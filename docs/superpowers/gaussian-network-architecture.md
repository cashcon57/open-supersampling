# OSS-Gaussian — Network Architecture

**Sprint:** 4 (pivoted 2026-05-02). **Status:** **NOT a path to single-image SR.** The 2D Gaussian splat representation cannot beat bicubic on single-image SR at our resource budget — verified across five independent paths in `docs/superpowers/experiments/2026-05-02-splats-cannot-SR-definitive.md`. OSS-SR has forked to a CNN-based pipeline (see `docs/superpowers/oss-sr-cnn-track.md` once it lands). This document remains the reference for the Gaussian track, which is being repurposed for **OSS Ray-Retracing (denoising / DLSS-RR replacement)** where Image-GS already beat OIDN on PSNR (D1 memo).
**Spec:** `docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md` (SR claims now stale — Ray-Retracing repurposing pending).
**Plan:** `docs/superpowers/plans/2026-05-01-gaussian-sprint-4-plan.md`.
**Live findings:** `docs/superpowers/experiments/2026-05-02-splats-cannot-SR-definitive.md`.

This document covers the param-network half of OSS-Gaussian: the small CNN
that turns LR + G-buffers into per-tile Gaussian parameters that the Sprint 1
Rasterizer consumes.

---

## 1. Data flow

```
                  ┌────────────────────────────────────────┐
LR color (3) ─────┤                                        │
depth (1) ────────┤                                        │
motion (2) ───────┤   GaussianParamNetwork (4-level UNet)  │   raw tensor
normals (3) ──────┤   c=(16,24,32,48) by default           ├────────────────┐
canvas state (3) ─┤   tile_size=16, K=5, bank_size=16      │  (B, K·22,     │
                  │                                        │   H/16, W/16)  │
                  └────────────────────────────────────────┘                │
                                                                             ▼
                                                          ┌──────────────────────────┐
                                                          │       OutputHead         │
                                                          │  ─ tanh-clipped Δμ → μ   │
                                                          │  ─ exp(log_scale)        │
                                                          │  ─ softmax(bank_logits)  │
                                                          │  ─ CovariancePriorBank   │
                                                          │  ─ sigmoid/softplus(c)   │
                                                          └────────────┬─────────────┘
                                                                       ▼
                                                              GaussianBatch
                                                                       ▼
                                                          ┌──────────────────────────┐
                                                          │        Rasterizer        │
                                                          │  (Sprint 1, CUDA / ref)  │
                                                          └────────────┬─────────────┘
                                                                       ▼
                                                                   HR image
                                                                       ▼
                                                              composite loss → ∇
```

`canvas state` is the bilinearly-warped previous-frame render, downsampled to
LR. Zeros at sequence boundary. Sprint 5's persistent canvas owns this signal.

The network only runs on **complex tiles** (Sprint 3's classifier mask). Simple
tiles bypass the network and go through bilinear upsample. The mask is applied
*outside* this module — all of `GaussianParamNetwork` runs at full LR resolution
to keep the convolutions dense; the simple-tile bypass happens at the
GaussianBatch level by zeroing out the unused rows.

---

## 2. Covariance Prior Bank vocabulary

The bank is 16 fixed (sx, sy, θ) entries. Each entry is converted to a 2×2 Σ on
demand via Σ = R diag(sx², sy²) Rᵀ.

| Idx | sx | sy | θ (deg) | Shape |
|----:|---:|---:|--------:|-------|
|  0  | 1.0 | 1.0 |  0  | small isotropic |
|  1  | 2.0 | 2.0 |  0  | medium isotropic |
|  2  | 4.0 | 4.0 |  0  | large isotropic |
|  3  | 4.0 | 1.0 |  0  | horizontal elongated, mid |
|  4  | 8.0 | 1.0 |  0  | horizontal elongated, large |
|  5  | 1.0 | 4.0 |  0  | vertical elongated, mid |
|  6  | 1.0 | 8.0 |  0  | vertical elongated, large |
|  7  | 4.0 | 1.0 | 45  | diagonal ↗ |
|  8  | 4.0 | 1.0 | 135 | diagonal ↘ |
|  9  | 3.0 | 1.0 | 45  | thin diagonal ↗ |
| 10  | 1.0 | 3.0 | 45  | thin diagonal (perpendicular) |
| 11  | 8.0 | 0.5 |  0  | narrow streak, horizontal |
| 12  | 8.0 | 0.5 | 45  | narrow streak, diagonal ↗ |
| 13  | 8.0 | 0.5 | 90  | narrow streak, vertical |
| 14  | 8.0 | 0.5 | 135 | narrow streak, diagonal ↘ |
| 15  | 1.5 | 1.5 |  0  | small-medium isotropic (neutral fallback) |

The network predicts a softmax distribution over the 16; final per-Gaussian
(sx, sy, θ) is the weighted geometric/circular mean of the entries (so the
parametrisation stays in the same space the renderer consumes).

Bank size and entries are ablated in Sprint 4 / T4.9. If size 32 wins by
> 0.3 dB PSNR we adopt it as the default.

### 2.1 Anisotropic G-buffer-conditioned bias (added 2026-05-02)

Implements Decision 2 from `docs/superpowers/experiments/2026-05-01-validation-decision-memo.md`. The naive-denoising D1 test showed the Gaussian prior over-smooths textured regions; biasing the bank softmax toward elongated entries aligned with surface gradients addresses this.

```
per-pixel depth (B,1,H,W) ─┐
per-pixel normals (B,3,H,W)┤  avg_pool(tile_size)  ┌─────────────────┐
                           ├──────────────────────►│ tile features   │
depth_gradient (∂z/∂x,∂z/∂y)                       │ (B, Ht, Wt, 5)  │
                                                   └────────┬────────┘
                                                            │
                                                  Linear(5 → bank_size)
                                                  zero-init
                                                            │
                                                            ▼
bank_logits (B,Ht,Wt,K,bank_size) ──────────► add ──► softmax ──► bank_w
```

- **5-channel feature** = 3 mean normal components + 2 mean depth-gradient components per tile.
- **Per-tile bias**, *not* per-Gaussian. The K Gaussians inside one 16×16 tile share the bias term — the tile is the geometric primitive that has a single dominant orientation.
- **Zero-init projection** so a freshly-enabled `OutputHead(enable_gbuffer_bias=True)` matches the disabled output bit-for-bit. The network learns when/how to use the bias; default is "no signal."
- **Backward compat**: `enable_gbuffer_bias=False` (the default) skips the bias module entirely. `OutputHead.decode(raw)` without `depth=`/`normals=` also bypasses it even when enabled.

Implementation: `oss/gaussian/network/output_head.py::GBufferCovarianceBias`. 7 tests in `tests/gaussian/test_network.py` cover the activation/deactivation states and gradient flow.

---

## 3. Tier scaling

A single trained model is the goal. The network's channel widths and number of
Gaussians per tile (K) are the tier knobs; bank_size is shared across tiers.

| Tier | GPU target | Channels (c0,c1,c2,c3) | K/tile | Gaussians budget | Output res |
|------|-----------|------------------------|--------|-----------------:|-----------:|
| Pico     | Steam Deck (RDNA 2 iGPU)  | (8, 16, 24, 32)  | 3 | 1 000  | 1280×800 |
| Lite     | M3 Max / RTX 4070         | (16, 24, 32, 40) | 5 | 5 000  | 1440p    |
| Standard | RTX 3080 Ti               | (16, 24, 32, 48) | 5 | 8 000  | 1440p / 4K |
| Ultra    | RTX 4090                  | (24, 32, 48, 64) | 8 | 15 000 | 4K       |

Pico/Lite/Standard share weights via channel-prune warm-start from the
standard checkpoint (Sprint 4 / T4.11). Ultra trains from scratch.

The Gaussians budget × K-per-tile relationship: `N_total ≈ K × n_complex_tiles`.
On a 1440p frame ~30% of 16×16 tiles are complex per Sprint 3's analysis; with
K=5 that gives 8 100 Gaussians ≈ Standard target.

> **Tier-collapse correction (2026-05-02):** The earlier "single trained model fans out to all tiers via prune/retrain" plan does not survive contact with smoke-test data. Pico (75K) at lr=3e-4, mild engine-aliased LR, 5 000 steps on SRGD ActionRPG produced flat 11–12 dB PSNR vs bicubic 33–37 dB — too undersized to learn SR from scratch at 540×960. The current production plan: **train Lite or Standard from scratch on 3080 Ti, then distil down to Pico for Steam Deck inference only.** Pico is no longer a from-scratch training target. See `docs/superpowers/experiments/2026-05-02-sprint4-smoke-findings.md` §4 for the per-step PSNR table.

---

## 4. Loss function

**Spec (target):**

```
L_total = L_hdr_l1
        + 0.1   · L_ssim
        + 0.05  · L_lpips
        + 0.1   · L_temporal
        + 0.001 · L_cov_reg
```

**Currently implemented in `oss/gaussian/train/train.py::composite_loss`:** `L1 + 0.1 · (1 − SSIM)` when `pytorch_msssim` is importable, otherwise `L1 + 0.1 · pooled_l1` as a dependency-free fallback. Real SSIM is now wired (commit on `v0.2-dev`, 2026-05-02). The earlier `ssim_proxy` metric was a misnamed pooled-L1; it's been removed and the field is now either `ssim` or `pooled_l1` depending on which path runs. **LPIPS, temporal, and covariance-regularisation terms are not yet wired** — they're spec'd but blocked behind the basic trainability gate.

| Term | What | Why | Status |
|------|------|-----|--------|
| `L_hdr_l1` | L1 on Reinhard-tonemapped HR images | Pixel accuracy under HDR-aware tone mapping. | LDR-only L1 wired; tone-map step pending. |
| `L_ssim` | 1 − SSIM(window=11) | Structural correctness, low-frequency stability. | ✓ wired (pytorch_msssim). |
| `L_lpips` | LPIPS-VGG | Perceptual quality at human-relevant scale. | Pending. |
| `L_temporal` | L1 between current prediction and motion-warped previous prediction in static regions | Eliminates ghosting and twinkle in flat areas. | Pending (gated on Sprint 5 canvas wiring). |
| `L_cov_reg` | `max(0, target_entropy − H(softmax(bank_logits)))`, target_entropy = 0.6 · log(K_bank) | Prevents bank collapse to one entry. | Pending. |

The coefficients match the Sprint 4 spec; ablation of `L_lpips` ∈ {0.0, 0.05,
0.10} is part of T4.4 to confirm 0.05 is the sweet spot.

---

## 5. Training resource estimate

**Compute budget (current):** RTX 3080 Ti 12 GB only. Lambda H100 spend is **postponed indefinitely** for v0 — Cyberpunk capture is gated behind first beating bicubic on this hardware. Master arch must work with the constraint that all training is local.

**Implications:**
- Batch size capped by 12 GB VRAM (no DDP, no bf16 mixed precision for now — gsplat 1.4.0 has fp16 NaN edge cases at SR resolution).
- No multi-week burst training. Multi-day continuous runs only.
- Ablation matrix (T4.9, T4.10) is shrunk to a single-variant pass; full ablation deferred to a v1 if v0 ships.
- Tier fan-out (T4.11) is **pico/lite distillation-only** post-MVP, not parallel from-scratch training.

**Cost model:** electricity, not cloud. ~$30–$50 across a multi-day production run. No accountable spend gate.

**Future (only if v0 succeeds and budget reopens):**
| Phase | H100-hours | $ at $3/h |
|-------|----------:|---------:|
| Standard tier full retrain on multi-game corpus | 20 | $60 |
| Ultra tier ablation + fine-tune | 15 | $45 |
| Bank/K ablation matrix | 18 | $54 |
| **Total (deferred)** | **~53** | **~$160** |

**Data volume on 3080 Ti:**
- Sintel: clean + flow only (~8 GB; depth supplement not staged — see findings memo).
- TartanAir: extracted partial; full triple available on `ocean/Easy/P*` only.
- Cyberpunk: not captured yet (gated behind trainability gate).
- SRGD: ~3 GB extracted, multi-scene; primary v0 dataset.
- HyperSim: 18K scenes, zips on G:\ — extraction deferred until SRGD signal proven.

**Wall-clock:** open-ended on local hardware. Multi-day runs feasible once trainability is unblocked.

---

## 6. Where this fits

- Sprint 1 (renderer, done) — provides the differentiable `Rasterizer` this
  network is trained against.
- Sprint 3 (classifier, parallel) — provides the complex/simple tile mask.
- Sprint 5 (persistent canvas, next) — consumes the trained checkpoint to
  spawn new Gaussians on disocclusion. The `OutputHead` decoder doubles as
  Sprint 5's Gaussian-spawn function.
- Sprint 7 (cross-platform ports) — re-exports the network for CoreML
  (M3 Max) and ncnn/Vulkan (Steam Deck). Bank decoding stays in PyTorch /
  host-side because it's a small post-process.

---

## 7. V0 → V2 staging (added 2026-05-02 from Codex 5.5 review)

The original spec wanted single-frame Gaussian SR + persistent canvas + frame extrapolation as one unified output. External review (Codex 5.5) flagged this as over-promising: contour-aware splat literature ([Image-GS](https://arxiv.org/abs/2407.01866), [Gaussian Billboards](https://arxiv.org/abs/2412.12734), [Contour-aware 2DGS](https://arxiv.org/abs/2512.23255)) shows plain colored-blob Gaussians blur high-frequency edges without explicit texture/contour priors. We now stage the work:

| Stage | Scope | Gate to next stage |
|-------|-------|--------------------|
| **V0** | Single-frame Gaussian SR: LR + depth + motion + normals → param net → splat raster. | Lite or Standard tier beats bicubic by ≥1 dB PSNR on aggressive engine-aliased LR (σ=1.5 + JPEG q=85) on at least one held-out scene. |
| **V0.5** *(fallback if V0 stalls)* | Add a small **pixel-residual head** that predicts a residual on the Gaussian-rendered HR. The bulk of the structure comes from the splats; the CNN cleans up high-frequency texture. Used by GSASR ([arXiv:2501.06838](https://arxiv.org/abs/2501.06838)) and GS-STVSR ([arXiv:2604.18047](https://arxiv.org/abs/2604.18047)) and is the most-likely cure for the "pure-splats blur edges" failure mode. | Same gate as V0 with the residual head enabled. |
| **V1** | Persistent canvas (Sprint 5 wiring) on top of V0 / V0.5. Flow-guided position and color evolution per [GS-STVSR](https://arxiv.org/abs/2604.18047), covariance resampling, adaptive motion windows. | Temporal stability ≥ baseline FSR2 Quality on a 10-second clip (no twinkling, no ghosting). |
| **V1.5** | Frame extrapolation via fractional-time canvas warp + a **learned disocclusion-repair head** for newly-revealed pixels, particles, transparencies, UI, and speculars. The earlier "free byproduct" framing was wrong — disocclusion is a real learning problem. | Quality ≥ DLSS-FG on a side-by-side perceptual review on Cyberpunk 2077. |
| **V2** | Ray-Retracing track using Gaussians as geometry-aware denoising and spatial reconstruction. Treat as a separate gate from SR. | Beats OIDN on PSNR + LPIPS on real path-traced NoiseBase frames. |

**Each stage has its own gate.** No work on V1+ until V0 (or V0.5) clears its bicubic gate. No community Cyberpunk capture until V0/V1 trains successfully on synthetic + game-engine datasets we already have.

---

## 8. References

- [GaussianSR — feed-forward 2D Gaussian fields for arbitrary-scale SR (arXiv:2407.18046)](https://arxiv.org/abs/2407.18046)
- [GSASR — image-conditioned Gaussian SR with CUDA rasterization (arXiv:2501.06838)](https://arxiv.org/abs/2501.06838)
- [GS-STVSR — continuous spatial + temporal upscaling via 2D Gaussian evolution (arXiv:2604.18047)](https://arxiv.org/abs/2604.18047)
- [Image-GS — texture-conditioned 2D Gaussian image rep (arXiv:2407.01866)](https://arxiv.org/abs/2407.01866)
- [Gaussian Billboards — texture-augmented splats (arXiv:2412.12734)](https://arxiv.org/abs/2412.12734)
- [Contour-aware 2DGS — explicit edge constraints for splat-based reconstruction (arXiv:2512.23255)](https://arxiv.org/abs/2512.23255)
- [NVIDIA Streamline — color/depth/motion resource tagging for DLSS-style features](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuide.md)
- [NVIDIA NRD — normal/roughness/viewZ/motion-conditioned ray-traced denoising](https://github.com/NVIDIA-RTX/NRD)
