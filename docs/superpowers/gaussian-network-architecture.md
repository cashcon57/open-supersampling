# OSS-Gaussian — Network Architecture

**Sprint:** 4. **Status:** design + skeleton landed; training pending Lambda
authorisation. **Spec:** `docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md`.
**Plan:** `docs/superpowers/plans/2026-05-01-gaussian-sprint-4-plan.md`.

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

---

## 4. Loss function

```
L_total = L_hdr_l1
        + 0.1   · L_ssim
        + 0.05  · L_lpips
        + 0.1   · L_temporal
        + 0.001 · L_cov_reg
```

| Term | What | Why |
|------|------|-----|
| `L_hdr_l1` | L1 on Reinhard-tonemapped HR images | Pixel accuracy under HDR-aware tone mapping. |
| `L_ssim` | 1 − SSIM(window=11) | Structural correctness, low-frequency stability. |
| `L_lpips` | LPIPS-VGG | Perceptual quality at human-relevant scale. |
| `L_temporal` | L1 between current prediction and motion-warped previous prediction in static regions | Eliminates ghosting and twinkle in flat areas. |
| `L_cov_reg` | `max(0, target_entropy − H(softmax(bank_logits)))`, target_entropy = 0.6 · log(K_bank) | Prevents bank collapse to one entry. |

The coefficients match the Sprint 4 spec; ablation of `L_lpips` ∈ {0.0, 0.05,
0.10} is part of T4.4 to confirm 0.05 is the sweet spot.

---

## 5. Training resource estimate

**Compute budget:** Lambda H100 SXM 80 GB. Single-node DDP (one GPU is enough
for our model size; we just want fast wall-clock).

| Phase | H100-hours | $ at $3/h |
|-------|----------:|---------:|
| Pretrain (T4.6) standard tier, 50 epochs Sintel + TartanAir | 15 | $45 |
| Fine-tune (T4.7) + Cyberpunk + temporal | 3 | $10 |
| Bank size ablation (T4.9, 3 variants × 5 epochs) | 9 | $27 |
| K ablation (T4.10, 3 variants × 5 epochs) | 9 | $27 |
| Tier fan-out (T4.11, Pico+Lite prune-retrain + Ultra from scratch) | 25 | $75 |
| Debug / restart buffer | ~10 | $30 |
| **Total** | **~71** | **~$210** |

The master plan budget is $50–$100; the above is the comprehensive run
including all ablations. Critical-path-only (skip ablations + Ultra) is ~$55.

**Data volume:**
- Sintel: 1064 frames + flow + depth (~8 GB).
- TartanAir: subsample to ~25 GB.
- Cyberpunk: depends on Sprint 2 hook capture rate; budget 30 GB.
- SRGD: ~3 GB.

Total: ~70 GB; fits comfortably on a single Lambda H100 instance's local SSD.

**Wall-clock:** ~4 weeks total sprint time. Of that, GPU-bound tasks are ~1.5
weeks of cloud time spread across the sprint to leave room for debug + iter.

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
