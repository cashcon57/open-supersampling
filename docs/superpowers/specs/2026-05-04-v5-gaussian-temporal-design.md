# v5 Gaussian Temporal Super-Resolution — Design Spec

**Status:** Design (research track in v5 dual-track race)
**Date:** 2026-05-04
**Author:** Cash Conway + Claude (Opus 4.7)
**Parent track:** [oss-gaussian-temporal-track.md](../oss-gaussian-temporal-track.md)

## Goal

Test the hypothesis that 2D Gaussians as a persistent temporal scene memory can outperform pixel-based temporal accumulation (the v5-pixel-temporal control track) for real-time super-resolution. The unique advantages we're testing:

- Analytical sub-pixel warping (no resample blur compounding across frames)
- Continuous representation with persistent positions (sub-pixel jitter accumulation is structural, not learned)
- Tractable token count for multi-frame attention (~5K Gaussians vs millions of pixels per frame)
- Densification under disocclusion (clean newly-visible region handling, no pixel-rejection blockiness)

This is **research-grade work**. No production deployment of Gaussian temporal SR exists. We are deliberately running this in parallel with the proven pixel track so that we have a safe fallback if this fails.

## Non-goals

- Production deployment in v5 cycle — even if this wins, we ship the pixel track first if both pass quality gates
- 3D scene reconstruction (we stay in 2D image space; no view-dependent effects)
- Frame extrapolation (separate track, OSS-FX)

## Success criteria

The Gaussian track wins or ties the pixel track on the **same fixed held-out batch** used for the v3-vs-v4 A/B:

- [ ] PSNR ≥ pixel track − 0.3 dB (tie acceptable)
- [ ] LPIPS ≤ pixel track − 0.01 (genuine perceptual win)
- [ ] Temporal stability (warp-then-diff between t and t+1) ≤ pixel track variance
- [ ] Inference latency ≤ 1.5× pixel track at 1080p→4K on RTX 3080 Ti

If Gaussian beats pixel, we ship Gaussian as v5. If Gaussian ties or loses, we ship pixel as v5 and continue Gaussian as v6+ research.

## Architecture

### Inputs (per frame)

Same per-frame G-buffer stack as pixel track:
```
LR color (3ch), Depth (1ch), Motion vec (2ch), Normals (3ch), Canvas hint (3ch)
```

Plus: **persistent Gaussian field** carried across frames — `K` Gaussians, each with `(μ_xy, Σ, c)` (mean, covariance, color). Initial K = 4096 per scene; densification can grow to 16K.

### Network

```
Frame t input ────┬─── G-buffer encoder ─── per-tile features (LR-resolution)
                  │
[ Gaussians_{t-1} ] ─── warp by motion vec (analytical) ─── Gaussians_warped
                  │
[ G-buffer features + Gaussians_warped (as tokens) ]
                  ├─── multi-frame transformer ─── attention(N=3-5 prev steps)
                  │
                  └─── densification head ─── new Gaussians for high-residual tiles
                              │
                              └─── Gaussians_t (updated field)
                              │
                              └─── differentiable rasterizer ─── output HR frame
```

**Components:**

1. **G-buffer encoder** — small CNN (~100K params) producing per-tile context features at LR resolution.
2. **Analytical Gaussian warp** — for each Gaussian, `μ' = μ + flow(μ)` (sample motion vector at the Gaussian's mean), `Σ' = J_flow · Σ · J_flow^T` (Jacobian-transformed covariance). No resampling blur.
3. **Multi-frame transformer** — ~500K params:
   - Tokens: `concat(Gaussian_params_at_t-N..t-1, encoded_g_buffer_features_at_t)`
   - Attention layers: 4-6 with rotary position embedding (positions are Gaussian means)
   - Output: per-token update `(Δμ, ΔΣ, Δc)` applied to current Gaussian set
4. **Densification head** — produces new Gaussians where residual error is high after warp+update:
   - Compute residual via differentiable raster of warped Gaussians vs current LR (upsampled)
   - Pick top-K_new tiles by residual magnitude
   - Spawn 1-2 new Gaussians per tile (initial mean = tile center, covariance = identity, color = tile mean)
   - Differentiable through soft top-K (Gumbel-Softmax or learned threshold)
5. **Pruning head** — removes Gaussians with weight below threshold (low contribution after warp). Hard threshold post-update; not differentiable but applied post-loss.
6. **Differentiable rasterizer** — reuses the Image-GS / gsplat tile-based rasterizer from V0 work. Produces HR output frame.

### Inference state

- Persistent Gaussian set: ~16K × 11 floats (μ_xy + Σ_3 + c_3 + opacity_1) × 4 bytes = ~700 KB
- Frame history of Gaussians (N=3-5 prev steps): ~3.5 MB
- Compared to pixel track's 24 MB HR buffer — Gaussian state is significantly smaller

## Loss

```
L_total = L_appearance + λ_temporal · L_temporal_consistency + λ_reg · L_gaussian_reg

L_appearance = w_l1 · L1(rendered_t, gt_hr_t)
             + w_ssim · (1 - SSIM(...))
             + w_lpips · LPIPS(...)

L_temporal_consistency = w_tc · L1( render(warp(Gaussians_t, motion_t→t+1)) · valid_mask,
                                    rendered_{t+1} · valid_mask )

L_gaussian_reg = w_pos · ||μ_drift||₂   (mean drift from t-1)
              + w_cov · max(0, det(Σ) - max_area)   (anti-collapse on huge Gaussians)
              + w_count · max(0, count - max_count)   (anti-explosion)
```

**Hyperparameters (start point):**
- `w_l1 = 1.0`, `w_ssim = 0.1`, `w_lpips = 0.1`
- `λ_temporal = 0.05`, `λ_reg = 0.01`
- `max_count = 16384`, `max_area = (8 px)²`

## Training

### Data

Same as pixel track: TartanAir Easy primary, Sintel secondary.

**Sequence sampling:** different from pixel track — Gaussian field needs **multi-frame context** (3-5 frames) for the transformer attention. Sample a trajectory window of 5-7 consecutive frames per training step. ~3× compute per step vs pixel track (which only needs 2-frame pairs).

### Schedule

1. **Phase 1 — single-frame Gaussian fitter (steps 0–20K):** train only the per-frame fitter to produce Gaussians from one frame. No temporal. This gets the rasterizer + densification stable in isolation. Reuse V0.5 splat infrastructure.
2. **Phase 2 — temporal warp + transformer warmup (steps 20K–50K):** add prev-frame Gaussian warp + small transformer (2 layers). Frozen fitter. Establishes the temporal update head can learn.
3. **Phase 3 — joint training (steps 50K–120K):** unfreeze fitter, full transformer (4-6 layers), full loss including temporal consistency. Densification active.
4. **Phase 4 — Sintel fine-tune (steps 120K–140K):** real-data polish.

Total: ~140K steps, ~24-48 hours on RTX 3080 Ti.

## Files

- **New module**: `oss/sr/gaussian_temporal/`
  - `__init__.py`
  - `gaussian_field.py` — persistent Gaussian state container
  - `analytical_warp.py` — Gaussian warp by motion vec (μ + Σ Jacobian transform)
  - `transformer.py` — multi-frame attention over Gaussian tokens
  - `densification.py` — residual-driven new-Gaussian spawning
  - `pruning.py` — opacity-threshold removal
  - `rasterizer.py` — wrap existing gsplat or Image-GS rasterizer
  - `dataset.py` — multi-frame trajectory window loader
- **Reused modules**: `oss/gaussian/network/`, `oss/gaussian/canvas/` (V0.5 splat code as starting base)
- **New script**: `scripts/sr_train_gaussian_temporal.py`
- **Updated**: `oss/sr/inference.py` — add Gaussian stateful inference path
- **Tests**:
  - `tests/sr/test_gaussian_warp.py` — analytical Gaussian warp roundtrip with synthetic flow
  - `tests/sr/test_gaussian_dataset.py` — multi-frame window loader
  - `tests/sr/test_gaussian_transformer.py` — attention forward + backward + token equivariance
  - `tests/sr/test_gaussian_densification.py` — residual-driven spawning is differentiable
  - `tests/sr/test_gaussian_full_step.py` — end-to-end train step on synthetic moving rectangle

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Per-frame fitter cost dominates | Warm-start from prev-frame Gaussians; refit only on residual-flagged tiles |
| Refit drift / temporal flicker | Strong drift regularization; warm-start strategy explicit |
| Differentiable densification is delicate | Heuristic insertion at fixed gradient threshold (3DGS-style) for v5; revisit later |
| Training data semantics (sub-pixel ground truth) | Use TartanAir HR + flow as supervision signal directly; no separate sub-pixel target needed |
| Gaussian count explosion | Hard cap + pruning + count-regularization loss term |
| Multi-frame transformer attention is expensive | Cap at 5 frames history; tokens limited to ~5K Gaussians; benchmark vs target latency early |
| **Two failure modes hard to distinguish** | The v5-pixel control track is the answer — if Gaussian fails, we know it's the architecture, not the dataset/training pipeline |

## Validation gates

1. Standalone fitter: produces sensible Gaussian set on a single frame (PSNR > 28 dB on rendered output)
2. Warp+transformer: doesn't degrade single-frame baseline when temporal consistency loss is OFF
3. Full system: meets success criteria above
4. Inference latency benchmark: measure end-to-end on 1080p→4K target

## Open questions

- Forward warp (splat-into) vs backward warp (sample-from)? **Default: forward analytical warp on Gaussian means.**
- Should the Gaussian color be RGB or learned features? **Default: 3ch RGB for v5; learned features = v6.**
- Reset on scene cut? **Yes — same trigger as pixel track.**
- Run pixel and Gaussian inference paths simultaneously and ensemble? **Out of scope for v5; could be v7.**

## Out-of-scope (deferred)

- 4D Gaussian Splatting (true 3D-aware temporal)
- View-dependent effects (specular, refraction)
- Cross-attention between Gaussians and pixel features (v6+)
- INT8 quantization of Gaussian params (post-quality)

## How this slots into the v5 race

- **Both tracks train on the same TartanAir + Sintel mix.**
- **Both tracks evaluate on the same fixed held-out batch.**
- **Comparison criterion: success criteria above. Gaussian must explicitly beat pixel; tie ≠ win.**
- If Gaussian wins: ship as v5, pixel becomes parallel research.
- If Gaussian ties or loses: ship pixel as v5, Gaussian continues as v6+ research input.
- Both tracks' weights, eval scripts, and memos archived regardless of outcome.
