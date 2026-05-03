# OSS-Gaussian-Temporal Track — Sprint 5/6 with Splats for Extrapolation

**Date created:** 2026-05-02
**Trigger:** system-level reframe in `docs/superpowers/experiments/2026-05-02-srcnn-beats-v05-and-gsasr.md` § "System-level reframe"
**Status:** scaffolding only — first decisive test queued.

## Why this track exists

The day's investigation produced two real findings:

1. SR-CNN beats every Gaussian-SR variant we tested **on per-frame SR alone**.
2. We never measured the *system-level* value of splats — which is what made the splat thesis interesting in the first place: cheap, structurally-coherent **frame extrapolation** by warping Gaussian centres along motion vectors, vs. running a separate CNN extrapolator at every fractional time step.

The per-component conclusion that "splats are dead" jumped over the entire reason to build splats. This track exists to test the system-level case before we kill it.

## The thesis (testable)

> A model that produces Gaussians per frame can extrapolate to t+α by warping Gaussian positions along the engine's motion vectors and re-rasterizing. This is structurally temporally coherent (no learned regularizer needed) and costs only the rasterizer call (~tens of µs), versus running a separate CNN extrapolator (~ms). Even if the per-frame SR PSNR is at parity (or slightly below) bicubic / SR-CNN, the system can still win on:
>
> - **Total extrapolation cost** (one model + cheap warp vs. SR model + extrapolation model + temporal consistency loss).
> - **Extrapolation quality** (geometric warp vs. learned warp, especially on disocclusions).
> - **VRAM and latency budgets** for shipping on Steam Deck / older GPUs.

## What we have not yet measured

- Quality of Gaussian-position-warp extrapolation against any baseline (bicubic + flow warp, separate CNN extrapolator, DLSS-FG).
- Whether GSASR-pretrained Gaussians (fed engine motion vectors at inference) extrapolate coherently to t+1.
- Whether *training* a Gaussian model with a temporal loss term (encouraging Gaussians to be temporally stable) materially improves extrapolation quality vs. our current per-frame-only training.

All three are decision-relevant. None of them are answered by the SR-only experiments to date.

## First decisive test (V0 of this track)

**Hypothesis:** Even with GSASR's pretrained-on-bicubic-clean weights — which lose by 0.04 dB to bicubic on per-frame SR over our LR — the *extrapolated* output (warp Gaussians along motion vectors, re-rasterize at t+1) is meaningfully better than a trivial bicubic+flow-warp baseline.

**Setup:**

- 24–48 SRGD frames where consecutive (frame N, frame N+1) pairs are available with motion vectors.
- For each pair:
  - Method A — Gaussian warp: feed frame N's LR through GSASR → get Gaussian set → warp positions along the engine motion vectors → rasterize at frame N+1's resolution → compare to frame N+1's HR ground truth.
  - Method B — bicubic + flow warp: bicubic-upsample frame N's LR → warp pixels along the same motion vectors → compare to frame N+1's HR ground truth.
- Compute PSNR / SSIM / LPIPS on the extrapolated frame vs. the GT frame N+1, mean across the 24–48 pairs.

**Decision rule:**

- **Method A beats Method B by ≥1 dB and on >75% of pairs:** the system-level case for splats is real. Justify investing in Sprint 5/6 (canvas + extrapolation) seriously.
- **Method A ≈ Method B (within ±0.5 dB):** splats add no extrapolation value; close the splat track for the SR/extrapolation product.
- **Method A loses to Method B by >0.5 dB:** splats actively hurt extrapolation; close the track.

**Time budget:** 2–3 days. Most of the time is plumbing the motion-vector warp; the inference step is reusing the already-installed GSASR.

## What carries over from prior work

- `oss/gaussian/renderer/` (Sprint 1) — the rasterizer can render at any output resolution.
- `oss/gaussian/canvas/{canvas,warp,prune_spawn,error_detection}.py` (Sprint 5 scaffold) — already has motion-vector warp on Gaussian positions.
- `oss/gaussian/extrapolation/{extrapolator,alpha_scheduler}.py` (Sprint 6 scaffold) — already has α-conditioned warp + scheduling.
- SRGD dataset adapter — has motion vectors as part of the 12-channel stack.

The infrastructure is all there. The missing piece is the experiment harness that pipes (GSASR weights → Gaussian set → existing warp + extrapolator → metrics).

## Decision tree from here

```
                    ┌─ A wins by ≥1 dB ──► Sprint 5/6 funded; build Gaussian-temporal stack as v1
First test ─────────┤
                    ├─ A ≈ B   ─────────► Splats provide no extrapolation value; ship SR-CNN + a
                    │                     standard CNN extrapolator (DLSS-FG-like) for v1.
                    │
                    └─ A loses ─────────► Close splat track for SR/extrapolation product.
                                          Gaussian work continues only in OSS-Gaussian-RR (denoising).
```

## What does NOT get done in this track

- **Retraining GSASR or any other Gaussian SR model on engine-aliased LR.** That's a 2–4 week investment; we don't make it without first showing extrapolation-side value above.
- **Sprint 5 (canvas) production wiring.** That happens *after* the first test passes the gate.
- **Comparison to DLSS-FG.** Useful but Cyberpunk-only; defer to after we know whether the basic Gaussian-warp story works at all.

## Design constraint — interpolation-readiness

V0 of this track focuses on **extrapolation** (predict frame N+α from frame N's Gaussians + motion vectors, no future-frame info). Interpolation (generate frame N+0.5 from BOTH frame N's and N+1's Gaussians) is a near-future option we want to be able to flip on without rebuilding the stack.

The two modes share most of their machinery:

| Operation | Extrapolation | Interpolation |
|-----------|---------------|---------------|
| Position update | `xy_t = xy_N + α · motion_N` | `xy_t = (1−α)·xy_N + α·xy_{N+1}` |
| Color update | reuse `feat_N` | lerp `feat_N`, `feat_{N+1}` |
| Covariance | reuse Σ_N (or evolve along flow) | lerp Σ_N, Σ_{N+1} |
| Disocclusion repair | learned head on motion-vector residual | optional repair on disagreements |
| Renderer call | identical | identical |

Concretely, this means the warp primitive in `oss/gaussian/canvas/warp.py` should expose at minimum:

```python
def warp(
    gauss: GaussianBatch,
    motion: Tensor,
    alpha: float,
    target_gauss: GaussianBatch | None = None,  # interpolation when not None
) -> GaussianBatch:
    """When target_gauss is None: extrapolation along motion. When set:
    interpolate between gauss (at α=0) and target_gauss (at α=1)."""
```

The same call site supports both modes; flipping requires only providing the second endpoint's Gaussians. This is cheap to design in now, expensive to retrofit later.

**Implementation rule:** any new code in `oss/gaussian/canvas/`, `oss/gaussian/extrapolation/`, or the inference pipeline must accept both `(gauss, motion, alpha)` *and* `(gauss_a, gauss_b, alpha)` shapes — not one or the other. Document with a tiny test for each path even if interpolation is initially unused.

## Why this isn't the same as the OSS-Gaussian-RR track

- **OSS-Gaussian-RR** (denoising / DLSS-RR replacement) uses Gaussians as a smoothing prior on noisy ray-traced HDR frames. The D1 result on synthetic noise validated that direction. Gated on NoiseBase data.
- **OSS-Gaussian-Temporal** (this track) uses Gaussians as a temporally-warpable HR feature/RGB representation for cheap frame extrapolation. The SR component need only be at parity for the system case to work; the value comes from extrapolation.

They are independent and parallel. RR doesn't need temporal warping; Temporal doesn't need denoising. Either may succeed or fail without affecting the other.

## Status flags

- ✅ Renderer + canvas + extrapolator scaffolds present from Sprints 1, 5, 6.
- ✅ GSASR installed and runnable on 3080 Ti (per the earlier GSASR memo).
- ⏳ First decisive test (Method A vs. Method B) — designed but not yet run.
- 🔒 Sprint 5 (canvas) production wiring — gated on first decisive test passing.
