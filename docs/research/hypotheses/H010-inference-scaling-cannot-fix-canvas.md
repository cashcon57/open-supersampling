# H010 — Inference-time Canvas Scaling Alone Cannot Produce Real OSS-FX Motion

**Status:** `validated` — confirmed on `srcnn-v6.2-pico-002` step-00070000 via the canvas-scale sweep (`scripts/sr_v6_canvas_scale_sweep.py`).
**Class:** Architecture-constraint gate. Decides whether v6.3 (magnitude scaling) is sufficient or whether v7 (native time-aware Gaussians) is required.
**Filed:** 2026-05-12
**Source:** Followup to H009 (canvas not utilized). The question H009 left open: can we recover canvas utilization at *inference* without retraining, by amplifying canvas_hr at the composite_head input?

## Claim

Multiplying canvas_hr by a constant scale at the composite_head input of v6.2-pico-002 produces a measurable but quantitatively insufficient shift between the model's output at α=1 and α=0.5. At every scale where output quality holds (PSNR drop ≤ 0.5 dB vs unscaled baseline), the extrap-vs-SR mean abs pixel diff stays at most ~1% of the inter-frame motion magnitude. Real frame extrapolation at α=0.5 would require this ratio to be ~50% (the canvas Gaussian visibly moves halfway across the motion field). The pre-trained canvas channels do not encode motion-aware temporal content; they cannot be made to do so by scaling alone.

## Procedure

`scripts/sr_v6_canvas_scale_sweep.py` hooks composite_head's `forward_pre_hook` and multiplies the canvas_hr portion of its input by a constant. For each scale value in {1, 5, 10, 50, 100, 500, 1000}, the script runs 16 held-out manifest pairs through the v6 inference path twice — once at α=1.0 (normal motion) and once at α=0.5 (canvas warped halfway) — and records:

- PSNR / LPIPS at α=1.0 (quality vs ground truth frame N+1)
- PSNR / LPIPS at α=0.5 (quality vs the same ground truth — for tracking only)
- mean abs pixel diff between α=1.0 and α=0.5 outputs (the "extrap_diff")
- mean abs pixel diff between consecutive α=1.0 outputs (the "inter_frame_diff", motion baseline)

The headline metric is the ratio `extrap_diff / inter_frame_diff`. A ratio of 1.0 would indicate the α=0.5 output captures the full motion delta (which is not what we want — α=0.5 should be *half* the motion). A ratio of 0.5 would be canonically correct for halfway extrapolation. A ratio of 0.0 means the canvas isn't moving anything.

## Result

| scale | PSNR @ α=1 | LPIPS @ α=1 | extrap_diff | extrap/inter-frame ratio |
| --- | ---: | ---: | ---: | ---: |
| 1.0 | 27.111 | 0.1134 | 0.00005 | **0.0005** |
| 5.0 | 27.111 | 0.1134 | 0.00008 | 0.0008 |
| 10.0 | 27.110 | 0.1135 | 0.00011 | 0.0011 |
| 50.0 | 27.068 | 0.1162 | 0.00043 | 0.0043 |
| 100.0 | 26.941 | 0.1209 | 0.00086 | 0.0086 |
| 500.0 | 24.731 | 0.1480 | 0.00463 | 0.0436 |
| 1000.0 | 22.018 | 0.1635 | 0.00845 | **0.0752** |

inter_frame_diff baseline (constant across scales): ~0.110

## Interpretation

The canvas channels DO carry signal — `extrap_diff` grows monotonically with scale, from 5e-5 at scale=1 (where the canvas is effectively ignored) to 8e-3 at scale=1000 (where signal is heavily amplified). So the model is *not* outputting random noise from the canvas pathway; there is real, scale-dependent content being passed through.

But:

1. **At quality-preserving scales (≤100), the ratio stays below 1%.** That's far below the ~50% needed for "the canvas-rasterized image at α=0.5 visually contains the halfway-motion frame." The canvas channels do not encode enough temporal motion to produce visible intermediate-frame content even when fully amplified.
2. **At higher scales, quality breaks before the ratio reaches anywhere close to 50%.** At scale=1000 the ratio reaches 7.5% but PSNR is below bicubic (22 dB vs bicubic 23.9). The trade-off is monotonically losing.
3. **Linear extrapolation of the ratio curve** suggests reaching ratio=0.5 would require scale ≈ 10⁴ or higher. Output quality at such scales is meaningless.

## Verdict

Inference-time canvas scaling alone is **not** a path to real OSS-FX motion on v6.2-pico-002.

This is a critical pre-training finding because it rules out a class of "fix v6.x without retraining" strategies. To make canvas-based frame extrapolation work, the canvas channels themselves must be *trained* to carry motion-aware temporal information. There are three identified paths:

1. **v6.3 retrain** — add magnitude scaling AT TRAINING TIME (composite_head sees pre-amplified canvas signal from day one) and a canvas-aware aux loss (canvas-only output must match GT) so the optimizer is pushed to encode useful content in canvas_hr. Same backbone, same canvas geometry. 2D.
2. **v7 retrain** — extend Gaussians to N-D (x, y, t) so temporal information is a NATIVE dimension of each Gaussian, not an emergent property of the canvas representation. The α=0.5 rendering becomes a structural property of the rasterizer, not a learned encoding. 3D.
3. **Parent-child spawner** (applies to either v6.3 or v7) — replace the conservative DisocclusionSpawner with loss-adaptive density allocation per Diolatzis et al. 2024, so the canvas gets capacity where the model needs it.

The v6.3-fine experiment (`docs/architecture/2026-05-12-v63-fine-finetune-spec.md`) is the cheapest gate that validates the first path; v7 is the architectural commitment if v6.3-fine succeeds OR if we want to skip the 2D bridge entirely.

## What this finding does NOT say

- Canvas channels are not "noise". They carry real spatial content; H009's activation statistics confirmed all 16 channels have non-zero variance.
- v6.x architecture is not "broken". It's a well-functioning SR model; the canvas just isn't doing the *temporal* job it was designed for.
- Magnitude scaling at training time won't help. v6.3 is testing exactly that, and the v6.3-fine experiment can pre-validate.

## Evidence

- Probe harness: `scripts/sr_v6_canvas_scale_sweep.py`
- Probe output: `<train-host-data>/checkpoints/srcnn-v6.2-pico-002/canvas-scale-sweep.json`
- Reproducible on any v6.2 ckpt with the held-out manifest in tree.

Reproduce:

```bash
python scripts/sr_v6_canvas_scale_sweep.py \
    --ckpt-temporal <train-host-data>/checkpoints/srcnn-v6.2-pico-002/step-00070000.pt \
    --tartanair-root <train-host-data>/datasets/tartanair_extracted \
    --manifest <train-host-data>/checkpoints/v5_held_out_manifest.json \
    --output canvas-scale-sweep.json \
    --device cuda
```
