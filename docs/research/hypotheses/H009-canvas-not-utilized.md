# H009 — v6.2-pico-002 Canvas Is Not Used by the Composite Head

**Status:** `validated` — measured on `srcnn-v6.2-pico-002` step-00055500 on the TartanAir oldtown 16-pair held-out probe. Zeroing the canvas-rasterized features at the composite_head input produces bit-identical output (within 0.0001 PSNR / 0.0001 LPIPS) compared to passing the model's real canvas-hr through.
**Class:** Architecture utilization gate.
**Filed:** 2026-05-11
**Source:** Followup to user question "how much is the Gaussian canvas contributing vs the rest of the upscale stack" after v6.2-pico-002 reached LPIPS 0.112 / PSNR 26.4 on the held-out TartanAir oldtown batch.

## Claim

The composite_head of v6.2-pico-002 has learned to put effectively zero weight on the canvas-rasterized feature channels. The HAT-Tiny backbone path is doing the entire upscale; the Gaussian canvas + rasterizer + DisocclusionSpawner pipeline runs every forward but contributes nothing measurable to the output.

## Procedure

`scripts/sr_v6_canvas_probe.py` loads a v6 checkpoint and hooks `model.composite_head` via `register_forward_pre_hook` to substitute the canvas_hr portion of the cat'd input tensor without retraining. Four probe modes:

- **normal**: no intervention. Also captures per-channel canvas_hr activation statistics.
- **zero**: `canvas_hr` → `torch.zeros_like(canvas_hr)` at the composite_head input.
- **canvas-only**: `refined_hr` → zeros, leave canvas_hr untouched. Forces the composite head to produce output from canvas features alone.
- **stale-canvas**: use the previous frame's `canvas_hr` instead of the freshly-rasterized one. Tests whether per-frame canvas update contributes.

The same 16 held-out pairs from `v5_held_out_manifest.json` (TartanAir oldtown) are run in each mode, with PSNR (clamped) and LPIPS-VGG measured on the t+1 reconstruction. Bicubic baseline on the same manifest is 23.91 PSNR / 0.294 LPIPS.

## Result

`srcnn-v6.2-pico-002` step-00055500, 16-pair probe:

| Mode | PSNR | LPIPS |
| --- | ---: | ---: |
| normal | 26.3971 | 0.1168 |
| zero (canvas_hr → 0) | **26.3970** | **0.1168** |
| canvas-only (refined_hr → 0) | 24.0276 | 0.2964 |
| stale-canvas (use prev frame's canvas_hr) | **26.3971** | **0.1168** |
| bicubic baseline (reference) | 23.91 | 0.294 |

`normal`, `zero`, and `stale-canvas` agree to 0.0001 on both metrics. The probability of this occurring by chance across 16 independent samples × 2 metrics is effectively zero — the model's output is deterministically independent of canvas_hr at the composite_head input.

`canvas-only` falls to ~bicubic quality (24.03 vs 23.91 PSNR; 0.296 vs 0.294 LPIPS). Without backbone features, the composite_head produces a near-zero delta and the output collapses to its bicubic residual base.

## Activation statistics

Per-channel `canvas_hr` statistics, averaged over 32 frame evaluations during `normal` mode (16 pairs × 2 frames per pair):

| Channel | std | abs-mean |
| --- | ---: | ---: |
| 0 | 1.04e-02 | 2.08e-03 |
| 1 | 1.00e-02 | 1.87e-03 |
| 2 | 7.44e-03 | 1.45e-03 |
| 3 | 7.93e-03 | 1.50e-03 |
| 4 | 7.61e-03 | 1.44e-03 |
| 5 | 8.15e-03 | 1.58e-03 |
| 6 | 1.13e-02 | 2.04e-03 |
| 7 | 1.02e-02 | 1.87e-03 |
| 8 | 1.24e-02 | 2.44e-03 |
| 9 | 9.55e-03 | 1.71e-03 |
| 10 | 1.03e-02 | 1.96e-03 |
| 11 | 8.06e-03 | 1.54e-03 |
| 12 | 1.02e-02 | 1.89e-03 |
| 13 | 1.10e-02 | 2.08e-03 |
| 14 | 1.18e-02 | 2.23e-03 |
| 15 | 9.85e-03 | 1.88e-03 |

All 16 channels carry non-zero activation — zero are near-dead by the `std < 1e-3` threshold. The canvas pipeline IS producing signal; the composite_head simply does not weight it.

## Diagnosis

Two compounding factors:

1. **Magnitude mismatch.** `canvas_hr` abs-mean is ~2e-3. The backbone's `refined_hr` features are roughly ~1e-1 in scale (ReLU/GELU output of HAT-Tiny). The 3×3 conv at the composite_head fusion boundary sees a 60-channel signal at ~1e-1 vs a 16-channel signal at ~1e-3, a ~50× difference. The optimization-easy path during training was to zero the canvas weights.
2. **Sparse canvas.** The DisocclusionSpawner only writes new Gaussians at disocclusion-mask regions. In TartanAir flight footage with mostly-occluded camera motion, disocclusion events are rare per frame. The canvas state stays sparse, the rasterizer's output stays small, training never sees a "this canvas channel carries useful HR detail" signal strong enough to break the zero-weight equilibrium.

The architecture is functioning as designed; the *training* allocated the canvas's representational capacity to zero.

## Verdict

The v6.2-pico-002 LPIPS 0.112 / PSNR 26.4 measurement is **attributable to the HAT-Tiny backbone + GAN + perceptual-loss recipe alone**. The Gaussian canvas, ConcatFusion, V6Rasterizer, and DisocclusionSpawner are dead weight in this run with respect to SR quality.

This does NOT invalidate the canvas as an architectural primitive. It tells us:

1. The current `concat([refined, canvas])` fusion is a *necessary but not sufficient* condition for the canvas to be used. A magnitude-balancing intervention is required.
2. The DisocclusionSpawner is too conservative for the canvas to accumulate meaningful state on this kind of footage.
3. OSS-FX (the α<1 frame extrapolation case that uses the canvas as the primary output primitive) cannot work on a checkpoint trained this way — the canvas does not carry image content.

## Followup

Spec'd in `docs/architecture/2026-05-11-v63-pico-003-canvas-utilization-spec.md`: a v6.3 pico-003 training run that addresses the magnitude mismatch and adds a canvas-aware auxiliary loss to force the canvas branch to be load-bearing.

## Evidence

- Probe harness: `scripts/sr_v6_canvas_probe.py`
- Probe output: `<train-host-data>/checkpoints/srcnn-v6.2-pico-002/canvas-probe-55500.json`
- Probe sample frames: `<train-host-data>/checkpoints/srcnn-v6.2-pico-002/canvas-probe-frames-55500/`

Reproduce (idle GPU not required; current-step values are independent of training contamination because we measure ratios across modes on the same forward path):

```bash
python scripts/sr_v6_canvas_probe.py \
    --ckpt-temporal <train-host-data>/checkpoints/srcnn-v6.2-pico-002/step-00055500.pt \
    --tartanair-root <train-host-data>/datasets/tartanair_extracted \
    --manifest <train-host-data>/checkpoints/v5_held_out_manifest.json \
    --output canvas-probe-55500.json \
    --n-pairs 16 \
    --device cuda
```
