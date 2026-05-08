# 2026-05-08 — Phase 4 Elegance K: quantized Gaussian state

## Question

Can Gaussian state use narrower representations without meaningful quality loss?

## Method

Computed first-order quantization bounds for xy, rotation, and scale. Real validation still needs trained scale/aniso/rotation distributions.

## Inputs

- Frame reference: `1920x1080`
- xy: int16 over max dimension 1920
- rot: 256 levels over `2pi`
- scale: fp16 relative precision
- Script: `tests/cuda/perf-math/k_quantized_state.py`

## Output

- xy int16 step: `0.05859553819391461 px`; half-step `0.029297769096957305 px`.
- rot uint8 step: `0.02454369260617026 rad = 1.40625 deg`; half-step `0.01227184630308513 rad`.
- fp16 relative precision: `0.0009765625`.
- Caveat: screen-wide fp16 xy is not equivalent to int16 fixed-point xy. At 4K, fp16 pixel centers can exceed 1 px half-ULP near the far edge; prefer fp32 xy or tile-local/fixed-point centers for geometry.
- Recommendation: xy int16 and fp16 scale are plausible; int8 rotation should be flag-only for anisotropic Gaussians until real ckpt stats bound `sx/sy`.

## Reproducibility

```bash
./venv-py312/bin/python tests/cuda/perf-math/k_quantized_state.py
```
