# 2026-05-08 — Phase 4 Elegance B: separable axis-aligned Gaussian

## Question

When can the rasterizer factor a Gaussian weight into independent x/y terms without visible error?

## Method

Substituted `cos(0)=1`, `sin(0)=0` into the conic equations and bounded the omitted cross term for small nonzero rotation.

## Inputs

- Conic: `a = c^2/sx^2 + s^2/sy^2`, `b = cs(1/sx^2 - 1/sy^2)`, `d = s^2/sx^2 + c^2/sy^2`
- Bound region: `|dx| <= 3sx`, `|dy| <= 3sy`
- Error target: `|delta weight| <= 1e-5`
- Script: `tests/cuda/perf-math/b_separable_gaussian.py`

## Output

- At exactly `rot=0`: `a=1/sx^2`, `b=0`, `d=1/sy^2`.
- `q = dx^2/sx^2 + dy^2/sy^2`.
- `exp(-0.5q)` factors exactly as `exp(-0.5dx^2/sx^2) * exp(-0.5dy^2/sy^2)`.
- Conservative cross-term bound: `|delta weight| <= 9 |sin(theta)cos(theta)| |sx/sy - sy/sx|`.
- Threshold examples:
  - `sx/sy=1.25`: `2.469e-6 rad`
  - `sx/sy=2`: `7.407e-7 rad`
  - `sx/sy=4`: `2.963e-7 rad`
- Recommendation: ship exact axis-aligned/isotropic fast path only. Do not use a broad small-rotation approximation without real ckpt rotation histograms.

## Reproducibility

```bash
./venv-py312/bin/python tests/cuda/perf-math/b_separable_gaussian.py
```
