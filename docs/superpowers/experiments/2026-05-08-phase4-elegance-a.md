# 2026-05-08 — Phase 4 Elegance A: Pade exp approximation

## Question

Can `[4/4]` Pade replace `expf(-0.5q)` over the rasterizer's `q in [0, 9]` working range with <`1e-3` absolute weight error?

## Method

Derived the standard diagonal Pade form for `exp(x)` and sampled `q in [0, 9]` at 1,000,001 points with float64 NumPy.

## Inputs

- Domain: `q in [0, 9]`, `x = -0.5q`
- Script: `tests/cuda/perf-math/a_pade_exp.py`
- Ship bar: max absolute error < `1e-3`

## Output

- Numerator coefficients: `[1, 1/2, 3/28, 1/84, 1/1680]`
- Denominator coefficients: `[1, -1/2, 3/28, -1/84, 1/1680]`
- Max abs error: `5.833315084608631e-4`
- Worst point: `q = 9.0`
- Approx/truth at worst point: `0.01169232804670317` vs `0.011108996538242306`
- Operation shape: Horner evaluation is 4 multiply-add steps for numerator, 4 for denominator, plus one divide/reciprocal. That replaces the SFU `expf` call in the hot weight path.
- Recommendation: ship-with-flag first, then default after CUDA parity.

## Reproducibility

```bash
./venv-py312/bin/python tests/cuda/perf-math/a_pade_exp.py
```
