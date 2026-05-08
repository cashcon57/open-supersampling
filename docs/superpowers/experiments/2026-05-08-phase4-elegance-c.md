# 2026-05-08 — Phase 4 Elegance C: LUT exp

## Question

Can a 256-entry LUT with linear interpolation replace `expf(-0.5q)` over `q in [0, 9]`?

## Method

Used the interpolation error bound `(Delta q)^2/8 * max |f''(q)|`, with `f(q)=exp(-0.5q)` and `max |f''|=0.25`, then sampled the LUT error.

## Inputs

- Entries: 256
- Domain: `q in [0, 9]`
- Storage: 512 bytes as fp16
- Script: `tests/cuda/perf-math/c_lut_exp.py`

## Output

- `Delta q = 0.03529411764705882`
- Analytical max abs error bound: `3.8927335640138406e-5`
- Measured max abs error: `3.858570372528014e-5`
- Worst sampled point: `q = 0.017622`
- Recommendation: ship if LUT is resident in constant/shared memory. Error is below bf16/fp16 accumulation noise.

## Reproducibility

```bash
./venv-py312/bin/python tests/cuda/perf-math/c_lut_exp.py
```
