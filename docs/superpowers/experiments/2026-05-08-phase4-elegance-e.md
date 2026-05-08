# 2026-05-08 — Phase 4 Elegance E: far-field q skip

## Question

Can the rasterizer skip Gaussian contributions where `q > 12`?

## Method

Bounded the skipped weight by `exp(-0.5q)` at the threshold.

## Inputs

- Threshold: `q > 12`
- Feature magnitude reference: `|feat| <= 3`
- Script: `tests/cuda/perf-math/e_far_field_skip.py`

## Output

- `exp(-6) = 0.0024787521766663585`
- Per-Gaussian L-inf contribution bound at feat=3: `0.0074362565299990755`
- This is not bit-exact if the contribution is zeroed.
- If `3sigma` AABB construction is restored, `q > 12` is mostly redundant for axis-aligned isotropic support because `3sigma` corresponds to `q=9`.
- Recommendation: ship-with-flag for the current full-frame native CUDA path, or supersede it by returning to pair-list AABB culling.

## Reproducibility

```bash
./venv-py312/bin/python tests/cuda/perf-math/e_far_field_skip.py
```
