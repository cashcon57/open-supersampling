# 2026-05-08 — Phase 4 Elegance D: sigma cull radius

## Question

Is a `2sigma` tile/AABB cull acceptable compared with the current `3sigma` convention?

## Method

Used the isotropic 2D Gaussian radial CDF `F(r)=1-exp(-r^2/(2sigma^2))` and bounded feature-space mass loss.

## Inputs

- Radii: `2sigma`, `2.5sigma`, `3sigma`, `sqrt(12)sigma`
- Feature magnitude reference: `|feat| <= 3`
- Script: `tests/cuda/perf-math/d_sigma_cull.py`

## Output

- `2sigma`: retains `86.466%`, drops `13.534%`, L-inf bound at feat=3 is `0.4060`.
- `2.5sigma`: retains `95.606%`, drops `4.394%`, bound `0.1318`.
- `3sigma`: retains `98.889%`, drops `1.111%`, bound `0.0333`.
- Recommendation: reject `2sigma`; keep `3sigma` default. `2.5sigma` is only a quality-flag candidate.

## Reproducibility

```bash
./venv-py312/bin/python tests/cuda/perf-math/d_sigma_cull.py
```
