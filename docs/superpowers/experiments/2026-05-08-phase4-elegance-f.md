# 2026-05-08 — Phase 4 Elegance F: top-K compositing

## Question

How many Gaussians per pixel capture 99% of the weight mass?

## Method

No v6.1 checkpoint was locally mounted. The reproducibility script therefore runs a deterministic untrained v6.1-pico-shaped fallback, computes per-pixel Gaussian weights, sorts them, and records `K` for 99% cumulative mass. The script accepts `--ckpt` for the real training-host run.

## Inputs

- Local ckpt status: no loadable v6/v6.1 checkpoint; only legacy `results/{oru,pico,ord,paired}` checkpoints found.
- Fallback input: random `1x9x32x32`, seed 0.
- Script: `tests/cuda/perf-math/f_topk_stats.py`
- Artifact: `docs/coordination/phase4-elegance-artifacts/f_topk_hist.png`

## Output

- Mode: `synthetic_fallback`
- Pixels: `4096`
- Gaussians considered: `16`
- `K99 p50/p95/p99`: `5 / 8 / 8`
- Fraction `K <= 4`: `0.315673828125`
- Fraction `K <= 8`: `0.993896484375`
- Recommendation: do not ship from fallback data. Real ckpt criterion remains: ship K=4 only if `K<=4` for >95% of pixels; otherwise consider K=8 or reject.

## Reproducibility

```bash
PYTHONPATH=. ./venv-py312/bin/python tests/cuda/perf-math/f_topk_stats.py --height 32 --width 32 --device cpu
```
