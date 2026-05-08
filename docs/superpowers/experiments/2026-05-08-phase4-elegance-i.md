# 2026-05-08 — Phase 4 Elegance I: precomputed tile masks

## Question

Can tile-id lists be cached across frames when Gaussian state drifts little?

## Method

No consecutive v6.1 frame tensors/checkpoints were locally mounted. The fallback script computes 3sigma tile AABBs for spawned Gaussians, applies synthetic center/scale drift for four transitions, and records exact AABB stability. Real validation should replace synthetic drift with five consecutive frames from a v6.1 checkpoint run.

## Inputs

- Local ckpt status: no loadable v6/v6.1 checkpoint.
- Fallback input: random `1x9x64x64`, seed 2.
- Drift fallback: 0.25 px/frame center noise and 0.5%/frame scale noise.
- Script: `tests/cuda/perf-math/i_tile_mask_drift.py`

## Output

- Mode: `synthetic_fallback`
- Gaussians: `64`
- Same-AABB fractions per transition: `[0.078125, 0.21875, 0.25, 0.25]`
- Mean cache-hit estimate: `0.19921875`
- Recommendation: do not ship from fallback data. Implement only after real consecutive-frame stats show a high AABB hit rate.

## Reproducibility

```bash
PYTHONPATH=. ./venv-py312/bin/python tests/cuda/perf-math/i_tile_mask_drift.py --height 64 --width 64 --device cpu
```
