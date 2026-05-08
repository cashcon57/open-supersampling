# 2026-05-08 — Phase 4 Elegance G: edge-only tile rendering

## Question

What fraction of LR tiles are edge/high-frequency tiles that would still need full Gaussian rasterization?

## Method

No v6.1 checkpoint was locally mounted. The fallback script runs an untrained v6.1-pico-shaped HAT-Tiny backbone, computes Sobel magnitude over LR feature energy, average-pools by LR tile, and thresholds at the 70th percentile. The script accepts `--ckpt` for the real training-host run.

## Inputs

- Local ckpt status: no loadable v6/v6.1 checkpoint.
- Fallback input: random `1x9x64x64`, seed 1.
- Threshold: 70th-percentile tile Sobel magnitude.
- Script: `tests/cuda/perf-math/g_edge_tiles.py`
- Artifact: `docs/coordination/phase4-elegance-artifacts/g_edge_tiles_hist.png`

## Output

- Mode: `synthetic_fallback`
- Tiles: `64`
- Threshold: `0.42431368231773375`
- Edge fraction: `0.296875`
- Flat fraction: `0.703125`
- Recommendation: promising only as a measurement surface. Needs real LR feature maps and then Tier 3 quality testing for flat-tile bilinear replacement.

## Reproducibility

```bash
PYTHONPATH=. ./venv-py312/bin/python tests/cuda/perf-math/g_edge_tiles.py --height 64 --width 64 --device cpu
```
