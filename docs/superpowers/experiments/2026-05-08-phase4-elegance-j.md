# 2026-05-08 — Phase 4 Elegance J: spatially varying budget

## Question

Can smooth/low-saliency regions run with fewer Gaussians without visible quality loss?

## Method

Deferred. Spatial budgets couple to the learned decoder and to scene-dependent texture/silhouette errors, so full-frame quality testing is required.

## Inputs

- Proposed fixture: held-out 64-frame TartanAir set.
- Metrics: PSNR, LPIPS, MS-SSIM, edge-region error, flat-region error.
- Script: `tests/cuda/perf-math/j_spatial_budget_plan.py`

## Output

- Tier reached: Tier 3 plan only.
- Recommendation: defer to Phase 4-frametest.

## Reproducibility

```bash
./venv-py312/bin/python tests/cuda/perf-math/j_spatial_budget_plan.py
```
