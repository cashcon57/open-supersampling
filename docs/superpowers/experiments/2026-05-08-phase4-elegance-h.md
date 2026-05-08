# 2026-05-08 — Phase 4 Elegance H: hierarchical 2K to 4K

## Question

Can a 2K intermediate followed by a cheaper 4K pass match direct 4K quality?

## Method

Deferred. This depends on the learned conv-head decoder and cannot be bounded from Gaussian math alone.

## Inputs

- Proposed fixture: held-out 64-frame TartanAir set.
- Metrics: PSNR, LPIPS, MS-SSIM, temporal error.
- Script: `tests/cuda/perf-math/h_hierarchical_plan.py`

## Output

- Tier reached: Tier 3 plan only.
- Recommendation: defer to Phase 4-frametest after Phase 4b/4c land.

## Reproducibility

```bash
./venv-py312/bin/python tests/cuda/perf-math/h_hierarchical_plan.py
```
