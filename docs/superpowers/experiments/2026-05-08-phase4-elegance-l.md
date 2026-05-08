# 2026-05-08 — Phase 4 Elegance L: decoupled feature compression

## Question

Can canvas/token features be compressed independently from geometry without final RGB quality loss?

## Method

Deferred. This is an information-loss change through the learned composite head; closed-form Gaussian math does not decide it.

## Inputs

- Proposed fixture: held-out 64-frame TartanAir set.
- Metrics: feature-space reconstruction error, final PSNR, LPIPS, MS-SSIM.
- Script: `tests/cuda/perf-math/l_feat_compression_plan.py`

## Output

- Tier reached: Tier 3 plan only.
- Recommendation: defer to Phase 4-frametest.

## Reproducibility

```bash
./venv-py312/bin/python tests/cuda/perf-math/l_feat_compression_plan.py
```
