# 2026-05-08 — Issue #11 closeout: R=16 default landed

**Status:** RESOLVED.

## Summary

Issue #11 blocked v6.2-pico-002 launch on the H002 SVD rank gate: top-8
centered singular-value energy on v6.1-pico-001 features measured 0.9479
(below the 0.95 gate). Top-16 hit 0.9918, comfortably above gate.

## Resolution

`V6Config.latent_rank` introduced with default **16**:

```python
latent_rank: int = 16   # gate-passing default; <= token_dim
```

Validation raises `ValueError` if `latent_rank` is non-positive or exceeds
`token_dim`. The rasterizer's `feature_dim` automatically equals
`latent_rank` whenever `latent_rank < token_dim`, so the composite head's
input channel count tracks it.

## Cross-check

- `V6Model(V6Config()).rasterizer.feature_dim == 16` ✓
- `tests/sr/v6/test_v62_mode_integration.py::test_v62_construction_uses_new_modules`
  asserts the rasterizer wires R=16 ✓
- 281-test v6 suite green; no regressions from the default switch.

## Follow-up

Pico-002 will also run an R=8 ablation branch later if and only if a
real bottleneck retraining step recovers >= the R=16 quality on RGB PSNR
+ LPIPS held-out. That experiment is downstream of pico-002 launch, not
a blocker.
