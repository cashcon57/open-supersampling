# 2026-05-08 — T2 DGP dictionary distribution

## Question

Does random-init `DGPDictionary(M=16, feat_dim=64)` collapse softmax usage onto one covariance prototype for random Gaussian features?

## Method

Feed 10,000 seeded Gaussian feature vectors through the DGP weight head, compute `softmax(logits)`, average prototype weights across the batch, and gate the largest average usage at `<=25%`.

## Inputs

- Module: `oss.sr.v6.dgp_dictionary.DGPDictionary`
- Dictionary: `M=16`, `feat_dim=64`
- Features: `torch.randn(10_000, 64)`, seed `20260508`
- Device: CPU
- Test: `tests/sr/v6/test_disocclusion_spawner.py::test_dgp_random_feature_usage_is_not_top1_collapsed`

## Output

- Average prototype usage: `[0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]`
- Top-1 usage: `0.0625`
- Gate: `0.0625 <= 0.25`, pass

The current DGP reset path initializes weight-head weights and bias to zero, so random features produce uniform logits and uniform prototype usage at initialization. This is a deliberate non-collapse baseline; the test will fail if later init changes push a random dictionary into top-1 dominance above 25%.

## Reproducibility

```bash
PYTHONPATH=. ./venv-py312/bin/python -m pytest tests/sr/v6/test_disocclusion_spawner.py -q
```
