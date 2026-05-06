# oss.sr.v6 — covariance-resampled online Gaussian-temporal SR

Per-module index. See `__init__.py` for the canonical reference and
`docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md` for
the full architectural design.

## Build order

1. `hat.py` — HAT spatial backbone (HAT-Tiny / HAT-Small / HAT-L)
2. `cross_attention.py` — pixel↔Gaussian fusion
3. `covariance_resampling.py` — GS-STVSR resampled Σ
4. `st_variation_score.py` — 4DGS-1K pruning score
5. `keyframe_active_mask.py` — 4DGS-1K key-frame active mask
6. `losses.py` + `discriminator.py` — Charbonnier + LPIPS + multi-scale VGG + wavelet L1 + Sobel edge + GAN hinge with UNetD
7. `ema.py` + `schedules.py` — EMA + cosine LR + warm restarts
8. `dataset.py` — TartanAir + Hypersim, 70/30 importance/uniform sampling
9. `model.py` — V6Model orchestrator
10. `scripts/sr_train_v6.py` — training entry point (lives outside this package)

## API contracts

| Module | Forward signature |
|---|---|
| `hat.HAT(in_channels, feat_dim, depth, num_heads, window_size)` | `(B, in_channels, H, W) -> (B, feat_dim, H, W)` |
| `cross_attention.PixelGaussianFusion(feat_dim, token_dim, num_heads)` | `((B, feat_dim, H, W), (B, K, token_dim)) -> (B, feat_dim, H, W)` |
| `covariance_resampling.resample_covariance(Sigma_t, J_t, Sigma_recon)` | pure-functional, see GS-STVSR §3.2 |
| `losses.V6CompositeLoss` | `(pred, target, prev_pred?, ...) -> dict[str, Tensor]` |
| `discriminator.UNetDiscriminator` | `(B, 3, H, W) -> (B, 1, H, W)` per-pixel real/fake |

## Test layout

`tests/sr/v6/test_<module>.py` — one test file per module. Smoke + shape +
gradient tests at minimum. End-to-end V6Model smoke gets its own file
once `model.py` is written.

## Out of scope for v6 (deferred to v6.1+)

- HDR-aware perceptual loss (training corpus is currently 8-bit sRGB)
- Per-Gaussian time-MLP heads
- Native 4D primitives (4D-Rotor)
- Gaussian Frosting
- GRTX (relevant to OSS-RG, not SR)
