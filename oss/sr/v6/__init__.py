"""v6 — covariance-resampled online Gaussian-temporal SR.

The v6 architecture is documented in
``docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md``.
Implementation roadmap with concrete action items in
``docs/research/2026-05-05-v6-external-baselines-integration-plan.md``.

Module layout (this package, in build order):

  hat.py
    HAT spatial backbone (Hybrid Attention Transformer, Chen et al. CVPR
    2023). Three sized variants — HAT-Tiny (~1M, Pico), HAT-Small (~5M,
    Standard), HAT-Base / HAT-L (~15M / ~17M, Heavy). HAT-L matches the
    pretrained checkpoint published by GSASR (ChrisDud0257/GSASR) so the
    teacher can warm-start instead of pretraining from scratch.

    API contract:
      forward(x: (B, in_channels, H, W)) -> (B, feat_dim, H, W)
      Default in_channels=9 (RGB + depth + motion + normals; v6 drops the
      SRGD-era canvas hint), feat_dim=180 (HAT-L convention).

  cross_attention.py
    Pixel-Gaussian fusion: pixel features attend to Gaussian-canvas
    tokens. Window cross-attention with ROPE positional encoding,
    mirroring GSASR's ``fea2gsropeamp_arch.py``.

    API contract:
      forward(pixel_features: (B, feat_dim, H, W),
              gaussian_tokens:  (B, K, token_dim))
        -> (B, feat_dim, H, W)

  gaussian_spawner.py
    GRAPE-style point-wise decoder. A single 1x1 conv predicts Gaussian
    params from HAT features, then tile pooling emits per-batch canvas
    write-back proposals.

    API contract:
      forward(features: (B, feat_dim, H, W)) -> GaussianSpawnState
      with positions/scales/colors batched over K LR tiles.

  covariance_resampling.py
    GS-STVSR (arXiv:2604.18047) ``Sigma'_output = J_t Sigma_t J_t^T +
    Sigma_recon`` resampled covariance computation. Used inside the
    analytical warp to make the temporal pass anti-shimmering by
    construction.

  st_variation_score.py
    4DGS-1K (arXiv:2503.16422) Spatial-Temporal Variation Score pruning.
    Score = SS_i * TS_i; rank globally and prune bottom 60-80%.

  keyframe_active_mask.py
    4DGS-1K key-frame active-Gaussian mask. Per K=10 frames precompute
    a binary visibility mask so non-keyframes skip inactive Gaussians.

  model.py
    V6Model orchestrator wiring HAT + canvas + cross-attention + warp
    (with covariance resampling) + ST-score pruning + active-mask
    rasterizer. The integration site for everything above.

  dataset.py
    Extends ``oss.sr.gaussian_temporal.dataset.TrajectoryWindowDataset``
    to include Hypersim alongside TartanAir. 70/30
    importance/uniform patch sampling.

  losses.py
    Charbonnier + LPIPS + multi-scale VGG + wavelet L1 + Sobel edge +
    temporal consistency. GAN hinge loss; pixel-only warmup until
    step 20K.

  discriminator.py
    UNet discriminator (per Real-ESRGAN, ICCV-W 2021) for adversarial
    training. Per-pixel real/fake.

  ema.py
    Exponential moving average wrapper (beta=0.999). Teacher only.

  schedules.py
    Cosine LR with 3 warm restarts (T_0=50K, T_mult=1).

A separate training script ``scripts/sr_train_v6.py`` consumes this
package; the AA-stack rasterizer modifications (AAA-Gaussians +
AA-2DGS + Analytic-Splatting) layer on top after the v6-skeleton
training proves the architecture converges.
"""
from __future__ import annotations

from oss.sr.v6.gaussian_spawner import GaussianSpawner, GaussianSpawnState

__all__: list[str] = ["GaussianSpawner", "GaussianSpawnState"]
