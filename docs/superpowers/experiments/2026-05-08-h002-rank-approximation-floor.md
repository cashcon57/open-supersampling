# 2026-05-08 - H002 Test A2: rank approximation floor

## Question

How much 64-channel splat feature error is introduced when the rasterizer emits
only rank `R` channels and a post-raster linear decode reconstructs `F=64`?

This isolates the approximation floor in H002 before any learned decoder,
checkpoint retraining, or RGB perceptual metric is involved.

## Method

- Located the existing F=64-capable rasterizer path:
  `oss.gaussian.renderer.rasterizer.Rasterizer`.
- Used the conic-capable reference path,
  `Rasterizer._render_reference(..., conic=...)`, as the correctness anchor.
- Ran a vectorized synthetic validation with the same sum-splat equation:

  ```text
  Y(p) = sum_g w_g(p) f_g
  Z_R(p) = sum_g w_g(p) ((f_g - mean(f)) B_R)
  Y_R(p) = Z_R(p) B_R^T + sum_g w_g(p) mean(f)
  ```

- Primary result uses a random-init decoder basis `B in R^{64 x R}` with a
  Moore-Penrose projection. The implementation samples Gaussian `B`, QR
  orthonormalizes the same column space for numerical conditioning, computes
  `z = feat @ Q`, rasterizes `z`, and decodes with `Q.T`.
- Oracle PCA is reported separately as the best same-fixture linear basis. This
  is an optimistic floor: a learned or fixed projection can do worse, but cannot
  beat this linear rank-R reconstruction on the same feature distribution.
- PSNR is feature-space PSNR against the full 64-channel raster, using the
  reference output max-min as `data_range`.

## Fixture

| Field | Value |
|---|---:|
| Seed | `663554` |
| Splats | `4096` |
| Output | `64x64` |
| Full channels | `64` |
| Ranks | `4, 8, 16, 32` |
| Device | `cpu` |
| dtype | `torch.float32` |

Gaussian positions were uniform in the output extent. Conics were random
positive-definite ellipses derived from random axes in `[1.2, 5.5]` pixels and
random rotation in `[0, pi]`.

Sanity check: the vectorized conic renderer matched
`Rasterizer._render_reference` on a small fixture with max absolute error
`7.15e-07`.

## Case 1: iid Gaussian features

`feat ~ N(0, 1)`, independent across all 64 channels.

Requested random-basis pseudo-inverse projection:

| Rank | Feature energy | PSNR vs full F=64 | MSE | Relative RMSE |
|---:|---:|---:|---:|---:|
| 4 | 0.0624 | 20.359 dB | 30.2206 | 0.9664 |
| 8 | 0.1260 | 20.565 dB | 28.8221 | 0.9438 |
| 16 | 0.2540 | 21.359 dB | 24.0068 | 0.8614 |
| 32 | 0.4981 | 23.127 dB | 15.9807 | 0.7028 |

Oracle PCA basis on the same fixture:

| Rank | Feature energy | PSNR vs full F=64 | MSE | Relative RMSE |
|---:|---:|---:|---:|---:|
| 4 | 0.0764 | 20.411 dB | 29.8635 | 0.9607 |
| 8 | 0.1504 | 20.794 dB | 27.3450 | 0.9193 |
| 16 | 0.2907 | 21.638 dB | 22.5121 | 0.8341 |
| 32 | 0.5526 | 23.518 dB | 14.6044 | 0.6718 |

Result: iid features are effectively high-rank. Even the best rank-32 linear
projection leaves large feature error. This is a negative control and confirms
that H002 is not an identity transformation for arbitrary F=64 features.

## Case 2: structured correlated features

`feat = latent_2 @ M + 0.1 * noise`, with Gaussian latent/noise. This represents
a learned feature tensor that has actually collapsed into a low-dimensional
contract.

Requested random-basis pseudo-inverse projection:

| Rank | Feature energy | PSNR vs full F=64 | MSE | Relative RMSE |
|---:|---:|---:|---:|---:|
| 4 | 0.0630 | 19.293 dB | 28.0511 | 0.9740 |
| 8 | 0.1014 | 19.483 dB | 26.8557 | 0.9531 |
| 16 | 0.2410 | 20.410 dB | 21.6919 | 0.8566 |
| 32 | 0.5125 | 22.080 dB | 14.7652 | 0.7067 |

Oracle PCA basis on the same fixture:

| Rank | Feature energy | PSNR vs full F=64 | MSE | Relative RMSE |
|---:|---:|---:|---:|---:|
| 4 | 0.9872 | 37.688 dB | 0.4060 | 0.1172 |
| 8 | 0.9910 | 39.244 dB | 0.2837 | 0.0980 |
| 16 | 0.9942 | 41.090 dB | 0.1855 | 0.0792 |
| 32 | 0.9977 | 45.542 dB | 0.0665 | 0.0474 |

Result: low intrinsic rank is not enough if the decoder basis is random and
unaligned; the random subspace still misses most signal. Once the basis is
aligned with the intrinsic feature manifold, rasterizing rank 8 and decoding
after accumulation preserves the full raster well. Extra rank mostly captures
the injected noise.

## Verdict

Test A2 validates the algebraic contract but also shows the hard quality
condition clearly:

- Rank projection commutes with sum-splat rasterization when the same linear
  basis is used before and after rasterization.
- A random-init basis is not a useful compression baseline: at R=8 it reaches
  only `20.565 dB` on iid features and `19.483 dB` on structured features.
- Low rank is not safe for arbitrary 64-channel splat features.
- H002 is viable only if training both makes the per-splat feature distribution
  intrinsically low-rank and aligns the encoder/decoder basis to that manifold.
  In this synthetic floor, the oracle rank-8 basis reaches `39.244 dB` feature
  PSNR on rank-2-plus-noise features; iid Gaussian features reach only
  `20.794 dB` even with oracle PCA.

Recommendation: proceed only as a retraining/bottleneck ablation. Pair this
memo with the real-checkpoint SVD in `2026-05-08-h002-svd-rank-analysis.md`,
then run the full R={4,8,16,32} bottleneck retrain against RGB PSNR/LPIPS. Do
not treat low-rank rasterization as a drop-in compression of the current F=64
state.

## Reproducibility

The validation was run from repo root with `./venv-py312/bin/python` as an
inline heredoc. No temporary helper file was retained.

Implementation details needed to reproduce the run:

- Build weights as `exp(-0.5 * (a dx^2 + 2 b dx dy + d dy^2))` for all
  `64x64` pixels and `4096` splats.
- Render full features with `weights @ feat`.
- For random-basis rows, sample `B_R = torch.randn(64, R)`, use QR to preserve
  the same random column space with stable conditioning, and project with
  `z = feat @ Q`.
- For oracle rows, fit `B_R` by CPU `torch.linalg.svd(feat - feat.mean(0))`.
- Render rank features with `weights @ z`.
- Decode with `Z_R @ B_R.T + (weights @ ones) * mean` when centering is used.
- Compute PSNR against the full `F=64` raster with reference max-min data
  range.
