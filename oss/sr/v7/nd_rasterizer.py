"""Reference Python implementation of N-D Gaussian time-slice rasterization.

Each Gaussian carries:
  mean      shape (3,)   (x, y, t)
  cov       shape (3, 3) full 3D covariance, PSD
  feature   shape (R,)   feature/colour to splat (matches v6.2 latent_rank)
  opacity   shape ()      [0, 1]

To render at time t = t_query, for each Gaussian we condition on t = t_query
to get the marginal 2D Gaussian in (x, y) space (mean shift + Schur
complement on covariance), weighted by the t-axis exp(-0.5 (t_query - t_i)^2 / V_tt).
Those 2D Gaussians are then splatted onto an HR image via the standard
EWA / image-gs path (here we use a simple Python splatter for correctness;
the production v7 path uses gsplat or a custom CUDA kernel).

This module is a SLOW correctness reference. v7-production rasterizer
will mirror this math in a CUDA / Triton kernel with LSH culling at 3D
where AABB starts failing. The math here is what the kernel must match.

Notation:
  V = [[V_xx, V_xy, V_xt],
       [V_xy, V_yy, V_yt],
       [V_xt, V_yt, V_tt]]
  V_xy_block = top-left 2x2  = [[V_xx, V_xy], [V_xy, V_yy]]
  V_xt_vec   = (V_xt, V_yt)  shape (2,)
  V_tt       = scalar
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class NDGaussian:
    """Convenience container for a single 3D Gaussian.

    Production code packs many of these into batched tensors; this exists
    for tests + clarity.
    """
    mean: torch.Tensor       # (3,) -- (x, y, t)
    cov: torch.Tensor        # (3, 3) -- PSD
    feature: torch.Tensor    # (R,)
    opacity: torch.Tensor    # scalar in [0, 1]


def time_marginal(
    mean: torch.Tensor,      # (..., 3)
    cov: torch.Tensor,       # (..., 3, 3)
    t_query: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the 2D (x, y) Gaussian conditional on t = t_query.

    Returns:
        mean_xy_cond  (..., 2)   shifted mean
        cov_xy_cond   (..., 2, 2) Schur-complement reduced covariance
        weight_t      (...,)      the t-axis Gaussian falloff weight
                                  = exp(-0.5 * (t_query - mean_t)^2 / V_tt)
                                  / sqrt(2 * pi * V_tt)
                                  -- the marginal density at t = t_query
    """
    # Split mean
    mean_xy = mean[..., :2]
    mean_t = mean[..., 2]

    # Split covariance into 2x2 + 2x1 + 1
    cov_xy_block = cov[..., :2, :2]
    cov_xt_vec = cov[..., :2, 2]        # (..., 2)
    cov_tt = cov[..., 2, 2]              # (...)

    # Avoid div-by-zero if V_tt is degenerate
    cov_tt_safe = cov_tt.clamp(min=1e-12)

    # Mean shift: mu_xy + V_xt / V_tt * (t_query - mu_t)
    delta_t = (float(t_query) - mean_t)               # (...,)
    mean_xy_cond = mean_xy + cov_xt_vec * (delta_t / cov_tt_safe).unsqueeze(-1)

    # Cov reduction: V_xy - V_xt @ V_xt^T / V_tt
    outer = cov_xt_vec.unsqueeze(-1) @ cov_xt_vec.unsqueeze(-2)  # (..., 2, 2)
    cov_xy_cond = cov_xy_block - outer / cov_tt_safe.unsqueeze(-1).unsqueeze(-1)

    # t-axis falloff weight (normalized marginal density)
    log_weight = -0.5 * delta_t * delta_t / cov_tt_safe \
                 - 0.5 * (math.log(2.0 * math.pi) + cov_tt_safe.log())
    weight_t = log_weight.exp()
    return mean_xy_cond, cov_xy_cond, weight_t


def _evaluate_2d_gaussian_at_pixels(
    mean_xy: torch.Tensor,    # (N, 2)
    cov_xy: torch.Tensor,     # (N, 2, 2)
    feature: torch.Tensor,    # (N, R)
    weight: torch.Tensor,     # (N,)
    opacity: torch.Tensor,    # (N,)
    image_hw: Tuple[int, int],
) -> torch.Tensor:
    """Naive O(H*W*N) reference splatter. Returns (R, H, W) features."""
    H, W = image_hw
    R = feature.shape[-1]
    device = mean_xy.device
    dtype = feature.dtype

    # Build pixel grid (W, H, 2) -> we'll index as (H, W, 2)
    ys = torch.arange(H, device=device, dtype=mean_xy.dtype) + 0.5
    xs = torch.arange(W, device=device, dtype=mean_xy.dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

    accum = torch.zeros((R, H, W), device=device, dtype=torch.float32)

    for i in range(mean_xy.shape[0]):
        m = mean_xy[i]                 # (2,)
        V = cov_xy[i].to(torch.float32) # (2, 2)
        # 3-sigma AABB cull for the reference (purely to bound the
        # Python loop)
        eig = torch.linalg.eigvalsh(V).clamp(min=1e-6)
        radius = 3.0 * eig.max().sqrt().item()
        x0 = int(max(0, math.floor(m[0].item() - radius)))
        x1 = int(min(W, math.ceil(m[0].item() + radius)))
        y0 = int(max(0, math.floor(m[1].item() - radius)))
        y1 = int(min(H, math.ceil(m[1].item() + radius)))
        if x1 <= x0 or y1 <= y0:
            continue

        sub = pixels[y0:y1, x0:x1, :] - m   # (h, w, 2)
        # gaussian = exp(-0.5 * sub @ V_inv @ sub)
        V_inv = torch.linalg.inv(V)
        # quadratic form per pixel
        v0 = sub[..., 0]
        v1 = sub[..., 1]
        q = (v0 * v0 * V_inv[0, 0]
             + v1 * v1 * V_inv[1, 1]
             + 2.0 * v0 * v1 * V_inv[0, 1])
        # Anti-aliased EWA pre-filter would add a small isotropic
        # convolution to V before inversion; we skip here -- this is a
        # correctness reference, not aliasing-correct.
        g = (-0.5 * q).exp()        # (h, w)
        contrib = g * float(opacity[i].item()) * float(weight[i].item())
        feat_i = feature[i].to(torch.float32)        # (R,)
        accum[:, y0:y1, x0:x1] = accum[:, y0:y1, x0:x1] + \
            contrib.unsqueeze(0) * feat_i.view(R, 1, 1)

    return accum.to(dtype=dtype)


def render_nd_time_slice(
    means: torch.Tensor,           # (N, 3)
    covs: torch.Tensor,            # (N, 3, 3)
    features: torch.Tensor,        # (N, R)
    opacities: torch.Tensor,       # (N,)
    t_query: float,
    image_hw: Tuple[int, int],
    time_falloff: bool = True,
) -> torch.Tensor:
    """Top-level entry: render the N-D mixture at a specified time slice.

    Returns: (R, H, W) feature image. Caller decodes feature -> RGB.

    Args:
        time_falloff: when True (default), Gaussians are weighted by
            their t-axis marginal density at t_query (so a Gaussian
            centered at t_i contributes most strongly at t_query = t_i
            and falls off as |t_query - t_i| grows). When False, all
            Gaussians contribute regardless of t -- used for testing.
    """
    mean_xy_cond, cov_xy_cond, weight_t = time_marginal(means, covs, t_query)
    weights = weight_t if time_falloff else torch.ones_like(weight_t)
    return _evaluate_2d_gaussian_at_pixels(
        mean_xy=mean_xy_cond,
        cov_xy=cov_xy_cond,
        feature=features,
        weight=weights,
        opacity=opacities,
        image_hw=image_hw,
    )
