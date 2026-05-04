"""Analytical Gaussian warp — μ shift + covariance Jacobian transform.

Reuses ``oss.gaussian.canvas.warp.warp_positions`` for the mean shift +
in-frame mask. Adds a 2×2 Jacobian sample at each Gaussian's mean and
applies ``Σ' = J Σ Jᵀ`` analytically.

Σ is parameterized as scale = exp(log_scale) along axes rotated by ``rotation``.
After warping we re-decompose JΣJᵀ via 2×2 SVD to recover the new (axis-aligned)
log_scale + rotation. Pure PyTorch.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from oss.gaussian.canvas.warp import warp_positions
from oss.sr.gaussian_temporal.gaussian_field import GaussianField


def _sample_jacobian(motion: torch.Tensor, mu: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Per-Gaussian 2x2 Jacobian J = I + ∂(motion)/∂(x,y) at each mean.

    Args:
        motion: (2, H, W).
        mu:     (N, 2).
        hw:     (H, W).

    Returns:
        (N, 2, 2) tensor — J = I + grad(motion).
    """
    n = mu.shape[0]
    h, w = hw
    if n == 0:
        return torch.zeros((0, 2, 2), device=motion.device, dtype=motion.dtype)

    # Finite-difference gradient of motion (same shape as motion).
    # We use forward differences with replicate padding at the borders.
    motion_b = motion.unsqueeze(0)  # (1, 2, H, W)
    pad = F.pad(motion_b, (1, 1, 1, 1), mode="replicate")
    dmdx = (pad[..., 1:-1, 2:] - pad[..., 1:-1, :-2]) * 0.5  # (1, 2, H, W)
    dmdy = (pad[..., 2:, 1:-1] - pad[..., :-2, 1:-1]) * 0.5

    # Sample dmdx, dmdy at each mu via grid_sample.
    x_norm = (mu[:, 0] / w) * 2.0 - 1.0
    y_norm = (mu[:, 1] / h) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, n, 1, 2)
    sx = F.grid_sample(dmdx, grid, mode="bilinear", padding_mode="border", align_corners=False)
    sy = F.grid_sample(dmdy, grid, mode="bilinear", padding_mode="border", align_corners=False)
    sx = sx[0, :, :, 0].t()  # (N, 2)
    sy = sy[0, :, :, 0].t()  # (N, 2)

    # J = I + [[du/dx, du/dy], [dv/dx, dv/dy]]
    j = torch.eye(2, device=motion.device, dtype=motion.dtype).unsqueeze(0).expand(n, -1, -1).clone()
    j[:, 0, 0] += sx[:, 0]
    j[:, 1, 0] += sx[:, 1]
    j[:, 0, 1] += sy[:, 0]
    j[:, 1, 1] += sy[:, 1]
    return j


def _decompose_covariance(field_log_scale: torch.Tensor, field_rotation: torch.Tensor) -> torch.Tensor:
    """Reconstruct Σ from (log_scale, rotation) -> (N, 2, 2)."""
    n = field_log_scale.shape[0]
    s = torch.exp(field_log_scale)  # (N, 2)
    cos = torch.cos(field_rotation)
    sin = torch.sin(field_rotation)
    r = torch.stack([
        torch.stack([cos, -sin], dim=-1),
        torch.stack([sin, cos], dim=-1),
    ], dim=-2)  # (N, 2, 2)
    s_diag = torch.diag_embed(s)  # (N, 2, 2)
    rs = r @ s_diag
    return rs @ rs.transpose(-1, -2)


def _recompose_covariance(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """SVD-decompose 2×2 Σ into (log_scale, rotation).

    The SVD returns singular values in descending order; we canonicalize by
    swapping when ``|u[0,0]| < |u[0,1]|`` so the first axis stays closest to
    +x. This preserves ``(log_scale, rotation)`` under identity warps for
    axis-aligned inputs (otherwise SVD silently swaps axes).
    """
    u, s, _ = torch.linalg.svd(sigma)
    # If the first eigenvector is closer to +y than +x, swap axes so the
    # parameterization stays near-identity for near-axis-aligned covariances.
    swap = u[:, 0, 0].abs() < u[:, 0, 1].abs()
    s0 = torch.where(swap, s[:, 1], s[:, 0])
    s1 = torch.where(swap, s[:, 0], s[:, 1])
    s_ordered = torch.stack([s0, s1], dim=-1)
    # Build the corresponding rotation matrix's first column.
    u00 = torch.where(swap, u[:, 0, 1], u[:, 0, 0])
    u10 = torch.where(swap, u[:, 1, 1], u[:, 1, 0])
    log_scale = 0.5 * torch.log(s_ordered.clamp(min=1e-8))  # eigenvalues are scale^2
    rotation = torch.atan2(u10, u00)
    return log_scale, rotation


def warp_field(field: GaussianField, motion: torch.Tensor, hw: tuple[int, int]) -> GaussianField:
    """Apply analytical warp to the Gaussian field.

    Args:
        field:  GaussianField to warp (NOT mutated).
        motion: (2, H, W) per-pixel motion vectors (dx, dy).
        hw:     (H, W) of motion field.

    Returns:
        New GaussianField with warped (mu, log_scale, rotation). alive
        flag is ANDed with in-frame mask from ``warp_positions``.
    """
    out = field.clone()
    new_mu, in_frame = warp_positions(field.mu, motion, hw=hw)
    out.mu = new_mu
    out.alive = field.alive & in_frame

    j = _sample_jacobian(motion, field.mu, hw=hw)  # (N, 2, 2)
    sigma = _decompose_covariance(field.log_scale, field.rotation)
    new_sigma = j @ sigma @ j.transpose(-1, -2)
    new_log_scale, new_rotation = _recompose_covariance(new_sigma)
    out.log_scale = new_log_scale
    out.rotation = new_rotation
    return out


__all__ = ["warp_field"]
