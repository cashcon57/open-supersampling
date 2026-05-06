"""Analytical motion warp for the v6 Gaussian canvas.

The persistent v6 canvas stores Gaussian centers in HR pixel coordinates
plus a 2D covariance parameterization as principal-axis ``scales`` and an
in-plane ``rotation``. Per frame we sample the LR engine motion field at
each Gaussian center after upsampling it to the HR output grid, then apply
the GS-STVSR covariance resampling equation

    Sigma' = J Sigma J^T + Sigma_recon

where ``J = I + grad(motion)`` at the Gaussian center.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from oss.sr.v6.covariance_resampling import resample_covariance
from oss.sr.v6.model import CanvasState


_IDENTITY_J_TOL = 1.0e-5
_EIGEN_EPS = 1.0e-12


def _validate_canvas(canvas: CanvasState) -> int:
    count = int(canvas.count)
    if count < 0:
        raise ValueError(f"canvas.count must be non-negative; got {canvas.count}")
    if canvas.positions.ndim != 2 or canvas.positions.shape[-1] != 2:
        raise ValueError(f"canvas.positions must be (N, 2); got {tuple(canvas.positions.shape)}")
    if canvas.scales.ndim != 2 or canvas.scales.shape[-1] != 2:
        raise ValueError(f"canvas.scales must be (N, 2); got {tuple(canvas.scales.shape)}")
    if canvas.rotations.ndim != 1:
        raise ValueError(f"canvas.rotations must be (N,); got {tuple(canvas.rotations.shape)}")
    if canvas.opacities.ndim != 1:
        raise ValueError(f"canvas.opacities must be (N,); got {tuple(canvas.opacities.shape)}")
    if canvas.colors.ndim != 2:
        raise ValueError(f"canvas.colors must be (N, F); got {tuple(canvas.colors.shape)}")

    lengths = (
        canvas.positions.shape[0],
        canvas.scales.shape[0],
        canvas.rotations.shape[0],
        canvas.opacities.shape[0],
        canvas.colors.shape[0],
    )
    if count > min(lengths):
        raise ValueError(f"canvas.count={count} exceeds tensor lengths {lengths}")
    return count


def _validate_motion(motion_lr: torch.Tensor, output_hw: Tuple[int, int]) -> tuple[int, int]:
    if motion_lr.ndim != 4 or motion_lr.shape[1] != 2:
        raise ValueError(f"motion_lr must be (B, 2, H, W); got {tuple(motion_lr.shape)}")
    if motion_lr.shape[0] < 1:
        raise ValueError("motion_lr batch dimension must be non-empty")
    h_hr, w_hr = int(output_hw[0]), int(output_hw[1])
    if h_hr <= 0 or w_hr <= 0:
        raise ValueError(f"output_hw must be positive; got {output_hw}")
    return h_hr, w_hr


def _motion_to_hr(motion_lr: torch.Tensor, output_hw: Tuple[int, int]) -> torch.Tensor:
    """Return the first shared-canvas motion field at HR resolution.

    ``CanvasState`` is unbatched, so v6 currently has one persistent canvas
    per rank. When a batched motion tensor is supplied, the first element is
    the only unambiguous field for that shared canvas.
    """
    h_hr, w_hr = output_hw
    motion = motion_lr[:1]
    if tuple(motion.shape[-2:]) != (h_hr, w_hr):
        motion = F.interpolate(motion, size=(h_hr, w_hr), mode="bilinear", align_corners=False)
    return motion[0]


def _grid_from_xy(xy: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    h, w = hw
    x_norm = (xy[:, 0] / float(w)) * 2.0 - 1.0
    y_norm = (xy[:, 1] / float(h)) * 2.0 - 1.0
    return torch.stack([x_norm, y_norm], dim=-1).view(1, xy.shape[0], 1, 2)


def _sample_motion(motion_hr: torch.Tensor, xy: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    n = xy.shape[0]
    if n == 0:
        return xy.new_zeros((0, 2))

    sample_dtype = torch.float32 if motion_hr.dtype in (torch.float16, torch.bfloat16) else motion_hr.dtype
    motion_sample = motion_hr.to(dtype=sample_dtype)
    xy_sample = xy.to(device=motion_hr.device, dtype=sample_dtype)
    grid = _grid_from_xy(xy_sample, hw)
    sampled = F.grid_sample(
        motion_sample.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.view(2, n).t().to(dtype=xy.dtype)


def _sample_jacobian(motion_hr: torch.Tensor, xy: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Sample ``J = I + grad(motion)`` at each Gaussian center."""
    n = xy.shape[0]
    if n == 0:
        return torch.zeros((0, 2, 2), device=motion_hr.device, dtype=torch.float32)

    motion_f = motion_hr.to(dtype=torch.float32)
    xy_f = xy.to(device=motion_hr.device, dtype=torch.float32)
    pad = F.pad(motion_f.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    dmdx = (pad[..., 1:-1, 2:] - pad[..., 1:-1, :-2]) * 0.5
    dmdy = (pad[..., 2:, 1:-1] - pad[..., :-2, 1:-1]) * 0.5

    grid = _grid_from_xy(xy_f, hw)
    sx = F.grid_sample(dmdx, grid, mode="bilinear", padding_mode="border", align_corners=False)
    sy = F.grid_sample(dmdy, grid, mode="bilinear", padding_mode="border", align_corners=False)
    sx = sx[0, :, :, 0].t()
    sy = sy[0, :, :, 0].t()

    j = torch.eye(2, device=motion_hr.device, dtype=torch.float32).expand(n, -1, -1).clone()
    j[:, 0, 0] += sx[:, 0]
    j[:, 1, 0] += sx[:, 1]
    j[:, 0, 1] += sy[:, 0]
    j[:, 1, 1] += sy[:, 1]
    return j


def _covariance_from_scales_rotations(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    scales_f = scales.to(dtype=torch.float32)
    rotations_f = rotations.to(dtype=torch.float32)
    cos = torch.cos(rotations_f)
    sin = torch.sin(rotations_f)
    r = torch.stack(
        [
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1),
        ],
        dim=-2,
    )
    sigma_axis = torch.diag_embed(scales_f.clamp(min=0.0).square())
    return r @ sigma_axis @ r.transpose(-1, -2)


def _scales_rotations_from_covariance(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form 2x2 symmetric eigendecomposition.

    Returns the largest principal axis first. This avoids the bf16/autocast
    hazards of running ``torch.linalg`` directly in low precision.
    """
    sigma_f = 0.5 * (sigma.to(dtype=torch.float32) + sigma.to(dtype=torch.float32).transpose(-1, -2))
    a = sigma_f[:, 0, 0]
    b = sigma_f[:, 0, 1]
    c = sigma_f[:, 1, 1]
    half_trace = 0.5 * (a + c)
    half_diff = 0.5 * (a - c)
    radius = torch.sqrt(half_diff.square() + b.square() + _EIGEN_EPS)

    eig0 = (half_trace + radius).clamp(min=_EIGEN_EPS)
    eig1 = (half_trace - radius).clamp(min=_EIGEN_EPS)
    scales = torch.sqrt(torch.stack([eig0, eig1], dim=-1))
    rotations = 0.5 * torch.atan2(2.0 * b, a - c)
    return scales, rotations


def _identity_j_mask(jacobian: torch.Tensor) -> torch.Tensor:
    n = jacobian.shape[0]
    if n == 0:
        return torch.zeros((0,), dtype=torch.bool, device=jacobian.device)
    eye = torch.eye(2, device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0)
    return (jacobian - eye).abs().reshape(n, -1).amax(dim=-1) < _IDENTITY_J_TOL


def warp_canvas(
    canvas: CanvasState,
    motion_lr: torch.Tensor,
    output_hw: tuple[int, int],
    sigma_recon: torch.Tensor | float = 0.5,
) -> CanvasState:
    """Warp canvas positions by motion vectors and resample covariances.

    Args:
        canvas: v6 ``CanvasState``. Only the first ``canvas.count`` entries
            are considered live and returned.
        motion_lr: ``(B, 2, H_lr, W_lr)`` engine motion vectors at LR.
            The current canvas contract is unbatched; the first batch item
            supplies the shared canvas motion field.
        output_hw: ``(H_hr, W_hr)`` output/canvas resolution.
        sigma_recon: isotropic scalar or ``(..., 2, 2)`` reconstruction
            covariance passed to ``resample_covariance``.

    Returns:
        A fresh ``CanvasState`` with out-of-frame Gaussians dropped.
    """
    h_hr, w_hr = _validate_motion(motion_lr, output_hw)
    output_hw = (h_hr, w_hr)
    count = _validate_canvas(canvas)

    positions = canvas.positions[:count]
    scales = canvas.scales[:count]
    rotations = canvas.rotations[:count]
    opacities = canvas.opacities[:count]
    colors = canvas.colors[:count]

    if count == 0:
        return CanvasState(
            positions=positions.clone(),
            scales=scales.clone(),
            rotations=rotations.clone(),
            opacities=opacities.clone(),
            colors=colors.clone(),
            count=0,
        )

    motion_hr = _motion_to_hr(motion_lr, output_hw)
    sampled_motion = _sample_motion(motion_hr, positions, output_hw)
    new_positions = positions + sampled_motion.to(device=positions.device, dtype=positions.dtype)
    in_frame = (
        (new_positions[:, 0] >= 0)
        & (new_positions[:, 0] < float(w_hr))
        & (new_positions[:, 1] >= 0)
        & (new_positions[:, 1] < float(h_hr))
    )

    jacobian = _sample_jacobian(motion_hr, positions, output_hw)
    identity_mask = _identity_j_mask(jacobian)
    new_scales = scales
    new_rotations = rotations
    non_identity = ~identity_mask
    if bool(non_identity.any()):
        sigma_old = _covariance_from_scales_rotations(scales[non_identity], rotations[non_identity])
        jacobian_f = jacobian[non_identity].to(dtype=torch.float32)
        if isinstance(sigma_recon, torch.Tensor):
            recon = sigma_recon.to(device=sigma_old.device, dtype=torch.float32)
        else:
            recon = float(sigma_recon)
        sigma_new = resample_covariance(sigma_old, jacobian_f, recon)
        decomposed_scales, decomposed_rotations = _scales_rotations_from_covariance(sigma_new)
        new_scales = scales.clone()
        new_rotations = rotations.clone()
        new_scales[non_identity] = decomposed_scales.to(device=scales.device, dtype=scales.dtype)
        new_rotations[non_identity] = decomposed_rotations.to(device=rotations.device, dtype=rotations.dtype)

    return CanvasState(
        positions=new_positions[in_frame],
        scales=new_scales[in_frame],
        rotations=new_rotations[in_frame],
        opacities=opacities[in_frame],
        colors=colors[in_frame],
        count=int(in_frame.sum().item()),
    )


__all__ = ["warp_canvas"]
