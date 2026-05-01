"""Warp pipeline for ORU-FX: three-tier fallback.

Tier 1 (best):   game motion vectors available → extrapolate + depth-aware warp
Tier 2 (good):   no motion vectors, depth available → RAFT-Small flow + depth-aware warp
Tier 3 (compat): color only → RAFT-Small flow + luminance-edge warp (GFFE-equivalent)

All tiers produce:
  warped  (B, 3, H, W)  — color frame warped to t+alpha
  depth   (B, 1, H, W)  — depth at render time, HR-scaled; zeros in Tier 3

Depth and motion vectors arrive at LR resolution (native render res).
Before warping, they are scaled to match the HR color frame:
  motion_vec_HR = motion_vec_LR * scale_factor  (bicubic resize + multiply)
  depth_HR      = F.interpolate(depth_LR, size=hr_size, mode="nearest")

The caller is responsible for providing the correct tier based on what the
DLL hook / Vulkan layer was able to extract at Present() time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _grid_sample_warp(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp frame using flow with bilinear sampling and border padding.

    Args:
        frame: (B, C, H, W)
        flow:  (B, 2, H, W) in pixels, (dx, dy)

    Returns:
        warped: (B, C, H, W)
    """
    B, C, H, W = frame.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=frame.device),
        torch.arange(W, dtype=torch.float32, device=frame.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)
    sample_grid = grid + flow                                   # (B, 2, H, W)
    # Normalize to [-1, 1]
    sample_grid[:, 0] = 2.0 * sample_grid[:, 0] / (W - 1) - 1.0
    sample_grid[:, 1] = 2.0 * sample_grid[:, 1] / (H - 1) - 1.0
    sample_grid = sample_grid.permute(0, 2, 3, 1)  # (B, H, W, 2)
    return F.grid_sample(frame, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)


def _depth_discontinuity_mask(depth: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Return mask of pixels near depth edges (True = discontinuity = unreliable warp).

    Uses max-pool to detect neighbors with very different depth.
    """
    d_max = F.max_pool2d(depth, kernel_size=3, stride=1, padding=1)
    d_min = -F.max_pool2d(-depth, kernel_size=3, stride=1, padding=1)
    return (d_max - d_min) > threshold  # (B, 1, H, W)


def _luminance_edge_mask(color: torch.Tensor, threshold: float = 0.15) -> torch.Tensor:
    """Luminance Laplacian edge mask — Tier 3 fallback when no depth available."""
    lum = 0.2126 * color[:, 0:1] + 0.7152 * color[:, 1:2] + 0.0722 * color[:, 2:3]
    laplacian_k = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=lum.dtype, device=lum.device
    ).view(1, 1, 3, 3)
    lap = F.conv2d(lum, laplacian_k, padding=1).abs()
    return lap > threshold


def _scale_to_hr(
    tensor: torch.Tensor, hr_h: int, hr_w: int, mode: str = "bilinear"
) -> torch.Tensor:
    return F.interpolate(tensor, size=(hr_h, hr_w), mode=mode, align_corners=False if mode == "bilinear" else None)


def warp_with_motion_vectors(
    color_hr: torch.Tensor,
    depth_lr: torch.Tensor,
    motion_vec_lr: torch.Tensor,
    alpha: float | torch.Tensor,
    scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tier 1: game motion vectors + depth.

    Args:
        color_hr:      (B, 3, H, W) full-res color from ORU-Pico output
        depth_lr:      (B, 1, h, w) LR depth from DLL hook
        motion_vec_lr: (B, 2, h, w) LR motion vectors (dx, dy in pixels at LR res)
        alpha:         scalar or (B,) temporal offset
        scale_factor:  upscale ratio (e.g. 2.0 for 2x SR)

    Returns:
        warped: (B, 3, H, W)
        depth:  (B, 1, H, W)  HR depth for SCN input
    """
    B, C, H, W = color_hr.shape
    a = float(alpha) if not isinstance(alpha, torch.Tensor) else alpha.mean().item()

    # Scale motion vectors to HR pixel space
    flow_hr = _scale_to_hr(motion_vec_lr * scale_factor, H, W, mode="bilinear") * a

    # Scale depth to HR (nearest to preserve sharp edges)
    depth_hr = F.interpolate(depth_lr, size=(H, W), mode="nearest")

    # Mask disoccluded pixels (depth discontinuities)
    disc_mask = _depth_discontinuity_mask(depth_hr)  # (B, 1, H, W)

    warped = _grid_sample_warp(color_hr, flow_hr)

    # Zero out warped color at discontinuities — SCN fills these
    warped = warped.masked_fill(disc_mask.expand_as(warped), 0.0)

    return warped, depth_hr


def warp_with_estimated_flow(
    color_hr: torch.Tensor,
    color_prev_hr: torch.Tensor,
    depth_lr: torch.Tensor | None,
    alpha: float | torch.Tensor,
    scale_factor: float,
    flow_estimator: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tier 2: RAFT-Small flow estimation + optional depth.

    Args:
        color_hr:       (B, 3, H, W) current frame (full-res)
        color_prev_hr:  (B, 3, H, W) previous frame (full-res)
        depth_lr:       (B, 1, h, w) or None
        alpha:          temporal offset
        scale_factor:   upscale ratio
        flow_estimator: pretrained RAFT-Small (frozen), expects HR inputs

    Returns:
        warped: (B, 3, H, W)
        depth:  (B, 1, H, W)  zeros if depth_lr is None
    """
    B, C, H, W = color_hr.shape
    a = float(alpha) if not isinstance(alpha, torch.Tensor) else alpha.mean().item()

    with torch.no_grad():
        # RAFT-Small: input range [0, 255] uint8-equivalent, returns list of flows
        flow_list = flow_estimator(
            (color_prev_hr * 255).clamp(0, 255),
            (color_hr * 255).clamp(0, 255),
        )
        flow = flow_list[-1]  # finest resolution, (B, 2, H, W)

    flow_extrap = flow * a

    if depth_lr is not None:
        depth_hr = F.interpolate(depth_lr, size=(H, W), mode="nearest")
        disc_mask = _depth_discontinuity_mask(depth_hr)
    else:
        depth_hr = torch.zeros(B, 1, H, W, device=color_hr.device, dtype=color_hr.dtype)
        disc_mask = _luminance_edge_mask(color_hr)

    warped = _grid_sample_warp(color_hr, flow_extrap)
    warped = warped.masked_fill(disc_mask.expand_as(warped), 0.0)

    return warped, depth_hr


def warp_color_only(
    color_hr: torch.Tensor,
    color_prev_hr: torch.Tensor,
    alpha: float | torch.Tensor,
    flow_estimator: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tier 3: color-only GFFE-equivalent fallback.

    Returns:
        warped: (B, 3, H, W)
        depth:  (B, 1, H, W)  always zeros
    """
    return warp_with_estimated_flow(
        color_hr, color_prev_hr, depth_lr=None, alpha=alpha,
        scale_factor=1.0, flow_estimator=flow_estimator,
    )
