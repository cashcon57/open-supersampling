"""Backward warp + LR→HR motion-vector upsample for the v5 pixel temporal track.

Convention: motion vectors are forward flow ``t-1 → t`` (LR pixel
displacements). At each pixel ``p`` of the current frame, the corresponding
prev-frame location is ``p − motion_hr(p)``, so ``warp_prev_hr`` does
``F.grid_sample(prev_hr, base_grid − motion_hr)`` (backward / pull warp).

This matches the dataset adapters: TartanAir's ``flow/NN_NM_flow.npy`` is
forward flow from frame N to frame N+1; Sintel's ``.flo`` files likewise
store forward flow.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def upsample_motion_to_hr(motion_lr: torch.Tensor, scale: int) -> torch.Tensor:
    """Bilinearly upsample LR motion to HR resolution and rescale magnitudes.

    Args:
        motion_lr: (B, 2, H_lr, W_lr) LR-pixel displacements (channel 0 = x, 1 = y).
        scale:     HR / LR ratio (positive int).

    Returns:
        (B, 2, scale*H_lr, scale*W_lr) HR-pixel displacements.
    """
    if motion_lr.dim() != 4 or motion_lr.shape[1] != 2:
        raise ValueError(f"motion_lr must be (B, 2, H, W); got {tuple(motion_lr.shape)}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1; got {scale}")
    motion_hr = F.interpolate(motion_lr, scale_factor=float(scale), mode="bilinear", align_corners=False)
    return motion_hr * float(scale)


def warp_prev_hr(prev_hr: torch.Tensor, motion_lr: torch.Tensor, scale: int) -> torch.Tensor:
    """Backward-warp prev-HR to align with current view via motion vectors.

    Args:
        prev_hr:   (B, 3, H_hr, W_hr).
        motion_lr: (B, 2, H_lr, W_lr) LR-pixel current→previous displacements.
        scale:     HR / LR ratio.

    Returns:
        (B, 3, H_hr, W_hr) prev_hr resampled at the current frame's pixel grid.

    Out-of-frame samples use ``padding_mode='border'`` (clamp to edge), which
    is the safest default for the head — it'll learn to mask via disocclusion.
    """
    if prev_hr.dim() != 4 or prev_hr.shape[1] != 3:
        raise ValueError(f"prev_hr must be (B, 3, H, W); got {tuple(prev_hr.shape)}")
    b, _, h_hr, w_hr = prev_hr.shape
    motion_hr = upsample_motion_to_hr(motion_lr, scale=scale)
    if motion_hr.shape[-2:] != (h_hr, w_hr):
        raise ValueError(
            f"motion HR shape {tuple(motion_hr.shape[-2:])} != prev_hr {(h_hr, w_hr)}"
        )

    # Build base grid in HR pixel coords, then add HR motion.
    yy, xx = torch.meshgrid(
        torch.arange(h_hr, device=prev_hr.device, dtype=prev_hr.dtype),
        torch.arange(w_hr, device=prev_hr.device, dtype=prev_hr.dtype),
        indexing="ij",
    )
    base_x = xx.unsqueeze(0).expand(b, -1, -1)
    base_y = yy.unsqueeze(0).expand(b, -1, -1)
    # Forward flow t-1 → t: at current pixel p, prev location is p − flow(p).
    sample_x = base_x - motion_hr[:, 0]
    sample_y = base_y - motion_hr[:, 1]

    # Normalize to [-1, 1] for grid_sample with align_corners=False:
    #   normalized = (2*pixel + 1) / N - 1
    norm_x = (2.0 * sample_x + 1.0) / w_hr - 1.0
    norm_y = (2.0 * sample_y + 1.0) / h_hr - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)  # (B, H_hr, W_hr, 2)

    return F.grid_sample(
        prev_hr,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


__all__ = ["upsample_motion_to_hr", "warp_prev_hr"]
