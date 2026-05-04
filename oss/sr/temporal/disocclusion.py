"""Disocclusion mask for the v5 pixel temporal track.

Per design spec §Architecture point 4:
    disoccl = sigmoid(alpha * |warped_depth_prev - depth_curr| + beta * ||motion|| - gamma)

Default init: alpha=10.0, beta=2.0, gamma=4.0. Empirically these put the
sigmoid in a regime where small static differences map to ~0 and large
disparities saturate to ~1. The trainer will adjust them.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from oss.sr.temporal.warp import upsample_motion_to_hr, warp_prev_hr


def _warp_prev_depth(depth_prev: torch.Tensor, motion_lr: torch.Tensor, scale: int) -> torch.Tensor:
    """Warp single-channel prev depth using the same backward-warp as RGB.

    Implementation: replicate channel to 3, warp, take channel 0. This avoids
    a second specialized warp implementation. Bilinear sampling is fine for
    depth here — disocclusion is the supervision target, not the depth itself.
    """
    rep = depth_prev.expand(-1, 3, -1, -1).contiguous()
    warped = warp_prev_hr(rep, motion_lr, scale=scale)
    return warped[:, :1]


class DisocclusionGate(nn.Module):
    """Disocclusion mask producer with three learnable scalar gates."""

    def __init__(
        self,
        alpha_init: float = 10.0,
        beta_init: float = 2.0,
        gamma_init: float = 4.0,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(
        self,
        depth_curr: torch.Tensor,
        depth_prev: torch.Tensor,
        motion_lr: torch.Tensor,
        scale: int,
    ) -> torch.Tensor:
        """Produce HR disocclusion mask.

        Args:
            depth_curr: (B, 1, H_hr, W_hr) — current frame depth at HR
                        (caller upsamples LR depth ahead of time if needed).
            depth_prev: (B, 1, H_hr, W_hr) — previous frame depth at HR.
            motion_lr:  (B, 2, H_lr, W_lr).
            scale:      HR / LR ratio.

        Returns:
            (B, 1, H_hr, W_hr) mask in [0, 1].
        """
        if depth_curr.shape != depth_prev.shape:
            raise ValueError(
                f"depth_curr {tuple(depth_curr.shape)} != depth_prev {tuple(depth_prev.shape)}"
            )
        warped_depth_prev = _warp_prev_depth(depth_prev, motion_lr, scale=scale)
        depth_diff = (warped_depth_prev - depth_curr).abs()  # (B, 1, H_hr, W_hr)

        motion_hr = upsample_motion_to_hr(motion_lr, scale=scale)
        motion_mag = motion_hr.norm(dim=1, keepdim=True)  # (B, 1, H_hr, W_hr)

        logit = self.alpha * depth_diff + self.beta * motion_mag - self.gamma
        return torch.sigmoid(logit)


__all__ = ["DisocclusionGate"]
