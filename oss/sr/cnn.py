"""OSS-SR CNN super-resolver — V0 backbone.

SRCNNSimple is a lightweight CNN that performs 2× single-image super-resolution
conditioned on G-buffer auxiliary channels.

Input layout (12 channels):
    [0:3]   LR RGB       (3 ch)
    [3:4]   depth        (1 ch)
    [4:6]   motion       (2 ch)
    [6:9]   normals      (3 ch)
    [9:12]  canvas_hint  (3 ch)

Architecture:
    head_conv  : Conv2d(12, hidden, 3, padding=1) → ReLU
    body       : N residual blocks  Conv→ReLU→Conv  with channel-wise skip
    upsample   : Conv2d(hidden, 3 * scale**2, 3, padding=1) → PixelShuffle(scale)
    bicubic skip: F.interpolate(lr, scale_factor=scale, mode='bicubic') added to output

The bicubic skip is a hard requirement: it provides a strong prior so the
network only has to learn the residual above bicubic, not reconstruct HR
from scratch.  Zero-init on the upsample conv bias ensures the first output
equals bicubic exactly (modulo floating-point rounding from the upsampled-LR
channel already included in the 12-ch stack).

Tier configurations
-------------------
    pico     : hidden=16, n_blocks=2   (~7 K params)
    lite     : hidden=32, n_blocks=4   (~47 K params)
    standard : hidden=64, n_blocks=8   (~306 K params)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Tier table: name → (hidden_channels, n_residual_blocks)
# ---------------------------------------------------------------------------

SR_TIER_CONFIGS: dict[str, tuple[int, int]] = {
    "pico": (16, 2),
    "lite": (32, 4),
    "standard": (64, 8),
}


class _ResBlock(nn.Module):
    """One plain residual block: Conv → ReLU → Conv + identity skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.relu(self.conv1(x), inplace=True))


class SRCNNSimple(nn.Module):
    """Small CNN super-resolver with G-buffer conditioning.

    Input:  (B, in_channels, h, w) — default 12-channel LR+G-buffer stack.
    Output: (B, 3, scale*h, scale*w) — HR RGB.

    The output is the PixelShuffle-upsampled feature map **plus** a bicubic
    upsample of the LR RGB channels [0:3].  At initialisation the upsample
    conv bias is zeroed so the additive residual starts at zero; early training
    therefore outputs ≈ bicubic.

    Args:
        in_channels:  Input channel count.  Default 12.
        scale:        Super-resolution scale factor.  Default 2.
        hidden:       Feature channel count in head + body.  Default 64.
        n_blocks:     Number of residual blocks in the body.  Default 8.
    """

    def __init__(
        self,
        in_channels: int = 12,
        scale: int = 2,
        hidden: int = 64,
        n_blocks: int = 8,
    ) -> None:
        super().__init__()
        if in_channels < 3:
            raise ValueError(f"in_channels must be >=3 (need at least RGB); got {in_channels}")
        if scale < 1:
            raise ValueError(f"scale must be >=1; got {scale}")
        if hidden < 1:
            raise ValueError(f"hidden must be >=1; got {hidden}")
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be >=0; got {n_blocks}")

        self.scale = scale
        self.in_channels = in_channels

        # Head: project input into feature space.
        self.head_conv = nn.Conv2d(in_channels, hidden, 3, padding=1)
        nn.init.kaiming_normal_(self.head_conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.head_conv.bias)

        # Body: stack of residual blocks.
        self.body = nn.Sequential(*[_ResBlock(hidden) for _ in range(n_blocks)])

        # Upsample tail: map to scale^2 * 3 channels, then PixelShuffle.
        self.upsample_conv = nn.Conv2d(hidden, 3 * scale * scale, 3, padding=1)
        # Depth-aware small init. The body's residual blocks accumulate
        # variance roughly with sqrt(n_blocks); the upsample-conv weight scale
        # has to compensate so the final residual stays in ±~0.05 magnitude
        # regardless of depth. Without this, deeper networks (e.g. standard
        # tier at n_blocks=8) hit clamp(0,1) at the output, which kills the
        # clamp gradient and freezes training. Lite (n_blocks=4) was fine at
        # std=0.01 because the body magnitude there is small enough.
        # Empirically: residual std ≈ 0.01 * sqrt(hidden) * sqrt(n_blocks).
        # Cap at 0.05 magnitude → std = 0.05 / (sqrt(hidden) * sqrt(n_blocks)).
        depth_safe_std = 0.05 / max(1.0, (hidden ** 0.5) * (max(1, n_blocks) ** 0.5))
        nn.init.normal_(self.upsample_conv.weight, mean=0.0, std=depth_safe_std)
        nn.init.zeros_(self.upsample_conv.bias)

        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, in_channels, h, w) — 12-channel LR+G-buffer stack.
               The first 3 channels must be LR RGB (used for bicubic skip).

        Returns:
            (B, 3, scale*h, scale*w) HR output, NOT clamped — caller should
            apply .clamp(0, 1) after loss if desired.
        """
        lr_rgb = x[:, :3, :, :]   # (B, 3, h, w) for bicubic skip

        feat = F.relu(self.head_conv(x), inplace=True)
        feat = self.body(feat)
        residual = self.pixel_shuffle(self.upsample_conv(feat))  # (B, 3, sh, sw)

        # Bicubic skip: gives the network a strong baseline to add onto.
        bicubic = F.interpolate(
            lr_rgb, scale_factor=self.scale, mode="bicubic", antialias=True
        )

        return bicubic + residual


def srcnn_for_tier(
    tier: str,
    in_channels: int = 12,
    scale: int = 2,
) -> SRCNNSimple:
    """Factory: instantiate SRCNNSimple for a hardware tier.

    Args:
        tier:        One of "pico", "lite", "standard".
        in_channels: Input channel count (default 12).
        scale:       SR scale factor (default 2).

    Returns:
        SRCNNSimple configured for the requested tier.

    Raises:
        ValueError: On unknown tier name.
    """
    if tier not in SR_TIER_CONFIGS:
        raise ValueError(
            f"Unknown SR tier {tier!r}. Valid choices: {sorted(SR_TIER_CONFIGS)}"
        )
    hidden, n_blocks = SR_TIER_CONFIGS[tier]
    return SRCNNSimple(in_channels=in_channels, scale=scale, hidden=hidden, n_blocks=n_blocks)


__all__ = ["SRCNNSimple", "srcnn_for_tier", "SR_TIER_CONFIGS"]
