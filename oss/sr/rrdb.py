"""OSS-SR RRDB backbone — V1 quality tier.

Implements a simplified RRDB (Residual-in-Residual Dense Block) network for
single-image super-resolution, based on the ESRGAN architecture (Wang et al.,
2018).  This is NOT heavily optimised — it is a correct, readable reference
implementation that brings proven ESRGAN-quality inductive biases to OSS-SR
with the same 12-channel G-buffer input as SRCNNSimple.

Reference: BasicSR / ESRGAN — Real-ESRGAN (Wang et al. 2021).
Status:    V1 candidate.  Use SRCNNSimple for production V0 training; switch to
           SRRRDB via ``--sr-backbone rrdb`` once SRCNNSimple baseline is solid.

Architecture
------------
Input  : (B, 12, h, w)    — 12-channel LR+G-buffer stack.
          First 3 channels must be LR RGB for the bicubic skip.
Output : (B, 3, 2h, 2w)   — HR RGB (unclamped; caller clamps to [0, 1]).

Modules
-------
    head_conv  : Conv2d(12, hidden, 3, padding=1)
    body       : n_rrdb RRDB blocks (each = 3 densely-connected residual units)
    upsample   : Conv2d(hidden, 3 * scale**2, 3, padding=1) → PixelShuffle
    bicubic skip: added to output (same as SRCNNSimple)

Default: 6 RRDB blocks, hidden=64, residual scaling beta=0.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dense block (3 conv layers with growing concatenated inputs)
# ---------------------------------------------------------------------------


class _DenseUnit(nn.Module):
    """One dense residual unit: 5-layer dense connection within hidden channels.

    Based on RDB in BasicSR — simplified to 5 layers and a single growth size
    equal to hidden//4 to keep parameter count manageable.
    """

    def __init__(self, in_channels: int, growth: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + 0 * growth, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + 1 * growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth, in_channels, 3, padding=1)
        for m in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu", a=0.2)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(torch.cat([x, x1], 1)), 0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(torch.cat([x, x1, x2], 1)), 0.2, inplace=True)
        x4 = F.leaky_relu(self.conv4(torch.cat([x, x1, x2, x3], 1)), 0.2, inplace=True)
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x   # residual scaling beta=0.2


# ---------------------------------------------------------------------------
# RRDB: Residual-in-Residual Dense Block (3 nested dense units)
# ---------------------------------------------------------------------------


class _RRDBBlock(nn.Module):
    """RRDB block: 3 DenseUnits with a global residual (beta=0.2 scaling)."""

    def __init__(self, channels: int, growth: int = 32) -> None:
        super().__init__()
        self.rdb1 = _DenseUnit(channels, growth)
        self.rdb2 = _DenseUnit(channels, growth)
        self.rdb3 = _DenseUnit(channels, growth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


# ---------------------------------------------------------------------------
# SRRRDB — top-level module
# ---------------------------------------------------------------------------


class SRRRDB(nn.Module):
    """ESRGAN-style RRDB network for single-image super-resolution.

    Based on BasicSR's RRDBNet; not heavily optimised.  Same 12-channel
    G-buffer input contract as SRCNNSimple; same bicubic skip tail.

    Args:
        in_channels:  Input channel count.  Default 12.
        scale:        Super-resolution scale factor.  Default 2.
        hidden:       Feature channel count in head + body.  Default 64.
        n_rrdb:       Number of RRDB blocks.  Default 6.
        growth:       Growth channel count inside each DenseUnit.  Default 32.
    """

    def __init__(
        self,
        in_channels: int = 12,
        scale: int = 2,
        hidden: int = 64,
        n_rrdb: int = 6,
        growth: int = 32,
    ) -> None:
        super().__init__()
        if in_channels < 3:
            raise ValueError(f"in_channels must be >=3; got {in_channels}")
        if scale < 1:
            raise ValueError(f"scale must be >=1; got {scale}")
        if hidden < 1:
            raise ValueError(f"hidden must be >=1; got {hidden}")
        if n_rrdb < 0:
            raise ValueError(f"n_rrdb must be >=0; got {n_rrdb}")

        self.scale = scale
        self.in_channels = in_channels

        # Head.
        self.head_conv = nn.Conv2d(in_channels, hidden, 3, padding=1)
        nn.init.kaiming_normal_(self.head_conv.weight, nonlinearity="leaky_relu", a=0.2)
        nn.init.zeros_(self.head_conv.bias)

        # RRDB body.
        self.body = nn.Sequential(*[_RRDBBlock(hidden, growth) for _ in range(n_rrdb)])

        # Post-body trunk conv (ESRGAN style: one extra conv after all RRDBs).
        self.trunk_conv = nn.Conv2d(hidden, hidden, 3, padding=1)
        nn.init.kaiming_normal_(self.trunk_conv.weight, nonlinearity="leaky_relu", a=0.2)
        nn.init.zeros_(self.trunk_conv.bias)

        # Upsample tail.
        self.upsample_conv = nn.Conv2d(hidden, 3 * scale * scale, 3, padding=1)
        nn.init.kaiming_normal_(self.upsample_conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.upsample_conv.bias)

        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, in_channels, h, w) — first 3 channels are LR RGB.

        Returns:
            (B, 3, scale*h, scale*w) — unclamped HR output.
        """
        lr_rgb = x[:, :3, :, :]

        feat = self.head_conv(x)
        trunk = self.trunk_conv(self.body(feat))
        feat = feat + trunk   # global residual (ESRGAN trunk skip)

        residual = self.pixel_shuffle(self.upsample_conv(feat))

        bicubic = F.interpolate(
            lr_rgb, scale_factor=self.scale, mode="bicubic", antialias=True
        )
        return bicubic + residual


__all__ = ["SRRRDB"]
