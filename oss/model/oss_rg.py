"""ORD - Open Ray Denoiser.

Architecture:
- Two-branch encoder (MUNet I3D 2025 layout): noisy radiance branch + aux
  G-buffer branch, late-fused at the bottleneck.
- U-Net body with 3 down/up stages.
- Kernel-prediction head (NPPD-style) producing denoised RGB.
- Penultimate feature tensor exposed as paired-mode handoff (32 ch FP16).

Tiered weight sets:
- "lite":     channels [16, 24, 32, 48], <200K params, RDNA2/PS5/Series X target
- "standard": channels [24, 32, 48, 64], <1M params, coop_matrix HW target
- "heavy":    channels [32, 48, 64, 96], <4M params, flagship target
"""
from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn

from .blocks import ConvBlock, DownBlock, UpBlock, KernelPredictionHead


_TIER_CHANNELS = {
    "lite":     [16, 24, 32, 48],
    "standard": [24, 32, 48, 64],
    "heavy":    [32, 48, 64, 96],
}

# Frozen handoff contract value (also defined in ors/handoff/contract.py - Task 4).
HANDOFF_FEATURE_CHANNELS = 32


class OSSRG(nn.Module):
    """Open Ray Denoiser.

    Inputs
    ------
    noisy : (B, 3, H, W) float32  - 1 spp noisy radiance (linear).
    aux : (B, 11, H, W) float32   - albedo(3) + normal(3) + depth(1) + roughness(1) +
                                    spec_hit_distance(1) + motion(2).
    history : (B, 3, H, W) float32 - prior frame's denoised RGB, motion-vector-warped
                                     externally (SVGF-style upstream reprojection).

    Outputs
    -------
    rgb : (B, 3, H, W) float32           - denoised RGB at native render resolution.
    features : (B, 32, H, W) float16     - penultimate-layer features for paired handoff.
    """

    def __init__(self, tier: Literal["lite", "standard", "heavy"] = "standard"):
        super().__init__()
        self.tier = tier
        c = _TIER_CHANNELS[tier]

        self.radiance_in = ConvBlock(6, c[0])           # noisy(3) + history(3)
        self.aux_in = ConvBlock(11, c[0])

        self.fuse = ConvBlock(c[0] * 2, c[0])
        self.enc1 = DownBlock(c[0], c[1])
        self.enc2 = DownBlock(c[1], c[2])
        self.enc3 = DownBlock(c[2], c[3])

        self.dec3 = UpBlock(c[3], c[2])
        self.dec2 = UpBlock(c[2] * 2, c[1])             # +skip
        self.dec1 = UpBlock(c[1] * 2, c[0])             # +skip

        # Penultimate feature projection - frozen handoff contract
        self.feature_proj = nn.Conv2d(c[0] * 2, HANDOFF_FEATURE_CHANNELS, 1)

        self.kpn = KernelPredictionHead(HANDOFF_FEATURE_CHANNELS, kernel_size=5)

    def forward(self, noisy: torch.Tensor, aux: torch.Tensor, history: torch.Tensor):
        rad = self.radiance_in(torch.cat([noisy, history], dim=1))
        aux_f = self.aux_in(aux)
        x0 = self.fuse(torch.cat([rad, aux_f], dim=1))

        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        d3 = self.dec3(x3)
        d2 = self.dec2(torch.cat([d3, x2], dim=1))
        d1 = self.dec1(torch.cat([d2, x1], dim=1))

        feats_full = torch.cat([d1, x0], dim=1)
        features = self.feature_proj(feats_full).to(torch.float16)

        # KPN runs in float for training stability; features stay FP16 for export.
        rgb = self.kpn(features.float(), noisy)
        return rgb, features
