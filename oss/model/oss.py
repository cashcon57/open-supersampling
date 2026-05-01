"""ORU - Open Ray Upscaler.

Architecture (Thomas/Liktor HPG 2022 derivative):
- Mode-specific input head (one of three: rgb, rgb_aux, features)
- Shared encoder (3 down stages) + shared decoder (3 up stages)
- Bilinear upsample to scale_factor*input then 3x3 RGB projection
- Standalone-competitive against FSR/XeSS in `rgb` mode; `features` mode unlocks
  joint-architecture quality+perf when paired with ORD.
"""
from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, DownBlock, UpBlock
from .oss_rg import HANDOFF_FEATURE_CHANNELS


_TIER_CHANNELS = {
    "lite":     [16, 24, 32, 48],
    "standard": [24, 32, 48, 64],
    "heavy":    [32, 48, 64, 96],
}

_VALID_SCALES = (1.3, 1.5, 1.7, 2.0)


class OSS(nn.Module):
    def __init__(
        self,
        input_mode: Literal["rgb", "rgb_aux", "features"] = "rgb",
        scale_factor: float = 2.0,
        tier: Literal["lite", "standard", "heavy"] = "standard",
        aux_channels: int = 6,
    ):
        super().__init__()
        if scale_factor not in _VALID_SCALES:
            raise ValueError(f"scale_factor must be one of {_VALID_SCALES}, got {scale_factor}")
        self.input_mode = input_mode
        self.scale_factor = scale_factor
        c = _TIER_CHANNELS[tier]

        if input_mode == "rgb":
            head_in = 3 + 1 + 2
            self.head = ConvBlock(head_in, c[0])
        elif input_mode == "rgb_aux":
            head_in = 3 + 1 + 2 + aux_channels
            self.head = ConvBlock(head_in, c[0])
        elif input_mode == "features":
            head_in = HANDOFF_FEATURE_CHANNELS + 1 + 2
            self.head = nn.Sequential(
                ConvBlock(head_in, c[0] * 2),
                ConvBlock(c[0] * 2, c[0]),
            )
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        self.enc1 = DownBlock(c[0], c[1])
        self.enc2 = DownBlock(c[1], c[2])
        self.enc3 = DownBlock(c[2], c[3])

        self.dec3 = UpBlock(c[3], c[2])
        self.dec2 = UpBlock(c[2] * 2, c[1])
        self.dec1 = UpBlock(c[1] * 2, c[0])

        self.out_proj = nn.Conv2d(c[0] * 2, 3, 3, padding=1)

    def _build_input(self, color, depth, motion, aux, features):
        if self.input_mode == "rgb":
            assert color is not None
            return torch.cat([color, depth, motion], dim=1)
        if self.input_mode == "rgb_aux":
            assert color is not None and aux is not None
            return torch.cat([color, depth, motion, aux], dim=1)
        if self.input_mode == "features":
            assert features is not None
            return torch.cat([features.float(), depth, motion], dim=1)
        raise RuntimeError("unreachable")

    def forward(
        self,
        color: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        motion: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self._build_input(color, depth, motion, aux, features)
        x0 = self.head(x)

        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        d3 = self.dec3(x3)
        d2 = self.dec2(torch.cat([d3, x2], dim=1))
        d1 = self.dec1(torch.cat([d2, x1], dim=1))

        feats_full = torch.cat([d1, x0], dim=1)
        # Floor rather than banker's-round for deterministic round-trip with the
        # downsample step in the paired trainer (T5). E.g. HR=64, scale=2.0 →
        # LR=32 → ORU upscale → 64. Banker's rounding on odd inputs at
        # non-integer scales produces platform-dependent sizes.
        out_h = int(feats_full.shape[2] * self.scale_factor)
        out_w = int(feats_full.shape[3] * self.scale_factor)
        upscaled = F.interpolate(feats_full, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return self.out_proj(upscaled)
