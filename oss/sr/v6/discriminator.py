"""UNet discriminator (Real-ESRGAN style).

Reference: Wang, Xie, Dong, Shan. "Real-ESRGAN: Training Real-World Blind
Super-Resolution with Pure Synthetic Data." ICCV-W 2021. The UNet
discriminator gives per-pixel real/fake predictions instead of a single
scalar — empirically more stable for SR / restoration GANs and yields
better local sharpness without the artifact-amplification of a global
PatchGAN.

Architecture
------------

  encoder:    3 -> 64 -> 128 -> 256          (stride-2 downsample at each step)
  bottleneck: 256 -> 512                     (stride-2 downsample)
  decoder:    512 -> 256 -> 128 -> 64        (bilinear upsample, skip-add)
  head:       64 -> 1                        (per-pixel logit)

Spectral normalization is applied to every conv. Output shape is
``(B, 1, H, W)`` matching the input spatial size.

bf16 safety: pure conv / leakyrelu — works under autocast with no special
handling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _sn_conv(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Module:
    return spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))


class _DownBlock(nn.Module):
    """Stride-2 conv + leaky relu (encoder block)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = _sn_conv(in_ch, out_ch, k=4, s=2, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv(x), 0.2, inplace=True)


class _UpBlock(nn.Module):
    """Bilinear upsample + 3×3 conv + leaky relu (decoder block)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = _sn_conv(in_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return F.leaky_relu(self.conv(x), 0.2, inplace=True)


class UNetDiscriminator(nn.Module):
    """UNet discriminator with spectral normalization (Real-ESRGAN, ICCV-W 2021).

    Args:
        in_channels: Number of input channels. Default 3 for RGB.
        base_channels: Width of the first encoder stage. Default 64. Subsequent
            stages double up to 8× at the bottleneck.

    Forward returns per-pixel logits of shape ``(B, 1, H, W)``.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels
        # Stem: 3×3 conv keeps spatial size, lifts to base_channels.
        self.stem = _sn_conv(in_channels, c, k=3, s=1, p=1)

        # Encoder.
        self.down1 = _DownBlock(c, c * 2)         # 64 -> 128, /2
        self.down2 = _DownBlock(c * 2, c * 4)     # 128 -> 256, /4

        # Bottleneck.
        self.bottleneck = _DownBlock(c * 4, c * 8)  # 256 -> 512, /8

        # Decoder. Channel counts are post-skip (concat would double; we
        # use add-style skip after a 1×1 channel match for simplicity and
        # to keep the param count tidy).
        self.up1 = _UpBlock(c * 8, c * 4)         # 512 -> 256, *2
        self.up2 = _UpBlock(c * 4, c * 2)         # 256 -> 128, *2
        self.up3 = _UpBlock(c * 2, c)             # 128 -> 64, *2

        # Per-pixel logit head.
        self.head = _sn_conv(c, 1, k=3, s=1, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = F.leaky_relu(self.stem(x), 0.2, inplace=True)  # (B, 64, H, W)
        s1 = self.down1(s0)                                  # (B, 128, H/2, W/2)
        s2 = self.down2(s1)                                  # (B, 256, H/4, W/4)
        b = self.bottleneck(s2)                              # (B, 512, H/8, W/8)

        u1 = self.up1(b)                                     # (B, 256, H/4, W/4)
        u1 = u1 + s2
        u2 = self.up2(u1)                                    # (B, 128, H/2, W/2)
        u2 = u2 + s1
        u3 = self.up3(u2)                                    # (B, 64, H, W)
        u3 = u3 + s0

        return self.head(u3)


__all__ = ["UNetDiscriminator"]
