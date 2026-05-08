# SPDX-License-Identifier: Apache-2.0
"""v6.2 latent decoder: R-latent splat + weight sum + reproject base -> delta RGB.

Per v6.2 arch spec section 3.2 step 9:
  I_t^HR(p) = I_base(p) + DeltaI(p)

This module computes DeltaI from the rasterizer's R-channel output.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from oss.sr.v6.activations import SqrSwish


class LatentDecoder(nn.Module):
    """1x1 -> depthwise 3x3 -> SqrSwish -> 1x1 -> RGB residual.

    Inputs:
        Z:      (B, R, H, W)  rasterizer R-latent splat
        m:      (B, 1, H, W)  per-pixel weight sum
        I_base: (B, 3, H, W)  reproject base

    Output: DeltaI (B, 3, H, W), added to I_base downstream.
    """

    def __init__(self, latent_R: int = 4, hidden: int = 32) -> None:
        super().__init__()
        if latent_R <= 0:
            raise ValueError(f"latent_R must be positive; got {latent_R}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive; got {hidden}")

        self.latent_R = int(latent_R)
        self.hidden = int(hidden)

        self.conv1 = nn.Conv2d(self.latent_R + 1 + 3, self.hidden, kernel_size=1)
        self.depthwise = nn.Conv2d(
            self.hidden,
            self.hidden,
            kernel_size=3,
            padding=1,
            groups=self.hidden,
        )
        self.act = SqrSwish()
        self.conv2 = nn.Conv2d(self.hidden, 3, kernel_size=1)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(
        self,
        Z: torch.Tensor,
        m: torch.Tensor,
        I_base: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = Z.shape
        if c != self.latent_R:
            raise ValueError(f"Z channels must be {self.latent_R}; got {c}")
        self._check_shape("m", m, (b, 1, h, w))
        self._check_shape("I_base", I_base, (b, 3, h, w))

        x = torch.cat([Z, m, I_base], dim=1)
        x = self.conv1(x)
        x = self.depthwise(x)
        x = self.act(x)
        return self.conv2(x)

    @staticmethod
    def _check_shape(
        name: str,
        tensor: torch.Tensor,
        expected: tuple[int, ...],
    ) -> None:
        if tensor.shape != expected:
            raise ValueError(
                f"{name} shape must be {expected}; got {tuple(tensor.shape)}"
            )


__all__ = ["LatentDecoder"]
