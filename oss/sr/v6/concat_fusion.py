# SPDX-License-Identifier: Apache-2.0
"""Concat-fusion module replacing global pixel-Gaussian cross-attention.

Per v6.2 arch spec section 3.2 step 6:

    F'(p) = F(p) + psi_theta([F(p), G(p), m(p), I_base(p), depth(p), MV(p)])
    psi_theta = 1x1 conv -> depthwise 3x3 -> SqrSwish -> 1x1 conv

Inputs:
  F:      (B, feat_dim, H, W)  HAT pixel features
  G:      (B, R, H, W)         rasterized canvas readout (R-latent splat)
  m:      (B, 1, H, W)         per-pixel sum of Gaussian weights
  I_base: (B, 3, H, W)         reproject base
  depth:  (B, 1, H, W)
  MV:     (B, 2, H, W)

Output: F' = F + psi_theta(concat(...)).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from oss.sr.v6.activations import SqrSwish


class ConcatFusion(nn.Module):
    """Residual concat-fusion block for v6.2 pixel/canvas feature exchange."""

    def __init__(
        self,
        feat_dim: int = 180,
        latent_R: int = 4,
        hidden: int = 64,
        mlp_ratio: float = 1.5,
    ) -> None:
        super().__init__()
        if feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive; got {feat_dim}")
        if latent_R <= 0:
            raise ValueError(f"latent_R must be positive; got {latent_R}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive; got {hidden}")

        self.feat_dim = feat_dim
        self.latent_R = latent_R
        self.hidden = hidden
        self.mlp_ratio = mlp_ratio

        in_channels = feat_dim + latent_R + 1 + 3 + 1 + 2
        self.proj_in = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.depthwise = nn.Conv2d(
            hidden, hidden, kernel_size=3, padding=1, groups=hidden
        )
        self.act = SqrSwish()
        self.proj_out = nn.Conv2d(hidden, feat_dim, kernel_size=1)

        # Start as an exact identity residual block.
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        F: torch.Tensor,
        G: torch.Tensor,
        m: torch.Tensor,
        I_base: torch.Tensor,
        depth: torch.Tensor,
        MV: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = F.shape
        if c != self.feat_dim:
            raise ValueError(f"F channels must be {self.feat_dim}; got {c}")
        self._check_shape("G", G, (b, self.latent_R, h, w))
        self._check_shape("m", m, (b, 1, h, w))
        self._check_shape("I_base", I_base, (b, 3, h, w))
        self._check_shape("depth", depth, (b, 1, h, w))
        self._check_shape("MV", MV, (b, 2, h, w))

        x = torch.cat([F, G, m, I_base, depth, MV], dim=1)
        x = self.proj_in(x)
        x = self.depthwise(x)
        x = self.act(x)
        x = self.proj_out(x)
        return F + x

    @staticmethod
    def _check_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
        if tensor.shape != expected:
            raise ValueError(f"{name} shape must be {expected}; got {tuple(tensor.shape)}")


__all__ = ["ConcatFusion"]
