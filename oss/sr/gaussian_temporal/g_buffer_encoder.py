"""Per-tile G-buffer encoder for the v5 Gaussian temporal track."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GBufferEncoder(nn.Module):
    def __init__(self, in_channels: int = 12, feat_dim: int = 64, tile_size: int = 16) -> None:
        super().__init__()
        if tile_size & (tile_size - 1) != 0:
            raise ValueError(f"tile_size must be a power of 2; got {tile_size}")
        self.tile_size = tile_size
        # log2(tile_size) stride-2 conv blocks.
        n_down = int(tile_size).bit_length() - 1
        widths = [16, 24, 32, max(48, feat_dim)][: n_down]
        widths[-1] = feat_dim  # final width matches feat_dim

        layers: list[nn.Module] = []
        prev = in_channels
        for w in widths:
            layers.append(nn.Conv2d(prev, w, 3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            prev = w
        # One mixing conv at output resolution.
        layers.append(nn.Conv2d(prev, feat_dim, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["GBufferEncoder"]
