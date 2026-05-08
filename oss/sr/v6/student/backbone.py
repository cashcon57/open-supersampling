"""v6.2 student backbone scaffold.

Stack:
  Stem: Conv 3x3 (in_channels -> channels)
  Body: FasterNetBlock stack
  Tail: Conv 1x1 (channels -> out_features)

Designed to replace HAT-Tiny in the v6.2 inference path after distillation.
For now, pico-002 continues to train with HAT-Tiny in path.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from oss.sr.v6.student.blocks import FasterNetBlock


class StudentBackbone(nn.Module):
    """Small FasterNet-based student feature extractor."""

    def __init__(
        self,
        in_channels: int = 9,
        channels: int = 48,
        depth: int = 4,
        out_features: int = 180,
    ) -> None:
        """Initialize the student backbone.

        Args:
            in_channels: Input feature dimension.
            channels: Width of the body.
            depth: Number of FasterNet blocks.
            out_features: Output feature dimension, matching HAT-Tiny output.
        """
        super().__init__()
        self.stem = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(*(FasterNetBlock(channels) for _ in range(depth)))
        self.tail = nn.Conv2d(channels, out_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract student features from ``x``."""
        x = self.stem(x)
        x = self.body(x)
        x = self.tail(x)
        return x

    def num_params(self) -> int:
        """Return the number of trainable and frozen parameters."""
        return sum(p.numel() for p in self.parameters())


__all__ = ["StudentBackbone"]
