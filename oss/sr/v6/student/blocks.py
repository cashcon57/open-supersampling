"""FasterNet block for the v6.2 student backbone.

Per Chen et al. "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural
Networks", CVPR 2023. Validated by AMD's FSR 4 (FasterNet48_NoGroup variant)
as a production-tested block for ML upscaling.

Independent reimplementation; pattern is well-published.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from oss.sr.v6.activations import SqrSwish


class FasterNetBlock(nn.Module):
    """Partial 3x3 conv + 1x1 MLP stack with residual output.

    Args:
        channels: Channel count. Input and output channel counts match.
        slice_ratio: Fraction of channels routed through the 3x3 conv.
        expansion: Inner channel multiplier for the 1x1 stack.
    """

    def __init__(
        self,
        channels: int,
        slice_ratio: float = 1.0 / 3.0,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.slice_channels = max(1, int(channels * slice_ratio))
        self.passthrough_channels = channels - self.slice_channels
        hidden = channels * expansion

        self.conv_pw = nn.Conv2d(self.slice_channels, self.slice_channels, kernel_size=3, padding=1)
        self.conv_expand = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = SqrSwish()
        self.conv_contract = nn.Conv2d(hidden, channels, kernel_size=1)

        nn.init.zeros_(self.conv_contract.weight)
        nn.init.zeros_(self.conv_contract.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``x``."""
        slice_part = x[:, : self.slice_channels]
        passthrough = x[:, self.slice_channels :]
        slice_part = self.conv_pw(slice_part)
        x_partial = torch.cat([slice_part, passthrough], dim=1)

        h = self.conv_expand(x_partial)
        h = self.act(h)
        h = self.conv_contract(h)
        return x + h


__all__ = ["FasterNetBlock"]
