"""V0.5 pixel-residual head.

A tiny CNN that takes the rendered Gaussian-splat output + a bicubic-upsampled
LR reference, and predicts a per-pixel RGB residual. Final HR output =
``(splat_render + residual).clamp(0, 1)``.

Why this exists: the V0 pure-splat architecture converges to a degenerate
local minimum (constant gray-blob output) that bicubic dominates. See
``docs/superpowers/experiments/2026-05-02-output-head-dead-init.md`` "Updated
decision". Both GSASR and GS-STVSR add a pixel-residual head for exactly this
reason — splats carry structure, the residual CNN paints high-frequency
texture.

Design:
- Input: 6 channels = 3 (rendered HR) + 3 (bicubic-upsampled LR HR).
- Hidden channels: configurable (default 32).
- 3 conv layers, kernel 3x3, padding 1, ReLU between them.
- Last layer is **zero-init** so the residual is identically zero at init —
  the V0.5 model bit-exactly matches the V0 model on step 0 and only learns
  to deviate via gradient.

Param count at default config: ~12 K (vs 178 K for the lite tier param net).
Cheap to add, cheap to ablate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelResidualHead(nn.Module):
    """Small CNN producing a per-pixel RGB residual for V0.5.

    Args:
        in_channels: number of input channels (default 6: rendered + lr_up).
        hidden_channels: channel count of the two hidden conv layers.
        out_channels: residual channel count (3 for RGB).
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 32,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >=1; got {in_channels}")
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be >=1; got {hidden_channels}")
        if out_channels < 1:
            raise ValueError(f"out_channels must be >=1; got {out_channels}")

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

        # Zero-init the last conv so the initial residual is identically zero.
        # This means a freshly-enabled V0.5 model bit-exactly matches V0 at
        # step 0; the residual head only learns to deviate via gradient.
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

        # Standard kaiming init for the hidden layers (ReLU activations).
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, rendered: torch.Tensor, lr_up: torch.Tensor) -> torch.Tensor:
        """Predict the residual.

        Args:
            rendered: (B, 3, H_hr, W_hr) splat-rendered HR output, in [0, 1].
            lr_up:    (B, 3, H_hr, W_hr) bicubic-upsampled LR reference, [0, 1].

        Returns:
            (B, 3, H_hr, W_hr) residual; can be positive or negative.
            Caller is responsible for ``(rendered + residual).clamp(0, 1)``.
        """
        if rendered.shape != lr_up.shape:
            raise ValueError(
                f"rendered shape {tuple(rendered.shape)} != "
                f"lr_up shape {tuple(lr_up.shape)}"
            )
        if rendered.dim() != 4:
            raise ValueError(
                f"expected (B, C, H, W); got rendered shape {tuple(rendered.shape)}"
            )

        x = torch.cat([rendered, lr_up], dim=1)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return self.conv3(x)


__all__ = ["PixelResidualHead"]
