"""Building blocks for ORD/ORU networks. Channel counts must be multiples of 8
for later cooperative-matrix tiling."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        assert out_ch % 8 == 0, f"out_ch={out_ch} must be multiple of 8"
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch // 8), num_channels=out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DownBlock(nn.Module):
    """2x downsample via stride-2 conv. Same activation as ConvBlock."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        assert out_ch % 8 == 0
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch // 8), out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UpBlock(nn.Module):
    """2x bilinear upsample + 3x3 conv (avoids checkerboard from transposed conv)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        assert out_ch % 8 == 0
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch // 8), out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.act(self.norm(self.conv(x)))


class KernelPredictionHead(nn.Module):
    """Predicts per-pixel reconstruction kernels (size K) over a noisy input.
    Output kernel weights are softmax-normalized so reconstructed pixels are
    convex combinations of neighbors - bounds the output, stabilizes training.
    Per Bako et al. 2017 / KPAL 2018, kernel prediction has provably better
    convergence than direct color regression.
    """
    def __init__(self, feature_ch: int, kernel_size: int = 5):
        super().__init__()
        self.k = kernel_size
        self.k2 = kernel_size * kernel_size
        self.predict = nn.Conv2d(feature_ch, self.k2, 3, padding=1)

    def forward(self, features: torch.Tensor, noisy_rgb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = noisy_rgb.shape
        weights = self.predict(features)
        weights = weights.softmax(dim=1)
        patches = F.unfold(noisy_rgb, kernel_size=self.k, padding=self.k // 2)
        patches = patches.reshape(B, C, self.k2, H, W)
        return (patches * weights.unsqueeze(1)).sum(dim=2)
