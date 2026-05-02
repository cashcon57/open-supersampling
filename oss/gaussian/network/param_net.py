"""GaussianParamNetwork — Sprint 4 / T4.2.

Lightweight CNN that predicts per-tile Gaussian parameters from an LR frame
plus G-buffers plus a canvas-state hint.

Pipeline (high level):
    LR(3) + depth(1) + motion(2) + normals(3) + canvas(3) = 12-channel input
        ↓ encoder (16 → 24 → 32 → 48), four levels at LR
        ↓ U-Net decoder (mirror)
        ↓ tile-aware output head: each 16×16 tile gets K Gaussians
    raw output: (B, K * (5 + bank_size + 3), H_tile, W_tile)
        where the per-Gaussian channels are
            (Δμx, Δμy, log_scale, rotation_offset_radians,
             bank_logits[bank_size], color[3])

This module emits the *raw* tensor only. ``output_head.OutputHead`` then
converts it into a renderer-ready ``GaussianBatch`` (using the
``CovariancePriorBank`` to decode the bank logits).

No training code lives here — this is the model class, the dataloader and
losses are introduced in T4.3 / T4.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from oss.model.blocks import ConvBlock, DownBlock, UpBlock


# Channel widths per the Sprint 4 spec — match Image-GS-style narrow CNN.
# Multiples of 8 keep FP16 packed-math friendly. Tier scaling lives in the
# ``param_net_for_tier`` factory so the same code serves Pico→Ultra.
_DEFAULT_CHANNELS: Tuple[int, int, int, int] = (16, 24, 32, 48)
DEFAULT_TILE_SIZE: int = 16  # must match renderer.TILE_SIZE
DEFAULT_K_PER_TILE: int = 5  # ablated to 3 / 5 / 8 in T4.10
DEFAULT_BANK_SIZE: int = 16  # default bank vocabulary

# Per-Gaussian raw channel count: Δμx, Δμy, log_scale, rot_offset,
# bank_logits[bank_size], color[3].
_NON_BANK_CHANNELS: int = 4 + 3  # 4 geometry + 3 color


def per_gaussian_channels(bank_size: int) -> int:
    """Total raw output channels per Gaussian for a given bank size."""
    return _NON_BANK_CHANNELS + bank_size


@dataclass(frozen=True)
class TierConfig:
    """One row of the tier-scaling table.

    See ``docs/superpowers/gaussian-network-architecture.md`` for the full
    table. Channels stay on multiples of 8.
    """
    name: str
    channels: Tuple[int, int, int, int]
    k_per_tile: int
    target_gaussians: int


TIER_CONFIGS: dict[str, TierConfig] = {
    # Steam Deck — 1K Gaussians, smallest CNN we still trust to converge.
    "pico":     TierConfig("pico",     (8, 16, 24, 32),  3,  1_000),
    "lite":     TierConfig("lite",     (16, 24, 32, 40), 5,  5_000),
    "standard": TierConfig("standard", (16, 24, 32, 48), 5,  8_000),
    "ultra":    TierConfig("ultra",    (24, 32, 48, 64), 8, 15_000),
}


class GaussianParamNetwork(nn.Module):
    """Tile-wise predictor of per-Gaussian parameters.

    Args:
        in_channels: total input channel count. Default 12 (LR 3 + depth 1 +
            motion 2 + normals 3 + canvas 3). Pass a different value if the
            caller wants to drop the canvas hint (sequence boundary) or add
            extra G-buffers.
        bank_size: width of the covariance prior bank softmax head.
        k_per_tile: number of Gaussians to emit per 16×16 tile.
        channels: (c0, c1, c2, c3) encoder widths. Decoder mirrors.
        tile_size: must match renderer.TILE_SIZE for downstream compatibility.

    Forward:
        x: (B, in_channels, H_lr, W_lr)
        Returns: (B, K * per_gaussian_channels(bank_size), H_tile, W_tile)
            where H_tile = H_lr // tile_size, W_tile = W_lr // tile_size.

    Spatial contract:
        H_lr and W_lr MUST be exact multiples of tile_size. The network
        downsamples 4×, then a final stride-(tile_size//4) projection takes
        the feature map to tile resolution. Concretely the U-Net runs at
        H_lr / 1, /2, /4, /8 — the final tile head pools H_lr/8 → H_lr/16
        when tile_size=16.
    """

    def __init__(
        self,
        in_channels: int = 12,
        bank_size: int = DEFAULT_BANK_SIZE,
        k_per_tile: int = DEFAULT_K_PER_TILE,
        channels: Tuple[int, int, int, int] = _DEFAULT_CHANNELS,
        tile_size: int = DEFAULT_TILE_SIZE,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >=1; got {in_channels}")
        if bank_size < 2:
            raise ValueError(f"bank_size must be >=2; got {bank_size}")
        if k_per_tile < 1:
            raise ValueError(f"k_per_tile must be >=1; got {k_per_tile}")
        if tile_size <= 0 or (tile_size & (tile_size - 1)) != 0:
            raise ValueError(f"tile_size must be a positive power of 2; got {tile_size}")
        if any(c % 8 for c in channels):
            raise ValueError(f"channels must all be multiples of 8; got {channels}")

        self.in_channels = in_channels
        self.bank_size = bank_size
        self.k_per_tile = k_per_tile
        self.tile_size = tile_size
        self.channels = tuple(channels)

        c = channels

        # ---- Encoder (4 levels, each with one refinement block). ------------
        self.stem = ConvBlock(in_channels, c[0])
        self.enc0 = ConvBlock(c[0], c[0])
        self.down1 = DownBlock(c[0], c[1])
        self.enc1 = ConvBlock(c[1], c[1])
        self.down2 = DownBlock(c[1], c[2])
        self.enc2 = ConvBlock(c[2], c[2])
        self.down3 = DownBlock(c[2], c[3])
        self.enc3 = ConvBlock(c[3], c[3])

        # ---- Decoder mirrors the encoder with skip connections. -------------
        self.up3 = UpBlock(c[3], c[2])
        self.dec3 = ConvBlock(c[2] * 2, c[2])
        self.up2 = UpBlock(c[2], c[1])
        self.dec2 = ConvBlock(c[1] * 2, c[1])
        self.up1 = UpBlock(c[1], c[0])
        self.dec1 = ConvBlock(c[0] * 2, c[0])

        # ---- Tile-aware output head. ----------------------------------------
        # The decoder ends at LR resolution. We pool to tile resolution with a
        # stride-tile_size avg-pool over a c[0] feature map, then a 1×1 conv
        # produces the per-tile raw parameter vector. Avg-pool is intentional:
        # we want each tile to see all LR pixels inside it, but learnable
        # spatial pooling adds parameters with no obvious win at this budget.
        self.out_channels = k_per_tile * per_gaussian_channels(bank_size)
        self.tile_proj = nn.Conv2d(c[0], c[0], kernel_size=tile_size, stride=tile_size)
        self.tile_norm = nn.GroupNorm(min(8, c[0] // 8), c[0])
        self.tile_act = nn.SiLU(inplace=True)
        self.head = nn.Conv2d(c[0], self.out_channels, kernel_size=1)

        self._init_head()

    def _init_head(self) -> None:
        """Initialise the head so K parallel Gaussians per tile START DIFFERENT.

        Pure zero-init (the prior implementation) was a dead-init symmetry
        failure: the K Gaussians within each tile shared identical
        (position, bank weights, color, scale, rotation), the loss gradient
        was symmetric across them, AdamW updated them in lockstep, and the
        symmetry never broke. Diagnostic output (`bank_entropy_norm=1.000`,
        `mean_dxy_norm=0.000`, `color_std<0.03`) was pinned across 500
        steps. See `docs/superpowers/experiments/2026-05-02-output-head-dead-init.md`.

        We now keep the WEIGHTS near zero (so the spatial features don't
        dominate at init — small Gaussian, std=1e-3) but apply a small
        Gaussian random BIAS (std=0.05) so each output channel — and
        therefore each of the K parallel decoder slots — starts at a
        different point. This is enough to break K-way symmetry without
        destabilising early training:
          - tanh(0.05) ≈ 0.05 → ±5% of a tile-size offset on positions.
          - softmax over a vector with std 0.05 is still nearly uniform but
            no longer perfectly so, giving the bank weights a gradient to
            follow.
          - sigmoid(0.05) ≈ 0.512, sigmoid(−0.05) ≈ 0.488 → tiny color
            differential, again enough to break symmetry.
        """
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.head.bias, mean=0.0, std=0.05)

    # ---- Forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W); got shape {tuple(x.shape)}")
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"in_channels mismatch: model={self.in_channels}, input={C}"
            )
        if H % self.tile_size or W % self.tile_size:
            raise ValueError(
                f"H={H} and W={W} must be exact multiples of tile_size={self.tile_size}"
            )
        # Encoder
        x0 = self.enc0(self.stem(x))                  # (B, c0, H,  W)
        x1 = self.enc1(self.down1(x0))                # (B, c1, H/2,W/2)
        x2 = self.enc2(self.down2(x1))                # (B, c2, H/4,W/4)
        x3 = self.enc3(self.down3(x2))                # (B, c3, H/8,W/8)

        # Decoder with skip concatenation
        d3 = self.dec3(torch.cat([self.up3(x3), x2], dim=1))  # (B, c2, H/4, W/4)
        d2 = self.dec2(torch.cat([self.up2(d3), x1], dim=1))  # (B, c1, H/2, W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), x0], dim=1))  # (B, c0, H,   W)

        # Tile pooling → per-tile feature → head
        t = self.tile_act(self.tile_norm(self.tile_proj(d1)))
        raw = self.head(t)
        # raw shape: (B, K * per_gauss_ch, H/tile_size, W/tile_size)
        return raw

    # ---- Diagnostics -------------------------------------------------------
    @property
    def raw_per_gaussian_channels(self) -> int:
        return per_gaussian_channels(self.bank_size)

    def output_shape(self, h_lr: int, w_lr: int) -> Tuple[int, int, int]:
        """Return (out_ch, H_tile, W_tile) for a given LR resolution."""
        return (self.out_channels, h_lr // self.tile_size, w_lr // self.tile_size)


def param_net_for_tier(tier: str, bank_size: int = DEFAULT_BANK_SIZE,
                       in_channels: int = 12, tile_size: int = DEFAULT_TILE_SIZE
                       ) -> GaussianParamNetwork:
    """Factory: build a GaussianParamNetwork sized for a hardware tier.

    Tiers map to the channel widths and K-per-tile defined in TIER_CONFIGS.
    The bank_size is independent of tier.
    """
    if tier not in TIER_CONFIGS:
        raise KeyError(f"unknown tier {tier!r}; available: {sorted(TIER_CONFIGS)}")
    cfg = TIER_CONFIGS[tier]
    return GaussianParamNetwork(
        in_channels=in_channels,
        bank_size=bank_size,
        k_per_tile=cfg.k_per_tile,
        channels=cfg.channels,
        tile_size=tile_size,
    )


__all__ = [
    "GaussianParamNetwork",
    "TierConfig",
    "TIER_CONFIGS",
    "param_net_for_tier",
    "per_gaussian_channels",
    "DEFAULT_TILE_SIZE",
    "DEFAULT_K_PER_TILE",
    "DEFAULT_BANK_SIZE",
]
