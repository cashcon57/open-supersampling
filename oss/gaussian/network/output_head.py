"""OutputHead — Sprint 4 / T4.2.

Converts the GaussianParamNetwork's raw tile-wise output tensor into a
``GaussianBatch`` ready for the Sprint-1 Rasterizer.

Layout of the raw tensor (per the param_net spec):
    (B, K * per_gauss_ch, H_tile, W_tile)

with per-Gaussian channels:
    [Δμx, Δμy, log_scale, rot_offset, bank_logits[bank_size], color[3]]

Decoded per-Gaussian quantities:
    μx = (tile_x + 0.5) * tile_size + tile_size * tanh(Δμx)
    μy = (tile_y + 0.5) * tile_size + tile_size * tanh(Δμy)
        → centers stay inside or just outside the tile (±tile_size).
    scale_factor = exp(clamp(log_scale, ±ln(8)))   ∈ [1/8, 8]
    rotation = bank_theta + rot_offset_clipped     (small rotational refinement)
    bank_weights = softmax(bank_logits)            (used by CovariancePriorBank)
    sx, sy, theta = CovariancePriorBank(bank_weights) ∗ scale_factor (sx, sy)
                    + rot_offset (theta)
    color = sigmoid(color)                         ∈ [0, 1] (LDR mode)

NOTE: HDR mode is selectable via ``color_activation`` ("sigmoid" or "softplus").
softplus is unbounded above for HDR training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from oss.gaussian.network.param_net import per_gaussian_channels
from oss.gaussian.network.prior_bank import CovariancePriorBank
from oss.gaussian.renderer import GaussianBatch


@dataclass(frozen=True)
class DecodedParams:
    """All decoded per-Gaussian tensors. Shapes are (B, N) or (B, N, ...)."""
    xy: torch.Tensor            # (B, N, 2)
    scale: torch.Tensor         # (B, N, 2)
    rot: torch.Tensor           # (B, N)
    feat: torch.Tensor          # (B, N, F)
    bank_weights: torch.Tensor  # (B, N, K)


class OutputHead(nn.Module):
    """Decode raw network output into renderer-ready Gaussian parameters.

    Args:
        bank: a ``CovariancePriorBank`` (must have the same bank_size that
            produced the raw logits).
        tile_size: 16 (must match renderer.TILE_SIZE).
        k_per_tile: number of Gaussians per tile (must match the param net).
        color_activation: "sigmoid" → LDR colors in [0, 1].
                          "softplus" → unbounded HDR-friendly.
        log_scale_clip: absolute clip on log_scale; final scale_factor is
            in [exp(-clip), exp(+clip)]. Default ln(8) ≈ 2.08.
    """

    def __init__(
        self,
        bank: CovariancePriorBank,
        tile_size: int = 16,
        k_per_tile: int = 5,
        color_activation: str = "sigmoid",
        log_scale_clip: float = math.log(8.0),
    ) -> None:
        super().__init__()
        if color_activation not in ("sigmoid", "softplus"):
            raise ValueError(
                f"color_activation must be 'sigmoid' or 'softplus'; "
                f"got {color_activation!r}"
            )
        self.bank = bank
        self.tile_size = tile_size
        self.k_per_tile = k_per_tile
        self.color_activation = color_activation
        self.log_scale_clip = float(log_scale_clip)

    def decode(self, raw: torch.Tensor) -> DecodedParams:
        """Decode the raw output tensor into ``DecodedParams``."""
        if raw.dim() != 4:
            raise ValueError(f"expected (B, C, H, W); got {tuple(raw.shape)}")
        B, C, Ht, Wt = raw.shape
        per_g = per_gaussian_channels(self.bank.bank_size)
        if C != self.k_per_tile * per_g:
            raise ValueError(
                f"raw channels ({C}) != k_per_tile ({self.k_per_tile}) * "
                f"per_gaussian_channels ({per_g})"
            )

        # Reshape to (B, K, per_g, Ht, Wt) → (B, Ht, Wt, K, per_g)
        x = raw.view(B, self.k_per_tile, per_g, Ht, Wt).permute(0, 3, 4, 1, 2)
        # Slice channels.
        d_xy = x[..., 0:2]                      # (B, Ht, Wt, K, 2)
        log_scale = x[..., 2]                   # (B, Ht, Wt, K)
        rot_off = x[..., 3]                     # (B, Ht, Wt, K)
        bank_logits = x[..., 4:4 + self.bank.bank_size]   # (B, Ht, Wt, K, K_bank)
        color = x[..., 4 + self.bank.bank_size:4 + self.bank.bank_size + 3]
        # color shape: (B, Ht, Wt, K, 3)

        # ---- Position decode -------------------------------------------------
        # Tile centers in pixel space.
        device = raw.device
        dtype = raw.dtype
        ts = float(self.tile_size)
        ys = (torch.arange(Ht, device=device, dtype=dtype) + 0.5) * ts
        xs = (torch.arange(Wt, device=device, dtype=dtype) + 0.5) * ts
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")  # (Ht, Wt)
        # Broadcast tile centers across (B, K).
        cx = cx[None, :, :, None].expand(B, Ht, Wt, self.k_per_tile)
        cy = cy[None, :, :, None].expand(B, Ht, Wt, self.k_per_tile)

        # ±tile_size offset envelope; tanh keeps the center within ±1 tile.
        dx = torch.tanh(d_xy[..., 0]) * ts
        dy = torch.tanh(d_xy[..., 1]) * ts
        mu_x = cx + dx
        mu_y = cy + dy

        # ---- Scale decode ----------------------------------------------------
        scale_factor = torch.exp(log_scale.clamp(-self.log_scale_clip,
                                                 self.log_scale_clip))

        # ---- Bank softmax ----------------------------------------------------
        bank_w = F.softmax(bank_logits, dim=-1)
        sx_bank, sy_bank, theta_bank, _cov = self.bank(bank_w)
        # Apply scale_factor multiplicatively to sx/sy; rot_offset adds in.
        sx = sx_bank * scale_factor
        sy = sy_bank * scale_factor
        # Clamp rotation offset to ±π/4 so it can't undo the bank's selection.
        rot = theta_bank + torch.tanh(rot_off) * (math.pi / 4.0)

        # ---- Color -----------------------------------------------------------
        if self.color_activation == "sigmoid":
            feat = torch.sigmoid(color)
        else:
            feat = F.softplus(color)

        # ---- Flatten (Ht, Wt, K) → N -----------------------------------------
        N = Ht * Wt * self.k_per_tile
        mu_x = mu_x.reshape(B, N)
        mu_y = mu_y.reshape(B, N)
        xy = torch.stack([mu_x, mu_y], dim=-1)            # (B, N, 2)
        scale = torch.stack([sx.reshape(B, N), sy.reshape(B, N)], dim=-1)  # (B, N, 2)
        rot = rot.reshape(B, N)
        feat = feat.reshape(B, N, 3)
        bank_weights = bank_w.reshape(B, N, self.bank.bank_size)
        return DecodedParams(xy=xy, scale=scale, rot=rot, feat=feat,
                             bank_weights=bank_weights)

    # ------------------------------------------------------------------------
    def to_gaussian_batch(self, raw: torch.Tensor, batch_index: int = 0
                          ) -> GaussianBatch:
        """Decode a single sample of the raw tensor into a ``GaussianBatch``.

        The Sprint 1 ``GaussianBatch`` is per-image (no batch dimension).
        Callers wanting B>1 should iterate or write a batched renderer wrapper.
        """
        if not (0 <= batch_index < raw.shape[0]):
            raise IndexError(f"batch_index {batch_index} out of range for B={raw.shape[0]}")
        d = self.decode(raw)
        return GaussianBatch(
            xy=d.xy[batch_index],
            scale=d.scale[batch_index],
            rot=d.rot[batch_index],
            feat=d.feat[batch_index],
        )


__all__ = ["OutputHead", "DecodedParams"]
