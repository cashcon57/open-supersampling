"""GRAPE-style per-tile Gaussian parameter decoder for v6.

The spawner turns HAT pixel features at LR resolution into one anisotropic
Gaussian proposal per LR tile. It mirrors the compact GRAPE pattern: a single
point-wise layer predicts per-pixel Gaussian parameters, then a tile pool
reduces those predictions to the canvas write-back granularity.

The output is batched because V6Model flattens / concatenates per-rank canvas
updates outside this module. This module owns no mutable state beyond its
single 1x1 convolution, so it is DDP-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GaussianSpawnState:
    """Batched CanvasState-compatible Gaussian proposals.

    Attributes:
        positions: ``(B, K, 2)`` HR pixel-space ``(x, y)`` centers.
        scales: ``(B, K, 2)`` positive anisotropic per-axis scales.
        rotations: ``(B, K)`` angle in radians, bounded to ``[-pi, pi]``.
        colors: ``(B, K, token_dim)`` embeddings consumed by canvas-to-token.
        confidence: ``(B, K)`` alive probability in ``[0, 1]``.
        count: number of proposals per batch element.
    """

    positions: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    colors: torch.Tensor
    confidence: torch.Tensor
    count: int

    @property
    def opacities(self) -> torch.Tensor:
        """Alias used by CanvasState-style consumers."""
        return self.confidence


class GaussianSpawner(nn.Module):
    """Decode HAT features into per-tile Gaussian proposals.

    Args:
        feat_dim: channel count of the HAT feature tensor.
        token_dim: output embedding dimension. Defaults to ``config.token_dim``
            when ``config`` is passed, otherwise 64.
        scale: LR-to-HR upscale factor. Defaults to ``config.scale`` when
            available, otherwise 2.
        tile_size_lr: tile edge length in LR pixels. Defaults to 8.
        config: optional V6Config-like object; read duck-typed to avoid a hard
            import cycle with ``model.py``.
    """

    def __init__(
        self,
        feat_dim: int,
        token_dim: int | None = None,
        scale: int | None = None,
        tile_size_lr: int | None = None,
        config: Any | None = None,
    ) -> None:
        super().__init__()
        if config is not None:
            if token_dim is None:
                token_dim = int(getattr(config, "token_dim"))
            if scale is None:
                scale = int(getattr(config, "scale"))
            if tile_size_lr is None:
                tile_size_lr = int(getattr(config, "tile_size_lr", 8))
        self.spawn_offset_random = bool(
            getattr(config, "spawn_offset_random", False)
        ) if config is not None else False
        if token_dim is None:
            token_dim = 64
        if scale is None:
            scale = 2
        if tile_size_lr is None:
            tile_size_lr = 8

        self.feat_dim = int(feat_dim)
        self.token_dim = int(token_dim)
        self.scale = int(scale)
        self.tile_size_lr = int(tile_size_lr)
        self.tile_size_hr = self.tile_size_lr * self.scale
        self.param_dim = 6 + self.token_dim

        if self.feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive; got {feat_dim}")
        if self.token_dim <= 0:
            raise ValueError(f"token_dim must be positive; got {token_dim}")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive; got {scale}")
        if self.tile_size_lr <= 0:
            raise ValueError(f"tile_size_lr must be positive; got {tile_size_lr}")

        self.conv = nn.Conv2d(self.feat_dim, self.param_dim, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize neutral proposals at tile centers.

        Scales start near half an HR tile, rotations / offsets / confidence
        logits start at zero, and colors start as a small neutral embedding.
        The conv weights are zero so first-frame proposals are geometrically
        stable before training.
        """
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        target_scale = torch.tensor(0.5 * float(self.tile_size_hr))
        scale_bias = torch.log(torch.expm1(target_scale)).item()
        with torch.no_grad():
            self.conv.bias[3:5].fill_(scale_bias)
            self.conv.bias[6:].fill_(0.01)

    def forward(
        self,
        features: torch.Tensor,
        spawn_offset_xy: torch.Tensor | None = None,
    ) -> GaussianSpawnState:
        """Return one Gaussian proposal per LR tile.

        Args:
            features: HAT features ``(B, feat_dim, H, W)`` at LR resolution.

        Returns:
            ``GaussianSpawnState`` with ``K = (H / tile_size_lr) *
            (W / tile_size_lr)`` proposals per batch element.
        """
        if features.dim() != 4:
            raise ValueError(
                f"features must be (B, feat_dim, H, W); got {tuple(features.shape)}"
            )
        if features.shape[1] != self.feat_dim:
            raise ValueError(
                f"expected feat_dim={self.feat_dim}, got {features.shape[1]}"
            )
        h, w = int(features.shape[-2]), int(features.shape[-1])
        if h % self.tile_size_lr != 0 or w % self.tile_size_lr != 0:
            raise ValueError(
                "feature height/width must be divisible by tile_size_lr; "
                f"got H={h}, W={w}, tile_size_lr={self.tile_size_lr}"
            )

        pixel_params = self.conv(features)
        pooled = F.avg_pool2d(
            pixel_params,
            kernel_size=self.tile_size_lr,
            stride=self.tile_size_lr,
        )
        pooled = pooled.flatten(2).transpose(1, 2).contiguous()

        # Parameter heads feed the rasterizer's quadratic/exp math. Decode in
        # fp32 even under bf16 autocast so large HAT-L activations cannot push
        # positions/scales into non-finite canvas state before rendering.
        with torch.autocast(device_type=features.device.type, enabled=False):
            pooled_f = pooled.float()
            conf_raw = pooled_f[..., 0]
            offset_raw = pooled_f[..., 1:3]
            scale_raw = pooled_f[..., 3:5]
            rotation_raw = pooled_f[..., 5]
            colors = pooled_f[..., 6:]

            centers = self._tile_centers(
                tile_h=h // self.tile_size_lr,
                tile_w=w // self.tile_size_lr,
                device=pooled_f.device,
                dtype=torch.float32,
            )
            if spawn_offset_xy is not None:
                spawn_offset_xy = spawn_offset_xy.to(
                    device=pooled_f.device,
                    dtype=torch.float32,
                )
                if spawn_offset_xy.shape == (2,):
                    centers_b = centers.unsqueeze(0) + spawn_offset_xy.view(1, 1, 2)
                elif spawn_offset_xy.shape == (features.shape[0], 2):
                    centers_b = centers.unsqueeze(0) + spawn_offset_xy.view(-1, 1, 2)
                else:
                    raise ValueError(
                        "spawn_offset_xy must be (2,) or (B, 2); got "
                        f"{tuple(spawn_offset_xy.shape)}"
                    )
            else:
                centers_b = centers.unsqueeze(0)
            offset_bound = pooled_f.new_tensor(0.5 * float(self.tile_size_hr))
            positions = centers_b + torch.tanh(offset_raw) * offset_bound
            max_xy = pooled_f.new_tensor([float(w * self.scale), float(h * self.scale)])
            positions = torch.minimum(positions.clamp_min(0.0), max_xy.view(1, 1, 2))

            max_scale = 4.0 * float(self.tile_size_hr)
            scales = F.softplus(scale_raw).clamp(min=1.0e-4, max=max_scale)
            rotations = torch.tanh(rotation_raw) * pooled_f.new_tensor(torch.pi)
            confidence = torch.sigmoid(conf_raw)

        return GaussianSpawnState(
            positions=positions,
            scales=scales,
            rotations=rotations,
            colors=colors,
            confidence=confidence,
            count=int(centers.shape[0]),
        )

    def _tile_centers(
        self,
        tile_h: int,
        tile_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.arange(tile_h, device=device, dtype=dtype)
        x = torch.arange(tile_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        half = torch.tensor(0.5, device=device, dtype=dtype)
        step = torch.tensor(float(self.tile_size_hr), device=device, dtype=dtype)
        centers_x = (xx + half) * step
        centers_y = (yy + half) * step
        return torch.stack((centers_x, centers_y), dim=-1).reshape(-1, 2)


__all__ = ["GaussianSpawnState", "GaussianSpawner"]
