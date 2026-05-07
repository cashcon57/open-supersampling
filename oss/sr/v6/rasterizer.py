"""v6 active-mask-aware Gaussian feature rasterizer."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from oss.gaussian.renderer import GaussianBatch, Rasterizer, TILE_SIZE
from oss.sr.v6.model import CanvasState


_MIN_PIXEL_SCALE = 1.0e-3


class V6Rasterizer(nn.Module):
    """Renders active subset of v6 CanvasState to HR feature image.

    Args:
        token_dim: feature channels per Gaussian. Matches ``V6Config.token_dim``.
        tile_size: tile edge length in pixels. Must match the underlying renderer.

    ``active_mask`` may be ``(N,)`` or ``(B, N)``. The v6 canvas is per-rank
    shared state, so a batched mask is treated as identical visibility for each
    batch element and the single rendered image is expanded to ``B``.
    """

    def __init__(
        self,
        token_dim: int,
        tile_size: int = TILE_SIZE,
        overlap: int = 0,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive; got {token_dim}")
        if tile_size != TILE_SIZE:
            raise ValueError(
                f"tile_size must match the underlying renderer ({TILE_SIZE}); got {tile_size}"
            )
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0; got {overlap}")
        self.token_dim = int(token_dim)
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.renderer = Rasterizer(tile_size=tile_size)

    def forward(
        self,
        canvas: CanvasState,
        active_mask: torch.Tensor,
        output_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Return ``(B, token_dim, H, W)`` feature image."""
        h, w = int(output_hw[0]), int(output_hw[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"output_hw must be positive; got {output_hw}")

        batch_size, mask_1d = self._normalize_active_mask(canvas, active_mask)
        device = canvas.colors.device
        out_dtype = canvas.colors.dtype

        n_live = self._live_count(canvas)
        if n_live == 0 or not bool(mask_1d[:n_live].any().item()):
            return torch.zeros(
                (batch_size, self.token_dim, h, w),
                device=device,
                dtype=out_dtype,
            )

        live_active = mask_1d[:n_live].to(device=device, dtype=torch.bool)
        colors = self._token_features(canvas.colors[:n_live])
        feat_dtype = colors.dtype

        # The reference rasterizer performs exp/quadratic math; keeping that in
        # fp32 avoids bf16 underflow/rounding while preserving autograd edges.
        positions = canvas.positions[:n_live][live_active].to(dtype=torch.float32)
        scales = canvas.scales[:n_live][live_active].to(dtype=torch.float32)
        rotations = canvas.rotations[:n_live][live_active].to(dtype=torch.float32)
        features = colors[live_active].to(dtype=torch.float32)

        positions, scales, rotations, features = self._sanitize_active_gaussians(
            positions,
            scales,
            rotations,
            features,
            output_hw=(h, w),
        )
        if positions.shape[0] == 0:
            return torch.zeros(
                (batch_size, self.token_dim, h, w),
                device=device,
                dtype=out_dtype,
            )

        gaussians = GaussianBatch(
            xy=positions,
            scale=scales,
            rot=rotations,
            feat=features,
        )

        if self.overlap > 0:
            rendered = self._render_overlapped(gaussians, output_hw=(h, w))
        else:
            rendered = self.renderer(gaussians, output_hw=(h, w))
        rendered = rendered.to(dtype=feat_dtype)
        return rendered.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def _render_overlapped(
        self,
        gaussians: GaussianBatch,
        output_hw: tuple[int, int],
    ) -> torch.Tensor:
        h, w = int(output_hw[0]), int(output_hw[1])
        device = gaussians.xy.device
        dtype = gaussians.feat.dtype
        accum = torch.zeros(
            (gaussians.feat_dim, h, w),
            device=device,
            dtype=torch.float32,
        )
        weight_accum = torch.zeros((1, h, w), device=device, dtype=torch.float32)
        overlap = int(self.overlap)

        # Rendering every owner tile independently is prohibitively expensive
        # once the CUDA backend chunks token_dim>12 into multiple kernel calls.
        # Instead, render a small set of full-frame crops whose 16px tile
        # origins are phase-shifted by the overlap width, then blend each crop
        # down near its own tile boundaries. This keeps the C++ kernel intact
        # while breaking the fixed renderer-grid phase in O(4) renders.
        origins = ((0, 0), (-overlap, 0), (0, -overlap), (-overlap, -overlap))
        for origin_y, origin_x in origins:
            crop_h = h - min(origin_y, 0)
            crop_w = w - min(origin_x, 0)
            local_xy = gaussians.xy.clone()
            local_xy[:, 0] = local_xy[:, 0] - float(origin_x)
            local_xy[:, 1] = local_xy[:, 1] - float(origin_y)
            local = GaussianBatch(
                xy=local_xy,
                scale=gaussians.scale,
                rot=gaussians.rot,
                feat=gaussians.feat,
            )
            rendered = self.renderer(local, output_hw=(crop_h, crop_w)).to(
                dtype=torch.float32
            )
            y_start = -min(origin_y, 0)
            x_start = -min(origin_x, 0)
            rendered = rendered[:, y_start:y_start + h, x_start:x_start + w]
            weight = self._phase_feather_mask(
                image_hw=(h, w),
                origin=(origin_y, origin_x),
                device=device,
            )
            accum += rendered * weight
            weight_accum += weight

        return (accum / weight_accum.clamp_min(1.0e-6)).to(dtype=dtype)

    def _phase_feather_mask(
        self,
        *,
        image_hw: tuple[int, int],
        origin: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        h, w = int(image_hw[0]), int(image_hw[1])
        origin_y, origin_x = int(origin[0]), int(origin[1])
        overlap = float(self.overlap)
        if overlap <= 0:
            return torch.ones((1, h, w), device=device, dtype=torch.float32)

        tile = float(self.tile_size)
        ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5 - float(origin_y)
        xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5 - float(origin_x)
        phase_y = torch.remainder(ys, tile)
        phase_x = torch.remainder(xs, tile)
        dist_y = torch.minimum(phase_y, tile - phase_y)
        dist_x = torch.minimum(phase_x, tile - phase_x)
        wy = self._cosine_ramp((dist_y / overlap).clamp(0.0, 1.0))
        wx = self._cosine_ramp((dist_x / overlap).clamp(0.0, 1.0))
        return (wy.view(1, h, 1) * wx.view(1, 1, w)).to(dtype=torch.float32)

    @staticmethod
    def _cosine_ramp(t: torch.Tensor) -> torch.Tensor:
        return 0.5 - 0.5 * torch.cos(t * math.pi)

    def _sanitize_active_gaussians(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        features: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = int(output_hw[0]), int(output_hw[1])
        if positions.shape[0] == 0:
            return positions, scales, rotations, features

        finite = (
            torch.isfinite(positions).all(dim=-1)
            & torch.isfinite(scales).all(dim=-1)
            & torch.isfinite(rotations)
            & torch.isfinite(features).all(dim=-1)
        )

        scales_safe = scales.abs().clamp(min=_MIN_PIXEL_SCALE, max=float(max(h, w)))
        radius = 3.0 * scales_safe.amax(dim=-1)
        overlaps = (
            (positions[:, 0] + radius >= 0.0)
            & (positions[:, 0] - radius < float(w))
            & (positions[:, 1] + radius >= 0.0)
            & (positions[:, 1] - radius < float(h))
        )
        valid = finite & overlaps
        if not bool(valid.any().item()):
            return (
                positions[:0],
                scales_safe[:0],
                rotations[:0],
                features[:0],
            )

        positions = positions[valid]
        scales_safe = scales_safe[valid]
        rotations = rotations[valid]
        features = features[valid]

        max_xy = positions.new_tensor([
            max(float(w) - 1.0e-4, 0.0),
            max(float(h) - 1.0e-4, 0.0),
        ])
        positions = torch.minimum(positions.clamp_min(0.0), max_xy)
        rotations = torch.atan2(torch.sin(rotations), torch.cos(rotations))
        return positions, scales_safe, rotations, features

    def _normalize_active_mask(
        self,
        canvas: CanvasState,
        active_mask: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        n_total = canvas.positions.shape[0]
        if active_mask.device != canvas.positions.device:
            active_mask = active_mask.to(device=canvas.positions.device)

        if active_mask.ndim == 1:
            if active_mask.shape[0] != n_total:
                raise ValueError(
                    f"active_mask must have length {n_total}; got {active_mask.shape[0]}"
                )
            return 1, active_mask.to(dtype=torch.bool)

        if active_mask.ndim == 2:
            batch_size, n_mask = active_mask.shape
            if n_mask != n_total:
                raise ValueError(
                    f"active_mask must have trailing length {n_total}; got {n_mask}"
                )
            mask_bool = active_mask.to(dtype=torch.bool)
            if batch_size == 0:
                raise ValueError("active_mask batch dimension must be positive")
            if batch_size > 1 and not torch.equal(mask_bool, mask_bool[:1].expand_as(mask_bool)):
                raise ValueError(
                    "batched active_mask rows must be identical because v6 renders "
                    "the shared per-rank canvas once and expands it across B"
                )
            return int(batch_size), mask_bool[0]

        raise ValueError(
            f"active_mask must be (N,) or (B, N); got {tuple(active_mask.shape)}"
        )

    def _live_count(self, canvas: CanvasState) -> int:
        n_total = int(canvas.positions.shape[0])
        return max(0, min(int(canvas.count), n_total))

    def _token_features(self, colors: torch.Tensor) -> torch.Tensor:
        if colors.ndim != 2:
            raise ValueError(f"canvas.colors must be (N, F); got {tuple(colors.shape)}")
        if colors.shape[-1] == self.token_dim:
            return colors
        if colors.shape[-1] > self.token_dim:
            return colors[..., : self.token_dim]
        pad = self.token_dim - colors.shape[-1]
        return torch.nn.functional.pad(colors, (0, pad))


__all__ = ["V6Rasterizer"]
