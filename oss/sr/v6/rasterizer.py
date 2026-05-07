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

        radius = 3.0 * gaussians.scale.to(dtype=torch.float32).amax(dim=-1)
        overlap = int(self.overlap)
        tile = int(self.tile_size)
        for y0 in range(0, h, tile):
            y1 = min(y0 + tile, h)
            crop_y0 = max(0, y0 - overlap)
            crop_y1 = min(h, y1 + overlap)
            for x0 in range(0, w, tile):
                x1 = min(x0 + tile, w)
                crop_x0 = max(0, x0 - overlap)
                crop_x1 = min(w, x1 + overlap)

                keep = (
                    (gaussians.xy[:, 0].to(dtype=torch.float32) + radius >= float(crop_x0))
                    & (gaussians.xy[:, 0].to(dtype=torch.float32) - radius < float(crop_x1))
                    & (gaussians.xy[:, 1].to(dtype=torch.float32) + radius >= float(crop_y0))
                    & (gaussians.xy[:, 1].to(dtype=torch.float32) - radius < float(crop_y1))
                )
                if not bool(keep.any().item()):
                    continue

                local_xy = gaussians.xy[keep].clone()
                local_xy[:, 0] = local_xy[:, 0] - float(crop_x0)
                local_xy[:, 1] = local_xy[:, 1] - float(crop_y0)
                local = GaussianBatch(
                    xy=local_xy,
                    scale=gaussians.scale[keep],
                    rot=gaussians.rot[keep],
                    feat=gaussians.feat[keep],
                )
                crop_h = crop_y1 - crop_y0
                crop_w = crop_x1 - crop_x0
                rendered = self.renderer(local, output_hw=(crop_h, crop_w)).to(
                    dtype=torch.float32
                )
                weight = self._feather_mask(
                    crop_hw=(crop_h, crop_w),
                    crop_origin=(crop_y0, crop_x0),
                    owner_box=(y0, y1, x0, x1),
                    image_hw=(h, w),
                    device=device,
                )
                accum[:, crop_y0:crop_y1, crop_x0:crop_x1] += rendered * weight
                weight_accum[:, crop_y0:crop_y1, crop_x0:crop_x1] += weight

        return (accum / weight_accum.clamp_min(1.0e-6)).to(dtype=dtype)

    def _feather_mask(
        self,
        *,
        crop_hw: tuple[int, int],
        crop_origin: tuple[int, int],
        owner_box: tuple[int, int, int, int],
        image_hw: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        crop_h, crop_w = int(crop_hw[0]), int(crop_hw[1])
        crop_y0, crop_x0 = int(crop_origin[0]), int(crop_origin[1])
        y0, y1, x0, x1 = (int(v) for v in owner_box)
        h, w = int(image_hw[0]), int(image_hw[1])
        overlap = float(self.overlap)
        if overlap <= 0:
            return torch.ones((1, crop_h, crop_w), device=device, dtype=torch.float32)

        ys = torch.arange(crop_y0, crop_y0 + crop_h, device=device, dtype=torch.float32) + 0.5
        xs = torch.arange(crop_x0, crop_x0 + crop_w, device=device, dtype=torch.float32) + 0.5
        wy = torch.ones_like(ys)
        wx = torch.ones_like(xs)

        if y0 > 0:
            left = ys < float(y0)
            t = ((ys - float(y0 - self.overlap)) / overlap).clamp(0.0, 1.0)
            wy = torch.where(left, self._cosine_ramp(t), wy)
        if y1 < h:
            right = ys >= float(y1)
            t = ((float(y1 + self.overlap) - ys) / overlap).clamp(0.0, 1.0)
            wy = torch.where(right, self._cosine_ramp(t), wy)
        if x0 > 0:
            left = xs < float(x0)
            t = ((xs - float(x0 - self.overlap)) / overlap).clamp(0.0, 1.0)
            wx = torch.where(left, self._cosine_ramp(t), wx)
        if x1 < w:
            right = xs >= float(x1)
            t = ((float(x1 + self.overlap) - xs) / overlap).clamp(0.0, 1.0)
            wx = torch.where(right, self._cosine_ramp(t), wx)

        return (wy.view(1, crop_h, 1) * wx.view(1, 1, crop_w)).to(dtype=torch.float32)

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
