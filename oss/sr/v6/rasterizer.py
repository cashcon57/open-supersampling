"""v6 active-mask-aware Gaussian feature rasterizer."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from oss.gaussian.renderer import GaussianBatch, Rasterizer, TILE_SIZE
from oss.sr.v6.model import CanvasState


class V6Rasterizer(nn.Module):
    """Renders active subset of v6 CanvasState to HR feature image.

    Args:
        token_dim: feature channels per Gaussian. Matches ``V6Config.token_dim``.
        tile_size: tile edge length in pixels. Must match the underlying renderer.

    ``active_mask`` may be ``(N,)`` or ``(B, N)``. The v6 canvas is per-rank
    shared state, so a batched mask is treated as identical visibility for each
    batch element and the single rendered image is expanded to ``B``.
    """

    def __init__(self, token_dim: int, tile_size: int = TILE_SIZE) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive; got {token_dim}")
        if tile_size != TILE_SIZE:
            raise ValueError(
                f"tile_size must match the underlying renderer ({TILE_SIZE}); got {tile_size}"
            )
        self.token_dim = int(token_dim)
        self.tile_size = int(tile_size)
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
        gaussians = GaussianBatch(
            xy=canvas.positions[:n_live][live_active].to(dtype=torch.float32),
            scale=canvas.scales[:n_live][live_active].to(dtype=torch.float32),
            rot=canvas.rotations[:n_live][live_active].to(dtype=torch.float32),
            feat=colors[live_active].to(dtype=torch.float32),
        )

        rendered = self.renderer(gaussians, output_hw=(h, w))
        rendered = rendered.to(dtype=feat_dtype)
        return rendered.unsqueeze(0).expand(batch_size, -1, -1, -1)

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
