"""Per-tile error detection for the persistent Gaussian canvas — Sprint 5 / T5.3.

Two pure functions:

- ``per_tile_mse(rendered, lr_upsampled, tile_size)`` — averaged squared
  error per 16×16 tile. Drives prune+spawn decisions.
- ``gaussians_error_from_tiles(xy, tile_err, tile_size, hw)`` — each
  Gaussian inherits the error of the tile its centre falls into. Out-of-
  frame Gaussians get ``+inf`` so they are always pruned first.

Per-tile (rather than per-Gaussian) attribution is a deliberate
simplification: the prune+spawn loop only needs *which tiles look wrong*
to drive replacement. Per-Gaussian attribution would require breaking
into the rasterizer's tile accumulator — a Sprint 5 v2 perf-tune
problem, not a v1 correctness problem.

Design doc: ``docs/superpowers/gaussian-canvas-design.md`` §3.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def per_tile_mse(
    rendered: torch.Tensor,
    lr_upsampled: torch.Tensor,
    tile_size: int = 16,
) -> torch.Tensor:
    """Mean squared error per ``tile_size × tile_size`` tile.

    Args:
        rendered:     ``(F, H, W)`` canvas render at full output resolution.
        lr_upsampled: ``(F, H, W)`` LR input upsampled (e.g. bilinear) to
                      match ``rendered``'s resolution.
        tile_size:    Edge length in pixels. Must divide H and W.

    Returns:
        ``(h, w)`` float tensor where ``h = H // tile_size`` and
        ``w = W // tile_size``. Each entry is the mean squared error
        across that tile, averaged over feature channels.
    """
    if rendered.shape != lr_upsampled.shape:
        raise ValueError(
            f"rendered {tuple(rendered.shape)} vs lr_upsampled "
            f"{tuple(lr_upsampled.shape)} shape mismatch"
        )
    if rendered.ndim != 3:
        raise ValueError(f"rendered must be (F, H, W); got {tuple(rendered.shape)}")
    _, H, W = rendered.shape
    if H % tile_size or W % tile_size:
        raise ValueError(
            f"H={H} W={W} must both be multiples of tile_size={tile_size}"
        )

    sq = (rendered - lr_upsampled).pow(2).mean(dim=0, keepdim=True)  # (1, H, W)
    sq = sq.unsqueeze(0)  # (1, 1, H, W) for avg_pool2d
    pooled = F.avg_pool2d(sq, kernel_size=tile_size, stride=tile_size)
    return pooled.squeeze(0).squeeze(0)  # (h, w)


def gaussians_error_from_tiles(
    xy: torch.Tensor,
    tile_err: torch.Tensor,
    tile_size: int,
    hw: Tuple[int, int],
) -> torch.Tensor:
    """Look up per-Gaussian error from a per-tile error map.

    Args:
        xy:        ``(N, 2)`` Gaussian centres in pixel space (post-warp).
        tile_err:  ``(h, w)`` per-tile error map from ``per_tile_mse``.
        tile_size: Edge length in pixels (must match what produced
                   ``tile_err``).
        hw:        ``(H, W)`` frame dimensions (must equal
                   ``tile_err.shape * tile_size``).

    Returns:
        ``(N,)`` float tensor of per-Gaussian errors. Gaussians outside
        the frame get ``+inf`` so the prune step always selects them.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be (N, 2); got {tuple(xy.shape)}")
    if tile_err.ndim != 2:
        raise ValueError(f"tile_err must be (h, w); got {tuple(tile_err.shape)}")
    H, W = hw
    h, w = tile_err.shape
    if h * tile_size != H or w * tile_size != W:
        raise ValueError(
            f"tile_err {tile_err.shape} * tile_size={tile_size} != hw={hw}"
        )

    n = xy.shape[0]
    if n == 0:
        return torch.zeros((0,), dtype=tile_err.dtype, device=tile_err.device)

    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < float(W))
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < float(H))
    )
    # Compute tile coords for inside Gaussians; clamp for safety.
    tx = (xy[:, 0] / tile_size).long().clamp_(0, w - 1)
    ty = (xy[:, 1] / tile_size).long().clamp_(0, h - 1)
    err = tile_err[ty, tx]
    inf = torch.full_like(err, float("inf"))
    return torch.where(inside, err, inf)


__all__ = ["per_tile_mse", "gaussians_error_from_tiles"]
