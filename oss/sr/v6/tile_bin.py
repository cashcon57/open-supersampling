"""v6 tile-binning wrappers."""

from __future__ import annotations

import torch


def tile_bin_counting_sort(
    tile_id: torch.Tensor,
    gid: torch.Tensor | None = None,
    num_tiles: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group gids by tile id with the native CUDA counting-sort kernel.

    Returns ``(sorted_gid, tile_offsets)``. If ``gid`` is omitted, it defaults
    to ``arange(N)`` on the same device as ``tile_id``.
    """
    if gid is None:
        gid = torch.arange(tile_id.numel(), device=tile_id.device, dtype=torch.int32)
    if num_tiles is None:
        if tile_id.numel() == 0:
            raise ValueError("num_tiles is required for empty tile_id")
        num_tiles = int(tile_id.max().item()) + 1

    from oss_cuda.tile_bin import tile_bin_counting_sort as _cuda_tile_bin

    return _cuda_tile_bin(tile_id, gid, int(num_tiles))
