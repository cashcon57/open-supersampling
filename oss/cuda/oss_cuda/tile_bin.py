"""Tile binning helpers backed by the OSS CUDA extension."""

from __future__ import annotations

from importlib import import_module

import torch

try:
    from . import _C

    _COMPILED = True
except ImportError:
    try:
        _C = import_module("oss_cuda._C")
        _COMPILED = True
    except ImportError:
        _C = None
        _COMPILED = False


def tile_bin_counting_sort(
    tile_id: torch.Tensor, gid: torch.Tensor, num_tiles: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group ``gid`` values by ``tile_id`` using CUDA counting sort.

    Returns ``(sorted_gid, tile_offsets)``. Values for tile ``t`` occupy
    ``sorted_gid[tile_offsets[t]:tile_offsets[t + 1]]``.
    """
    if not _COMPILED:
        raise RuntimeError(
            "oss_cuda extension not compiled. Run: pip install -e ./oss/cuda"
        )
    if not tile_id.is_cuda or not gid.is_cuda:
        raise RuntimeError("tile_id and gid must be CUDA tensors")
    return _C.tile_bin_counting_sort(
        tile_id.contiguous().to(torch.int32),
        gid.contiguous().to(torch.int32),
        int(num_tiles),
    )
