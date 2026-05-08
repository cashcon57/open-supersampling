"""Phase 2 sub-phase 2a: kernel-by-kernel test helpers, not in production path."""

from importlib import import_module

try:
    from . import _C
except ImportError:
    _C = import_module("oss_cuda._C")


def preprocess_only(xy, scale, rot, H, W, tile_size=16):
    return _C.preprocess_only(xy, scale, rot, H, W, tile_size)


def pair_construction_only(xy, scale, rot, H, W, tile_size=16):
    return _C.pair_construction_only(xy, scale, rot, H, W, tile_size)
