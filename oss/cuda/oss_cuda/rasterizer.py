"""
OSS custom rasterizer -- autograd Function wrapper around the C++ extension.

Phase 1: forward delegates to oss.gaussian.renderer.rasterizer.Rasterizer._render_reference.
Backward not yet implemented -- raises NotImplementedError.
"""

from __future__ import annotations

from importlib import import_module

import torch
from torch.autograd import Function

# Phase 1 import gate. The compiled extension exposes _C.rasterize_forward.
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


def _phase1_ref_forward(xy, scale, rot, feat, h, w, tile_size, topk_norm):
    """Called from C++ binding in Phase 1 only."""
    from oss.gaussian.renderer.rasterizer import GaussianBatch, Rasterizer

    batch = GaussianBatch(xy=xy, scale=scale, rot=rot, feat=feat)
    rast = Rasterizer(tile_size=int(tile_size), topk_norm=bool(topk_norm))
    return rast._render_reference(batch, int(h), int(w))


class _RasterizeGaussians(Function):
    @staticmethod
    def forward(ctx, xy, scale, rot, feat, h, w, tile_size, topk_norm):
        if not _COMPILED:
            raise RuntimeError(
                "oss_cuda extension not compiled. "
                "Run: pip install -e ./oss/cuda"
            )
        out = _C.rasterize_forward(xy, scale, rot, feat, h, w, tile_size, topk_norm)
        ctx.save_for_backward(xy, scale, rot, feat)
        ctx.h, ctx.w = int(h), int(w)
        ctx.tile_size, ctx.topk_norm = int(tile_size), bool(topk_norm)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError(
            "Rasterizer backward is not implemented in Phase 1. "
            "Use OSS_USE_CUDA_KERNELS=0 (default) for training."
        )


def rasterize_gaussians(xy, scale, rot, feat, h, w, tile_size=16, topk_norm=True):
    """Phase-1 stub: forward only, calls Python reference."""
    return _RasterizeGaussians.apply(xy, scale, rot, feat, h, w, tile_size, topk_norm)
