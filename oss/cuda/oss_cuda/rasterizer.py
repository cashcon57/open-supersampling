"""
OSS custom rasterizer -- autograd Function wrapper around the C++ extension.

Phase 2c: forward calls the native CUDA rasterizer. The Python reference is
kept as an OSS_CUDA_RASTER_DEBUG=1 fallback through Phase 2d.
Backward not yet implemented -- raises NotImplementedError.
"""

from __future__ import annotations

from importlib import import_module

import torch
from torch.autograd import Function

# Import gate. The compiled extension exposes _C.rasterize_forward.
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
    """Called from C++ binding only when OSS_CUDA_RASTER_DEBUG=1."""
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
        # TODO(Phase 3): save sorted ids, tile offsets, conics, and weight_sum
        # once the binding returns the backward scratch tuple.
        ctx.save_for_backward(xy, scale, rot, feat)
        ctx.h, ctx.w = int(h), int(w)
        ctx.tile_size, ctx.topk_norm = int(tile_size), bool(topk_norm)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError(
            "Rasterizer backward is not implemented in Phase 2c. "
            "Use OSS_USE_CUDA_KERNELS=0 (default) for training."
        )


def rasterize_gaussians(xy, scale, rot, feat, h, w, tile_size=16, topk_norm=True):
    """Forward-only CUDA rasterizer wrapper."""
    return _RasterizeGaussians.apply(xy, scale, rot, feat, h, w, tile_size, topk_norm)
