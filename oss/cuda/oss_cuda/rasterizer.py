"""
OSS custom rasterizer -- autograd Function wrapper around the C++ extension.

Phase 3c native CUDA forward/backward is the live implementation.
"""

from __future__ import annotations

from importlib import import_module

import torch
from torch.autograd import Function

# Import gate. The compiled extension exposes _C.rasterize_forward/backward.
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


class _RasterizeGaussians(Function):
    @staticmethod
    def forward(ctx, xy, scale, rot, feat, h, w, tile_size, topk_norm):
        if not _COMPILED:
            raise RuntimeError(
                "oss_cuda extension not compiled. "
                "Run: pip install -e ./oss/cuda"
            )
        out, gaussian_idx_sorted, tile_offsets, conic = _C.rasterize_forward(
            xy, scale, rot, feat, h, w, tile_size, topk_norm
        )
        ctx.save_for_backward(
            xy, scale, rot, feat, conic, gaussian_idx_sorted, tile_offsets
        )
        ctx.h, ctx.w = int(h), int(w)
        ctx.tile_size, ctx.topk_norm = int(tile_size), bool(topk_norm)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        xy, scale, rot, feat, conic, gaussian_idx_sorted, tile_offsets = (
            ctx.saved_tensors
        )
        d_xy, d_conic, d_feat = _C.rasterize_backward(
            xy,
            scale,
            rot,
            feat,
            conic,
            gaussian_idx_sorted,
            tile_offsets,
            grad_out.contiguous(),
            ctx.h,
            ctx.w,
            ctx.tile_size,
        )
        d_scale, d_rot = _C.conic_to_scale_rot_grad(scale, rot, d_conic)
        return d_xy, d_scale, d_rot, d_feat, None, None, None, None


def rasterize_gaussians(xy, scale, rot, feat, h, w, tile_size=16, topk_norm=True):
    """CUDA rasterizer wrapper with Phase 3c native backward."""
    return _RasterizeGaussians.apply(xy, scale, rot, feat, h, w, tile_size, topk_norm)
