"""OSS-Gaussian renderer module.

Public API:
- GaussianBatch — typed container for a batch of 2D Gaussians.
- Rasterizer    — tile-based top-K renderer (CUDA via vendored gsplat,
                  with a slow PyTorch reference fallback).
- TILE_SIZE     — the renderer's tile edge length in pixels (16, hardcoded
                  in the upstream Image-GS CUDA kernel).
"""

from oss.gaussian.renderer.rasterizer import (
    GaussianBatch,
    Rasterizer,
    TILE_SIZE,
)

__all__ = ["GaussianBatch", "Rasterizer", "TILE_SIZE"]
