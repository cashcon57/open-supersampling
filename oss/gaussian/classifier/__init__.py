"""OSS-Gaussian tile classifier (Sprint 3).

Per-frame 16x16 tile mask classifying complex tiles (need Gaussian param
prediction) vs simple tiles (bilinear passthrough). Pure PyTorch; runs on
CPU + CUDA from one code path.
"""

from .classifier import (
    DEFAULT_TILE_SIZE,
    FeatureWeights,
    TileClassifier,
    overlay_mask,
)

__all__ = [
    "TileClassifier",
    "FeatureWeights",
    "DEFAULT_TILE_SIZE",
    "overlay_mask",
]
