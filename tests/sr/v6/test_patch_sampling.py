"""Tests for ``oss.sr.v6.patch_sampling``."""
from __future__ import annotations

import pytest
import torch

from oss.sr.v6.patch_sampling import (
    importance_weighted_patch_indices,
    sobel_gradient_magnitude,
)


def test_returns_requested_count() -> None:
    image = torch.rand(3, 64, 64)
    coords = importance_weighted_patch_indices(
        image, patch_size=16, num_patches=32, importance_ratio=0.7,
    )
    assert len(coords) == 32


def test_coords_within_bounds() -> None:
    image = torch.rand(3, 64, 96)
    patch = 24
    coords = importance_weighted_patch_indices(
        image, patch_size=patch, num_patches=20, importance_ratio=0.7,
    )
    H, W = image.shape[-2:]
    for top, left in coords:
        assert isinstance(top, int) and isinstance(left, int)
        assert 0 <= top <= H - patch
        assert 0 <= left <= W - patch


def test_image_smaller_than_patch_raises() -> None:
    image = torch.rand(3, 16, 16)
    with pytest.raises(ValueError, match="smaller than patch_size"):
        importance_weighted_patch_indices(image, patch_size=32, num_patches=4)


def test_zero_patches_returns_empty() -> None:
    image = torch.rand(3, 64, 64)
    assert importance_weighted_patch_indices(image, 16, 0) == []


def test_invalid_ratios_raise() -> None:
    image = torch.rand(3, 64, 64)
    with pytest.raises(ValueError, match="importance_ratio"):
        importance_weighted_patch_indices(image, 16, 4, importance_ratio=-0.1)
    with pytest.raises(ValueError, match="importance_ratio"):
        importance_weighted_patch_indices(image, 16, 4, importance_ratio=1.5)


def test_invalid_patch_size_raises() -> None:
    image = torch.rand(3, 64, 64)
    with pytest.raises(ValueError, match="patch_size"):
        importance_weighted_patch_indices(image, patch_size=0, num_patches=4)


def test_uniform_image_falls_back_to_uniform() -> None:
    # Constant image: gradient is zero everywhere; importance pool is empty.
    image = torch.full((3, 64, 64), 0.5)
    coords = importance_weighted_patch_indices(
        image, patch_size=16, num_patches=10, importance_ratio=0.7,
    )
    assert len(coords) == 10
    # All coords still valid.
    for top, left in coords:
        assert 0 <= top <= 64 - 16
        assert 0 <= left <= 64 - 16


def test_importance_ratio_respected_via_high_contrast_region() -> None:
    """A region with a sharp edge should get more importance-pool draws.

    We construct an image where the right half is bright and the left half
    is dark. Sobel magnitude is concentrated in the central column. With
    importance_ratio=1.0 (all draws importance-weighted), the average top-
    left x of the sampled patches should be near the edge column, far from
    the uniform-distribution mean.
    """
    H, W, patch = 64, 128, 16
    image = torch.zeros(3, H, W)
    image[:, :, W // 2:] = 1.0  # vertical step edge in the middle

    g = torch.Generator().manual_seed(0)
    importance_only = importance_weighted_patch_indices(
        image, patch_size=patch, num_patches=200, importance_ratio=1.0, generator=g,
    )
    g2 = torch.Generator().manual_seed(0)
    uniform_only = importance_weighted_patch_indices(
        image, patch_size=patch, num_patches=200, importance_ratio=0.0, generator=g2,
    )

    mean_imp_left = sum(c[1] for c in importance_only) / len(importance_only)
    mean_uni_left = sum(c[1] for c in uniform_only) / len(uniform_only)

    # The edge sits around column W/2 - patch/2 ≈ 56 in anchor space.
    # The uniform mean across anchors is roughly (W - patch) / 2 = 56 too,
    # but the importance distribution should spike near the edge column,
    # giving lower variance — which translates to a tighter mean. Validate
    # by checking the importance coords cluster around the edge.
    edge_col = W // 2 - patch // 2
    # Most importance-only patches should be within `patch` of the edge col.
    near_edge = sum(1 for c in importance_only if abs(c[1] - edge_col) <= patch)
    # Without overfitting, we just require importance-mode clusters more than uniform.
    near_edge_uni = sum(1 for c in uniform_only if abs(c[1] - edge_col) <= patch)
    assert near_edge > near_edge_uni, (
        f"importance not clustering near edge (importance={near_edge}, uniform={near_edge_uni})"
    )

    # Sanity: both means in valid range
    assert 0 <= mean_imp_left <= W - patch
    assert 0 <= mean_uni_left <= W - patch


def test_sobel_gradient_magnitude_shape() -> None:
    image = torch.rand(3, 32, 48)
    mag = sobel_gradient_magnitude(image)
    assert mag.shape == (1, 32, 48)
    assert (mag >= 0).all()


def test_more_patches_than_anchors_pads() -> None:
    # 4x4 image with patch=3 → only 2x2 = 4 anchor positions.
    image = torch.rand(3, 4, 4)
    coords = importance_weighted_patch_indices(
        image, patch_size=3, num_patches=10, importance_ratio=0.7,
    )
    # Must still return exactly 10.
    assert len(coords) == 10
    for top, left in coords:
        assert 0 <= top <= 1 and 0 <= left <= 1
