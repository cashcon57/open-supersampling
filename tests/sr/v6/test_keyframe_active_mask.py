"""Tests for v6 KeyframeActiveMaskCache (4DGS-1K)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from oss.sr.v6.keyframe_active_mask import KeyframeActiveMaskCache


@dataclass
class FakeCanvas:
    positions: torch.Tensor
    scales: torch.Tensor
    count: int
    output_hw: tuple[int, int]


def _canvas(positions, scales=None, output_hw=(64, 64), count=None):
    n = positions.shape[0]
    if scales is None:
        scales = torch.ones(n, 2)
    return FakeCanvas(
        positions=positions,
        scales=scales,
        count=n if count is None else count,
        output_hw=output_hw,
    )


def test_mask_shape_and_dtype():
    canvas = _canvas(torch.tensor([[10.0, 10.0], [50.0, 50.0]]))
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    mask = cache.get_mask(0, canvas, view_matrix=torch.eye(2))
    assert mask.shape == (2,)
    assert mask.dtype == torch.bool


def test_inside_viewport_marked_active():
    """A Gaussian centered well inside the viewport should be active."""
    canvas = _canvas(
        torch.tensor([[32.0, 32.0]]),
        scales=torch.tensor([[1.0, 1.0]]),
    )
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    mask = cache.get_mask(0, canvas, view_matrix=torch.eye(2))
    assert bool(mask[0]) is True


def test_far_outside_viewport_marked_inactive():
    canvas = _canvas(
        torch.tensor([[1000.0, 1000.0]]),
        scales=torch.tensor([[1.0, 1.0]]),
    )
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    mask = cache.get_mask(0, canvas, view_matrix=torch.eye(2))
    assert bool(mask[0]) is False


def test_3sigma_extent_overlap_counts():
    """A Gaussian centered just outside the viewport but with a 3-sigma
    envelope that crosses the boundary should still be active."""
    # Center at x=66 (outside W=64). Scale=1 -> 3-sigma=3, bbox [63, 69].
    # x+r >= 0 (yes), x-r < 64 (63 < 64) -> active.
    canvas = _canvas(
        torch.tensor([[66.0, 32.0]]),
        scales=torch.tensor([[1.0, 1.0]]),
    )
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    mask = cache.get_mask(0, canvas, view_matrix=torch.eye(2))
    assert bool(mask[0]) is True

    # Same x but tiny scale -> bbox doesn't reach viewport.
    canvas2 = _canvas(
        torch.tensor([[66.0, 32.0]]),
        scales=torch.tensor([[0.1, 0.1]]),
    )
    cache2 = KeyframeActiveMaskCache(keyframe_interval=10)
    mask2 = cache2.get_mask(0, canvas2, view_matrix=torch.eye(2))
    assert bool(mask2[0]) is False


def test_keyframe_recomputes_others_reuse_cache():
    canvas = _canvas(torch.tensor([[32.0, 32.0]]))
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    mask_kf = cache.get_mask(0, canvas, view_matrix=torch.eye(2))
    assert bool(mask_kf[0]) is True

    # Move the canvas off-screen — but on a non-keyframe, the cache should
    # return the previous (still-active) mask.
    canvas.positions = torch.tensor([[5000.0, 5000.0]])
    mask_intermediate = cache.get_mask(3, canvas, view_matrix=torch.eye(2))
    assert bool(mask_intermediate[0]) is True

    # Hit the next keyframe boundary -> recompute, should now be inactive.
    mask_recompute = cache.get_mask(10, canvas, view_matrix=torch.eye(2))
    assert bool(mask_recompute[0]) is False


def test_reset_invalidates_cache():
    canvas = _canvas(torch.tensor([[32.0, 32.0]]))
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    cache.get_mask(0, canvas, view_matrix=torch.eye(2))

    canvas.positions = torch.tensor([[5000.0, 5000.0]])
    cache.reset()
    # frame_index=3 is not a keyframe, but reset() forces recompute.
    mask = cache.get_mask(3, canvas, view_matrix=torch.eye(2))
    assert bool(mask[0]) is False


def test_view_matrix_2x3_affine():
    """View matrix as (2, 3) [R | t] with translation."""
    canvas = _canvas(torch.tensor([[0.0, 0.0]]))
    # Translate by (32, 32) — center -> (32, 32) inside viewport.
    vm = torch.tensor([[1.0, 0.0, 32.0], [0.0, 1.0, 32.0]])
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    mask = cache.get_mask(0, canvas, view_matrix=vm)
    assert bool(mask[0]) is True


def test_view_matrix_3x3_homogeneous():
    canvas = _canvas(torch.tensor([[0.0, 0.0]]))
    vm = torch.tensor([[1.0, 0.0, 32.0], [0.0, 1.0, 32.0], [0.0, 0.0, 1.0]])
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    mask = cache.get_mask(0, canvas, view_matrix=vm)
    assert bool(mask[0]) is True


def test_dead_tail_padded_false():
    """Slots beyond canvas.count are always inactive."""
    pos = torch.tensor([[32.0, 32.0], [32.0, 32.0], [32.0, 32.0]])
    scales = torch.ones(3, 2)
    canvas = FakeCanvas(positions=pos, scales=scales, count=2, output_hw=(64, 64))
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    mask = cache.get_mask(0, canvas, view_matrix=torch.eye(2))
    assert mask.shape == (3,)
    assert bool(mask[0]) is True
    assert bool(mask[1]) is True
    assert bool(mask[2]) is False  # dead tail


def test_empty_canvas():
    pos = torch.zeros(0, 2)
    scales = torch.zeros(0, 2)
    canvas = FakeCanvas(positions=pos, scales=scales, count=0, output_hw=(64, 64))
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    mask = cache.get_mask(0, canvas, view_matrix=torch.eye(2))
    assert mask.shape == (0,)
    assert mask.dtype == torch.bool


def test_isotropic_scales_supported():
    """``scales`` may be (N,) instead of (N, 2)."""
    canvas = FakeCanvas(
        positions=torch.tensor([[32.0, 32.0]]),
        scales=torch.tensor([1.0]),  # 1D
        count=1,
        output_hw=(64, 64),
    )
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    mask = cache.get_mask(0, canvas, view_matrix=torch.eye(2))
    assert bool(mask[0]) is True


def test_invalid_keyframe_interval():
    with pytest.raises(ValueError):
        KeyframeActiveMaskCache(keyframe_interval=0)
    with pytest.raises(ValueError):
        KeyframeActiveMaskCache(keyframe_interval=-5)


def test_invalid_view_matrix_shape():
    canvas = _canvas(torch.tensor([[32.0, 32.0]]))
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    with pytest.raises(ValueError):
        cache.get_mask(0, canvas, view_matrix=torch.zeros(4, 4))


def test_canvas_count_resize_invalidates():
    """If canvas N changes, cached mask must not be reused even on
    a non-keyframe."""
    canvas = _canvas(torch.tensor([[32.0, 32.0], [32.0, 32.0]]))
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    cache.get_mask(0, canvas, view_matrix=torch.eye(2))

    # Resize canvas to a different N.
    canvas.positions = torch.tensor([[32.0, 32.0]])
    canvas.scales = torch.ones(1, 2)
    canvas.count = 1
    mask = cache.get_mask(3, canvas, view_matrix=torch.eye(2))  # non-keyframe
    assert mask.shape == (1,)


def test_bf16_positions():
    canvas = _canvas(torch.tensor([[32.0, 32.0]]).to(torch.bfloat16))
    cache = KeyframeActiveMaskCache(keyframe_interval=10)
    mask = cache.get_mask(0, canvas, view_matrix=torch.eye(2))
    assert mask.dtype == torch.bool
    assert bool(mask[0]) is True
