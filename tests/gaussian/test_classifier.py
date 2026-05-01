"""Sprint 3 tile classifier tests.

All tests run on CPU. They use synthetic inputs designed so the expected
"complex" region is known a priori, avoiding brittle dependence on real frames
(Sprint 2 G-buffer dumps land later).

Spec: docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md
Plan: docs/superpowers/plans/2026-05-01-gaussian-sprint-3-plan.md (T3.5)
"""

from __future__ import annotations

import math

import pytest
import torch

from oss.gaussian.classifier import (
    DEFAULT_TILE_SIZE,
    FeatureWeights,
    TileClassifier,
    overlay_mask,
)


T = DEFAULT_TILE_SIZE


def _smooth_frame(h: int, w: int) -> torch.Tensor:
    """A horizontal gradient image — low Sobel response, but not zero."""
    xs = torch.linspace(0.0, 1.0, w)
    img = xs.view(1, 1, 1, w).expand(1, 3, h, w).clone()
    return img


def _flat_depth(h: int, w: int, value: float = 5.0) -> torch.Tensor:
    return torch.full((1, 1, h, w), value)


def _zero_motion(h: int, w: int) -> torch.Tensor:
    return torch.zeros(1, 2, h, w)


def _tile_slice(tile_y: int, tile_x: int, count_y: int = 1, count_x: int = 1):
    return (
        slice(tile_y * T, (tile_y + count_y) * T),
        slice(tile_x * T, (tile_x + count_x) * T),
    )


# --------------------------------------------------------------- shape contract


@pytest.mark.parametrize("h,w,b", [(128, 128, 1), (256, 384, 2), (720, 1280, 1)])
def test_shape_correctness(h: int, w: int, b: int) -> None:
    classifier = TileClassifier()
    frame = torch.zeros(b, 3, h, w)
    depth = torch.ones(b, 1, h, w)
    motion = torch.zeros(b, 2, h, w)
    mask = classifier(frame, depth, motion)
    assert mask.shape == (b, h // T, w // T)
    assert mask.dtype == torch.bool


def test_optional_normals_runs_with_or_without() -> None:
    classifier = TileClassifier()
    frame = torch.rand(1, 3, 64, 64)
    depth = torch.rand(1, 1, 64, 64) + 0.1
    motion = torch.rand(1, 2, 64, 64)
    normals = torch.nn.functional.normalize(torch.rand(1, 3, 64, 64), dim=1)
    m1 = classifier(frame, depth, motion)
    m2 = classifier(frame, depth, motion, normals)
    assert m1.shape == m2.shape == (1, 64 // T, 64 // T)


# --------------------------------------------------------- input rejection


def test_rejects_non_multiple_dims() -> None:
    classifier = TileClassifier()
    # 130 isn't a multiple of 16.
    with pytest.raises(ValueError, match="multiples of tile_size"):
        classifier(
            torch.zeros(1, 3, 130, 128),
            torch.zeros(1, 1, 130, 128),
            torch.zeros(1, 2, 130, 128),
        )


def test_rejects_mismatched_batch() -> None:
    classifier = TileClassifier()
    with pytest.raises(ValueError, match="batch sizes disagree"):
        classifier(
            torch.zeros(1, 3, 64, 64),
            torch.zeros(2, 1, 64, 64),
            torch.zeros(1, 2, 64, 64),
        )


def test_rejects_wrong_channels() -> None:
    classifier = TileClassifier()
    with pytest.raises(ValueError, match=r"frame must be"):
        classifier(
            torch.zeros(1, 4, 64, 64),
            torch.zeros(1, 1, 64, 64),
            torch.zeros(1, 2, 64, 64),
        )


# --------------------------------------------------------- positive cases


def test_smooth_vs_noisy_patch() -> None:
    """A smooth image with a noisy patch — noisy tiles must be marked complex."""
    h, w = 64, 64
    frame = _smooth_frame(h, w)
    # Drop a high-noise patch covering tiles (1,1)..(2,2) (a 2x2 tile region).
    ys, xs = _tile_slice(1, 1, count_y=2, count_x=2)
    torch.manual_seed(0)
    frame[:, :, ys, xs] = torch.rand(1, 3, 2 * T, 2 * T)
    classifier = TileClassifier(target_complex_fraction=0.30)
    mask = classifier(frame, _flat_depth(h, w), _zero_motion(h, w))
    # Every tile in the noisy patch must be flagged.
    assert mask[0, 1:3, 1:3].all(), f"noisy patch missed: {mask[0]}"


def test_depth_discontinuity_marks_step() -> None:
    """Flat color, but depth has a step — those tiles must be flagged.

    The step is placed at y=24 (mid-tile of tile-row 1) so the discontinuity
    falls strictly inside tile-row 1's interior; that tile-row must light up.
    """
    h, w = 64, 64
    frame = torch.full((1, 3, h, w), 0.5)
    depth = torch.full((1, 1, h, w), 2.0)
    depth[:, :, 24:, :] = 20.0  # step inside tile-row 1 (y=16..31)
    classifier = TileClassifier(target_complex_fraction=0.20)
    mask = classifier(frame, depth, _zero_motion(h, w))
    boundary_row = mask[0, 1, :]
    assert boundary_row.any(), f"depth-step tile-row 1 didn't fire: {mask[0]}"


def test_motion_marks_moving_region() -> None:
    """Static color/depth, but a region moves — that region must be flagged."""
    h, w = 64, 64
    frame = torch.full((1, 3, h, w), 0.5)
    depth = _flat_depth(h, w)
    motion = torch.zeros(1, 2, h, w)
    ys, xs = _tile_slice(0, 0, count_y=2, count_x=2)
    motion[:, :, ys, xs] = 5.0  # strong motion in upper-left 2x2 tile region
    classifier = TileClassifier(target_complex_fraction=0.30)
    mask = classifier(frame, depth, motion)
    assert mask[0, 0:2, 0:2].any(), f"moving tiles missed: {mask[0]}"


# --------------------------------------------------------- threshold honoring


def test_threshold_honors_target_fraction_uniformly() -> None:
    """Across many seeds, mask fraction should land within +/-5% of target."""
    h, w = 256, 256
    target = 0.30
    classifier = TileClassifier(target_complex_fraction=target)

    fracs = []
    for seed in range(20):
        gen = torch.Generator().manual_seed(seed)
        frame = torch.rand(1, 3, h, w, generator=gen)
        depth = torch.rand(1, 1, h, w, generator=gen) + 0.1
        motion = torch.rand(1, 2, h, w, generator=gen)
        mask = classifier(frame, depth, motion)
        fracs.append(mask.float().mean().item())
    avg = sum(fracs) / len(fracs)
    assert abs(avg - target) <= 0.05, f"avg complex fraction {avg} too far from {target}"


def test_threshold_zero_and_one_edge_cases() -> None:
    h, w = 64, 64
    frame = torch.rand(1, 3, h, w)
    depth = torch.rand(1, 1, h, w) + 0.1
    motion = torch.rand(1, 2, h, w)

    none = TileClassifier(target_complex_fraction=0.0)
    all_ = TileClassifier(target_complex_fraction=1.0)
    assert not none(frame, depth, motion).any()
    assert all_(frame, depth, motion).all()


# --------------------------------------------------------- visualization


def test_overlay_mask_shape_and_blend() -> None:
    h, w = 32, 32
    frame = torch.zeros(1, 3, h, w)  # black
    mask = torch.zeros(1, h // T, w // T, dtype=torch.bool)
    mask[0, 0, 0] = True  # one tile complex
    out = overlay_mask(frame, mask, tile_size=T, color=(1.0, 0.0, 0.0), alpha=0.5)
    assert out.shape == frame.shape
    # The complex tile should now have a red component ~0.5; rest still black.
    assert math.isclose(out[0, 0, 0, 0].item(), 0.5, abs_tol=1e-6)
    assert out[0, 0, T, T].item() == 0.0  # outside complex tile


def test_overlay_mask_rejects_size_mismatch() -> None:
    frame = torch.zeros(1, 3, 32, 32)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)  # 4*16 = 64 != 32
    with pytest.raises(ValueError, match="does not match frame"):
        overlay_mask(frame, mask, tile_size=T)


# --------------------------------------------------------- weights wiring


def test_feature_weights_zero_disables_feature() -> None:
    """Zeroing the depth weight should change the per-tile score map."""
    h, w = 64, 64
    frame = torch.full((1, 3, h, w), 0.5)
    depth = torch.full((1, 1, h, w), 2.0)
    depth[:, :, 24:, :] = 20.0  # step inside tile-row 1
    motion = _zero_motion(h, w)

    on = TileClassifier(target_complex_fraction=0.30)
    off = TileClassifier(
        target_complex_fraction=0.30,
        weights=FeatureWeights(rgb_grad=1.0, depth_disc=0.0, motion=0.5, normal_var=0.25),
    )
    # Compare the underlying score maps, not the mask: at very low feature
    # variance the threshold step can collapse both to all-False due to ties.
    s_on = on.score(frame, depth, motion)
    s_off = off.score(frame, depth, motion)
    assert not torch.allclose(s_on, s_off)
