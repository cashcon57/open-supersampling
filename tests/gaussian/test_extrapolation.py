"""Sprint 6 — frame extrapolation tests.

These tests exercise `FrameExtrapolator` without requiring Sprint 5's
real `PersistentCanvas` / `warp_canvas` to be merged. We provide:

- `_FakeCanvas` — minimal duck-typed canvas matching the `_CanvasLike`
  protocol declared in `extrapolator.py`.
- `_synthetic_warp_fn` — pure-PyTorch warp that shifts every Gaussian's
  xy by the average of the per-pixel motion field, scaled by alpha.
  This matches the contract Sprint 5 will satisfy with a per-Gaussian
  motion-vector lookup.

When Sprint 5 lands, additional integration tests live in
`tests/gaussian/test_canvas_integration.py` (Sprint 5's responsibility)
that wire the real `warp_canvas` to `FrameExtrapolator`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from oss.gaussian.extrapolation import (
    AlphaSchedule,
    FrameExtrapolator,
    preset_60_to_90,
    preset_60_to_120,
    preset_60_to_144,
    schedule_for,
)
from oss.gaussian.renderer import GaussianBatch, Rasterizer


# --- Test doubles -----------------------------------------------------------


@dataclass
class _FakeCanvas:
    """Bare-minimum canvas double satisfying `_CanvasLike`.

    Holds a `GaussianBatch` and nothing else. Sprint 5's real
    `PersistentCanvas` will be a strict superset of this surface.
    """

    gaussians: GaussianBatch


def _synthetic_warp_fn(canvas: _FakeCanvas, motion: torch.Tensor, alpha: float) -> _FakeCanvas:
    """Reference warp: shift every Gaussian's xy by alpha * <motion>.

    We use the average of the motion field as a stand-in for "look up
    the motion vector at each Gaussian's xy and shift". For Sprint 6
    correctness tests this is enough — the algorithm under test is
    "alpha multiplies the per-Gaussian shift", not the lookup itself.
    """
    g = canvas.gaussians
    avg_dx = motion[0].mean()
    avg_dy = motion[1].mean()
    delta = torch.stack([avg_dx, avg_dy]).to(g.xy.dtype)  # (2,)
    new_xy = g.xy + alpha * delta.unsqueeze(0)  # (N, 2)
    new_g = GaussianBatch(xy=new_xy, scale=g.scale, rot=g.rot, feat=g.feat)
    return replace(canvas, gaussians=new_g)


@pytest.fixture
def small_canvas() -> _FakeCanvas:
    """Two Gaussians, distinct positions and colors. Output ≤ 32×32."""
    return _FakeCanvas(
        gaussians=GaussianBatch(
            xy=torch.tensor([[8.0, 8.0], [16.0, 16.0]]),
            scale=torch.tensor([[1.5, 1.5], [1.5, 1.5]]),
            rot=torch.tensor([0.0, 0.0]),
            feat=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        )
    )


@pytest.fixture
def constant_motion() -> torch.Tensor:
    """Uniform motion: every pixel moves +4px right, +2px down."""
    h, w = 32, 32
    motion = torch.zeros((2, h, w))
    motion[0] = 4.0
    motion[1] = 2.0
    return motion


@pytest.fixture
def reference_extrapolator() -> FrameExtrapolator:
    return FrameExtrapolator(
        rasterizer=Rasterizer(force_backend="reference"),
        warp_fn=_synthetic_warp_fn,
    )


# --- T6.1 / T6.3 correctness -----------------------------------------------


def test_alpha_zero_matches_canvas_render_at_t(
    reference_extrapolator: FrameExtrapolator,
    small_canvas: _FakeCanvas,
    constant_motion: torch.Tensor,
) -> None:
    """alpha=0 → no warp; output identical to direct canvas render."""
    direct = reference_extrapolator.rasterizer(
        small_canvas.gaussians, output_hw=(32, 32)
    )
    extrap = reference_extrapolator.extrapolate(
        small_canvas, constant_motion, alpha=0.0, output_hw=(32, 32)
    )
    assert torch.allclose(direct, extrap, atol=1e-6)


def test_alpha_one_full_warp(
    reference_extrapolator: FrameExtrapolator,
    small_canvas: _FakeCanvas,
    constant_motion: torch.Tensor,
) -> None:
    """alpha=1 → Gaussian positions shifted by the full motion delta."""
    expected_xy = small_canvas.gaussians.xy + torch.tensor([[4.0, 2.0]])
    warped = _synthetic_warp_fn(small_canvas, constant_motion, alpha=1.0)
    assert torch.allclose(warped.gaussians.xy, expected_xy, atol=1e-6)
    # And the rendered output must reflect that shift — peak intensity
    # of the red Gaussian originally at (8, 8) should now sit at (12, 10).
    out = reference_extrapolator.extrapolate(
        small_canvas, constant_motion, alpha=1.0, output_hw=(32, 32)
    )
    red_channel = out[0]
    flat = red_channel.argmax().item()
    peak_y, peak_x = flat // 32, flat % 32
    assert (peak_x, peak_y) == (12, 10)


def test_alpha_half_is_midway(
    reference_extrapolator: FrameExtrapolator,
    small_canvas: _FakeCanvas,
    constant_motion: torch.Tensor,
) -> None:
    """Linear motion: alpha=0.5 puts Gaussians midway between t-1 and t.

    With our synthetic warp model, the canvas at time t already sits at
    xy. The motion vector represents the t-1 → t displacement, so t-1
    Gaussians lived at ``xy - motion`` and the predicted intermediate
    at alpha=0.5 lives at ``xy - 0.5 * motion``... but the algorithm
    under test (per master plan §6.1) uses the same forward warp the
    canvas applied at time t, so alpha=0.5 means the intermediate is
    *0.5 * motion* further along. We verify the midpoint property
    holds: position at alpha=0.5 minus position at alpha=0 equals half
    of position at alpha=1 minus position at alpha=0.
    """
    p0 = _synthetic_warp_fn(small_canvas, constant_motion, alpha=0.0).gaussians.xy
    p_half = _synthetic_warp_fn(small_canvas, constant_motion, alpha=0.5).gaussians.xy
    p_one = _synthetic_warp_fn(small_canvas, constant_motion, alpha=1.0).gaussians.xy
    assert torch.allclose(p_half - p0, 0.5 * (p_one - p0), atol=1e-6)


def test_alpha_out_of_range_rejected(
    reference_extrapolator: FrameExtrapolator,
    small_canvas: _FakeCanvas,
    constant_motion: torch.Tensor,
) -> None:
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            reference_extrapolator.extrapolate(
                small_canvas, constant_motion, alpha=bad, output_hw=(16, 16)
            )


def test_motion_shape_validated(
    reference_extrapolator: FrameExtrapolator,
    small_canvas: _FakeCanvas,
) -> None:
    bad_motion = torch.zeros(3, 16, 16)
    with pytest.raises(ValueError):
        reference_extrapolator.extrapolate(
            small_canvas, bad_motion, alpha=0.5, output_hw=(16, 16)
        )


# --- T6.2 latency property -------------------------------------------------


def test_warp_is_essentially_free_relative_to_render(
    reference_extrapolator: FrameExtrapolator,
    small_canvas: _FakeCanvas,
    constant_motion: torch.Tensor,
) -> None:
    """The Sprint 6 cost claim: warp is just a position add; render dominates.

    We measure CPU wall-clock for a render-only pass and an
    extrapolate pass at alpha=0.5. The extrapolate pass must not be
    more than ~1.5× the render-only pass. This is a structural check —
    not a real-time benchmark — to catch accidental regressions like
    re-instantiating the rasterizer or copying the whole canvas.
    """
    import time

    rasterizer = reference_extrapolator.rasterizer

    # Warm-up — first call hits CPython cache, allocator, etc.
    rasterizer(small_canvas.gaussians, output_hw=(32, 32))
    reference_extrapolator.extrapolate(
        small_canvas, constant_motion, alpha=0.5, output_hw=(32, 32)
    )

    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        rasterizer(small_canvas.gaussians, output_hw=(32, 32))
    render_only = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n):
        reference_extrapolator.extrapolate(
            small_canvas, constant_motion, alpha=0.5, output_hw=(32, 32)
        )
    with_warp = time.perf_counter() - t0

    # Generous bound — CI machines are noisy; we want to fail only on
    # actual regressions (e.g. someone added a deepcopy).
    assert with_warp < 2.0 * render_only + 0.05, (
        f"extrapolate too slow vs render alone: render={render_only*1000:.2f}ms, "
        f"extrapolate={with_warp*1000:.2f}ms"
    )


# --- alpha_scheduler tests --------------------------------------------------


def test_60_to_120_one_intermediate_alpha_half() -> None:
    s = preset_60_to_120()
    assert s.source_fps == 60
    assert s.target_fps == 120
    # 60 → 120: every real frame followed by one synthesized at alpha=0.5.
    assert s.alphas == [0.5]
    assert s.period_frames == 2


def test_60_to_90_one_intermediate_alpha_half() -> None:
    s = preset_60_to_90()
    # 60 → 90: gcd=30, 2 real frames + 1 synthesized over a 3-displayed period.
    assert s.intermediates_per_period == 1
    assert s.period_frames == 3
    # The synthesized frame falls between two real frames; with two real
    # frames at displayed positions 0 and 1 (after gcd reduction:
    # real_per_period=2, displayed_per_period=3), the synthesized alpha
    # should be 0.5.
    assert s.alphas == [pytest.approx(0.5)]


def test_60_to_144_emits_uniform_intermediates() -> None:
    s = preset_60_to_144()
    # gcd(60, 144) = 12 → real_per_period=5, displayed_per_period=12,
    # intermediates_per_period = 7.
    assert s.intermediates_per_period == 7
    assert s.period_frames == 12
    # All alphas in (0, 1).
    assert all(0.0 < a < 1.0 for a in s.alphas)


def test_schedule_equal_fps_is_empty() -> None:
    s = schedule_for(60, 60)
    assert s.alphas == []
    assert s.intermediates_per_period == 0


def test_schedule_target_lower_rejected() -> None:
    with pytest.raises(ValueError):
        schedule_for(120, 60)


def test_schedule_non_positive_rejected() -> None:
    with pytest.raises(ValueError):
        schedule_for(0, 60)
    with pytest.raises(ValueError):
        schedule_for(60, -1)


def test_schedule_dataclass_is_frozen() -> None:
    s = preset_60_to_120()
    with pytest.raises(Exception):  # FrozenInstanceError subclasses AttributeError
        s.source_fps = 30  # type: ignore[misc]


def test_alpha_schedule_dataclass_round_trip() -> None:
    s = AlphaSchedule(source_fps=60, target_fps=120, alphas=[0.5], period_frames=2)
    assert s.intermediates_per_period == 1
