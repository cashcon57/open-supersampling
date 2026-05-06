"""Tests for v6 Spatial-Temporal Variation Score pruning (4DGS-1K)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from oss.sr.v6.st_variation_score import (
    STVScoreState,
    init_st_score_state,
    prune_by_st_score,
    st_variation_score,
    update_st_score,
)


@dataclass
class FakeCanvas:
    """Minimal duck-typed canvas with the documented v6 interface."""

    positions: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    count: int


def _make_fake_canvas(n: int) -> FakeCanvas:
    return FakeCanvas(
        positions=torch.arange(n * 2, dtype=torch.float32).reshape(n, 2),
        scales=torch.ones(n, 2),
        rotations=torch.zeros(n),
        opacities=torch.full((n,), 0.5),
        colors=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
        count=n,
    )


def test_init_state_shape_and_dtype():
    s = init_st_score_state(5, dtype=torch.float32)
    assert s.spatial_accumulator.shape == (5,)
    assert s.spatial_accumulator.dtype == torch.float32
    assert s.lifespan_count.shape == (5,)
    assert s.lifespan_count.dtype == torch.int64
    assert s.frames_observed == 0


def test_update_accumulates_alpha_T():
    """SS_i is sum_p (alpha * T) per Gaussian, summed across update calls."""
    n, p = 3, 4
    state = init_st_score_state(n)
    alpha = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0],
    ])
    T = torch.ones(n, p)
    active = torch.tensor([True, True, False])

    update_st_score(state, alpha, T, active)
    # Gaussian 0: 2.0; Gaussian 1: 2.0; Gaussian 2: 0.0
    assert torch.allclose(state.spatial_accumulator, torch.tensor([2.0, 2.0, 0.0]))
    assert state.lifespan_count.tolist() == [1, 1, 0]
    assert state.frames_observed == 1

    update_st_score(state, alpha, T, active)
    assert torch.allclose(state.spatial_accumulator, torch.tensor([4.0, 4.0, 0.0]))
    assert state.lifespan_count.tolist() == [2, 2, 0]
    assert state.frames_observed == 2


def test_score_is_product_SS_TS():
    state = init_st_score_state(3)
    state.spatial_accumulator = torch.tensor([1.0, 2.0, 3.0])
    state.lifespan_count = torch.tensor([4, 5, 0], dtype=torch.int64)
    score = st_variation_score(state)
    assert torch.allclose(score, torch.tensor([4.0, 10.0, 0.0]))


def test_prune_keeps_highest_scoring():
    n = 10
    canvas = _make_fake_canvas(n)
    state = init_st_score_state(n)
    # Score determined entirely by SS for fixed TS=1.
    state.spatial_accumulator = torch.tensor([0.1, 5.0, 0.2, 4.0, 0.3, 3.0, 0.4, 2.0, 0.5, 1.0])
    state.lifespan_count = torch.ones(n, dtype=torch.int64)
    state.frames_observed = 1

    canvas, new_state = prune_by_st_score(canvas, state, prune_fraction=0.7)
    # n_keep = round(10 * 0.3) = 3.
    assert canvas.count == 3
    # Top three SS values are 5.0, 4.0, 3.0 at indices 1, 3, 5.
    assert canvas.positions.shape == (3, 2)
    expected_pos = torch.tensor([[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]])
    assert torch.allclose(canvas.positions, expected_pos)
    assert new_state.spatial_accumulator.tolist() == [5.0, 4.0, 3.0]
    assert new_state.frames_observed == 1


def test_prune_fraction_zero_is_noop():
    n = 4
    canvas = _make_fake_canvas(n)
    state = init_st_score_state(n)
    state.spatial_accumulator = torch.tensor([3.0, 1.0, 4.0, 2.0])
    state.lifespan_count = torch.ones(n, dtype=torch.int64)

    canvas, new_state = prune_by_st_score(canvas, state, prune_fraction=0.0)
    assert canvas.count == n
    assert new_state.spatial_accumulator.tolist() == [3.0, 1.0, 4.0, 2.0]


def test_prune_keeps_at_least_one():
    """Even fraction=0.99 with N=2 must keep >=1 to avoid empty canvas."""
    n = 2
    canvas = _make_fake_canvas(n)
    state = init_st_score_state(n)
    state.spatial_accumulator = torch.tensor([0.0, 5.0])
    state.lifespan_count = torch.ones(n, dtype=torch.int64)
    canvas, _ = prune_by_st_score(canvas, state, prune_fraction=0.99)
    assert canvas.count >= 1


def test_empty_canvas():
    canvas = _make_fake_canvas(0)
    state = init_st_score_state(0)
    canvas, new_state = prune_by_st_score(canvas, state, prune_fraction=0.7)
    assert canvas.count == 0
    assert new_state.spatial_accumulator.shape == (0,)


def test_all_zero_alpha_yields_zero_score():
    n, p = 5, 3
    state = init_st_score_state(n)
    alpha = torch.zeros(n, p)
    T = torch.ones(n, p)
    active = torch.ones(n, dtype=torch.bool)
    update_st_score(state, alpha, T, active)
    score = st_variation_score(state)
    assert torch.allclose(score, torch.zeros(n))


def test_invalid_prune_fraction_rejected():
    n = 4
    canvas = _make_fake_canvas(n)
    state = init_st_score_state(n)
    with pytest.raises(ValueError):
        prune_by_st_score(canvas, state, prune_fraction=-0.1)
    with pytest.raises(ValueError):
        prune_by_st_score(canvas, state, prune_fraction=1.0)


def test_update_validates_shapes():
    state = init_st_score_state(3)
    with pytest.raises(ValueError):
        update_st_score(
            state,
            torch.zeros(2, 4),  # wrong N
            torch.zeros(3, 4),
            torch.ones(3, dtype=torch.bool),
        )
    with pytest.raises(ValueError):
        update_st_score(
            state,
            torch.zeros(3, 4),
            torch.zeros(3, 5),  # mismatched P
            torch.ones(3, dtype=torch.bool),
        )
    with pytest.raises(ValueError):
        update_st_score(
            state,
            torch.zeros(3, 4),
            torch.zeros(3, 4),
            torch.ones(2, dtype=torch.bool),  # wrong N
        )


def test_canvas_missing_attrs_rejected():
    @dataclass
    class BadCanvas:
        positions: torch.Tensor
        count: int

    bad = BadCanvas(positions=torch.zeros(3, 2), count=3)
    state = init_st_score_state(3)
    state.spatial_accumulator = torch.tensor([1.0, 2.0, 3.0])
    state.lifespan_count = torch.ones(3, dtype=torch.int64)
    with pytest.raises(AttributeError):
        prune_by_st_score(bad, state, prune_fraction=0.5)


def test_bf16_safe():
    n, p = 4, 3
    state = init_st_score_state(n, dtype=torch.bfloat16)
    alpha = torch.full((n, p), 0.5, dtype=torch.bfloat16)
    T = torch.full((n, p), 0.5, dtype=torch.bfloat16)
    active = torch.ones(n, dtype=torch.bool)
    update_st_score(state, alpha, T, active)
    # 0.25 * 3 = 0.75 per Gaussian.
    assert torch.allclose(
        state.spatial_accumulator.float(),
        torch.full((n,), 0.75),
        atol=0.05,
    )


def test_score_gradient_flow():
    """SS_i contributions must remain differentiable wrt alpha and T."""
    n, p = 2, 3
    alpha = torch.full((n, p), 0.5, requires_grad=True)
    T = torch.full((n, p), 0.5, requires_grad=True)
    active = torch.ones(n, dtype=torch.bool)
    contrib = (alpha * T).sum(dim=1)  # what update_st_score computes
    loss = contrib.sum()
    loss.backward()
    assert alpha.grad is not None
    assert T.grad is not None
    assert torch.isfinite(alpha.grad).all()
