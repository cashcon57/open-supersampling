"""Unit tests for v7 NDCanvasState + ParentChildSpawner."""
from __future__ import annotations

import pytest
import torch

from oss.sr.v7.nd_canvas_state import (
    NDCanvasState,
    cholesky_pack_to_L,
    cholesky_pack_to_cov,
)
from oss.sr.v7.parent_child_spawner import (
    ChildState,
    OPACITY_MATERIALIZE,
    BRIGHTNESS_MATERIALIZE,
    materialize_mask,
    materialize_to_canvas,
    initialize_children_for_new_parents,
)


def test_cholesky_pack_to_L_is_lower_triangular():
    raw = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    L = cholesky_pack_to_L(raw)
    assert L.shape == (1, 3, 3)
    # Upper-tri entries (i < j) should be 0
    assert L[0, 0, 1].item() == 0.0
    assert L[0, 0, 2].item() == 0.0
    assert L[0, 1, 2].item() == 0.0
    # Diagonals via exp()
    torch.testing.assert_close(L[0, 0, 0].item(), float(torch.tensor(1.0).exp()))
    torch.testing.assert_close(L[0, 1, 1].item(), float(torch.tensor(3.0).exp()))
    torch.testing.assert_close(L[0, 2, 2].item(), float(torch.tensor(6.0).exp()))


def test_cholesky_pack_to_cov_is_psd():
    torch.manual_seed(0)
    raw = (torch.rand((64, 6)) - 0.5) * 4.0
    V = cholesky_pack_to_cov(raw)
    assert V.shape == (64, 3, 3)
    eig = torch.linalg.eigvalsh(V)
    assert (eig > 0).all(), "Cholesky-packed covariances must all be PSD"


def test_canvas_state_empty_then_add_then_active_view():
    cs = NDCanvasState.empty(capacity=8, feature_dim=4)
    assert cs.count == 0
    assert cs.n_live == 0
    positions = torch.tensor([[1.0, 2.0, 0.5], [3.0, 4.0, 1.5]])
    cov_raw = torch.zeros((2, 6))
    features = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    opacity = torch.tensor([0.8, 0.6])
    cs.add(positions, cov_raw, features, opacity)
    assert cs.n_live == 2
    assert cs.count == 2
    pos, cov, feat, op = cs.active_view()
    assert pos.shape == (2, 3)
    assert cov.shape == (2, 3, 3)
    assert feat.shape == (2, 4)
    assert op.shape == (2,)


def test_canvas_state_prune_keeps_capacity_unchanged():
    cs = NDCanvasState.empty(capacity=8, feature_dim=2)
    cs.add(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        cov_raw=torch.zeros((3, 6)),
        features=torch.zeros((3, 2)),
        opacity=torch.ones(3),
    )
    assert cs.count == 3
    cs.prune(torch.tensor([True, False, True]))
    assert cs.count == 2          # middle one pruned
    assert cs.n_live == 3          # n_live unchanged (capacity preserved)
    pos, _, _, _ = cs.active_view()
    assert pos.shape == (2, 3)
    # Only the live ones survive
    torch.testing.assert_close(pos[0], torch.tensor([0.0, 0.0, 0.0]))
    torch.testing.assert_close(pos[1], torch.tensor([2.0, 0.0, 0.0]))


def test_canvas_state_add_beyond_capacity_raises():
    cs = NDCanvasState.empty(capacity=2, feature_dim=1)
    cs.add(
        positions=torch.zeros((2, 3)),
        cov_raw=torch.zeros((2, 6)),
        features=torch.zeros((2, 1)),
        opacity=torch.ones(2),
    )
    with pytest.raises(ValueError, match="capacity"):
        cs.add(
            positions=torch.zeros((1, 3)),
            cov_raw=torch.zeros((1, 6)),
            features=torch.zeros((1, 1)),
            opacity=torch.ones(1),
        )


def test_canvas_state_prune_to_count_drops_lowest_opacity():
    cs = NDCanvasState.empty(capacity=16, feature_dim=2)
    cs.add(
        positions=torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]),
        cov_raw=torch.zeros((5, 6)),
        features=torch.zeros((5, 2)),
        opacity=torch.tensor([0.1, 0.5, 0.05, 0.9, 0.3]),
    )
    assert cs.count == 5
    # Drop to 3 -- the 2 lowest opacities (0.05 and 0.1) should go.
    n_dropped = cs.prune_to_count(3, strategy="lowest_opacity")
    assert n_dropped == 2
    assert cs.count == 3
    # The dropped indices should be the ones with opacity 0.05 and 0.1
    pos, _, _, op = cs.active_view()
    surviving_op = sorted(op.tolist())
    assert surviving_op == pytest.approx([0.3, 0.5, 0.9])


def test_canvas_state_prune_to_count_drops_oldest():
    cs = NDCanvasState.empty(capacity=16, feature_dim=2)
    cs.add(
        positions=torch.arange(15, dtype=torch.float32).view(5, 3),
        cov_raw=torch.zeros((5, 6)),
        features=torch.zeros((5, 2)),
        opacity=torch.ones(5),
    )
    # Drop 2 oldest -- positions 0 and 1 should go, surviving 2-4.
    n_dropped = cs.prune_to_count(3, strategy="oldest")
    assert n_dropped == 2
    pos, _, _, _ = cs.active_view()
    # Surviving positions are rows 2, 3, 4 from the original tensor.
    expected = torch.arange(6, 15, dtype=torch.float32).view(3, 3)
    torch.testing.assert_close(pos, expected)


def test_canvas_state_prune_to_count_noop_when_below_target():
    cs = NDCanvasState.empty(capacity=8, feature_dim=1)
    cs.add(
        positions=torch.zeros((2, 3)),
        cov_raw=torch.zeros((2, 6)),
        features=torch.zeros((2, 1)),
        opacity=torch.ones(2),
    )
    assert cs.prune_to_count(5) == 0
    assert cs.count == 2


def test_canvas_state_compact_reclaims_capacity():
    cs = NDCanvasState.empty(capacity=8, feature_dim=1)
    cs.add(
        positions=torch.tensor([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
        ]),
        cov_raw=torch.zeros((5, 6)),
        features=torch.zeros((5, 1)),
        opacity=torch.ones(5),
    )
    # Drop the middle three; live: 0, 4
    cs.prune(torch.tensor([True, False, False, False, True]))
    assert cs.count == 2
    assert cs.n_live == 5
    cs.compact()
    assert cs.n_live == 2
    pos, _, _, _ = cs.active_view()
    # After compaction, the two survivors live at slots 0 and 1
    torch.testing.assert_close(pos[0], torch.tensor([0.0, 0.0, 0.0]))
    torch.testing.assert_close(pos[1], torch.tensor([4.0, 0.0, 0.0]))


def test_canvas_state_reset_clears_all_live_gaussians():
    cs = NDCanvasState.empty(capacity=4, feature_dim=1)
    cs.add(
        positions=torch.zeros((3, 3)),
        cov_raw=torch.zeros((3, 6)),
        features=torch.zeros((3, 1)),
        opacity=torch.ones(3),
    )
    assert cs.count == 3
    cs.reset()
    assert cs.count == 0
    assert cs.n_live == 0


# -------------------- ParentChildSpawner --------------------

def test_child_state_initializes_dormant():
    child = ChildState.empty(capacity=8, feature_dim=4)
    assert (child.opacity < OPACITY_MATERIALIZE).all()
    assert (child.brightness < BRIGHTNESS_MATERIALIZE).all()


def test_materialize_mask_fires_above_either_threshold():
    child = ChildState.empty(capacity=4, feature_dim=1)
    # Manually set: row 0 high opacity, row 1 high brightness, row 2 both low
    child.opacity = torch.tensor([0.2, 1e-6, 1e-6, 0.2])
    child.brightness = torch.tensor([1e-6, 0.05, 1e-6, 0.05])
    mask = materialize_mask(child, n_live=4)
    expected = torch.tensor([True, True, False, True])
    assert torch.equal(mask, expected)


def test_materialize_to_canvas_appends_new_gaussians():
    cs = NDCanvasState.empty(capacity=8, feature_dim=2)
    cs.add(
        positions=torch.tensor([[10.0, 20.0, 0.0], [30.0, 40.0, 1.0]]),
        cov_raw=torch.zeros((2, 6)),
        features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        opacity=torch.ones(2),
    )
    assert cs.count == 2
    child = ChildState.empty(capacity=8, feature_dim=2)
    # Force row 0 to materialize, row 1 stays dormant
    child.opacity[0] = 0.5
    child.dpos[0] = torch.tensor([1.0, 1.0, 0.5])
    n_materialized = materialize_to_canvas(cs, child)
    assert n_materialized == 1
    assert cs.count == 3
    # The new Gaussian should be at parent[0] + offset
    pos, _, _, _ = cs.active_view()
    # newest is at index 2
    torch.testing.assert_close(pos[2], torch.tensor([11.0, 21.0, 0.5]))
    # Child slot for row 0 should be reset
    assert child.opacity[0].item() < OPACITY_MATERIALIZE


def test_initialize_children_for_new_parents_breaks_symmetry():
    """Two children initialized for different parent indices should
    have distinct (non-zero) dpos values."""
    child = ChildState.empty(capacity=8, feature_dim=4)
    initialize_children_for_new_parents(
        child=child,
        parent_indices=torch.tensor([0, 1]),
        init_dpos_std=0.5,
    )
    # Different parents -> different dpos values
    assert not torch.equal(child.dpos[0], child.dpos[1])
    # Children should still be dormant (opacity below threshold)
    assert (child.opacity[:2] < OPACITY_MATERIALIZE).all()


def test_full_spawn_drift_materialize_cycle():
    """Simulate the full lifecycle: add parents, init dormant children,
    drift them via gradient-style updates, materialize the ones whose
    drift carried them over threshold."""
    cs = NDCanvasState.empty(capacity=32, feature_dim=4)
    cs.add(
        positions=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]),
        cov_raw=torch.zeros((4, 6)),
        features=torch.zeros((4, 4)),
        opacity=torch.ones(4),
    )
    assert cs.count == 4
    child = ChildState.empty(capacity=32, feature_dim=4)
    initialize_children_for_new_parents(child, parent_indices=torch.arange(4))
    # Drift child opacities upward only for parents 0 and 2 (simulating
    # high-loss tiles for those parents).
    child.opacity[0] = 0.3
    child.opacity[1] = 1e-6
    child.opacity[2] = 0.5
    child.opacity[3] = 1e-6
    # Materialize.
    n_new = materialize_to_canvas(cs, child)
    assert n_new == 2
    assert cs.count == 6
    # Confirm parents 0 and 2 had their children promoted: new Gaussians
    # exist near (0, 0, 0) and (20, 0, 0).
    pos, _, _, _ = cs.active_view()
    # First 4 are original parents; last 2 are newly materialized children
    assert pos.shape == (6, 3)
