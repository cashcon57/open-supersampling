"""Pre-v6.3.1 / v7 unit tests for the parent-child deferred-materialization
spawner from Diolatzis et al. 2024.

The mechanism: every active Gaussian carries a dormant child Gaussian
expressed in the parent's reference frame. During training the child's
opacity / brightness drifts; when it crosses a fixed threshold the
child "materializes" into a full top-level Gaussian. Loss-adaptive
density control without explicit splitting heuristics.

These tests verify the mechanism in isolation -- no model, no
training loop -- so we know the materialization criterion fires
correctly before we wire it into the canvas state.
"""
from __future__ import annotations

import pytest
import torch


# Threshold values from the paper §3.3
OPACITY_MATERIALIZE = 0.1
BRIGHTNESS_MATERIALIZE = 0.01


def make_dormant_child(parent_opacity: torch.Tensor, parent_brightness: torch.Tensor,
                      child_opacity_init: float = 1e-6,
                      child_brightness_init: float = 1e-6):
    """Initialize a child Gaussian with negligible opacity / brightness."""
    n = parent_opacity.shape[0]
    return {
        "opacity": torch.full((n,), child_opacity_init),
        "brightness": torch.full((n,), child_brightness_init),
    }


def materialize_mask(child: dict) -> torch.Tensor:
    """Returns a (N,) bool mask of children that should be promoted to
    independent Gaussians on this step."""
    op_pass = child["opacity"] > OPACITY_MATERIALIZE
    br_pass = child["brightness"] > BRIGHTNESS_MATERIALIZE
    return op_pass | br_pass


def test_dormant_children_do_not_materialize_at_init():
    """At initialization, no children should be at materialization
    threshold."""
    parent_op = torch.tensor([0.5, 0.5, 0.5, 0.5])
    parent_br = torch.tensor([0.3, 0.3, 0.3, 0.3])
    child = make_dormant_child(parent_op, parent_br)
    assert not materialize_mask(child).any(), \
        "newly-initialized children should not be materialized"


def test_child_materializes_when_opacity_threshold_crossed():
    """Opacity drift past 0.1 should fire materialization."""
    child = {
        "opacity": torch.tensor([0.05, 0.099, 0.11, 0.5]),
        "brightness": torch.tensor([1e-6, 1e-6, 1e-6, 1e-6]),
    }
    mask = materialize_mask(child)
    expected = torch.tensor([False, False, True, True])
    assert torch.equal(mask, expected), \
        f"opacity threshold materialization wrong; mask={mask.tolist()}"


def test_child_materializes_when_brightness_threshold_crossed():
    """Brightness drift past 0.01 should fire materialization even if
    opacity is still tiny."""
    child = {
        "opacity": torch.tensor([1e-6, 1e-6, 1e-6, 1e-6]),
        "brightness": torch.tensor([0.005, 0.0099, 0.011, 0.5]),
    }
    mask = materialize_mask(child)
    expected = torch.tensor([False, False, True, True])
    assert torch.equal(mask, expected), \
        f"brightness threshold materialization wrong; mask={mask.tolist()}"


def test_either_threshold_alone_fires_materialization():
    """An OR over the two thresholds: passing either one suffices."""
    child = {
        "opacity":     torch.tensor([0.2, 1e-6, 1e-6, 0.2]),
        "brightness":  torch.tensor([1e-6, 0.05, 1e-6, 0.05]),
    }
    expected = torch.tensor([True, True, False, True])
    assert torch.equal(materialize_mask(child), expected)


def test_density_growth_under_simulated_gradient_drift():
    """Simulate 300 training iterations where children whose parents
    sit on high-loss tiles drift their opacity upward; children on
    low-loss tiles stay dormant. Materialization should fire only on
    the high-loss children -- this is the loss-adaptive density
    control claim from the paper.
    """
    torch.manual_seed(0)
    n = 64
    parent_loss_per_tile = torch.cat([
        torch.full((32,), 0.5),    # high-loss tiles
        torch.full((32,), 0.001),  # low-loss tiles
    ])
    parent_op = torch.full((n,), 0.5)
    parent_br = torch.full((n,), 0.3)
    child = make_dormant_child(parent_op, parent_br)

    # Each step: child opacity drifts proportional to parent's local loss.
    # Real training would have a gradient-based update; this synthesizes
    # the effect to verify the spawning behavior.
    drift_rate = 0.01
    for step in range(300):
        child["opacity"] = child["opacity"] + drift_rate * parent_loss_per_tile
        child["brightness"] = child["brightness"] + drift_rate * parent_loss_per_tile * 0.05

    mask = materialize_mask(child)
    # Expect: the 32 high-loss children materialize, the 32 low-loss
    # ones stay dormant.
    assert mask[:32].all(), "high-loss children should materialize"
    assert not mask[32:].any(), "low-loss children should stay dormant"


def test_total_gaussian_count_growth_pattern():
    """Simulate three rounds of spawn -> drift -> materialize. Verify
    the canvas grows monotonically and only where loss demanded it."""
    torch.manual_seed(0)
    n_initial = 16
    active = n_initial   # All initially active.
    loss_per_gaussian = torch.cat([
        torch.full((8,), 0.5),     # high-loss half
        torch.full((8,), 0.001),   # low-loss half
    ])
    drift_rate = 0.01
    materialized_total = 0
    for round_idx in range(3):
        # New dormant children for all currently-active Gaussians.
        child = {
            "opacity":    torch.full((active,), 1e-6),
            "brightness": torch.full((active,), 1e-6),
        }
        # Drift 300 steps.
        for step in range(300):
            # Use a loss vector matching the current 'active' set;
            # always 50/50 split.
            split = active // 2
            loss = torch.cat([
                torch.full((split,), 0.5),
                torch.full((active - split,), 0.001),
            ])
            child["opacity"] = child["opacity"] + drift_rate * loss
            child["brightness"] = child["brightness"] + drift_rate * loss * 0.05
        new_mat = int(materialize_mask(child).sum().item())
        materialized_total += new_mat
        active += new_mat   # Materialized children become new active Gaussians.
    # Each round should add ~half of current active (the high-loss half).
    # Cumulative materialization should be substantially more than 0
    # but bounded (no runaway).
    assert materialized_total > 0, "no Gaussians materialized -- spawner inert"
    assert materialized_total < 4 * n_initial, (
        f"runaway materialization: {materialized_total} new Gaussians from "
        f"{n_initial} initial; spawner is not gated"
    )
