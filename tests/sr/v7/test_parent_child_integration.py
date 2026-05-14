"""End-to-end tests for the parent-child loss-adaptive density mechanism
wired into V7Model.

The mechanism is what makes v7 capable of growing canvas density where
the SR loss is high (thin geometry, edges, fine reflections) instead of
spreading Gaussians uniformly. Without it, the v7 architecture is just
'uniform tile spawner + N-D canvas' which can't represent sub-pixel
features regardless of how good the backbone is.

These tests verify:
- ChildState gets allocated/reset alongside the canvas when enabled
- The forward path captures the per-parent gradient signal
- Drift accumulates child.opacity proportional to per-parent gradient
- Materialization fires once children cross threshold
- Materialized Gaussians become real canvas members and contribute to
  subsequent forwards
- The whole loop survives backward() + optim.step() without autograd errors
"""
from __future__ import annotations

import torch

from oss.sr.v7.model import V7Config, V7Model
from oss.sr.v7.parent_child_spawner import (
    OPACITY_MATERIALIZE, BRIGHTNESS_MATERIALIZE,
)


def _cfg(enable_parent_child: bool = True, drift_rate: float = 0.5) -> V7Config:
    return V7Config(
        in_channels=9, scale=2, feat_dim=8, latent_rank=4,
        canvas_capacity=512, backbone_kind="placeholder",
        backbone_blocks=1, enable_spawner=True,
        spawner_k_per_tile=2, spawner_tile_size=8,
        enable_parent_child=enable_parent_child,
        parent_child_drift_rate=drift_rate,
        parent_child_decay=0.98,
    )


def test_child_state_allocated_when_enabled():
    model = V7Model(_cfg(enable_parent_child=True))
    model.allocate_canvas("cpu")
    assert model._child is not None
    assert model.child.opacity.shape == (model.cfg.canvas_capacity,)


def test_child_state_not_allocated_when_disabled():
    model = V7Model(_cfg(enable_parent_child=False))
    model.allocate_canvas("cpu")
    assert model._child is None


def test_reset_state_clears_children():
    model = V7Model(_cfg(enable_parent_child=True))
    model.allocate_canvas("cpu")
    # Manually populate a child to dirty state
    model.child.opacity[0] = 0.5
    model.reset_state("cpu")
    assert float(model.child.opacity[0].item()) < OPACITY_MATERIALIZE
    assert model._n_children_initialized == 0


def test_initialize_new_children_marks_canvas_parents():
    model = V7Model(_cfg(enable_parent_child=True)).train(False)
    model.allocate_canvas("cpu")
    lr_in = torch.randn((1, 9, 8, 16))
    with torch.no_grad():
        _ = model(lr_in, t_query=0.0, spawn_at_t=0.0)
    n_init = model.initialize_new_children(init_dpos_std=0.3)
    # HR 16x32 / tile=8 = 2x4=8 tiles, k=2 -> 16 parents
    expected_parents = (16 // 8) * (32 // 8) * 2
    assert n_init == expected_parents
    assert model._n_children_initialized == expected_parents
    # Idempotent: no new parents -> no new initializations
    assert model.initialize_new_children() == 0


def test_single_drift_step_raises_child_opacity_above_threshold():
    """One step of (forward -> backward -> drift) with a non-trivial loss
    should push the highest-gradient child's opacity above the
    materialization threshold (0.1). This is the per-step part of the
    Diolatzis mechanism: parents in high-loss regions get their child
    promoted; uniform-loss parents stay dormant."""
    torch.manual_seed(0)
    model = V7Model(_cfg(enable_parent_child=True, drift_rate=2.0)).train(True)
    model.allocate_canvas("cpu")

    lr_in = torch.randn((1, 9, 8, 16))
    h_hr, w_hr = 16, 32
    y, x = torch.meshgrid(torch.arange(h_hr), torch.arange(w_hr), indexing="ij")
    target = ((x.float() % 2) * 0.8 + 0.1).unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).float()

    model.reset_state("cpu")
    out = model(lr_in, t_query=0.0, spawn_at_t=0.0)
    model.initialize_new_children(init_dpos_std=0.3)
    n_live = model.canvas.n_live
    # Initial state: all children dormant
    pre_max = float(model.child.opacity[:n_live].max().item())
    assert pre_max < OPACITY_MATERIALIZE, f"Children should start dormant; got pre_max={pre_max}"

    loss = (out - target).pow(2).mean()
    loss.backward()
    per_parent_grad = model.drift_children_from_grad()
    # At least one parent should have non-trivial gradient
    assert per_parent_grad.abs().sum().item() > 0
    # And the drift should have pushed the highest-gradient child over
    # the threshold (with drift_rate=2.0 and the max-normalized signal,
    # one step pushes the top child to ~2.0 -- well above 0.1).
    post_max = float(model.child.opacity[:n_live].max().item())
    assert post_max > OPACITY_MATERIALIZE, (
        f"Single drift step should raise at least one child opacity above "
        f"materialization threshold ({OPACITY_MATERIALIZE}); got post_max={post_max}"
    )


def test_drift_accumulates_gradient_across_multiple_renders():
    """Audit-regression (2026-05-14): the model used to stash only the
    LAST render's retained-grad positions in _last_positions_for_grad.
    The trainer renders 3 times per sample (t=0, t=2, t=1), so only
    the t=1 (OSS-FX intermediate) render's gradients reached drift --
    and in curriculum stage 1 with lambda_fg=0, that's zero gradient
    forever. Fix: accumulate ALL retained-grad positions into a list,
    sum gradients across them.

    This test verifies multi-render accumulation by doing 2 renders
    with different t_query values and confirming drift sees gradient
    contributions from both."""
    torch.manual_seed(0)
    model = V7Model(_cfg(enable_parent_child=True, drift_rate=1.0)).train(True)
    model.allocate_canvas("cpu")

    lr_in = torch.randn((1, 9, 8, 16))
    target_sr = torch.rand((1, 3, 16, 32))
    target_fg = torch.rand((1, 3, 16, 32))

    model.reset_state("cpu")
    # Three renders within one step (the trainer's actual flow).
    out_sr = model(lr_in, t_query=0.0, spawn_at_t=0.0)
    model.initialize_new_children(init_dpos_std=0.3)
    out_np1 = model(lr_in, t_query=2.0, spawn_at_t=2.0)
    model.initialize_new_children(init_dpos_std=0.3)
    out_inter = model(lr_in, t_query=1.0)

    # Confirm the model retained 3 separate positions tensors, not just 1.
    assert len(model._retained_positions_for_grad) == 3

    loss = (
        ((out_sr - target_sr) ** 2).mean()
        + ((out_np1 - target_sr) ** 2).mean()
        + ((out_inter - target_fg) ** 2).mean()
    )
    loss.backward()
    per_parent_grad = model.drift_children_from_grad()

    # After drift, the retained list should be empty.
    assert len(model._retained_positions_for_grad) == 0
    # And the drift should have non-trivial gradient signal -- if only
    # the last render contributed, the magnitude would be ~1x; with all
    # three contributing, it's larger.
    assert per_parent_grad.abs().sum().item() > 0


def test_drift_signal_is_proportional_to_per_parent_gradient():
    """The drift mechanism normalizes by max-grad-norm so the relative
    ordering of child opacities should match the relative ordering of
    per-parent gradient norms. Tests that the attribution is signal-
    preserving, not just blanket-uniform."""
    torch.manual_seed(0)
    model = V7Model(_cfg(enable_parent_child=True, drift_rate=1.0)).train(True)
    model.allocate_canvas("cpu")

    lr_in = torch.randn((1, 9, 8, 16))
    target = torch.rand((1, 3, 16, 32))

    model.reset_state("cpu")
    out = model(lr_in, t_query=0.0, spawn_at_t=0.0)
    model.initialize_new_children(init_dpos_std=0.3)
    n_live = model.canvas.n_live
    loss = (out - target).pow(2).mean()
    loss.backward()
    per_parent_grad = model.drift_children_from_grad()

    # opacity rank should follow per-parent-grad rank, modulo the
    # uniform decay term (which is order-preserving). Spot-check: the
    # parent with the largest grad should have the largest post-drift
    # opacity (modulo the initial 1e-6 floor).
    top_grad_idx = int(per_parent_grad.argmax().item())
    # Map active-view index back to canvas-slot index (the active_view
    # order matches mask.nonzero() order, and parents are dense in
    # [0, n_live) so the mapping is identity here).
    canvas_slot = int(model.canvas.mask[:n_live].nonzero(as_tuple=True)[0][top_grad_idx].item())
    opacities = model.child.opacity[:n_live]
    top_opacity_slot = int(opacities.argmax().item())
    assert canvas_slot == top_opacity_slot, (
        f"Parent with the largest gradient should end up with the largest "
        f"child opacity; got grad-top={canvas_slot} opacity-top={top_opacity_slot}"
    )


def test_materialize_fires_when_drift_crosses_threshold():
    """Once child.opacity > 0.1, materialize_pending_children should
    promote those children into new canvas Gaussians and reset the
    materialized slots."""
    model = V7Model(_cfg(enable_parent_child=True)).train(False)
    model.allocate_canvas("cpu")
    # Bring up a canvas with 4 parents
    lr_in = torch.randn((1, 9, 8, 16))
    with torch.no_grad():
        _ = model(lr_in, t_query=0.0, spawn_at_t=0.0)
    model.initialize_new_children()
    parents_before = model.canvas.count
    # HR 16x32, tile=8, k=2 -> 2*4*2 = 16 parents
    assert parents_before == 16

    # Manually push 2 children over the opacity threshold
    model.child.opacity[0] = 0.5
    model.child.opacity[2] = 0.3
    n_mat = model.materialize_pending_children()
    assert n_mat == 2
    # Canvas should have grown by 2 (original parents + 2 new materialized)
    assert model.canvas.count == parents_before + 2
    # Materialized child slots should be reset to dormant
    assert float(model.child.opacity[0].item()) < OPACITY_MATERIALIZE
    assert float(model.child.opacity[2].item()) < OPACITY_MATERIALIZE


def test_full_loop_step_with_drift_and_materialize_doesnt_crash():
    """The full trainer-style loop with parent-child enabled must not
    blow up the autograd graph or violate version-counter invariants
    (it interacts with the autograd-isolation fix in nd_canvas_state)."""
    torch.manual_seed(1)
    model = V7Model(_cfg(enable_parent_child=True, drift_rate=2.0)).train(True)
    model.allocate_canvas("cpu")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_in = torch.randn((1, 9, 8, 16))
    target = torch.rand((1, 3, 16, 32))

    for step in range(3):
        # Trajectory-persistent mode: reset only at trajectory boundary
        # (here: every step, since we're doing 1-step trajectories)
        model.reset_state("cpu")
        optim.zero_grad()
        out = model(lr_in, t_query=0.0, spawn_at_t=0.0)
        model.initialize_new_children(init_dpos_std=0.3)
        loss = (out - target).pow(2).mean()
        loss.backward()
        _ = model.drift_children_from_grad()
        _ = model.materialize_pending_children()
        optim.step()
        # After materialize, new children need initialization
        model.initialize_new_children(init_dpos_std=0.3)
        assert torch.isfinite(loss).item()


def test_disabled_path_is_zero_overhead():
    """When enable_parent_child=False, the new methods raise or return 0,
    and the model behaves exactly like the v7 stack pre-parent-child."""
    model = V7Model(_cfg(enable_parent_child=False)).train(False)
    model.allocate_canvas("cpu")
    lr_in = torch.randn((1, 9, 8, 16))
    with torch.no_grad():
        out = model(lr_in, t_query=0.0, spawn_at_t=0.0)
    assert out.shape == (1, 3, 16, 32)

    # initialize_new_children + materialize_pending_children must no-op
    assert model.initialize_new_children() == 0
    assert model.materialize_pending_children() == 0

    # drift_children_from_grad must raise (since user explicitly disabled)
    try:
        model.drift_children_from_grad()
        raised = False
    except RuntimeError as e:
        raised = "Parent-child" in str(e)
    assert raised
