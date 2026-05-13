"""Regression tests for the v7 canvas autograd-isolation bug.

When the trainer does

    for b in range(batch_size):
        model.reset_state(device)
        out_b = model(lr, t_query=0, spawn_at_t=0)
        loss_b = (out_b - gt_b).pow(2).mean()
        total = total + loss_b
    total.backward()

and / or runs more than one optimizer step, the canvas state's in-place
index-assignment `positions[a:b] = ...` increments the underlying storage's
version counter. Sample b's `add()` mutates the same storage that sample
a's autograd CopySlices node still references, so backward fails with
"modified by an inplace operation" or "backward through the graph a
second time."

The fix in `NDCanvasState.reset()` clones-detaches the canvas tensors so
each boundary starts on fresh storage, leaving the prior graph's tensor
references stable.

These tests would FAIL on the pre-fix version of `NDCanvasState.reset()`
that only zeroed `mask` and `n_live`.
"""
from __future__ import annotations

import torch

from oss.sr.v7.model import V7Config, V7Model


def _tiny_cfg() -> V7Config:
    return V7Config(
        in_channels=9, scale=2, feat_dim=8, latent_rank=4,
        canvas_capacity=256, backbone_kind="placeholder",
        backbone_blocks=1, enable_spawner=True,
        spawner_k_per_tile=2, spawner_tile_size=8,
    )


def _synth_input(B: int = 1, H_lr: int = 8, W_lr: int = 16) -> torch.Tensor:
    return torch.rand((B, 9, H_lr, W_lr))


def test_two_step_training_does_not_corrupt_canvas_storage():
    """Run 2 full forward + backward + optim.step() iterations. Without
    the reset() detach, step 2's backward dies with a version-counter error."""
    model = V7Model(_tiny_cfg()).train(True)
    model.allocate_canvas("cpu")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(2):
        model.reset_state("cpu")
        lr_in = _synth_input()
        gt = torch.rand((1, 3, 16, 32))
        out = model(lr_in, t_query=0.0, spawn_at_t=0.0)
        loss = (out - gt).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()


def test_batched_accumulation_does_not_corrupt_canvas_storage():
    """Trainer's per-sample reset + accumulate-loss + single-backward path.
    Without the reset() detach, sample 1's `add()` corrupts the storage
    sample 0's CopySlices references, so the final backward fails."""
    model = V7Model(_tiny_cfg()).train(True)
    model.allocate_canvas("cpu")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    optim.zero_grad()
    total = None
    for _ in range(3):
        model.reset_state("cpu")
        lr_in = _synth_input()
        gt = torch.rand((1, 3, 16, 32))
        out = model(lr_in, t_query=0.0, spawn_at_t=0.0)
        loss = (out - gt).pow(2).mean()
        total = loss if total is None else total + loss
    total.backward()
    optim.step()


def test_per_step_multi_spawn_then_backward_survives():
    """Mirrors the trainer's actual 3-spawn-per-step pattern (spawn at t=0,
    spawn at t=2, render at t=1) across two steps. The intermediate render
    relies on canvas content from both spawns, so the graph has a longer
    tail; if reset between steps doesn't detach, step 2 detonates."""
    model = V7Model(_tiny_cfg()).train(True)
    model.allocate_canvas("cpu")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(2):
        model.reset_state("cpu")
        n_lr_in = _synth_input()
        np1_lr_in = _synth_input()
        np1_gt = torch.rand((1, 3, 16, 32))
        n_half_gt = torch.rand((1, 3, 16, 32))

        _ = model(n_lr_in, t_query=0.0, spawn_at_t=0.0)
        out_np1 = model(np1_lr_in, t_query=2.0, spawn_at_t=2.0)
        out_inter = model(np1_lr_in, t_query=1.0)

        loss = (out_np1 - np1_gt).pow(2).mean() + (out_inter - n_half_gt).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()


def test_canvas_reset_breaks_grad_history():
    """After reset_state, the canvas tensors should have no grad_fn so that
    subsequent in-place writes can't collide with a freed graph."""
    model = V7Model(_tiny_cfg()).train(True)
    model.allocate_canvas("cpu")

    lr_in = _synth_input()
    _ = model(lr_in, t_query=0.0, spawn_at_t=0.0)
    # Canvas tensors should NOT have a grad_fn directly (they are leaf
    # storage), but they will have a nonzero version counter from the
    # in-place add.
    pre_version = model.canvas.positions._version
    assert pre_version > 0

    model.reset_state("cpu")
    # After reset, positions points at fresh storage (version 0) and is
    # detached.
    assert model.canvas.positions._version == 0
    assert model.canvas.positions.grad_fn is None
