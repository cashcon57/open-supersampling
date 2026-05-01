"""Unit tests for ``temporal_consistency_loss``.

Three properties we verify:
1. Identical frames + zero motion -> loss ~= 0 (warp is identity).
2. Different frames + zero motion -> loss > 0 (no false-zero from grid_sample).
3. Gradients flow back to ``pred_t`` (so the loss can train the model).
"""
from __future__ import annotations

import torch

from oss.train.losses import temporal_consistency_loss


def test_temporal_consistency_zero_motion_identical_frames() -> None:
    pred = torch.rand(1, 3, 32, 32)
    motion = torch.zeros(1, 2, 16, 16)
    loss = temporal_consistency_loss(pred, pred, motion, scale_factor=2.0)
    assert loss.item() < 1e-3


def test_temporal_consistency_zero_motion_different_frames() -> None:
    pred_t = torch.rand(1, 3, 32, 32)
    pred_prev = torch.rand(1, 3, 32, 32)
    motion = torch.zeros(1, 2, 16, 16)
    loss = temporal_consistency_loss(pred_t, pred_prev, motion, scale_factor=2.0)
    assert loss.item() > 0


def test_temporal_consistency_grad_flow() -> None:
    pred_t = torch.rand(1, 3, 32, 32, requires_grad=True)
    pred_prev = torch.rand(1, 3, 32, 32)
    motion = torch.randn(1, 2, 16, 16) * 0.1
    loss = temporal_consistency_loss(pred_t, pred_prev, motion, scale_factor=2.0)
    loss.backward()
    assert pred_t.grad is not None
