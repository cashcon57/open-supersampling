"""Tests for DisocclusionGate."""
from __future__ import annotations

import torch

from oss.sr.temporal import DisocclusionGate


def test_module_has_three_scalar_params() -> None:
    gate = DisocclusionGate()
    params = list(gate.parameters())
    # alpha, beta, gamma — each scalar
    assert len(params) == 3
    for p in params:
        assert p.numel() == 1
        assert p.requires_grad


def test_output_shape_and_range() -> None:
    gate = DisocclusionGate()
    depth_curr = torch.rand(2, 1, 16, 16)
    depth_prev = torch.rand(2, 1, 16, 16)
    motion = torch.randn(2, 2, 8, 8) * 0.5
    mask = gate(depth_curr=depth_curr, depth_prev=depth_prev, motion_lr=motion, scale=2)
    assert mask.shape == (2, 1, 16, 16)
    assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_static_frame_low_mask() -> None:
    gate = DisocclusionGate()
    depth = torch.rand(1, 1, 16, 16)
    motion = torch.zeros(1, 2, 8, 8)
    mask = gate(depth_curr=depth, depth_prev=depth, motion_lr=motion, scale=2)
    # Default init: alpha, beta small positive, gamma large positive → low mask.
    assert mask.mean() < 0.2


def test_large_depth_disparity_high_mask() -> None:
    gate = DisocclusionGate()
    depth_curr = torch.zeros(1, 1, 16, 16)
    depth_prev = torch.ones(1, 1, 16, 16) * 5.0  # huge disparity
    motion = torch.zeros(1, 2, 8, 8)
    # Force alpha large so depth-diff dominates and the mask saturates.
    with torch.no_grad():
        gate.alpha.fill_(50.0)
        gate.gamma.fill_(0.0)
    mask = gate(depth_curr=depth_curr, depth_prev=depth_prev, motion_lr=motion, scale=2)
    assert mask.mean() > 0.9


def test_gradient_flow_to_params() -> None:
    gate = DisocclusionGate()
    depth_curr = torch.rand(1, 1, 8, 8)
    depth_prev = torch.rand(1, 1, 8, 8)
    motion = torch.randn(1, 2, 4, 4)
    mask = gate(depth_curr=depth_curr, depth_prev=depth_prev, motion_lr=motion, scale=2)
    mask.mean().backward()
    assert gate.alpha.grad is not None and torch.isfinite(gate.alpha.grad).all()
    assert gate.beta.grad is not None and torch.isfinite(gate.beta.grad).all()
    assert gate.gamma.grad is not None and torch.isfinite(gate.gamma.grad).all()
