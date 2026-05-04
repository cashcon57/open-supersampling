"""Tests for TemporalHead."""
from __future__ import annotations

import torch

from oss.sr.temporal import TemporalHead


def test_param_count_under_budget() -> None:
    head = TemporalHead()
    n = sum(p.numel() for p in head.parameters())
    assert n <= 60_000, f"TemporalHead has {n} params (budget 60_000)"


def test_forward_shape() -> None:
    head = TemporalHead()
    current_sr = torch.rand(2, 3, 16, 16)
    warped_prev = torch.rand(2, 3, 16, 16)
    disocclusion = torch.rand(2, 1, 16, 16)
    depth_hr = torch.rand(2, 1, 16, 16)
    out = head(current_sr=current_sr, warped_prev=warped_prev,
               disocclusion=disocclusion, depth_hr=depth_hr)
    assert out.shape == (2, 3, 16, 16)


def test_initial_output_close_to_current_sr() -> None:
    head = TemporalHead()
    current_sr = torch.rand(1, 3, 16, 16)
    warped_prev = torch.rand(1, 3, 16, 16)
    disocclusion = torch.rand(1, 1, 16, 16)
    depth_hr = torch.rand(1, 1, 16, 16)
    out = head(current_sr=current_sr, warped_prev=warped_prev,
               disocclusion=disocclusion, depth_hr=depth_hr)
    delta = (out - current_sr).abs().mean().item()
    assert delta < 0.1, f"Initial residual too large: {delta}"


def test_grad_flow() -> None:
    head = TemporalHead()
    current_sr = torch.rand(1, 3, 8, 8, requires_grad=True)
    warped_prev = torch.rand(1, 3, 8, 8)
    disocclusion = torch.rand(1, 1, 8, 8)
    depth_hr = torch.rand(1, 1, 8, 8)
    out = head(current_sr=current_sr, warped_prev=warped_prev,
               disocclusion=disocclusion, depth_hr=depth_hr)
    out.mean().backward()
    assert current_sr.grad is not None and torch.isfinite(current_sr.grad).all()
    for p in head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
