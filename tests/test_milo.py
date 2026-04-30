from __future__ import annotations

import torch

from ors.train.milo import milo_loss


def test_milo_zero_at_identity() -> None:
    x = torch.rand(1, 3, 64, 64)
    loss = milo_loss(x, x)
    assert loss.item() < 1e-6


def test_milo_grad_flow() -> None:
    pred = torch.rand(1, 3, 64, 64, requires_grad=True)
    target = torch.rand(1, 3, 64, 64)
    loss = milo_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_milo_returns_scalar() -> None:
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    loss = milo_loss(pred, target)
    assert loss.ndim == 0
