import torch
import pytest

from oss.model.ord import ORD


def test_ord_forward_shapes():
    m = ORD(tier="standard").eval()
    B, H, W = 2, 64, 64
    noisy = torch.randn(B, 3, H, W)
    aux = torch.randn(B, 11, H, W)
    history = torch.randn(B, 3, H, W)
    rgb, feats = m(noisy, aux, history)
    assert rgb.shape == (B, 3, H, W)
    assert feats.shape == (B, 32, H, W)
    assert feats.dtype == torch.float16


def test_ord_param_budget_standard():
    n = sum(p.numel() for p in ORD(tier="standard").parameters())
    assert n < 1_000_000, f"standard tier exceeds 1M params: {n}"


def test_ord_param_budget_lite():
    n = sum(p.numel() for p in ORD(tier="lite").parameters())
    assert n < 200_000, f"lite tier exceeds 200K params: {n}"


def test_ord_backward():
    m = ORD(tier="standard").train()
    noisy = torch.randn(1, 3, 32, 32, requires_grad=True)
    aux = torch.randn(1, 11, 32, 32)
    history = torch.randn(1, 3, 32, 32)
    rgb, feats = m(noisy, aux, history)
    (rgb.mean() + feats.float().mean()).backward()
    assert noisy.grad is not None
