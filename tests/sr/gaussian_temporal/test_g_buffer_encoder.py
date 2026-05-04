"""Tests for the G-buffer encoder."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import GBufferEncoder


def test_param_count() -> None:
    enc = GBufferEncoder(in_channels=12, feat_dim=64, tile_size=16)
    n = sum(p.numel() for p in enc.parameters())
    assert n <= 120_000, f"GBufferEncoder has {n} params (budget 120_000)"


def test_forward_shape() -> None:
    enc = GBufferEncoder(in_channels=12, feat_dim=64, tile_size=16)
    x = torch.rand(2, 12, 64, 64)
    feats = enc(x)
    assert feats.shape == (2, 64, 4, 4)


def test_grad_flow() -> None:
    enc = GBufferEncoder(in_channels=12, feat_dim=64, tile_size=16)
    x = torch.rand(1, 12, 32, 32, requires_grad=True)
    feats = enc(x)
    feats.mean().backward()
    for p in enc.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
