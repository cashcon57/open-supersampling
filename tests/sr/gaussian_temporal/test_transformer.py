"""Tests for GaussianMultiFrameTransformer."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import GaussianField, GaussianMultiFrameTransformer


def _live_field(n: int) -> GaussianField:
    f = GaussianField(capacity=n)
    f.alive[:] = True
    f.mu = torch.rand(n, 2) * 16.0
    f.log_scale = torch.zeros(n, 2)
    f.rotation = torch.zeros(n)
    f.color = torch.rand(n, 3)
    f.opacity = torch.ones(n)
    return f


def test_param_budget() -> None:
    t = GaussianMultiFrameTransformer(d_model=128, n_heads=4, n_layers=4, history_len=5)
    n = sum(p.numel() for p in t.parameters())
    assert 400_000 <= n <= 600_000, f"transformer param count {n} out of budget"


def test_forward_keys_and_shapes() -> None:
    t = GaussianMultiFrameTransformer(d_model=128, n_heads=4, n_layers=2, history_len=2)
    f_curr = _live_field(8)
    history = [_live_field(8), _live_field(8)]
    feats = torch.rand(1, 128, 4, 4)
    upd = t(field_curr=f_curr, history=history, tile_features=feats)
    assert set(upd.keys()) == {"dmu", "dlog_scale", "drot", "dcolor"}
    assert upd["dmu"].shape == (8, 2)
    assert upd["dlog_scale"].shape == (8, 2)
    assert upd["drot"].shape == (8,)
    assert upd["dcolor"].shape == (8, 3)


def test_permutation_equivariance() -> None:
    torch.manual_seed(0)
    t = GaussianMultiFrameTransformer(d_model=64, n_heads=2, n_layers=2, history_len=1)
    f = _live_field(8)
    feats = torch.rand(1, 64, 2, 2)
    history = [_live_field(8)]
    upd_a = t(field_curr=f, history=history, tile_features=feats)["dmu"].detach()

    perm = torch.randperm(8)
    f2 = _live_field(8)
    f2.mu = f.mu[perm].clone()
    f2.color = f.color[perm].clone()
    upd_b = t(field_curr=f2, history=history, tile_features=feats)["dmu"].detach()
    assert torch.allclose(upd_a[perm], upd_b, atol=1e-4)
