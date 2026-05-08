"""Tests for v6.2 disocclusion-driven spawning."""
from __future__ import annotations

import torch

from oss.sr.v6.dgp_dictionary import DGPDictionary
from oss.sr.v6.disocclusion_spawner import DisocclusionSpawner


def test_dgp_outputs_positive_definite() -> None:
    """DGP outputs must satisfy a > 0, d > 0, ad - b^2 > 0."""
    dgp = DGPDictionary(M=16, feat_dim=64)
    feat = torch.randn(100, 64)
    conic, scale = dgp(feat)
    a, b, d = conic.unbind(-1)
    assert (a > 0).all(), "a must be > 0"
    assert (d > 0).all(), "d must be > 0"
    assert (a * d - b * b > 0).all(), "ad - b^2 must be > 0"
    assert (scale >= dgp.scale_min).all()
    assert (scale <= dgp.scale_max).all()


def test_dgp_prototypes_are_positive_definite() -> None:
    """Prototype conics must be positive-definite before softmax mixing."""
    dgp = DGPDictionary(M=16, feat_dim=64)
    a, b, d = dgp.prototypes_abd.unbind(-1)
    assert (a > 0).all()
    assert (d > 0).all()
    assert (a * d - b * b > 0).all()


def test_disocclusion_spawn_at_pixel_center() -> None:
    """Spawn xy must be at exact integer + 0.5 pixel centers."""
    spawner = DisocclusionSpawner(feat_dim=64)
    B, H, W = 1, 8, 8
    depth_t = torch.zeros(B, 1, H, W)
    depth_prev = torch.zeros(B, 1, H, W)
    depth_t[0, 0, 3, 5] = 1.0
    MV = torch.zeros(B, 2, H, W)
    feat = torch.randn(B, 64, H, W)
    residual = torch.ones(B, 1, H, W) * 0.5

    out = spawner(depth_t, depth_prev, MV, feat, residual)

    assert out["xy"].shape[0] == 1
    torch.testing.assert_close(out["xy"][0], torch.tensor([5.5, 3.5]))
    fractional = out["xy"] - out["xy"].floor()
    torch.testing.assert_close(fractional, torch.full_like(fractional, 0.5))


def test_disocclusion_birth_cap() -> None:
    """max_births_per_frame must be enforced per batch element."""
    spawner = DisocclusionSpawner(feat_dim=64, max_births_per_frame=10)
    B, H, W = 1, 32, 32
    depth_t = torch.ones(B, 1, H, W)
    depth_prev = torch.zeros(B, 1, H, W)
    MV = torch.zeros(B, 2, H, W)
    feat = torch.randn(B, 64, H, W)
    residual = torch.ones(B, 1, H, W)

    out = spawner(depth_t, depth_prev, MV, feat, residual)

    assert out["n_births"].item() <= 10
