"""Tests for v6 GRAPE-style Gaussian spawner."""
from __future__ import annotations

import pickle

import pytest
import torch

from oss.sr.v6.gaussian_spawner import GaussianSpawner
from oss.sr.v6.model import V6Config


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("scale", [2, 3])
@pytest.mark.parametrize("tile_size_lr", [4, 8])
def test_output_shape_correctness(batch: int, scale: int, tile_size_lr: int) -> None:
    spawner = GaussianSpawner(
        feat_dim=12,
        token_dim=16,
        scale=scale,
        tile_size_lr=tile_size_lr,
    )
    features = torch.randn(batch, 12, 16, 24)
    out = spawner(features)
    k = (16 // tile_size_lr) * (24 // tile_size_lr)

    assert out.positions.shape == (batch, k, 2)
    assert out.scales.shape == (batch, k, 2)
    assert out.rotations.shape == (batch, k)
    assert out.colors.shape == (batch, k, 16)
    assert out.confidence.shape == (batch, k)
    assert out.opacities.shape == out.confidence.shape
    assert out.count == k


def test_uses_v6_config_defaults() -> None:
    cfg = V6Config(scale=3, token_dim=32, tile_size_lr=4)
    spawner = GaussianSpawner(feat_dim=8, config=cfg)
    out = spawner(torch.randn(1, 8, 8, 12))
    assert out.colors.shape == (1, 6, 32)


def test_softplus_scales_are_positive() -> None:
    spawner = GaussianSpawner(feat_dim=8, token_dim=4, scale=2, tile_size_lr=4)
    out = spawner(torch.randn(2, 8, 8, 8))
    assert (out.scales > 0).all()


def test_initial_scales_are_half_hr_tile() -> None:
    spawner = GaussianSpawner(feat_dim=8, token_dim=4, scale=2, tile_size_lr=4)
    out = spawner(torch.randn(1, 8, 8, 8))
    expected = torch.full_like(out.scales, 4.0)
    torch.testing.assert_close(out.scales, expected, atol=1.0e-5, rtol=1.0e-5)


def test_rotations_bounded_to_pi() -> None:
    spawner = GaussianSpawner(feat_dim=8, token_dim=4, scale=2, tile_size_lr=4)
    out = spawner(torch.randn(2, 8, 8, 8))
    assert (out.rotations >= -torch.pi).all()
    assert (out.rotations <= torch.pi).all()


def test_confidence_in_unit_range() -> None:
    spawner = GaussianSpawner(feat_dim=8, token_dim=4, scale=2, tile_size_lr=4)
    out = spawner(torch.randn(2, 8, 8, 8))
    assert (out.confidence >= 0).all()
    assert (out.confidence <= 1).all()


def test_positions_are_hr_tile_centers_at_initialization() -> None:
    spawner = GaussianSpawner(feat_dim=8, token_dim=4, scale=2, tile_size_lr=4)
    out = spawner(torch.randn(1, 8, 8, 8))
    expected = torch.tensor(
        [[[4.0, 4.0], [12.0, 4.0], [4.0, 12.0], [12.0, 12.0]]]
    )
    torch.testing.assert_close(out.positions, expected)


def test_gradient_flow_from_colors_to_conv_weight() -> None:
    spawner = GaussianSpawner(feat_dim=8, token_dim=4, scale=2, tile_size_lr=4)
    features = torch.randn(2, 8, 8, 8)
    out = spawner(features)
    loss = out.colors.sum()
    loss.backward()
    assert spawner.conv.weight.grad is not None
    assert torch.isfinite(spawner.conv.weight.grad).all()
    assert float(spawner.conv.weight.grad.abs().sum()) > 0.0


def test_gradient_flow_from_positions_to_conv_weight() -> None:
    spawner = GaussianSpawner(feat_dim=8, token_dim=4, scale=2, tile_size_lr=4)
    features = torch.randn(2, 8, 8, 8)
    out = spawner(features)
    loss = out.positions.sum()
    loss.backward()
    assert spawner.conv.weight.grad is not None
    assert torch.isfinite(spawner.conv.weight.grad).all()
    assert float(spawner.conv.weight.grad.abs().sum()) > 0.0


def test_bf16_autocast_forward_produces_finite_output() -> None:
    spawner = GaussianSpawner(feat_dim=8, token_dim=4, scale=2, tile_size_lr=4)
    features = torch.randn(2, 8, 8, 8)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = spawner(features)
    for tensor in (
        out.positions,
        out.scales,
        out.rotations,
        out.colors,
        out.confidence,
    ):
        assert torch.isfinite(tensor).all()
        assert tensor.dtype == torch.bfloat16


def test_pickle_round_trip_preserves_behavior() -> None:
    spawner = GaussianSpawner(feat_dim=8, token_dim=4, scale=2, tile_size_lr=4)
    features = torch.randn(2, 8, 8, 8)
    before = spawner(features)

    restored = pickle.loads(pickle.dumps(spawner))
    after = restored(features)

    torch.testing.assert_close(after.positions, before.positions)
    torch.testing.assert_close(after.scales, before.scales)
    torch.testing.assert_close(after.rotations, before.rotations)
    torch.testing.assert_close(after.colors, before.colors)
    torch.testing.assert_close(after.confidence, before.confidence)


def test_rejects_non_divisible_feature_grid() -> None:
    spawner = GaussianSpawner(feat_dim=8, token_dim=4, scale=2, tile_size_lr=4)
    with pytest.raises(ValueError, match="divisible"):
        spawner(torch.randn(1, 8, 10, 8))
