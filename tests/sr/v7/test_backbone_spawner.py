"""Tests for the v7 BackboneSpawner module."""
from __future__ import annotations

import pytest
import torch

from oss.sr.v7.backbone_spawner import BackboneSpawner
from oss.sr.v7.nd_canvas_state import cholesky_pack_to_cov


def test_backbone_spawner_output_shapes():
    spawner = BackboneSpawner(feat_dim=16, latent_rank=8, k_per_tile=4, tile_size=8)
    refined_hr = torch.randn((1, 16, 32, 48))   # 32/8=4 by 48/8=6 tiles = 24 tiles
    out = spawner(refined_hr, t=5.0)
    K = 4 * 4 * 6
    assert out["positions"].shape == (K, 3)
    assert out["cov_raw"].shape == (K, 6)
    assert out["features"].shape == (K, 8)
    assert out["opacity"].shape == (K,)


def test_backbone_spawner_positions_within_image_bounds():
    spawner = BackboneSpawner(feat_dim=16, latent_rank=4, k_per_tile=2, tile_size=8)
    refined_hr = torch.randn((1, 16, 32, 48))
    out = spawner(refined_hr, t=0.0)
    xs = out["positions"][:, 0]
    ys = out["positions"][:, 1]
    assert xs.min().item() >= 0.0
    assert xs.max().item() < 48.0
    assert ys.min().item() >= 0.0
    assert ys.max().item() < 32.0


def test_backbone_spawner_positions_cluster_around_tile_anchors():
    """For tile (i, j), all k Gaussians' xy should land within
    [j*ts, j*ts+ts) x [i*ts, i*ts+ts)."""
    spawner = BackboneSpawner(feat_dim=8, latent_rank=4, k_per_tile=3, tile_size=8)
    refined_hr = torch.randn((1, 8, 16, 16))    # 2 x 2 tiles
    out = spawner(refined_hr, t=0.0)
    positions = out["positions"]
    # Tile (i=0, j=0): x ∈ [0, 8), y ∈ [0, 8). First k_per_tile rows.
    for k in range(3):
        x, y = positions[k, 0].item(), positions[k, 1].item()
        assert 0.0 <= x < 8.0, f"k={k} x={x} should be in [0, 8)"
        assert 0.0 <= y < 8.0, f"k={k} y={y} should be in [0, 8)"
    # Tile (i=0, j=1): x ∈ [8, 16), y ∈ [0, 8). Rows 3..5.
    for k in range(3, 6):
        x, y = positions[k, 0].item(), positions[k, 1].item()
        assert 8.0 <= x < 16.0, f"k={k} x={x} should be in [8, 16)"
        assert 0.0 <= y < 8.0, f"k={k} y={y} should be in [0, 8)"


def test_backbone_spawner_covariance_is_psd():
    spawner = BackboneSpawner(feat_dim=16, latent_rank=4, k_per_tile=2, tile_size=8)
    refined_hr = torch.randn((1, 16, 16, 24))
    out = spawner(refined_hr, t=1.0)
    cov = cholesky_pack_to_cov(out["cov_raw"])
    eig = torch.linalg.eigvalsh(cov)
    assert (eig > 0).all(), "spawner-produced covariances must all be PSD"


def test_backbone_spawner_opacity_in_unit_interval():
    spawner = BackboneSpawner(feat_dim=8, latent_rank=4, k_per_tile=2, tile_size=8)
    refined_hr = torch.randn((1, 8, 16, 16))
    out = spawner(refined_hr, t=0.0)
    assert out["opacity"].min().item() >= 0.0
    assert out["opacity"].max().item() <= 1.0


def test_backbone_spawner_initial_opacity_low():
    """Init bias of -3 should produce opacities ~sigmoid(-3) = 0.047."""
    spawner = BackboneSpawner(feat_dim=8, latent_rank=4, k_per_tile=2, tile_size=8,
                              opacity_init_bias=-3.0)
    refined_hr = torch.zeros((1, 8, 16, 16))  # zero input -> output approximately bias
    out = spawner(refined_hr, t=0.0)
    mean_op = out["opacity"].mean().item()
    expected = torch.sigmoid(torch.tensor(-3.0)).item()
    assert abs(mean_op - expected) < 0.02, (
        f"expected opacity init near {expected:.3f}, got {mean_op:.3f}"
    )


def test_backbone_spawner_t_threaded_through_positions():
    spawner = BackboneSpawner(feat_dim=8, latent_rank=4, k_per_tile=1, tile_size=8)
    refined_hr = torch.zeros((1, 8, 16, 16))
    out_t0 = spawner(refined_hr, t=0.0)
    out_t5 = spawner(refined_hr, t=5.0)
    # xy should be identical at t=0 vs t=5 (deterministic from refined_hr);
    # the t channel should differ.
    torch.testing.assert_close(out_t0["positions"][:, :2], out_t5["positions"][:, :2])
    assert (out_t0["positions"][:, 2] == 0.0).all()
    assert (out_t5["positions"][:, 2] == 5.0).all()


def test_backbone_spawner_gradient_flow():
    """Backprop through spawner's outputs should update spawner params."""
    spawner = BackboneSpawner(feat_dim=8, latent_rank=4, k_per_tile=2, tile_size=8)
    refined_hr = torch.randn((1, 8, 16, 16), requires_grad=True)
    out = spawner(refined_hr, t=0.0)
    loss = out["positions"].sum() + out["features"].sum() + out["opacity"].sum()
    loss.backward()
    # Check at least the output conv got a gradient.
    assert spawner.out.weight.grad is not None
    assert spawner.out.weight.grad.abs().sum().item() > 0.0
    # And refined_hr did too.
    assert refined_hr.grad is not None
    assert refined_hr.grad.abs().sum().item() > 0.0


def test_backbone_spawner_rejects_batched_input():
    spawner = BackboneSpawner(feat_dim=8, latent_rank=4, k_per_tile=2, tile_size=8)
    refined_hr = torch.randn((2, 8, 16, 16))   # B=2
    with pytest.raises(ValueError, match="B=1"):
        spawner(refined_hr, t=0.0)
