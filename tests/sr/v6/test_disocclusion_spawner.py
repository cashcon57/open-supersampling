"""Tests for v6.2 disocclusion-driven spawning."""
from __future__ import annotations

import torch

from oss.sr.v6.dgp_dictionary import DGPDictionary
from oss.sr.v6.disocclusion_spawner import DisocclusionSpawner


PIXEL_Y = 3
PIXEL_X = 5


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


def test_dgp_random_feature_usage_is_not_top1_collapsed() -> None:
    """Random features should not collapse random-init DGP usage to one prototype."""
    dgp = DGPDictionary(M=16, feat_dim=64)
    generator = torch.Generator().manual_seed(20260508)
    feat = torch.randn(10_000, 64, generator=generator)

    with torch.no_grad():
        logits = dgp.weight_head(feat)
        weights = torch.softmax(logits, dim=-1)
        usage = weights.mean(dim=0)

    assert usage.shape == (16,)
    torch.testing.assert_close(usage.sum(), torch.tensor(1.0))
    assert float(usage.max().item()) <= 0.25


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


def _single_pixel_scene() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    B, H, W = 1, 8, 8
    depth_t = torch.zeros(B, 1, H, W)
    depth_prev = torch.zeros(B, 1, H, W)
    MV = torch.zeros(B, 2, H, W)
    feat = torch.randn(B, 64, H, W)
    residual = torch.zeros(B, 1, H, W)
    depth_t[0, 0, PIXEL_Y, PIXEL_X] = 1.0
    residual[0, 0, PIXEL_Y, PIXEL_X] = 1.0
    return depth_t, depth_prev, MV, feat, residual


def test_disocclusion_zero_mv_masks_and_spawns_only_new_pixel() -> None:
    """A single new depth sample with zero MV creates exactly one birth."""
    spawner = DisocclusionSpawner(feat_dim=64)
    depth_t, depth_prev, MV, feat, residual = _single_pixel_scene()

    mask = spawner.compute_disocclusion_mask(depth_t, depth_prev, MV)
    out = spawner(depth_t, depth_prev, MV, feat, residual)

    assert mask.sum().item() == 1
    assert mask[0, 0, PIXEL_Y, PIXEL_X].item() == 1
    assert out["n_births"].item() == 1
    torch.testing.assert_close(out["xy"][0], torch.tensor([5.5, 3.5]))
    torch.testing.assert_close(out["velocity"][0], MV[0, :, PIXEL_Y, PIXEL_X])


def test_disocclusion_mv_pointing_at_pixel_reuses_visible_history() -> None:
    """Matching depth at p - MV means the pixel scrolled in from history."""
    spawner = DisocclusionSpawner(feat_dim=64)
    depth_t, depth_prev, MV, feat, residual = _single_pixel_scene()
    depth_prev[0, 0, PIXEL_Y, PIXEL_X - 1] = 1.0
    MV[0, :, PIXEL_Y, PIXEL_X - 1] = torch.tensor([1.0, 0.0])
    MV[0, :, PIXEL_Y, PIXEL_X] = torch.tensor([1.0, 0.0])

    mask = spawner.compute_disocclusion_mask(depth_t, depth_prev, MV)
    out = spawner(depth_t, depth_prev, MV, feat, residual)

    assert mask.sum().item() == 0
    assert out["n_births"].item() == 0
    assert out["xy"].shape == (0, 2)


def test_disocclusion_mv_from_offscreen_spawns_scrolled_in_pixel() -> None:
    """Offscreen p - MV samples are missing history and remain disoccluded."""
    spawner = DisocclusionSpawner(feat_dim=64)
    depth_t, depth_prev, MV, feat, residual = _single_pixel_scene()
    depth_prev[0, 0, PIXEL_Y, 0] = 1.0
    MV[0, :, PIXEL_Y, 0] = torch.tensor([-1.0, 0.0])
    MV[0, :, PIXEL_Y, PIXEL_X] = torch.tensor([6.0, 0.0])

    mask = spawner.compute_disocclusion_mask(depth_t, depth_prev, MV)
    out = spawner(depth_t, depth_prev, MV, feat, residual)

    assert mask.sum().item() == 1
    assert mask[0, 0, PIXEL_Y, PIXEL_X].item() == 1
    assert out["n_births"].item() == 1
    torch.testing.assert_close(out["xy"][0], torch.tensor([5.5, 3.5]))
    torch.testing.assert_close(out["velocity"][0], torch.tensor([6.0, 0.0]))


def test_disocclusion_mv_pointing_away_keeps_revealed_pixel_visible() -> None:
    """Depth on p + MV must not be mistaken for the current pixel's history."""
    spawner = DisocclusionSpawner(feat_dim=64)
    depth_t, depth_prev, MV, feat, residual = _single_pixel_scene()
    depth_prev[0, 0, PIXEL_Y, PIXEL_X + 1] = 1.0
    MV[0, :, PIXEL_Y, PIXEL_X] = torch.tensor([1.0, 0.0])
    MV[0, :, PIXEL_Y, PIXEL_X + 1] = torch.tensor([1.0, 0.0])

    mask = spawner.compute_disocclusion_mask(depth_t, depth_prev, MV)
    out = spawner(depth_t, depth_prev, MV, feat, residual)

    assert mask.sum().item() == 1
    assert mask[0, 0, PIXEL_Y, PIXEL_X].item() == 1
    assert out["n_births"].item() == 1
    torch.testing.assert_close(out["xy"][0], torch.tensor([5.5, 3.5]))
    torch.testing.assert_close(out["velocity"][0], MV[0, :, PIXEL_Y, PIXEL_X])


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
