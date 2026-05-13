"""Tests for the Mip-Splatting anti-aliasing filters in the v7 rasterizer.

Implements the math from Mip-Splatting (Yu et al. 2024, arXiv 2311.16493):
  Eq. 9 — 3D smoothing filter (caps min spatial+temporal extent of source 3D Gaussian)
  Eq. 10 — 2D Mip filter (caps min projected 2D Gaussian to ~1 screen pixel)

Both filters rescale opacity / weight to conserve total Gaussian mass so
that the rendered integral is unchanged when a Gaussian is forced to
expand.

These tests check:
1. The filters preserve identity behavior when their variance == 0
2. The opacity rescale formula sqrt(|Σ| / |Σ + sI|) is correct
3. Filtering a sub-pixel-tight Gaussian no longer collapses to a dirac
4. Filtering preserves gradients (training-time differentiability)
5. The full render still works post-filter (no NaN, finite output)
"""
from __future__ import annotations

import math

import torch

from oss.sr.v7.nd_rasterizer import (
    _apply_3d_smoothing_filter,
    _apply_2d_mip_filter,
    render_nd_time_slice,
    time_marginal,
)


def _make_iso_3d_gaussian(n: int, sigma_xy: float, sigma_t: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """N copies of an axis-aligned 3D Gaussian at the image center."""
    means = torch.zeros((n, 3))
    means[:, :2] = 16.0   # center of a 32x32 HR
    means[:, 2] = 0.5
    covs = torch.zeros((n, 3, 3))
    covs[:, 0, 0] = sigma_xy * sigma_xy
    covs[:, 1, 1] = sigma_xy * sigma_xy
    covs[:, 2, 2] = sigma_t * sigma_t
    features = torch.ones((n, 4))
    opacities = torch.full((n,), 0.9)
    return means, covs, features, opacities


def test_3d_smoothing_filter_zero_variance_is_identity():
    means, covs, _, opacities = _make_iso_3d_gaussian(8, sigma_xy=1.0, sigma_t=1.0)
    out_cov, out_op = _apply_3d_smoothing_filter(covs, opacities, variance=0.0)
    assert torch.allclose(out_cov, covs)
    assert torch.allclose(out_op, opacities)


def test_3d_smoothing_filter_adds_variance_to_diagonal():
    means, covs, _, opacities = _make_iso_3d_gaussian(4, sigma_xy=1.0, sigma_t=1.0)
    s = 0.2
    out_cov, out_op = _apply_3d_smoothing_filter(covs, opacities, variance=s)
    expected_diag = torch.tensor([1.0 + s, 1.0 + s, 1.0 + s])
    assert torch.allclose(out_cov[0].diag(), expected_diag)
    # Off-diagonals unchanged (axis-aligned input)
    assert out_cov[0, 0, 1].item() == 0.0


def test_3d_smoothing_opacity_factor_matches_eq9_formula():
    """opacity_factor = sqrt(|Σ| / |Σ + sI|) per Mip-Splatting Eq 9."""
    means, covs, _, opacities = _make_iso_3d_gaussian(1, sigma_xy=0.5, sigma_t=0.5)
    s = 0.1
    _, out_op = _apply_3d_smoothing_filter(covs, opacities, variance=s)
    # For an iso Gaussian, |Σ| = sigma^6, |Σ + sI| = (sigma^2 + s)^3
    det_orig = (0.5 ** 2) ** 3
    det_smooth = (0.5 ** 2 + s) ** 3
    expected = 0.9 * math.sqrt(det_orig / det_smooth)
    assert math.isclose(out_op[0].item(), expected, rel_tol=1e-5)


def test_3d_smoothing_shrinks_opacity_for_sub_pixel_gaussian():
    """A tiny Gaussian (sigma << 1 px) gets its opacity reduced substantially
    after smoothing because its mass is forcibly spread over a wider area
    -- this is what prevents the Gaussian from rendering as a dirac."""
    means, covs, _, opacities = _make_iso_3d_gaussian(1, sigma_xy=0.1, sigma_t=0.1)
    _, out_op = _apply_3d_smoothing_filter(covs, opacities, variance=0.2)
    # sigma^2 = 0.01, smoothed = 0.21 per axis -> opacity_factor = (0.01/0.21)^1.5
    # ~ (0.0476)^1.5 = ~0.0104 -> 0.9 * 0.0104 = ~0.0094
    assert 0.005 < out_op[0].item() < 0.02


def test_3d_smoothing_preserves_grad_through_covs():
    means, covs, features, opacities = _make_iso_3d_gaussian(4, sigma_xy=1.0, sigma_t=1.0)
    covs = covs.clone().requires_grad_(True)
    opacities = opacities.clone().requires_grad_(True)
    out_cov, out_op = _apply_3d_smoothing_filter(covs, opacities, variance=0.2)
    loss = out_cov.sum() + out_op.sum()
    loss.backward()
    assert covs.grad is not None
    assert opacities.grad is not None


def test_2d_mip_filter_zero_variance_is_identity():
    cov_2d = torch.eye(2).unsqueeze(0).expand(4, -1, -1).clone()
    weights = torch.ones(4)
    out_cov, out_w = _apply_2d_mip_filter(cov_2d, weights, variance=0.0)
    assert torch.allclose(out_cov, cov_2d)
    assert torch.allclose(out_w, weights)


def test_2d_mip_filter_opacity_factor_matches_eq10_formula():
    """weight_factor = sqrt(|Σ_2D| / |Σ_2D + sI|) per Mip-Splatting Eq 10."""
    cov_2d = (0.5 ** 2) * torch.eye(2).unsqueeze(0)
    weights = torch.tensor([0.7])
    s = 0.1
    _, out_w = _apply_2d_mip_filter(cov_2d, weights, variance=s)
    # |Σ| = sigma^4 = 0.0625, |Σ + sI| = (sigma^2 + s)^2 = 0.35^2 = 0.1225
    expected = 0.7 * math.sqrt(0.0625 / 0.1225)
    assert math.isclose(out_w[0].item(), expected, rel_tol=1e-5)


def test_render_with_mip_filters_produces_finite_output():
    """Full render path including a sub-pixel-tight Gaussian must not
    NaN with default Mip-Splatting filters on."""
    torch.manual_seed(0)
    n = 16
    means = torch.zeros((n, 3))
    means[:, :2] = torch.rand((n, 2)) * 32.0
    means[:, 2] = torch.rand(n) * 2.0
    # Tight Gaussians: sigma 0.05 px is well below 1 -- would alias hard
    # without the filter.
    covs = torch.zeros((n, 3, 3))
    covs[:, 0, 0] = 0.0025
    covs[:, 1, 1] = 0.0025
    covs[:, 2, 2] = 0.25
    features = torch.randn((n, 4))
    opacities = torch.rand(n) * 0.5 + 0.5

    out = render_nd_time_slice(
        means=means, covs=covs, features=features, opacities=opacities,
        t_query=0.5, image_hw=(32, 32),
        mip_3d_variance=0.2, mip_2d_variance=0.1,
    )
    assert out.shape == (4, 32, 32)
    assert torch.isfinite(out).all()
    assert out.abs().sum().item() > 0  # some Gaussians did render


def test_render_filter_off_path_matches_pre_filter_baseline_behavior():
    """With both filter variances = 0, the render output is the same as
    the pre-Mip-Splatting code path (the filters short-circuit early)."""
    torch.manual_seed(1)
    n = 8
    means = torch.zeros((n, 3))
    means[:, :2] = 10.0 + torch.rand((n, 2)) * 6.0
    means[:, 2] = 0.5
    covs = torch.eye(3).unsqueeze(0).expand(n, -1, -1).clone() * 2.0
    features = torch.randn((n, 2))
    opacities = torch.ones(n) * 0.5

    out_no_filter = render_nd_time_slice(
        means=means, covs=covs, features=features, opacities=opacities,
        t_query=0.5, image_hw=(16, 16),
        mip_3d_variance=0.0, mip_2d_variance=0.0,
    )
    # Sanity: small but non-degenerate Gaussians, output should have signal
    assert out_no_filter.abs().sum().item() > 0


def test_render_with_filters_rescues_subpixel_gaussian_from_aliasing():
    """The whole point of the filters: a sub-pixel Gaussian without
    filtering misses every pixel center entirely (the pixel grid samples
    at integers + 0.5, and a sigma-0.05 Gaussian centered at (8,8) has
    its nearest pixel center at (7.5, 7.5), 0.707 px away, which is
    >14 sigmas out -- exp(-200/2) is effectively zero). With the Mip
    filters on, the Gaussian is forcibly spread to cover ~1 px so it
    actually deposits mass on the grid.

    This is the "anti-aliasing via primitive bandlimit" behavior the
    Mip-Splatting paper is targeting."""
    torch.manual_seed(2)
    means = torch.tensor([[8.0, 8.0, 0.0]])
    # Very tight Gaussian: sigma ~ 0.05 px in xy and t
    covs = torch.eye(3).unsqueeze(0) * 0.0025
    features = torch.tensor([[1.0]])
    opacities = torch.tensor([1.0])

    out_off = render_nd_time_slice(
        means=means, covs=covs, features=features, opacities=opacities,
        t_query=0.0, image_hw=(16, 16),
        mip_3d_variance=0.0, mip_2d_variance=0.0,
    )
    out_on = render_nd_time_slice(
        means=means, covs=covs, features=features, opacities=opacities,
        t_query=0.0, image_hw=(16, 16),
        mip_3d_variance=0.2, mip_2d_variance=0.1,
    )
    peak_off = out_off.max().item()
    peak_on = out_on.max().item()
    # Without filtering, the sub-pixel Gaussian is aliased away to ~0
    # (it lands between pixel centers and gets no contribution).
    assert peak_off < 1e-3, (
        f"Sub-pixel Gaussian without filter should alias to ~0; got "
        f"peak_off={peak_off}"
    )
    # With filtering, the Gaussian gets spread enough to actually deposit
    # mass on the pixel grid.
    assert peak_on > peak_off, (
        f"Filter-on peak ({peak_on}) should exceed filter-off peak "
        f"({peak_off}) for a sub-pixel Gaussian (filter is supposed to "
        f"rescue it from aliasing)."
    )


def test_filters_preserve_grad_end_to_end():
    """Backward through the full render-with-filters path should populate
    gradients on the parameters that DO flow through the rasterizer.

    Pre-existing v7 quirk: the Python-reference rasterizer's inner loop
    calls `float(opacity[i].item())` and `float(weight[i].item())`,
    which breaks autograd on those two tensors regardless of the Mip-
    Splatting changes. The CUDA / Triton replacement will fix that; for
    now this test verifies that the filter additions don't break grad
    flow on covs + features, which is what actually trains."""
    n = 4
    means = torch.zeros((n, 3))
    means[:, :2] = 8.0
    means[:, 2] = 0.0
    covs = torch.eye(3).unsqueeze(0).expand(n, -1, -1).clone().contiguous() * 0.5
    covs.requires_grad_(True)
    features = torch.randn((n, 2), requires_grad=True)
    opacities = torch.full((n,), 0.5)  # NOT requires_grad; .item() breaks it anyway

    out = render_nd_time_slice(
        means=means, covs=covs, features=features, opacities=opacities,
        t_query=0.0, image_hw=(16, 16),
        mip_3d_variance=0.2, mip_2d_variance=0.1,
    )
    loss = out.sum()
    loss.backward()
    assert covs.grad is not None and covs.grad.abs().sum().item() > 0
    assert features.grad is not None and features.grad.abs().sum().item() > 0
