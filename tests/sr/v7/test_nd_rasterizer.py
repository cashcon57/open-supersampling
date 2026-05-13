"""Correctness tests for v7 N-D time-slice rasterizer.

The reference implementation lives at oss.sr.v7.nd_rasterizer. These
tests pin the math without touching any production model or training
data -- they're synthetic-only, fast, and load-bearing for the v7
architecture.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from oss.sr.v7.nd_rasterizer import (
    NDGaussian,
    time_marginal,
    render_nd_time_slice,
)


def _make_isotropic_3d_gaussian(x, y, t, sigma_xy, sigma_t, feature_value=1.0, R=1):
    mean = torch.tensor([float(x), float(y), float(t)])
    cov = torch.diag(torch.tensor([sigma_xy * sigma_xy, sigma_xy * sigma_xy, sigma_t * sigma_t]))
    feature = torch.full((R,), float(feature_value))
    opacity = torch.tensor(1.0)
    return NDGaussian(mean=mean, cov=cov, feature=feature, opacity=opacity)


# -------------------- time_marginal correctness --------------------

def test_time_marginal_diagonal_covariance_no_xy_shift():
    """For a diagonal 3D covariance (no V_xt or V_yt), conditioning on
    any t value should leave mean_xy unchanged (no t->xy coupling)."""
    means = torch.tensor([[10.0, 20.0, 5.0]])
    covs = torch.diag(torch.tensor([4.0, 4.0, 1.0])).unsqueeze(0)
    for t_query in (5.0, 4.5, 5.5, 6.0):
        m_xy, V_xy, w_t = time_marginal(means, covs, t_query=t_query)
        torch.testing.assert_close(m_xy[0], torch.tensor([10.0, 20.0]))


def test_time_marginal_coupled_xy_t_shifts_mean_correctly():
    """For a covariance with V_xt = c (x correlated with t), shifting
    t_query by delta_t should shift the conditional mean_x by
    c * delta_t / V_tt."""
    # Construct a covariance where V_xt = 0.5, V_yt = 0, V_tt = 1
    # so x is correlated with t with slope 0.5
    cov = torch.tensor([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 1.0],
    ])
    # Make it PSD: check
    assert (torch.linalg.eigvalsh(cov) > 0).all()
    means = torch.tensor([[10.0, 20.0, 0.0]])
    covs = cov.unsqueeze(0)
    delta_t = 1.5
    m_xy, _, _ = time_marginal(means, covs, t_query=delta_t)
    # Expected shift: c * delta_t / V_tt = 0.5 * 1.5 / 1.0 = 0.75
    expected_x = 10.0 + 0.5 * 1.5 / 1.0
    torch.testing.assert_close(m_xy[0, 0].item(), expected_x, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(m_xy[0, 1].item(), 20.0, rtol=1e-5, atol=1e-5)


def test_time_marginal_weight_peaks_at_mean_t():
    """The t-axis weight should peak at t_query == mean_t and fall
    off as |t_query - mean_t| grows."""
    means = torch.tensor([[10.0, 20.0, 5.0]])
    covs = torch.diag(torch.tensor([4.0, 4.0, 1.0])).unsqueeze(0)
    _, _, w_at_5 = time_marginal(means, covs, t_query=5.0)
    _, _, w_at_5_5 = time_marginal(means, covs, t_query=5.5)
    _, _, w_at_6 = time_marginal(means, covs, t_query=6.0)
    _, _, w_at_8 = time_marginal(means, covs, t_query=8.0)
    assert w_at_5.item() > w_at_5_5.item() > w_at_6.item() > w_at_8.item()


def test_time_marginal_schur_reduction_decreases_variance():
    """When x and t are correlated, conditioning on t REDUCES the
    conditional variance of x (we've learned something about x by
    observing t)."""
    cov = torch.tensor([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 1.0],
    ])
    means = torch.tensor([[0.0, 0.0, 0.0]])
    covs = cov.unsqueeze(0)
    _, V_xy_cond, _ = time_marginal(means, covs, t_query=0.0)
    # Conditional V_xx = 1 - 0.5^2 / 1.0 = 0.75 (less than marginal 1.0)
    torch.testing.assert_close(V_xy_cond[0, 0, 0].item(), 0.75, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(V_xy_cond[0, 1, 1].item(), 1.0, rtol=1e-5, atol=1e-5)


# -------------------- render_nd_time_slice correctness --------------------

def test_render_single_gaussian_centered_pixel_is_max():
    """A single isotropic 3D Gaussian rendered at its center time should
    produce a 2D Gaussian image with the brightest pixel near (x, y)."""
    g = _make_isotropic_3d_gaussian(x=15.5, y=10.5, t=0.0,
                                     sigma_xy=2.5, sigma_t=1.0,
                                     feature_value=1.0, R=1)
    out = render_nd_time_slice(
        means=g.mean.unsqueeze(0),
        covs=g.cov.unsqueeze(0),
        features=g.feature.unsqueeze(0),
        opacities=g.opacity.unsqueeze(0),
        t_query=0.0,
        image_hw=(20, 30),
    )
    # Output shape
    assert out.shape == (1, 20, 30)
    # Argmax (over the 20x30 image) should be near (10, 15) — note xy
    # convention: pixel (h, w) corresponds to y, x.
    flat_argmax = out[0].flatten().argmax().item()
    arg_y = flat_argmax // 30
    arg_x = flat_argmax % 30
    assert abs(arg_x - 15) <= 1, f"expected x~15 got {arg_x}"
    assert abs(arg_y - 10) <= 1, f"expected y~10 got {arg_y}"


def test_render_falls_off_in_time():
    """Same Gaussian, evaluated at t_query = mean_t vs t_query far away,
    should produce a much dimmer image at the far t."""
    g = _make_isotropic_3d_gaussian(x=15.5, y=10.5, t=0.0,
                                     sigma_xy=2.5, sigma_t=1.0,
                                     feature_value=1.0, R=1)
    out_at_t0 = render_nd_time_slice(
        means=g.mean.unsqueeze(0), covs=g.cov.unsqueeze(0),
        features=g.feature.unsqueeze(0), opacities=g.opacity.unsqueeze(0),
        t_query=0.0, image_hw=(20, 30),
    )
    out_at_t3 = render_nd_time_slice(
        means=g.mean.unsqueeze(0), covs=g.cov.unsqueeze(0),
        features=g.feature.unsqueeze(0), opacities=g.opacity.unsqueeze(0),
        t_query=3.0, image_hw=(20, 30),     # 3 sigma_t away
    )
    peak_t0 = out_at_t0.max().item()
    peak_t3 = out_at_t3.max().item()
    # 3-sigma fall-off in 1D: exp(-9/2) = ~0.011
    # The 2D image preserves the t-weight as a global scale on every pixel.
    assert peak_t3 < 0.1 * peak_t0, (
        f"expected dramatic falloff at 3-sigma; got peak_t0={peak_t0:.4f} "
        f"peak_t3={peak_t3:.4f}"
    )


def test_render_moving_gaussian_shifts_with_time_when_coupled():
    """A Gaussian with V_xt non-zero should appear to MOVE in screen
    space as t_query varies. This is the OSS-FX primitive in action:
    a Gaussian whose (x, t) covariance encodes motion produces
    different screen positions at different times -- without any
    external motion-vector warp."""
    # V_xt = 1.0, V_tt = 1.0  -> dx/dt = 1.0 (one pixel per unit time)
    cov = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 2.0],     # Force V_tt > V_xt for PSD: |V_xt|^2 < V_xx * V_tt
    ])
    eig = torch.linalg.eigvalsh(cov)
    assert (eig > 0).all(), f"test cov not PSD: eig={eig}"
    means = torch.tensor([[15.0, 10.0, 0.0]])
    covs = cov.unsqueeze(0)
    features = torch.ones((1, 1))
    opacities = torch.tensor([1.0])

    # Render at t_query = 0 (no shift) and t_query = 2 (expect shift +1)
    out_0 = render_nd_time_slice(
        means, covs, features, opacities,
        t_query=0.0, image_hw=(20, 40), time_falloff=False,
    )
    out_2 = render_nd_time_slice(
        means, covs, features, opacities,
        t_query=2.0, image_hw=(20, 40), time_falloff=False,
    )
    # Argmax along x should shift by ~1 pixel (V_xt/V_tt * delta_t = 1/2 * 2 = 1)
    x_at_0 = out_0[0].flatten().argmax().item() % 40
    x_at_2 = out_2[0].flatten().argmax().item() % 40
    assert (x_at_2 - x_at_0) >= 1, (
        f"expected screen-space shift of >=1 px at t_query=2; "
        f"got x_at_0={x_at_0}, x_at_2={x_at_2}"
    )


def test_render_returns_correct_shape_and_dtype():
    """Smoke test: arbitrary N, R, image dims work without shape errors."""
    torch.manual_seed(0)
    N = 8
    R = 4
    H, W = 32, 48
    means = torch.rand((N, 3)) * torch.tensor([float(W), float(H), 2.0])
    # Build PSD covariances via random Cholesky factors
    L = torch.randn((N, 3, 3))
    # Make lower-triangular with positive diagonal
    L = torch.tril(L) + torch.diag_embed(L.diagonal(dim1=-2, dim2=-1).abs() + 1.0)
    covs = L @ L.transpose(-1, -2)
    features = torch.randn((N, R))
    opacities = torch.rand((N,))
    out = render_nd_time_slice(
        means, covs, features, opacities,
        t_query=1.0, image_hw=(H, W),
    )
    assert out.shape == (R, H, W)
    assert out.dtype == features.dtype


def test_render_features_blend_when_overlapping():
    """Two overlapping Gaussians with feature values [1.0] and [4.0]
    should produce blended output in the overlap region."""
    g1 = _make_isotropic_3d_gaussian(x=10.0, y=10.0, t=0.0, sigma_xy=3.0, sigma_t=1.0, feature_value=1.0)
    g2 = _make_isotropic_3d_gaussian(x=10.0, y=10.0, t=0.0, sigma_xy=3.0, sigma_t=1.0, feature_value=4.0)
    means = torch.stack([g1.mean, g2.mean])
    covs = torch.stack([g1.cov, g2.cov])
    features = torch.stack([g1.feature, g2.feature])
    opacities = torch.stack([g1.opacity, g2.opacity])
    out = render_nd_time_slice(
        means, covs, features, opacities,
        t_query=0.0, image_hw=(20, 20),
    )
    # Pixel near (10, 10) should have brightness > each individual
    # Gaussian alone since they ADD here (alpha-compositing not yet
    # in the reference implementation -- additive feature splatting).
    peak = out.max().item()
    # Each contributes ~1 * exp(0) * weight_t, ~4 * exp(0) * weight_t
    # Together ~ 5 * weight_t. Just verify > each-alone behavior.
    out_g1_alone = render_nd_time_slice(
        g1.mean.unsqueeze(0), g1.cov.unsqueeze(0),
        g1.feature.unsqueeze(0), g1.opacity.unsqueeze(0),
        t_query=0.0, image_hw=(20, 20),
    )
    out_g2_alone = render_nd_time_slice(
        g2.mean.unsqueeze(0), g2.cov.unsqueeze(0),
        g2.feature.unsqueeze(0), g2.opacity.unsqueeze(0),
        t_query=0.0, image_hw=(20, 20),
    )
    peak_alone_max = max(out_g1_alone.max().item(), out_g2_alone.max().item())
    assert peak > peak_alone_max, (
        f"blended peak ({peak:.4f}) should exceed each alone "
        f"({peak_alone_max:.4f})"
    )
