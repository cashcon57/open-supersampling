"""Tests for Analytic-Splatting per-pixel area integral."""

from __future__ import annotations

import math

import pytest
import torch

from oss.sr.v6.aa_analytic_splat import analytic_pixel_integral, logistic_cdf


def _analytic_pixel_integral_eigh_reference(
    delta_x: torch.Tensor,
    delta_y: torch.Tensor,
    sigma_2d_inv: torch.Tensor,
) -> torch.Tensor:
    """Previous direct-eigh implementation, retained as an anisotropic oracle."""
    eigvals_inv, eigvecs = torch.linalg.eigh(sigma_2d_inv)
    eigvals_inv = torch.clamp(eigvals_inv, min=1.0e-12)
    sigma_1 = torch.rsqrt(eigvals_inv[..., 0])
    sigma_2 = torch.rsqrt(eigvals_inv[..., 1])

    delta_x_b, delta_y_b = torch.broadcast_tensors(delta_x, delta_y)
    delta = torch.stack((delta_x_b, delta_y_b), dim=-1)
    u = (eigvecs.unsqueeze(-3) * delta.unsqueeze(-1)).sum(dim=-2)

    int_x = logistic_cdf((u[..., 0] + 0.5) / sigma_1.unsqueeze(-1)) - logistic_cdf(
        (u[..., 0] - 0.5) / sigma_1.unsqueeze(-1)
    )
    int_y = logistic_cdf((u[..., 1] + 0.5) / sigma_2.unsqueeze(-1)) - logistic_cdf(
        (u[..., 1] - 0.5) / sigma_2.unsqueeze(-1)
    )
    return 2.0 * math.pi * sigma_1.unsqueeze(-1) * sigma_2.unsqueeze(-1) * int_x * int_y


def test_logistic_cdf_at_zero() -> None:
    out = logistic_cdf(torch.tensor(0.0)).item()
    # S(0) = 1 / (1 + exp(0)) = 0.5.
    assert abs(out - 0.5) < 1.0e-6


def test_logistic_cdf_approximates_normal() -> None:
    """At x = 1, the true normal CDF is ~0.8413. The conditioned
    logistic should be within ~1e-3."""
    out = logistic_cdf(torch.tensor(1.0)).item()
    true_cdf = 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    assert abs(out - true_cdf) < 2.0e-3


def test_shape_correctness_diagonal_sigma() -> None:
    sigma_inv = torch.eye(2).expand(4, 2, 2).contiguous()
    delta_x = torch.zeros(4, 9)
    delta_y = torch.zeros(4, 9)
    out = analytic_pixel_integral(delta_x, delta_y, sigma_inv)
    assert out.shape == (4, 9)


def test_shape_correctness_broadcast_per_gaussian_grid() -> None:
    """Common case: N Gaussians, each evaluated on the same P-pixel grid."""
    sigma_inv = torch.eye(2).expand(3, 2, 2).contiguous().unsqueeze(1)  # (3, 1, 2, 2)
    delta_x = torch.zeros(3, 9)
    delta_y = torch.zeros(3, 9)
    out = analytic_pixel_integral(delta_x, delta_y, sigma_inv.squeeze(1))
    # Without broadcasting tricks: simplest shape (3, 9).
    assert out.shape == (3, 9)


def test_centre_value_sigma_one() -> None:
    """At delta=(0,0), Sigma=I, the un-normalised pixel integral is
    2*pi * (S(0.5) - S(-0.5))^2.

    With the conditioned logistic CDF,
    S(0.5) - S(-0.5) = 2 * (S(0.5) - 0.5).
    Numerically S(0.5) ~= 0.6915, so diff ~= 0.383, squared * 2pi ~= 0.921.
    But since logistic is symmetric S(x) - S(-x) = 2 S(x) - 1, fine.

    Cross-check against the true erf-based area integral:
    erf(0.5 / sqrt(2)) ~= 0.38292; 2*pi * 0.38292^2 ~= 0.921.
    """
    sigma_inv = torch.eye(2).unsqueeze(0)
    delta_x = torch.zeros(1, 1)
    delta_y = torch.zeros(1, 1)
    out = analytic_pixel_integral(delta_x, delta_y, sigma_inv).item()
    expected = 2.0 * math.pi * (math.erf(0.5 / math.sqrt(2.0)) ** 2)
    # Logistic vs true CDF -- expect within ~0.5%.
    assert abs(out - expected) / expected < 5.0e-3


def test_far_from_centre_decays() -> None:
    """Far from the splat centre, the integral decays toward zero."""
    sigma_inv = torch.eye(2).unsqueeze(0)
    delta_x = torch.tensor([[5.0]])
    delta_y = torch.tensor([[5.0]])
    out = analytic_pixel_integral(delta_x, delta_y, sigma_inv).item()
    assert out < 1.0e-4


def test_diagonal_factorises_separably() -> None:
    """For diagonal Sigma_inv, the 2D integral is the product of two
    1D integrals -- verify by checking against a direct 1D computation."""
    sigma_inv = torch.diag(torch.tensor([1.0, 1.0 / 4.0])).unsqueeze(0)
    # sigma_x = 1, sigma_y = 2.
    delta_x = torch.tensor([[0.5]])
    delta_y = torch.tensor([[0.0]])
    out = analytic_pixel_integral(delta_x, delta_y, sigma_inv).item()

    # Expected per-axis integrals using the conditioned logistic CDF.
    sx, sy = 1.0, 2.0
    int_x = logistic_cdf(torch.tensor((0.5 + 0.5) / sx)).item() - logistic_cdf(
        torch.tensor((0.5 - 0.5) / sx)
    ).item()
    int_y = logistic_cdf(torch.tensor((0.0 + 0.5) / sy)).item() - logistic_cdf(
        torch.tensor((0.0 - 0.5) / sy)
    ).item()
    expected = 2.0 * math.pi * sx * sy * int_x * int_y
    assert abs(out - expected) < 1.0e-5


def test_monte_carlo_cross_check_centre() -> None:
    """Within 1% of a 1024-sample Monte Carlo integral on a unit-Gaussian
    centre case."""
    sigma_inv = torch.eye(2).unsqueeze(0)
    delta_x = torch.zeros(1, 1)
    delta_y = torch.zeros(1, 1)
    analytical = analytic_pixel_integral(delta_x, delta_y, sigma_inv).item()

    # MC: integrate exp(-0.5 * (x^2 + y^2)) (un-normalised Gaussian, peak=1)
    # over [-0.5, 0.5]^2 (a unit pixel centred at the splat).
    torch.manual_seed(42)
    n = 1024
    pts = torch.rand(n, 2) - 0.5  # uniform in [-0.5, 0.5]^2
    densities = torch.exp(-0.5 * (pts[:, 0] ** 2 + pts[:, 1] ** 2))
    # MC estimate: mean * area (area = 1).
    mc_estimate = densities.mean().item() * 1.0

    rel_err = abs(analytical - mc_estimate) / mc_estimate
    assert rel_err < 0.01, f"analytical={analytical} mc={mc_estimate} rel_err={rel_err}"


def test_monte_carlo_cross_check_offset() -> None:
    """Within 1% of MC when the pixel centre is offset from the splat."""
    sigma_inv = torch.eye(2).unsqueeze(0)
    dx, dy = 0.7, -0.3
    delta_x = torch.tensor([[dx]])
    delta_y = torch.tensor([[dy]])
    analytical = analytic_pixel_integral(delta_x, delta_y, sigma_inv).item()

    torch.manual_seed(43)
    n = 16384  # higher count for a more stable estimate
    pts = torch.rand(n, 2) - 0.5  # uniform in [-0.5, 0.5]^2 around pixel centre
    # Splat centre is at (-dx, -dy) relative to pixel centre, so densities are
    # exp(-0.5 * ((pts.x + dx)^2 + (pts.y + dy)^2)).
    densities = torch.exp(
        -0.5 * ((pts[:, 0] + dx) ** 2 + (pts[:, 1] + dy) ** 2)
    )
    mc_estimate = densities.mean().item() * 1.0

    rel_err = abs(analytical - mc_estimate) / mc_estimate
    assert rel_err < 0.01, f"analytical={analytical} mc={mc_estimate} rel_err={rel_err}"


def test_anisotropic_rotation() -> None:
    """A diagonal Sigma rotated 45 deg should give the same on-centre
    pixel integral as the un-rotated diagonal Sigma -- the integral is
    invariant under rotation of the covariance about the splat centre
    when evaluated at delta=(0,0)."""
    diag_sigma_inv = torch.diag(torch.tensor([1.0, 1.0 / 4.0]))
    theta = torch.tensor(math.pi / 4.0)
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    rotated = R @ diag_sigma_inv @ R.T

    out_diag = analytic_pixel_integral(
        torch.zeros(1, 1), torch.zeros(1, 1), diag_sigma_inv.unsqueeze(0)
    ).item()
    out_rotated = analytic_pixel_integral(
        torch.zeros(1, 1), torch.zeros(1, 1), rotated.unsqueeze(0)
    ).item()
    # Should match within numerical precision of eigendecomposition.
    assert abs(out_diag - out_rotated) < 1.0e-5


def test_anisotropic_closed_form_matches_eigh_reference() -> None:
    """Away from repeated eigenvalues, the closed-form path matches direct eigh."""
    diag_sigma_inv = torch.diag(torch.tensor([1.0 / 9.0, 1.0 / 2.25]))
    theta = torch.tensor(0.37)
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    sigma_inv = (R @ diag_sigma_inv @ R.T).unsqueeze(0)
    delta_x = torch.tensor([[-1.25, -0.2, 0.0, 0.75, 1.5]])
    delta_y = torch.tensor([[0.5, -0.75, 0.0, 0.25, -1.0]])

    out = analytic_pixel_integral(delta_x, delta_y, sigma_inv)
    expected = _analytic_pixel_integral_eigh_reference(delta_x, delta_y, sigma_inv)

    assert torch.allclose(out, expected, atol=1.0e-5, rtol=1.0e-5)


def test_exactly_isotropic_backward_has_no_nan_gradients() -> None:
    """Repeated eigenvalues must not poison training gradients."""
    sigma_inv = torch.eye(2).unsqueeze(0).requires_grad_(True)
    delta_x = torch.tensor([[0.0, 0.25, -0.5]], requires_grad=True)
    delta_y = torch.tensor([[0.0, -0.25, 0.5]], requires_grad=True)

    out = analytic_pixel_integral(delta_x, delta_y, sigma_inv)
    out.sum().backward()

    assert torch.isfinite(out).all()
    assert sigma_inv.grad is not None
    assert delta_x.grad is not None
    assert delta_y.grad is not None
    assert torch.isfinite(sigma_inv.grad).all()
    assert torch.isfinite(delta_x.grad).all()
    assert torch.isfinite(delta_y.grad).all()


def test_zero_variance_handled() -> None:
    """Very large Sigma_inv (near-zero variance) should not produce NaN."""
    sigma_inv = (1.0e8 * torch.eye(2)).unsqueeze(0)
    out = analytic_pixel_integral(
        torch.zeros(1, 1), torch.zeros(1, 1), sigma_inv
    )
    assert torch.isfinite(out).all()


def test_invalid_sigma_inv_shape_raises() -> None:
    with pytest.raises(ValueError):
        analytic_pixel_integral(
            torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 3, 3)
        )


def test_bf16_safe() -> None:
    sigma_inv = torch.eye(2, dtype=torch.bfloat16).unsqueeze(0)
    delta_x = torch.zeros(1, 4, dtype=torch.bfloat16)
    delta_y = torch.zeros(1, 4, dtype=torch.bfloat16)
    out = analytic_pixel_integral(delta_x, delta_y, sigma_inv)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
