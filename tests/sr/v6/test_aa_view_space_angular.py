"""Tests for AAA-Gaussians view-space angular bounds."""

from __future__ import annotations

import pytest
import torch

from oss.sr.v6.aa_view_space_angular import angular_bounds


def test_shape_correctness() -> None:
    sigma = torch.eye(2).expand(5, 2, 2).contiguous()
    mean = torch.zeros(5, 2)
    out = angular_bounds(sigma, mean, image_size=(64, 64))
    assert out.shape == (5, 4)


def test_isotropic_3sigma_bounds_at_origin() -> None:
    """Sigma = I, sigma_radius=3 => half-extent 3 along each axis,
    centred at (0,0)."""
    sigma = torch.eye(2).unsqueeze(0)
    mean = torch.zeros(1, 2)
    out = angular_bounds(sigma, mean, image_size=(64, 64), sigma_radius=3.0)
    expected = torch.tensor([[-3.0, -3.0, 3.0, 3.0]])
    torch.testing.assert_close(out, expected, atol=1.0e-6, rtol=1.0e-6)


def test_anisotropic_diagonal() -> None:
    """Sigma = diag(4, 9), sigma_radius=1 => half-extent 2 in x, 3 in y."""
    sigma = torch.diag(torch.tensor([4.0, 9.0])).unsqueeze(0)
    mean = torch.tensor([[10.0, 20.0]])
    out = angular_bounds(sigma, mean, image_size=(64, 64), sigma_radius=1.0)
    expected = torch.tensor([[10.0 - 2.0, 20.0 - 3.0, 10.0 + 2.0, 20.0 + 3.0]])
    torch.testing.assert_close(out, expected, atol=1.0e-6, rtol=1.0e-6)


def test_bbox_unclamped_off_screen() -> None:
    """AAA contract: bbox is NOT clamped to image_size. A Gaussian
    centred outside the screen returns a bbox that is also outside."""
    sigma = torch.eye(2).unsqueeze(0)
    mean = torch.tensor([[-50.0, -50.0]])
    out = angular_bounds(sigma, mean, image_size=(64, 64), sigma_radius=3.0)
    # Bbox should be entirely negative, not clamped to >= 0.
    assert out[0, 0].item() < 0.0
    assert out[0, 1].item() < 0.0
    assert out[0, 2].item() < 0.0
    assert out[0, 3].item() < 0.0


def test_bbox_extends_past_screen_edge() -> None:
    """Gaussian centred at the edge with non-trivial extent must
    return a bbox that extends past the edge -- no clamping."""
    sigma = torch.eye(2).unsqueeze(0)
    mean = torch.tensor([[63.0, 63.0]])
    out = angular_bounds(sigma, mean, image_size=(64, 64), sigma_radius=3.0)
    # Bbox max should exceed image_size.
    assert out[0, 2].item() > 64.0
    assert out[0, 3].item() > 64.0


def test_zero_variance() -> None:
    """Zero covariance collapses bbox to a point at the mean."""
    sigma = torch.zeros(1, 2, 2)
    mean = torch.tensor([[5.0, 7.0]])
    out = angular_bounds(sigma, mean, image_size=(64, 64), sigma_radius=3.0)
    expected = torch.tensor([[5.0, 7.0, 5.0, 7.0]])
    torch.testing.assert_close(out, expected, atol=1.0e-6, rtol=1.0e-6)


def test_degenerate_one_axis_zero() -> None:
    """One-axis-zero sigma collapses bbox to a horizontal line."""
    sigma = torch.diag(torch.tensor([4.0, 0.0])).unsqueeze(0)
    mean = torch.tensor([[10.0, 20.0]])
    out = angular_bounds(sigma, mean, image_size=(64, 64), sigma_radius=1.0)
    expected = torch.tensor([[8.0, 20.0, 12.0, 20.0]])
    torch.testing.assert_close(out, expected, atol=1.0e-6, rtol=1.0e-6)


def test_invalid_sigma_shape_raises() -> None:
    with pytest.raises(ValueError):
        angular_bounds(torch.zeros(5, 3, 3), torch.zeros(5, 2), image_size=(64, 64))


def test_invalid_n_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        angular_bounds(torch.zeros(5, 2, 2), torch.zeros(7, 2), image_size=(64, 64))


def test_invalid_sigma_radius_raises() -> None:
    with pytest.raises(ValueError):
        angular_bounds(
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2),
            image_size=(64, 64),
            sigma_radius=0.0,
        )


def test_bf16_safe() -> None:
    sigma = torch.eye(2, dtype=torch.bfloat16).unsqueeze(0)
    mean = torch.zeros(1, 2, dtype=torch.bfloat16)
    out = angular_bounds(sigma, mean, image_size=(64, 64), sigma_radius=3.0)
    assert out.dtype == torch.bfloat16


def test_negative_diag_clamped() -> None:
    """Tiny negative diagonals (bf16 round-off) are clamped to 0
    instead of producing NaN from sqrt."""
    sigma = torch.tensor([[[-1.0e-9, 0.0], [0.0, 1.0]]])
    mean = torch.zeros(1, 2)
    out = angular_bounds(sigma, mean, image_size=(64, 64), sigma_radius=3.0)
    # x-extent is 0 (sqrt(-eps) clamped to 0), y-extent is 3.
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0, 0].item(), 0.0, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(out[0, 2].item(), 0.0, atol=1.0e-6, rtol=1.0e-6)
