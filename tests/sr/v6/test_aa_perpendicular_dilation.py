"""Tests for AAA-Gaussians perpendicular-ray dilation (2D)."""

from __future__ import annotations

import pytest
import torch

from oss.sr.v6.aa_perpendicular_dilation import perpendicular_dilation


def test_shape_correctness_single() -> None:
    sigma = torch.eye(2)
    view = torch.tensor([1.0, 0.0])
    out = perpendicular_dilation(sigma, view, epsilon=0.5)
    assert out.shape == (2, 2)


def test_shape_correctness_batched() -> None:
    sigma = torch.eye(2).expand(7, 2, 2).contiguous()
    view = torch.zeros(7, 2)
    view[..., 0] = 1.0
    out = perpendicular_dilation(sigma, view, epsilon=0.5)
    assert out.shape == (7, 2, 2)


def test_shape_correctness_multi_dim() -> None:
    # Leading dims (B, N) should broadcast.
    sigma = torch.eye(2).expand(3, 5, 2, 2).contiguous()
    view = torch.zeros(3, 5, 2)
    view[..., 0] = 1.0
    out = perpendicular_dilation(sigma, view, epsilon=0.25)
    assert out.shape == (3, 5, 2, 2)


def test_view_along_x_dilates_y() -> None:
    """Hand-computed: view along +x => perpendicular is +y => epsilon
    is added to Sigma[1,1] only."""
    sigma = torch.eye(2)
    view = torch.tensor([1.0, 0.0])
    out = perpendicular_dilation(sigma, view, epsilon=0.5)
    expected = torch.tensor([[1.0, 0.0], [0.0, 1.5]])
    torch.testing.assert_close(out, expected, atol=1.0e-6, rtol=1.0e-6)


def test_view_along_y_dilates_x() -> None:
    """View along +y => perpendicular is -x (or +x, same outer product)."""
    sigma = torch.eye(2)
    view = torch.tensor([0.0, 1.0])
    out = perpendicular_dilation(sigma, view, epsilon=0.5)
    expected = torch.tensor([[1.5, 0.0], [0.0, 1.0]])
    torch.testing.assert_close(out, expected, atol=1.0e-6, rtol=1.0e-6)


def test_view_diagonal() -> None:
    """View along (1,1)/sqrt(2): perpendicular is (-1,1)/sqrt(2). The
    rank-1 update epsilon * n_perp n_perp^T is then
    epsilon/2 * [[1, -1], [-1, 1]]. With sigma=I and epsilon=2 this
    gives [[2, -1], [-1, 2]]."""
    sigma = torch.eye(2)
    view = torch.tensor([1.0, 1.0])
    out = perpendicular_dilation(sigma, view, epsilon=2.0)
    expected = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
    torch.testing.assert_close(out, expected, atol=1.0e-6, rtol=1.0e-6)


def test_zero_epsilon_is_identity() -> None:
    sigma = torch.tensor([[2.0, 0.5], [0.5, 1.5]])
    view = torch.tensor([3.0, -2.0])
    out = perpendicular_dilation(sigma, view, epsilon=0.0)
    torch.testing.assert_close(out, sigma, atol=1.0e-6, rtol=1.0e-6)


def test_eigenvalue_along_perpendicular() -> None:
    """The dilated covariance's quadratic form along n_perp must
    increase by exactly epsilon over the original sigma."""
    sigma = torch.tensor([[2.0, 0.5], [0.5, 1.5]])
    view = torch.tensor([1.0, 0.5])
    eps = 0.7
    out = perpendicular_dilation(sigma, view, epsilon=eps)
    # n_perp orthogonal to view, unit length.
    v = view / torch.linalg.norm(view)
    n_perp = torch.tensor([-v[1].item(), v[0].item()])
    q_before = (n_perp @ sigma @ n_perp).item()
    q_after = (n_perp @ out @ n_perp).item()
    assert abs((q_after - q_before) - eps) < 1.0e-6


def test_eigenvalue_along_view_unchanged() -> None:
    """The quadratic form along the view direction must not change."""
    sigma = torch.tensor([[2.0, 0.5], [0.5, 1.5]])
    view = torch.tensor([1.0, 0.5])
    out = perpendicular_dilation(sigma, view, epsilon=0.7)
    v = view / torch.linalg.norm(view)
    q_before = (v @ sigma @ v).item()
    q_after = (v @ out @ v).item()
    assert abs(q_after - q_before) < 1.0e-6


def test_zero_view_direction_fallback() -> None:
    """Degenerate view falls back to dilating along x (perp to fallback +x is y)."""
    sigma = torch.eye(2)
    view = torch.zeros(2)
    out = perpendicular_dilation(sigma, view, epsilon=0.5)
    # Fallback is +x view, so dilation is along y.
    expected = torch.tensor([[1.0, 0.0], [0.0, 1.5]])
    torch.testing.assert_close(out, expected, atol=1.0e-6, rtol=1.0e-6)


def test_zero_variance_sigma() -> None:
    """Dilation of a zero covariance produces a rank-1 covariance with
    eigenvalue epsilon along n_perp and zero along v."""
    sigma = torch.zeros(2, 2)
    view = torch.tensor([1.0, 0.0])
    out = perpendicular_dilation(sigma, view, epsilon=0.5)
    expected = torch.tensor([[0.0, 0.0], [0.0, 0.5]])
    torch.testing.assert_close(out, expected, atol=1.0e-6, rtol=1.0e-6)


def test_view_normalisation_invariance() -> None:
    """The result must be invariant to the magnitude of view_direction."""
    sigma = torch.tensor([[2.0, 0.5], [0.5, 1.5]])
    out_a = perpendicular_dilation(sigma, torch.tensor([1.0, 2.0]), epsilon=0.5)
    out_b = perpendicular_dilation(sigma, torch.tensor([100.0, 200.0]), epsilon=0.5)
    torch.testing.assert_close(out_a, out_b, atol=1.0e-6, rtol=1.0e-6)


def test_invalid_sigma_shape_raises() -> None:
    with pytest.raises(ValueError):
        perpendicular_dilation(torch.zeros(3, 3), torch.tensor([1.0, 0.0]))


def test_invalid_view_shape_raises() -> None:
    with pytest.raises(ValueError):
        perpendicular_dilation(torch.eye(2), torch.tensor([1.0, 0.0, 0.0]))


def test_bf16_safe() -> None:
    sigma = torch.eye(2, dtype=torch.bfloat16)
    view = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)
    out = perpendicular_dilation(sigma, view, epsilon=0.5)
    assert out.dtype == torch.bfloat16
    # Allow generous tolerance for bf16.
    expected = torch.tensor([[1.0, 0.0], [0.0, 1.5]], dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=2.0e-2, rtol=2.0e-2)
