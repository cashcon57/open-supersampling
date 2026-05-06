"""Tests for v6 covariance resampling (GS-STVSR)."""
from __future__ import annotations

import pytest
import torch

from oss.sr.v6.covariance_resampling import (
    isotropic_sigma_recon,
    resample_covariance,
)


def test_shape_and_dtype_unbatched():
    sigma_t = torch.eye(2)
    j = torch.eye(2) * 2.0
    out = resample_covariance(sigma_t, j, 0.25)
    assert out.shape == (2, 2)
    assert out.dtype == torch.float32


def test_shape_and_dtype_batched():
    sigma_t = torch.eye(2).expand(4, 7, 2, 2).contiguous()
    j = (torch.eye(2) * 2.0).expand(4, 7, 2, 2).contiguous()
    out = resample_covariance(sigma_t, j, 0.25)
    assert out.shape == (4, 7, 2, 2)


def test_canonical_handcomputed_example():
    """Sigma_t = I, J = 2I, Sigma_recon = 0.25 I  =>  4I + 0.25 I."""
    sigma_t = torch.eye(2, dtype=torch.float64)
    j = torch.eye(2, dtype=torch.float64) * 2.0
    out = resample_covariance(sigma_t, j, 0.25)
    expected = torch.eye(2, dtype=torch.float64) * (4.0 + 0.25)
    assert torch.allclose(out, expected, atol=1e-9)


def test_recon_tensor_input():
    sigma_t = torch.eye(2)
    j = torch.eye(2)
    recon = torch.tensor([[0.1, 0.0], [0.0, 0.3]])
    out = resample_covariance(sigma_t, j, recon)
    expected = torch.tensor([[1.1, 0.0], [0.0, 1.3]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_anisotropic_jacobian():
    """J = diag(2, 3), Sigma_t = I, no recon-floor effect."""
    sigma_t = torch.eye(2, dtype=torch.float64)
    j = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64))
    out = resample_covariance(sigma_t, j, 0.0)
    expected = torch.diag(torch.tensor([4.0, 9.0], dtype=torch.float64))
    assert torch.allclose(out, expected, atol=1e-6)


def test_singular_jacobian_floored():
    """A degenerate (rank-1) Jacobian should not produce a singular output."""
    sigma_t = torch.eye(2, dtype=torch.float64)
    j = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    out = resample_covariance(sigma_t, j, 0.0)
    # Eigenvalues should be >= floor; det > 0.
    eigvals = torch.linalg.eigvalsh(out)
    assert (eigvals > 0).all()
    assert torch.linalg.det(out) > 0


def test_isotropic_sigma_recon_value():
    s = isotropic_sigma_recon(0.5)
    expected = torch.eye(2) * (0.25**2)
    assert torch.allclose(s, expected)


def test_isotropic_sigma_recon_rejects_nonpositive():
    with pytest.raises(ValueError):
        isotropic_sigma_recon(0.0)
    with pytest.raises(ValueError):
        isotropic_sigma_recon(-1.0)


def test_bad_shapes_rejected():
    with pytest.raises(ValueError):
        resample_covariance(torch.zeros(3, 3), torch.eye(2), 0.0)
    with pytest.raises(ValueError):
        resample_covariance(torch.eye(2), torch.zeros(3, 3), 0.0)


def test_bf16_safe():
    sigma_t = (torch.eye(2) * 1.5).to(torch.bfloat16)
    j = (torch.eye(2) * 2.0).to(torch.bfloat16)
    out = resample_covariance(sigma_t, j, 0.25)
    assert out.dtype == torch.bfloat16
    # 4 * 1.5 + 0.25 = 6.25 on each diag.
    assert torch.allclose(out.float(), torch.eye(2) * 6.25, atol=0.1)


def test_gradient_flow():
    sigma_t = torch.eye(2, requires_grad=True)
    j = (torch.eye(2) * 2.0).requires_grad_(True)
    out = resample_covariance(sigma_t, j, 0.25)
    loss = out.sum()
    loss.backward()
    assert sigma_t.grad is not None
    assert j.grad is not None
    assert torch.isfinite(sigma_t.grad).all()
    assert torch.isfinite(j.grad).all()


def test_isotropic_sigma_recon_dtype_device():
    s = isotropic_sigma_recon(1.0, device="cpu", dtype=torch.float64)
    assert s.dtype == torch.float64
    assert s.shape == (2, 2)
