"""Analytical Gaussian warp tests."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import GaussianField, warp_field


def _make_field(n: int) -> GaussianField:
    f = GaussianField(capacity=n)
    f.alive[:] = True
    f.mu = torch.tensor([[float(i + 1), float(i + 1)] for i in range(n)])
    f.log_scale = torch.zeros(n, 2)
    f.rotation = torch.zeros(n)
    f.color = torch.rand(n, 3)
    f.opacity = torch.ones(n)
    return f


def test_identity_flow_unchanged() -> None:
    f = _make_field(4)
    motion = torch.zeros(2, 16, 16)
    g = warp_field(f, motion, hw=(16, 16))
    assert torch.allclose(g.mu, f.mu, atol=1e-5)
    assert torch.allclose(g.log_scale, f.log_scale, atol=1e-5)


def test_translation_preserves_covariance() -> None:
    f = _make_field(4)
    f.log_scale = torch.tensor([[0.5, 0.2], [0.3, 0.4], [0.1, 0.1], [0.6, 0.6]])
    motion = torch.zeros(2, 16, 16)
    motion[0] = 1.0   # constant +1 px in x
    motion[1] = -2.0  # constant -2 px in y
    g = warp_field(f, motion, hw=(16, 16))
    # mu shifted; log_scale identical (J = I).
    assert torch.allclose(g.mu, f.mu + torch.tensor([1.0, -2.0]), atol=1e-5)
    assert torch.allclose(g.log_scale, f.log_scale, atol=1e-4)


def test_jacobian_warp_matches_numerical() -> None:
    """Smooth flow (linear gradient in x) → analytic Σ' should match numerical."""
    h, w = 32, 32
    motion = torch.zeros(2, h, w)
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")
    motion[0] = 0.1 * xx   # u(x, y) = 0.1 x  → du/dx = 0.1
    motion[1] = 0.05 * yy  # v(x, y) = 0.05 y → dv/dy = 0.05
    f = _make_field(1)
    f.mu = torch.tensor([[16.0, 16.0]])
    f.log_scale = torch.tensor([[0.0, 0.0]])
    g = warp_field(f, motion, hw=(h, w))
    # J = diag(1.1, 1.05); axis-aligned scales become 1.1 and 1.05.
    expected_log = torch.tensor([[torch.log(torch.tensor(1.1)).item(),
                                  torch.log(torch.tensor(1.05)).item()]])
    assert torch.allclose(g.log_scale, expected_log, atol=5e-3)


def test_out_of_frame_marked_dead() -> None:
    f = _make_field(2)
    f.mu = torch.tensor([[1.0, 1.0], [30.0, 30.0]])
    motion = torch.zeros(2, 16, 16)
    motion[0] = 100.0  # huge x flow
    g = warp_field(f, motion, hw=(16, 16))
    assert g.alive[0].item() is False
