"""Tests for pixel-Gaussian fusion (oss.sr.v6.cross_attention)."""
from __future__ import annotations

import pytest
import torch

from oss.sr.v6.cross_attention import PixelGaussianFusion


@pytest.mark.parametrize(
    "feat_dim,num_heads",
    [
        (60, 4),    # Tiny / Pico
        (120, 6),   # Small / Standard
        (180, 6),   # L / Heavy
    ],
)
def test_forward_shape(feat_dim: int, num_heads: int) -> None:
    fusion = PixelGaussianFusion(
        feat_dim=feat_dim, token_dim=64, num_heads=num_heads, window_size=16
    )
    x = torch.randn(2, feat_dim, 32, 32)
    g = torch.randn(2, 7, 64)
    out = fusion(x, g)
    assert out.shape == (2, feat_dim, 32, 32)


def test_empty_canvas_is_identity() -> None:
    """K = 0 (no Gaussians yet) must short-circuit to identity."""
    fusion = PixelGaussianFusion(feat_dim=180, token_dim=64, num_heads=6, window_size=16)
    x = torch.randn(2, 180, 32, 32)
    g = torch.zeros(2, 0, 64)
    out = fusion(x, g)
    assert out.shape == x.shape
    assert torch.equal(out, x)


def test_gradient_flow() -> None:
    fusion = PixelGaussianFusion(feat_dim=60, token_dim=32, num_heads=4, window_size=16)
    x = torch.randn(1, 60, 32, 32, requires_grad=True)
    g = torch.randn(1, 5, 32, requires_grad=True)
    out = fusion(x, g)
    loss = out.pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss).item()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    assert g.grad is not None and torch.isfinite(g.grad).all().item()
    grads = [p.grad for p in fusion.parameters() if p.grad is not None]
    assert len(grads) > 0
    for gg in grads:
        assert torch.isfinite(gg).all().item()


def test_unaligned_spatial_size() -> None:
    """H, W not divisible by window_size — internal padding handles it."""
    fusion = PixelGaussianFusion(feat_dim=60, token_dim=32, num_heads=4, window_size=16)
    x = torch.randn(1, 60, 23, 31)
    g = torch.randn(1, 4, 32)
    out = fusion(x, g)
    assert out.shape == (1, 60, 23, 31)


def test_variable_k_per_call() -> None:
    """K varies frame to frame; layer must accept any K >= 0."""
    fusion = PixelGaussianFusion(feat_dim=60, token_dim=32, num_heads=4, window_size=16)
    x = torch.randn(1, 60, 16, 16)
    for k in (1, 3, 17, 100):
        out = fusion(x, torch.randn(1, k, 32))
        assert out.shape == (1, 60, 16, 16)


def test_input_validation() -> None:
    fusion = PixelGaussianFusion(feat_dim=60, token_dim=32, num_heads=4, window_size=16)
    with pytest.raises(ValueError):
        fusion(torch.randn(1, 30, 16, 16), torch.randn(1, 4, 32))  # wrong feat_dim
    with pytest.raises(ValueError):
        fusion(torch.randn(1, 60, 16, 16), torch.randn(1, 4, 16))  # wrong token_dim
    with pytest.raises(ValueError):
        fusion(torch.randn(2, 60, 16, 16), torch.randn(1, 4, 32))  # B mismatch


def test_bf16_forward_runs() -> None:
    fusion = PixelGaussianFusion(feat_dim=60, token_dim=32, num_heads=4, window_size=16)
    x = torch.randn(1, 60, 32, 32)
    g = torch.randn(1, 5, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = fusion(x, g)
    assert out.shape == (1, 60, 32, 32)
