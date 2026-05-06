"""Tests for the HAT spatial backbone (oss.sr.v6.hat)."""
from __future__ import annotations

import pytest
import torch

from oss.sr.v6.hat import HAT, hat_l, hat_small, hat_tiny


@pytest.mark.parametrize(
    "factory,expected_dim,min_params,max_params",
    [
        (hat_tiny, 60, 0.3e6, 2.0e6),
        (hat_small, 120, 3.0e6, 7.0e6),
        (hat_l, 180, 14.0e6, 20.0e6),
    ],
)
def test_factory_shapes_and_param_count(
    factory, expected_dim: int, min_params: float, max_params: float
) -> None:
    model = factory()
    n_params = sum(p.numel() for p in model.parameters())
    assert min_params <= n_params <= max_params, (
        f"{factory.__name__}: {n_params/1e6:.2f}M params not in "
        f"[{min_params/1e6:.1f}M, {max_params/1e6:.1f}M]"
    )

    x = torch.randn(2, 9, 32, 32)
    out = model(x)
    assert out.shape == (2, expected_dim, 32, 32)


def test_forward_unaligned_input_size() -> None:
    """Window-size doesn't divide H, W — internal padding must handle it."""
    model = hat_tiny()
    x = torch.randn(1, 9, 23, 31)
    out = model(x)
    # Output must match input spatial size (padding stripped on the way out).
    assert out.shape == (1, 60, 23, 31)


def test_gradient_flow() -> None:
    model = hat_tiny()
    x = torch.randn(1, 9, 32, 32, requires_grad=True)
    out = model(x)
    loss = out.pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss).item()
    # Some parameters must have non-None, finite grads.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    for g in grads:
        assert torch.isfinite(g).all().item(), "non-finite gradient encountered"


def test_custom_in_channels() -> None:
    """The 9-channel default is configurable for ablations."""
    model = HAT(in_channels=3, embed_dim=60, depth=1, num_heads=4, window_size=16)
    x = torch.randn(1, 3, 16, 16)
    out = model(x)
    assert out.shape == (1, 60, 16, 16)


def test_in_channels_mismatch_raises() -> None:
    model = hat_tiny()
    x = torch.randn(1, 6, 16, 16)  # wrong channel count
    with pytest.raises(ValueError):
        model(x)


def test_bf16_forward_runs() -> None:
    """bf16 autocast should not crash the layer (DDP / mixed-precision sanity)."""
    model = hat_tiny()
    x = torch.randn(1, 9, 32, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(x)
    assert out.shape == (1, 60, 32, 32)
