"""Tests for ``oss.sr.v6.discriminator.UNetDiscriminator``."""
from __future__ import annotations

import pytest
import torch

from oss.sr.v6.discriminator import UNetDiscriminator


@pytest.mark.parametrize("h, w", [(64, 64), (128, 96)])
def test_output_shape_matches_input_spatial(h: int, w: int):
    """Per-pixel logits: (B, 1, H, W) matching input spatial size."""
    d = UNetDiscriminator()
    x = torch.randn(2, 3, h, w)
    out = d(x)
    assert out.shape == (2, 1, h, w)


def test_gradient_flow():
    d = UNetDiscriminator()
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    out = d(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Some param should also have grad.
    any_param_grad = any(
        (p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        for p in d.parameters()
    )
    assert any_param_grad


def test_spectral_norm_present():
    """Every conv must be wrapped with spectral norm — sanity check the
    parametrization is registered."""
    d = UNetDiscriminator()
    found = 0
    for m in d.modules():
        if isinstance(m, torch.nn.Conv2d):
            # spectral_norm registers a 'weight_u' parametrization (legacy
            # API) or a parametrization on .weight (modern API). Check both.
            has_sn = (
                hasattr(m, "weight_u")
                or (hasattr(m, "parametrizations") and "weight" in m.parametrizations)
            )
            assert has_sn, f"conv {m} missing spectral norm"
            found += 1
    assert found > 0


def test_in_channels_configurable():
    d = UNetDiscriminator(in_channels=9, base_channels=32)
    x = torch.randn(1, 9, 64, 64)
    out = d(x)
    assert out.shape == (1, 1, 64, 64)


def test_eval_mode_runs():
    d = UNetDiscriminator()
    d.train(False)
    with torch.no_grad():
        out = d(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)
