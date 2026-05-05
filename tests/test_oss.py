import pytest
import torch

from oss.model.oss import OSS


def test_oru_rgb_mode():
    oru = OSS(input_mode="rgb", scale_factor=2.0, tier="standard")
    oru.eval()
    color = torch.randn(1, 3, 32, 32)
    depth = torch.randn(1, 1, 32, 32)
    motion = torch.randn(1, 2, 32, 32)
    out = oru(color=color, depth=depth, motion=motion)
    assert out.shape == (1, 3, 64, 64)


def test_oru_rgb_aux_mode():
    oru = OSS(input_mode="rgb_aux", scale_factor=2.0, tier="standard")
    oru.eval()
    color = torch.randn(1, 3, 32, 32)
    depth = torch.randn(1, 1, 32, 32)
    motion = torch.randn(1, 2, 32, 32)
    aux = torch.randn(1, 6, 32, 32)
    out = oru(color=color, depth=depth, motion=motion, aux=aux)
    assert out.shape == (1, 3, 64, 64)


def test_oru_feature_handoff_mode():
    oru = OSS(input_mode="features", scale_factor=2.0, tier="standard")
    oru.eval()
    features = torch.randn(1, 32, 32, 32, dtype=torch.float16)
    depth = torch.randn(1, 1, 32, 32)
    motion = torch.randn(1, 2, 32, 32)
    out = oru(features=features, depth=depth, motion=motion)
    assert out.shape == (1, 3, 64, 64)


def test_oru_param_budget():
    """Standard-tier OSS pixel model param budget.

    Budget bumped from 1.5M to 3.0M after the standard tier was scaled up
    to ~2.6M params (see oss/model/oss.py header docstring: '"standard":
    [96, 144, 224, 320], # ~2.6M params'). Pico/lite tiers have their own
    smaller budgets enforced separately.
    """
    oru = OSS(input_mode="rgb", scale_factor=2.0, tier="standard")
    n = sum(p.numel() for p in oru.parameters())
    assert n < 3_000_000, f"standard tier exceeds 3M params: {n}"


def test_oru_invalid_scale_rejected():
    with pytest.raises(ValueError):
        OSS(input_mode="rgb", scale_factor=3.0, tier="standard")
