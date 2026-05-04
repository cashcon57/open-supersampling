"""Tests for the rasterizer wrapper around the existing OSS Gaussian renderer.

Covers Task 6 acceptance criteria:
- Output shape parity with `Rasterizer` smoke test (returns (1, F, H, W))
- Only alive Gaussians contribute (dead rows masked out)
- Gradient flows through `field.color` to `output.mean()`
- Empty alive mask returns a zero tensor
"""
from __future__ import annotations

import math

import pytest
import torch

from oss.gaussian.renderer import GaussianBatch, Rasterizer
from oss.sr.gaussian_temporal import render_field
from oss.sr.gaussian_temporal.gaussian_field import GaussianField


def _seed_field(capacity: int = 8, n_alive: int = 3, h: int = 16, w: int = 16) -> GaussianField:
    field = GaussianField(capacity=capacity, device="cpu")
    # Mark the first n_alive rows as alive.
    field.alive[:n_alive] = True
    # Spread positions across the image so they actually hit the frame.
    xs = torch.linspace(2.0, float(w - 2), steps=n_alive)
    ys = torch.linspace(2.0, float(h - 2), steps=n_alive)
    field.mu[:n_alive, 0] = xs
    field.mu[:n_alive, 1] = ys
    # log_scale -> scale = exp(log_scale); choose a small positive scale.
    field.log_scale[:n_alive] = math.log(1.5)
    field.rotation[:n_alive] = 0.0
    field.color[:n_alive] = 0.5
    field.opacity[:n_alive] = 0.8
    return field


def test_render_field_shape_matches_rasterizer_smoke() -> None:
    """Wrapper output has shape (1, F, H, W) — same F/H/W as the underlying Rasterizer."""
    h, w = 16, 16
    field = _seed_field(capacity=8, n_alive=3, h=h, w=w)
    out = render_field(field, output_hw=(h, w))
    assert out.shape == (1, 3, h, w), f"expected (1, 3, {h}, {w}), got {tuple(out.shape)}"

    # Parity smoke: directly invoke Rasterizer with the same alive set and confirm
    # the wrapper output equals rasterizer-output.unsqueeze(0).
    alive = field.alive
    feat = field.color[alive] * field.opacity[alive].unsqueeze(-1)
    batch = GaussianBatch(
        xy=field.mu[alive],
        scale=torch.exp(field.log_scale[alive]),
        rot=field.rotation[alive],
        feat=feat,
    )
    rast = Rasterizer()
    direct = rast(batch, output_hw=(h, w))
    assert direct.shape == (3, h, w)
    assert torch.allclose(out, direct.unsqueeze(0), atol=1e-5)


def test_render_field_masks_dead_rows() -> None:
    """Dead rows must not contribute. Toggling a dead row's color does not change output."""
    h, w = 16, 16
    field = _seed_field(capacity=8, n_alive=3, h=h, w=w)
    out_before = render_field(field, output_hw=(h, w))

    # Mutate a known-DEAD row (index 5). It must not affect output.
    assert not bool(field.alive[5])
    field.color[5] = torch.tensor([1.0, 0.0, 0.0])
    field.opacity[5] = 1.0
    field.mu[5] = torch.tensor([float(w) / 2.0, float(h) / 2.0])
    field.log_scale[5] = math.log(2.0)
    out_after = render_field(field, output_hw=(h, w))

    assert torch.allclose(out_before, out_after, atol=1e-6), (
        "dead rows must not contribute to the rendered output"
    )


def test_render_field_grad_flows_through_color() -> None:
    """`output.mean().backward()` produces a finite gradient on `field.color`."""
    h, w = 16, 16
    field = _seed_field(capacity=8, n_alive=3, h=h, w=w)
    field.color = field.color.clone().detach().requires_grad_(True)

    out = render_field(field, output_hw=(h, w))
    loss = out.mean()
    loss.backward()

    assert field.color.grad is not None, "no gradient on field.color"
    # Alive-row grads must be finite and non-zero (alive rows contributed).
    alive = field.alive
    alive_grad = field.color.grad[alive]
    assert torch.isfinite(alive_grad).all(), "non-finite gradient on alive rows"
    assert alive_grad.abs().sum().item() > 0.0, "alive-row gradient is identically zero"


def test_render_field_empty_alive_returns_zeros() -> None:
    """For a field with zero alive Gaussians, the wrapper returns a zero tensor."""
    h, w = 16, 16
    field = GaussianField(capacity=8, device="cpu")
    # No rows are alive (default).
    assert field.count_alive() == 0
    out = render_field(field, output_hw=(h, w))
    assert out.shape == (1, 3, h, w)
    assert torch.equal(out, torch.zeros(1, 3, h, w))
