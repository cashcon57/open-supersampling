"""Tests for the v6 active-mask-aware feature rasterizer."""
from __future__ import annotations

import torch

from oss.sr.v6.model import CanvasState
from oss.sr.v6.rasterizer import V6Rasterizer


def _canvas_state(
    count: int,
    token_dim: int,
    *,
    capacity: int | None = None,
    requires_grad: bool = False,
    dtype: torch.dtype = torch.float32,
) -> CanvasState:
    n = count if capacity is None else capacity
    positions = torch.zeros(n, 2, dtype=dtype)
    scales = torch.ones(n, 2, dtype=dtype)
    rotations = torch.zeros(n, dtype=dtype)
    opacities = torch.ones(n, dtype=dtype)
    colors = torch.zeros(n, token_dim, dtype=dtype)

    if count > 0:
        positions[:count] = torch.tensor([[5.0, 6.0]], dtype=dtype).expand(count, 2)
        colors[:count, 0] = 1.0

    if requires_grad:
        positions.requires_grad_()
        colors.requires_grad_()

    return CanvasState(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        count=count,
    )


def test_empty_canvas_returns_batched_zeros():
    token_dim = 8
    canvas = _canvas_state(count=0, token_dim=token_dim, capacity=4)
    active_mask = torch.ones(3, 4, dtype=torch.bool)

    out = V6Rasterizer(token_dim)(canvas, active_mask, output_hw=(12, 10))

    assert out.shape == (3, token_dim, 12, 10)
    assert torch.equal(out, torch.zeros_like(out))


def test_single_gaussian_renders_peak_at_known_pixel():
    token_dim = 4
    canvas = _canvas_state(count=1, token_dim=token_dim)
    rast = V6Rasterizer(token_dim)

    out = rast(canvas, torch.ones(1, dtype=torch.bool), output_hw=(16, 16))
    peak = out[0, 0].argmax()
    peak_y = int(peak // out.shape[-1])
    peak_x = int(peak % out.shape[-1])

    assert (peak_y, peak_x) == (6, 5)
    assert out[0, 0, 6, 5] > out[0, 0, 6, 4]
    assert out[0, 0, 6, 5] > out[0, 0, 5, 5]


def test_active_mask_filters_inactive_gaussians():
    token_dim = 3
    canvas = _canvas_state(count=2, token_dim=token_dim)
    canvas.positions[0] = torch.tensor([4.0, 4.0])
    canvas.positions[1] = torch.tensor([10.0, 10.0])
    canvas.colors[0, 0] = 0.0
    canvas.colors[1, 0] = 5.0

    inactive = V6Rasterizer(token_dim)(
        canvas,
        torch.tensor([True, False]),
        output_hw=(16, 16),
    )
    active = V6Rasterizer(token_dim)(
        canvas,
        torch.tensor([False, True]),
        output_hw=(16, 16),
    )

    assert inactive[0, 0, 10, 10] < 1e-3
    assert active[0, 0, 10, 10] > 4.9


def test_backward_reaches_canvas_colors_and_positions():
    token_dim = 5
    canvas = _canvas_state(count=1, token_dim=token_dim, requires_grad=True)

    out = V6Rasterizer(token_dim)(canvas, torch.ones(1, dtype=torch.bool), (16, 16))
    out.sum().backward()

    assert canvas.colors.grad is not None
    assert canvas.positions.grad is not None
    assert torch.isfinite(canvas.colors.grad).all()
    assert torch.isfinite(canvas.positions.grad).all()
    assert float(canvas.colors.grad.abs().sum()) > 0.0
    assert float(canvas.positions.grad.abs().sum()) > 0.0


def test_output_shape_for_scale_factor_variations():
    token_dim = 6
    canvas = _canvas_state(count=1, token_dim=token_dim)
    mask = torch.ones(2, 1, dtype=torch.bool)
    rast = V6Rasterizer(token_dim)

    for scale_factor in (2, 3, 4):
        h, w = 8 * scale_factor, 10 * scale_factor
        out = rast(canvas, mask, output_hw=(h, w))
        assert out.shape == (2, token_dim, h, w)


def test_bf16_autocast_forward_produces_finite_output():
    token_dim = 4
    canvas = _canvas_state(count=1, token_dim=token_dim, dtype=torch.bfloat16)
    rast = V6Rasterizer(token_dim)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = rast(canvas, torch.ones(1, dtype=torch.bool), (16, 16))

    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
