"""Tests for v6 CanvasState analytical warp."""
from __future__ import annotations

import torch

from oss.sr.v6.canvas_warp import warp_canvas
from oss.sr.v6.model import CanvasState


def _canvas(n: int = 4, *, dtype: torch.dtype = torch.float32) -> CanvasState:
    positions = torch.stack(
        [
            torch.linspace(8.0, 32.0, n, dtype=dtype),
            torch.linspace(10.0, 34.0, n, dtype=dtype),
        ],
        dim=-1,
    )
    return CanvasState(
        positions=positions,
        scales=torch.stack(
            [
                torch.linspace(1.0, 1.6, n, dtype=dtype),
                torch.linspace(0.6, 1.2, n, dtype=dtype),
            ],
            dim=-1,
        ),
        rotations=torch.linspace(0.0, 0.3, n, dtype=dtype),
        opacities=torch.linspace(0.2, 1.0, n, dtype=dtype),
        colors=torch.randn(n, 8, dtype=dtype),
        count=n,
    )


def _sigma(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(rotations)
    sin = torch.sin(rotations)
    r = torch.stack(
        [
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1),
        ],
        dim=-2,
    )
    return r @ torch.diag_embed(scales.square()) @ r.transpose(-1, -2)


def test_zero_motion_identity() -> None:
    canvas = _canvas()
    motion = torch.zeros(1, 2, 32, 32)
    out = warp_canvas(canvas, motion, output_hw=(64, 64))
    assert out.count == canvas.count
    assert torch.allclose(out.positions, canvas.positions, atol=1.0e-5)
    assert torch.allclose(out.scales, canvas.scales, atol=1.0e-5)
    assert torch.allclose(out.rotations, canvas.rotations, atol=1.0e-5)
    assert torch.allclose(out.colors, canvas.colors)


def test_constant_x_motion_shifts_positions() -> None:
    canvas = _canvas()
    motion = torch.zeros(1, 2, 32, 32)
    motion[:, 0].fill_(5.0)
    out = warp_canvas(canvas, motion, output_hw=(64, 64))
    expected = canvas.positions + torch.tensor([5.0, 0.0])
    assert torch.allclose(out.positions, expected, atol=1.0e-5)
    assert torch.allclose(out.scales, canvas.scales, atol=1.0e-5)
    assert torch.allclose(out.rotations, canvas.rotations, atol=1.0e-5)


def test_out_of_frame_gaussians_are_dropped() -> None:
    canvas = _canvas()
    motion = torch.zeros(1, 2, 64, 64)
    motion[:, 0].fill_(40.0)
    out = warp_canvas(canvas, motion, output_hw=(64, 64))
    assert out.count < canvas.count
    assert out.positions.shape[0] == out.count
    assert (out.positions[:, 0] < 64.0).all()


def test_covariance_resample_pure_scaling_jacobian() -> None:
    canvas = _canvas(n=1)
    h = w = 64
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    motion = torch.zeros(1, 2, h, w)
    motion[0, 0] = 0.5 * xx
    motion[0, 1] = 0.5 * yy

    out = warp_canvas(canvas, motion, output_hw=(h, w), sigma_recon=0.25)
    sigma_old = _sigma(canvas.scales, canvas.rotations)
    sigma_new = _sigma(out.scales, out.rotations)
    expected = (1.5 ** 2) * sigma_old + 0.25 * torch.eye(2)
    assert torch.allclose(sigma_new, expected, atol=2.0e-4)


def test_gradient_flow_from_positions_to_motion_lr() -> None:
    canvas = _canvas(n=3)
    motion = torch.zeros(1, 2, 32, 32, requires_grad=True)
    out = warp_canvas(canvas, motion, output_hw=(64, 64))
    out.positions.sum().backward()
    assert motion.grad is not None
    assert torch.isfinite(motion.grad).all()
    assert float(motion.grad.abs().sum()) > 0.0


def test_bf16_autocast_forward_finite() -> None:
    canvas = _canvas(dtype=torch.bfloat16)
    motion = torch.zeros(1, 2, 32, 32, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = warp_canvas(canvas, motion, output_hw=(64, 64))
    assert out.positions.dtype == torch.bfloat16
    assert out.scales.dtype == torch.bfloat16
    assert torch.isfinite(out.positions.float()).all()
    assert torch.isfinite(out.scales.float()).all()
