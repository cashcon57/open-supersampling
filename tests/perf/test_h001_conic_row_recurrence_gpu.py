"""GPU fp32 parity checks for H001 conic row recurrence."""
from __future__ import annotations

import math

import pytest
import torch


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required"),
]


def _random_positive_definite_conics(
    n_configs: int, tile: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sx = torch.rand(n_configs, device=device, dtype=torch.float32) * 3.5 + 0.5
    sy = torch.rand(n_configs, device=device, dtype=torch.float32) * 3.5 + 0.5
    theta = torch.rand(n_configs, device=device, dtype=torch.float32) * math.pi
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    inv_sx2 = 1.0 / (sx * sx)
    inv_sy2 = 1.0 / (sy * sy)

    a = cos_t * cos_t * inv_sx2 + sin_t * sin_t * inv_sy2
    b = cos_t * sin_t * (inv_sx2 - inv_sy2)
    d = sin_t * sin_t * inv_sx2 + cos_t * cos_t * inv_sy2
    cx = torch.rand(n_configs, device=device, dtype=torch.float32) * float(tile)
    cy = torch.rand(n_configs, device=device, dtype=torch.float32) * float(tile)
    return a, b, d, cx, cy


def _naive_weights_tile(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    tile: int,
) -> torch.Tensor:
    x = torch.arange(tile, device=a.device, dtype=torch.float32).view(1, 1, tile)
    y = torch.arange(tile, device=a.device, dtype=torch.float32).view(1, tile, 1)
    dx = x - cx.view(-1, 1, 1)
    dy = y - cy.view(-1, 1, 1)
    q = (
        a.view(-1, 1, 1) * dx * dx
        + 2.0 * b.view(-1, 1, 1) * dx * dy
        + d.view(-1, 1, 1) * dy * dy
    )
    return torch.exp(-0.5 * q)


def _recurrence_weights_tile(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    tile: int,
) -> torch.Tensor:
    y = torch.arange(tile, device=a.device, dtype=torch.float32).view(1, tile)
    dx0 = -cx
    dy = y - cy.view(-1, 1)

    q0 = (
        a.view(-1, 1) * dx0.view(-1, 1) * dx0.view(-1, 1)
        + 2.0 * b.view(-1, 1) * dx0.view(-1, 1) * dy
        + d.view(-1, 1) * dy * dy
    )
    delta_q = a.view(-1, 1) * (2.0 * dx0.view(-1, 1) + 1.0) + 2.0 * b.view(-1, 1) * dy

    out = torch.empty((a.numel(), tile, tile), device=a.device, dtype=torch.float32)
    weight = torch.exp(-0.5 * q0)
    ratio = torch.exp(-0.5 * delta_q)
    ratio_step = torch.exp(-a).view(-1, 1)
    out[:, :, 0] = weight

    for col in range(1, tile):
        weight = weight * ratio
        out[:, :, col] = weight
        ratio = ratio * ratio_step

    return out


@pytest.mark.parametrize("tile", [16, 32])
def test_gpu_fp32_recurrence_matches_naive_for_tile(tile: int) -> None:
    torch.manual_seed(0xA300 + tile)
    device = torch.device("cuda")
    n_configs = 100

    a, b, d, cx, cy = _random_positive_definite_conics(n_configs, tile, device)
    naive = _naive_weights_tile(a, b, d, cx, cy, tile)
    recurrence = _recurrence_weights_tile(a, b, d, cx, cy, tile)

    max_abs_err = torch.max(torch.abs(naive - recurrence)).item()
    assert max_abs_err < 1e-4, (
        f"H001 GPU fp32 row-recurrence drift exceeds 1e-4 for {tile}x{tile} tile: "
        f"{max_abs_err:.3e}"
    )
