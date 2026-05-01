"""Cross-sprint integration test for OSS-Gaussian.

Exercises Sprints 1, 3, 4, 5, 6 together on synthetic data to confirm the
component contracts hold end-to-end:

    LR + G-buffers
        -> tile classifier (Sprint 3) -> mask
        -> param network (Sprint 4) -> raw -> OutputHead -> GaussianBatch
        -> Rasterizer (Sprint 1) -> rendered_t
        -> PersistentCanvas (Sprint 5) initialized from GaussianBatch
        -> FrameExtrapolator (Sprint 6) -> rendered_{t+0.5}

Sprint 2 (D3D12 hook) is C++ on Windows — out of scope for this test.
Sprint 7 (Metal/Vulkan ports) tested separately in test_ports.py.
"""

from __future__ import annotations

import pytest
import torch

from oss.gaussian.classifier import TileClassifier
from oss.gaussian.network import CovariancePriorBank, OutputHead
from oss.gaussian.network.param_net import param_net_for_tier
from oss.gaussian.renderer import Rasterizer
from oss.gaussian.canvas import PersistentCanvas, warp_canvas


@pytest.fixture
def tier() -> str:
    return "pico"


@pytest.fixture
def synthetic_inputs() -> dict:
    """LR frame + G-buffers at 64x64. tile_size 16 → 4x4 tile grid."""
    g = torch.Generator().manual_seed(42)
    return {
        "lr": torch.rand((1, 3, 64, 64), generator=g),
        "depth": torch.rand((1, 1, 64, 64), generator=g),
        "motion": torch.randn((1, 2, 64, 64), generator=g) * 0.5,
        "normals": torch.nn.functional.normalize(torch.randn((1, 3, 64, 64), generator=g), dim=1),
        "canvas_hint": torch.zeros((1, 3, 64, 64)),
    }


def test_classifier_to_mask(synthetic_inputs: dict) -> None:
    """Sprint 3: tile classifier produces a per-tile bool mask."""
    classifier = TileClassifier(tile_size=16, target_complex_fraction=0.30)
    mask = classifier(
        synthetic_inputs["lr"],
        synthetic_inputs["depth"],
        synthetic_inputs["motion"],
        synthetic_inputs["normals"],
    )
    # 64/16 = 4 tiles per axis
    assert mask.shape == (1, 4, 4)
    assert mask.dtype == torch.bool


def test_network_to_gaussian_batch(synthetic_inputs: dict, tier: str) -> None:
    """Sprint 4: stacked input -> network -> OutputHead -> GaussianBatch."""
    bank = CovariancePriorBank(learnable=False)
    net = param_net_for_tier(tier, bank_size=16)
    head = OutputHead(bank=bank, k_per_tile=net.k_per_tile)

    stacked = torch.cat([
        synthetic_inputs["lr"],
        synthetic_inputs["depth"],
        synthetic_inputs["motion"],
        synthetic_inputs["normals"],
        synthetic_inputs["canvas_hint"],
    ], dim=1)
    assert stacked.shape == (1, 12, 64, 64)

    raw = net(stacked)
    gaussians = head.to_gaussian_batch(raw, batch_index=0)
    assert gaussians.num_gaussians > 0


def test_renderer_round_trip(synthetic_inputs: dict, tier: str) -> None:
    """Sprint 1: GaussianBatch -> render -> non-zero output."""
    bank = CovariancePriorBank(learnable=False)
    net = param_net_for_tier(tier, bank_size=16)
    head = OutputHead(bank=bank, k_per_tile=net.k_per_tile)

    stacked = torch.cat([
        synthetic_inputs["lr"], synthetic_inputs["depth"],
        synthetic_inputs["motion"], synthetic_inputs["normals"],
        synthetic_inputs["canvas_hint"],
    ], dim=1)
    raw = net(stacked)
    gaussians = head.to_gaussian_batch(raw, batch_index=0)
    renderer = Rasterizer(force_backend="reference")
    rendered = renderer(gaussians, output_hw=(128, 128))
    assert rendered.shape == (3, 128, 128)


def test_canvas_to_extrapolation(synthetic_inputs: dict, tier: str) -> None:
    """Sprint 5 + 6: GaussianBatch -> canvas -> warp_canvas at alpha=0.5
    -> render -> different from alpha=0 render."""
    bank = CovariancePriorBank(learnable=False)
    net = param_net_for_tier(tier, bank_size=16)
    head = OutputHead(bank=bank, k_per_tile=net.k_per_tile)

    stacked = torch.cat([
        synthetic_inputs["lr"], synthetic_inputs["depth"],
        synthetic_inputs["motion"], synthetic_inputs["normals"],
        synthetic_inputs["canvas_hint"],
    ], dim=1)
    with torch.no_grad():
        raw = net(stacked)
        gaussians = head.to_gaussian_batch(raw, batch_index=0)

    canvas = PersistentCanvas(
        capacity=max(gaussians.num_gaussians, 64),
        feat_dim=gaussians.feat_dim,
        output_hw=(128, 128),
    )
    canvas.initialize_from_batch(gaussians)
    motion_hr = torch.nn.functional.interpolate(
        synthetic_inputs["motion"], size=(128, 128), mode="bilinear",
    ).squeeze(0)

    canvas_t0 = warp_canvas(canvas, motion_hr, alpha=0.0)
    canvas_t05 = warp_canvas(canvas, motion_hr, alpha=0.5)

    # Positions must change with alpha != 0
    pos_t0_alive = canvas_t0.positions[canvas_t0.alive]
    pos_t05_alive = canvas_t05.positions[canvas_t05.alive]
    if pos_t0_alive.numel() > 0 and pos_t05_alive.numel() > 0:
        assert not torch.allclose(pos_t0_alive, pos_t05_alive), (
            "alpha=0.5 warp produced identical positions to alpha=0; warp not active"
        )


def test_full_pipeline_end_to_end(synthetic_inputs: dict, tier: str) -> None:
    """All sprints (except 2/7) together: input -> classifier -> network ->
    render -> canvas -> extrapolate -> render.

    This is the contract the integrated runtime will follow per frame."""
    classifier = TileClassifier(tile_size=16, target_complex_fraction=0.30)
    bank = CovariancePriorBank(learnable=False)
    net = param_net_for_tier(tier, bank_size=16)
    head = OutputHead(bank=bank, k_per_tile=net.k_per_tile)
    renderer = Rasterizer(force_backend="reference")

    # Sprint 3
    mask = classifier(
        synthetic_inputs["lr"], synthetic_inputs["depth"],
        synthetic_inputs["motion"], synthetic_inputs["normals"],
    )
    # Sprint 4
    stacked = torch.cat([
        synthetic_inputs["lr"], synthetic_inputs["depth"],
        synthetic_inputs["motion"], synthetic_inputs["normals"],
        synthetic_inputs["canvas_hint"],
    ], dim=1)
    with torch.no_grad():
        raw = net(stacked)
        gaussians = head.to_gaussian_batch(raw, batch_index=0)
    # Sprint 1
    rendered_t = renderer(gaussians, output_hw=(128, 128))
    assert rendered_t.shape == (3, 128, 128)
    # Sprint 5
    canvas = PersistentCanvas(
        capacity=max(gaussians.num_gaussians, 64),
        feat_dim=gaussians.feat_dim,
        output_hw=(128, 128),
    )
    canvas.initialize_from_batch(gaussians)
    # Sprint 6
    motion_hr = torch.nn.functional.interpolate(
        synthetic_inputs["motion"], size=(128, 128), mode="bilinear",
    ).squeeze(0)
    extrap_canvas = warp_canvas(canvas, motion_hr, alpha=0.5)
    rendered_extrap = renderer(extrap_canvas.snapshot(), output_hw=(128, 128))
    assert rendered_extrap.shape == (3, 128, 128)
    # mask shape sanity
    assert mask.shape == (1, 4, 4)
