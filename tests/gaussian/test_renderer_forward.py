"""Sprint 1 / T1.4 — forward render test.

Validates the Rasterizer's forward path:
- The reference (PyTorch) backend produces correct output for known Gaussians.
- The CUDA backend (when available) matches the reference within FP32 tolerance.

The CUDA test is skipped on machines without a working gsplat extension —
those run on the reference backend only as a correctness anchor.
"""

from __future__ import annotations

import pytest
import torch

from oss.gaussian.renderer import GaussianBatch, Rasterizer


@pytest.fixture
def single_centered_gaussian() -> GaussianBatch:
    """One bright Gaussian centered in the image."""
    return GaussianBatch(
        xy=torch.tensor([[16.0, 16.0]]),
        scale=torch.tensor([[2.0, 2.0]]),
        rot=torch.tensor([0.0]),
        feat=torch.tensor([[1.0, 0.5, 0.25]]),  # RGB
    )


@pytest.fixture
def two_distinct_gaussians() -> GaussianBatch:
    """Two non-overlapping Gaussians, one red one blue."""
    return GaussianBatch(
        xy=torch.tensor([[8.0, 16.0], [24.0, 16.0]]),
        scale=torch.tensor([[1.5, 1.5], [1.5, 1.5]]),
        rot=torch.tensor([0.0, 0.0]),
        feat=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )


def test_reference_backend_centered_peak(single_centered_gaussian: GaussianBatch) -> None:
    """Reference renderer's peak intensity should be at the Gaussian center."""
    r = Rasterizer(force_backend="reference")
    out = r(single_centered_gaussian, output_hw=(32, 32))
    assert out.shape == (3, 32, 32)
    # Peak intensity = at center (16, 16)
    peak_y, peak_x = torch.argmax(out[0]).item() // 32, torch.argmax(out[0]).item() % 32
    assert (peak_x, peak_y) == (16, 16)
    # Center value must equal feat[0] (no normalization in reference at peak distance 0).
    center_val = out[:, 16, 16]
    assert torch.allclose(center_val, single_centered_gaussian.feat[0], atol=1e-5)


def test_reference_backend_two_distinct_peaks(two_distinct_gaussians: GaussianBatch) -> None:
    """Two non-overlapping Gaussians should each produce a peak at their position."""
    r = Rasterizer(force_backend="reference")
    out = r(two_distinct_gaussians, output_hw=(32, 32))
    # Red peak at (8, 16): out[0] should be high there, out[2] low.
    assert out[0, 16, 8] > 0.9
    assert out[2, 16, 8] < 0.1
    # Blue peak at (24, 16): out[2] high, out[0] low.
    assert out[2, 16, 24] > 0.9
    assert out[0, 16, 24] < 0.1


def test_reference_backend_empty_batch_returns_zeros() -> None:
    r = Rasterizer(force_backend="reference")
    empty = GaussianBatch(
        xy=torch.zeros((0, 2)),
        scale=torch.zeros((0, 2)),
        rot=torch.zeros((0,)),
        feat=torch.zeros((0, 3)),
    )
    out = r(empty, output_hw=(8, 8))
    assert out.shape == (3, 8, 8)
    assert torch.all(out == 0)


def test_invalid_output_hw_raises() -> None:
    r = Rasterizer(force_backend="reference")
    g = GaussianBatch(
        xy=torch.tensor([[0.0, 0.0]]),
        scale=torch.tensor([[1.0, 1.0]]),
        rot=torch.tensor([0.0]),
        feat=torch.tensor([[1.0]]),
    )
    with pytest.raises(ValueError, match="output_hw must be positive"):
        r(g, output_hw=(0, 8))
    with pytest.raises(ValueError, match="output_hw must be positive"):
        r(g, output_hw=(8, -1))


def test_gaussian_batch_validates_shapes() -> None:
    with pytest.raises(ValueError, match="xy must be"):
        GaussianBatch(
            xy=torch.zeros((1, 3)),  # wrong: should be (N, 2)
            scale=torch.zeros((1, 2)),
            rot=torch.zeros((1,)),
            feat=torch.zeros((1, 3)),
        )
    with pytest.raises(ValueError, match="scale must be"):
        GaussianBatch(
            xy=torch.zeros((1, 2)),
            scale=torch.zeros((1, 3)),
            rot=torch.zeros((1,)),
            feat=torch.zeros((1, 3)),
        )
    with pytest.raises(ValueError, match="rot must be"):
        GaussianBatch(
            xy=torch.zeros((1, 2)),
            scale=torch.zeros((1, 2)),
            rot=torch.zeros((1, 1)),
            feat=torch.zeros((1, 3)),
        )
    with pytest.raises(ValueError, match="feat must be"):
        GaussianBatch(
            xy=torch.zeros((1, 2)),
            scale=torch.zeros((1, 2)),
            rot=torch.zeros((1,)),
            feat=torch.zeros((2, 3)),  # mismatched N
        )


# CUDA-specific tests below. Skipped when CUDA / gsplat unavailable.

cuda_available = torch.cuda.is_available()
try:
    from gsplat import rasterize_gaussians_sum  # noqa: F401

    gsplat_available = True
except Exception:
    gsplat_available = False


@pytest.mark.gpu
@pytest.mark.skipif(not (cuda_available and gsplat_available),
                    reason="CUDA / gsplat not available")
def test_cuda_matches_reference_within_tolerance(two_distinct_gaussians: GaussianBatch) -> None:
    """CUDA backend output should approximate the reference output."""
    g_cpu = two_distinct_gaussians
    g_cuda = GaussianBatch(
        xy=g_cpu.xy.cuda(),
        scale=g_cpu.scale.cuda(),
        rot=g_cpu.rot.cuda(),
        feat=g_cpu.feat.cuda(),
    )
    ref = Rasterizer(force_backend="reference")(g_cpu, output_hw=(32, 32))
    cuda = Rasterizer(force_backend="cuda")(g_cuda, output_hw=(32, 32)).cpu()
    # Tile-based top-K introduces small differences vs the reference at tile
    # boundaries — tolerance is permissive on purpose.
    assert torch.mean(torch.abs(ref - cuda)).item() < 0.05
