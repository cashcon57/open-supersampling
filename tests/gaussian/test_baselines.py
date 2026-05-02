"""Tests for OSS-Gaussian baseline upscalers."""

from __future__ import annotations

import pytest
import torch

from oss.gaussian.bench.baselines import (
    BicubicUpscaler,
    DLSSFrameGenUpscaler,
    DLSSQualityUpscaler,
    FSR2Upscaler,
    LanczosUpscaler,
    REGISTRY,
    bicubic_upscale,
    lanczos_upscale,
    make,
)


@pytest.fixture
def lr_frame() -> torch.Tensor:
    return torch.rand((1, 3, 32, 32))


def test_bicubic_upscale_shape(lr_frame: torch.Tensor) -> None:
    out = bicubic_upscale(lr_frame, scale=2.0)
    assert out.shape == (1, 3, 64, 64)


def test_bicubic_upscaler_returns_result(lr_frame: torch.Tensor) -> None:
    r = BicubicUpscaler()(lr_frame, scale=2.0)
    assert r.name == "bicubic"
    assert r.image.shape == (1, 3, 64, 64)
    assert r.elapsed_ms == 0.0  # caller times this


def test_lanczos_upscaler_returns_result(lr_frame: torch.Tensor) -> None:
    # Even without kornia installed, falls back to bicubic — should not raise.
    r = LanczosUpscaler()(lr_frame, scale=2.0)
    assert r.image.shape == (1, 3, 64, 64)


def test_lanczos_function(lr_frame: torch.Tensor) -> None:
    out = lanczos_upscale(lr_frame, scale=3.0)
    assert out.shape == (1, 3, 96, 96)


def test_vendor_baselines_raise_clearly(lr_frame: torch.Tensor) -> None:
    """SDK-gated baselines should raise NotImplementedError with a clear
    message until their SDK shim lands. Catch silently failing wiring."""
    for cls in (FSR2Upscaler, DLSSQualityUpscaler, DLSSFrameGenUpscaler):
        with pytest.raises(NotImplementedError, match="not yet wired"):
            cls()(lr_frame, scale=2.0)


def test_registry_keys_match_classes() -> None:
    assert set(REGISTRY.keys()) == {
        "bicubic", "lanczos", "fsr2_quality", "dlss_sr_quality", "dlss_fg",
    }
    for k in REGISTRY:
        assert make(k).name == k


def test_make_rejects_unknown() -> None:
    with pytest.raises(KeyError, match="unknown baseline"):
        make("not-a-baseline")


def test_bicubic_preserves_dtype_and_device(lr_frame: torch.Tensor) -> None:
    f32 = lr_frame.float()
    out = bicubic_upscale(f32, 2.0)
    assert out.dtype == f32.dtype
    assert out.device == f32.device


def test_bicubic_handles_non_unit_scale(lr_frame: torch.Tensor) -> None:
    out = bicubic_upscale(lr_frame, 1.5)
    assert out.shape == (1, 3, 48, 48)
