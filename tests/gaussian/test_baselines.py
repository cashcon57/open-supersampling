"""Tests for OSS-Gaussian baseline upscalers."""

from __future__ import annotations

import csv
from pathlib import Path

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
from oss.gaussian.bench import run_baselines


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


# ---- run_baselines.py end-to-end smoke ------------------------------------


def _save_png(path: Path, tensor: torch.Tensor) -> None:
    from torchvision.utils import save_image
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.clamp(0, 1), str(path))


def test_run_baselines_on_synthetic_sintel_fixture(tmp_path: Path) -> None:
    """End-to-end: build a tiny synthetic Sintel-shaped tree, invoke
    ``run_baselines.main()``, verify the CSV contains one row per
    (baseline, sequence) pair with finite metrics + latency.

    Catches integration regressions in the bench script itself: argparse
    wiring, dataset traversal, metric construction, CSV schema.
    """
    # 3 sequences x 2 frames each, HR 64x64 (multiple of scale=2).
    H, W = 64, 64
    sintel_root = tmp_path / "sintel"
    pass_dir = sintel_root / "training" / "clean"
    sequences = ["alley_1", "bandage_2", "mountain_1"]
    for seq in sequences:
        for i in range(1, 3):
            _save_png(pass_dir / seq / f"frame_{i:04d}.png", torch.rand(3, H, W))

    out_csv = tmp_path / "bench_results.csv"
    rc = run_baselines.main([
        "--sintel-root", str(sintel_root),
        "--scale", "2.0",
        "--output", str(out_csv),
        "--baselines", "bicubic", "lanczos",
        "--max-sequences", str(len(sequences)),
        "--runs", "3",
        "--warmup", "1",
        "--device", "cpu",  # deterministic, avoid MPS in CI / test
    ])
    assert rc == 0
    assert out_csv.is_file()

    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(sequences) * 2  # bicubic + lanczos per sequence

    seen = {(r["baseline"], r["sequence"]) for r in rows}
    assert seen == {(b, s) for b in ("bicubic", "lanczos") for s in sequences}

    for r in rows:
        # PSNR / SSIM / latency must be finite and sensible.
        psnr_v = float(r["psnr_db"])
        ssim_v = float(r["ssim"])
        mean_ms = float(r["mean_ms"])
        assert psnr_v > 0.0, f"non-positive PSNR for {r}"
        assert 0.0 <= ssim_v <= 1.0, f"SSIM out of range for {r}"
        assert mean_ms > 0.0
        # LPIPS may be missing if the user has a broken install — but it's
        # listed in pyproject.toml as a hard dep so should always be present.
        assert r["lpips_vgg"] not in ("", None), "LPIPS missing — check pyproject deps"


def test_run_baselines_missing_root_returns_error(tmp_path: Path) -> None:
    rc = run_baselines.main([
        "--sintel-root", str(tmp_path / "does-not-exist"),
        "--output", str(tmp_path / "x.csv"),
        "--runs", "1",
        "--warmup", "1",
        "--device", "cpu",
    ])
    assert rc == 2
