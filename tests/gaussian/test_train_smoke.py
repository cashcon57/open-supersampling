"""Tests for the Sprint 4 smoke-test trainer wiring.

Covers:
- evaluate_against_bicubic returns the expected dict shape and types.
- PSNR correctness on a known pair (model == GT -> high PSNR; bicubic is lower).
- --smoke-test CLI mode implies pico tier and the 3-hour wall-clock limit.
- build_dataloader sequence filter works correctly on a synthetic fixture.

All tests run on CPU with no real Sintel dataset required.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from oss.gaussian.train.train import (
    TrainArgs,
    _psnr,
    build_dataloader,
    evaluate_against_bicubic,
)


# ---------------------------------------------------------------------------
# Helpers: build tiny synthetic Sintel-layout fixture on disk
# ---------------------------------------------------------------------------

HR_H, HR_W = 64, 96   # multiple of 2 -- safe for scale=2 box downsample


def _save_png(path: Path, tensor: torch.Tensor) -> None:
    from torchvision.utils import save_image
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.clamp(0, 1), str(path))


def _write_flo(path: Path, flow: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _, H, W = flow.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<f", 202021.25))
        f.write(struct.pack("<ii", W, H))
        arr = flow.permute(1, 2, 0).contiguous().numpy().astype("float32")
        f.write(arr.tobytes())


def _write_dpt(path: Path, depth: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _, H, W = depth.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<f", 202021.25))
        f.write(struct.pack("<ii", W, H))
        arr = depth.squeeze(0).contiguous().numpy().astype("float32")
        f.write(arr.tobytes())


def _make_sintel_seq(root: Path, seq: str, n_frames: int = 4) -> None:
    """Write n_frames of clean/depth/flow data for a single sequence."""
    pass_dir = root / "training" / "clean"
    flow_dir = root / "training" / "flow"
    depth_dir = root / "training" / "depth"
    for i in range(1, n_frames + 1):
        rgb = torch.rand(3, HR_H, HR_W)
        _save_png(pass_dir / seq / f"frame_{i:04d}.png", rgb)
        _write_dpt(depth_dir / seq / f"frame_{i:04d}.dpt", torch.rand(1, HR_H, HR_W) * 10.0)
        if i < n_frames:  # last frame has no flow in real Sintel
            _write_flo(flow_dir / seq / f"frame_{i:04d}.flo", torch.randn(2, HR_H, HR_W) * 2.0)


# ---------------------------------------------------------------------------
# Fixture: minimal TrainArgs for testing (no real data needed for unit tests)
# ---------------------------------------------------------------------------


def _minimal_args(tmp_path: Path, **overrides: Any) -> TrainArgs:
    """Build a TrainArgs suitable for unit testing.

    Defaults to --use-synthetic-batch to avoid needing real data, but
    tests that exercise the DataLoader path supply their own dataset_root.
    """
    defaults: dict[str, Any] = dict(
        tier="pico",
        max_steps=5,
        batch_size=2,
        learning_rate=3e-4,
        output_dir=tmp_path / "out",
        dataset_root=tmp_path / "datasets",
        bank_size=16,
        k_per_tile=5,
        log_every=1,
        ckpt_every=100,
        seed=0,
        device="cpu",
        use_synthetic_batch=True,
        dataset="sintel",
        sintel_sequence=None,
        srgd_scene=None,
        force_lr_synth=False,
        renderer_backend="auto",
        enable_gbuffer_bias=False,
        enable_engine_aliased_lr=False,
        lr_synth_blur_sigma=0.5,
        lr_synth_jpeg=False,
        lr_synth_jpeg_quality=85,
        enable_pixel_residual=False,
        pixel_residual_hidden=32,
        score_every=0,
        max_time_seconds=None,
        smoke_test=False,
    )
    defaults.update(overrides)
    return TrainArgs(**defaults)


# ---------------------------------------------------------------------------
# 1. evaluate_against_bicubic -- expected keys and types
# ---------------------------------------------------------------------------


def _make_fake_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Build a collated batch dict matching what DataLoader produces."""
    lr = torch.rand(batch_size, 3, 32, 48)
    depth = torch.rand(batch_size, 1, 32, 48)
    motion = torch.rand(batch_size, 2, 32, 48) * 2.0
    normals = torch.rand(batch_size, 3, 32, 48) * 2.0 - 1.0
    canvas = torch.zeros(batch_size, 3, 32, 48)
    gt_hr = torch.rand(batch_size, 3, 64, 96)
    return {
        "lr_frame": lr,
        "depth": depth,
        "motion": motion,
        "normals": normals,
        "canvas_hint": canvas,
        "gt_hr_frame": gt_hr,
    }


def test_evaluate_against_bicubic_returns_expected_keys(tmp_path: Path) -> None:
    """evaluate_against_bicubic must return all required keys with correct types."""
    from oss.gaussian.network import CovariancePriorBank, OutputHead
    from oss.gaussian.network.param_net import param_net_for_tier

    bank = CovariancePriorBank(learnable=False)
    net = param_net_for_tier("pico", bank_size=16)
    head = OutputHead(bank=bank, k_per_tile=net.k_per_tile, enable_gbuffer_bias=False)

    # Single-batch "dataloader" (just a list of one batch dict).
    fake_batch = _make_fake_batch(batch_size=2)
    fake_loader = [fake_batch]

    result = evaluate_against_bicubic(
        net, head, bank, fake_loader, device="cpu", n_samples=2
    )

    assert isinstance(result, dict), "result must be a dict"
    for key in (
        "model_psnr_mean",
        "bicubic_psnr_mean",
        "model_psnr_per_sample",
        "bicubic_psnr_per_sample",
        "model_beats_bicubic_count",
    ):
        assert key in result, f"missing key {key!r}"

    assert isinstance(result["model_psnr_mean"], float)
    assert isinstance(result["bicubic_psnr_mean"], float)
    assert isinstance(result["model_psnr_per_sample"], list)
    assert isinstance(result["bicubic_psnr_per_sample"], list)
    assert isinstance(result["model_beats_bicubic_count"], int)
    assert len(result["model_psnr_per_sample"]) == len(result["bicubic_psnr_per_sample"])
    assert 0 <= result["model_beats_bicubic_count"] <= len(result["model_psnr_per_sample"])


# ---------------------------------------------------------------------------
# 2. PSNR correctness on a known pair
# ---------------------------------------------------------------------------


def test_evaluate_against_bicubic_psnr_correctness_on_known_pair(tmp_path: Path) -> None:
    """When model output == GT, model PSNR should be very high (>40 dB).

    We test _psnr directly on:
      - A perfect prediction (model == GT) -> PSNR should be huge.
      - Bicubic upsample of a downsampled image -> moderate PSNR.
    """
    torch.manual_seed(0)
    gt = torch.rand(3, 64, 96)

    # Perfect prediction: model output == GT.
    perfect_pred = gt.clone()
    perfect_psnr = _psnr(perfect_pred, gt)
    assert perfect_psnr > 40.0, (
        f"Perfect prediction should have PSNR > 40 dB, got {perfect_psnr:.2f} dB"
    )

    # Bicubic baseline: downsample then upsample -> moderate PSNR.
    lr = F.interpolate(gt.unsqueeze(0), scale_factor=0.5, mode="bilinear", align_corners=False)
    bicubic = F.interpolate(lr, size=(64, 96), mode="bicubic", antialias=True).squeeze(0).clamp(0, 1)
    bicubic_psnr = _psnr(bicubic, gt)

    # Bicubic on a clean downsample should be reasonable (typically 25-40 dB)
    # but should not reach the perfect-prediction level.
    assert bicubic_psnr < perfect_psnr, (
        f"Bicubic PSNR ({bicubic_psnr:.2f}) should be less than perfect ({perfect_psnr:.2f})"
    )
    assert bicubic_psnr > 10.0, (
        f"Bicubic PSNR should be > 10 dB on a clean pair, got {bicubic_psnr:.2f}"
    )

    # Verify PSNR clamping: identical tensors should not produce inf.
    psnr_identical = _psnr(gt, gt)
    assert psnr_identical < float("inf"), "PSNR on identical tensors must be finite (clamped)"
    assert psnr_identical > 100.0, "PSNR on identical tensors should be very large"


# ---------------------------------------------------------------------------
# 3. --smoke-test implies pico tier and 3-hour time bound
# ---------------------------------------------------------------------------


def test_smoke_test_args_implies_pico_tier_and_time_bound(tmp_path: Path) -> None:
    """Parsing --smoke-test sets tier='pico', batch_size=2, max_time_seconds=10800."""
    args = TrainArgs.from_cli([
        "--smoke-test",
        "--sintel-sequence", "alley_1",
        "--dataset-root", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
        "--max-steps", "100",
    ])

    assert args.tier == "pico", f"smoke-test must set tier=pico, got {args.tier!r}"
    assert args.batch_size == 2, f"smoke-test must set batch_size=2, got {args.batch_size}"
    assert args.max_time_seconds == 10800, (
        f"smoke-test must set max_time_seconds=10800 (3 hr), "
        f"got {args.max_time_seconds}"
    )
    assert args.enable_gbuffer_bias is True, "smoke-test must enable gbuffer bias"
    assert args.enable_engine_aliased_lr is True, "smoke-test must enable engine-aliased LR"
    assert args.use_synthetic_batch is False, "smoke-test must use real data (not synthetic)"
    assert args.smoke_test is True


def test_smoke_test_explicit_time_override_is_respected(tmp_path: Path) -> None:
    """Explicit --max-time-seconds should override smoke-test default (3 hr)."""
    args = TrainArgs.from_cli([
        "--smoke-test",
        "--sintel-sequence", "alley_1",
        "--dataset-root", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
        "--max-steps", "100",
        "--max-time-seconds", "60",
    ])
    assert args.max_time_seconds == 60


# ---------------------------------------------------------------------------
# 4. build_dataloader sequence filter
# ---------------------------------------------------------------------------


def test_build_dataloader_filters_to_single_sequence(tmp_path: Path) -> None:
    """Requesting a specific sequence should return only frames from that sequence."""
    sintel_root = tmp_path / "datasets"
    mpi_root = sintel_root / "MPI-Sintel-complete"

    # Create two sequences: alley_1 (3 frames -> 2 valid) and bamboo_1 (4 frames -> 3 valid).
    _make_sintel_seq(mpi_root, "alley_1", n_frames=3)
    _make_sintel_seq(mpi_root, "bamboo_1", n_frames=4)

    args = _minimal_args(
        tmp_path,
        use_synthetic_batch=False,
        sintel_sequence="alley_1",
        dataset_root=sintel_root,
        batch_size=1,
        enable_engine_aliased_lr=False,
    )

    loader = build_dataloader(args)
    ds = loader.dataset

    # alley_1 has 3 frames but last has no flow, so 2 valid items.
    assert len(ds) == 2, (
        f"Expected 2 items for alley_1 (3 frames, last has no flow), got {len(ds)}"
    )

    # Verify all items belong to alley_1.
    for item in ds._items:  # type: ignore[attr-defined]
        frame_path = item[0]
        assert frame_path.parent.name == "alley_1", (
            f"Expected alley_1, got {frame_path.parent.name}"
        )


def test_build_dataloader_raises_on_missing_sequence(tmp_path: Path) -> None:
    """Requesting a non-existent sequence name should raise ValueError."""
    sintel_root = tmp_path / "datasets"
    mpi_root = sintel_root / "MPI-Sintel-complete"
    _make_sintel_seq(mpi_root, "alley_1", n_frames=3)

    args = _minimal_args(
        tmp_path,
        use_synthetic_batch=False,
        sintel_sequence="nonexistent_seq",
        dataset_root=sintel_root,
        batch_size=1,
        enable_engine_aliased_lr=False,
    )

    with pytest.raises(ValueError, match="No frames found for sequence"):
        build_dataloader(args)
