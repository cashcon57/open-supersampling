"""Subprocess smoke tests for the v5 temporal ONNX export script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from oss.sr.temporal import TemporalSRModel

onnx = pytest.importorskip("onnx", reason="onnx not installed")


def test_temporal_onnx_export_help_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/sr_export_temporal_onnx.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"--help exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "usage" in proc.stdout.lower()
    assert "--ckpt" in proc.stdout
    assert "--lr-h" in proc.stdout
    assert "--lr-w" in proc.stdout


def test_temporal_onnx_export_synthetic_ckpt(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = TemporalSRModel(
        in_channels=12, scale=2, tier="pico", backbone_kind="simple",
    )
    ckpt = tmp_path / "tiny-temporal.pt"
    torch.save(
        {
            "temporal_model": model.state_dict(),
            "args": {
                "in_channels": 12,
                "scale": 2,
                "tier": "pico",
                "backbone_kind": "simple",
            },
        },
        ckpt,
    )
    out = tmp_path / "tiny-temporal.onnx"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/sr_export_temporal_onnx.py",
            "--ckpt",
            str(ckpt),
            "--output",
            str(out),
            "--lr-h",
            "64",
            "--lr-w",
            "64",
            "--opset",
            "17",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"export exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    assert out.exists()
    assert out.stat().st_size > 0
    exported = onnx.load(str(out))
    onnx.checker.check_model(exported)
