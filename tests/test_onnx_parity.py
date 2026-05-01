"""Verify ONNX export round-trip preserves model behavior."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from oss.model.oss_pico import OSSPico


def test_onnx_export_round_trip(tmp_path):
    """Train a fresh OSS-Pico (random weights), save ckpt, export, compare."""
    pytest.importorskip("onnxruntime")  # skip if onnxruntime not installed locally

    model = OSSPico().train(False)
    ckpt = tmp_path / "test_pico.pth"
    torch.save({"model": model.state_dict(), "config": {"scale_factor": 2.0, "tier": "pico"}}, ckpt)

    onnx_path = tmp_path / "test_pico.onnx"

    result = subprocess.run(
        [sys.executable, "-m", "ors.export.onnx_export", "--ckpt", str(ckpt), "--out", str(onnx_path)],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"export failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0
    # Module's _validate_parity prints its check; non-zero exit means failed assertion
    assert "ONNX/PyTorch parity within tolerance" in result.stdout
