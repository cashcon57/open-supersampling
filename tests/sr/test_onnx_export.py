"""Tests for ONNX export roundtrip correctness.

All tests skip if onnxruntime is not installed. They run on CPU and use
purely synthetic (untrained) SRCNNSimple models -- no checkpoint required.

Coverage:
1. test_export_roundtrip_shape     -- export at any tier, load with ORT,
                                      verify output shape (B, 3, 128, 192)
                                      for 64x96 LR input.
2. test_export_dynamic_axes_work   -- same ONNX file accepts 64x96 AND
                                      128x192 inputs.
3. test_export_fp16_roundtrip      -- FP16 ONNX output within 1e-2 of FP32.
"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Skip entire module if onnxruntime is not installed.
ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
onnx = pytest.importorskip("onnx", reason="onnx not installed")

from oss.sr.cnn import SRCNNSimple

# Attempt to import onnxconverter_common for the FP16 test.
try:
    from onnxconverter_common.float16 import convert_float_to_float16
    _HAS_OCC = True
except ImportError:
    _HAS_OCC = False

# Import the ONNX-safe wrapper from the export script.
import importlib.util

_EXPORT_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "sr_export_onnx.py"
_spec = importlib.util.spec_from_file_location("sr_export_onnx", str(_EXPORT_SCRIPT))
_export_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_export_mod)
SRCNNExportWrapper = _export_mod.SRCNNExportWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(tier: str = "lite") -> SRCNNSimple:
    """Create an untrained model (no checkpoint needed)."""
    configs = {"pico": (16, 2), "lite": (32, 4), "standard": (64, 8)}
    hidden, n_blocks = configs[tier]
    model = SRCNNSimple(in_channels=12, scale=2, hidden=hidden, n_blocks=n_blocks)
    model.train(False)
    return model


def _export_to_tmpfile(model: SRCNNSimple, tmp_dir: Path, opset: int = 17) -> Path:
    """Export model to a temp ONNX file. Returns path."""
    out_path = tmp_dir / "test_model.onnx"
    wrapper = SRCNNExportWrapper(model, use_bilinear_fallback=False)
    wrapper.train(False)
    dummy = torch.zeros(1, 12, 64, 96)

    dynamic_axes = {
        "input":  {0: "batch", 2: "h", 3: "w"},
        "output": {0: "batch", 2: "h", 3: "w"},
    }
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, str(out_path),
                opset_version=opset,
                input_names=["input"], output_names=["output"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )
    except Exception:
        # Fallback to bilinear if antialias=True fails on this platform.
        wrapper_fb = SRCNNExportWrapper(model, use_bilinear_fallback=True)
        wrapper_fb.train(False)
        with torch.no_grad():
            torch.onnx.export(
                wrapper_fb, dummy, str(out_path),
                opset_version=opset,
                input_names=["input"], output_names=["output"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )
    return out_path


def _ort_session(onnx_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _ort_run(session: ort.InferenceSession, x_np: np.ndarray) -> np.ndarray:
    inp_name = session.get_inputs()[0].name
    return session.run(None, {inp_name: x_np})[0]


# ---------------------------------------------------------------------------
# 1. Shape roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["pico", "lite", "standard"])
def test_export_roundtrip_shape(tier: str) -> None:
    """Export at each tier, run at 64x96 LR, verify output (B, 3, 128, 192)."""
    model = _make_model(tier)
    with tempfile.TemporaryDirectory() as td:
        onnx_path = _export_to_tmpfile(model, Path(td))
        session = _ort_session(onnx_path)

        x_np = np.zeros((1, 12, 64, 96), dtype=np.float32)
        out = _ort_run(session, x_np)

    assert out.shape == (1, 3, 128, 192), (
        f"tier={tier}: expected (1,3,128,192), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 2. Dynamic axes
# ---------------------------------------------------------------------------


def test_export_dynamic_axes_work() -> None:
    """Same ONNX file must accept both 64x96 and 128x192 inputs."""
    model = _make_model("lite")
    with tempfile.TemporaryDirectory() as td:
        onnx_path = _export_to_tmpfile(model, Path(td))
        session = _ort_session(onnx_path)

        out_small = _ort_run(session, np.zeros((1, 12, 64, 96), dtype=np.float32))
        out_large = _ort_run(session, np.zeros((1, 12, 128, 192), dtype=np.float32))

    assert out_small.shape == (1, 3, 128, 192), f"Small input: got {out_small.shape}"
    assert out_large.shape == (1, 3, 256, 384), f"Large input: got {out_large.shape}"


# ---------------------------------------------------------------------------
# 3. FP16 numeric fidelity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OCC, reason="onnxconverter_common not installed")
def test_export_fp16_roundtrip() -> None:
    """FP16 ONNX output must be within 1e-2 of FP32 ONNX output."""
    torch.manual_seed(42)
    model = _make_model("lite")

    with tempfile.TemporaryDirectory() as td:
        fp32_path = _export_to_tmpfile(model, Path(td))

        # FP16 conversion
        fp32_model = onnx.load(str(fp32_path))
        fp16_model = convert_float_to_float16(fp32_model, keep_io_types=True)
        fp16_path = Path(td) / "test_model_fp16.onnx"
        onnx.save(fp16_model, str(fp16_path))

        sess_fp32 = _ort_session(fp32_path)
        sess_fp16 = _ort_session(fp16_path)

        # Use a random input (not zero) to exercise the residual path.
        x_np = np.random.default_rng(42).uniform(0.0, 1.0, (1, 12, 64, 96)).astype(np.float32)

        out_fp32 = _ort_run(sess_fp32, x_np)
        out_fp16 = _ort_run(sess_fp16, x_np)

    max_diff = float(np.abs(out_fp32 - out_fp16.astype(np.float32)).max())
    atol = 1e-2
    assert max_diff <= atol, (
        f"FP16 vs FP32 max|diff|={max_diff:.3e} exceeds tolerance {atol:.0e}"
    )
