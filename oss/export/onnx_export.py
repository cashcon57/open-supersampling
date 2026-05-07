"""Export OSS-Pico from PyTorch checkpoint to ONNX.

Usage:
    python -m ors.export.onnx_export --ckpt <path-to-pth> --out <path-to-onnx>

Validates ONNX inference matches PyTorch reference within FP32 tolerance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from oss.model.oss_pico import OSSPico


ExportInputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _make_export_inputs(
    batch: int,
    h_lr: int,
    w_lr: int,
    h_hr: int,
    w_hr: int,
) -> ExportInputs:
    """Build deterministic, physically plausible inputs for tracing/parity."""

    gen = torch.Generator(device="cpu").manual_seed(20260507)
    color_lr = torch.rand(batch, 3, h_lr, w_lr, generator=gen)
    depth_lr = torch.rand(batch, 1, h_lr, w_lr, generator=gen)
    motion_lr = (torch.rand(batch, 2, h_lr, w_lr, generator=gen) - 0.5) * 2.0
    normals_lr = torch.rand(batch, 3, h_lr, w_lr, generator=gen) * 2.0 - 1.0
    normals_lr = F.normalize(normals_lr, dim=1, eps=1e-6)
    albedo_lr = torch.rand(batch, 3, h_lr, w_lr, generator=gen) * 0.95 + 0.05
    history_hr = torch.rand(batch, 3, h_hr, w_hr, generator=gen)
    hidden_zero = torch.zeros(batch, 24, h_lr // 4, w_lr // 4)
    return color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_zero


def export(ckpt_path: Path, out_path: Path, validate: bool = True):
    """Load checkpoint, export to ONNX, optionally validate parity.

    Args:
        ckpt_path: Path to .pth checkpoint containing {model, config}.
        out_path: Path to write .onnx file.
        validate: If True, compare PyTorch vs ONNX Runtime outputs.
    """
    # Load checkpoint
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    scale_factor = float(state.get("config", {}).get("scale_factor", 2.0))
    model = OSSPico(scale_factor=scale_factor).eval()
    model.load_state_dict(state["model"])

    # Dummy inputs for tracing
    B = 1
    H_lr, W_lr = 64, 64  # canonical export shape; runtime can use dynamic axes
    H_hr = int(H_lr * scale_factor)
    W_hr = int(W_lr * scale_factor)

    # hidden_state: None at sequence start; we trace the explicit-zeros path for ONNX.
    color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_zero = _make_export_inputs(
        B,
        H_lr,
        W_lr,
        H_hr,
        W_hr,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export with dynamic batch + spatial axes.
    # dynamo=True is required: the legacy TorchScript exporter cannot
    # trace the wavelet-stack circular padding (see
    # ``oss/model/wavelet.py::SWT2D``) — pad-then-conv loses static kernel
    # shape, and the legacy path raises SymbolicValueError("ONNX export
    # of convolution for kernel of unknown shape"). The dynamo path
    # correctly handles the dynamic shape and exports cleanly.
    torch.onnx.export(
        model,
        (color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_zero),
        str(out_path),
        dynamo=True,
        # opset 18 (was 17): the Resize op used by upsample paths has no
        # v17 adapter in the current onnx-c-api version converter (see
        # `No Adapter To Version $17 for Resize`), so the exporter fails
        # to convert and silently leaves the graph at v18 — but the v18
        # graph then evaluates with a small numerical shift vs the
        # PyTorch eager pass, breaking the rgb-parity assertion downstream.
        # Asking for v18 directly skips the failed-conversion step and
        # produces a graph whose runtime output matches eager within 2e-3.
        opset_version=18,
        input_names=["color_lr", "depth_lr", "motion_lr", "normals_lr", "albedo_lr", "history_hr", "hidden_state"],
        output_names=["rgb_hr", "new_hidden_state"],
        dynamic_axes={
            "color_lr":     {0: "batch", 2: "h_lr", 3: "w_lr"},
            "depth_lr":     {0: "batch", 2: "h_lr", 3: "w_lr"},
            "motion_lr":    {0: "batch", 2: "h_lr", 3: "w_lr"},
            "normals_lr":   {0: "batch", 2: "h_lr", 3: "w_lr"},
            "albedo_lr":    {0: "batch", 2: "h_lr", 3: "w_lr"},
            "history_hr":   {0: "batch", 2: "h_hr", 3: "w_hr"},
            "hidden_state": {0: "batch", 2: "h_lr_quarter", 3: "w_lr_quarter"},
            "rgb_hr":       {0: "batch", 2: "h_hr", 3: "w_hr"},
            "new_hidden_state": {0: "batch", 2: "h_lr_quarter", 3: "w_lr_quarter"},
        },
        do_constant_folding=True,
    )

    size_kb = out_path.stat().st_size / 1024
    print(f"exported ONNX to {out_path} ({size_kb:.1f} KB)")

    if validate:
        _validate_parity(model, out_path, color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_zero)


def _validate_parity(model, onnx_path, color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_zero):
    """Compare PyTorch vs ONNX Runtime output on the same inputs."""
    import numpy as np
    import onnxruntime as ort

    # PyTorch reference
    with torch.no_grad():
        rgb_pt, hidden_pt = model(color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_zero)

    # ONNX Runtime
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_inputs = {
        "color_lr": color_lr.numpy(),
        "depth_lr": depth_lr.numpy(),
        "motion_lr": motion_lr.numpy(),
        "normals_lr": normals_lr.numpy(),
        "albedo_lr": albedo_lr.numpy(),
        "history_hr": history_hr.numpy(),
        "hidden_state": hidden_zero.numpy(),
    }
    rgb_onnx, hidden_onnx = sess.run(["rgb_hr", "new_hidden_state"], onnx_inputs)

    # Tolerances suitable for FP32-traced graph; relaxed for radiance demodulation
    # (element-wise division + multiplication introduces rounding across FP32 boundaries).
    rgb_diff = np.abs(rgb_pt.numpy() - rgb_onnx).max()
    hidden_diff = np.abs(hidden_pt.numpy() - hidden_onnx).max()
    print(f"max abs diff — rgb: {rgb_diff:.6e}, hidden: {hidden_diff:.6e}")
    assert rgb_diff < 2e-3, f"RGB parity failed: {rgb_diff}"
    assert hidden_diff < 1e-4, f"hidden_state parity failed: {hidden_diff}"
    print("✓ ONNX/PyTorch parity within tolerance")


def main():
    p = argparse.ArgumentParser(description="Export OSS-Pico to ONNX from checkpoint.")
    p.add_argument("--ckpt", type=Path, required=True, help="Path to .pth checkpoint")
    p.add_argument("--out", type=Path, default=Path("oru_pico.onnx"), help="Output .onnx path")
    p.add_argument("--no-validate", action="store_true", help="Skip parity validation")
    args = p.parse_args()
    export(args.ckpt, args.out, validate=not args.no_validate)


if __name__ == "__main__":
    main()
