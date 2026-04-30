"""Export ORU-Pico from PyTorch checkpoint to ONNX.

Usage:
    python -m ors.export.onnx_export --ckpt <path-to-pth> --out <path-to-onnx>

Validates ONNX inference matches PyTorch reference within FP32 tolerance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ors.model.oru_pico import ORUPico


def export(ckpt_path: Path, out_path: Path, validate: bool = True):
    """Load checkpoint, export to ONNX, optionally validate parity.

    Args:
        ckpt_path: Path to .pth checkpoint containing {model, config}.
        out_path: Path to write .onnx file.
        validate: If True, compare PyTorch vs ONNX Runtime outputs.
    """
    # Load checkpoint
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ORUPico().train(False)
    model.load_state_dict(state["model"])

    # Dummy inputs for tracing
    B = 1
    H_lr, W_lr = 64, 64  # canonical export shape; runtime can use dynamic axes
    scale = state.get("config", {}).get("scale_factor", 2.0)
    H_hr = int(H_lr * scale)
    W_hr = int(W_lr * scale)

    color_lr = torch.randn(B, 3, H_lr, W_lr)
    depth_lr = torch.randn(B, 1, H_lr, W_lr)
    motion_lr = torch.randn(B, 2, H_lr, W_lr)
    normals_lr = torch.randn(B, 3, H_lr, W_lr)
    albedo_lr = torch.randn(B, 3, H_lr, W_lr)
    history_hr = torch.randn(B, 3, H_hr, W_hr)
    # hidden_state: None at sequence start; we trace the explicit-zeros path for ONNX.
    hidden_zero = torch.zeros(B, 24, H_lr // 4, W_lr // 4)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export with dynamic batch + spatial axes
    torch.onnx.export(
        model,
        (color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_zero),
        str(out_path),
        opset_version=17,
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
    p = argparse.ArgumentParser(description="Export ORU-Pico to ONNX from checkpoint.")
    p.add_argument("--ckpt", type=Path, required=True, help="Path to .pth checkpoint")
    p.add_argument("--out", type=Path, default=Path("oru_pico.onnx"), help="Output .onnx path")
    p.add_argument("--no-validate", action="store_true", help="Skip parity validation")
    args = p.parse_args()
    export(args.ckpt, args.out, validate=not args.no_validate)


if __name__ == "__main__":
    main()
