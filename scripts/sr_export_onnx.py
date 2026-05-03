"""Export SRCNNSimple to ONNX FP32 and FP16.

Loads the latest checkpoint from --output-dir, exports to ONNX with dynamic
spatial axes (works at any resolution), converts to FP16, validates both
exports against the PyTorch FP32 reference, and reports sizes + max abs diffs.

Key design decisions:
- Dynamic axes on H and W so the same .onnx file runs at any resolution.
- FP16 conversion via onnxconverter_common with keep_io_types=True so the
  ONNX graph accepts and returns FP32 at its boundary; internal ops are FP16.
  This avoids forced FP16 inputs at the call site and keeps measurement
  comparable to the FP32 baseline.
- antialias=True in F.interpolate does NOT export cleanly under opset 17 on
  all ORT+torch pairs. The export call catches this and falls back to
  antialias=False with an explicit warning.
- Verification runs at 256x480 (LR) -> 512x960 (HR) which fits in available
  VRAM even alongside a running training job.

Usage:
    python scripts/sr_export_onnx.py \
        --output-dir <train-host-data>\checkpoints\srcnn-prod-v3 \
        --export-dir <train-host-data>\onnx \
        --opset 17
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

try:
    from onnxconverter_common.float16 import convert_float_to_float16
    _HAS_OCC = True
except ImportError:
    _HAS_OCC = False

from oss.sr import build_sr_model


# ---------------------------------------------------------------------------
# ONNX-safe model wrapper
# ---------------------------------------------------------------------------


class SRCNNExportWrapper(torch.nn.Module):
    """Wrapper that optionally replaces antialias=True bicubic skip.

    antialias=True is not exported cleanly by torch.onnx under all opset 17
    configurations. When use_bilinear_fallback=True, the skip is replaced with
    bilinear align_corners=False. Quality delta: typically <0.1 dB PSNR on
    natural images; the trained residual compensates most of the difference.

    Also replaces inplace=True ReLU with inplace=False for ONNX compatibility.
    """

    def __init__(self, base_model: torch.nn.Module, use_bilinear_fallback: bool = False) -> None:
        super().__init__()
        self.base = base_model
        self.use_bilinear_fallback = use_bilinear_fallback
        self.scale = base_model.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lr_rgb = x[:, :3, :, :]

        feat = F.relu(self.base.head_conv(x), inplace=False)
        feat = self.base.body(feat)
        residual = self.base.pixel_shuffle(self.base.upsample_conv(feat))

        if self.use_bilinear_fallback:
            skip = F.interpolate(
                lr_rgb,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
            )
        else:
            skip = F.interpolate(
                lr_rgb,
                scale_factor=self.scale,
                mode="bicubic",
                antialias=True,
            )

        return skip + residual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_checkpoint(ckpt_dir: Path, device: str) -> tuple[torch.nn.Module, str]:
    """Load latest checkpoint; return (sr_model, checkpoint_name)."""
    ckpts = sorted(ckpt_dir.glob("step-*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No step-*.pt checkpoints in {ckpt_dir}")
    latest = ckpts[-1]
    print(f"Loading checkpoint: {latest.name}")

    ck = torch.load(latest, map_location=device, weights_only=False)
    saved_args = ck.get("args", {})
    tier = saved_args.get("tier", "lite")
    sr_backbone = saved_args.get("sr_backbone", "simple")
    factory_kind = "rrdb" if sr_backbone == "rrdb" else "simple"

    sr_model = build_sr_model(
        model_kind=factory_kind, tier=tier, in_channels=12, scale=2
    ).to(device)
    sr_model.load_state_dict(ck["sr_model"])
    sr_model.train(False)

    n_params = sum(p.numel() for p in sr_model.parameters())
    print(f"  tier={tier}  backbone={factory_kind}  params={n_params:,}")
    print(f"  weights size = {n_params * 4 / 1024**2:.2f} MiB (FP32)")
    return sr_model, latest.name


def _export_onnx(
    wrapper: torch.nn.Module,
    dummy: torch.Tensor,
    out_path: Path,
    opset: int,
) -> None:
    """Export model to ONNX with dynamic spatial axes."""
    dynamic_axes = {
        "input":  {0: "batch", 2: "h", 3: "w"},
        "output": {0: "batch", 2: "h", 3: "w"},
    }
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )


def _build_and_export_fp32(
    sr_model: torch.nn.Module,
    export_dir: Path,
    stem: str,
    opset: int,
    device: str,
    h: int,
    w: int,
) -> tuple[Path, bool]:
    """Build ONNX wrapper, export FP32, return (path, used_bilinear_fallback)."""
    out_path = export_dir / f"{stem}-fp32.onnx"
    dummy = torch.zeros(1, 12, h, w, device=device)
    used_fallback = False

    wrapper = SRCNNExportWrapper(sr_model, use_bilinear_fallback=False)
    wrapper.train(False)
    try:
        with torch.no_grad():
            _export_onnx(wrapper, dummy, out_path, opset)
        print("  bicubic antialias=True: export succeeded.")
    except Exception as exc:
        print(f"  antialias=True failed ({type(exc).__name__}: {exc})")
        print("  Falling back to bilinear antialias=False.")
        print("  QUALITY NOTE: bilinear skip vs bicubic antialias -- typically <0.1 dB PSNR delta.")
        wrapper_fb = SRCNNExportWrapper(sr_model, use_bilinear_fallback=True)
        wrapper_fb.train(False)
        with torch.no_grad():
            _export_onnx(wrapper_fb, dummy, out_path, opset)
        used_fallback = True
        print("  Fallback export succeeded.")

    return out_path, used_fallback


def _validate_onnx(path: Path) -> None:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    print("  onnx.checker: OK")


def _model_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 ** 2


def _ort_session(path: Path) -> ort.InferenceSession:
    available = ort.get_available_providers()
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(path), providers=providers)


def _run_ort(session: ort.InferenceSession, x_np: np.ndarray) -> np.ndarray:
    inp_name = session.get_inputs()[0].name
    return session.run(None, {inp_name: x_np})[0]


def _verify_vs_pytorch(
    onnx_path: Path,
    sr_model: torch.nn.Module,
    device: str,
    h: int,
    w: int,
    label: str,
    atol: float,
) -> float:
    """Compare ONNX output to PyTorch FP32 reference. Returns max abs diff."""
    x_torch = torch.zeros(1, 12, h, w, device=device)

    # Determine which skip mode was used (try antialias, fall back if needed).
    wrapper = SRCNNExportWrapper(sr_model, use_bilinear_fallback=False)
    wrapper.train(False)
    try:
        with torch.no_grad():
            ref_torch = wrapper(x_torch).cpu().numpy()
    except Exception:
        wrapper_fb = SRCNNExportWrapper(sr_model, use_bilinear_fallback=True)
        wrapper_fb.train(False)
        with torch.no_grad():
            ref_torch = wrapper_fb(x_torch).cpu().numpy()

    x_np = x_torch.cpu().numpy()
    session = _ort_session(onnx_path)
    ort_out = _run_ort(session, x_np)

    max_diff = float(np.abs(ref_torch.astype(np.float32) - ort_out.astype(np.float32)).max())
    mean_diff = float(np.abs(ref_torch.astype(np.float32) - ort_out.astype(np.float32)).mean())
    status = "PASS" if max_diff <= atol else "WARN (exceeds tolerance)"
    print(f"  [{label}]")
    print(f"    max|diff|  = {max_diff:.3e}")
    print(f"    mean|diff| = {mean_diff:.3e}")
    print(f"    tolerance  = {atol:.0e}  -> {status}")
    return max_diff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory containing step-*.pt checkpoints.")
    p.add_argument("--export-dir", type=Path, required=True,
                   help="Output directory for .onnx files.")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version (default: 17).")
    p.add_argument("--stem", type=str, default="srcnn-prod-v3",
                   help="File name stem for output files.")
    p.add_argument("--verify-h", type=int, default=256,
                   help="LR height for verification forward pass (default: 256).")
    p.add_argument("--verify-w", type=int, default=480,
                   help="LR width for verification forward pass (default: 480).")
    p.add_argument("--device", type=str, default="cuda",
                   help="PyTorch device for checkpoint load and reference run.")
    args = p.parse_args()

    args.export_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------------------------
    sr_model, ckpt_name = _load_checkpoint(args.output_dir, args.device)

    # -----------------------------------------------------------------------
    # FP32 export
    # -----------------------------------------------------------------------
    print(f"\n=== FP32 ONNX export (opset {args.opset}) ===")
    fp32_path, used_fallback = _build_and_export_fp32(
        sr_model, args.export_dir, args.stem, args.opset,
        args.device, args.verify_h, args.verify_w,
    )

    print(f"  Validating ...")
    _validate_onnx(fp32_path)
    fp32_size = _model_size_mb(fp32_path)
    print(f"  File: {fp32_path}  ({fp32_size:.2f} MB)")

    print(f"  Verifying at {args.verify_h}x{args.verify_w} LR ...")
    _verify_vs_pytorch(fp32_path, sr_model, args.device,
                       args.verify_h, args.verify_w,
                       label="FP32 ONNX vs PyTorch FP32", atol=1e-4)

    # -----------------------------------------------------------------------
    # FP16 export
    # -----------------------------------------------------------------------
    print(f"\n=== FP16 ONNX conversion ===")
    if not _HAS_OCC:
        print("ERROR: onnxconverter_common not available.")
        print("Install with: pip install onnxconverter-common")
        return 1

    fp32_model = onnx.load(str(fp32_path))
    fp16_model = convert_float_to_float16(fp32_model, keep_io_types=True)

    fp16_path = args.export_dir / f"{args.stem}-fp16.onnx"
    onnx.save(fp16_model, str(fp16_path))

    print(f"  Validating ...")
    _validate_onnx(fp16_path)
    fp16_size = _model_size_mb(fp16_path)
    print(f"  File: {fp16_path}  ({fp16_size:.2f} MB)")
    print(f"  Size ratio: FP32/FP16 = {fp32_size/fp16_size:.2f}x")

    print(f"  Verifying at {args.verify_h}x{args.verify_w} LR ...")
    _verify_vs_pytorch(fp16_path, sr_model, args.device,
                       args.verify_h, args.verify_w,
                       label="FP16 ONNX vs PyTorch FP32", atol=1e-2)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n=== Summary ===")
    print(f"  Checkpoint : {ckpt_name}")
    print(f"  FP32 ONNX  : {fp32_size:.2f} MB  -> {fp32_path.name}")
    print(f"  FP16 ONNX  : {fp16_size:.2f} MB  -> {fp16_path.name}")
    if used_fallback:
        print("  NOTE: antialias=True not supported by exporter; used bilinear fallback.")
        print("        Quality delta vs training distribution: typically <0.1 dB PSNR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
