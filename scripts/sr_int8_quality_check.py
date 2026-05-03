"""INT8 quality gate: compare PyTorch FP32 vs TRT FP16 vs TRT INT8 on held-out frames.

Runs 16 SRGD frames from a held-out scene (CitySample by default, which the
model was NOT trained on) through all three inference paths and reports:
  - PSNR (dB) vs ground-truth HR
  - LPIPS (lower is better, requires lpips package)

Quality thresholds from the spec:
  - INT8 vs FP32 reference: PSNR drop > 1 dB  → deal-breaker
  - INT8 vs FP32 reference: LPIPS delta > 0.05 → deal-breaker

The FP32 reference uses PyTorch (the original training path).
TRT FP16 is measured via ORT TensorrtExecutionProvider (same as sr_bench_onnx.py).
TRT INT8 uses the native TRT engine from sr_export_trt_int8.py.

Usage:
    python scripts/sr_int8_quality_check.py \\
        --checkpoint <train-host-data>\\checkpoints\\srcnn-prod-v3 \\
        --fp16-onnx <train-host-data>\\onnx\\srcnn-prod-v3-fp16.onnx \\
        --trt-int8-engine <train-host-data>\\onnx\\srcnn-prod-v3-int8.trt \\
        --dataset-root <train-host-data>\\datasets\\srgd \\
        --eval-scene CitySample \\
        --n-samples 16
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows (default console is cp1252 which lacks Unicode)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np

# ---------------------------------------------------------------------------
# Windows DLL fix — before any ORT/TRT import.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import torch as _torch_tmp
    _torch_lib = Path(_torch_tmp.__file__).parent / "lib"
    if _torch_lib.exists():
        os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(_torch_lib))
        except (OSError, AttributeError):
            pass
    _conda_bin = Path(sys.executable).parent.parent / "bin"
    if _conda_bin.exists():
        os.environ["PATH"] = str(_conda_bin) + os.pathsep + os.environ.get("PATH", "")
    try:
        import tensorrt as _trt_side  # noqa: F401
    except ImportError:
        pass

import torch
import torch.nn.functional as F
import onnxruntime as ort

from oss.gaussian.data.lr_synthesis import EngineAliasedLRSynth
from oss.gaussian.data.srgd import SRGDGaussianDataset
from oss.sr import build_sr_model

# Load TRTEngine from sr_export_trt_int8.py
import importlib.util as _ilu
_int8_script = Path(__file__).parent / "sr_export_trt_int8.py"
_spec_int8 = _ilu.spec_from_file_location("sr_export_trt_int8", str(_int8_script))
_int8_mod = _ilu.module_from_spec(_spec_int8)
_spec_int8.loader.exec_module(_int8_mod)
TRTEngine = _int8_mod.TRTEngine


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = float(F.mse_loss(pred.float().clamp(0, 1), target.float().clamp(0, 1)).item())
    mse = max(mse, 1e-12)
    return float(-10.0 * math.log10(mse))


def _lpips_fn():
    """Return an LPIPS metric instance (net=alex, spatial=False)."""
    try:
        import lpips
        return lpips.LPIPS(net="alex", verbose=False)
    except ImportError:
        return None


def _compute_lpips(fn, pred: torch.Tensor, target: torch.Tensor) -> float | None:
    """Compute LPIPS on (3, H, W) tensors in [0,1]. Returns None if lpips unavailable."""
    if fn is None:
        return None
    # LPIPS expects (N,3,H,W) in [-1,1].
    a = pred.float().clamp(0, 1).unsqueeze(0) * 2 - 1
    b = target.float().clamp(0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return float(fn(a.cpu(), b.cpu()).item())


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _infer_pytorch(sr_model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """FP32 PyTorch inference. x: (1, 12, H, W) on device."""
    with torch.no_grad():
        return sr_model(x).clamp(0, 1)


def _infer_ort_fp16(session: ort.InferenceSession, x: torch.Tensor) -> torch.Tensor:
    """ORT FP16 inference. x: (1, 12, H, W) on CPU or GPU."""
    inp_name = session.get_inputs()[0].name
    x_np = x.cpu().numpy().astype(np.float32)
    out_np = session.run(None, {inp_name: x_np})[0]
    return torch.from_numpy(out_np).clamp(0, 1)


def _infer_trt_int8(engine: "TRTEngine", x: torch.Tensor) -> torch.Tensor:
    """TRT INT8 inference. x: (1, 12, H, W)."""
    x_np = x.cpu().numpy().astype(np.float32)
    out_np = engine.infer(x_np)
    return torch.from_numpy(out_np).clamp(0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path(r"<train-host-data>\checkpoints\srcnn-prod-v3"),
                   help="Checkpoint dir (step-*.pt). Latest is used.")
    p.add_argument("--fp16-onnx", type=Path, default=Path(r"<train-host-data>\onnx\srcnn-prod-v3-fp16.onnx"),
                   help="FP16 ONNX for ORT TRT FP16 path.")
    p.add_argument("--trt-int8-engine", type=Path, default=Path(r"<train-host-data>\onnx\srcnn-prod-v3-int8.trt"),
                   help="Serialized TRT INT8 engine from sr_export_trt_int8.py.")
    p.add_argument("--dataset-root", type=Path, default=Path(r"<train-host-data>\datasets\srgd"))
    p.add_argument("--eval-scene", type=str, default="CitySample",
                   help="Held-out SRGD scene (default: CitySample).")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--eval-h", type=int, default=720,
                   help="Resize LR input to this height before inference (must fit an engine profile).")
    p.add_argument("--eval-w", type=int, default=1280,
                   help="Resize LR input to this width before inference.")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    # -----------------------------------------------------------------------
    # Load PyTorch model
    # -----------------------------------------------------------------------
    ckpt_dir = args.checkpoint
    ckpts = sorted(ckpt_dir.glob("step-*.pt"))
    if not ckpts:
        print(f"ERROR: no checkpoints in {ckpt_dir}")
        return 1
    latest_ckpt = ckpts[-1]
    print(f"Checkpoint: {latest_ckpt.name}")
    ck = torch.load(latest_ckpt, map_location=args.device, weights_only=False)
    saved_args = ck.get("args", {})
    tier = saved_args.get("tier", "lite")
    sr_backbone = saved_args.get("sr_backbone", "simple")
    factory_kind = "rrdb" if sr_backbone == "rrdb" else "simple"
    sr_model = build_sr_model(model_kind=factory_kind, tier=tier, in_channels=12, scale=2).to(args.device)
    sr_model.load_state_dict(ck["sr_model"])
    sr_model.train(False)
    print(f"  tier={tier}  backbone={factory_kind}")

    # -----------------------------------------------------------------------
    # ORT FP16 session (via ORT TRT EP if available, else CUDA)
    # -----------------------------------------------------------------------
    fp16_ok = args.fp16_onnx.exists()
    if fp16_ok:
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" in available:
            fp16_providers = [
                ("TensorrtExecutionProvider", {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(args.fp16_onnx.parent / (args.fp16_onnx.stem + ".trt-cache")),
                }),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            print("  TRT FP16 path: TensorrtExecutionProvider (FP16)")
        else:
            fp16_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("  TRT FP16 path: CUDAExecutionProvider (TRT EP not available)")
        fp16_session = ort.InferenceSession(str(args.fp16_onnx), providers=fp16_providers)
    else:
        fp16_session = None
        print(f"  WARN: FP16 ONNX not found at {args.fp16_onnx} — TRT FP16 column will be skipped.")

    # -----------------------------------------------------------------------
    # TRT INT8 engine
    # -----------------------------------------------------------------------
    int8_ok = args.trt_int8_engine.exists()
    if int8_ok:
        try:
            int8_engine = TRTEngine(args.trt_int8_engine)
            print(f"  TRT INT8 engine loaded: {args.trt_int8_engine}")
        except Exception as exc:
            print(f"  WARN: TRT INT8 engine load failed: {exc}")
            int8_engine = None
            int8_ok = False
    else:
        int8_engine = None
        print(f"  WARN: TRT INT8 engine not found at {args.trt_int8_engine}")

    # -----------------------------------------------------------------------
    # LPIPS
    # -----------------------------------------------------------------------
    lpips_fn = _lpips_fn()
    if lpips_fn is None:
        print("  WARN: lpips not installed — LPIPS column will be N/A. Install: pip install lpips")

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    lr_synth = EngineAliasedLRSynth(
        scale=2.0, enable_jitter=True, enable_taa_blur=True,
        enable_jpeg=True, jpeg_quality=85, blur_sigma=1.5,
    )
    candidates = [args.dataset_root, args.dataset_root / "srgd"]
    srgd_root = next(
        (c for c in candidates if (c / "data" / "GameEngineData").is_dir()), None
    )
    if srgd_root is None:
        print(f"ERROR: SRGD dataset not found under {candidates}")
        return 1

    ds = SRGDGaussianDataset(
        root=srgd_root, scale=2.0, lr_synth=lr_synth,
        scene=args.eval_scene, force_synth_lr=True,
    )
    print(f"  eval_scene={args.eval_scene}  total_frames={len(ds)}  eval_res={args.eval_h}x{args.eval_w}")

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    fp32_psnrs: list[float] = []
    fp16_psnrs: list[float] = []
    int8_psnrs: list[float] = []
    fp32_lpips: list[float] = []
    fp16_lpips: list[float] = []
    int8_lpips: list[float] = []

    # Use fixed stride so samples spread evenly across the scene.
    stride = max(1, len(ds) // args.n_samples)
    indices = [i * stride for i in range(args.n_samples) if i * stride < len(ds)]
    if len(indices) < args.n_samples:
        # top up from the front if scene has fewer than n_samples * stride frames
        indices += list(range(len(indices), min(args.n_samples, len(ds))))
    indices = indices[:args.n_samples]

    print(f"\nEvaluating {len(indices)} frames from {args.eval_scene} ...")

    eval_h, eval_w = args.eval_h, args.eval_w
    eval_hr_h, eval_hr_w = eval_h * 2, eval_w * 2

    for frame_idx in indices:
        ex = ds[frame_idx]
        normals = ex.normals if ex.normals is not None else torch.zeros(
            (3, *ex.lr_frame.shape[-2:]), dtype=torch.float32
        )
        x_12ch = torch.cat([ex.lr_frame, ex.depth, ex.motion, normals, ex.canvas_hint], dim=0)

        # Resize to evaluation resolution if needed.  The engine profiles only
        # cover {800,720,900,1080}×{1280,1280,1600,1920} LR shapes; inputs
        # from some scenes may be at a different native resolution.
        h_src, w_src = x_12ch.shape[-2], x_12ch.shape[-1]
        if (h_src, w_src) != (eval_h, eval_w):
            x_12ch = F.interpolate(
                x_12ch.unsqueeze(0),
                size=(eval_h, eval_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        x_batch = x_12ch.unsqueeze(0).to(args.device)  # (1, 12, eval_h, eval_w)

        # GT HR resized to match the 2× upscaled output of the model.
        gt_hr_raw = ex.gt_hr_frame  # (3, H_hr, W_hr) at native HR res
        if gt_hr_raw.shape[-2:] != (eval_hr_h, eval_hr_w):
            gt_hr = F.interpolate(
                gt_hr_raw.unsqueeze(0),
                size=(eval_hr_h, eval_hr_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).to(args.device)
        else:
            gt_hr = gt_hr_raw.to(args.device)

        # FP32 reference
        fp32_out = _infer_pytorch(sr_model, x_batch)[0]  # (3, H_hr, W_hr)
        fp32_psnrs.append(_psnr(fp32_out, gt_hr))
        if lpips_fn is not None:
            fp32_lpips.append(_compute_lpips(lpips_fn, fp32_out.cpu(), gt_hr.cpu()))

        # TRT FP16
        if fp16_session is not None:
            fp16_out = _infer_ort_fp16(fp16_session, x_batch)[0]
            fp16_psnrs.append(_psnr(fp16_out, gt_hr.cpu()))
            if lpips_fn is not None:
                fp16_lpips.append(_compute_lpips(lpips_fn, fp16_out, gt_hr.cpu()))

        # TRT INT8
        if int8_engine is not None:
            int8_out_t = _infer_trt_int8(int8_engine, x_batch)[0]
            int8_psnrs.append(_psnr(int8_out_t, gt_hr.cpu()))
            if lpips_fn is not None:
                int8_lpips.append(_compute_lpips(lpips_fn, int8_out_t, gt_hr.cpu()))

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    def _mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else float("nan")

    def _fmt(v: float, decimals: int = 2) -> str:
        return f"{v:.{decimals}f}" if v == v else "  N/A"

    fp32_p = _mean(fp32_psnrs)
    fp16_p = _mean(fp16_psnrs) if fp16_psnrs else float("nan")
    int8_p = _mean(int8_psnrs) if int8_psnrs else float("nan")
    fp32_l = _mean(fp32_lpips) if fp32_lpips else float("nan")
    fp16_l = _mean(fp16_lpips) if fp16_lpips else float("nan")
    int8_l = _mean(int8_lpips) if int8_lpips else float("nan")

    def _delta_psnr(ref: float, val: float) -> str:
        if val == val and ref == ref:
            return f"{val - ref:+.2f} dB"
        return "   N/A"

    def _delta_lpips(ref: float, val: float) -> str:
        if val == val and ref == ref:
            return f"{val - ref:+.3f}"
        return "   N/A"

    print()
    print("=" * 72)
    print(f"Quality report - {args.eval_scene} ({len(indices)} held-out frames)")
    print("=" * 72)
    print(f"  {'Method':<18}  {'PSNR (dB)':>10}  {'dPSNR':>10}  {'LPIPS':>8}  {'dLPIPS':>8}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    print(f"  {'PyTorch FP32':<18}  {_fmt(fp32_p):>10}  {'(ref)':>10}  {_fmt(fp32_l, 3):>8}  {'(ref)':>8}")
    if fp16_psnrs:
        print(f"  {'TRT FP16':<18}  {_fmt(fp16_p):>10}  {_delta_psnr(fp32_p, fp16_p):>10}  {_fmt(fp16_l, 3):>8}  {_delta_lpips(fp32_l, fp16_l):>8}")
    if int8_psnrs:
        print(f"  {'TRT INT8':<18}  {_fmt(int8_p):>10}  {_delta_psnr(fp32_p, int8_p):>10}  {_fmt(int8_l, 3):>8}  {_delta_lpips(fp32_l, int8_l):>8}")
    print()

    # Gate evaluation
    passed = True
    if int8_psnrs:
        psnr_drop = fp32_p - int8_p
        lpips_delta = (int8_l - fp32_l) if (int8_l == int8_l and fp32_l == fp32_l) else 0.0
        print(f"INT8 quality gate (vs FP32 reference):")
        if psnr_drop > 1.0:
            print(f"  FAIL: PSNR drop {psnr_drop:.2f} dB > 1.0 dB threshold — INT8 quality unacceptable")
            passed = False
        else:
            print(f"  PASS: PSNR drop {psnr_drop:.2f} dB ≤ 1.0 dB")
        if lpips_fn is not None and lpips_delta > 0.05:
            print(f"  FAIL: LPIPS delta {lpips_delta:.3f} > 0.05 threshold — INT8 quality unacceptable")
            passed = False
        elif lpips_fn is not None:
            print(f"  PASS: LPIPS delta {lpips_delta:.3f} ≤ 0.05")
        if passed:
            print("  VERDICT: INT8 quality ACCEPTABLE — safe to ship")
        else:
            print("  VERDICT: INT8 quality UNACCEPTABLE — do not ship INT8; fall back to FP16")
    else:
        print("INT8 results not available (engine not loaded).")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
