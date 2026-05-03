"""Benchmark PyTorch FP32 vs ONNX FP32 vs ONNX FP16 at 4 representative resolutions.

Mirrors the measurement methodology in sr_inference_vram.py:
- 3 warmup passes, then 5 timed passes (matching vram script)
- torch.cuda.synchronize() before timing stop
- Peak VRAM via torch.cuda.max_memory_allocated() for PyTorch paths
- ORT CUDA paths: VRAM estimated via torch.cuda.memory_allocated() delta

VRAM safety: the benchmark uses batch=1 and releases tensors between runs.
At standard tier (~306K params) peak VRAM is ~300-500 MB per run -- well
within the ~7-8 GB headroom available alongside the running training job.
1080p inputs are included but skipped automatically if they would OOM.

Usage:
    python scripts/sr_bench_onnx.py --onnx-dir <train-host-data>\onnx
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

import onnxruntime as ort

from oss.sr import build_sr_model

# ---------------------------------------------------------------------------
# Import the export wrapper for consistent skip behaviour
# ---------------------------------------------------------------------------
import sys
import importlib.util

_EXPORT_SCRIPT = Path(__file__).parent / "sr_export_onnx.py"
_spec = importlib.util.spec_from_file_location("sr_export_onnx", str(_EXPORT_SCRIPT))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SRCNNExportWrapper = _mod.SRCNNExportWrapper


# ---------------------------------------------------------------------------
# Representative resolutions
# ---------------------------------------------------------------------------

RESOLUTIONS = [
    ("Steam Deck  (800x1280 LR -> 1600x2560)",  800, 1280),
    ("720p        (720x1280 LR -> 1440x2560)",  720, 1280),
    ("900p        (900x1600 LR -> 1800x3200)",  900, 1600),
    ("1080p / 4K  (1080x1920 LR -> 2160x3840)", 1080, 1920),
]

N_WARMUP = 3
N_RUNS = 5


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _free_vram() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _bench_pytorch(
    sr_model: torch.nn.Module,
    device: str,
    h: int,
    w: int,
) -> dict:
    """Benchmark PyTorch FP32 inference. Returns stats dict."""
    _free_vram()
    torch.cuda.reset_peak_memory_stats(device)

    wrapper = SRCNNExportWrapper(sr_model, use_bilinear_fallback=False)
    wrapper.train(False)
    # Try bicubic antialias=True, fall back if platform doesn't support it.
    x = torch.zeros(1, 12, h, w, device=device)

    def _forward(m, inp):
        try:
            return m(inp)
        except Exception:
            fb = SRCNNExportWrapper(sr_model, use_bilinear_fallback=True)
            fb.train(False)
            return fb(inp)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _forward(wrapper, x)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        t0 = time.monotonic()
        for _ in range(N_RUNS):
            _forward(wrapper, x)
        torch.cuda.synchronize(device)
        elapsed = (time.monotonic() - t0) / N_RUNS

    peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
    return {"ms": elapsed * 1000, "fps": 1.0 / elapsed, "peak_mib": peak_mib, "label": "PyTorch FP32"}


def _ort_providers() -> list[str]:
    available = ort.get_available_providers()
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _bench_ort(
    onnx_path: Path,
    device: str,
    h: int,
    w: int,
    label: str,
) -> dict:
    """Benchmark ONNX Runtime. Returns stats dict."""
    _free_vram()

    providers = _ort_providers()
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    inp_name = session.get_inputs()[0].name

    x_np = np.zeros((1, 12, h, w), dtype=np.float32)

    # Warmup
    for _ in range(N_WARMUP):
        session.run(None, {inp_name: x_np})

    # Measure VRAM before timed runs.
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    vram_before = torch.cuda.memory_allocated(device)

    t0 = time.monotonic()
    for _ in range(N_RUNS):
        session.run(None, {inp_name: x_np})
    torch.cuda.synchronize(device)
    elapsed = (time.monotonic() - t0) / N_RUNS

    peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
    # If no CUDA provider or VRAM is zero, report peak as N/A.
    if "CUDAExecutionProvider" not in providers:
        peak_mib = float("nan")

    return {"ms": elapsed * 1000, "fps": 1.0 / elapsed, "peak_mib": peak_mib, "label": label}


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def _fmt_mib(v: float) -> str:
    if v != v:  # NaN
        return "    N/A"
    return f"{v:7.1f}"


def _print_table(rows: list[dict], resolution_label: str) -> None:
    print(f"\n{resolution_label}")
    print(f"  {'Backend':<28}  {'peak VRAM':>9}  {'ms/frame':>9}  {'fps':>6}  {'size MB':>8}")
    print(f"  {'-'*28}  {'-'*9}  {'-'*9}  {'-'*6}  {'-'*8}")
    for r in rows:
        size_s = f"{r['size_mb']:8.2f}" if r.get("size_mb") else "       —"
        print(f"  {r['label']:<28}  {_fmt_mib(r['peak_mib'])} MiB  {r['ms']:9.2f}  {r['fps']:6.1f}  {size_s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", type=Path, required=True,
                   help="Directory containing srcnn-prod-v3-fp32.onnx and -fp16.onnx")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Checkpoint dir (to load PyTorch model). "
                        "Defaults to discovering model from ONNX metadata.")
    p.add_argument("--stem", type=str, default="srcnn-prod-v3",
                   help="ONNX file stem (default: srcnn-prod-v3).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir-ckpt", type=Path, default=Path("<train-host-data>/checkpoints/srcnn-prod-v3"),
                   help="Checkpoint dir for PyTorch model load.")
    args = p.parse_args()

    fp32_onnx = args.onnx_dir / f"{args.stem}-fp32.onnx"
    fp16_onnx = args.onnx_dir / f"{args.stem}-fp16.onnx"

    if not fp32_onnx.exists():
        print(f"ERROR: {fp32_onnx} not found. Run sr_export_onnx.py first.")
        return 1
    if not fp16_onnx.exists():
        print(f"ERROR: {fp16_onnx} not found. Run sr_export_onnx.py first.")
        return 1

    fp32_mb = fp32_onnx.stat().st_size / 1024**2
    fp16_mb = fp16_onnx.stat().st_size / 1024**2

    # Load PyTorch model from checkpoint.
    ckpt_dir = args.output_dir_ckpt
    ckpts = sorted(ckpt_dir.glob("step-*.pt"))
    if not ckpts:
        print(f"ERROR: No checkpoints in {ckpt_dir}")
        return 1
    latest = ckpts[-1]
    print(f"Checkpoint: {latest.name}")
    ck = torch.load(latest, map_location=args.device, weights_only=False)
    saved_args = ck.get("args", {})
    tier = saved_args.get("tier", "lite")
    sr_backbone = saved_args.get("sr_backbone", "simple")
    factory_kind = "rrdb" if sr_backbone == "rrdb" else "simple"
    sr_model = build_sr_model(model_kind=factory_kind, tier=tier, in_channels=12, scale=2).to(args.device)
    sr_model.load_state_dict(ck["sr_model"])
    sr_model.train(False)

    n_params = sum(p.numel() for p in sr_model.parameters())
    pt_size_mb = n_params * 4 / 1024**2
    print(f"  tier={tier}  backbone={factory_kind}  params={n_params:,}  FP32 weights={pt_size_mb:.2f} MiB")
    print(f"  FP32 ONNX: {fp32_mb:.2f} MB")
    print(f"  FP16 ONNX: {fp16_mb:.2f} MB")

    providers = _ort_providers()
    print(f"  ORT providers: {providers}")

    print("\n" + "=" * 80)
    print("Benchmark: PyTorch FP32  vs  ONNX-RT FP32  vs  ONNX-RT FP16")
    print("=" * 80)

    all_tables = []

    for res_label, h, w in RESOLUTIONS:
        rows: list[dict] = []

        # PyTorch FP32
        try:
            r = _bench_pytorch(sr_model, args.device, h, w)
            r["size_mb"] = pt_size_mb
            rows.append(r)
        except torch.cuda.OutOfMemoryError:
            rows.append({"label": "PyTorch FP32", "ms": float("nan"), "fps": 0.0, "peak_mib": float("nan"), "size_mb": pt_size_mb})
            torch.cuda.empty_cache()
            print(f"  PyTorch FP32 @ {h}x{w}: OOM")

        # ONNX FP32
        try:
            r = _bench_ort(fp32_onnx, args.device, h, w, label="ONNX-RT FP32")
            r["size_mb"] = fp32_mb
            rows.append(r)
        except Exception as exc:
            rows.append({"label": "ONNX-RT FP32", "ms": float("nan"), "fps": 0.0, "peak_mib": float("nan"), "size_mb": fp32_mb})
            print(f"  ONNX-RT FP32 @ {h}x{w}: {exc}")

        # ONNX FP16
        try:
            r = _bench_ort(fp16_onnx, args.device, h, w, label="ONNX-RT FP16")
            r["size_mb"] = fp16_mb
            rows.append(r)
        except Exception as exc:
            rows.append({"label": "ONNX-RT FP16", "ms": float("nan"), "fps": 0.0, "peak_mib": float("nan"), "size_mb": fp16_mb})
            print(f"  ONNX-RT FP16 @ {h}x{w}: {exc}")

        _print_table(rows, res_label)
        all_tables.append((res_label, h, w, rows))

    print("\n" + "=" * 80)
    print("Done. Speedup summary (ONNX FP16 vs PyTorch FP32):")
    for res_label, h, w, rows in all_tables:
        by_label = {r["label"]: r for r in rows}
        pt_ms = by_label.get("PyTorch FP32", {}).get("ms", float("nan"))
        fp16_ms = by_label.get("ONNX-RT FP16", {}).get("ms", float("nan"))
        if pt_ms == pt_ms and fp16_ms == fp16_ms and fp16_ms > 0:
            speedup = pt_ms / fp16_ms
            print(f"  {res_label:<45}  {speedup:.2f}x faster")
        else:
            print(f"  {res_label:<45}  N/A")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
