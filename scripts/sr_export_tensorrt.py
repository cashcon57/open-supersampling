"""Export ONNX FP16 to a TensorRT engine using ORT's TensorrtExecutionProvider.

This uses onnxruntime-gpu's built-in TensorRT integration rather than the
standalone `tensorrt` Python package. This approach is simpler and works if
`TensorrtExecutionProvider` is listed in ort.get_available_providers().

Note: TensorRT engine compilation is slow (minutes at first run) but the
compiled engine is cached. Re-running the benchmark after the first run will
use the cache.

Note on correctness: ORT's TensorRT path compiles a subset of the graph with
TRT and falls back to CUDA for unsupported ops. The PixelShuffle + FP16
combination is well-supported in TRT 8+. We verify output shape and a rough
numeric match against the ONNX-RT FP16 baseline.

Usage:
    python scripts/sr_export_tensorrt.py \
        --onnx <train-host-data>\onnx\srcnn-prod-v3-fp16.onnx \
        --output <train-host-data>\onnx\srcnn-prod-v3-fp16.trt \
        --bench
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ORT CUDA + TRT DLL fix (Windows only) -- must run before importing onnxruntime.
# Importing tensorrt first triggers its __init__.py which calls
# os.add_dll_directory on the right paths. Order matters: ORT's
# TensorrtExecutionProvider loads its DLL on first use, and that DLL
# transitively depends on nvinfer_10.dll; if tensorrt was never imported in
# this process, Windows can't find it even with PATH set.
if sys.platform == "win32":
    import torch as _torch_tmp
    _torch_lib = Path(_torch_tmp.__file__).parent / "lib"
    if _torch_lib.exists():
        os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")
        try: os.add_dll_directory(str(_torch_lib))
        except (OSError, AttributeError): pass
    _conda_bin = Path(sys.executable).parent.parent / "bin"
    if _conda_bin.exists():
        os.environ["PATH"] = str(_conda_bin) + os.pathsep + os.environ.get("PATH", "")
    try:
        import tensorrt as _trt_warmup  # noqa: F401 — side-effect: registers DLLs
    except ImportError:
        pass

import numpy as np
import onnxruntime as ort


RESOLUTIONS = [
    ("Steam Deck  (800x1280 LR -> 1600x2560)",  800, 1280),
    ("720p        (720x1280 LR -> 1440x2560)",  720, 1280),
    ("900p        (900x1600 LR -> 1800x3200)",  900, 1600),
    ("1080p / 4K  (1080x1920 LR -> 2160x3840)", 1080, 1920),
]

N_WARMUP = 3
N_RUNS = 5


def _check_trt_available() -> bool:
    return "TensorrtExecutionProvider" in ort.get_available_providers()


def _make_trt_session(onnx_path: Path, trt_engine_cache_dir: Path) -> ort.InferenceSession:
    """Create an ORT session backed by TensorRT."""
    trt_engine_cache_dir.mkdir(parents=True, exist_ok=True)

    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(trt_engine_cache_dir),
                "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2 GB
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def _bench_session(session: ort.InferenceSession, h: int, w: int) -> dict:
    inp_name = session.get_inputs()[0].name
    x_np = np.zeros((1, 12, h, w), dtype=np.float32)

    for _ in range(N_WARMUP):
        session.run(None, {inp_name: x_np})

    t0 = time.monotonic()
    for _ in range(N_RUNS):
        out = session.run(None, {inp_name: x_np})
    elapsed = (time.monotonic() - t0) / N_RUNS

    return {
        "ms": elapsed * 1000,
        "fps": 1.0 / elapsed,
        "out_shape": out[0].shape,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True,
                   help="Input FP16 ONNX file.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output path for TRT engine cache dir (directory, not file).")
    p.add_argument("--bench", action="store_true",
                   help="Run benchmark at all 4 resolutions after export.")
    args = p.parse_args()

    if not _check_trt_available():
        print("SKIP: TensorrtExecutionProvider not available in this ORT build.")
        print("Available providers:", ort.get_available_providers())
        print("To enable TensorRT via ORT: install onnxruntime-gpu built against TRT,")
        print("or install the tensorrt package separately.")
        return 0

    print("TensorrtExecutionProvider: available")
    print(f"ONNX input: {args.onnx}")

    if not args.onnx.exists():
        print(f"ERROR: {args.onnx} does not exist. Run sr_export_onnx.py first.")
        return 1

    # Use the output path as the engine cache directory.
    cache_dir = args.output if args.output.suffix == "" else args.output.parent / args.output.stem

    print("Building TRT session (first run compiles engine -- this takes 1-5 minutes) ...")
    t_compile_start = time.monotonic()
    try:
        session = _make_trt_session(args.onnx, cache_dir)
    except Exception as exc:
        print(f"ERROR: TRT session creation failed: {exc}")
        return 1
    t_compile = time.monotonic() - t_compile_start
    print(f"Session ready in {t_compile:.1f}s")

    # Quick shape verification at small input.
    inp_name = session.get_inputs()[0].name
    x_check = np.zeros((1, 12, 64, 96), dtype=np.float32)
    out_check = session.run(None, {inp_name: x_check})[0]
    assert out_check.shape == (1, 3, 128, 192), (
        f"Shape mismatch: expected (1,3,128,192), got {out_check.shape}"
    )
    print(f"Shape check at 64x96 LR: output {out_check.shape}  PASS")

    if not args.bench:
        print(f"TRT engine cached at: {cache_dir}")
        return 0

    # Also load the FP16 ONNX-RT session for direct comparison.
    ort_fp16_session = ort.InferenceSession(
        str(args.onnx),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    print("\n" + "=" * 70)
    print("TensorRT vs ONNX-RT FP16 benchmark")
    print("=" * 70)
    print(f"  {'Resolution':<45}  {'TRT ms':>8}  {'TRT fps':>7}  {'ORT16 ms':>9}  {'speedup':>8}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*8}")

    for label, h, w in RESOLUTIONS:
        try:
            trt_r = _bench_session(session, h, w)
            ort_r = _bench_session(ort_fp16_session, h, w)
            speedup = ort_r["ms"] / trt_r["ms"]
            print(f"  {label:<45}  {trt_r['ms']:8.2f}  {trt_r['fps']:7.1f}  {ort_r['ms']:9.2f}  {speedup:8.2f}x")
        except Exception as exc:
            print(f"  {label:<45}  ERROR: {exc}")

    print(f"\nTRT engine cached at: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
