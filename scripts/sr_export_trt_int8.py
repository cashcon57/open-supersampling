"""Export SRCNNSimple to a TensorRT INT8 engine via post-training calibration.

Uses the native TensorRT Python bindings (tensorrt package).  Memory management
uses PyTorch's CUDA allocator (torch.Tensor.data_ptr()) rather than pycuda, so
pycuda does not need to be installed.

WHY NATIVE TRT OVER ORT TRT-EP INT8:
ORT's TensorrtExecutionProvider exposes trt_int8_enable and
trt_int8_calibration_table_name options, but on TRT 10.x the calibration table
it generates is an internal flatbuffers format that ORT generates and manages
itself — there is no Python hook to drive the calibration data loop externally.
Without controlling which samples drive calibration we cannot match the training
distribution.  The native TRT Python API (IInt8EntropyCalibrator2) gives us
full control, so we use it and then wrap the resulting .trt engine for inference.

Pipeline:
    1. Load FP32 ONNX.
    2. Build calibration batches from N SRGD frames at a fixed LR resolution
       using EngineAliasedLRSynth(blur_sigma=1.5, enable_jpeg=True, jpeg_quality=85).
       Save to .npy cache; skip re-generation on subsequent runs.
    3. Implement IInt8EntropyCalibrator2 with get_batch() returning device pointers
       from pinned PyTorch tensors (avoids pycuda dependency).
    4. Build TRT engine with BuilderFlag.INT8 (+ FP16 for mixed-precision fallback).
       Dynamic shape profile: min=(1,12,256,256) opt=(1,12,calib_h,calib_w)
       max=(1,12,1080,1920) — covers all 4 bench resolutions.
    5. Serialize engine to disk.
    6. Validate: compare INT8 output to FP16 ORT reference; report PSNR.

Usage:
    python scripts/sr_export_trt_int8.py \\
        --onnx <train-host-data>\\onnx\\srcnn-prod-v3-fp32.onnx \\
        --output <train-host-data>\\onnx\\srcnn-prod-v3-int8.trt \\
        --dataset-root <train-host-data>\\datasets\\srgd \\
        --calib-scene ActionRPG \\
        --calib-samples 64 \\
        --calib-h 720 --calib-w 1280 \\
        --bench
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Windows DLL path fix — must happen before importing tensorrt or onnxruntime.
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
    # Must import tensorrt BEFORE onnxruntime so nvinfer_10.dll is registered.
    try:
        import tensorrt as _trt_side_effect  # noqa: F401
    except ImportError:
        pass

import onnxruntime as ort
import torch
import torch.cuda


# ---------------------------------------------------------------------------
# TRT import guard
# ---------------------------------------------------------------------------

try:
    import tensorrt as trt
    _TRT_OK = True
except ImportError:
    _TRT_OK = False


# ---------------------------------------------------------------------------
# Calibration data builder
# ---------------------------------------------------------------------------

CALIB_ARRAY_DTYPE = np.float32


def _build_calib_array(
    dataset_root: Path,
    scene: str,
    n_samples: int,
    calib_h: int,
    calib_w: int,
    calib_cache: Path,
) -> np.ndarray:
    """Return calibration batch array (N, 12, H, W) float32, loading from cache if present."""

    if calib_cache.exists():
        print(f"  Loading calibration array from cache: {calib_cache}")
        arr = np.load(str(calib_cache))
        print(f"  Shape: {arr.shape}  dtype: {arr.dtype}")
        return arr

    print(f"  Building calibration array from SRGD/{scene} (n={n_samples}, {calib_h}x{calib_w}) ...")

    from oss.gaussian.data.lr_synthesis import EngineAliasedLRSynth
    from oss.gaussian.data.srgd import SRGDGaussianDataset
    import torch.nn.functional as F

    lr_synth = EngineAliasedLRSynth(
        scale=2.0,
        enable_jitter=True,
        enable_taa_blur=True,
        enable_jpeg=True,
        jpeg_quality=85,
        blur_sigma=1.5,
    )

    # Try dataset root directly, then <root>/srgd subfolder.
    candidates = [dataset_root, dataset_root / "srgd"]
    srgd_root = next(
        (c for c in candidates if (c / "data" / "GameEngineData").is_dir()),
        None,
    )
    if srgd_root is None:
        raise FileNotFoundError(
            f"SRGD dataset not found under {candidates}. "
            "Expected <root>/data/GameEngineData/<scene>/*.png"
        )

    ds = SRGDGaussianDataset(
        root=srgd_root,
        scale=2.0,
        lr_synth=lr_synth,
        scene=scene,
        force_synth_lr=True,
    )
    print(f"  SRGD {scene}: {len(ds)} frames available")

    batches: list[np.ndarray] = []
    for i in range(n_samples):
        ex = ds[i % len(ds)]
        # Stack 12-ch input.
        normals = ex.normals if ex.normals is not None else torch.zeros(
            (3, *ex.lr_frame.shape[-2:]), dtype=torch.float32
        )
        x_12ch = torch.cat(
            [ex.lr_frame, ex.depth, ex.motion, normals, ex.canvas_hint], dim=0
        )  # (12, H_lr, W_lr)

        # Resize to calibration resolution if the source frame differs.
        h_lr, w_lr = x_12ch.shape[-2:]
        if (h_lr, w_lr) != (calib_h, calib_w):
            x_12ch = F.interpolate(
                x_12ch.unsqueeze(0),
                size=(calib_h, calib_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        batches.append(x_12ch.numpy().astype(CALIB_ARRAY_DTYPE))

    arr = np.stack(batches, axis=0)  # (N, 12, H, W)
    calib_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(calib_cache), arr)
    print(f"  Saved calibration array to {calib_cache}")
    return arr


# ---------------------------------------------------------------------------
# IInt8EntropyCalibrator2 — pycuda-free, uses torch.Tensor device pointers.
# ---------------------------------------------------------------------------

class _SRCNNCalibrator(trt.IInt8EntropyCalibrator2 if _TRT_OK else object):
    """Calibrator that serves one batch at a time from a pre-built numpy array.

    Uses a pinned PyTorch CUDA tensor as the device-side buffer.
    get_batch() uploads each batch via tensor.copy_() and returns data_ptr().
    TRT read_calibration_cache / write_calibration_cache skip the calibration
    loop on re-runs if the cache file already exists.
    """

    def __init__(
        self,
        calib_array: np.ndarray,
        batch_size: int,
        cache_path: Path,
    ) -> None:
        super().__init__()
        self._array = calib_array       # (N, 12, H, W) float32 on CPU
        self._batch_size = batch_size
        self._n_batches = len(calib_array) // batch_size
        self._current = 0
        self._cache_path = cache_path

        # Allocate a persistent CUDA tensor for one batch; reused across calls.
        n, c, h, w = calib_array.shape
        self._cuda_buf: torch.Tensor = torch.empty(
            (batch_size, c, h, w), dtype=torch.float32, device="cuda"
        )

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names: list[str]):  # type: ignore[override]
        if self._current >= self._n_batches:
            return None
        batch = self._array[
            self._current * self._batch_size : (self._current + 1) * self._batch_size
        ]
        # Copy from CPU numpy → pinned memory → CUDA buffer.
        cpu_t = torch.from_numpy(batch)
        self._cuda_buf.copy_(cpu_t)
        torch.cuda.synchronize()
        self._current += 1
        return [int(self._cuda_buf.data_ptr())]

    def read_calibration_cache(self):
        if self._cache_path.exists():
            print(f"    Loading TRT calibration cache: {self._cache_path}")
            return self._cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_bytes(cache)
        print(f"    Wrote TRT calibration cache ({len(cache)/1024:.1f} KB): {self._cache_path}")


# ---------------------------------------------------------------------------
# Engine builder
# ---------------------------------------------------------------------------


def _build_int8_engine(
    onnx_path: Path,
    output_path: Path,
    calib_array: np.ndarray,
    calib_cache_path: Path,
    calib_h: int,
    calib_w: int,
    batch_size: int = 1,
    workspace_gib: float = 2.0,
    also_fp16: bool = True,
    verbose: bool = False,
) -> None:
    """Build a TRT INT8 engine from ONNX and serialize it to disk."""
    log_level = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"  Parsing ONNX: {onnx_path}")
    with open(str(onnx_path), "rb") as f:
        onnx_bytes = f.read()
    ok = parser.parse(onnx_bytes)
    if not ok:
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed: {errs}")
    n_inputs = network.num_inputs
    inp_names = [network.get_input(i).name for i in range(n_inputs)]
    print(f"  ONNX parsed OK.  Inputs: {inp_names}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gib * 1024 ** 3),
    )
    config.set_flag(trt.BuilderFlag.INT8)
    if also_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Calibrator.  TRT skips the calibration loop if read_calibration_cache()
    # returns non-None data (i.e., the cache file exists from a prior run).
    calibrator = _SRCNNCalibrator(
        calib_array=calib_array,
        batch_size=batch_size,
        cache_path=calib_cache_path,
    )
    config.int8_calibrator = calibrator

    # Dynamic shape profiles.  TRT INT8 with a single wide min/max range causes
    # extremely long kernel-search times (6+ hours observed on RTX 3080 Ti).
    # Solution: use 4 separate narrow optimization profiles, one per benchmark
    # resolution.  The engine covers all 4 shapes without TRT needing to explore
    # the full (256→1920) width range in one go.
    inp_name = inp_names[0]
    bench_shapes = [
        (800, 1280),   # Steam Deck LR
        (720, 1280),   # 720p LR  (calibration shape == opt for this profile)
        (900, 1600),   # 900p LR
        (1080, 1920),  # 1080p LR
    ]
    for bh, bw in bench_shapes:
        profile = builder.create_optimization_profile()
        profile.set_shape(
            inp_name,
            min=(1, 12, max(64, bh - 64), max(64, bw - 64)),
            opt=(1, 12, bh, bw),
            max=(1, 12, min(1080, bh + 64), min(1920, bw + 64)),
        )
        config.add_optimization_profile(profile)

    print("  Building TRT INT8 engine (2-10 min on first run, fast if calib cache exists) ...")
    t0 = time.monotonic()
    engine_mem = builder.build_serialized_network(network, config)
    elapsed = time.monotonic() - t0
    if engine_mem is None:
        raise RuntimeError(
            "TRT engine build returned None. "
            "Check ONNX opset compatibility and available VRAM."
        )
    # TRT 10.x returns IHostMemory, not bytes.  Convert via numpy memoryview.
    engine_bytes = bytes(engine_mem)
    print(f"  Engine built in {elapsed:.1f}s  ({len(engine_bytes)/1024**2:.1f} MB)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(engine_bytes)
    print(f"  Engine serialized to: {output_path}")


# ---------------------------------------------------------------------------
# Engine inference wrapper — pycuda-free, uses PyTorch CUDA tensors.
# ---------------------------------------------------------------------------


class TRTEngine:
    """Inference wrapper around a serialized TRT INT8 engine.

    Supports single-profile (wide dynamic range) and multi-profile engines.
    For multi-profile engines built with 4 narrow profiles (one per benchmark
    resolution), selects the best matching profile for each (H, W).

    Uses PyTorch CUDA tensors for I/O buffers; no pycuda dependency.
    Thread-safety: NOT thread-safe (shares execution context + buffers).
    """

    def __init__(self, engine_path: Path) -> None:
        if not _TRT_OK:
            raise RuntimeError("tensorrt not installed.")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(str(engine_path), "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream()

        # Cache tensor names.
        self._inp_name: str = self._engine.get_tensor_name(0)
        self._out_name: str = self._engine.get_tensor_name(1)
        self._n_profiles: int = self._engine.num_optimization_profiles

        # Buffers are allocated on-demand in _ensure_buffers.
        self._d_input: torch.Tensor | None = None
        self._d_output: torch.Tensor | None = None
        self._buf_shape: tuple | None = None
        self._active_profile: int = 0

    def _select_profile(self, h: int, w: int) -> int:
        """Pick the optimization profile whose opt shape is closest to (h, w).

        For single-profile engines this always returns 0.  For the 4-profile
        engine built by _build_int8_engine, it returns the profile index whose
        opt (H, W) has the minimum L1 distance to the requested shape.
        """
        if self._n_profiles <= 1:
            return 0
        best_idx, best_dist = 0, float("inf")
        for i in range(self._n_profiles):
            shape = self._engine.get_tensor_profile_shape(
                self._inp_name, i
            )  # returns (min, opt, max) each as a tuple
            opt_h, opt_w = shape[1][2], shape[1][3]  # opt (N,C,H,W)
            dist = abs(opt_h - h) + abs(opt_w - w)
            if dist < best_dist:
                best_dist, best_idx = dist, i
        return best_idx

    def _ensure_buffers(self, n: int, c: int, h: int, w: int) -> None:
        h_out, w_out = h * 2, w * 2
        new_shape = (n, c, h, w)
        if self._buf_shape != new_shape:
            self._d_input = torch.empty((n, c, h, w), dtype=torch.float32, device="cuda")
            self._d_output = torch.empty((n, 3, h_out, w_out), dtype=torch.float32, device="cuda")
            self._buf_shape = new_shape

    def infer(self, x_np: np.ndarray) -> np.ndarray:
        """Run inference on a (1, 12, H, W) float32 numpy array.

        Returns (1, 3, H*2, W*2) float32 numpy array.
        """
        n, c, h, w = x_np.shape
        self._ensure_buffers(n, c, h, w)

        # Upload input.
        self._d_input.copy_(torch.from_numpy(x_np))

        # Switch optimization profile if needed (required before set_input_shape
        # when using multiple profiles).
        profile_idx = self._select_profile(h, w)
        if profile_idx != self._active_profile:
            self._context.set_optimization_profile_async(
                profile_idx, self._stream.cuda_stream
            )
            self._active_profile = profile_idx
            self._stream.synchronize()

        # Set dynamic shapes for this specific input resolution.
        self._context.set_input_shape(self._inp_name, (n, c, h, w))
        self._context.set_tensor_address(self._inp_name, self._d_input.data_ptr())
        self._context.set_tensor_address(self._out_name, self._d_output.data_ptr())

        with torch.cuda.stream(self._stream):
            ok = self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
        self._stream.synchronize()

        return self._d_output.cpu().numpy()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    import math
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    mse = max(mse, 1e-12)
    return float(-10.0 * math.log10(mse))


def _validate_vs_fp16(
    engine: TRTEngine,
    fp16_onnx_path: Path,
    h: int,
    w: int,
) -> float:
    """Compare INT8 output to FP16 ORT reference on random input. Returns PSNR."""
    rng = np.random.default_rng(seed=42)
    x_np = rng.random((1, 12, h, w), dtype=np.float32)

    # FP16 ORT reference (plain CUDA, not TRT EP, for speed).
    fp16_session = ort.InferenceSession(
        str(fp16_onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    inp_name = fp16_session.get_inputs()[0].name
    fp16_out = fp16_session.run(None, {inp_name: x_np})[0]

    # INT8 TRT inference.
    int8_out = engine.infer(x_np)

    psnr_val = _psnr(fp16_out, int8_out)
    max_diff = float(np.abs(fp16_out - int8_out).max())
    mean_diff = float(np.abs(fp16_out - int8_out).mean())
    print(f"  [INT8 vs FP16 ORT @ {h}x{w} LR]")
    print(f"    PSNR(INT8, FP16)  = {psnr_val:.2f} dB")
    print(f"    max|diff|         = {max_diff:.4f}")
    print(f"    mean|diff|        = {mean_diff:.6f}")
    quality = "PASS" if psnr_val >= 30.0 else "WARN: PSNR < 30 dB — INT8 quality may be marginal"
    print(f"    Status: {quality}")
    return psnr_val


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

RESOLUTIONS = [
    ("Steam Deck  (800x1280 LR -> 1600x2560)",  800, 1280),
    ("720p        (720x1280 LR -> 1440x2560)",  720, 1280),
    ("900p        (900x1600 LR -> 1800x3200)",  900, 1600),
    ("1080p / 4K  (1080x1920 LR -> 2160x3840)", 1080, 1920),
]

N_WARMUP = 3
N_RUNS = 5


def _bench_trt_engine(engine: TRTEngine, h: int, w: int) -> dict:
    x_np = np.zeros((1, 12, h, w), dtype=np.float32)
    for _ in range(N_WARMUP):
        engine.infer(x_np)
    t0 = time.monotonic()
    for _ in range(N_RUNS):
        out = engine.infer(x_np)
    elapsed = (time.monotonic() - t0) / N_RUNS
    return {"ms": elapsed * 1000, "fps": 1.0 / elapsed, "out_shape": out.shape}


def _bench_ort_fp16(fp16_onnx_path: Path, h: int, w: int) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    session = ort.InferenceSession(
        str(fp16_onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    inp_name = session.get_inputs()[0].name
    x_np = np.zeros((1, 12, h, w), dtype=np.float32)
    for _ in range(N_WARMUP):
        session.run(None, {inp_name: x_np})
    t0 = time.monotonic()
    for _ in range(N_RUNS):
        session.run(None, {inp_name: x_np})
    elapsed = (time.monotonic() - t0) / N_RUNS
    return {"ms": elapsed * 1000, "fps": 1.0 / elapsed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, default=Path(r"<train-host-data>\onnx\srcnn-prod-v3-fp32.onnx"),
                   help="Input FP32 ONNX file.")
    p.add_argument("--fp16-onnx", type=Path, default=Path(r"<train-host-data>\onnx\srcnn-prod-v3-fp16.onnx"),
                   help="FP16 ONNX file for validation baseline.")
    p.add_argument("--output", type=Path, default=Path(r"<train-host-data>\onnx\srcnn-prod-v3-int8.trt"),
                   help="Output TRT INT8 engine path.")
    p.add_argument("--dataset-root", type=Path, default=Path(r"<train-host-data>\datasets\srgd"),
                   help="SRGD dataset root.")
    p.add_argument("--calib-scene", type=str, default="ActionRPG",
                   help="SRGD scene for calibration (default: ActionRPG).")
    p.add_argument("--calib-samples", type=int, default=64,
                   help="Number of calibration frames (default: 64).")
    p.add_argument("--calib-h", type=int, default=720,
                   help="Calibration LR height (default: 720).")
    p.add_argument("--calib-w", type=int, default=1280,
                   help="Calibration LR width (default: 1280).")
    p.add_argument("--workspace-gib", type=float, default=2.0,
                   help="TRT builder workspace in GiB (default: 2.0).")
    p.add_argument("--bench", action="store_true",
                   help="Run speed benchmark at all 4 standard resolutions after export.")
    p.add_argument("--skip-build", action="store_true",
                   help="Skip engine build if --output already exists; go straight to bench/validate.")
    p.add_argument("--verbose", action="store_true",
                   help="Use TRT VERBOSE logger to expose detailed build errors.")
    args = p.parse_args()

    if not _TRT_OK:
        print("ERROR: tensorrt Python package not found.")
        print("Install with: pip install tensorrt (TRT 10.x wheel)")
        return 1

    print(f"TensorRT version: {trt.__version__}")
    print(f"Input ONNX:       {args.onnx}")
    print(f"Output engine:    {args.output}")

    if not args.onnx.exists():
        print(f"ERROR: {args.onnx} not found. Run sr_export_onnx.py first.")
        return 1

    # -----------------------------------------------------------------------
    # Build calibration array
    # -----------------------------------------------------------------------
    calib_cache_npy = (
        args.output.parent
        / f"calib_{args.calib_scene}_{args.calib_samples}x{args.calib_h}x{args.calib_w}.npy"
    )
    trt_calib_cache = (
        args.output.parent
        / f"calib_{args.calib_scene}_{args.calib_samples}x{args.calib_h}x{args.calib_w}.trt_calib"
    )

    print("\n=== Calibration data ===")
    calib_array = _build_calib_array(
        dataset_root=args.dataset_root,
        scene=args.calib_scene,
        n_samples=args.calib_samples,
        calib_h=args.calib_h,
        calib_w=args.calib_w,
        calib_cache=calib_cache_npy,
    )

    # -----------------------------------------------------------------------
    # Build INT8 engine
    # -----------------------------------------------------------------------
    if args.skip_build and args.output.exists():
        print(f"\n=== Skipping build (engine exists at {args.output}) ===")
    else:
        print("\n=== Building TRT INT8 engine ===")
        _build_int8_engine(
            onnx_path=args.onnx,
            output_path=args.output,
            calib_array=calib_array,
            calib_cache_path=trt_calib_cache,
            calib_h=args.calib_h,
            calib_w=args.calib_w,
            batch_size=1,
            workspace_gib=args.workspace_gib,
            also_fp16=True,
            verbose=args.verbose,
        )

    # -----------------------------------------------------------------------
    # Load engine and validate
    # -----------------------------------------------------------------------
    print("\n=== Loading engine ===")
    engine = TRTEngine(args.output)
    print(f"  Engine loaded from {args.output}")

    print("\n=== Validating INT8 output vs FP16 ORT reference ===")
    if args.fp16_onnx.exists():
        _validate_vs_fp16(engine, args.fp16_onnx, args.calib_h, args.calib_w)
    else:
        print(f"  SKIP: FP16 ONNX not found at {args.fp16_onnx}; skipping quality validation.")

    # -----------------------------------------------------------------------
    # Benchmark
    # -----------------------------------------------------------------------
    if not args.bench:
        print(f"\nDone. INT8 engine at: {args.output}")
        return 0

    print("\n=== Benchmark: TRT INT8 vs ORT FP16 CUDA ===")
    print(f"  {'Resolution':<45}  {'INT8 ms':>8}  {'FP16 ms':>8}  {'speedup':>8}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*8}")

    for label, h, w in RESOLUTIONS:
        try:
            int8_r = _bench_trt_engine(engine, h, w)
        except Exception as exc:
            print(f"  {label:<45}  INT8 ERROR: {exc}")
            continue

        try:
            fp16_r = _bench_ort_fp16(args.fp16_onnx, h, w)
            speedup_s = f"{fp16_r['ms'] / int8_r['ms']:.2f}x"
            fp16_ms_s = f"{fp16_r['ms']:8.2f}"
        except Exception as exc:
            speedup_s = "  N/A"
            fp16_ms_s = "    N/A "
            print(f"    ORT FP16 bench failed at {h}x{w}: {exc}")

        print(f"  {label:<45}  {int8_r['ms']:8.2f}  {fp16_ms_s}  {speedup_s:>8}")

    print(f"\nINT8 engine: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
