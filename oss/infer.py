"""Unified inference session with automatic backend selection.

Backend priority order (highest to lowest):
  NVIDIA GPU  -> TensorRT engine (build + cache FP16) -> ORT CUDA EP
  AMD Linux   -> MIGraphX -> ORT ROCm EP -> NCNN Vulkan
  AMD Windows -> ORT DirectML EP
  Apple       -> CoreML (compute_units=ALL, loads .mlpackage)
  Intel       -> ORT OpenVINO EP
  fallback    -> NCNN Vulkan -> ORT CPU

Usage
-----
    from oss.infer import InferenceSession
    sess = InferenceSession("model.onnx")          # auto-detect
    out  = sess.run({"color": arr, "depth": arr2}) # dict[str, np.ndarray]
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------

def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _nvidia_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _tensorrt_available() -> bool:
    return _has_module("tensorrt")


def _ort_ep_available(ep: str) -> bool:
    """Check if an ORT Execution Provider is in the installed runtime."""
    try:
        import onnxruntime as ort
        return ep in ort.get_available_providers()
    except Exception:
        return False


def _migraphx_available() -> bool:
    if _has_module("migraphx"):
        return True
    return Path("/opt/rocm/lib/libmigraphx_c.so").exists()


def _amd_gpu_linux() -> bool:
    """True when we're on Linux and an AMD GPU is likely present via ROCm."""
    if sys.platform != "linux":
        return False
    # ROCm runtime drops /opt/rocm when installed.
    return Path("/opt/rocm").is_dir()


def _directml_available() -> bool:
    return _ort_ep_available("DmlExecutionProvider")


def _apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _coreml_available() -> bool:
    return _has_module("coremltools")


def _openvino_available() -> bool:
    return _ort_ep_available("OpenVINOExecutionProvider")


def _ncnn_available() -> bool:
    return _has_module("ncnn") and _has_module("pnnx")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _select_backend(model_path: Path) -> str:
    """Return a backend identifier string for the given model file.

    The model_path extension is used to detect CoreML (.mlpackage) vs ONNX.
    """
    if model_path.suffix == ".mlpackage":
        return "coreml"

    if _nvidia_gpu():
        if _tensorrt_available():
            return "tensorrt"
        if _ort_ep_available("CUDAExecutionProvider"):
            return "ort_cuda"

    if _amd_gpu_linux():
        if _migraphx_available():
            return "migraphx"
        if _ort_ep_available("ROCMExecutionProvider"):
            return "ort_rocm"
        if _ncnn_available():
            return "ncnn_vulkan"

    if _directml_available():
        return "directml"

    if _apple_silicon() and _coreml_available():
        return "coreml_ort"

    if _openvino_available():
        return "openvino"

    if _ncnn_available():
        return "ncnn_vulkan"

    return "ort_cpu"


# ---------------------------------------------------------------------------
# TensorRT engine builder
# ---------------------------------------------------------------------------

def _model_hash(model_path: Path) -> str:
    h = hashlib.sha256()
    h.update(model_path.read_bytes())
    return h.hexdigest()[:16]


def _gpu_arch() -> str:
    """Return a short CUDA device arch string, e.g. 'sm_89'."""
    try:
        import torch
        cap = torch.cuda.get_device_capability()
        return f"sm_{cap[0]}{cap[1]}"
    except Exception:
        return "unknown"


def _trt_cache_path(model_path: Path) -> Path:
    cache_dir = Path.home() / ".cache" / "oss" / "engines"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"{_model_hash(model_path)}_{_gpu_arch()}.trt"
    return cache_dir / name


def _build_trt_engine(onnx_path: Path) -> bytes:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"TRT ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    # Dynamic shapes: allow any batch/height/width within generous bounds.
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = inp.shape
        # Replace -1 (dynamic) dims with concrete min/opt/max.
        min_shape = tuple(1 if d < 0 else d for d in shape)
        opt_shape = tuple(4 if d < 0 else d for d in shape)
        max_shape = tuple(16 if d < 0 else d for d in shape)
        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build returned None")
    return bytes(serialized)


def _load_trt_session(model_path: Path):
    """Return a TRT runtime + context pair, building and caching the engine."""
    import tensorrt as trt

    cache = _trt_cache_path(model_path)
    if cache.exists():
        log.info("TRT cache hit: %s", cache)
        engine_bytes = cache.read_bytes()
    else:
        log.info("Building TRT engine from %s ...", model_path)
        engine_bytes = _build_trt_engine(model_path)
        cache.write_bytes(engine_bytes)
        log.info("TRT engine cached at %s", cache)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine  = runtime.deserialize_cuda_engine(engine_bytes)
    return engine


# ---------------------------------------------------------------------------
# Backend runners
# (Each returns a callable: inputs_dict -> np.ndarray or list of np.ndarray)
# ---------------------------------------------------------------------------

def _make_ort_runner(model_path: Path, providers: list[str]):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=providers)
    output_names = [o.name for o in sess.get_outputs()]

    def run(inputs: dict[str, np.ndarray]) -> np.ndarray:
        results = sess.run(output_names, inputs)
        return results[0] if len(results) == 1 else results

    return run


def _make_trt_runner(model_path: Path):
    import tensorrt as trt
    import torch

    engine = _load_trt_session(model_path)
    context = engine.create_execution_context()

    def run(inputs: dict[str, np.ndarray]) -> np.ndarray:
        bindings = []
        device_bufs = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                arr = inputs[name]
                t = torch.as_tensor(arr).cuda().contiguous()
                device_bufs[name] = t
                context.set_tensor_address(name, t.data_ptr())
            else:
                shape = context.get_tensor_shape(name)
                t = torch.empty(tuple(shape), dtype=torch.float32, device="cuda")
                device_bufs[name] = t
                context.set_tensor_address(name, t.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        context.execute_async_v3(stream)
        torch.cuda.synchronize()

        output_names = [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]
        results = [device_bufs[n].cpu().numpy() for n in output_names]
        return results[0] if len(results) == 1 else results

    return run


def _make_migraphx_runner(model_path: Path):
    import migraphx

    prog = migraphx.parse_onnx(str(model_path))
    prog.compile(migraphx.get_target("gpu"))
    params = prog.get_parameter_names()

    def run(inputs: dict[str, np.ndarray]) -> np.ndarray:
        mg_inputs = {k: migraphx.argument(inputs[k]) for k in params if k in inputs}
        results = prog.run(mg_inputs)
        out = [np.array(r) for r in results]
        return out[0] if len(out) == 1 else out

    return run


def _make_coreml_runner(model_path: Path):
    """Load a .mlpackage directly (produced by export_oss.py --format coreml)."""
    import coremltools as ct

    mlmodel = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.ALL)
    output_names = [o.name for o in mlmodel.get_spec().description.output]

    def run(inputs: dict[str, np.ndarray]) -> np.ndarray:
        preds = mlmodel.predict(inputs)
        results = [preds[n] for n in output_names]
        return results[0] if len(results) == 1 else results

    return run


def _make_coreml_ort_runner(model_path: Path):
    """CoreML via ORT CoreMLExecutionProvider (ONNX model, not .mlpackage)."""
    return _make_ort_runner(model_path, ["CoreMLExecutionProvider", "CPUExecutionProvider"])


def _make_ncnn_runner(model_path: Path):
    """NCNN Vulkan via the existing VulkanPicoRuntime scaffold.

    The NCNN path requires a pre-converted .param/.bin pair alongside the
    model_path stem (model_path.with_suffix('.ncnn.param') etc.). If the
    converted artifacts don't exist the runner falls back to ORT CPU.
    """
    param = model_path.with_suffix(".ncnn.param")
    bin_  = model_path.with_suffix(".ncnn.bin")
    if not (param.exists() and bin_.exists()):
        log.warning(
            "NCNN artifacts not found for %s; falling back to ORT CPU. "
            "Run pnnx to convert first.", model_path
        )
        return _make_ort_runner(model_path.with_suffix(".onnx"), ["CPUExecutionProvider"])

    from oss.inference.vulkan.runtime import VulkanPicoRuntime
    runtime = VulkanPicoRuntime.from_artifacts(param, bin_, input_shapes=())

    def run(inputs: dict[str, np.ndarray]) -> np.ndarray:
        ordered = list(inputs.values())
        results = runtime.forward(*ordered)
        return results[0] if isinstance(results, (list, tuple)) and len(results) == 1 else results

    return run


# ---------------------------------------------------------------------------
# InferenceSession
# ---------------------------------------------------------------------------

_BACKEND_BUILDERS = {
    "tensorrt":   _make_trt_runner,
    "ort_cuda":   lambda p: _make_ort_runner(p, ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    "migraphx":   _make_migraphx_runner,
    "ort_rocm":   lambda p: _make_ort_runner(p, ["ROCMExecutionProvider", "CPUExecutionProvider"]),
    "ncnn_vulkan": _make_ncnn_runner,
    "directml":   lambda p: _make_ort_runner(p, ["DmlExecutionProvider", "CPUExecutionProvider"]),
    "coreml":     _make_coreml_runner,
    "coreml_ort": _make_coreml_ort_runner,
    "openvino":   lambda p: _make_ort_runner(p, ["OpenVINOExecutionProvider", "CPUExecutionProvider"]),
    "ort_cpu":    lambda p: _make_ort_runner(p, ["CPUExecutionProvider"]),
}


class InferenceSession:
    """Backend-agnostic inference session for any OSS ONNX / CoreML model.

    Args:
        model_path:  Path to .onnx or .mlpackage file.
        backend:     Force a specific backend key (see _BACKEND_BUILDERS).
                     When None (default), the best available backend is
                     auto-detected.
    """

    def __init__(
        self,
        model_path: str | Path,
        backend: Optional[str] = None,
    ) -> None:
        self._path = Path(model_path)
        self._backend = backend or _select_backend(self._path)
        log.info("InferenceSession: model=%s backend=%s", self._path.name, self._backend)

        builder = _BACKEND_BUILDERS.get(self._backend)
        if builder is None:
            raise ValueError(
                f"Unknown backend {self._backend!r}; "
                f"choices: {list(_BACKEND_BUILDERS)}"
            )
        self._runner = builder(self._path)

    @property
    def backend(self) -> str:
        return self._backend

    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Run inference.

        Args:
            inputs: Dict mapping input names to float32 NCHW numpy arrays.

        Returns:
            Single output array, or a list of arrays for multi-output models.
        """
        return self._runner(inputs)
