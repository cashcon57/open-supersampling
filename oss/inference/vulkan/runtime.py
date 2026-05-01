"""Vulkan compute inference runtime for OSS-Pico, via NCNN + PNNX.

This is the v0.2-alpha scaffold. It picks NCNN as the runtime (BSD-3-Clause,
first-class Vulkan backend, mobile/RDNA-2 tuned) and PNNX as the PyTorch ->
NCNN exporter. See ``docs/research/2026-04-30-vulkan-runtime-eval.md`` for
the full evaluation.

Public surface
--------------

``run_pico_vulkan(color_lr, depth_lr, motion_lr, normals_lr, history_hr,
hidden_state, *, model=None, runtime=None) -> (rgb_hr, new_hidden_state)``

  Functional one-shot wrapper. Builds a runtime on demand from ``model`` (a
  ``OSSPico`` instance in inference mode) the first time it is called for a
  given model identity, caches the converted NCNN files in the user cache,
  and runs a single forward pass returning numpy arrays. If a
  ``VulkanPicoRuntime`` is passed in via ``runtime=``, that is used directly
  and ``model=`` is ignored.

``VulkanPicoRuntime``

  Persistent runtime: holds the loaded ``ncnn.Net``, the workdir of the
  converted artifacts, and a thin ``forward()`` method matching the same
  signature. Use this for sequence inference where we want to keep the model
  resident across many frames.

The runtime is a *scaffold*. It does not yet ship custom HLSL/SPIR-V kernels;
it leans on NCNN's own auto-generated SPIR-V for the ops in the Pico graph.
Future work (v0.2-beta) is to drop in hand-tuned kernels for the kernel-
prediction head and the recurrent cell where they bottleneck.

NCNN will use Vulkan whenever a Vulkan ICD is present (Mesa RADV on Steam
Deck, MoltenVK on macOS, lavapipe on a software-only host). Otherwise it
silently falls back to its CPU codepath, which is still suitable for the
parity test we care about here.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime availability probes
# ---------------------------------------------------------------------------


def _ncnn_available() -> bool:
    try:
        import ncnn  # noqa: F401
        return True
    except Exception:
        return False


def _pnnx_available() -> bool:
    try:
        import pnnx  # noqa: F401
        return True
    except Exception:
        return False


def vulkan_available() -> bool:
    """True if NCNN can see at least one Vulkan-capable GPU.

    Returns False on macOS dev boxes without MoltenVK installed and on
    headless CI without a software ICD. The runtime still works in that
    case (NCNN CPU fallback); this probe is mainly for diagnostics and
    pytest skip-marker decisions.
    """
    if not _ncnn_available():
        return False
    try:
        import ncnn
        return ncnn.get_gpu_count() > 0
    except Exception:
        return False


def runtime_available() -> bool:
    """True iff we can build *and* run a Pico graph here.

    Both ncnn (for execution) and pnnx (for the exporter) must be importable.
    Vulkan itself is not required -- NCNN's CPU path is a valid fallback for
    the parity scaffold.
    """
    return _ncnn_available() and _pnnx_available()


def _should_use_vulkan(prefer_vulkan: bool) -> bool:
    if not prefer_vulkan:
        return False
    if os.environ.get("ORS_DISABLE_NCNN_VULKAN") == "1":
        return False
    # MoltenVK is optional on macOS and the current NCNN path is not stable on
    # all dev boxes, so default to the documented CPU fallback unless the user
    # explicitly opts in.
    if sys.platform == "darwin" and os.environ.get("ORS_ENABLE_NCNN_VULKAN") != "1":
        return False
    return vulkan_available()


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


def _default_cache_root() -> Path:
    base = os.environ.get("ORS_VULKAN_CACHE")
    if base:
        return Path(base)
    return Path.home() / ".cache" / "ors" / "vulkan"


def _model_fingerprint(model: nn.Module, input_shapes: Tuple[Tuple[int, ...], ...]) -> str:
    """Hash a model's state_dict + traced input shapes into a stable cache key.

    The same weights + same export shapes -> same fingerprint -> we can reuse
    the converted NCNN files without re-running PNNX.
    """
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous().numpy()
        h.update(k.encode())
        h.update(str(t.shape).encode())
        h.update(str(t.dtype).encode())
        h.update(t.tobytes())
    for s in input_shapes:
        h.update(str(s).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Conversion (PyTorch -> NCNN .param + .bin via PNNX)
# ---------------------------------------------------------------------------


@dataclass
class _ConvertedArtifacts:
    """Filesystem paths to a converted NCNN model + the export shapes used."""
    workdir: Path
    param: Path
    bin: Path
    # Shapes the model was traced at -- callers must reuse these (PNNX bakes
    # spatial dims into the graph).
    input_shapes: Tuple[Tuple[int, ...], ...]


def _convert_with_pnnx(
    model: nn.Module,
    input_shapes: Tuple[Tuple[int, ...], ...],
    workdir: Path,
) -> _ConvertedArtifacts:
    """Run PNNX on ``model`` to emit ``pico.ncnn.{param,bin}`` in ``workdir``.

    PNNX is invoked via its Python wrapper, which under the hood does:
    ``torch.jit.trace(model, dummy_inputs) -> .pt -> pnnx CLI -> NCNN``.
    """
    import pnnx  # local import; gated by runtime_available()

    workdir.mkdir(parents=True, exist_ok=True)

    dummy_inputs = tuple(torch.zeros(*s) for s in input_shapes)

    # PNNX writes its outputs in the *current working directory*. We chdir
    # into ``workdir`` for the duration of the call (and restore after) so
    # the artifacts land where the caller expects, regardless of where the
    # process was launched from.
    prev_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        # ``check_trace=False`` avoids PNNX's internal numerical comparison
        # which can be noisy with our recurrent graph. We do our own parity
        # check at the end-to-end level in tests.
        pnnx.export(
            model,
            "pico.pt",
            inputs=dummy_inputs,
            fp16=False,
            check_trace=False,
        )
    finally:
        os.chdir(prev_cwd)

    param = workdir / "pico.ncnn.param"
    bin_ = workdir / "pico.ncnn.bin"
    if not param.exists() or not bin_.exists():
        raise RuntimeError(
            f"PNNX did not emit NCNN artifacts in {workdir}. "
            f"Got: {sorted(p.name for p in workdir.iterdir())}"
        )
    return _ConvertedArtifacts(
        workdir=workdir, param=param, bin=bin_, input_shapes=input_shapes
    )


# ---------------------------------------------------------------------------
# Persistent runtime
# ---------------------------------------------------------------------------


class VulkanPicoRuntime:
    """A loaded OSS-Pico graph ready to run forward passes via NCNN.

    Construct with either:

    - ``VulkanPicoRuntime.from_model(model, input_shapes)`` -- traces a
      PyTorch ``OSSPico`` and converts it via PNNX, caching the converted
      NCNN files under ``$ORS_VULKAN_CACHE`` (default ``~/.cache/ors/vulkan``).

    - ``VulkanPicoRuntime.from_artifacts(param_path, bin_path, input_shapes)``
      -- loads pre-converted NCNN artifacts directly. Use this when an
      offline export pipeline has already produced ``pico.ncnn.{param,bin}``.

    The runtime is *not* thread-safe -- create one per worker thread. NCNN's
    own ``Extractor`` is per-call so concurrent calls within one runtime would
    need extra synchronisation; for v0.2-alpha we serialise with an internal
    lock and accept the contention.
    """

    # PNNX emits these blob names in the order it traced the forward
    # signature. They are part of the ABI between this runtime and the
    # converted artifacts.
    _INPUT_NAMES = ("in0", "in1", "in2", "in3", "in4", "in5", "in6")
    _OUTPUT_NAMES = ("out0", "out1")

    def __init__(self, artifacts: _ConvertedArtifacts, *, prefer_vulkan: bool = True):
        if not _ncnn_available():
            raise RuntimeError(
                "ncnn is not importable. `pip install ncnn` (or install the "
                "ors[vulkan] extra)."
            )
        import ncnn

        self._artifacts = artifacts
        self._lock = threading.Lock()
        self._net = ncnn.Net()
        # NCNN flips this on automatically when a Vulkan ICD is visible.
        # We set the flag explicitly so behaviour is deterministic.
        self._net.opt.use_vulkan_compute = _should_use_vulkan(prefer_vulkan)
        # Threaded CPU path defaults are fine; expose if we need them later.
        self._net.load_param(str(artifacts.param))
        self._net.load_model(str(artifacts.bin))
        log.info(
            "VulkanPicoRuntime loaded (vulkan=%s, gpu_count=%d, workdir=%s)",
            self._net.opt.use_vulkan_compute,
            self._gpu_count(),
            artifacts.workdir,
        )

    # --- factories -------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        input_shapes: Tuple[Tuple[int, ...], ...],
        *,
        cache_root: Optional[Path] = None,
        prefer_vulkan: bool = True,
    ) -> "VulkanPicoRuntime":
        if not runtime_available():
            raise RuntimeError(
                "ncnn and/or pnnx not importable; install with "
                "`pip install ncnn pnnx` (or `ors[vulkan]`)."
            )
        cache_root = Path(cache_root) if cache_root is not None else _default_cache_root()
        fp = _model_fingerprint(model, input_shapes)
        workdir = cache_root / f"pico-{fp}"
        param = workdir / "pico.ncnn.param"
        bin_ = workdir / "pico.ncnn.bin"
        if param.exists() and bin_.exists():
            log.info("reusing cached NCNN artifacts at %s", workdir)
            artifacts = _ConvertedArtifacts(
                workdir=workdir, param=param, bin=bin_, input_shapes=input_shapes
            )
        else:
            log.info("converting OSS-Pico via PNNX -> %s", workdir)
            # PNNX expects the model to be in inference mode; we set it here
            # without mutating the caller's state by using a transient toggle.
            was_training = model.training
            try:
                model.train(False)
                artifacts = _convert_with_pnnx(model, input_shapes, workdir)
            finally:
                model.train(was_training)
        return cls(artifacts, prefer_vulkan=prefer_vulkan)

    @classmethod
    def from_artifacts(
        cls,
        param_path: Path,
        bin_path: Path,
        input_shapes: Tuple[Tuple[int, ...], ...],
        *,
        prefer_vulkan: bool = True,
    ) -> "VulkanPicoRuntime":
        artifacts = _ConvertedArtifacts(
            workdir=Path(param_path).parent,
            param=Path(param_path),
            bin=Path(bin_path),
            input_shapes=input_shapes,
        )
        return cls(artifacts, prefer_vulkan=prefer_vulkan)

    # --- introspection ---------------------------------------------------

    def _gpu_count(self) -> int:
        try:
            import ncnn
            return int(ncnn.get_gpu_count())
        except Exception:
            return 0

    @property
    def using_vulkan(self) -> bool:
        return bool(self._net.opt.use_vulkan_compute)

    @property
    def workdir(self) -> Path:
        return self._artifacts.workdir

    # --- inference -------------------------------------------------------

    def forward(
        self,
        color_lr: np.ndarray,
        depth_lr: np.ndarray,
        motion_lr: np.ndarray,
        normals_lr: np.ndarray,
        albedo_lr: np.ndarray,
        history_hr: np.ndarray,
        hidden_state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one forward pass. Inputs are NCHW float32 numpy arrays with B=1.

        NCNN's ``Mat`` is CHW (no batch). We squeeze the leading batch
        dimension on the way in and re-add it on the way out, matching the
        contract that the rest of the codebase uses (PyTorch NCHW).
        """
        import ncnn

        inputs = (color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_state)
        for i, (arr, expected) in enumerate(zip(inputs, self._artifacts.input_shapes)):
            if arr.shape != expected:
                raise ValueError(
                    f"input {i} shape mismatch: got {arr.shape}, "
                    f"runtime was traced at {expected}. PNNX bakes spatial "
                    f"dims into the graph; export a new runtime for new shapes."
                )
            if arr.dtype != np.float32:
                raise ValueError(
                    f"input {i} dtype must be float32, got {arr.dtype}"
                )

        with self._lock:
            with self._net.create_extractor() as ex:
                for name, arr in zip(self._INPUT_NAMES, inputs):
                    # Drop the batch dim -- NCNN Mats are CHW.
                    chw = np.ascontiguousarray(arr[0])
                    ex.input(name, ncnn.Mat(chw).clone())
                _, mat0 = ex.extract(self._OUTPUT_NAMES[0])
                _, mat1 = ex.extract(self._OUTPUT_NAMES[1])
                # NCNN mats may reference extractor-owned memory; copy them
                # before the extractor context exits.
                rgb_hr = np.array(mat0.numpy(), copy=True)[None, ...].astype(np.float32, copy=False)
                new_hidden = np.array(mat1.numpy(), copy=True)[None, ...].astype(np.float32, copy=False)
        return rgb_hr, new_hidden


# ---------------------------------------------------------------------------
# Functional one-shot API
# ---------------------------------------------------------------------------


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(arr)


def run_pico_vulkan(
    color_lr,
    depth_lr,
    motion_lr,
    normals_lr,
    albedo_lr,
    history_hr,
    hidden_state,
    *,
    model: Optional[nn.Module] = None,
    runtime: Optional[VulkanPicoRuntime] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run OSS-Pico under Vulkan/NCNN. Same signature as ``OSSPico.forward``.

    Either ``runtime=`` (a pre-built ``VulkanPicoRuntime``) or ``model=`` (a
    PyTorch ``OSSPico`` we should convert on demand) must be provided. When
    converting on demand, this function caches the converted artifacts so
    repeated calls with the same weights are cheap.

    Inputs may be torch tensors *or* numpy arrays. Outputs are numpy float32
    NCHW with batch=1.
    """
    if runtime is None and model is None:
        raise ValueError("run_pico_vulkan requires either model= or runtime=")

    arrs = tuple(
        _to_numpy(x)
        for x in (color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_state)
    )

    if runtime is None:
        shapes = tuple(a.shape for a in arrs)
        runtime = VulkanPicoRuntime.from_model(model, shapes)
    return runtime.forward(*arrs)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _ephemeral_workdir() -> Path:
    """Convenience for tests: a per-process temp dir for converted artifacts."""
    return Path(tempfile.mkdtemp(prefix="ors-vulkan-"))
