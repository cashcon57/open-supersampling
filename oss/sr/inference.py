"""Runtime-optimised inference engine for OSS-SR.

The training-time forward path is shaped for clarity and gradient flow.
The shipped inference path needs every speedup we can get without changing
the math:

- **FP16 weights + ops** (2x memory, 2-4x throughput on Tensor Cores).
- **Channels-last memory format** (1.5-2x on Tensor Cores via better
  memory access patterns).
- **CUDA Graphs** (eliminates kernel launch overhead — significant for
  small models like ours).
- **Pre-allocated input/output buffers** (one allocation, reused).
- **Lean checkpoints** (drop zero-input channels via
  ``scripts/sr_make_lean.py``).

This module is what a game-side integration loads. Use:

    engine = SRInferenceEngine.from_checkpoint(
        ckpt_path,
        target_lr_shape=(1080, 1920),  # (H, W) of LR input
        fp16=True,
        channels_last=True,
        cuda_graphs=True,
    )
    out_hr = engine(lr_tensor)  # (1, 3, 2160, 3840)

For the ONNX/TensorRT export path, see ``scripts/sr_export_onnx.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from oss.sr import build_sr_model


@dataclass
class _CudaGraphState:
    """Captured CUDA graph + its pre-allocated input/output buffers."""

    graph: "torch.cuda.CUDAGraph"
    static_input: torch.Tensor
    static_output: torch.Tensor


class SRInferenceEngine:
    """High-throughput inference for OSS-SR.

    Use ``from_checkpoint`` to construct.  Inference is single-batch;
    repeated calls reuse the same model + (optionally) the same captured
    CUDA graph.

    Notes:
        - When ``cuda_graphs=True``, the input shape is fixed at
          construction time. Calling with a differently-shaped input
          re-captures the graph automatically (cheap, but not free —
          prefer one engine per fixed resolution).
        - When ``channels_last=True``, the engine converts the model and
          input on demand. Output is contiguous in channels-first format
          regardless.
    """

    def __init__(
        self,
        sr_model: torch.nn.Module,
        device: str,
        fp16: bool,
        channels_last: bool,
        cuda_graphs: bool,
        in_channels: int,
        scale: int,
    ) -> None:
        self.device = device
        self.fp16 = fp16
        self.channels_last = channels_last
        self.cuda_graphs = cuda_graphs
        self.in_channels = in_channels
        self.scale = scale

        sr_model = sr_model.to(device).train(False)
        if fp16:
            sr_model = sr_model.half()
        if channels_last:
            sr_model = sr_model.to(memory_format=torch.channels_last)
        for p in sr_model.parameters():
            p.requires_grad_(False)
        self.model = sr_model

        self._dtype = torch.float16 if fp16 else torch.float32
        self._graph: Optional[_CudaGraphState] = None
        self._graph_shape: Optional[tuple[int, int, int, int]] = None

    # ---- construction --------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Path,
        device: str = "cuda",
        fp16: bool = True,
        channels_last: bool = True,
        cuda_graphs: bool = False,
        target_lr_shape: Optional[tuple[int, int]] = None,
    ) -> "SRInferenceEngine":
        """Load a trained SR-CNN checkpoint with all the inference speedups
        applied.

        Args:
            ckpt_path:        Path to ``step-XXXXX.pt`` (full or lean).
            device:           ``"cuda"`` or ``"cpu"``.
            fp16:             Convert weights + ops to FP16. Recommended.
            channels_last:    Use channels-last memory format. Recommended.
            cuda_graphs:      Capture a CUDA graph at first call. Massive
                              latency reduction for small models if the
                              input shape is fixed.
            target_lr_shape:  ``(H, W)`` of expected LR input. When set,
                              the engine warms and (if ``cuda_graphs``)
                              captures the graph here. Optional; if None,
                              the graph is captured on first call.
        """
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_args = ck.get("args", {})
        tier = saved_args.get("tier", "lite")
        sr_backbone = saved_args.get("sr_backbone", "simple")
        in_channels = int(ck.get("lean_in_channels", saved_args.get("lean_in_channels", 12)))
        scale = 2  # current architecture default

        factory_kind = "rrdb" if (sr_backbone == "rrdb") else "simple"
        sr_model = build_sr_model(
            model_kind=factory_kind, tier=tier, in_channels=in_channels, scale=scale
        )
        sr_model.load_state_dict(ck["sr_model"])

        engine = cls(
            sr_model=sr_model,
            device=device,
            fp16=fp16,
            channels_last=channels_last,
            cuda_graphs=cuda_graphs,
            in_channels=in_channels,
            scale=scale,
        )

        if target_lr_shape is not None:
            h, w = target_lr_shape
            engine.warm(h, w, n_warmup=3)

        return engine

    # ---- inference ----------------------------------------------------------

    def warm(self, lr_h: int, lr_w: int, n_warmup: int = 3) -> None:
        """Run a few warm-up forwards at the target shape and (if enabled)
        capture the CUDA graph. Call once before timed inference."""
        x = torch.zeros(1, self.in_channels, lr_h, lr_w, device=self.device, dtype=self._dtype)
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = self.model(x)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)

        if self.cuda_graphs and self.device.startswith("cuda"):
            self._capture_graph(lr_h, lr_w)

    def _capture_graph(self, lr_h: int, lr_w: int) -> None:
        """Capture a CUDA graph for the given input shape."""
        static_input = torch.zeros(
            1, self.in_channels, lr_h, lr_w, device=self.device, dtype=self._dtype
        )
        if self.channels_last:
            static_input = static_input.to(memory_format=torch.channels_last)

        # Run once outside the graph context to allocate output + caches.
        with torch.no_grad():
            _ = self.model(static_input)
        torch.cuda.synchronize(self.device)

        # Now capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.no_grad():
                static_output = self.model(static_input)

        self._graph = _CudaGraphState(
            graph=graph,
            static_input=static_input,
            static_output=static_output,
        )
        self._graph_shape = tuple(static_input.shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on a single LR input.

        Args:
            x: (1, C, H, W) LR + G-buffer stack. C must equal
               ``self.in_channels``. Caller can pass FP32 — we convert.

        Returns:
            (1, 3, scale*H, scale*W) HR output. FP32 contiguous.
        """
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(f"expected (1, C, H, W); got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels; got {x.shape[1]}"
            )
        x = x.to(self.device, dtype=self._dtype, non_blocking=True)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        if self.cuda_graphs and self._graph is not None and tuple(x.shape) == self._graph_shape:
            # Replay path: copy input into the captured buffer, replay graph.
            self._graph.static_input.copy_(x)
            self._graph.graph.replay()
            out = self._graph.static_output
        else:
            with torch.no_grad():
                out = self.model(x)

        # Always return FP32 contiguous for downstream code.
        return out.float().contiguous()


__all__ = ["SRInferenceEngine"]


# ============================================================================
# v5 pixel-temporal stateful inference
# ============================================================================

from oss.sr.temporal import TemporalSRModel, make_first_frame_prev_hr


class TemporalSRInferenceEngine:
    """Stateful inference engine for v5 pixel-temporal SR.

    Carries ``prev_hr_output`` and ``prev_depth_hr`` across calls. Auto-resets
    when mean motion magnitude exceeds ``scene_cut_motion_threshold`` (in LR
    pixels).
    """

    def __init__(
        self,
        model: TemporalSRModel,
        device: str,
        fp16: bool,
        scene_cut_motion_threshold: float,
    ) -> None:
        self.device = device
        self.fp16 = fp16
        self._dtype = torch.float16 if fp16 else torch.float32
        self.scene_cut_motion_threshold = float(scene_cut_motion_threshold)
        self.last_call_was_scene_cut = False

        model = model.to(device).train(False)
        if fp16:
            model = model.half()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        self._prev_hr: Optional[torch.Tensor] = None
        self._prev_depth_hr: Optional[torch.Tensor] = None

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Path,
        device: str = "cuda",
        fp16: bool = True,
        scene_cut_motion_threshold: float = 32.0,
    ) -> "TemporalSRInferenceEngine":
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ck.get("args", {})
        in_channels = int(saved.get("in_channels", 12))
        scale = int(saved.get("scale", 2))
        tier = saved.get("tier", "standard")
        backbone_kind = "rrdb" if saved.get("sr_backbone") == "rrdb" else "simple"
        model = TemporalSRModel(
            in_channels=in_channels, scale=scale, tier=tier, backbone_kind=backbone_kind
        )
        model.load_state_dict(ck["temporal_model"])
        return cls(model=model, device=device, fp16=fp16,
                   scene_cut_motion_threshold=scene_cut_motion_threshold)

    def reset(self) -> None:
        self._prev_hr = None
        self._prev_depth_hr = None

    def __call__(
        self,
        lr_inputs: torch.Tensor,
        depth_hr_curr: torch.Tensor,
        motion_lr: torch.Tensor,
    ) -> torch.Tensor:
        lr_inputs = lr_inputs.to(self.device, dtype=self._dtype, non_blocking=True)
        depth_hr_curr = depth_hr_curr.to(self.device, dtype=self._dtype, non_blocking=True)
        motion_lr = motion_lr.to(self.device, dtype=self._dtype, non_blocking=True)

        # Scene-cut detection.
        mean_mag = float(motion_lr.norm(dim=1).mean().item())
        self.last_call_was_scene_cut = (
            self._prev_hr is not None and mean_mag > self.scene_cut_motion_threshold
        )
        if self.last_call_was_scene_cut:
            self.reset()

        # First-frame init or use stored state.
        if self._prev_hr is None:
            prev_hr = make_first_frame_prev_hr(lr_inputs[:, :3], scale=self.model.scale)
            prev_depth = depth_hr_curr
        else:
            prev_hr = self._prev_hr
            prev_depth = self._prev_depth_hr

        with torch.no_grad():
            out = self.model(
                lr_inputs=lr_inputs,
                prev_hr=prev_hr,
                depth_hr_curr=depth_hr_curr,
                depth_hr_prev=prev_depth,
                motion_lr=motion_lr,
            )

        # Persist state for next call (detached, fp16 if engine fp16).
        self._prev_hr = out.detach()
        self._prev_depth_hr = depth_hr_curr.detach()
        return out.float().contiguous()


__all__ = list(set(__all__) | {"TemporalSRInferenceEngine"})  # type: ignore[name-defined]
