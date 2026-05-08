#!/usr/bin/env python
"""End-to-end v6.1 production-shape CUDA benchmark.

Measures the full v6.1 forward shape as explicit stages so Phase 4 perf work
can track both total latency and the main architectural components.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from oss.sr.v6.model import CanvasState, V6Config, V6Model


DEFAULT_OUT = ROOT / "docs/coordination/bench-baseline-v6-e2e.json"
COMPONENTS = (
    "hat_tiny_lr_forward",
    "cross_attention_fusion",
    "spawner_write_back",
    "rasterizer_composite_head",
)
SHAPES = {
    "smoke": {"lr_h": 64, "lr_w": 96, "scale": 2, "canvas_count": 16_384},
    "prod": {"lr_h": 1080, "lr_w": 1920, "scale": 2, "canvas_count": 16_384},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return float("nan")
    ordered = sorted(float(x) for x in samples)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "median_ms": float(statistics.median(samples)) if samples else float("nan"),
        "p90_ms": _percentile(samples, 0.90),
        "min_ms": min(samples) if samples else float("nan"),
        "max_ms": max(samples) if samples else float("nan"),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return out, (time.perf_counter() - start) * 1000.0


class _NvtxRange:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, exc_type, exc, tb) -> None:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def _clone_canvas(canvas: CanvasState) -> CanvasState:
    return CanvasState(
        positions=canvas.positions.clone(),
        scales=canvas.scales.clone(),
        rotations=canvas.rotations.clone(),
        opacities=canvas.opacities.clone(),
        colors=canvas.colors.clone(),
        count=int(canvas.count),
    )


def _detach_canvas(canvas: CanvasState) -> CanvasState:
    return CanvasState(
        positions=canvas.positions.detach(),
        scales=canvas.scales.detach(),
        rotations=canvas.rotations.detach(),
        opacities=canvas.opacities.detach(),
        colors=canvas.colors.detach(),
        count=int(canvas.count),
    )


def _synthetic_canvas(
    *,
    count: int,
    token_dim: int,
    output_hw: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> CanvasState:
    gen = torch.Generator(device=device).manual_seed(0xA61)
    h, w = int(output_hw[0]), int(output_hw[1])
    positions = torch.rand((count, 2), generator=gen, device=device, dtype=torch.float32)
    positions[:, 0].mul_(float(w))
    positions[:, 1].mul_(float(h))
    base_scale = max(2.0, min(h, w) / 540.0)
    scales = torch.rand((count, 2), generator=gen, device=device, dtype=torch.float32)
    scales = scales.mul_(base_scale * 3.0).add_(base_scale)
    rotations = torch.rand((count,), generator=gen, device=device, dtype=torch.float32)
    rotations = rotations.mul_(2.0 * math.pi).sub_(math.pi)
    opacities = torch.ones((count,), device=device, dtype=torch.float32)
    colors = torch.randn((count, token_dim), generator=gen, device=device, dtype=torch.float32)
    return CanvasState(
        positions=positions.to(dtype=dtype),
        scales=scales.to(dtype=dtype),
        rotations=rotations.to(dtype=dtype),
        opacities=opacities.to(dtype=dtype),
        colors=colors.to(dtype=dtype),
        count=int(count),
    )


def _canvas_from_checkpoint(
    ckpt_path: Path,
    *,
    token_dim: int,
    output_hw: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> CanvasState | None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    candidates: list[Any] = []
    if isinstance(ckpt, dict):
        for key in ("canvas", "canvas_state", "_canvas_state", "spawner_state", "gaussian_state"):
            if key in ckpt:
                candidates.append(ckpt[key])
    else:
        candidates.append(ckpt)

    for state in candidates:
        if isinstance(state, CanvasState):
            canvas = state
        elif isinstance(state, dict):
            def first_present(*keys: str) -> Any:
                for key in keys:
                    if key in state:
                        return state[key]
                return None

            pos = first_present("positions", "xy")
            scales = first_present("scales", "scale")
            rotations = first_present("rotations", "rot")
            colors = first_present("colors", "features", "feat")
            opacities = first_present("opacities", "confidence")
            if pos is None or scales is None or rotations is None or colors is None:
                continue
            if opacities is None:
                opacities = torch.ones(pos.shape[0], device=device, dtype=pos.dtype)
            canvas = CanvasState(
                positions=pos,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
                colors=colors,
                count=int(state.get("count", pos.shape[0])),
            )
        else:
            continue

        count = int(canvas.count)
        colors = canvas.colors
        if colors.shape[-1] < token_dim:
            colors = F.pad(colors, (0, token_dim - colors.shape[-1]))
        elif colors.shape[-1] > token_dim:
            colors = colors[..., :token_dim]
        h, w = output_hw
        positions = canvas.positions[:count].to(device=device, dtype=torch.float32)
        positions[:, 0].clamp_(0.0, float(w) - 1.0e-4)
        positions[:, 1].clamp_(0.0, float(h) - 1.0e-4)
        return CanvasState(
            positions=positions.to(dtype=dtype),
            scales=canvas.scales[:count].to(device=device, dtype=dtype),
            rotations=canvas.rotations[:count].to(device=device, dtype=dtype),
            opacities=canvas.opacities[:count].to(device=device, dtype=dtype),
            colors=colors[:count].to(device=device, dtype=dtype),
            count=count,
        )
    return None


def _load_model_weights(model: V6Model, ckpt_path: Path) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return {"loaded": False, "reason": "checkpoint is not a dict"}
    load_info: dict[str, Any] = {"loaded": False}
    state = ckpt.get("generator") or ckpt.get("model") or ckpt.get("state_dict")
    if isinstance(state, dict):
        result = model.load_state_dict(state, strict=False)
        load_info.update(
            {
                "loaded": True,
                "source": "generator/model/state_dict",
                "missing": list(result.missing_keys),
                "unexpected": list(result.unexpected_keys),
            }
        )

    spawner_state = ckpt.get("spawner") or ckpt.get("gaussian_spawner")
    if isinstance(spawner_state, dict):
        result = model.gaussian_spawner.load_state_dict(spawner_state, strict=False)
        load_info["spawner"] = {
            "loaded": True,
            "missing": list(result.missing_keys),
            "unexpected": list(result.unexpected_keys),
        }
        load_info["loaded"] = True

    if not load_info["loaded"]:
        load_info["reason"] = "no generator/model/state_dict/spawner key"
    return load_info


def _make_input(
    *,
    lr_h: int,
    lr_w: int,
    in_channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(0xF00D)
    rgb = torch.rand((1, 3, lr_h, lr_w), generator=gen, device=device, dtype=torch.float32)
    gbuffers = torch.zeros((1, in_channels - 3, lr_h, lr_w), device=device, dtype=torch.float32)
    if in_channels >= 6:
        gbuffers[:, 0:1].uniform_(0.1, 1.0, generator=gen)
    return torch.cat([rgb, gbuffers], dim=1).to(dtype=dtype).contiguous()


class V6E2ERunner:
    def __init__(
        self,
        *,
        model: V6Model,
        lr_inputs: torch.Tensor,
        base_canvas: CanvasState,
        device: torch.device,
        fusion_chunk_windows: int,
    ) -> None:
        self.model = model
        self.lr_inputs = lr_inputs
        self.base_canvas = base_canvas
        self.device = device
        self.fusion_chunk_windows = max(1, int(fusion_chunk_windows))
        self.output_hw = (
            int(lr_inputs.shape[-2]) * int(model.scale),
            int(lr_inputs.shape[-1]) * int(model.scale),
        )

    def reset_canvas(self) -> None:
        self.model._canvas_state = _clone_canvas(self.base_canvas)
        self.model._st_state = None
        self.model.keyframe_mask.reset()

    def fusion_forward(self, feats: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Exact window-aligned fusion with bounded peak memory.

        PixelGaussianFusion is window-local on the pixel side and global on the
        Gaussian-token side, so splitting the LR feature image on window
        boundaries preserves the math while avoiding a full 1080p x 16k-token
        attention expansion in one call.
        """
        ws = int(self.model.fusion.window_size)
        chunk = max(ws, self.fusion_chunk_windows * ws)
        _, _, h, w = feats.shape
        if h <= chunk and w <= chunk:
            return self.model.fusion(feats, tokens)

        rows: list[torch.Tensor] = []
        for y0 in range(0, h, chunk):
            cols: list[torch.Tensor] = []
            y1 = min(y0 + chunk, h)
            for x0 in range(0, w, chunk):
                x1 = min(x0 + chunk, w)
                tile = feats[:, :, y0:y1, x0:x1].contiguous()
                cols.append(self.model.fusion(tile, tokens))
            rows.append(torch.cat(cols, dim=-1))
        return torch.cat(rows, dim=-2).contiguous()

    def run_once(self, *, collect_times: bool) -> tuple[torch.Tensor, dict[str, float]]:
        times: dict[str, float] = {}
        self.reset_canvas()
        model = self.model
        lr_inputs = self.lr_inputs

        def hat_stage() -> torch.Tensor:
            with _NvtxRange("hat_tiny_lr_forward"):
                feats = model.backbone(lr_inputs)
                return model.activation(model.pixel_head(feats))

        if collect_times:
            feats, times["hat_tiny_lr_forward"] = _timed(self.device, hat_stage)
        else:
            feats = hat_stage()

        b, _, h_lr, w_lr = feats.shape
        output_hw = (h_lr * model.scale, w_lr * model.scale)
        warped_canvas = model._warped_canvas(motion_lr=None, output_hw=output_hw)
        active_mask = model._active_mask(warped_canvas, frame_index=0, output_hw=output_hw)

        def fusion_stage() -> torch.Tensor:
            with _NvtxRange("cross_attention_fusion"):
                tokens = model._tokens_from_canvas(feats, warped_canvas, active_mask)
                return self.fusion_forward(feats, tokens)

        if collect_times:
            refined, times["cross_attention_fusion"] = _timed(self.device, fusion_stage)
        else:
            refined = fusion_stage()

        def spawn_stage() -> tuple[CanvasState, torch.Tensor]:
            with _NvtxRange("spawner_write_back"):
                spawned = model.gaussian_spawner(
                    refined,
                    spawn_offset_xy=model._spawn_offset_for(refined),
                )
                spawned_canvas = model._flatten_spawned(spawned)
                old_count = 0 if warped_canvas is None else int(warped_canvas.count)
                render_canvas = model._concat_canvas(warped_canvas, spawned_canvas)
                model._canvas_state = render_canvas
                model.keyframe_mask.reset()
                render_active = model._render_active_mask(
                    old_active=active_mask,
                    old_count=old_count,
                    new_count=int(spawned_canvas.count),
                    canvas=render_canvas,
                )
                return render_canvas, render_active

        if collect_times:
            (render_canvas, render_active), times["spawner_write_back"] = _timed(
                self.device,
                spawn_stage,
            )
        else:
            render_canvas, render_active = spawn_stage()

        def raster_composite_stage() -> torch.Tensor:
            with _NvtxRange("rasterizer_composite_head"):
                canvas_hr = model.rasterizer(
                    render_canvas,
                    render_active.unsqueeze(0).expand(b, -1),
                    output_hw=output_hw,
                )
                refined_hr = F.interpolate(
                    refined,
                    size=output_hw,
                    mode="bilinear",
                    align_corners=False,
                )
                canvas_hr = canvas_hr.to(dtype=refined_hr.dtype)
                bicubic_hr = F.interpolate(
                    lr_inputs[:, :3],
                    size=output_hw,
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                ).clamp(min=0.0)
                delta = model.composite_head(torch.cat([refined_hr, canvas_hr], dim=1))
                rgb_hr = bicubic_hr + delta
                if model.cfg.color_activation in ("sdr", "sigmoid"):
                    return rgb_hr.clamp(0.0, 1.0)
                return rgb_hr.clamp(min=0.0)

        if collect_times:
            out, times["rasterizer_composite_head"] = _timed(
                self.device,
                raster_composite_stage,
            )
        else:
            out = raster_composite_stage()

        self.model._canvas_state = _detach_canvas(model._canvas_state)
        return out, times

    def timed_iteration(self) -> tuple[dict[str, float], float]:
        _sync(self.device)
        total_start = time.perf_counter()
        out, times = self.run_once(collect_times=True)
        _sync(self.device)
        total_ms = (time.perf_counter() - total_start) * 1000.0
        if not bool(torch.isfinite(out).all().detach().item()):
            raise RuntimeError("non-finite output from v6.1 benchmark forward")
        return times, total_ms


def _benchmark(runner: V6E2ERunner, warmup: int, iterations: int) -> dict[str, Any]:
    component_samples = {name: [] for name in COMPONENTS}
    total_samples: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            runner.run_once(collect_times=False)
        _sync(runner.device)
        for _ in range(iterations):
            samples, total_ms = runner.timed_iteration()
            for name in COMPONENTS:
                component_samples[name].append(float(samples[name]))
            total_samples.append(float(total_ms))
    return {
        "components": {name: _summary(values) for name, values in component_samples.items()},
        "total_wallclock": _summary(total_samples),
        "samples_ms": {
            "components": component_samples,
            "total_wallclock": total_samples,
        },
    }


def _device_info(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "name": torch.cuda.get_device_name(device),
        "compute_capability": f"sm_{props.major}{props.minor}",
        "multiprocessor_count": int(props.multi_processor_count),
        "total_memory_bytes": int(props.total_memory),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def _rasterizer_backend(allow_reference: bool) -> str:
    import oss.gaussian.renderer.rasterizer as renderer_mod

    def patch_renderer(rasterize_gaussians: Callable[..., torch.Tensor]) -> None:
        if getattr(renderer_mod.Rasterizer, "_oss_phase4a_patched", False):
            return
        original_call = renderer_mod.Rasterizer.__call__

        def oss_cuda_call(self, gaussians, output_hw):  # type: ignore[no-untyped-def]
            h, w = int(output_hw[0]), int(output_hw[1])
            if gaussians.device.type == "cuda":
                return rasterize_gaussians(
                    gaussians.xy,
                    gaussians.scale.clamp_min(1.0e-3),
                    gaussians.rot,
                    gaussians.feat,
                    h,
                    w,
                    int(getattr(self, "tile_size", 16)),
                    bool(getattr(self, "topk_norm", True)),
                )
            return original_call(self, gaussians, output_hw)

        renderer_mod.Rasterizer.__call__ = oss_cuda_call
        renderer_mod.Rasterizer._oss_phase4a_patched = True

    custom_enabled = getattr(renderer_mod, "_custom_rasterizer_enabled", lambda: False)
    if custom_enabled():
        try:
            from oss.cuda.oss_cuda import rasterize_gaussians  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "OSS_USE_CUDA_KERNELS requests the native rasterizer, but "
                "oss.cuda.oss_cuda is not importable. Build with "
                "`pip install --no-build-isolation -e ./oss/cuda`."
            ) from exc
        patch_renderer(rasterize_gaussians)
        return "oss_cuda"
    try:
        from oss.cuda.oss_cuda import rasterize_gaussians
    except Exception:
        pass
    else:
        os.environ.setdefault("OSS_USE_CUDA_KERNELS", "rasterizer")
        patch_renderer(rasterize_gaussians)
        return "oss_cuda"
    if renderer_mod._GSPLAT_AVAILABLE:
        return "gsplat"
    if allow_reference:
        return "reference"
    raise RuntimeError(
        "No CUDA rasterizer backend is importable. Install gsplat or build "
        "./oss/cuda; use --allow-reference-rasterizer only for tiny debugging."
    )


def _host_key(explicit: str | None, device: torch.device) -> str:
    if explicit:
        return explicit
    name = socket.gethostname().split(".")[0].lower() or platform.node().lower()
    cc = torch.cuda.get_device_capability(device)
    return f"{name}-sm_{cc[0]}{cc[1]}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _merge_result(output_json: Path, host_key: str, host_result: dict[str, Any]) -> dict[str, Any]:
    payload = _read_json(output_json)
    if not payload:
        payload = {
            "phase": "4a",
            "benchmark": "v6.1_end_to_end_forward",
            "schema_version": 1,
            "generated_at": _utc_now(),
            "hosts": {},
        }
    payload["updated_at"] = _utc_now()
    payload.setdefault("hosts", {})[host_key] = host_result
    payload["latest_host"] = host_key
    return payload


def _find_baseline_host(baseline: dict[str, Any], host_key: str, device_name: str) -> dict[str, Any] | None:
    hosts = baseline.get("hosts")
    if not isinstance(hosts, dict):
        return None
    if isinstance(hosts.get(host_key), dict):
        return hosts[host_key]
    for item in hosts.values():
        if isinstance(item, dict) and item.get("device", {}).get("name") == device_name:
            return item
    return None


def _check_regression(
    *,
    baseline_json: Path,
    host_key: str,
    current: dict[str, Any],
    threshold: float,
) -> None:
    baseline = _read_json(baseline_json)
    base_host = _find_baseline_host(
        baseline,
        host_key,
        current.get("device", {}).get("name", ""),
    )
    if base_host is None:
        raise RuntimeError(f"no matching baseline host in {baseline_json} for {host_key}")
    base_ms = float(base_host["metrics"]["total_wallclock"]["median_ms"])
    cur_ms = float(current["metrics"]["total_wallclock"]["median_ms"])
    limit = base_ms * (1.0 + threshold)
    if cur_ms > limit:
        raise AssertionError(
            f"v6.1 e2e total regression: current={cur_ms:.3f}ms "
            f"baseline={base_ms:.3f}ms limit={limit:.3f}ms"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=sorted(SHAPES), default="prod")
    parser.add_argument("--iterations", "--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--host-name", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--canvas-count", type=int, default=None)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--fusion-chunk-windows",
        type=int,
        default=4,
        help=(
            "Window-aligned chunk edge for cross-attention. The v6 fusion math "
            "is unchanged, but peak memory stays bounded for 1080p x 16k tokens."
        ),
    )
    parser.add_argument("--allow-missing-cuda", action="store_true")
    parser.add_argument("--allow-reference-rasterizer", action="store_true")
    parser.add_argument("--check-regression", action="store_true")
    parser.add_argument("--regression-threshold", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        if args.allow_missing_cuda:
            print("CUDA not available; skipping v6.1 e2e benchmark")
            return 0
        raise RuntimeError("CUDA device required for v6.1 e2e benchmark")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    rasterizer_backend = _rasterizer_backend(args.allow_reference_rasterizer)
    tensor_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    shape = copy.deepcopy(SHAPES[args.shape])
    canvas_count = int(args.canvas_count or shape["canvas_count"])
    lr_h = int(shape["lr_h"])
    lr_w = int(shape["lr_w"])
    scale = int(shape["scale"])
    output_hw = (lr_h * scale, lr_w * scale)

    cfg = V6Config(
        backbone="hat-tiny",
        scale=scale,
        canvas_capacity=canvas_count,
        token_dim=64,
        cross_attention_heads=6,
        window_size=16,
        rasterizer_overlap=0,
        color_activation="hdr",
    )
    model = V6Model(cfg).to(device=device, dtype=tensor_dtype).eval()
    weight_load: dict[str, Any] = {"loaded": False}
    if args.checkpoint is not None:
        weight_load = _load_model_weights(model, args.checkpoint)
        model = model.to(device=device, dtype=tensor_dtype).eval()

    base_canvas = None
    canvas_source = "synthetic"
    if args.checkpoint is not None:
        base_canvas = _canvas_from_checkpoint(
            args.checkpoint,
            token_dim=cfg.token_dim,
            output_hw=output_hw,
            device=device,
            dtype=tensor_dtype,
        )
        if base_canvas is not None:
            canvas_source = str(args.checkpoint)
    if base_canvas is None:
        base_canvas = _synthetic_canvas(
            count=canvas_count,
            token_dim=cfg.token_dim,
            output_hw=output_hw,
            device=device,
            dtype=tensor_dtype,
        )

    lr_inputs = _make_input(
        lr_h=lr_h,
        lr_w=lr_w,
        in_channels=cfg.in_channels,
        device=device,
        dtype=tensor_dtype,
    )
    runner = V6E2ERunner(
        model=model,
        lr_inputs=lr_inputs,
        base_canvas=base_canvas,
        device=device,
        fusion_chunk_windows=args.fusion_chunk_windows,
    )

    metrics = _benchmark(runner, warmup=args.warmup, iterations=args.iterations)
    host_key = _host_key(args.host_name, device)
    host_result = {
        "host": {
            "key": host_key,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "device": _device_info(device),
        "shape": {
            "name": args.shape,
            "lr": {"height": lr_h, "width": lr_w, "channels": cfg.in_channels},
            "hr": {"height": output_hw[0], "width": output_hw[1], "channels": 3},
            "scale": scale,
            "canvas_count": int(base_canvas.count),
            "fusion_chunk_windows": int(args.fusion_chunk_windows),
        },
        "config": asdict(cfg),
        "benchmark": {
            "warmup": int(args.warmup),
            "iterations": int(args.iterations),
            "dtype": args.dtype,
            "component_order": list(COMPONENTS),
            "timing": "torch.cuda.synchronize + time.perf_counter",
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "checkpoint_load": weight_load,
            "canvas_source": canvas_source,
            "rasterizer_backend": rasterizer_backend,
        },
        "metrics": metrics,
        "measured_at": _utc_now(),
    }
    if args.check_regression:
        _check_regression(
            baseline_json=args.baseline_json,
            host_key=host_key,
            current=host_result,
            threshold=float(args.regression_threshold),
        )

    payload = _merge_result(args.output_json, host_key, host_result)
    _write_payload(args.output_json, payload)
    print(json.dumps(host_result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
