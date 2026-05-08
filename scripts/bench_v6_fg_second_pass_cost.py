#!/usr/bin/env python3
"""Microbench v6 rasterizer cost for a foreground second pass.

Synthetic shape is fixed to the H005 target unless overridden:
N=4096 gaussians, H=540, W=960.  For each latent rank R in {4, 8, 64},
the harness renders the same synthetic canvas twice with different motion
offsets:

- pass 1: alpha=1.0 display-frame offset
- pass 2: alpha=0.5 mid-interval extrapolated offset

Timings use CUDA events and report median milliseconds.  If torch, CUDA, or a
CUDA rasterizer backend is unavailable, the script prints an explicit NOT RUN
record and exits without fabricating timing data.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RANKS = (4, 8, 64)
DEFAULT_N = 4096
DEFAULT_H = 540
DEFAULT_W = 960
GATE_RATIO = 1.5


class NotRun(RuntimeError):
    """Raised when the benchmark cannot produce real CUDA timing data."""


@dataclass(frozen=True)
class BenchInputs:
    canvas: Any
    motion: Any
    active_mask: Any


def _import_torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - report the import blocker verbatim
        raise NotRun(f"torch unavailable: {type(exc).__name__}: {exc}") from exc
    return torch


def _enable_available_cuda_backend() -> str:
    """Return the backend the existing renderer wrapper should use."""
    requested = os.environ.get("OSS_USE_CUDA_KERNELS", "").strip().lower()
    if requested not in {"", "0", "false", "off", "none"}:
        try:
            from oss.cuda.oss_cuda import rasterize_gaussians  # noqa: F401
            from oss.cuda.oss_cuda import rasterizer as oss_rasterizer
        except Exception as exc:  # noqa: BLE001
            raise NotRun(
                "OSS_USE_CUDA_KERNELS requests oss_cuda, but it is unavailable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not getattr(oss_rasterizer, "_COMPILED", False):
            raise NotRun("OSS CUDA rasterizer requested but compiled extension is unavailable")
        return "oss_cuda"

    from oss.gaussian.renderer import rasterizer as renderer_mod

    if getattr(renderer_mod, "_GSPLAT_AVAILABLE", False):
        return "gsplat"

    try:
        from oss.cuda.oss_cuda import rasterizer as oss_rasterizer

        if getattr(oss_rasterizer, "_COMPILED", False):
            os.environ.setdefault("OSS_USE_CUDA_KERNELS", "rasterizer")
            return "oss_cuda"
    except Exception:
        pass

    import_error = getattr(renderer_mod, "_GSPLAT_IMPORT_ERROR", None)
    raise NotRun(
        "no CUDA rasterizer backend available; build oss_cuda or gsplat. "
        f"gsplat import error: {import_error}"
    )


def _make_canvas(*, torch: Any, n: int, h: int, w: int, r: int, device: Any) -> BenchInputs:
    from oss.sr.v6.model import CanvasState

    generator = torch.Generator(device=device)
    generator.manual_seed(0xB50000 + int(r))
    margin = 16.0
    xy_extent = torch.tensor(
        [max(float(w) - 2.0 * margin, 1.0), max(float(h) - 2.0 * margin, 1.0)],
        device=device,
        dtype=torch.float32,
    )
    positions = torch.rand(n, 2, device=device, dtype=torch.float32, generator=generator)
    positions = positions * xy_extent + margin
    scales = torch.rand(n, 2, device=device, dtype=torch.float32, generator=generator) * 5.0 + 0.5
    rotations = torch.rand(n, device=device, dtype=torch.float32, generator=generator) * 6.283185307179586
    opacities = torch.ones(n, device=device, dtype=torch.float32)
    colors = torch.randn(n, r, device=device, dtype=torch.float32, generator=generator)
    motion = torch.empty(n, 2, device=device, dtype=torch.float32).uniform_(
        -3.0,
        3.0,
        generator=generator,
    )
    canvas = CanvasState(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        count=int(n),
    )
    active_mask = torch.ones(n, device=device, dtype=torch.bool)
    return BenchInputs(canvas=canvas, motion=motion, active_mask=active_mask)


def _offset_canvas(base_canvas: Any, motion: Any, alpha: float) -> Any:
    from oss.sr.v6.model import CanvasState

    return CanvasState(
        positions=base_canvas.positions + motion * float(alpha),
        scales=base_canvas.scales,
        rotations=base_canvas.rotations,
        opacities=base_canvas.opacities,
        colors=base_canvas.colors,
        count=int(base_canvas.count),
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
    return float(ordered[idx])


def _time_rank(
    *,
    torch: Any,
    rank: int,
    n: int,
    h: int,
    w: int,
    warmup: int,
    iters: int,
    device: Any,
) -> dict[str, Any]:
    from oss.sr.v6.rasterizer import V6Rasterizer

    inputs = _make_canvas(torch=torch, n=n, h=h, w=w, r=rank, device=device)
    display_canvas = _offset_canvas(inputs.canvas, inputs.motion, alpha=1.0)
    mid_canvas = _offset_canvas(inputs.canvas, inputs.motion, alpha=0.5)
    rasterizer = V6Rasterizer(token_dim=64, latent_rank=rank).to(device=device).eval()

    def run_pass1() -> Any:
        return rasterizer(
            display_canvas,
            inputs.active_mask,
            (h, w),
            return_weight_sum=False,
        )

    def run_pass2() -> Any:
        return rasterizer(
            mid_canvas,
            inputs.active_mask,
            (h, w),
            return_weight_sum=False,
        )

    with torch.inference_mode():
        for _ in range(warmup):
            run_pass1()
            run_pass2()
        torch.cuda.synchronize(device)

        pass1_ms: list[float] = []
        pass2_ms: list[float] = []
        total_ms: list[float] = []
        for _ in range(iters):
            total_start = torch.cuda.Event(enable_timing=True)
            pass1_start = torch.cuda.Event(enable_timing=True)
            pass1_end = torch.cuda.Event(enable_timing=True)
            pass2_start = torch.cuda.Event(enable_timing=True)
            pass2_end = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)

            total_start.record()
            pass1_start.record()
            display_out = run_pass1()
            pass1_end.record()
            pass2_start.record()
            mid_out = run_pass2()
            pass2_end.record()
            total_end.record()
            total_end.synchronize()

            if display_out.shape[-2:] != (h, w) or mid_out.shape[-2:] != (h, w):
                raise RuntimeError(
                    "unexpected rasterizer output shapes: "
                    f"{tuple(display_out.shape)}, {tuple(mid_out.shape)}"
                )
            pass1_ms.append(float(pass1_start.elapsed_time(pass1_end)))
            pass2_ms.append(float(pass2_start.elapsed_time(pass2_end)))
            total_ms.append(float(total_start.elapsed_time(total_end)))

    single = statistics.median(pass1_ms)
    total = statistics.median(total_ms)
    ratio = total / single if single > 0.0 else float("inf")
    return {
        "R": int(rank),
        "pass1_alpha": 1.0,
        "pass2_alpha": 0.5,
        "pass1_median_ms": single,
        "pass2_median_ms": statistics.median(pass2_ms),
        "total_median_ms": total,
        "ratio_total_over_pass1": ratio,
        "pass1_p95_ms": _percentile(pass1_ms, 95.0),
        "pass2_p95_ms": _percentile(pass2_ms, 95.0),
        "total_p95_ms": _percentile(total_ms, 95.0),
        "verdict": "PASS" if ratio <= GATE_RATIO else "FAIL",
    }


def _not_run_payload(reason: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "NOT RUN",
        "reason": reason,
        "shape": {"N": int(args.n), "H": int(args.h), "W": int(args.w)},
        "ranks": list(RANKS),
        "gate": "two-pass total median <= 1.5x pass1 median",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--h", type=int, default=DEFAULT_H)
    parser.add_argument("--w", type=int, default=DEFAULT_W)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", action="store_true", help="Only print JSON payload")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        torch = _import_torch()
        if not torch.cuda.is_available():
            raise NotRun("torch.cuda.is_available() is false")
        if args.iters <= 0:
            raise ValueError("--iters must be positive")
        if args.warmup < 0:
            raise ValueError("--warmup must be non-negative")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        backend = _enable_available_cuda_backend()

        results = [
            _time_rank(
                torch=torch,
                rank=rank,
                n=args.n,
                h=args.h,
                w=args.w,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            for rank in RANKS
        ]
        overall = "PASS" if all(row["verdict"] == "PASS" for row in results) else "FAIL"
        payload = {
            "status": overall,
            "shape": {"N": int(args.n), "H": int(args.h), "W": int(args.w)},
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "device": torch.cuda.get_device_name(device),
            "cuda_runtime": torch.version.cuda,
            "torch": torch.__version__,
            "backend": backend,
            "gate": {
                "ratio_limit": GATE_RATIO,
                "definition": "two-pass total median <= 1.5x pass1 median",
            },
            "results": results,
        }
    except NotRun as exc:
        payload = _not_run_payload(str(exc), args)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
        if payload["status"] == "FAIL":
            print("FAIL: GitHub issue required for second-pass cost gate failure.")
        elif payload["status"] == "NOT RUN":
            print(f"NOT RUN: {payload['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
