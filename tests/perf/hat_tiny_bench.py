#!/usr/bin/env python
"""Benchmark the v6.1 HAT-Tiny backbone used by pico-001 training."""
from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on the active environment.
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR: Exception | None = exc
else:
    TORCH_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class BenchResult:
    name: str
    median_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=270, help="LR input height.")
    parser.add_argument("--width", type=int, default=480, help="LR input width.")
    parser.add_argument("--channels", type=int, default=9, help="Input channels.")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile variants even when torch.compile is available.",
    )
    return parser.parse_args(argv)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot percentile empty sample")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def load_hat_tiny(in_channels: int) -> torch.nn.Module:
    try:
        from oss.sr.v6.hat import hat_tiny
    except Exception as exc:  # pragma: no cover - environment-dependent message path.
        raise RuntimeError(f"could not import oss.sr.v6.hat.hat_tiny: {exc}") from exc

    try:
        model = hat_tiny(in_channels=in_channels)
    except Exception as exc:  # pragma: no cover - environment-dependent message path.
        raise RuntimeError(f"could not construct HAT-Tiny backbone: {exc}") from exc

    return model


def benchmark_forward(
    name: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
) -> BenchResult:
    if iters < 1:
        raise ValueError("--iters must be >= 1")
    if warmup < 0:
        raise ValueError("--warmup must be >= 0")

    model.eval()
    timings_ms: list[float] = []

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()

        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x)
            end.record()
            end.synchronize()
            timings_ms.append(float(start.elapsed_time(end)))

    return BenchResult(
        name=name,
        median_ms=statistics.median(timings_ms),
        p50_ms=percentile(timings_ms, 0.50),
        p95_ms=percentile(timings_ms, 0.95),
        p99_ms=percentile(timings_ms, 0.99),
    )


def maybe_compile(model: torch.nn.Module) -> torch.nn.Module:
    compile_fn: Callable[..., torch.nn.Module] | None = getattr(torch, "compile", None)
    if compile_fn is None:
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    return compile_fn(model, mode="reduce-overhead")


def print_results(results: list[BenchResult]) -> None:
    print()
    print(f"{'variant':<20} {'median':>10} {'p50':>10} {'p95':>10} {'p99':>10}")
    print("-" * 64)
    for r in results:
        print(
            f"{r.name:<20} "
            f"{r.median_ms:>9.3f} "
            f"{r.p50_ms:>9.3f} "
            f"{r.p95_ms:>9.3f} "
            f"{r.p99_ms:>9.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if torch is None:
        print(f"PyTorch is unavailable; cannot load v6.1 HAT-Tiny backbone: {TORCH_IMPORT_ERROR}")
        return 2

    if not torch.cuda.is_available():
        print("CUDA is unavailable; HAT-Tiny benchmark requires a CUDA GPU.")
        return 0

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    try:
        base = load_hat_tiny(in_channels=args.channels)
    except RuntimeError as exc:
        print(f"Could not load v6.1 HAT-Tiny backbone: {exc}")
        return 2

    shape = (args.batch_size, args.channels, args.height, args.width)
    print("Benchmarking oss.sr.v6.hat.hat_tiny")
    print(f"input={shape} warmup={args.warmup} iters={args.iters} device={torch.cuda.get_device_name(device)}")

    results: list[BenchResult] = []
    variants: list[tuple[str, torch.dtype]] = [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
    ]

    for precision_name, dtype in variants:
        x = torch.randn(shape, device=device, dtype=dtype)
        model = base.to(device=device, dtype=dtype)

        try:
            results.append(
                benchmark_forward(
                    name=f"{precision_name}/eager",
                    model=model,
                    x=x,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )
        except Exception as exc:
            print(f"Skipping {precision_name}/eager: {exc}")

        if args.no_compile:
            continue

        try:
            compiled_model = maybe_compile(model)
            results.append(
                benchmark_forward(
                    name=f"{precision_name}/compile",
                    model=compiled_model,
                    x=x,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )
        except Exception as exc:
            print(f"Skipping {precision_name}/compile: {exc}")

    if not results:
        print("No benchmark variants completed.")
        return 3

    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
