#!/usr/bin/env python3
"""Microbench the v6.2 T5 student backbone on synthetic CUDA fp16 input."""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


HAT_TINY_FP16_MEDIAN_MS = 107.548
HAT_TINY_FP16_P99_MS = 129.716


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--channels", type=int, default=9)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on host env
        print(f"NOT RUN: torch unavailable ({type(exc).__name__}: {exc})")
        return 0

    if not torch.cuda.is_available():
        print("NOT RUN: CUDA unavailable")
        print(f"torch: {torch.__version__}")
        print(f"python: {platform.python_version()}")
        print(f"host: {platform.node()}")
        return 0

    from oss.sr.v6.student import StudentBackbone

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    model = StudentBackbone(
        in_channels=args.channels,
        channels=48,
        depth=4,
        out_features=180,
    ).eval().to(device=device, dtype=torch.float16)
    x = torch.randn(
        args.batch,
        args.channels,
        args.height,
        args.width,
        device=device,
        dtype=torch.float16,
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(x)
        torch.cuda.synchronize()

        times_ms: list[float] = []
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

    median_ms = statistics.median(times_ms)
    p99_ms = percentile(times_ms, 0.99)
    gate_ms = HAT_TINY_FP16_MEDIAN_MS / 2.0
    verdict = "PASS" if median_ms <= gate_ms else "FAIL"

    print("T5 student backbone CUDA fp16 microbench")
    print(f"input: B={args.batch}, C={args.channels}, H={args.height}, W={args.width}")
    print(
        "model: "
        f"StudentBackbone(in_channels={args.channels}, channels=48, depth=4, out_features=180)"
    )
    print(f"params: {model.num_params():,}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_runtime: {torch.version.cuda}")
    print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"median_ms: {median_ms:.3f}")
    print(f"p99_ms: {p99_ms:.3f}")
    print(f"hat_tiny_fp16_median_ms: {HAT_TINY_FP16_MEDIAN_MS:.3f}")
    print(f"hat_tiny_fp16_p99_ms: {HAT_TINY_FP16_P99_MS:.3f}")
    print(f"student_2x_gate_median_ms: {gate_ms:.3f}")
    print(f"verdict: {verdict}")
    if verdict == "FAIL":
        print("issue_required: GitHub issue required for failed acceptance gate")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
