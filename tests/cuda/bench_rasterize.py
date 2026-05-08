"""Python fallback benchmark for Phase 2c rasterizer gate.

Prefer `tests/cuda/cpp/bench_oss_cuda` when nvbench builds. This script exists
for hosts where nvbench/CMake linkage is unavailable and records the required
PyTorch-reference comparison in the same JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_inputs(n: int, h: int, w: int, f: int, device: torch.device):
    torch.manual_seed(0xC0DA)
    xy = torch.rand(n, 2, device=device, dtype=torch.float32) * torch.tensor(
        [float(w), float(h)], device=device
    )
    scale = torch.rand(n, 2, device=device, dtype=torch.float32) * 5.0 + 0.5
    rot = torch.rand(n, device=device, dtype=torch.float32) * 6.28
    feat = torch.randn(n, f, device=device, dtype=torch.float32)
    return xy, scale, rot, feat


def _time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/coordination/bench-baseline.json"))
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--h", type=int, default=540)
    parser.add_argument("--w", type=int, default=960)
    parser.add_argument("--f", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--ref-iters", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    device = torch.device("cuda:0")
    xy, scale, rot, feat = _make_inputs(args.n, args.h, args.w, args.f, device)

    from oss.cuda.oss_cuda import rasterize_gaussians
    from oss.gaussian.renderer.rasterizer import GaussianBatch, Rasterizer

    rast = Rasterizer(tile_size=16, topk_norm=True)
    batch = GaussianBatch(xy=xy, scale=scale, rot=rot, feat=feat)

    kernel_ms = _time_cuda(
        lambda: rasterize_gaussians(xy, scale, rot, feat, args.h, args.w, 16, True),
        args.warmup,
        args.iters,
    )
    ref_ms = _time_cuda(lambda: rast._render_reference(batch, args.h, args.w), 1, args.ref_iters)
    speedup = ref_ms / kernel_ms
    if speedup < 5.0:
        raise AssertionError(f"rasterize speedup {speedup:.2f}x is below 5x gate")

    payload = {
        "phase": "2c",
        "shape": {"N": args.n, "H": args.h, "W": args.w, "F": args.f},
        "pytorch_ref_ms": ref_ms,
        "kernel_ms": kernel_ms,
        "speedup": speedup,
        "warmup": args.warmup,
        "iters": args.iters,
        "ref_iters": args.ref_iters,
        "device": torch.cuda.get_device_name(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
