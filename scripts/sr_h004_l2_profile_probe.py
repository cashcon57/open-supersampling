#!/usr/bin/env python
"""Synthetic forward-only rasterizer target for H004 Nsight L2 profiling.

This script intentionally does little reporting itself. It exists so Nsight
Compute can attach to one deterministic v6 rasterizer forward loop at the
requested H004 dimensions.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _make_canvas(
    torch,
    canvas_cls,
    *,
    n: int,
    h: int,
    w: int,
    token_dim: int,
    device,
    seed: int,
):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    positions = torch.empty((n, 2), device=device, dtype=torch.float32)
    positions[:, 0].uniform_(0.0, float(w), generator=gen)
    positions[:, 1].uniform_(0.0, float(h), generator=gen)

    scales = torch.empty((n, 2), device=device, dtype=torch.float32)
    scales[:, 0].uniform_(1.5, 6.0, generator=gen)
    scales[:, 1].uniform_(1.5, 6.0, generator=gen)
    rotations = torch.empty((n,), device=device, dtype=torch.float32)
    rotations.uniform_(-math.pi, math.pi, generator=gen)

    colors = torch.randn((n, token_dim), device=device, dtype=torch.float32, generator=gen)
    opacities = torch.ones((n,), device=device, dtype=torch.float32)
    return canvas_cls(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        count=n,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--h", type=int, default=540)
    parser.add_argument("--w", type=int, default=960)
    parser.add_argument("--rank", type=int, choices=(8, 64), required=True)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260508)
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - explicit probe precondition
        print(f"NOT RUN: torch unavailable: {type(exc).__name__}: {exc}")
        return 0

    if not torch.cuda.is_available():
        print("NOT RUN: torch.cuda.is_available() is False")
        return 0
    device = torch.device(args.device)
    if device.type != "cuda":
        print(f"NOT RUN: CUDA device required; got {device}")
        return 0

    from oss.sr.v6.model import CanvasState
    from oss.sr.v6.rasterizer import V6Rasterizer

    torch.cuda.set_device(device)
    canvas = _make_canvas(
        torch,
        CanvasState,
        n=args.n,
        h=args.h,
        w=args.w,
        token_dim=args.token_dim,
        device=device,
        seed=args.seed,
    )
    active_mask = torch.ones((args.n,), device=device, dtype=torch.bool)
    rasterizer = (
        V6Rasterizer(token_dim=args.token_dim, latent_rank=args.rank)
        .to(device)
        .eval()
    )

    print(
        "H004 L2 profile probe: "
        f"N={args.n} H={args.h} W={args.w} R={args.rank} "
        f"token_dim={args.token_dim} warmup={args.warmup} iters={args.iters}"
    )
    print(f"device={torch.cuda.get_device_name(device)} torch={torch.__version__}")

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = rasterizer(canvas, active_mask, (args.h, args.w))
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            _ = rasterizer(canvas, active_mask, (args.h, args.w))
        end.record()
        torch.cuda.synchronize(device)

    total_ms = float(start.elapsed_time(end))
    print(f"avg_ms={(total_ms / max(1, args.iters)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
