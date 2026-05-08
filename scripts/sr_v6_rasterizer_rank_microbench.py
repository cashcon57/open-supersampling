#!/usr/bin/env python3
"""Microbenchmark v6 rasterizer latency across latent ranks.

Synthetic case:
  - N=4096 Gaussians
  - H=540, W=960
  - random xy/conic/feat
  - latent_rank sweep R in {4, 8, 16, 32, 64}

Timings use CUDA events and report median/p99 milliseconds plus speedup
relative to the R=64 row. If torch or CUDA is unavailable, the script prints
an explicit NOT RUN line and exits successfully without fabricating numbers.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RANKS = (4, 8, 16, 32, 64)


@dataclass(frozen=True)
class BenchStats:
    median_ms: float
    p99_ms: float


def _nearest_rank_p99(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute p99 of an empty sample")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(0.99 * len(ordered)) - 1))
    return ordered[idx]


def _stats(values: list[float]) -> BenchStats:
    return BenchStats(
        median_ms=float(statistics.median(values)),
        p99_ms=float(_nearest_rank_p99(values)),
    )


def _format_speedup(baseline_ms: float | None, current_ms: float | None) -> str:
    if baseline_ms is None or current_ms is None:
        return "n/a"
    return f"{baseline_ms / current_ms:.2f}x"


def _loss_from_output(output):
    if isinstance(output, tuple):
        latent, weight_sum = output
        return latent.float().mean() + weight_sum.float().mean()
    return output.float().mean()


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
    requires_grad: bool,
):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    xy = torch.empty(n, 2, device=device, dtype=torch.float32)
    xy[:, 0].uniform_(0.0, float(w), generator=gen)
    xy[:, 1].uniform_(0.0, float(h), generator=gen)

    # Generate a positive-definite random conic via scale+rotation, then feed
    # the equivalent scale/rotation contract expected by V6Rasterizer.
    scale = torch.empty(n, 2, device=device, dtype=torch.float32)
    scale[:, 0].uniform_(1.5, 6.0, generator=gen)
    scale[:, 1].uniform_(1.5, 6.0, generator=gen)
    rot = torch.empty(n, device=device, dtype=torch.float32).uniform_(
        -math.pi,
        math.pi,
        generator=gen,
    )
    conic = _scale_rot_to_conic(torch, scale, rot)

    feat = torch.randn(n, token_dim, device=device, dtype=torch.float32, generator=gen)
    opacities = torch.ones(n, device=device, dtype=torch.float32)

    if requires_grad:
        xy.requires_grad_()
        scale.requires_grad_()
        rot.requires_grad_()
        feat.requires_grad_()

    canvas = canvas_cls(
        positions=xy,
        scales=scale,
        rotations=rot,
        opacities=opacities,
        colors=feat,
        count=n,
    )
    return canvas, conic


def _scale_rot_to_conic(torch, scale, rot):
    cos_t = torch.cos(rot)
    sin_t = torch.sin(rot)
    inv_sx2 = 1.0 / scale[:, 0].square()
    inv_sy2 = 1.0 / scale[:, 1].square()
    a = cos_t.square() * inv_sx2 + sin_t.square() * inv_sy2
    b = cos_t * sin_t * (inv_sx2 - inv_sy2)
    d = sin_t.square() * inv_sx2 + cos_t.square() * inv_sy2
    return torch.stack((a, b, d), dim=-1).contiguous()


def _clear_grads(canvas) -> None:
    for tensor in (canvas.positions, canvas.scales, canvas.rotations, canvas.colors):
        tensor.grad = None


def _time_forward(
    torch,
    rasterizer,
    canvas,
    active_mask,
    output_hw: tuple[int, int],
    *,
    warmup: int,
    iters: int,
    device,
) -> BenchStats:
    for _ in range(warmup):
        _ = rasterizer(canvas, active_mask, output_hw)
    torch.cuda.synchronize(device)

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = rasterizer(canvas, active_mask, output_hw)
        end.record()
        torch.cuda.synchronize(device)
        times.append(float(start.elapsed_time(end)))
    return _stats(times)


def _time_backward(
    torch,
    rasterizer,
    canvas,
    active_mask,
    output_hw: tuple[int, int],
    *,
    warmup: int,
    iters: int,
    device,
) -> BenchStats:
    for _ in range(warmup):
        _clear_grads(canvas)
        loss = _loss_from_output(rasterizer(canvas, active_mask, output_hw))
        loss.backward()
    torch.cuda.synchronize(device)

    times: list[float] = []
    for _ in range(iters):
        _clear_grads(canvas)
        loss = _loss_from_output(rasterizer(canvas, active_mask, output_hw))
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize(device)
        times.append(float(start.elapsed_time(end)))
    return _stats(times)


def _print_table(title: str, rows: list[tuple[int, BenchStats | None, str | None]]) -> None:
    baseline = next(
        (stats.median_ms for rank, stats, _ in rows if rank == 64 and stats is not None),
        None,
    )
    print(f"\n{title}")
    print(
        f"{'R':>4s}  {'median_ms':>10s}  {'p99_ms':>10s}  "
        f"{'speedup_vs_R64':>15s}  status"
    )
    print("-" * 59)
    for rank, stats, error in rows:
        if stats is None:
            print(f"{rank:4d}  {'n/a':>10s}  {'n/a':>10s}  {'n/a':>15s}  ERROR: {error}")
            continue
        print(
            f"{rank:4d}  {stats.median_ms:10.3f}  {stats.p99_ms:10.3f}  "
            f"{_format_speedup(baseline, stats.median_ms):>15s}  ok"
        )


def _parse_ranks(raw: str) -> tuple[int, ...]:
    ranks = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not ranks:
        raise argparse.ArgumentTypeError("at least one rank is required")
    if any(rank <= 0 for rank in ranks):
        raise argparse.ArgumentTypeError("all ranks must be positive")
    return ranks


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--h", type=int, default=540)
    parser.add_argument("--w", type=int, default=960)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--ranks", type=_parse_ranks, default=RANKS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-backward", action="store_true", help="Skip backward sweep.")
    args = parser.parse_args(argv)

    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - explicit benchmark precondition report
        print(f"NOT RUN: torch unavailable: {type(exc).__name__}: {exc}")
        return 0

    if args.device != "cuda" and not args.device.startswith("cuda:"):
        print(
            "NOT RUN: CUDA device required for CUDA-event timing; "
            f"got --device={args.device!r}"
        )
        return 0
    if not torch.cuda.is_available():
        print("NOT RUN: torch.cuda.is_available() is False")
        return 0

    from oss.gaussian.renderer import rasterizer as renderer_mod
    from oss.sr.v6.model import CanvasState
    from oss.sr.v6.rasterizer import V6Rasterizer

    device = torch.device(args.device)
    torch.cuda.set_device(device.index if device.index is not None else 0)
    torch.backends.cuda.matmul.allow_tf32 = False

    print("OpenSuperSampling v6 rasterizer latent-rank microbench")
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
    print(f"OSS_USE_CUDA_KERNELS={os.environ.get('OSS_USE_CUDA_KERNELS', '')!r}")
    custom_cuda_enabled = renderer_mod._custom_rasterizer_enabled()
    print(
        "renderer_gsplat_available="
        f"{renderer_mod._GSPLAT_AVAILABLE} import_error={renderer_mod._GSPLAT_IMPORT_ERROR}"
    )
    print(f"renderer_oss_cuda_enabled={custom_cuda_enabled}")
    print(
        f"synthetic: N={args.n} H={args.h} W={args.w} token_dim={args.token_dim} "
        f"warmup={args.warmup} iters={args.iters} ranks={','.join(map(str, args.ranks))}"
    )
    print(
        "note: random conic is generated from scale/rotation; "
        "V6Rasterizer consumes scale/rotation."
    )

    if not renderer_mod._GSPLAT_AVAILABLE and not custom_cuda_enabled:
        print(
            "NOT RUN: no CUDA rasterizer backend available "
            "(gsplat unavailable and OSS_USE_CUDA_KERNELS does not enable rasterizer)"
        )
        return 0

    active_mask = torch.ones(args.n, dtype=torch.bool, device=device)
    forward_rows: list[tuple[int, BenchStats | None, str | None]] = []
    backward_rows: list[tuple[int, BenchStats | None, str | None]] = []

    for rank in args.ranks:
        torch.cuda.empty_cache()
        canvas, _conic = _make_canvas(
            torch,
            CanvasState,
            n=args.n,
            h=args.h,
            w=args.w,
            token_dim=args.token_dim,
            device=device,
            seed=args.seed,
            requires_grad=False,
        )
        rasterizer = V6Rasterizer(token_dim=args.token_dim, latent_rank=rank).to(
            device
        )
        try:
            stats = _time_forward(
                torch,
                rasterizer,
                canvas,
                active_mask,
                (args.h, args.w),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            forward_rows.append((rank, stats, None))
        except Exception as exc:  # noqa: BLE001 - keep sweep going
            forward_rows.append((rank, None, f"{type(exc).__name__}: {exc}"))

    _print_table("Forward", forward_rows)

    if not args.no_backward:
        for rank in args.ranks:
            torch.cuda.empty_cache()
            canvas, _conic = _make_canvas(
                torch,
                CanvasState,
                n=args.n,
                h=args.h,
                w=args.w,
                token_dim=args.token_dim,
                device=device,
                seed=args.seed,
                requires_grad=True,
            )
            rasterizer = V6Rasterizer(token_dim=args.token_dim, latent_rank=rank).to(
                device
            )
            try:
                stats = _time_backward(
                    torch,
                    rasterizer,
                    canvas,
                    active_mask,
                    (args.h, args.w),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
                backward_rows.append((rank, stats, None))
            except Exception as exc:  # noqa: BLE001 - unsupported backward is expected
                backward_rows.append((rank, None, f"{type(exc).__name__}: {exc}"))

        _print_table("Backward", backward_rows)

    r4 = next((stats for rank, stats, _ in forward_rows if rank == 4), None)
    r64 = next((stats for rank, stats, _ in forward_rows if rank == 64), None)
    if r4 is not None and r64 is not None:
        speedup = r64.median_ms / r4.median_ms
        verdict = "PASS" if speedup >= 4.0 else "FAIL"
        print(f"\nGate: R=4 forward speedup vs R=64 = {speedup:.2f}x => {verdict}")
    else:
        print("\nGate: NOT RUN: missing successful R=4 or R=64 forward timing")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
