#!/usr/bin/env python3
"""Benchmark torch.sort tile binning against the OSS CUDA counting-sort op."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


N = 16_384
NUM_TILES = 2_000
WARMUP_ITERS = 100
MEASURE_ITERS = 1_000


def percentile_ms(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def timed_cuda_ms(fn, warmup_iters: int, measure_iters: int) -> list[float]:
    import torch

    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(measure_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--num-tiles", type=int, default=NUM_TILES)
    parser.add_argument("--warmup-iters", type=int, default=WARMUP_ITERS)
    parser.add_argument("--measure-iters", type=int, default=MEASURE_ITERS)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cuda_pkg = repo_root / "oss" / "cuda"
    if str(cuda_pkg) not in sys.path:
        sys.path.insert(0, str(cuda_pkg))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        import torch
    except ImportError as exc:
        print(f"NOT RUN: torch unavailable ({exc})")
        return 0

    if not torch.cuda.is_available():
        print("NOT RUN: torch CUDA unavailable")
        return 0

    try:
        from oss.sr.v6.tile_bin import tile_bin_counting_sort
    except Exception as exc:
        print(f"NOT RUN: oss_cuda tile_bin_counting_sort unavailable ({exc})")
        return 0

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    tile_id = torch.randint(
        0,
        args.num_tiles,
        (args.n,),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    gid = torch.arange(args.n, device="cuda", dtype=torch.int32)

    def torch_sort_baseline() -> tuple[torch.Tensor, torch.Tensor]:
        _, order = torch.sort(tile_id)
        sorted_gid = gid[order]
        return sorted_gid, order

    try:
        sorted_gid, offsets = tile_bin_counting_sort(tile_id, gid, args.num_tiles)
    except ModuleNotFoundError as exc:
        print(f"NOT RUN: oss_cuda tile_bin_counting_sort unavailable ({exc})")
        return 0
    except RuntimeError as exc:
        if "oss_cuda extension not compiled" in str(exc):
            print(f"NOT RUN: oss_cuda tile_bin_counting_sort unavailable ({exc})")
            return 0
        raise
    expected_counts = torch.bincount(tile_id.to(torch.int64), minlength=args.num_tiles)
    expected_offsets = torch.empty(args.num_tiles + 1, device="cuda", dtype=torch.int32)
    expected_offsets[0] = 0
    expected_offsets[1:] = torch.cumsum(expected_counts, dim=0).to(torch.int32)

    if not torch.equal(offsets, expected_offsets):
        raise RuntimeError("counting-sort tile offsets mismatch")
    if not torch.equal(torch.sort(sorted_gid).values, gid):
        raise RuntimeError("counting-sort gid permutation mismatch")
    grouped_tile_id = tile_id[sorted_gid]
    if grouped_tile_id.numel() > 1 and not bool(
        torch.all(grouped_tile_id[:-1] <= grouped_tile_id[1:]).item()
    ):
        raise RuntimeError("counting-sort grouped tile ids mismatch")

    torch_ms = timed_cuda_ms(torch_sort_baseline, args.warmup_iters, args.measure_iters)
    count_ms = timed_cuda_ms(
        lambda: tile_bin_counting_sort(tile_id, gid, args.num_tiles),
        args.warmup_iters,
        args.measure_iters,
    )

    torch_median = percentile_ms(torch_ms, 0.5)
    torch_p99 = percentile_ms(torch_ms, 0.99)
    count_median = percentile_ms(count_ms, 0.5)
    count_p99 = percentile_ms(count_ms, 0.99)
    speedup = torch_median / count_median if count_median > 0 else float("inf")
    verdict = "PASS" if speedup >= 3.0 else "FAIL"

    print(
        f"N={args.n} num_tiles={args.num_tiles} "
        f"warmup={args.warmup_iters} measured={args.measure_iters}"
    )
    print(f"torch.sort median_ms={torch_median:.6f} p99_ms={torch_p99:.6f}")
    print(f"counting_sort median_ms={count_median:.6f} p99_ms={count_p99:.6f}")
    print(f"speedup_median={speedup:.3f}x verdict={verdict}")
    if verdict == "FAIL":
        print("Acceptance failed: GitHub issue required if this run is authoritative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
