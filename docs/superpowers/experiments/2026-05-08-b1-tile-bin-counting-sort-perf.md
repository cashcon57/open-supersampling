# B1 Tile-Bin Counting-Sort Perf

Date: 2026-05-08
Owner: B1

## Method

Added a native CUDA tile-binning variant that groups `gid` values by
`tile_id` with a count -> exclusive prefix sum -> scatter pipeline:

- Input: `N = 16,384`
- Tile ids: random uniform `tile_id in [0, 2000)`
- Gids: `gid = arange(N)`
- Baseline: `torch.sort(tile_id)` plus `gid[order]`
- Variant: `oss.sr.v6.tile_bin.tile_bin_counting_sort(tile_id, gid, 2000)`
- Timing: CUDA events, 100 warmup iterations, 1000 measured iterations
- Reported metrics: median and p99 milliseconds

The benchmark script validates the CUDA output before timing by checking exact
tile offsets, gid permutation, and nondecreasing output tile ids.

## Results

Measured on the reachable 3080 Ti host:

- Host: `Cash-PC`
- GPU: NVIDIA GeForce RTX 3080 Ti
- Driver: 595.79
- PyTorch: 2.4.1
- CUDA runtime: 12.4

Command:

```bash
python scripts/bench_tile_bin_counting_sort.py --warmup-iters 100 --measure-iters 1000
```

Output:

```text
N=16384 num_tiles=2000 warmup=100 measured=1000
torch.sort median_ms=0.256832 p99_ms=0.532256
counting_sort median_ms=0.653280 p99_ms=1.667072
speedup_median=0.393x verdict=FAIL
Acceptance failed: GitHub issue required if this run is authoritative.
```

Local macOS checkout attempts before the remote run were `NOT RUN` because the
system Python had no PyTorch and `venv-py312` had PyTorch without CUDA.

## Verdict

**FAIL** versus the acceptance target: counting-sort median runtime must be at
least 3x faster than the `torch.sort` baseline.

The measured counting-sort implementation is slower than `torch.sort` on this
workload (`0.393x` speedup). Blocker filed:

- https://github.com/cashcon57/open-supersampling/issues/15
