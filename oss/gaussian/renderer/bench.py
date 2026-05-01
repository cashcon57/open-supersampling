"""Sprint 1 / T1.6 — Rasterizer performance benchmark.

Measures the renderer's wall-clock time on the configurations OSS-Gaussian
will actually use in production:
- Gaussian counts: 1K, 5K, 8K, 15K
- Output resolutions: 1080p, 1440p, 4K
- Tile size 16, top-K = 10

Outputs a CSV with mean / p50 / p95 / p99 ms over 100 runs (after 10 warm-ups)
to oss/gaussian/renderer/bench/bench_results_<gpu_name>.csv.

Usage:
    python -m oss.gaussian.renderer.bench

Skips quietly with a clear message if CUDA / gsplat is not available — the
reference backend is too slow to bench meaningfully (O(N×H×W) Python loop).
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from oss.gaussian.renderer import GaussianBatch, Rasterizer


@dataclass
class BenchConfig:
    num_gaussians: int
    height: int
    width: int

    @property
    def label(self) -> str:
        return f"N{self.num_gaussians}_{self.width}x{self.height}"


CONFIGS: list[BenchConfig] = [
    BenchConfig(1_000, 1080, 1920),
    BenchConfig(1_000, 1440, 2560),
    BenchConfig(1_000, 2160, 3840),
    BenchConfig(5_000, 1080, 1920),
    BenchConfig(5_000, 1440, 2560),
    BenchConfig(5_000, 2160, 3840),
    BenchConfig(8_000, 1080, 1920),
    BenchConfig(8_000, 1440, 2560),
    BenchConfig(8_000, 2160, 3840),
    BenchConfig(15_000, 1080, 1920),
    BenchConfig(15_000, 1440, 2560),
    BenchConfig(15_000, 2160, 3840),
]

NUM_WARMUP = 10
NUM_RUNS = 100


def _make_random_batch(n: int, h: int, w: int, device: torch.device) -> GaussianBatch:
    """Random valid Gaussians spread across the output canvas."""
    g = torch.Generator(device=device).manual_seed(0)
    return GaussianBatch(
        xy=torch.stack([
            torch.rand(n, generator=g, device=device) * w,
            torch.rand(n, generator=g, device=device) * h,
        ], dim=-1),
        scale=torch.rand((n, 2), generator=g, device=device) * 4.0 + 1.0,
        rot=torch.rand(n, generator=g, device=device) * torch.pi,
        feat=torch.rand((n, 3), generator=g, device=device),
    )


def _bench_one(config: BenchConfig, renderer: Rasterizer, device: torch.device) -> dict:
    """Run a single config: warmup → 100 timed runs → return statistics."""
    batch = _make_random_batch(config.num_gaussians, config.height, config.width, device)
    output_hw = (config.height, config.width)

    for _ in range(NUM_WARMUP):
        _ = renderer(batch, output_hw=output_hw)
    torch.cuda.synchronize()

    timings_ms = torch.zeros(NUM_RUNS, dtype=torch.float64)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(NUM_RUNS):
        start.record()
        _ = renderer(batch, output_hw=output_hw)
        end.record()
        torch.cuda.synchronize()
        timings_ms[i] = start.elapsed_time(end)

    sorted_ms = timings_ms.sort().values
    return {
        "label": config.label,
        "num_gaussians": config.num_gaussians,
        "height": config.height,
        "width": config.width,
        "mean_ms": float(timings_ms.mean()),
        "p50_ms": float(sorted_ms[int(NUM_RUNS * 0.50)]),
        "p95_ms": float(sorted_ms[int(NUM_RUNS * 0.95)]),
        "p99_ms": float(sorted_ms[int(NUM_RUNS * 0.99)]),
        "min_ms": float(timings_ms.min()),
        "max_ms": float(timings_ms.max()),
    }


def _gpu_name() -> str:
    if torch.cuda.is_available():
        raw = torch.cuda.get_device_name(0)
        return raw.replace(" ", "_").replace("/", "_")
    return "cpu"


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available — Rasterizer reference backend is too slow to bench. "
              "Skipping.", file=sys.stderr)
        return 0
    try:
        from gsplat import rasterize_gaussians_sum  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"gsplat not importable — bench requires the CUDA renderer. {e!r}",
              file=sys.stderr)
        return 0

    device = torch.device("cuda")
    renderer = Rasterizer(force_backend="cuda")

    out_dir = Path(__file__).parent / "bench"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"bench_results_{_gpu_name()}.csv"

    print(f"Bench: {len(CONFIGS)} configs × {NUM_RUNS} runs (after {NUM_WARMUP} warmups) "
          f"on {torch.cuda.get_device_name(0)}", file=sys.stderr)

    rows: list[dict] = []
    for cfg in CONFIGS:
        row = _bench_one(cfg, renderer, device)
        rows.append(row)
        print(f"  {row['label']:<22}  "
              f"mean={row['mean_ms']:6.2f}ms  p95={row['p95_ms']:6.2f}ms  "
              f"p99={row['p99_ms']:6.2f}ms",
              file=sys.stderr)

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
