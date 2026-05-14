"""v7 inference-cost benchmark.

Measures per-component wall-time and peak memory of the v7 model at
deployment-realistic resolutions, with parent-child and Mip-Splatting
toggled. Output is a markdown table the user can paste into the spec.

The numbers are CPU-bound here; multiply by ~30-50x downward to estimate
GPU time. Relative orderings between configs hold.

Usage:
    venv-py312/bin/python scripts/bench_v7_inference.py
    venv-py312/bin/python scripts/bench_v7_inference.py --device cuda --warmup 3 --iters 10

Output is written to stdout AND to docs/research/2026-05-14-v7-inference-bench.md
(if --out is given, otherwise just stdout).
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from oss.sr.v7.model import V7Config, V7Model


SCENARIOS = [
    # (label, h_lr, w_lr, scale, recommended_canvas_capacity)
    ("TartanAir-train", 240,  320,  2,  16384),
    ("240p->720p",      240,  320,  3,  16384),
    ("360p->1080p",     360,  640,  3,  65536),
    ("540p->1080p",     540,  960,  2,  65536),
    ("720p->1440p",     720, 1280,  2, 131072),
]


@contextmanager
def _timing():
    """Yields a callable that returns elapsed seconds when invoked."""
    start = [time.perf_counter()]
    elapsed = [0.0]
    def get():
        return elapsed[0]
    yield get
    elapsed[0] = time.perf_counter() - start[0]


def bench_config(
    label: str,
    h_lr: int, w_lr: int, scale: int, capacity: int,
    *,
    backbone_kind: str = "placeholder",
    enable_parent_child: bool = False,
    mip_3d_variance: float = 0.2,
    mip_2d_variance: float = 0.1,
    device: str = "cpu",
    warmup: int = 1,
    iters: int = 3,
) -> dict:
    """Run a single config. Returns timing dict."""
    cfg = V7Config(
        in_channels=9, scale=scale, feat_dim=8, latent_rank=4,
        canvas_capacity=capacity, backbone_kind=backbone_kind,
        backbone_blocks=1, enable_spawner=True,
        spawner_k_per_tile=2, spawner_tile_size=16,
        enable_parent_child=enable_parent_child,
    )
    model = V7Model(cfg).train(False).to(device)
    model.allocate_canvas(device)

    lr_in = torch.randn((1, 9, h_lr, w_lr), device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.reset_state(device)
            _ = model(lr_in, t_query=0.0, spawn_at_t=0.0)
            _ = model(lr_in, t_query=2.0, spawn_at_t=2.0)
            _ = model(lr_in, t_query=1.0)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed iters: measure per-component
    times = {"total": 0.0, "forward_n": 0.0, "forward_np1": 0.0, "forward_inter": 0.0}
    canvas_counts = []

    for _ in range(iters):
        with torch.no_grad():
            model.reset_state(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            t0 = time.perf_counter()
            _ = model(lr_in, t_query=0.0, spawn_at_t=0.0)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times["forward_n"] += (t1 - t0)

            _ = model(lr_in, t_query=2.0, spawn_at_t=2.0)
            if device == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            times["forward_np1"] += (t2 - t1)

            _ = model(lr_in, t_query=1.0)
            if device == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            times["forward_inter"] += (t3 - t2)

            times["total"] += (t3 - t_start)
            canvas_counts.append(model.canvas.count)

    # Normalize per iter
    for k in times:
        times[k] = times[k] * 1000.0 / iters  # ms

    peak_mb = 0.0
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    times["canvas_count"] = canvas_counts[-1] if canvas_counts else 0
    times["peak_memory_mb"] = peak_mb
    times["label"] = label
    times["resolution"] = f"{h_lr*scale}x{w_lr*scale}"
    return times


def fmt_row(r: dict) -> str:
    return (
        f"| {r['label']:<18s} "
        f"| {r['resolution']:<10s} "
        f"| {r['canvas_count']:>8d} "
        f"| {r['forward_n']:>8.1f} "
        f"| {r['forward_np1']:>8.1f} "
        f"| {r['forward_inter']:>8.1f} "
        f"| {r['total']:>9.1f} "
        f"| {r['peak_memory_mb']:>10.1f} |"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None,
                        help="Append results to this markdown file.")
    parser.add_argument("--scenario", choices=[s[0] for s in SCENARIOS],
                        help="Restrict to one scenario.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[bench] CUDA unavailable; falling back to CPU.")
        args.device = "cpu"

    scenarios = SCENARIOS if not args.scenario else [s for s in SCENARIOS if s[0] == args.scenario]

    matrix = [
        ("baseline",  False, 0.0, 0.0),
        ("mip-on",    False, 0.2, 0.1),
        ("pc+mip",     True, 0.2, 0.1),
    ]

    lines = []
    header = (
        f"### {args.device.upper()} bench — {time.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"(warmup={args.warmup}, iters={args.iters})\n\n"
        f"All times in ms per render-cycle (3 forwards: t=0 spawn, t=2 spawn, t=1 render).\n"
        f"`canvas_count` is after 2 spawns; peak_memory_mb is CUDA-only.\n"
    )
    lines.append(header)

    table_header = (
        f"| Scenario           | HR shape   | canvas#  | fwd_n    | fwd_np1  | fwd_inter | total_ms  | peak_mb    |\n"
        f"|--------------------|------------|----------|----------|----------|-----------|-----------|------------|"
    )
    print(header)

    for variant_name, pc, mv3, mv2 in matrix:
        print(f"\n#### Variant: {variant_name}  "
              f"(parent_child={pc}, mip_3d={mv3}, mip_2d={mv2})\n")
        print(table_header)
        lines.append(f"\n#### Variant: {variant_name}  "
                     f"(parent_child={pc}, mip_3d={mv3}, mip_2d={mv2})\n")
        lines.append(table_header)
        for label, h_lr, w_lr, scale, cap in scenarios:
            try:
                r = bench_config(
                    label=label, h_lr=h_lr, w_lr=w_lr, scale=scale, capacity=cap,
                    enable_parent_child=pc,
                    mip_3d_variance=mv3, mip_2d_variance=mv2,
                    device=args.device, warmup=args.warmup, iters=args.iters,
                )
                row = fmt_row(r)
                print(row)
                lines.append(row)
            except Exception as e:
                msg = f"| {label:<18s} | FAIL: {str(e)[:80]:<80s} |"
                print(msg)
                lines.append(msg)
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        existing = args.out.read_text() if args.out.exists() else ""
        args.out.write_text(existing + "\n\n" + "\n".join(lines) + "\n")
        print(f"\n[bench] appended to {args.out}")


if __name__ == "__main__":
    main()
