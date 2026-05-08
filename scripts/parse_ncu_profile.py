#!/usr/bin/env python3
"""Summarize Nsight Compute CSV export for the Phase 4a baseline doc."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


ALIASES = {
    "duration": [
        "gpu__time_duration.sum",
        "gpu__time_duration.avg",
        "gpu__time_duration",
    ],
    "occupancy": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "achieved_occupancy",
    ],
    "registers": [
        "launch__registers_per_thread",
        "launch__registers_per_thread.avg",
    ],
    "bandwidth": [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "compute": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "tensor": [
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed_pipe_tensor.sum",
        "sm__inst_executed_pipe_tensor.sum",
    ],
}


UNIT_SCALE_TO_US = {
    "ns": 0.001,
    "nsecond": 0.001,
    "nseconds": 0.001,
    "us": 1.0,
    "usecond": 1.0,
    "useconds": 1.0,
    "microsecond": 1.0,
    "microseconds": 1.0,
    "ms": 1000.0,
    "msecond": 1000.0,
    "mseconds": 1000.0,
    "s": 1_000_000.0,
    "second": 1_000_000.0,
    "seconds": 1_000_000.0,
}


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip().replace(",", "").replace("%", "")
    if not text or text.upper() in {"N/A", "NA", "INF", "-INF"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def norm_key(row: dict[str, str], *names: str) -> str:
    lowered = {key.strip().lower(): value for key, value in row.items() if key}
    for name in names:
        value = lowered.get(name.lower())
        if value:
            return value.strip()
    return ""


def duration_to_us(value: float | None, unit: str) -> float | None:
    if value is None:
        return None
    scale = UNIT_SCALE_TO_US.get(unit.strip().lower(), 1.0)
    return value * scale


def metric_bucket(metric_name: str) -> str | None:
    for bucket, names in ALIASES.items():
        if metric_name in names:
            return bucket
    return None


def read_ncu_csv(path: Path) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            kernel = norm_key(row, "Kernel Name", "Kernel", "Name")
            metric = norm_key(row, "Metric Name", "Metric")
            if not kernel or not metric:
                continue
            if metric_bucket(metric) is None:
                continue
            value = parse_number(norm_key(row, "Metric Value", "Value"))
            if value is None:
                continue
            if metric in ALIASES["duration"]:
                value = duration_to_us(value, norm_key(row, "Metric Unit", "Unit"))
            grouped[kernel][metric].append(value)
    return grouped


def aggregate(values: list[float], *, sum_values: bool = False) -> float | None:
    if not values:
        return None
    if sum_values:
        return sum(values)
    return sum(values) / len(values)


def pick_metric(
    metrics: dict[str, list[float]],
    bucket: str,
    *,
    sum_values: bool = False,
) -> float | None:
    for metric_name in ALIASES[bucket]:
        value = aggregate(metrics[metric_name], sum_values=sum_values)
        if value is not None:
            return value
    return None


def fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "TODO"
    if abs(value) >= 100:
        return f"{value:.1f}{suffix}"
    return f"{value:.2f}{suffix}"


def render_markdown(grouped: dict[str, dict[str, list[float]]], top_n: int) -> str:
    rows = []
    for kernel, metrics in grouped.items():
        duration_us = pick_metric(metrics, "duration", sum_values=True)
        rows.append(
            {
                "kernel": kernel,
                "duration_us": duration_us,
                "occupancy": pick_metric(metrics, "occupancy"),
                "registers": pick_metric(metrics, "registers"),
                "bandwidth": pick_metric(metrics, "bandwidth"),
                "compute": pick_metric(metrics, "compute"),
                "tensor": pick_metric(metrics, "tensor"),
            }
        )
    rows.sort(key=lambda row: row["duration_us"] or -1.0, reverse=True)
    rows = rows[:top_n]

    lines = [
        "| Rank | Kernel | Time (us) | Occupancy % | Registers/thread | DRAM or memory throughput % peak | SM compute % peak | Tensor pipe/utilization |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | `{kernel}` | {time} | {occ} | {regs} | {bw} | {compute} | {tensor} |".format(
                rank=index,
                kernel=row["kernel"],
                time=fmt(row["duration_us"]),
                occ=fmt(row["occupancy"], "%"),
                regs=fmt(row["registers"]),
                bw=fmt(row["bandwidth"], "%"),
                compute=fmt(row["compute"], "%"),
                tensor=fmt(row["tensor"], "%" if row["tensor"] is not None else ""),
            )
        )
    if not rows:
        lines.append("| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |")

    hottest = rows[0]["kernel"] if rows else "TODO"
    lines.extend(
        [
            "",
            f"- Hottest kernel: `{hottest}`",
            "- Tensor core utilization note: `TODO` if tensor-pipe counters are instruction counts rather than percent counters.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV from `ncu --import REPORT --page raw --csv`")
    parser.add_argument("--top", type=int, default=5, help="number of kernels to print")
    args = parser.parse_args()

    grouped = read_ncu_csv(args.csv)
    print(render_markdown(grouped, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
