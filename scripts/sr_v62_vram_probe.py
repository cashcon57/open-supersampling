#!/usr/bin/env python
"""Measure v6.2 inference VRAM at target output resolutions.

This follows the existing ``scripts/sr_inference_vram.py`` pattern, but uses
the v6 orchestrator directly and writes both a compact JSON result file and
per-resolution ``torch.cuda.memory_summary()`` dumps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from oss.sr.v6.model import V6Config, V6Model  # noqa: E402


CASES = (
    ("1080p output", 540, 960),
    ("1440p output", 720, 1280),
    ("4K output", 1080, 1920),
)


def _latest_checkpoint(output_dir: Path) -> Path:
    ckpts = sorted(output_dir.glob("step-*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no step-*.pt checkpoints in {output_dir}")
    return ckpts[-1]


def _require_v62_config_surface() -> None:
    if not is_dataclass(V6Config):
        raise RuntimeError("V6Config is not a dataclass; cannot verify v6.2 fields")
    names = {f.name for f in fields(V6Config)}
    missing = {"latent_rank", "spawner_mode", "fusion_mode"} - names
    if missing:
        raise RuntimeError(
            "v6.2 model wiring is not present in V6Config; missing fields: "
            + ", ".join(sorted(missing))
        )


def _load_state(
    model: V6Model,
    ckpt: Path,
    device: torch.device,
    strict: bool,
) -> dict[str, Any]:
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(payload).__name__}")
    state = (
        payload.get("generator")
        or payload.get("model")
        or payload.get("state_dict")
        or payload
    )
    result = model.load_state_dict(state, strict=strict)
    return {
        "path": str(ckpt),
        "step": payload.get("step"),
        "kind": payload.get("kind"),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def _dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _memory_stats(device: torch.device) -> dict[str, float | int]:
    stats = torch.cuda.memory_stats(device)
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "max_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "active_peak_mib": stats.get("active_bytes.all.peak", 0) / 1024**2,
        "requested_peak_mib": stats.get("requested_bytes.all.peak", 0) / 1024**2,
        "inactive_split_peak_mib": stats.get("inactive_split_bytes.all.peak", 0)
        / 1024**2,
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
        "num_ooms": int(stats.get("num_ooms", 0)),
    }


def measure_case(
    model: V6Model,
    *,
    label: str,
    lr_h: int,
    lr_w: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    runs: int,
    summary_dir: Path,
) -> dict[str, Any]:
    model.eval()
    model.reset_state(device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    x = torch.zeros((1, model.cfg.in_channels, lr_h, lr_w), device=device, dtype=dtype)
    motion = torch.zeros((1, 2, lr_h, lr_w), device=device, dtype=dtype)

    with torch.inference_mode():
        for i in range(warmup):
            _ = model(x, motion_lr=None if i == 0 else motion, frame_index=i)
        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        start = time.monotonic()
        out = None
        for i in range(runs):
            out = model(x, motion_lr=motion, frame_index=warmup + i)
        torch.cuda.synchronize(device)
        elapsed = (time.monotonic() - start) / runs

    if out is None:
        raise RuntimeError("runs must be >= 1")

    stats = _memory_stats(device)
    summary_path = summary_dir / f"{label.lower().replace(' ', '-')}.memory_summary.txt"
    summary_path.write_text(torch.cuda.memory_summary(device=device), encoding="utf-8")

    return {
        "label": label,
        "lr_hw": [lr_h, lr_w],
        "output_hw": list(out.shape[-2:]),
        "ms_per_frame": elapsed * 1000.0,
        "fps": 1.0 / elapsed,
        "memory_summary": str(summary_path),
        **stats,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="Explicit v6.2 checkpoint.")
    src.add_argument("--output-dir", type=Path, help="Directory containing step-*.pt.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    p.add_argument(
        "--v62",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the v6.2 path: R-latent raster + concat fusion + disocclusion spawner.",
    )
    p.add_argument("--backbone", default="hat-tiny")
    p.add_argument("--latent-rank", type=int, default=4)
    p.add_argument("--canvas-capacity", type=int, default=16_000)
    p.add_argument("--in-channels", type=int, default=9)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("artifacts/v62-vram"),
        help="Output directory for JSON and memory_summary dumps.",
    )
    args = p.parse_args()

    if not args.v62:
        print("FAIL: this probe only measures the v6.2 path; pass --v62")
        return 1

    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is false")
        return 1
    device = torch.device(args.device)
    if device.type != "cuda":
        print(f"FAIL: this probe requires a CUDA device, got {device}")
        return 1

    try:
        _require_v62_config_surface()
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1

    checkpoint = args.checkpoint or _latest_checkpoint(args.output_dir)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    cfg = V6Config(
        in_channels=args.in_channels,
        scale=args.scale,
        backbone=args.backbone,
        canvas_capacity=args.canvas_capacity,
        latent_rank=args.latent_rank,
        spawner_mode="disocclusion_only",
        fusion_mode="concat",
    )
    dtype = _dtype(args.dtype)
    model = V6Model(cfg).to(device=device, dtype=dtype)
    ckpt_info = _load_state(model, checkpoint, device, strict=args.strict)

    results = []
    for label, lr_h, lr_w in CASES:
        try:
            results.append(
                measure_case(
                    model,
                    label=label,
                    lr_h=lr_h,
                    lr_w=lr_w,
                    device=device,
                    dtype=dtype,
                    warmup=args.warmup,
                    runs=args.runs,
                    summary_dir=args.summary_dir,
                )
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            results.append(
                {"label": label, "lr_hw": [lr_h, lr_w], "error": f"OOM: {exc}"}
            )

    payload = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "config": {
            "backbone": args.backbone,
            "latent_rank": args.latent_rank,
            "canvas_capacity": args.canvas_capacity,
            "fusion_mode": "concat",
            "spawner_mode": "disocclusion_only",
            "scale": args.scale,
            "in_channels": args.in_channels,
        },
        "checkpoint": ckpt_info,
        "results": results,
    }
    json_path = args.summary_dir / "v62-vram-results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"checkpoint: {checkpoint}")
    print(f"device: {payload['device']}  torch={payload['torch']}  dtype={args.dtype}")
    print(
        f"{'case':16s} {'LR':>12s} {'output':>12s} {'peak alloc':>12s} "
        f"{'peak reserved':>14s} {'ms/frame':>10s}"
    )
    print("-" * 84)
    for row in results:
        if "error" in row:
            print(
                f"{row['label']:16s} {str(row['lr_hw']):>12s} "
                f"{'ERROR':>12s} {row['error']}"
            )
            continue
        print(
            f"{row['label']:16s} {str(row['lr_hw']):>12s} "
            f"{str(row['output_hw']):>12s} {row['max_allocated_mib']:10.1f} MiB "
            f"{row['max_reserved_mib']:12.1f} MiB {row['ms_per_frame']:9.2f}"
        )
    print(f"json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
