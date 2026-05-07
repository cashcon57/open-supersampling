"""Backfill dashboard score rows for v5 temporal checkpoints.

This is a thin batch wrapper around ``scripts/sr_temporal_held_out.py``. It
keeps the battle-tested v5 eval path, but avoids that script's shared
``held_out_results.json`` side effect so multiple checkpoint slices can run in
parallel and write independent ``score_log.slice-*.json`` files.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import sr_temporal_held_out as held_out  # noqa: E402


def _step_from_ckpt(path: Path) -> int:
    m = re.search(r"step-(\d+)\.pt$", path.name)
    if not m:
        raise ValueError(f"checkpoint name does not contain a step: {path}")
    return int(m.group(1))


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [dict(row) for row in data]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(rows, f, indent=2)
    tmp.replace(path)


def _row_from_result(
    *,
    ckpt: Path,
    baseline: Path,
    manifest: Path,
    result: dict[str, list[float]],
) -> dict[str, Any]:
    psnr_model = result["psnr_temporal"]
    psnr_bicubic = result["psnr_bicubic"]
    lpips_model = result["lpips_temporal"]
    lpips_bicubic = result["lpips_bicubic"]
    return {
        "step": _step_from_ckpt(ckpt),
        "model_psnr_mean": _mean(psnr_model),
        "bicubic_psnr_mean": _mean(psnr_bicubic),
        "model_lpips_mean": _mean(lpips_model),
        "bicubic_lpips_mean": _mean(lpips_bicubic),
        "model_beats_bicubic_count": sum(
            1 for m, b in zip(psnr_model, psnr_bicubic) if m > b
        ),
        "model_beats_bicubic_lpips_count": sum(
            1 for m, b in zip(lpips_model, lpips_bicubic) if m < b
        ),
        "n_samples": len(psnr_model),
        "manifest": str(manifest),
        "ckpt": str(ckpt),
        "ckpt_baseline": str(baseline),
    }


def _parse_ckpts(values: list[str]) -> list[Path]:
    ckpts: list[Path] = []
    for value in values:
        if "," in value:
            ckpts.extend(Path(part.strip()) for part in value.split(",") if part.strip())
        else:
            ckpts.append(Path(value))
    return ckpts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpts", nargs="+", required=True, help="Checkpoint paths to evaluate.")
    p.add_argument("--ckpt-baseline", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True, help="Per-slice score rows JSON.")
    p.add_argument("--tartanair-root", type=Path, default=None)
    p.add_argument("--sintel-root", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=float, default=held_out.DEFAULT_SCALE)
    p.add_argument("--enable-jpeg", action="store_true")
    p.add_argument("--blur-sigma", type=float, default=0.5)
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--resume", action="store_true", help="Keep existing rows in --out-json.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tartanair_root is None and args.sintel_root is None:
        print("FAIL: provide at least one of --tartanair-root / --sintel-root")
        return 1

    import torch

    torch.manual_seed(args.seed)
    ckpts = sorted(_parse_ckpts(args.ckpts), key=_step_from_ckpt)
    existing = _read_rows(args.out_json) if args.resume else []
    rows_by_step = {int(row["step"]): row for row in existing}

    manifest_paths = held_out._split_manifest_paths(args.manifest)
    lr_synth_args = held_out._lr_synth_args_from_cli(args)
    loaders = held_out._build_manifest_loaders(
        manifest_paths,
        tartanair_root=args.tartanair_root,
        sintel_root=args.sintel_root,
        batch_size=args.batch_size,
        scale=args.scale,
        lr_synth_args=lr_synth_args,
    )
    if not loaders:
        print("FAIL: no manifest loaders produced samples")
        return 1

    print(f"Loading baseline: {args.ckpt_baseline}")
    model_baseline = held_out._load_baseline(args.ckpt_baseline, args.device)

    lpips_fn = None
    try:
        import lpips  # type: ignore[import-not-found]

        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(args.device)
        lpips_fn.train(False)
    except Exception as e:
        print(f"WARN: LPIPS unavailable ({e}) - LPIPS columns will be NaN")

    per_loader = max(1, args.n_samples)
    for ckpt in ckpts:
        step = _step_from_ckpt(ckpt)
        if step in rows_by_step:
            print(f"skip existing step={step}")
            continue
        print(f"Loading temporal: {ckpt}")
        model_temporal = held_out._load_temporal(ckpt, args.device)
        per_dataset_results: dict[str, dict[str, list[float]]] = {}
        for name, loader in loaders:
            print(f"-- evaluating {name} step={step} (target {per_loader}) --")
            per_dataset_results[name] = held_out._eval_loader(
                loader,
                model_temporal=model_temporal,
                model_baseline=model_baseline,
                lpips_fn=lpips_fn,
                n_samples_remaining=per_loader,
                device=args.device,
            )
        merged = held_out._merge_results(*per_dataset_results.values())
        row = _row_from_result(
            ckpt=ckpt,
            baseline=args.ckpt_baseline,
            manifest=args.manifest,
            result=merged,
        )
        if math.isnan(row["model_lpips_mean"]):
            print(f"FAIL: LPIPS was unavailable for step={step}")
            return 1
        rows_by_step[step] = row
        _write_rows(args.out_json, [rows_by_step[k] for k in sorted(rows_by_step)])
        print(
            f"held-out: step={step} "
            f"model_psnr={row['model_psnr_mean']:.3f} "
            f"bicubic_psnr={row['bicubic_psnr_mean']:.3f} "
            f"model_lpips={row['model_lpips_mean']:.4f} "
            f"bicubic_lpips={row['bicubic_lpips_mean']:.4f} "
            f"beats={row['model_beats_bicubic_count']}/{row['n_samples']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
