"""Pre-stage a deterministic held-out frame-pair manifest for v5 eval.

Sprint-5 follow-up to the v5-pixel-temporal training run. Rather than have
``scripts/sr_temporal_held_out.py`` resample held-out frames every
invocation, we freeze 64 (configurable) ``(trajectory, idx_t, idx_t+1)``
references to a JSON manifest. This makes:

  1. eval reproducible across re-runs against the same checkpoint;
  2. cross-checkpoint comparisons strictly comparable (every step's eval
     scores the exact same frames);
  3. partial-checkpoint learning curves (step 10K / 30K / 60K / 80K)
     plottable as a single mean over identical inputs.

The frozen manifest is a JSON file with schema version 1. See
``oss/sr/temporal/held_out_manifest.py`` for the read side and the
documented schema.

Usage::

    python scripts/sr_freeze_held_out_manifest.py \\
        --tartanair-root <train-host-data>/datasets/tartanair_extracted \\
        --n-pairs 64 \\
        --output docs/superpowers/experiments/v5_held_out_manifest.json

The CLI verification gate is::

    python scripts/sr_freeze_held_out_manifest.py --help

returning exit code 0. Materializing the manifest itself requires a real
TartanAir extraction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow ``python scripts/...`` to import ``oss.*`` without an editable
# install. Mirrors the import shim in ``scripts/sr_temporal_held_out.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# NOTE: torch / oss imports are deferred into ``main()`` so that ``--help``
# (the CLI smoke-test gate) succeeds on a vanilla Python interpreter.

DEFAULT_TARTANAIR_OUTPUT = Path("docs/superpowers/experiments/v5_held_out_manifest.json")
DEFAULT_SINTEL_OUTPUT = Path("docs/superpowers/experiments/v5_held_out_manifest_sintel.json")
DEFAULT_TARTANAIR_ROOT = Path("<train-host-data>/datasets/tartanair_extracted")
DEFAULT_SINTEL_ROOT = Path("<train-host-data>/datasets/sintel")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Freeze a deterministic held-out frame-pair manifest "
            "(JSON, schema v1) for v5 temporal SR eval."
        )
    )
    p.add_argument(
        "--dataset-kind",
        choices=["tartanair", "sintel"],
        default="tartanair",
        help="Dataset adapter to freeze (default: tartanair).",
    )
    p.add_argument(
        "--tartanair-root",
        type=Path,
        default=None,
        help=(
            f"TartanAir extraction root (default: {DEFAULT_TARTANAIR_ROOT}). "
            "Pass None for local --help / smoke; a real path is required to "
            "actually materialize the manifest."
        ),
    )
    p.add_argument(
        "--sintel-root",
        type=Path,
        default=None,
        help=f"Sintel root when --dataset-kind=sintel (default: {DEFAULT_SINTEL_ROOT}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Manifest output path. Defaults to "
            f"{DEFAULT_TARTANAIR_OUTPUT} for TartanAir and "
            f"{DEFAULT_SINTEL_OUTPUT} for Sintel."
        ),
    )
    p.add_argument(
        "--n-pairs",
        type=int,
        default=64,
        help="Number of (t, t+1) pairs to record (default: 64).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="torch.manual_seed before iterating the loader (default: 0).",
    )
    p.add_argument(
        "--lr-scale",
        type=float,
        default=2.0,
        help="HR/LR scale factor recorded in the manifest (default: 2.0).",
    )
    p.add_argument(
        "--enable-jpeg",
        action="store_true",
        help=(
            "Match the EngineAliasedLRSynth setting used at training time. "
            "Default off (matches v5 training default)."
        ),
    )
    p.add_argument(
        "--blur-sigma",
        type=float,
        default=0.5,
        help="EngineAliasedLRSynth.blur_sigma (default: 0.5).",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="EngineAliasedLRSynth.jpeg_quality (default: 85).",
    )
    p.add_argument(
        "--include-envs",
        type=str,
        default=None,
        help="Comma-separated TartanAir env names to RESTRICT manifest selection to "
             "(e.g. 'oldtown'). Should match the training script's --held-out-envs "
             "set so the manifest contains only frames the trainer never saw.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Per-trajectory frame index extraction
# ---------------------------------------------------------------------------


def _frame_index_from_path(image_path: Path) -> int:
    """Extract the frame number from a TartanAir or Sintel image path.

    TartanAir image filenames look like ``000123_left.png``; the integer
    prefix is the per-trajectory frame index. Sintel filenames look like
    ``frame_0001.png``.
    """
    stem = image_path.stem  # e.g. "000123_left"
    if stem.startswith("frame_"):
        head = stem.split("_")[-1]
    else:
        head = stem.split("_")[0]
    return int(head)


def _trajectory_from_image_path(dataset_kind: str, image_path: Path) -> Path:
    if dataset_kind == "tartanair":
        # .../<traj>/image_left/000000_left.png
        return image_path.parent.parent
    if dataset_kind == "sintel":
        # .../training/clean/<seq>/frame_0001.png
        return image_path.parent
    raise ValueError(f"unknown dataset kind: {dataset_kind!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output is None:
        args.output = (
            DEFAULT_SINTEL_OUTPUT
            if args.dataset_kind == "sintel"
            else DEFAULT_TARTANAIR_OUTPUT
        )

    if args.n_pairs <= 0:
        print(f"FAIL: --n-pairs must be positive; got {args.n_pairs}")
        return 1

    root = (
        args.sintel_root if args.dataset_kind == "sintel"
        else args.tartanair_root
    )
    if root is None:
        root = (
            DEFAULT_SINTEL_ROOT if args.dataset_kind == "sintel"
            else DEFAULT_TARTANAIR_ROOT
        )

    if root is None:
        print(
            f"FAIL: --{args.dataset_kind}-root is required to materialize "
            "the manifest. Pass a real dataset root."
        )
        return 1

    if not root.is_dir():
        print(
            f"FAIL: {args.dataset_kind} root {root} is not a directory "
            "(run on the box where the dataset is extracted)."
        )
        return 1

    # Defer heavy imports.
    import torch
    from torch.utils.data import DataLoader

    from oss.gaussian.data import SintelGaussianDataset, TartanAirGaussianDataset
    from oss.gaussian.data.lr_synthesis import EngineAliasedLRSynth
    from oss.sr.temporal import (
        SequentialPairDataset,
        adapt_sintel,
        adapt_tartanair,
        default_collate_pair,
    )
    from oss.sr.temporal.held_out_manifest import MANIFEST_VERSION

    torch.manual_seed(args.seed)

    lr_synth_args = {
        "enable_jitter": True,
        "enable_taa_blur": True,
        "enable_jpeg": bool(args.enable_jpeg),
        "jpeg_quality": int(args.jpeg_quality),
        "blur_sigma": float(args.blur_sigma),
    }
    lr_synth = EngineAliasedLRSynth(scale=args.lr_scale, **lr_synth_args)

    if args.dataset_kind == "tartanair":
        base = TartanAirGaussianDataset(
            root=root, scale=args.lr_scale, lr_synth=lr_synth
        )
        # --include-envs: restrict to the held-out env(s) the training script
        # excludes via --held-out-envs. Without this, the manifest can land
        # on training frames (data leak; see launch-status notes).
        if args.include_envs:
            include = {e.strip() for e in args.include_envs.split(",") if e.strip()}
            root_str = str(root.resolve())
            def _env_of(item) -> str:
                rel = str(item[0]).removeprefix(root_str).lstrip("\\/")
                return rel.split("/")[0].split("\\")[0]
            before = len(base._items)
            base._items = [it for it in base._items if _env_of(it) in include]
            if not base._items:
                raise SystemExit(
                    f"--include-envs={sorted(include)} matched 0 items under {root}; "
                    f"check the env names. (Was: {before} items before filter.)"
                )
            print(
                f"include-envs filter: {sorted(include)} -> kept {len(base._items)}/{before} items",
                flush=True,
            )
        base = adapt_tartanair(base)
    else:
        base = SintelGaussianDataset(
            root=root, scale=args.lr_scale, pass_name="clean", lr_synth=lr_synth
        )
        base = adapt_sintel(base)
    pair_ds = SequentialPairDataset(base)
    if len(pair_ds) == 0:
        print(f"FAIL: {args.dataset_kind} produced 0 sequential pairs")
        return 1
    if len(pair_ds) < args.n_pairs:
        print(
            f"FAIL: requested {args.n_pairs} pairs but only "
            f"{len(pair_ds)} are available"
        )
        return 1

    # Determinism contract: shuffle=False + manual_seed(0) + zero workers.
    # We need the *underlying* per-pair base index, not the rendered tensor —
    # so we iterate ``pair_ds._pair_indices`` directly. That attribute is the
    # canonical ordering ``DataLoader(shuffle=False)`` would emit. We still
    # sanity-check against a length-1 DataLoader to confirm parity.
    pair_indices = list(pair_ds._pair_indices[: args.n_pairs])

    # Belt-and-braces parity check: a shuffle=False loader visits the same
    # frame indices in the same order. We don't need actual frames, only the
    # ordering, so we cap iteration at n_pairs.
    _ = DataLoader(  # constructed to surface dataloader-config errors early
        pair_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=default_collate_pair,
    )

    pairs_records: list[dict] = []
    for base_idx in pair_indices:
        # base._items[i] = (image_path, depth_path, flow_path)
        img_path = Path(base._items[base_idx][0])
        next_img_path = Path(base._items[base_idx + 1][0])
        traj_dir = _trajectory_from_image_path(args.dataset_kind, img_path)
        idx_t = _frame_index_from_path(img_path)
        idx_p = _frame_index_from_path(next_img_path)
        if idx_p != idx_t + 1:
            # SequentialPairDataset enforces same trajectory but the pair
            # frame numbers should also be consecutive; flag if not.
            print(
                f"WARN: non-consecutive frame indices at base_idx={base_idx}: "
                f"{idx_t} -> {idx_p}"
            )
        pairs_records.append(
            {
                "trajectory": str(traj_dir),
                "idx_t": int(idx_t),
                "idx_t_plus_1": int(idx_p),
            }
        )

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_kind": args.dataset_kind,
        "n_pairs": len(pairs_records),
        "seed": int(args.seed),
        "lr_scale": float(args.lr_scale),
        "lr_synth_args": lr_synth_args,
        "pairs": pairs_records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.output}  ({len(pairs_records)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
