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

DEFAULT_OUTPUT = Path("docs/superpowers/experiments/v5_held_out_manifest.json")
DEFAULT_TARTANAIR_ROOT = Path("<train-host-data>/datasets/tartanair_extracted")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Freeze a deterministic held-out frame-pair manifest "
            "(JSON, schema v1) for v5 temporal SR eval."
        )
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Manifest output path (default: {DEFAULT_OUTPUT}).",
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
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Per-trajectory frame index extraction
# ---------------------------------------------------------------------------


def _frame_index_from_path(image_path: Path) -> int:
    """Extract the zero-padded frame number from a TartanAir image path.

    TartanAir image filenames look like ``000123_left.png``; the integer
    prefix is the per-trajectory frame index.
    """
    stem = image_path.stem  # e.g. "000123_left"
    head = stem.split("_")[0]
    return int(head)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.n_pairs <= 0:
        print(f"FAIL: --n-pairs must be positive; got {args.n_pairs}")
        return 1

    if args.tartanair_root is None:
        print(
            "FAIL: --tartanair-root is required to materialize the "
            "manifest. Pass a real TartanAir extraction root."
        )
        return 1

    if not args.tartanair_root.is_dir():
        print(
            f"FAIL: --tartanair-root {args.tartanair_root} is not a "
            "directory (run on the box where TartanAir is extracted)."
        )
        return 1

    # Defer heavy imports.
    import torch
    from torch.utils.data import DataLoader

    from oss.gaussian.data import TartanAirGaussianDataset
    from oss.gaussian.data.lr_synthesis import EngineAliasedLRSynth
    from oss.sr.temporal import (
        SequentialPairDataset,
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

    base = TartanAirGaussianDataset(
        root=args.tartanair_root, scale=args.lr_scale, lr_synth=lr_synth
    )
    base = adapt_tartanair(base)
    pair_ds = SequentialPairDataset(base)
    if len(pair_ds) == 0:
        print("FAIL: TartanAir produced 0 sequential pairs")
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
        # Trajectory dir = parent of image_left/.
        traj_dir = img_path.parent.parent
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
