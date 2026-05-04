"""Pre-staged held-out manifest loader for v5 temporal eval.

The manifest file pins the exact TartanAir frame pairs used by
``scripts/sr_temporal_held_out.py`` so that:

  - re-running the eval on a different checkpoint of the same training
    run yields a strictly comparable score (no resampling jitter);
  - the partial-checkpoint learning curve (step 10K / 30K / 60K / 80K)
    can plot mean PSNR/LPIPS as a single curve over identical inputs.

Schema (``manifest_version: 1``):

    {
      "manifest_version": 1,
      "n_pairs": 64,
      "seed": 0,
      "lr_scale": 2.0,
      "lr_synth_args": {
          "enable_jitter": true,
          "enable_taa_blur": true,
          "enable_jpeg": false,
          "jpeg_quality": 85,
          "blur_sigma": 0.5
      },
      "pairs": [
          {"trajectory": "<abs trajectory dir>", "idx_t": 0, "idx_t_plus_1": 1},
          ...
      ]
    }

The freezer script ``scripts/sr_freeze_held_out_manifest.py`` writes
manifests of this shape; this module is the read side.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

__all__ = ["MANIFEST_VERSION", "load_manifest", "manifest_to_pairs"]


MANIFEST_VERSION = 1

# Per-manifest-entry fields that must be present.
_REQUIRED_PAIR_FIELDS = ("trajectory", "idx_t", "idx_t_plus_1")
# Top-level fields the loader validates.
_REQUIRED_TOP_LEVEL = (
    "manifest_version",
    "n_pairs",
    "seed",
    "lr_scale",
    "pairs",
)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest JSON file and validate its top-level shape.

    Returns the parsed manifest dict (with ``pairs`` as a list of dicts).
    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: on schema / version / pair-count mismatches.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"manifest not found: {p}")
    with p.open("r") as f:
        manifest = json.load(f)

    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in manifest]
    if missing:
        raise ValueError(
            f"manifest at {p} is missing required field(s) {missing}"
        )

    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ValueError(
            f"manifest_version mismatch at {p}: got "
            f"{manifest['manifest_version']!r}, expected {MANIFEST_VERSION}"
        )

    pairs = manifest["pairs"]
    if not isinstance(pairs, list):
        raise ValueError(
            f"manifest 'pairs' must be a list; got {type(pairs).__name__}"
        )
    if manifest["n_pairs"] != len(pairs):
        raise ValueError(
            f"manifest n_pairs={manifest['n_pairs']} disagrees with "
            f"len(pairs)={len(pairs)} at {p}"
        )

    for i, entry in enumerate(pairs):
        for field in _REQUIRED_PAIR_FIELDS:
            if field not in entry:
                raise ValueError(
                    f"manifest pair[{i}] is missing field {field!r}"
                )

    # ``lr_synth_args`` is optional in older drafts; default to {} for
    # forward-compat with bare manifests written by hand.
    manifest.setdefault("lr_synth_args", {})

    return manifest


def manifest_to_pairs(
    manifest: Mapping[str, Any], base_dataset: Any
) -> list[tuple[int, int]]:
    """Resolve manifest trajectory references to base-dataset indices.

    For each manifest entry we walk ``base_dataset`` once (O(N) — the
    held-out set is small) building a lookup from
    ``(trajectory_key, frame_index_within_trajectory) -> base_idx``,
    then map every ``(trajectory, idx_t)`` / ``(trajectory, idx_t_plus_1)``
    reference to its base-dataset slot.

    The base dataset must expose:

      - ``__len__()``,
      - ``trajectory_key(idx) -> str`` (matches manifest's ``trajectory``),
      - either ``frame_index(idx) -> int`` (per-trajectory frame number)
        OR a per-trajectory monotonic insertion order, in which case
        ``frame_index`` is inferred from each trajectory's first
        appearance.

    Args:
        manifest: dict from ``load_manifest``.
        base_dataset: e.g. an ``adapt_tartanair(TartanAirGaussianDataset(...))``.

    Returns:
        ``[(idx_t, idx_t_plus_1), ...]`` — list of base-dataset index
        pairs, one per manifest entry, in manifest order.

    Raises:
        KeyError: if any manifest trajectory or per-trajectory frame
            index is not found in ``base_dataset``.
    """
    n = len(base_dataset)

    use_frame_index = hasattr(base_dataset, "frame_index")
    # (trajectory, frame_idx_within_traj) -> base_idx
    lookup: dict[tuple[str, int], int] = {}
    # Track per-trajectory ordinal so we can synthesize a frame index
    # when the dataset doesn't provide ``frame_index``.
    ordinals: dict[str, int] = {}
    for i in range(n):
        traj = base_dataset.trajectory_key(i)
        if use_frame_index:
            fidx = int(base_dataset.frame_index(i))
        else:
            fidx = ordinals.get(traj, 0)
            ordinals[traj] = fidx + 1
        lookup[(traj, fidx)] = i

    pairs: list[tuple[int, int]] = []
    for entry in manifest["pairs"]:
        traj = entry["trajectory"]
        idx_t = int(entry["idx_t"])
        idx_p = int(entry["idx_t_plus_1"])
        try:
            b_t = lookup[(traj, idx_t)]
            b_p = lookup[(traj, idx_p)]
        except KeyError as e:
            raise KeyError(
                f"trajectory/frame {e.args[0]!r} from manifest not found "
                f"in base dataset (saw {len(lookup)} (traj, frame) entries)"
            ) from e
        pairs.append((b_t, b_p))
    return pairs
