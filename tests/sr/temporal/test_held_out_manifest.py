"""Tests for the v5 pre-staged held-out manifest.

The manifest freezes the (trajectory, idx_t, idx_t_plus_1) frame-pair list
that ``scripts/sr_temporal_held_out.py`` and the partial-checkpoint
learning-curve eval will replay deterministically.

Coverage:

  - Manifest round-trip: write a synthetic manifest with the expected
    schema, load it back via ``load_manifest``, verify field shapes and
    that ``seed`` / ``lr_scale`` / ``manifest_version`` survive.
  - ``manifest_to_pairs`` resolves trajectory paths to base-dataset
    indices correctly when the underlying dataset reports those paths via
    ``trajectory_key`` (we use a stub base dataset here so the test runs
    without TartanAir on disk).
  - ``manifest_to_pairs`` raises a clear error if a manifest references
    a trajectory the base dataset doesn't expose.
  - The freezer script's ``--help`` exits 0 (argparse + import-time
    smoke; the actual freezing requires real data and is gated behind
    ``--tartanair-root``).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from oss.sr.temporal.held_out_manifest import (
    load_manifest,
    manifest_to_pairs,
)


# ---------------------------------------------------------------------------
# Manifest round-trip
# ---------------------------------------------------------------------------


def _synth_manifest() -> dict:
    return {
        "manifest_version": 1,
        "n_pairs": 3,
        "seed": 0,
        "lr_scale": 2.0,
        "lr_synth_args": {
            "enable_jitter": True,
            "enable_taa_blur": True,
            "enable_jpeg": False,
            "jpeg_quality": 85,
            "blur_sigma": 0.5,
        },
        "pairs": [
            {"trajectory": "/data/env/level/P000", "idx_t": 0, "idx_t_plus_1": 1},
            {"trajectory": "/data/env/level/P000", "idx_t": 5, "idx_t_plus_1": 6},
            {"trajectory": "/data/env/level/P001", "idx_t": 2, "idx_t_plus_1": 3},
        ],
    }


def test_manifest_round_trip_preserves_fields(tmp_path: Path) -> None:
    """``load_manifest`` recovers every documented top-level field."""
    src = _synth_manifest()
    p = tmp_path / "v5_held_out_manifest.json"
    p.write_text(json.dumps(src, indent=2))

    loaded = load_manifest(p)

    assert loaded["manifest_version"] == 1
    assert loaded["n_pairs"] == 3
    assert loaded["seed"] == 0
    assert loaded["lr_scale"] == pytest.approx(2.0)
    assert loaded["lr_synth_args"] == src["lr_synth_args"]
    assert isinstance(loaded["pairs"], list)
    assert len(loaded["pairs"]) == 3
    for entry in loaded["pairs"]:
        assert set(entry.keys()) >= {"trajectory", "idx_t", "idx_t_plus_1"}
        assert isinstance(entry["trajectory"], str)
        assert isinstance(entry["idx_t"], int)
        assert isinstance(entry["idx_t_plus_1"], int)
        assert entry["idx_t_plus_1"] == entry["idx_t"] + 1


def test_load_manifest_rejects_unknown_version(tmp_path: Path) -> None:
    bad = _synth_manifest()
    bad["manifest_version"] = 999
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="manifest_version"):
        load_manifest(p)


def test_load_manifest_validates_pair_count(tmp_path: Path) -> None:
    bad = _synth_manifest()
    bad["n_pairs"] = 99  # claim more pairs than the list contains
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="n_pairs"):
        load_manifest(p)


# ---------------------------------------------------------------------------
# manifest_to_pairs
# ---------------------------------------------------------------------------


class _StubDataset:
    """Stand-in for TartanAirGaussianDataset.

    Exposes ``trajectory_key(idx) -> str`` matching the manifest's
    ``trajectory`` field. The actual data is irrelevant for resolution.
    """

    def __init__(self, frames: list[tuple[str, int]]) -> None:
        # frames is list of (trajectory, frame_idx_within_traj).
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def trajectory_key(self, idx: int) -> str:
        return self._frames[idx][0]

    def frame_index(self, idx: int) -> int:
        # Public helper so the resolver can match (trajectory, frame_index)
        # tuples against manifest entries.
        return self._frames[idx][1]


def _make_stub() -> _StubDataset:
    # Two trajectories, three frames each, interleaved insertion order.
    return _StubDataset(
        [
            ("/data/env/level/P000", 0),
            ("/data/env/level/P000", 1),
            ("/data/env/level/P001", 0),
            ("/data/env/level/P000", 5),
            ("/data/env/level/P000", 6),
            ("/data/env/level/P001", 2),
            ("/data/env/level/P001", 3),
        ]
    )


def test_manifest_to_pairs_resolves_indices() -> None:
    """Each manifest entry resolves to (idx_t, idx_t+1) base-dataset slots."""
    manifest = _synth_manifest()
    base = _make_stub()
    pairs = manifest_to_pairs(manifest, base)

    # Same number of pairs as manifest entries.
    assert len(pairs) == len(manifest["pairs"])

    # First entry: (P000, frame 0) -> base idx 0; (P000, frame 1) -> base idx 1.
    assert pairs[0] == (0, 1)
    # Second entry: (P000, frame 5) -> base idx 3; (P000, frame 6) -> base idx 4.
    assert pairs[1] == (3, 4)
    # Third entry: (P001, frame 2) -> base idx 5; (P001, frame 3) -> base idx 6.
    assert pairs[2] == (5, 6)


def test_manifest_to_pairs_raises_on_missing_trajectory() -> None:
    manifest = _synth_manifest()
    manifest["pairs"][0]["trajectory"] = "/data/does/not/exist"
    base = _make_stub()
    with pytest.raises(KeyError, match="not found"):
        manifest_to_pairs(manifest, base)


# ---------------------------------------------------------------------------
# Freezer script smoke
# ---------------------------------------------------------------------------


def test_freezer_script_help_exits_zero() -> None:
    """``python scripts/sr_freeze_held_out_manifest.py --help`` must exit 0."""
    proc = subprocess.run(
        [sys.executable, "scripts/sr_freeze_held_out_manifest.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"--help exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    out = proc.stdout.lower()
    assert "usage" in out
    # All advertised flags must be discoverable from --help.
    for flag in (
        "--dataset-kind",
        "--output",
        "--tartanair-root",
        "--sintel-root",
        "--n-pairs",
        "--seed",
        "--lr-scale",
    ):
        assert flag in proc.stdout, f"flag {flag!r} missing from --help:\n{proc.stdout}"
