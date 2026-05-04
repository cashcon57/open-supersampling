"""Argparse smoke-test for the held-out eval entry point.

The full eval requires real datasets + checkpoints, so we can't exercise it
end-to-end on CI. The minimum we guarantee is that the script imports cleanly
and ``--help`` exits 0 — i.e. argparse is well-formed and there are no
import-time errors. This is the verification gate Task 8 of the
v5-pixel-temporal plan calls out.
"""
from __future__ import annotations

import subprocess
import sys


def test_held_out_help_exits_zero() -> None:
    """``python scripts/sr_temporal_held_out.py --help`` must exit 0."""
    proc = subprocess.run(
        [sys.executable, "scripts/sr_temporal_held_out.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"--help exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    # Sanity-check argparse actually produced usage text.
    assert "usage" in proc.stdout.lower(), (
        f"argparse usage missing from --help output:\n{proc.stdout}"
    )
