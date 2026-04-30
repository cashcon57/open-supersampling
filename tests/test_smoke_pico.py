"""Run train_pico --smoke-test as a subprocess; verify a checkpoint lands."""
from __future__ import annotations

import subprocess
import sys


def test_smoke_pico_runs(tmp_path) -> None:
    out_dir = tmp_path / "pico_out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ors.train.train_pico",
            "--smoke-test",
            "--out",
            str(out_dir),
            "--sequence-length",
            "4",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"smoke train failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (out_dir / "oru_pico.pth").exists()
