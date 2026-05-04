"""Smoke-test the training entry runs end-to-end on CPU."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_smoke_train(tmp_path: Path) -> None:
    out = tmp_path / "smoke"
    rc = subprocess.run(
        [sys.executable, "scripts/sr_train_temporal.py",
         "--smoke", "--device", "cpu", "--max-steps", "5",
         "--output-dir", str(out)],
        check=False,
    ).returncode
    assert rc == 0, "smoke train returned non-zero"
    assert (out / "metrics.json").exists()
    assert (out / "score_log.json").exists()
    assert any(out.glob("step-*.pt"))
