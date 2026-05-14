from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_public_dashboard as dashboard


ROOT = Path(__file__).resolve().parents[1]


def _build_fixture(tmp_path: Path) -> Path:
    out_dir = tmp_path / "dashboard"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_public_dashboard.py"),
            "--runs-dir",
            str(ROOT / "tests" / "fixtures" / "public-dashboard" / "runs"),
            "--out",
            str(out_dir),
        ],
        check=True,
        cwd=ROOT,
    )
    return out_dir / "data.json"


def test_public_dashboard_fixture_passes_schema(tmp_path: Path) -> None:
    data_json = _build_fixture(tmp_path)

    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_data_schema.py"), str(data_json)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "OK schema_version=2026-05-07 runs=8 models=" in proc.stdout
    data = json.loads(data_json.read_text(encoding="utf-8"))
    assert len(data["models"]) >= 5
    assert all(run["slug"] == run["name"] for run in data["runs"])
    latest = next(run for run in data["runs"] if run["name"] == "srcnn-v6.1-pico-001")["score_log"][-1]
    assert len(latest["per_frame"]["psnr"]) == 8
    assert 0.0 <= latest["stats"]["beats_bicubic_wilson95_lo"] <= 1.0


def test_run_storage_slug_rejects_path_significant_names() -> None:
    with pytest.raises(ValueError):
        dashboard.run_storage_slug("future/run?bad#name%")


def test_gpu_mem_log_rolls_30_min_window() -> None:
    previous = {
        "gpu_mem_log": [[1020 + i * 60, 7000 + i] for i in range(32)],
    }
    gpu_status = {
        "captured_at": "1970-01-01T00:48:00Z",
        "memory_used_mib": 7561,
    }

    log = dashboard.build_gpu_mem_log(previous, gpu_status)

    assert len(log) == 31
    assert log[0][0] == 1080
    assert log[-1] == [2880, 7561]


def test_public_dashboard_schema_rejects_missing_models(tmp_path: Path) -> None:
    data_json = _build_fixture(tmp_path)
    data = json.loads(data_json.read_text(encoding="utf-8"))
    data.pop("models")
    broken = tmp_path / "missing-models.json"
    broken.write_text(json.dumps(data), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_data_schema.py"), str(broken)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 1
    assert "models: required field missing" in proc.stderr
