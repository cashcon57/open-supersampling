from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
    assert "OK schema_version=2026-05-07 runs=6 models=" in proc.stdout
    data = json.loads(data_json.read_text(encoding="utf-8"))
    assert len(data["models"]) >= 5
    latest = next(run for run in data["runs"] if run["name"] == "srcnn-v6.1-pico-001")["score_log"][-1]
    assert len(latest["per_frame"]["psnr"]) == 8
    assert 0.0 <= latest["stats"]["beats_bicubic_wilson95_lo"] <= 1.0


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
