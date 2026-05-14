"""Tests for the v7 lane in scripts/build_public_dashboard.py.

Covers:
- v7 history.jsonl parsing (new schema: sr_charbonnier, canvas_count, etc.)
- v7 score_log_v7.json parsing (alpha_1_sr / alpha_0_5_oss_fx / bicubic baseline
  + delta_oss_fx_over_bicubic_psnr_db headline)
- --print-active-run --version v7 resolves the configured v7 run name.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import build_public_dashboard as dashboard


ROOT = Path(__file__).resolve().parents[2]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v7_history_jsonl_parsing(tmp_path: Path) -> None:
    """Synthesize a v7 history.jsonl with the new schema and verify the
    canonical keys round-trip through read_v7_history + parse_v7_history_row.
    """
    run_dir = tmp_path / "srcnn-v7.0-pico-005"
    rows = [
        {
            "step": 100,
            "total": 0.123,
            "sr_charbonnier": 0.045,
            "sr_lpips": 0.011,
            "sr_sobel": 0.002,
            "fg_charbonnier": 0.0,
            "lambda_fg": 0.0,
            "lambda_fg_lpips": 0.0,
            "lambda_temp": 0.0,
            "canvas_count": 4800,
            "canvas_mean_opacity": 0.05,
            "canvas_mean_L_diag": 1.4,
            "materialized": 12,
            "elapsed_s": 510,
        },
        {
            "step": 200,
            "total": 0.108,
            "sr_charbonnier": 0.041,
            "sr_lpips": 0.010,
            "sr_sobel": 0.0018,
            "canvas_count": 5100,
            "canvas_mean_opacity": 0.06,
            "canvas_mean_L_diag": 1.5,
            "materialized": 18,
            "elapsed_s": 1020,
        },
    ]
    _write_jsonl(run_dir / "history.jsonl", rows)

    parsed = dashboard.read_v7_history(run_dir)

    assert len(parsed) == 2
    first = parsed[0]
    # All v7-specific fields land
    for key in (
        "step",
        "total",
        "sr_charbonnier",
        "sr_lpips",
        "sr_sobel",
        "canvas_count",
        "canvas_mean_opacity",
        "canvas_mean_L_diag",
        "materialized",
        "elapsed_s",
        "lambda_fg",
        "lambda_fg_lpips",
        "lambda_temp",
    ):
        assert key in first, f"v7 history row missing {key!r}"
    assert first["sr_charbonnier"] == 0.045
    assert first["canvas_count"] == 4800
    assert first["materialized"] == 12
    # Second row tolerates absent lambda_* fields (early-row schema variance).
    assert parsed[1]["materialized"] == 18


def test_v7_score_log_parsing(tmp_path: Path) -> None:
    """Synthesize a score_log_v7.json with one eval row and verify the
    alpha_1_sr / alpha_0_5_oss_fx / bicubic baseline triples round-trip
    flattened, plus the delta_oss_fx_over_bicubic_psnr_db headline survives.
    """
    run_dir = tmp_path / "srcnn-v7.0-pico-005"
    payload = [
        {
            "step": 12345,
            "n_triplets": 64,
            "alpha_1_sr": {"psnr": 28.7, "ssim": 0.812, "lpips": 0.198},
            "alpha_0_5_oss_fx": {"psnr": 27.2, "ssim": 0.795, "lpips": 0.215},
            "alpha_0_5_bicubic_baseline": {"psnr": 25.8, "ssim": 0.770, "lpips": 0.260},
            "delta_oss_fx_over_bicubic_psnr_db": 1.4,
            "canvas_health_final": {"canvas_count": 5200},
        }
    ]
    _write_json(run_dir / "score_log_v7.json", payload)

    parsed = dashboard.read_v7_score_log(run_dir)

    assert len(parsed) == 1
    row = parsed[0]
    assert row["step"] == 12345
    assert row["n_triplets"] == 64
    # Flattened alpha=1 SR triple
    assert row["alpha_1_sr_psnr"] == 28.7
    assert row["alpha_1_sr_ssim"] == 0.812
    assert row["alpha_1_sr_lpips"] == 0.198
    # Flattened alpha=0.5 OSS-FX + bicubic baseline triples
    assert row["alpha_0_5_oss_fx_psnr"] == 27.2
    assert row["alpha_0_5_bicubic_psnr"] == 25.8
    assert row["alpha_0_5_bicubic_lpips"] == 0.260
    # Headline OSS-FX delta is preserved verbatim under its canonical key.
    assert row["delta_oss_fx_over_bicubic_psnr_db"] == 1.4


def test_print_active_run_version_v7_returns_configured_run() -> None:
    """--print-active-run --version v7 must resolve to the v7 entry in
    RUN_CONFIG (the v7 eval supervisor depends on this contract).
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_public_dashboard.py"),
            "--print-active-run",
            "--version",
            "v7",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert proc.stdout.strip() == "srcnn-v7.0-pico-005"

    # Default --version v6 keeps legacy behavior: returns whichever active
    # run wins by RUN_CONFIG insertion order. Must NOT be the v7 run.
    proc_v6 = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_public_dashboard.py"),
            "--print-active-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    stdout = proc_v6.stdout.strip()
    assert stdout, "default --print-active-run must still print an active run"
    # The legacy active run is a v6.x entry, not the v7 one.
    assert "v7" not in stdout


def test_v7_build_run_surfaces_score_log_and_arch_version(tmp_path: Path) -> None:
    """End-to-end: build_run() reads v7 history + score_log_v7.json and
    tags the run with arch_version='v7'.
    """
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "srcnn-v7.0-pico-005"
    _write_jsonl(
        run_dir / "history.jsonl",
        [
            {"step": 100, "total": 0.12, "sr_charbonnier": 0.04, "canvas_count": 4800,
             "materialized": 10, "elapsed_s": 500},
        ],
    )
    _write_json(
        run_dir / "score_log_v7.json",
        [
            {
                "step": 100,
                "n_triplets": 32,
                "alpha_1_sr": {"psnr": 28.0, "ssim": 0.80, "lpips": 0.20},
                "alpha_0_5_oss_fx": {"psnr": 27.0, "ssim": 0.78, "lpips": 0.22},
                "alpha_0_5_bicubic_baseline": {"psnr": 25.7, "ssim": 0.76, "lpips": 0.26},
                "delta_oss_fx_over_bicubic_psnr_db": 1.3,
            }
        ],
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    dashboard.configure_paths(runs_dir, out_dir)
    run = dashboard.build_run("srcnn-v7.0-pico-005", previous=None)

    assert run is not None
    assert run["arch_version"] == "v7"
    assert run["latest_step"] == 100
    assert run["score_log_v7"]
    assert run["score_log_v7"][0]["delta_oss_fx_over_bicubic_psnr_db"] == 1.3
    # v7 history fields land in latest_metrics so headlineValue can render them.
    assert run["latest_metrics"].get("sr_charbonnier") == 0.04
    assert run["latest_metrics"].get("canvas_count") == 4800
