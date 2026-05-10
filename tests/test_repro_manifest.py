from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_public_dashboard as dashboard
from scripts.emit_repro_manifest import emit_manifest
from tools.check_data_schema import validate


def _torch():
    return pytest.importorskip("torch")


def _write_ckpt(tmp_path: Path, *, step: int = 1) -> tuple[Path, Path]:
    torch = _torch()
    run_dir = tmp_path / "runs" / "srcnn-v6.1-pico-001"
    ckpt_dir = run_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manifest = ckpt_dir / "train_manifest.json"
    manifest.write_text(json.dumps({"pairs": [1, 2, 3]}), encoding="utf-8")
    ckpt = ckpt_dir / f"step-{step:08d}.pt"
    torch.save(
        {
            "kind": "v6",
            "args": {
                "backbone": "hat-tiny",
                "manifest": str(manifest),
                "spawn_offset_random": True,
                "rasterizer_overlap": 8,
            },
            "generator": {
                "conv.weight": torch.zeros(2, 3, 3, 3),
                "conv.bias": torch.zeros(2),
            },
            "discriminator": {"weight": torch.zeros(999)},
            "rng": {"torch": torch.tensor([1, 2, 3], dtype=torch.uint8)},
        },
        ckpt,
    )
    return ckpt, manifest


def test_emit_repro_manifest_reads_cpu_checkpoint_and_hashes_manifest(tmp_path: Path) -> None:
    ckpt, data_manifest = _write_ckpt(tmp_path)

    manifest = emit_manifest(ckpt)

    assert manifest["dataset_sha"] == hashlib.sha256(data_manifest.read_bytes()).hexdigest()
    assert "--backbone hat-tiny" in manifest["cli_invocation"]
    assert manifest["model_arch"] == "v6:hat-tiny"
    assert manifest["param_count"] == 56
    assert manifest["rng_state"]["torch"]["values"] == [1, 2, 3]
    assert manifest["cuda_version"] is None or isinstance(manifest["cuda_version"], str)


def test_public_dashboard_includes_repro_manifest_from_latest_ckpt(tmp_path: Path) -> None:
    _write_ckpt(tmp_path, step=1)
    latest_ckpt, _data_manifest = _write_ckpt(tmp_path, step=2)
    run_dir = latest_ckpt.parents[1]
    (run_dir / "metrics.json").write_text('{"step": 2, "loss": 0.1}', encoding="utf-8")
    out_dir = tmp_path / "public"
    dashboard.configure_paths(tmp_path / "runs", out_dir)

    run = dashboard.build_run("srcnn-v6.1-pico-001", previous=None)

    assert run is not None
    assert run["repro_manifest"]["1"]["param_count"] == 56
    assert "--rasterizer-overlap 8" in run["repro_manifest"]["2"]["cli_invocation"]


def test_schema_accepts_repro_manifest_field() -> None:
    data = {
        "schema_version": "2026-05-07",
        "generated_at": "2026-05-07T00:00:00Z",
        "runs": [
            {
                "name": "run",
                "slug": "run",
                "label": "Run",
                "active": False,
                "latest_step": 0,
                "max_target_steps": None,
                "latest_metrics": {},
                "history": {},
                "loss_curve": [],
                "score_log": [],
                "viz_pngs": [],
                "viz_columns": [],
                "events": [],
                "cross_version_points": [],
                "gpu_status": None,
                "gpu_mem_log": [],
                "repro_manifest": {},
                "cost": {
                    "kwh": 0.0,
                    "usd": 0.0,
                    "gpu_hours": 0.0,
                    "projections": {
                        "B200": {"gpu_hours_to_dlss4_quality": 0.0, "usd_at_runpod_rate": 0.0},
                        "H100": {"gpu_hours_to_dlss4_quality": 0.0, "usd_at_runpod_rate": 0.0},
                        "A100": {"gpu_hours_to_dlss4_quality": 0.0, "usd_at_runpod_rate": 0.0},
                        "4090": {"gpu_hours_to_dlss4_quality": 0.0, "usd_at_runpod_rate": 0.0},
                    },
                },
            }
        ],
        "models": [],
    }

    code, errors, _warnings = validate(data)

    assert code == 0
    assert errors == []
