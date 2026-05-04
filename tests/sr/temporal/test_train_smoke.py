"""Smoke-test the training entry runs end-to-end on CPU."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from oss.sr.temporal import TemporalSRModel
from scripts.sr_train_temporal import (
    apply_phase,
    lr_multiplier_for_phase,
    phase_for_step,
)


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
    with (out / "score_log.json").open() as f:
        assert json.load(f) == []
    assert any(out.glob("step-*.pt"))


def test_phase_schedule_and_lr_scaling() -> None:
    assert phase_for_step(0, warmup_steps=10, joint_end=20) == 1
    assert phase_for_step(10, warmup_steps=10, joint_end=20) == 1
    assert phase_for_step(11, warmup_steps=10, joint_end=20) == 2
    assert phase_for_step(20, warmup_steps=10, joint_end=20) == 2
    assert phase_for_step(21, warmup_steps=10, joint_end=20) == 3
    assert lr_multiplier_for_phase(1) == 1.0
    assert lr_multiplier_for_phase(2) == 0.1
    assert lr_multiplier_for_phase(3) == 0.01

    model = TemporalSRModel(in_channels=12, scale=2, tier="standard")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    apply_phase(model, optim, base_lr=1e-4, prev_phase=-1, cur_phase=1)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(pg["lr"] == 1e-4 for pg in optim.param_groups)

    apply_phase(model, optim, base_lr=1e-4, prev_phase=1, cur_phase=2)
    assert all(p.requires_grad for p in model.backbone.parameters())
    assert all(pg["lr"] == pytest.approx(1e-5) for pg in optim.param_groups)

    apply_phase(model, optim, base_lr=1e-4, prev_phase=2, cur_phase=3)
    assert all(p.requires_grad for p in model.backbone.parameters())
    assert all(pg["lr"] == pytest.approx(1e-6) for pg in optim.param_groups)


def test_smoke_auto_resume(tmp_path: Path) -> None:
    out = tmp_path / "resume"
    first = subprocess.run(
        [sys.executable, "scripts/sr_train_temporal.py",
         "--smoke", "--device", "cpu", "--max-steps", "1",
         "--output-dir", str(out)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    assert (out / "step-00000001.pt").exists()

    second = subprocess.run(
        [sys.executable, "scripts/sr_train_temporal.py",
         "--smoke", "--device", "cpu", "--max-steps", "2",
         "--output-dir", str(out)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0, second.stderr
    assert "final_step=2" in second.stdout
    assert (out / "step-00000002.pt").exists()
