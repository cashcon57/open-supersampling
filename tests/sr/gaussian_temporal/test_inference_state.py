"""Stateful inference for v5-gaussian-temporal (Task 10).

Mirrors ``tests/sr/temporal/test_inference_state.py`` but the carried state
is a ``GaussianField`` rather than a ``prev_hr`` tensor.
"""
from __future__ import annotations

from pathlib import Path

import torch

from oss.sr.gaussian_temporal import GaussianTemporalSRModel
from oss.sr.inference import GaussianTemporalSRInferenceEngine


def _save_gaussian_temporal_ckpt(tmp_path: Path) -> Path:
    model = GaussianTemporalSRModel(in_channels=12, scale=2, max_count=2048)
    ckpt = tmp_path / "gaussian_temporal.pt"
    torch.save(
        {
            "gaussian_temporal_model": model.state_dict(),
            "args": {"in_channels": 12, "scale": 2, "max_count": 2048},
        },
        ckpt,
    )
    return ckpt


def test_first_call_uses_no_prev_field(tmp_path: Path) -> None:
    ckpt = _save_gaussian_temporal_ckpt(tmp_path)
    eng = GaussianTemporalSRInferenceEngine.from_checkpoint(ckpt, device="cpu", fp16=False)
    # Pre-call: state is empty.
    assert eng._prev_field is None
    lr = torch.rand(1, 12, 32, 32)
    motion = torch.zeros(1, 2, 32, 32)
    out = eng(lr_inputs=lr, motion_lr=motion)
    assert out.shape == (1, 3, 64, 64)
    # Post-call: state stored.
    assert eng._prev_field is not None


def test_reset_clears_state(tmp_path: Path) -> None:
    ckpt = _save_gaussian_temporal_ckpt(tmp_path)
    eng = GaussianTemporalSRInferenceEngine.from_checkpoint(ckpt, device="cpu", fp16=False)
    lr = torch.rand(1, 12, 32, 32)
    motion = torch.zeros(1, 2, 32, 32)
    eng(lr_inputs=lr, motion_lr=motion)
    assert eng._prev_field is not None
    eng.reset()
    assert eng._prev_field is None


def test_scene_cut_auto_reset(tmp_path: Path) -> None:
    ckpt = _save_gaussian_temporal_ckpt(tmp_path)
    eng = GaussianTemporalSRInferenceEngine.from_checkpoint(
        ckpt, device="cpu", fp16=False, scene_cut_motion_threshold=4.0,
    )
    lr = torch.rand(1, 12, 32, 32)
    eng(lr_inputs=lr, motion_lr=torch.zeros(1, 2, 32, 32))
    # Big motion → scene cut. The engine should flag and reset state internally.
    big_motion = torch.full((1, 2, 32, 32), 10.0)
    eng(lr_inputs=lr, motion_lr=big_motion)
    assert eng.last_call_was_scene_cut is True
