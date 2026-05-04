"""Stateful inference for v5-pixel-temporal."""
from __future__ import annotations

from pathlib import Path

import torch

from oss.sr import build_sr_model
from oss.sr.inference import TemporalSRInferenceEngine
from oss.sr.temporal import TemporalSRModel


def _save_temporal_ckpt(tmp_path: Path) -> Path:
    backbone = build_sr_model(model_kind="simple", tier="standard", in_channels=12, scale=2)
    model = TemporalSRModel(in_channels=12, scale=2, tier="standard")
    model.backbone.load_state_dict(backbone.state_dict())
    ckpt = tmp_path / "temporal.pt"
    torch.save(
        {
            "temporal_model": model.state_dict(),
            "args": {"tier": "standard", "sr_backbone": "simple", "in_channels": 12, "scale": 2},
        },
        ckpt,
    )
    return ckpt


def test_first_call_uses_bilinear_init(tmp_path: Path) -> None:
    ckpt = _save_temporal_ckpt(tmp_path)
    eng = TemporalSRInferenceEngine.from_checkpoint(ckpt, device="cpu", fp16=False)
    lr = torch.rand(1, 12, 8, 8)
    depth_hr = torch.rand(1, 1, 16, 16)
    motion = torch.zeros(1, 2, 8, 8)
    out = eng(lr_inputs=lr, depth_hr_curr=depth_hr, motion_lr=motion)
    assert out.shape == (1, 3, 16, 16)
    assert eng._prev_hr is not None  # state stored


def test_reset_clears_state(tmp_path: Path) -> None:
    ckpt = _save_temporal_ckpt(tmp_path)
    eng = TemporalSRInferenceEngine.from_checkpoint(ckpt, device="cpu", fp16=False)
    lr = torch.rand(1, 12, 8, 8)
    depth_hr = torch.rand(1, 1, 16, 16)
    motion = torch.zeros(1, 2, 8, 8)
    eng(lr_inputs=lr, depth_hr_curr=depth_hr, motion_lr=motion)
    eng.reset()
    assert eng._prev_hr is None
    assert eng._prev_depth_hr is None


def test_scene_cut_auto_reset(tmp_path: Path) -> None:
    ckpt = _save_temporal_ckpt(tmp_path)
    eng = TemporalSRInferenceEngine.from_checkpoint(
        ckpt, device="cpu", fp16=False, scene_cut_motion_threshold=4.0,
    )
    lr = torch.rand(1, 12, 8, 8)
    depth_hr = torch.rand(1, 1, 16, 16)
    eng(lr_inputs=lr, depth_hr_curr=depth_hr, motion_lr=torch.zeros(1, 2, 8, 8))
    # Big motion → scene cut. The engine should flag and reset state internally.
    big_motion = torch.full((1, 2, 8, 8), 10.0)
    eng(lr_inputs=lr, depth_hr_curr=depth_hr, motion_lr=big_motion)
    assert eng.last_call_was_scene_cut is True
