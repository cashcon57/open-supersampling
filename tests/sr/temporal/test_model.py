"""Tests for TemporalSRModel."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from oss.sr import build_sr_model
from oss.sr.temporal import TemporalSRModel, make_first_frame_prev_hr


def _make_inputs(batch: int = 1, lr: int = 8, in_ch: int = 12, scale: int = 2):
    h_hr, w_hr = lr * scale, lr * scale
    return {
        "lr_inputs": torch.rand(batch, in_ch, lr, lr),
        "prev_hr": torch.rand(batch, 3, h_hr, w_hr),
        "depth_hr_curr": torch.rand(batch, 1, h_hr, w_hr),
        "depth_hr_prev": torch.rand(batch, 1, h_hr, w_hr),
        "motion_lr": torch.randn(batch, 2, lr, lr) * 0.1,
    }


def test_forward_shape() -> None:
    model = TemporalSRModel(in_channels=12, scale=2, tier="standard")
    out = model(**_make_inputs(batch=2, lr=8))
    assert out.shape == (2, 3, 16, 16)


def test_make_first_frame_prev_hr() -> None:
    lr_rgb = torch.rand(2, 3, 8, 8)
    prev_hr = make_first_frame_prev_hr(lr_rgb, scale=2)
    assert prev_hr.shape == (2, 3, 16, 16)


def test_freeze_backbone_toggle() -> None:
    model = TemporalSRModel(in_channels=12, scale=2, tier="standard")
    model.freeze_backbone(True)
    backbone_params = list(model.backbone.parameters())
    assert all(not p.requires_grad for p in backbone_params)
    head_params = list(model.head.parameters())
    assert all(p.requires_grad for p in head_params)
    model.freeze_backbone(False)
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_load_v4_warm_start(tmp_path: Path) -> None:
    # Build a v4-style checkpoint and round-trip it.
    src = build_sr_model(model_kind="simple", tier="standard", in_channels=12, scale=2)
    ckpt = tmp_path / "v4_synth.pt"
    torch.save(
        {"sr_model": src.state_dict(), "args": {"tier": "standard", "sr_backbone": "simple"}},
        ckpt,
    )
    model = TemporalSRModel.load_v4_warm_start(ckpt, in_channels=12, scale=2)
    for k, v in src.state_dict().items():
        assert torch.equal(model.backbone.state_dict()[k], v)


def test_grad_flow_with_frozen_backbone() -> None:
    model = TemporalSRModel(in_channels=12, scale=2, tier="standard")
    model.freeze_backbone(True)
    inputs = _make_inputs(batch=1, lr=8)
    out = model(**inputs)
    out.mean().backward()
    # Head + gate get grads; backbone does not.
    for p in model.head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    for p in model.gate.parameters():
        assert p.grad is not None
    for p in model.backbone.parameters():
        assert p.grad is None or torch.equal(p.grad, torch.zeros_like(p))
