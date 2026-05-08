# SPDX-License-Identifier: Apache-2.0
"""Integration smoke for the v6.2 architectural mode.

Drives a 3-frame trajectory through V6Model in fusion_mode='concat' +
spawner_mode='disocclusion' + latent_rank=16 to assert the new code
paths run end-to-end on CPU without NaN/Inf and produce non-empty canvas
state.
"""
from __future__ import annotations

import torch

from oss.sr.v6.model import V6Config, V6Model


def _v62_config() -> V6Config:
    return V6Config(
        backbone="hat-tiny",
        fusion_mode="concat",
        spawner_mode="disocclusion",
        latent_rank=16,
    )


def test_v62_construction_uses_new_modules() -> None:
    model = V6Model(_v62_config())
    assert type(model.fusion).__name__ == "ConcatFusion"
    assert model.gaussian_spawner is None
    assert type(model.disocclusion_spawner).__name__ == "DisocclusionSpawner"
    assert model.rasterizer.latent_rank == 16
    assert model.rasterizer.feature_dim == 16


def test_v62_forward_three_frames_finite() -> None:
    torch.manual_seed(0)
    model = V6Model(_v62_config())
    model.train(False)
    B, C, H, W = 1, 9, 16, 16
    lr_in = torch.rand(B, C, H, W) * 0.5
    lr_in[:, 3:4] = torch.linspace(0.2, 0.8, H * W).reshape(1, 1, H, W)
    mv = torch.zeros(B, 2, H, W)

    with torch.no_grad():
        out0 = model(lr_in, motion_lr=None, frame_index=0)
        out1 = model(lr_in, motion_lr=mv, frame_index=1)
        out2 = model(lr_in + 0.05, motion_lr=mv + 0.1, frame_index=2)

    for tag, out in (("frame0", out0), ("frame1", out1), ("frame2", out2)):
        assert out.shape == (B, 3, H * 2, W * 2), f"{tag}: bad shape {out.shape}"
        assert torch.isfinite(out).all().item(), f"{tag}: non-finite output"

    canvas = model._canvas_state
    assert canvas is not None, "canvas should be populated after frame 1"
    assert canvas.count > 0, "disocclusion spawner should have produced births"


def test_v62_validation_rejects_bad_modes() -> None:
    import pytest

    for cfg_kwargs in (
        {"fusion_mode": "bogus"},
        {"spawner_mode": "bogus"},
        {"latent_rank": 0},
        {"latent_rank": 999},
    ):
        with pytest.raises(ValueError):
            V6Model(V6Config(backbone="hat-tiny", **cfg_kwargs))
