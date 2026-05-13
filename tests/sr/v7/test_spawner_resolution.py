"""Multi-resolution tests for the v7 BackboneSpawner.

Production deployment scenarios:
  240p LR -> 720p HR  (320x240 -> 1280x720, scale 3)
  360p LR -> 1080p HR (640x360 -> 1920x1080, scale 3)
  540p LR -> 1080p HR (960x540 -> 1920x1080, scale 2)
  720p LR -> 1440p HR (1280x720 -> 2560x1440, scale 2)
  1080p LR -> 4K HR   (1920x1080 -> 3840x2160, scale 2)

The spawner used to require H_HR and W_HR divisible by tile_size, which
errored for almost every realistic deployment shape (1080 / 16 = 67.5).
The current implementation reflect-pads to the next tile boundary so any
HR works.
"""
from __future__ import annotations

import math

import pytest
import torch

from oss.sr.v7.model import V7Config, V7Model


def _tiny_cfg(canvas_capacity: int, tile_size: int = 16, k_per_tile: int = 2) -> V7Config:
    return V7Config(
        in_channels=9, scale=2, feat_dim=8, latent_rank=4,
        canvas_capacity=canvas_capacity, backbone_kind="placeholder",
        backbone_blocks=1, enable_spawner=True,
        spawner_k_per_tile=k_per_tile, spawner_tile_size=tile_size,
    )


def test_spawner_handles_HR_not_divisible_by_tile_size():
    """1080p HR (1080x1920) at tile_size=16 needs 67.5 vertical tiles.
    Pre-fix: ValueError. Post-fix: reflect-pad to 1088 and spawn 68x120
    tiles' worth of Gaussians."""
    cfg = _tiny_cfg(canvas_capacity=32768, tile_size=16, k_per_tile=1)
    cfg.scale = 2
    model = V7Model(cfg).train(False)
    model.allocate_canvas("cpu")
    lr_in = torch.randn((1, 9, 540, 960))   # HR will be 1080x1920
    with torch.no_grad():
        _ = model(lr_in, t_query=0.0, spawn_at_t=0.0)
    # Expected: ceil(1080/16) * ceil(1920/16) * k_per_tile = 68 * 120 * 1 = 8160
    assert model.canvas.count == 68 * 120 * 1


def test_spawner_pad_only_when_needed():
    """When HR IS exactly divisible by tile_size, no padding happens and
    the spawn count is the un-padded tile product."""
    cfg = _tiny_cfg(canvas_capacity=4096, tile_size=16, k_per_tile=1)
    cfg.scale = 2
    model = V7Model(cfg).train(False)
    model.allocate_canvas("cpu")
    # HR 640x480 (divisible by 16 in both dims)
    lr_in = torch.randn((1, 9, 240, 320))
    with torch.no_grad():
        _ = model(lr_in, t_query=0.0, spawn_at_t=0.0)
    assert model.canvas.count == (480 // 16) * (640 // 16) * 1


def test_spawner_padding_does_not_NaN_outputs():
    """Reflect-pad at the spawner input must not produce NaN positions /
    covariances / features / opacities for the tiles in the padded
    region."""
    cfg = _tiny_cfg(canvas_capacity=32768, tile_size=16, k_per_tile=2)
    cfg.scale = 2
    model = V7Model(cfg).train(False)
    model.allocate_canvas("cpu")
    lr_in = torch.randn((1, 9, 270, 480))   # HR 540x960 — both NON divisible
    with torch.no_grad():
        _ = model(lr_in, t_query=0.0, spawn_at_t=0.0)
    pos, cov, feat, op = model.canvas.active_view()
    assert torch.isfinite(pos).all()
    assert torch.isfinite(cov).all()
    assert torch.isfinite(feat).all()
    assert torch.isfinite(op).all()


@pytest.mark.parametrize("h_lr,w_lr,scale,cap", [
    # (h_lr, w_lr, scale, capacity_required_for_2_spawn_cycle_with_headroom)
    # Capacities chosen so spawn1 + spawn2 fits with at least 1.2x headroom
    # (the trainer wants room for parent-child materializations on top of
    # the base 2-spawn cycle).
    (240, 320, 3, 16384),    # 240p -> 720p  (spawn1=5400, total=10800)
    (360, 640, 3, 65536),    # 360p -> 1080p (spawn1=16320, total=32640, padded)
    (540, 960, 2, 65536),    # 540p -> 1080p (spawn1=16320, total=32640)
    (720, 1280, 2, 131072),  # 720p -> 1440p (spawn1=28800, total=57600)
])
def test_two_spawn_cycle_fits_capacity_at_deployment_resolutions(h_lr, w_lr, scale, cap):
    """Trainer does spawn-at-t=0 then spawn-at-t=2 per sample; capacity
    must hold both. Exercised here for the realistic deployment-HR cases
    so the rigid capacity won't crash the inference path."""
    cfg = V7Config(
        in_channels=9, scale=scale, feat_dim=8, latent_rank=4,
        canvas_capacity=cap, backbone_kind="placeholder",
        backbone_blocks=1, enable_spawner=True,
        spawner_k_per_tile=2, spawner_tile_size=16,
    )
    model = V7Model(cfg).train(False)
    model.allocate_canvas("cpu")
    lr_in = torch.randn((1, 9, h_lr, w_lr))
    with torch.no_grad():
        _ = model(lr_in, t_query=0.0, spawn_at_t=0.0)
        spawn1 = model.canvas.count
        _ = model(lr_in, t_query=2.0, spawn_at_t=2.0)
        spawn2 = model.canvas.count
    assert spawn1 > 0
    assert spawn2 == 2 * spawn1, (
        f"2-spawn cycle should double active count; got {spawn1} -> {spawn2}"
    )
    assert spawn2 < cap, (
        f"Capacity {cap} should leave headroom for a 2-spawn cycle "
        f"({spawn2} actives); pick a larger --canvas-capacity for this HR."
    )


def test_default_v7_config_fits_tartanair_native_HR_two_spawn_cycle():
    """The whole point of the new defaults: V7Config() out of the box
    must handle TartanAir's 480x640 HR for both spawns. Pre-fix this
    was an overflow."""
    cfg = V7Config()  # all defaults
    # Make it tiny enough to run on CPU in unit tests
    cfg.backbone_kind = "placeholder"
    cfg.backbone_blocks = 1
    cfg.feat_dim = 8
    cfg.latent_rank = 4
    model = V7Model(cfg).train(False)
    model.allocate_canvas("cpu")
    lr_in = torch.randn((1, 9, 240, 320))   # HR 480x640, scale=2
    with torch.no_grad():
        _ = model(lr_in, t_query=0.0, spawn_at_t=0.0)
        _ = model(lr_in, t_query=2.0, spawn_at_t=2.0)
    # Defaults: tile=16, k=2 -> 30x40 = 1200 tiles -> 2400 per spawn -> 4800 total
    assert model.canvas.count == 4800
    assert model.canvas.count < cfg.canvas_capacity, (
        "Default canvas_capacity must hold 2-spawn cycle at TartanAir HR"
    )
