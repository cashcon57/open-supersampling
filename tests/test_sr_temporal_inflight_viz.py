from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.sr_temporal_inflight_viz import _comparison_panels, main


def test_comparison_panels_adds_v6_gaussian_and_two_error_maps() -> None:
    h, w = 3, 5
    panel = torch.zeros(3, h, w)

    panels, labels = _comparison_panels(
        lr_up=panel,
        bicubic=panel,
        pixel=panel,
        gaussian=panel,
        v6=panel,
        gt=panel,
        err_rgb=panel,
        err_rgb_v6=panel,
    )
    strip = torch.cat(panels, dim=-1)

    assert strip.shape == (3, h, w * 8)
    assert labels == [
        "LR-bilinear",
        "bicubic",
        "v5-pixel-temporal",
        "v5-Gaussian",
        "v6",
        "GT",
        "|err v5|",
        "|err v6|",
    ]


def test_comparison_panels_keeps_legacy_v5_only_strip() -> None:
    h, w = 3, 5
    panel = torch.zeros(3, h, w)

    panels, labels = _comparison_panels(
        lr_up=panel,
        bicubic=panel,
        baseline=panel,
        pixel=panel,
        gt=panel,
        err_rgb=panel,
    )
    strip = torch.cat(panels, dim=-1)

    assert strip.shape == (3, h, w * 6)
    assert labels == [
        "LR-bilinear",
        "bicubic",
        "v4-baseline",
        "v5-temporal",
        "GT",
        "|err| heatmap",
    ]


def _write_tartanair_fixture(root: Path) -> Path:
    traj = root / "oldtown" / "Easy" / "P000"
    image_dir = traj / "image_left"
    depth_dir = traj / "depth_left"
    flow_dir = traj / "flow"
    image_dir.mkdir(parents=True)
    depth_dir.mkdir()
    flow_dir.mkdir()

    h = w = 16
    yy, xx = np.mgrid[0:h, 0:w]
    for idx in range(2):
        rgb = np.stack([
            (xx + idx * 8) % 256,
            (yy * 2 + idx * 4) % 256,
            np.full((h, w), 64 + idx * 32),
        ], axis=-1).astype(np.uint8)
        Image.fromarray(rgb).save(image_dir / f"{idx:06d}_left.png")
        depth = np.linspace(0.1, 1.0, h * w, dtype=np.float32).reshape(h, w)
        np.save(depth_dir / f"{idx:06d}_left_depth.npy", depth)
        flow = np.zeros((h, w, 2), dtype=np.float32)
        flow[..., 0] = 0.25
        np.save(flow_dir / f"{idx:06d}_{idx + 1:06d}_flow.npy", flow)
    return traj


def test_v6_primary_renders_png_with_required_columns(tmp_path: Path) -> None:
    from oss.sr.temporal import TemporalSRModel
    from oss.sr.v6.model import V6Config, V6Model

    root = tmp_path / "tartanair"
    traj = _write_tartanair_fixture(root)
    manifest = tmp_path / "held_out_manifest.json"
    manifest.write_text(json.dumps({
        "manifest_version": 1,
        "dataset_kind": "tartanair",
        "n_pairs": 1,
        "seed": 0,
        "lr_scale": 2.0,
        "lr_synth_args": {
            "enable_jitter": False,
            "enable_taa_blur": False,
            "enable_jpeg": False,
            "jpeg_quality": 85,
            "blur_sigma": 0.5,
        },
        "pairs": [
            {"trajectory": str(traj), "idx_t": 0, "idx_t_plus_1": 1},
        ],
    }))

    v5 = TemporalSRModel(in_channels=12, scale=2, tier="pico", backbone_kind="simple")
    v5_ckpt = tmp_path / "v5.pt"
    torch.save({
        "args": {
            "in_channels": 12,
            "scale": 2,
            "tier": "pico",
            "backbone_kind": "simple",
            "zero_gbuffer_into_backbone": False,
        },
        "temporal_model": v5.state_dict(),
    }, v5_ckpt)

    v6_cfg = V6Config(
        backbone="hat-tiny",
        in_channels=9,
        scale=2,
        canvas_capacity=64,
        tile_size_lr=8,
        tile_size_hr=16,
        keyframe_interval=2,
        prune_every=10,
    )
    v6 = V6Model(v6_cfg)
    output_dir = tmp_path / "srcnn-v6-heavy-001"
    output_dir.mkdir()
    v6_ckpt = output_dir / "step-00000100.pt"
    torch.save({
        "step": 100,
        "kind": "v6",
        "args": {
            "backbone": "hat-tiny",
            "trajectory_length": 2,
            "held_out_envs": ["oldtown"],
        },
        "v6_config": v6_cfg.__dict__,
        "model_state_dict": v6.state_dict(),
    }, v6_ckpt)

    rc = main([
        "--output-dir", str(output_dir),
        "--primary-version", "v6",
        "--manifest", str(manifest),
        "--tartanair-root", str(root),
        "--ckpt-v5", str(v5_ckpt),
        "--n-pairs", "1",
        "--traj-length", "2",
        "--once",
    ])

    assert rc == 0
    png = output_dir / "viz" / "step-00000100.png"
    assert png.is_file()
    with Image.open(png) as img:
        assert img.size == (16 * 7, 16)
