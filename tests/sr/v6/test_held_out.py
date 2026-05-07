from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _write_tartanair_fixture(root: Path, *, n_pairs: int = 4) -> Path:
    from torchvision.io import write_png

    traj = root / "oldtown" / "Easy" / "P000"
    img_dir = traj / "image_left"
    depth_dir = traj / "depth_left"
    flow_dir = traj / "flow"
    img_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    flow_dir.mkdir(parents=True)

    h, w = 32, 32
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    for idx in range(n_pairs + 1):
        rgb = torch.stack(
            [
                (xx + idx * 7).remainder(256),
                (yy * 3 + idx * 5).remainder(256),
                ((xx + yy) * 2 + idx * 11).remainder(256),
            ],
            dim=0,
        ).to(torch.uint8)
        write_png(rgb, str(img_dir / f"{idx:06d}_left.png"))

        depth = np.linspace(0.1, 2.0 + 0.1 * idx, h * w, dtype=np.float32).reshape(h, w)
        np.save(depth_dir / f"{idx:06d}_left_depth.npy", depth)

        flow = np.zeros((h, w, 2), dtype=np.float32)
        flow[..., 0] = 0.25
        flow[..., 1] = -0.125
        np.save(flow_dir / f"{idx:06d}_{idx + 1:06d}_flow.npy", flow)

    return traj


def _write_manifest(path: Path, traj: Path, *, n_pairs: int = 4) -> None:
    manifest = {
        "manifest_version": 1,
        "dataset_kind": "tartanair",
        "n_pairs": n_pairs,
        "seed": 0,
        "lr_scale": 2.0,
        "lr_synth_args": {
            "enable_jitter": True,
            "enable_taa_blur": True,
            "enable_jpeg": False,
            "jpeg_quality": 85,
            "blur_sigma": 0.5,
        },
        "pairs": [
            {
                "trajectory": str(traj),
                "idx_t": idx,
                "idx_t_plus_1": idx + 1,
            }
            for idx in range(n_pairs)
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_v6_ckpt(path: Path) -> None:
    from oss.sr.v6.model import V6Config, V6Model

    cfg = V6Config(
        in_channels=9,
        scale=2,
        backbone="hat-tiny",
        canvas_capacity=64,
        token_dim=32,
        cross_attention_heads=4,
        window_size=16,
        color_activation="sdr",
    )
    model = V6Model(cfg)
    torch.save(
        {
            "kind": "v6",
            "step": 100,
            "args": {"backbone": "hat-tiny", "in_channels": 9, "scale": 2},
            "v6_config": cfg.__dict__,
            "generator": model.state_dict(),
        },
        path,
    )


def test_v6_held_out_writes_dashboard_score_log(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.sr_v6_held_out as held_out

    data_root = tmp_path / "tartanair"
    traj = _write_tartanair_fixture(data_root, n_pairs=4)
    manifest = tmp_path / "v5_held_out_manifest.json"
    _write_manifest(manifest, traj, n_pairs=4)

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    ckpt = output_dir / "step-00000100.pt"
    _write_v6_ckpt(ckpt)

    monkeypatch.setenv("OSS_V6_HELD_OUT_LPIPS_FALLBACK", "1")
    rc = held_out.main(
        [
            "--output-dir",
            str(output_dir),
            "--ckpt",
            str(ckpt),
            "--manifest",
            str(manifest),
            "--tartanair-root",
            str(data_root),
            "--device",
            "cpu",
            "--batch-size",
            "2",
        ]
    )
    assert rc == 0

    rows = json.loads((output_dir / "score_log.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    row = rows[0]
    for key in (
        "step",
        "model_psnr_mean",
        "model_lpips_mean",
        "model_ssim_mean",
        "bicubic_psnr_mean",
        "bicubic_lpips_mean",
        "bicubic_ssim_mean",
        "model_beats_bicubic_count",
        "model_beats_bicubic_lpips_count",
    ):
        assert key in row
    assert row["step"] == 100
    assert len(row["model_psnr_per_sample"]) == 4
    assert torch.isfinite(torch.tensor(row["model_psnr_mean"]))
    assert torch.isfinite(torch.tensor(row["model_lpips_mean"]))
