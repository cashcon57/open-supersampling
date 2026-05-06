"""Tests for ``oss.sr.v6.dataset`` Hypersim integration + 60/30 mix."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from oss.sr.v6.dataset import (
    HypersimDataset,
    MixedTartanAirHypersimDataset,
    TrajectoryDataset,
    TrajectoryMixedDataset,
    _TartanAirV6Wrapper,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_hypersim_fixture(root: Path, n_scenes: int = 2, n_frames: int = 3) -> Path:
    """Build a Hypersim-like directory tree under ``root``."""
    from torchvision.utils import save_image

    for s in range(n_scenes):
        scene = root / f"ai_001_{s:03d}"
        cam_color = scene / "images" / "scene_cam_00_final_preview"
        cam_geom = scene / "images" / "scene_cam_00_geometry_hdf5"
        cam_color.mkdir(parents=True, exist_ok=True)
        cam_geom.mkdir(parents=True, exist_ok=True)
        for f in range(n_frames):
            fid = f"{f:04d}"
            color = torch.rand(3, 32, 48)
            save_image(color, cam_color / f"frame.{fid}.tonemap.jpg")
            depth = np.random.rand(32, 48).astype("float32") * 5.0
            np.save(cam_geom / f"frame.{fid}.depth_meters.npy", depth)
            normals = np.random.randn(32, 48, 3).astype("float32")
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True).clip(min=1e-6)
            np.save(cam_geom / f"frame.{fid}.normal_cam.npy", normals)
    return root


def _make_tartanair_fixture(root: Path, n_traj: int = 1, n_frames: int = 4) -> Path:
    """Build a TartanAir-like directory tree under ``root``."""
    from torchvision.utils import save_image

    for t in range(n_traj):
        traj = root / "fixtureenv" / "Easy" / f"P{t:03d}"
        img_dir = traj / "image_left"
        depth_dir = traj / "depth_left"
        flow_dir = traj / "flow"
        for d in (img_dir, depth_dir, flow_dir):
            d.mkdir(parents=True, exist_ok=True)
        for f in range(n_frames):
            idx = f"{f:06d}"
            color = torch.rand(3, 32, 48)
            save_image(color, img_dir / f"{idx}_left.png")
            depth = np.random.rand(32, 48).astype("float32")
            np.save(depth_dir / f"{idx}_left_depth.npy", depth)
            if f + 1 < n_frames:
                next_idx = f"{f + 1:06d}"
                flow = np.random.randn(32, 48, 2).astype("float32")
                np.save(flow_dir / f"{idx}_{next_idx}_flow.npy", flow)
    return root


# ---------------------------------------------------------------------------
# HypersimDataset
# ---------------------------------------------------------------------------


def test_hypersim_dataset_construction(tmp_path: Path) -> None:
    root = _make_hypersim_fixture(tmp_path / "hypersim")
    ds = HypersimDataset(root=root, scale=2.0)
    assert len(ds) == 2 * 3  # n_scenes * n_frames


def test_hypersim_dataset_yields_v6_keys(tmp_path: Path) -> None:
    root = _make_hypersim_fixture(tmp_path / "hypersim")
    ds = HypersimDataset(root=root, scale=2.0)
    item = ds[0]
    for key in ("lr_frame", "gt_hr_frame", "depth", "normals", "motion", "canvas_hint"):
        assert key in item, f"missing key {key}"

    # motion and canvas_hint must be all zeros for Hypersim
    assert torch.equal(item["motion"], torch.zeros_like(item["motion"]))
    assert torch.equal(item["canvas_hint"], torch.zeros_like(item["canvas_hint"]))

    # spatial shapes
    assert item["lr_frame"].shape == (3, 16, 24)
    assert item["gt_hr_frame"].shape == (3, 32, 48)
    assert item["depth"].shape == (1, 16, 24)
    assert item["normals"].shape == (3, 16, 24)
    assert item["motion"].shape == (2, 16, 24)
    assert item["canvas_hint"].shape == (3, 16, 24)


def test_hypersim_dataset_held_out(tmp_path: Path) -> None:
    root = _make_hypersim_fixture(tmp_path / "hypersim", n_scenes=4, n_frames=2)
    ds = HypersimDataset(
        root=root, scale=2.0, held_out_scenes=["ai_001_000", "ai_001_001"],
    )
    # 2 of 4 scenes excluded → 2 scenes * 2 frames = 4 items
    assert len(ds) == 4


def test_hypersim_dataset_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        HypersimDataset(root=tmp_path / "does-not-exist", scale=2.0)


# ---------------------------------------------------------------------------
# MixedTartanAirHypersimDataset
# ---------------------------------------------------------------------------


def test_mixed_dataset_interleaves_with_ratio(tmp_path: Path) -> None:
    from oss.gaussian.data.tartanair import TartanAirGaussianDataset

    h_root = _make_hypersim_fixture(tmp_path / "hypersim", n_scenes=2, n_frames=2)
    t_root = _make_tartanair_fixture(tmp_path / "tartanair", n_traj=1, n_frames=4)

    ds_t = _TartanAirV6Wrapper(TartanAirGaussianDataset(root=t_root, scale=2.0))
    ds_h = HypersimDataset(root=h_root, scale=2.0)

    mixed = MixedTartanAirHypersimDataset(
        tartanair=ds_t, hypersim=ds_h,
        tartanair_ratio=0.667, hypersim_ratio=0.333, seed=42,
    )
    assert len(mixed) == len(ds_t) + len(ds_h)

    # Sample many indices and check the empirical ratio is in the right ballpark.
    n_samples = 1000
    counts = {"tartanair": 0, "hypersim": 0}
    for i in range(n_samples):
        src = mixed._pick_source(i)
        counts[src] += 1
    # Allow generous tolerance — 1000 samples around 66.7% target.
    t_frac = counts["tartanair"] / n_samples
    assert 0.60 < t_frac < 0.74, f"expected ~0.667 TartanAir frac, got {t_frac}"


def test_mixed_dataset_yields_dict(tmp_path: Path) -> None:
    from oss.gaussian.data.tartanair import TartanAirGaussianDataset

    h_root = _make_hypersim_fixture(tmp_path / "hypersim", n_scenes=1, n_frames=2)
    t_root = _make_tartanair_fixture(tmp_path / "tartanair", n_traj=1, n_frames=3)

    ds_t = _TartanAirV6Wrapper(TartanAirGaussianDataset(root=t_root, scale=2.0))
    ds_h = HypersimDataset(root=h_root, scale=2.0)

    mixed = MixedTartanAirHypersimDataset(
        tartanair=ds_t, hypersim=ds_h, seed=7,
    )

    for i in range(len(mixed)):
        item = mixed[i]
        # Both sources must produce the v6 dict shape.
        for key in ("lr_frame", "gt_hr_frame", "depth", "normals", "motion", "canvas_hint"):
            assert key in item


def test_mixed_dataset_single_source(tmp_path: Path) -> None:
    h_root = _make_hypersim_fixture(tmp_path / "hypersim", n_scenes=1, n_frames=2)
    ds_h = HypersimDataset(root=h_root, scale=2.0)
    mixed = MixedTartanAirHypersimDataset(tartanair=None, hypersim=ds_h)
    assert len(mixed) == 2
    assert mixed._pick_source(0) == "hypersim"


def test_trajectory_dataset_yields_consecutive_window(tmp_path: Path) -> None:
    h_root = _make_hypersim_fixture(tmp_path / "hypersim", n_scenes=1, n_frames=4)
    ds_h = HypersimDataset(root=h_root, scale=2.0)
    traj = TrajectoryDataset(ds_h, trajectory_length=3)

    item = traj[0]
    assert item["lr_frame"].shape[:2] == (3, 3)
    assert item["gt_hr_frame"].shape[:2] == (3, 3)
    assert item["motion"].shape[:2] == (3, 2)
    assert torch.equal(item["motion"][0], torch.zeros_like(item["motion"][0]))


def test_trajectory_mixed_dataset_indexes_windows(tmp_path: Path) -> None:
    h_root = _make_hypersim_fixture(tmp_path / "hypersim", n_scenes=1, n_frames=4)
    ds_h = HypersimDataset(root=h_root, scale=2.0)
    mixed = TrajectoryMixedDataset(
        tartanair=None,
        hypersim=ds_h,
        trajectory_length=2,
    )

    assert len(mixed) == 3
    item = mixed[0]
    assert item["lr_frame"].shape[0] == 2


# ---------------------------------------------------------------------------
# sr_train_v6.py smoke
# ---------------------------------------------------------------------------


def _read_metrics_json(path: Path) -> list[dict]:
    import json as _json
    return [_json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_sr_train_v6_smoke_runs_training_loop(tmp_path: Path) -> None:
    """`--smoke` must run the v6 training loop and exit zero."""
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "sr_train_v6.py"
    assert script.is_file(), f"script not found at {script}"

    out_dir = tmp_path / "out"
    res = subprocess.run(
        [sys.executable, str(script), "--smoke", "--output-dir", str(out_dir)],
        cwd=str(repo_root),
        capture_output=True, text=True, timeout=120,
    )
    combined = res.stdout + res.stderr
    assert res.returncode == 0, (
        f"smoke failed: code={res.returncode}\nstdout={res.stdout}\nstderr={res.stderr}"
    )
    assert "v6 training:" in combined and "final_step=5" in combined
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists(), "metrics.json was not created"
    metrics = _read_metrics_json(metrics_path)
    assert [row["step"] for row in metrics] == [1, 2, 3, 4, 5]
    assert (out_dir / "step-00000005.pt").exists()
