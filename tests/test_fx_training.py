"""Tests for OSS-FX training pipeline: dataset shapes, losses, smoke train."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from oss.model.oss_fx import HISTORY_CH


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _write_png(path: Path, h: int = 64, w: int = 96, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(str(path))


def _write_flo(path: Path, h: int = 64, w: int = 96, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    flow = rng.standard_normal((h, w, 2)).astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<f", 202021.25))
        f.write(struct.pack("<ii", w, h))
        f.write(flow.tobytes())


def _make_sintel_root(tmp_path: Path, n_seqs: int = 2, n_frames: int = 6) -> Path:
    root = tmp_path / "sintel"
    for pass_name in ("clean", "final"):
        for si in range(n_seqs):
            seq_name = f"seq_{si:02d}"
            frame_dir = root / "training" / pass_name / seq_name
            frame_dir.mkdir(parents=True)
            for fi in range(1, n_frames + 1):
                _write_png(frame_dir / f"frame_{fi:04d}.png", seed=si * 100 + fi)

    for si in range(n_seqs):
        seq_name = f"seq_{si:02d}"
        flow_dir = root / "training" / "flow" / seq_name
        flow_dir.mkdir(parents=True)
        for fi in range(1, n_frames):
            _write_flo(flow_dir / f"frame_{fi:04d}.flo", seed=si * 100 + fi)

    return root


def _make_vimeo_root(tmp_path: Path, n_seqs: int = 4) -> Path:
    root = tmp_path / "vimeo"
    seq_paths = []
    for g in range(2):
        for s in range(n_seqs // 2):
            rel = f"{g+1:05d}/{s+1:04d}"
            d = root / "sequences" / rel
            d.mkdir(parents=True)
            for i in range(1, 8):
                _write_png(d / f"im{i}.png", seed=g * 10 + s * 7 + i)
            seq_paths.append(rel)

    (root / "sep_trainlist.txt").write_text("\n".join(seq_paths[:n_seqs]))
    (root / "sep_testlist.txt").write_text("\n".join(seq_paths[:max(1, n_seqs // 4)]))
    return root


# ---------------------------------------------------------------------------
# test_sintel_dataset_shape
# ---------------------------------------------------------------------------

def test_sintel_dataset_shape(tmp_path):
    sintel_root = _make_sintel_root(tmp_path, n_seqs=2, n_frames=6)
    from oss.data.sintel_fx import SintelFxDataset

    ds = SintelFxDataset(
        root=sintel_root,
        split="train",
        alpha_range=(0.2, 0.8),
        resolution=(64, 96),
        augment=False,
    )
    assert len(ds) > 0, "dataset must have at least one item"

    item = ds[0]
    required_keys = {"warped", "depth", "history", "alpha", "target", "frame_t"}
    assert required_keys == set(item.keys()), f"unexpected keys: {set(item.keys())}"

    H, W = 64, 96
    assert item["warped"].shape  == (3, H, W),         f"warped: {item['warped'].shape}"
    assert item["depth"].shape   == (1, H, W),         f"depth: {item['depth'].shape}"
    assert item["history"].shape == (HISTORY_CH, H, W), f"history: {item['history'].shape}"
    assert item["alpha"].shape   == (),                 f"alpha: {item['alpha'].shape}"
    assert item["target"].shape  == (3, H, W),         f"target: {item['target'].shape}"
    assert item["frame_t"].shape == (3, H, W),         f"frame_t: {item['frame_t'].shape}"

    assert 0.0 < item["alpha"].item() <= 1.0
    assert item["history"].sum().item() == 0.0


# ---------------------------------------------------------------------------
# test_vimeo_dataset_shape
# ---------------------------------------------------------------------------

def test_vimeo_dataset_shape(tmp_path):
    vimeo_root = _make_vimeo_root(tmp_path, n_seqs=4)
    from oss.data.vimeo90k_fx import Vimeo90kFxDataset

    ds = Vimeo90kFxDataset(
        root=vimeo_root,
        split="train",
        alpha_range=(0.2, 0.8),
        resolution=(64, 96),
        augment=False,
    )
    assert len(ds) > 0

    item = ds[0]
    required_keys = {"warped", "depth", "history", "alpha", "target", "frame_t"}
    assert required_keys == set(item.keys())

    H, W = 64, 96
    assert item["warped"].shape  == (3, H, W)
    assert item["depth"].shape   == (1, H, W)
    assert item["history"].shape == (HISTORY_CH, H, W)
    assert item["alpha"].ndim    == 0
    assert item["target"].shape  == (3, H, W)
    assert item["frame_t"].shape == (3, H, W)


# ---------------------------------------------------------------------------
# test_losses_fx
# ---------------------------------------------------------------------------

def test_losses_fx():
    from oss.train.losses_fx import extrapolation_loss

    B, H, W = 2, 32, 32
    pred      = torch.rand(B, 3, H, W)
    target    = torch.rand(B, 3, H, W)
    pred_prev = torch.rand(B, 3, H, W)
    alpha     = torch.rand(B) * 0.85 + 0.1

    loss = extrapolation_loss(pred, target, pred_prev, alpha)
    assert loss.ndim == 0, "loss must be scalar"
    assert loss.item() > 0.0
    loss.backward()


# ---------------------------------------------------------------------------
# test_train_fx_smoke
# ---------------------------------------------------------------------------

def test_train_fx_smoke(tmp_path):
    from oss.train.train_fx import train_fx

    out_dir = str(tmp_path / "fx_out")
    train_fx(
        sintel_root="",
        vimeo_root="",
        out_dir=out_dir,
        epochs=1,
        batch_size=2,
        lr=3e-4,
        device="cpu",
        smoke=True,
    )
    assert (Path(out_dir) / "oss_fx.pth").exists()
    ckpt = torch.load(Path(out_dir) / "oss_fx.pth", map_location="cpu")
    assert "model" in ckpt
    assert ckpt["config"]["history_ch"] == HISTORY_CH
