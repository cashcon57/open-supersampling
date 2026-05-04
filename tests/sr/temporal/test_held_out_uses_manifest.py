"""Held-out eval manifest path tests.

The full eval needs real TartanAir/Sintel data and checkpoints. These tests
exercise the manifest-specific loader branch against a tiny stub base dataset
so the CI path proves that ``scripts/sr_temporal_held_out.py --manifest`` uses
``load_manifest`` + ``manifest_to_pairs`` instead of default pair enumeration.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import scripts.sr_temporal_held_out as held_out


class _StubBaseDataset:
    def __init__(self) -> None:
        self.frames = [
            ("/data/env/Easy/P000", 0),
            ("/data/env/Easy/P000", 1),
            ("/data/env/Easy/P000", 2),
            ("/data/env/Easy/P001", 0),
            ("/data/env/Easy/P001", 1),
        ]

    def __len__(self) -> int:
        return len(self.frames)

    def trajectory_key(self, idx: int) -> str:
        return self.frames[idx][0]

    def frame_index(self, idx: int) -> int:
        return self.frames[idx][1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        value = float(idx)
        lr = torch.full((3, 2, 2), value)
        return {
            "lr_frame": lr,
            "depth": torch.full((1, 2, 2), value),
            "motion": torch.full((2, 2, 2), value),
            "normals": torch.full((3, 2, 2), value),
            "canvas_hint": torch.zeros(3, 2, 2),
            "gt_hr_frame": torch.full((3, 4, 4), value),
        }


def _write_manifest(
    path: Path,
    *,
    dataset_kind: str = "tartanair",
    lr_scale: float = 2.0,
    extra_lr: dict[str, Any] | None = None,
) -> None:
    lr_synth_args = dict(held_out.DEFAULT_LR_SYNTH_ARGS)
    if extra_lr:
        lr_synth_args.update(extra_lr)
    manifest = {
        "manifest_version": 1,
        "dataset_kind": dataset_kind,
        "n_pairs": 2,
        "seed": 0,
        "lr_scale": lr_scale,
        "lr_synth_args": lr_synth_args,
        "pairs": [
            {
                "trajectory": "/data/env/Easy/P000",
                "idx_t": 1,
                "idx_t_plus_1": 2,
            },
            {
                "trajectory": "/data/env/Easy/P001",
                "idx_t": 0,
                "idx_t_plus_1": 1,
            },
        ],
    }
    path.write_text(json.dumps(manifest))


def test_manifest_loader_replays_explicit_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "held_out_manifest.json"
    _write_manifest(manifest_path)

    monkeypatch.setattr(
        held_out,
        "_build_manifest_base_dataset",
        lambda *args, **kwargs: _StubBaseDataset(),
    )

    loader = held_out._build_manifest_loader(
        "tartanair",
        tmp_path,
        manifest_path,
        batch_size=2,
        scale=2.0,
        lr_synth_args=held_out.DEFAULT_LR_SYNTH_ARGS,
    )

    batch = next(iter(loader))
    assert batch["t_lr"].shape == (2, 3, 2, 2)
    assert batch["tp1_gt_hr"].shape == (2, 3, 4, 4)
    # Manifest pair order is P000 frame 1 -> 2, then P001 frame 0 -> 1.
    assert torch.equal(batch["t_lr"][:, 0, 0, 0], torch.tensor([1.0, 3.0]))
    assert torch.equal(batch["tp1_lr"][:, 0, 0, 0], torch.tensor([2.0, 4.0]))


def test_manifest_loader_rejects_config_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "held_out_manifest.json"
    _write_manifest(manifest_path, extra_lr={"enable_jpeg": True})
    monkeypatch.setattr(
        held_out,
        "_build_manifest_base_dataset",
        lambda *args, **kwargs: _StubBaseDataset(),
    )

    with pytest.raises(ValueError, match="lr_synth_args mismatch"):
        held_out._build_manifest_loader(
            "tartanair",
            tmp_path,
            manifest_path,
            batch_size=2,
            scale=2.0,
            lr_synth_args=held_out.DEFAULT_LR_SYNTH_ARGS,
        )


def test_manifest_loader_rejects_scale_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "held_out_manifest.json"
    _write_manifest(manifest_path, lr_scale=4.0)
    monkeypatch.setattr(
        held_out,
        "_build_manifest_base_dataset",
        lambda *args, **kwargs: _StubBaseDataset(),
    )

    with pytest.raises(ValueError, match="lr_scale mismatch"):
        held_out._build_manifest_loader(
            "tartanair",
            tmp_path,
            manifest_path,
            batch_size=2,
            scale=2.0,
            lr_synth_args=held_out.DEFAULT_LR_SYNTH_ARGS,
        )


def test_dual_manifest_loader_routes_each_manifest_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tartan_manifest = tmp_path / "held_out_tartanair.json"
    sintel_manifest = tmp_path / "held_out_sintel.json"
    _write_manifest(tartan_manifest, dataset_kind="tartanair")
    _write_manifest(sintel_manifest, dataset_kind="sintel")

    calls: list[str] = []

    def _fake_base(kind: str, *args: Any, **kwargs: Any) -> _StubBaseDataset:
        calls.append(kind)
        return _StubBaseDataset()

    monkeypatch.setattr(held_out, "_build_manifest_base_dataset", _fake_base)

    loaders = held_out._build_manifest_loaders(
        held_out._split_manifest_paths(f"{tartan_manifest},{sintel_manifest}"),
        tartanair_root=tmp_path / "tartan",
        sintel_root=tmp_path / "sintel",
        batch_size=2,
        scale=2.0,
        lr_synth_args=held_out.DEFAULT_LR_SYNTH_ARGS,
    )

    assert [name for name, _loader in loaders] == ["tartanair", "sintel"]
    assert calls == ["tartanair", "sintel"]
    batches = [next(iter(loader)) for _name, loader in loaders]
    assert [batch["t_lr"].shape for batch in batches] == [(2, 3, 2, 2), (2, 3, 2, 2)]
