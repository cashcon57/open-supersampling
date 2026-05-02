"""SRGD adapter for OSS-Gaussian Sprint 4.

SRGD (Synthetic Resolution Gradient Dataset) provides paired LR/HR frames
with no G-buffers. We synthesize the missing channels:
    - depth: zeros (no depth in dataset)
    - motion: zeros (single-frame pairs)
    - normals: zeros (no surface info)

Used as a small-weight (5%) anchor for the SR-only signal during training.

Canonical layout:
    root/
        hr/<name>.png
        lr/<name>.png         # optional — if missing we box-downsample HR
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn.functional as F

from .base import (
    DEPTH_CHANNELS,
    GaussianDataset,
    GaussianTrainingExample,
    MOTION_CHANNELS,
    NORMAL_CHANNELS,
)

if TYPE_CHECKING:
    from .lr_synthesis import EngineAliasedLRSynth


def _load_png(path: Path) -> torch.Tensor:
    from torchvision.io import read_image
    img = read_image(str(path)).float() / 255.0
    return img[:3]


class SRGDGaussianDataset(GaussianDataset):
    """SRGD synthetic SR pairs."""

    name = "srgd"

    def __init__(
        self,
        root: Path | str,
        scale: float = 2.0,
        synthetic: bool = False,
        lr_synth: "EngineAliasedLRSynth | None" = None,
        scene: str | None = None,
    ) -> None:
        super().__init__(root=root, scale=scale, lr_synth=lr_synth)
        self.synthetic = synthetic
        self.scene = scene
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"SRGD dataset not found at {self.root}.\n"
                "Expected layouts:\n"
                "  - <root>/hr/*.png (canonical), or\n"
                "  - <root>/data/GameEngineData/<scene>/*.png paired with\n"
                "    <root>/data/DownscaleData/<scene>/*.png (3080 Ti layout)."
            )

        # Try canonical hr/ layout first.
        hr_dir = self.root / "hr"
        if hr_dir.is_dir():
            lr_dir = self.root / "lr"
            self._items: List[Tuple[Path, Path | None]] = []
            for hr in sorted(hr_dir.glob("*.png")):
                lr_candidate = lr_dir / hr.name if lr_dir.is_dir() else None
                self._items.append(
                    (hr, lr_candidate if (lr_candidate and lr_candidate.exists()) else None)
                )
            if not self._items:
                raise FileNotFoundError(f"SRGD HR directory {hr_dir} contains no PNGs.")
            return

        # Fall back to GameEngineData/DownscaleData layout (per-scene).
        ge_root = self.root / "data" / "GameEngineData"
        ds_root = self.root / "data" / "DownscaleData"
        if not ge_root.is_dir():
            raise FileNotFoundError(
                f"Neither {hr_dir} nor {ge_root} exist. "
                "Cannot determine SRGD layout."
            )

        if scene is not None:
            scene_dirs = [ge_root / scene]
            if not scene_dirs[0].is_dir():
                raise FileNotFoundError(
                    f"SRGD scene {scene!r} not found at {scene_dirs[0]}"
                )
        else:
            scene_dirs = sorted(p for p in ge_root.iterdir() if p.is_dir())

        self._items = []
        for scene_dir in scene_dirs:
            ds_scene = ds_root / scene_dir.name if ds_root.is_dir() else None
            for hr in sorted(scene_dir.glob("*.png")):
                lr_candidate = (ds_scene / hr.name) if ds_scene is not None else None
                self._items.append(
                    (hr, lr_candidate if (lr_candidate and lr_candidate.exists()) else None)
                )
        if not self._items:
            raise FileNotFoundError(
                f"SRGD GameEngineData under {ge_root} contains no PNGs"
                + (f" for scene {scene!r}" if scene else "")
                + "."
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> GaussianTrainingExample:
        hr_path, lr_path = self._items[idx]
        hr = _load_png(hr_path)
        if lr_path is not None:
            lr = _load_png(lr_path)
            # If shapes don't line up with scale, fall back to synthesis/box-downsample.
            target_h = hr.shape[-2] // int(round(self.scale))
            target_w = hr.shape[-1] // int(round(self.scale))
            if lr.shape[-2:] != (target_h, target_w):
                lr = (
                    self.lr_synth.synthesize(hr, idx)
                    if self.lr_synth is not None
                    else self._box_downsample(hr, self.scale)
                )
        else:
            lr = (
                self.lr_synth.synthesize(hr, idx)
                if self.lr_synth is not None
                else self._box_downsample(hr, self.scale)
            )

        lr_h, lr_w = lr.shape[-2:]
        depth_lr = torch.zeros((DEPTH_CHANNELS, lr_h, lr_w), dtype=torch.float32)
        motion_lr = torch.zeros((MOTION_CHANNELS, lr_h, lr_w), dtype=torch.float32)
        normals_lr = torch.zeros((NORMAL_CHANNELS, lr_h, lr_w), dtype=torch.float32)
        # third axis of normals points "up" in the absence of geometry — keeps
        # the network from learning a degenerate zero-everywhere normal.
        normals_lr[2] = 1.0
        canvas = self._zero_canvas(lr)

        return GaussianTrainingExample(
            lr_frame=lr.contiguous().float(),
            depth=depth_lr,
            motion=motion_lr,
            normals=normals_lr,
            canvas_hint=canvas.contiguous().float(),
            gt_hr_frame=hr.contiguous().float(),
            metadata={
                "source": "srgd",
                "frame_path": str(hr_path),
                "scale": self.scale,
                "g_buffers_synthetic": True,
            },
        )


__all__ = ["SRGDGaussianDataset"]
