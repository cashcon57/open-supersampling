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
    ) -> None:
        super().__init__(root=root, scale=scale, lr_synth=lr_synth)
        self.synthetic = synthetic
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"SRGD dataset not found at {self.root}.\n"
                "Expected layout: <root>/hr/*.png (and optionally <root>/lr/*.png)."
            )
        hr_dir = self.root / "hr"
        if not hr_dir.is_dir():
            raise FileNotFoundError(
                f"SRGD HR directory missing: {hr_dir}. "
                "Expected layout: <root>/hr/*.png."
            )
        lr_dir = self.root / "lr"
        self._items: List[Tuple[Path, Path | None]] = []
        for hr in sorted(hr_dir.glob("*.png")):
            lr_candidate = lr_dir / hr.name if lr_dir.is_dir() else None
            self._items.append((hr, lr_candidate if (lr_candidate and lr_candidate.exists()) else None))
        if not self._items:
            raise FileNotFoundError(
                f"SRGD HR directory {hr_dir} contains no PNGs."
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
