"""MPI Sintel adapter for OSS-Gaussian Sprint 4.

Sintel ships HR (1024x436) RGB frames + depth + flow per sequence. We use:
    - clean pass frames as HR target
    - depth (.dpt) as the depth G-buffer
    - flow (.flo) as the motion vector G-buffer
    - LR := box-downsample(HR, scale)
    - normals := derived from depth gradient

Canonical layout:
    root/
        training/
            clean/<seq>/frame_NNNN.png
            depth/<seq>/frame_NNNN.dpt
            flow/<seq>/frame_NNNN.flo
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import (
    CANVAS_CHANNELS,
    GaussianDataset,
    GaussianTrainingExample,
)

if TYPE_CHECKING:
    from .lr_synthesis import EngineAliasedLRSynth


# ---- Format readers (Sintel-canonical .flo and .dpt). ----------------------


def _read_flo(path: Path) -> Tensor:
    with open(path, "rb") as f:
        magic = struct.unpack("<f", f.read(4))[0]
        if abs(magic - 202021.25) > 0.001:
            raise ValueError(f"Invalid .flo magic {magic} in {path}")
        w, h = struct.unpack("<ii", f.read(8))
        data = torch.frombuffer(f.read(h * w * 2 * 4), dtype=torch.float32)
    return data.view(h, w, 2).permute(2, 0, 1).clone()  # (2, H, W)


def _read_dpt(path: Path) -> Tensor:
    """Sintel depth (.dpt): 4-byte magic + ints + float32 row-major depth."""
    with open(path, "rb") as f:
        magic = struct.unpack("<f", f.read(4))[0]
        if abs(magic - 202021.25) > 0.001:
            raise ValueError(f"Invalid .dpt magic {magic} in {path}")
        w, h = struct.unpack("<ii", f.read(8))
        data = torch.frombuffer(f.read(h * w * 4), dtype=torch.float32)
    return data.view(1, h, w).clone()  # (1, H, W)


def _load_png(path: Path) -> Tensor:
    from torchvision.io import read_image
    img = read_image(str(path)).float() / 255.0
    return img[:3]


def _normalize_depth(depth: Tensor) -> Tensor:
    """Sintel depth is in metres (large dynamic range). Normalize to [0,1]."""
    d = depth.clamp(min=0.0)
    p99 = torch.quantile(d.flatten(), 0.99).clamp(min=1e-6)
    return (d / p99).clamp(0.0, 1.0)


# ---- Dataset ---------------------------------------------------------------


class SintelGaussianDataset(GaussianDataset):
    """Sintel dataset adapter for the Gaussian param network.

    Args:
        root: path to Sintel "MPI-Sintel-complete" extracted root.
        scale: HR/LR ratio (the LR side is HR // scale via box downsample).
        pass_name: "clean" or "final" (clean by default — less film grain).
        synthetic: TEST-ONLY. When True, ``root`` is interpreted as a tiny
            fake on-disk fixture (see ``tests/gaussian/test_datasets.py``)
            generated with the canonical structure.
    """

    name = "sintel"

    def __init__(
        self,
        root: Path | str,
        scale: float = 2.0,
        pass_name: str = "clean",
        synthetic: bool = False,
        lr_synth: "EngineAliasedLRSynth | None" = None,
    ) -> None:
        super().__init__(root=root, scale=scale, lr_synth=lr_synth)
        self.pass_name = pass_name
        self.synthetic = synthetic

        pass_dir = self.root / "training" / pass_name
        flow_dir = self.root / "training" / "flow"
        depth_dir = self.root / "training" / "depth"

        if not pass_dir.is_dir():
            raise FileNotFoundError(
                f"Sintel dataset not found at {self.root}: missing {pass_dir}.\n"
                f"Expected layout: <root>/training/{{clean,final,flow,depth}}/<seq>/...\n"
                f"Download from http://sintel.is.tue.mpg.de/."
            )

        self._items: List[Tuple[Path, Path, Path]] = []
        for seq_dir in sorted(pass_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            seq = seq_dir.name
            frames = sorted(seq_dir.glob("frame_*.png"))
            for i, frame in enumerate(frames):
                stem = frame.stem
                flow_path = flow_dir / seq / f"{stem}.flo"
                depth_path = depth_dir / seq / f"{stem}.dpt"
                # Last frame in each sequence has no flow_t; skip it.
                if not flow_path.exists() or not depth_path.exists():
                    continue
                self._items.append((frame, depth_path, flow_path))

        if not self._items:
            raise FileNotFoundError(
                f"Sintel root {self.root} found but no (frame, depth, flow) "
                "triples discovered. Did the download finish?"
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> GaussianTrainingExample:
        frame_path, depth_path, flow_path = self._items[idx]
        hr = _load_png(frame_path)                # (3, H, W)
        depth_hr = _read_dpt(depth_path)          # (1, H, W)
        flow_hr = _read_flo(flow_path)            # (2, H, W)

        # LR: use engine-aliased synthesis when available, box-downsample otherwise.
        lr = (
            self.lr_synth.synthesize(hr, idx)
            if self.lr_synth is not None
            else self._box_downsample(hr, self.scale)
        )
        lr_h, lr_w = lr.shape[-2:]

        # Depth + motion + normals at LR resolution.
        depth_lr = self._box_downsample(_normalize_depth(depth_hr), self.scale)
        # Flow values are pixel offsets at HR; rescale magnitude for LR.
        motion_lr = (
            F.interpolate(
                flow_hr.unsqueeze(0), size=(lr_h, lr_w), mode="bilinear", align_corners=False
            ).squeeze(0)
            / float(self.scale)
        )
        normals_lr = self._depth_to_normals(depth_lr)
        canvas = self._zero_canvas(lr)

        return GaussianTrainingExample(
            lr_frame=lr.contiguous().float(),
            depth=depth_lr.contiguous().float(),
            motion=motion_lr.contiguous().float(),
            normals=normals_lr.contiguous().float(),
            canvas_hint=canvas.contiguous().float(),
            gt_hr_frame=hr.contiguous().float(),
            metadata={
                "source": "sintel",
                "frame_path": str(frame_path),
                "scale": self.scale,
            },
        )


__all__ = ["SintelGaussianDataset"]
