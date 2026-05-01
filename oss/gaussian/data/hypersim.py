"""HyperSim adapter for OSS-Gaussian Sprint 4.

HyperSim is photorealistic synthetic indoor scenes. Frames are independent
(no per-frame motion), so motion vectors are zero. Used for pretraining only.

Canonical layout (per scene / cam):
    root/<scene>/images/scene_cam_NN_final_preview/frame.NNNN.tonemap.jpg
    root/<scene>/images/scene_cam_NN_geometry_hdf5/frame.NNNN.depth_meters.hdf5
    root/<scene>/images/scene_cam_NN_geometry_hdf5/frame.NNNN.normal_cam.hdf5

For practicality (and h5py is already a dep), we accept either:
    - .hdf5 files (real layout)
    - .npy fallback (test fixtures)

Motion is always zero (3 ch wide spatial, fits the 12-ch contract).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from .base import GaussianDataset, GaussianTrainingExample, MOTION_CHANNELS


def _load_image(path: Path) -> torch.Tensor:
    from torchvision.io import read_image
    img = read_image(str(path)).float() / 255.0
    return img[:3]


def _load_array(path: Path, key_hint: str) -> np.ndarray:
    """Load HyperSim per-pixel data. Supports .hdf5 (real) and .npy (fixtures)."""
    if path.suffix.lower() in {".npy"}:
        return np.load(path)
    if path.suffix.lower() in {".hdf5", ".h5"}:
        import h5py
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            # HyperSim uses "dataset" as the canonical key; fall back to first key.
            key = "dataset" if "dataset" in keys else keys[0]
            return np.array(f[key])
    raise ValueError(f"unsupported HyperSim payload extension: {path}")


def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    d = depth.clamp(min=0.0)
    p99 = torch.quantile(d.flatten(), 0.99).clamp(min=1e-6)
    return (d / p99).clamp(0.0, 1.0)


class HyperSimGaussianDataset(GaussianDataset):
    """HyperSim adapter (static frames; motion=0)."""

    name = "hypersim"

    def __init__(
        self,
        root: Path | str,
        scale: float = 2.0,
        synthetic: bool = False,
    ) -> None:
        super().__init__(root=root, scale=scale)
        self.synthetic = synthetic
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"HyperSim dataset not found at {self.root}.\n"
                "Expected layout: <root>/<scene>/images/scene_cam_NN_final_preview/*.jpg\n"
                "Download from https://github.com/apple/ml-hypersim."
            )

        self._items: List[Tuple[Path, Path, Path | None]] = []
        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir():
                continue
            images_root = scene_dir / "images"
            if not images_root.is_dir():
                continue
            for cam_dir in sorted(images_root.iterdir()):
                if not cam_dir.is_dir() or "_final_preview" not in cam_dir.name:
                    continue
                geom_dir = images_root / cam_dir.name.replace("_final_preview", "_geometry_hdf5")
                # frame.NNNN.tonemap.jpg OR frame.NNNN.color.jpg OR fixture .jpg
                color_files = sorted(cam_dir.glob("frame.*.tonemap.jpg")) or sorted(
                    cam_dir.glob("frame.*.jpg")
                )
                for color in color_files:
                    # frame.0000.tonemap.jpg → "0000"
                    parts = color.name.split(".")
                    if len(parts) < 3:
                        continue
                    fid = parts[1]
                    depth_path = None
                    for ext in (".depth_meters.hdf5", ".depth_meters.npy"):
                        cand = geom_dir / f"frame.{fid}{ext}"
                        if cand.exists():
                            depth_path = cand
                            break
                    if depth_path is None:
                        continue
                    self._items.append((color, depth_path, None))

        if not self._items:
            raise FileNotFoundError(
                f"HyperSim root {self.root} found but no (color, depth) pairs "
                "were discovered. Did the download finish?"
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> GaussianTrainingExample:
        color_path, depth_path, _ = self._items[idx]
        hr = _load_image(color_path)
        depth_arr = _load_array(depth_path, key_hint="depth")
        depth_hr = torch.from_numpy(depth_arr).float()
        if depth_hr.dim() == 2:
            depth_hr = depth_hr.unsqueeze(0)
        elif depth_hr.dim() == 3 and depth_hr.shape[-1] == 1:
            depth_hr = depth_hr.permute(2, 0, 1)

        lr = self._box_downsample(hr, self.scale)
        lr_h, lr_w = lr.shape[-2:]
        depth_lr = self._box_downsample(_normalize_depth(depth_hr), self.scale)
        # Static frames → motion is zero.
        motion_lr = torch.zeros((MOTION_CHANNELS, lr_h, lr_w), dtype=torch.float32)
        normals_lr = self._depth_to_normals(depth_lr)
        canvas = self._zero_canvas(lr)

        return GaussianTrainingExample(
            lr_frame=lr.contiguous().float(),
            depth=depth_lr.contiguous().float(),
            motion=motion_lr,
            normals=normals_lr.contiguous().float(),
            canvas_hint=canvas.contiguous().float(),
            gt_hr_frame=hr.contiguous().float(),
            metadata={
                "source": "hypersim",
                "frame_path": str(color_path),
                "scale": self.scale,
                "static": True,
            },
        )


__all__ = ["HyperSimGaussianDataset"]
