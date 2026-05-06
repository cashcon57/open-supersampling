"""TartanAir adapter for OSS-Gaussian Sprint 4.

TartanAir is a synthetic indoor/outdoor dataset with full G-buffers at
640x480 per scene, distributed under environments/levels/trajectories.

Canonical layout (per environment / Easy or Hard / P000..):
    root/<env>/<level>/<traj>/
        image_left/000000_left.png
        depth_left/000000_left_depth.npy
        flow/000000_000001_flow.npy        # forward flow to next frame
        seg_left/000000_left_seg.npy       # optional, ignored
        pose_left.txt                      # not used here

For our purposes:
    - HR target := image_left
    - LR := box-downsample(HR, scale)
    - depth := depth_left (in metres) → normalize to [0, 1]
    - motion := flow_NN.npy (pixel offsets) — already screen-space at HR
      (TartanAir's flow is per-pixel and screen-space; the prior plan note
      about scene-flow conversion applies only to TartanAir-V2 stereo flow,
      not the standard ``flow/`` tensors used here)
    - normals := derived from depth gradient
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .base import GaussianDataset, GaussianTrainingExample

if TYPE_CHECKING:
    from .lr_synthesis import EngineAliasedLRSynth


def _load_png(path: Path) -> torch.Tensor:
    from torchvision.io import read_image
    img = read_image(str(path)).float() / 255.0
    return img[:3]


def _load_npy_chw(path: Path, channels: int) -> torch.Tensor:
    try:
        arr = np.load(path)
    except Exception as e:
        raise ValueError(f"could not load npy at {path}: {e}") from e
    if arr.ndim == 2:
        # (H, W) → (1, H, W) — depth case
        t = torch.from_numpy(arr).float().unsqueeze(0)
    elif arr.ndim == 3:
        # could be (H, W, C) or (C, H, W)
        if arr.shape[0] in (1, 2, 3) and arr.shape[-1] not in (1, 2, 3):
            t = torch.from_numpy(arr).float()
        else:
            t = torch.from_numpy(arr).float().permute(2, 0, 1)
    else:
        raise ValueError(f"unsupported npy shape {arr.shape} at {path}")
    if t.shape[0] != channels:
        raise ValueError(
            f"{path}: expected {channels} channels; got {t.shape[0]} (shape {tuple(t.shape)})"
        )
    return t.contiguous()


def _normalize_depth_metres(depth: torch.Tensor) -> torch.Tensor:
    d = depth.clamp(min=0.0)
    p99 = torch.quantile(d.flatten(), 0.99).clamp(min=1e-6)
    return (d / p99).clamp(0.0, 1.0)


class TartanAirGaussianDataset(GaussianDataset):
    """TartanAir adapter.

    Args:
        root: path containing TartanAir environments.
        scale: HR/LR ratio (LR is HR // scale).
        synthetic: TEST-ONLY fixture mode (see test_datasets.py).
    """

    name = "tartanair"

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
                f"TartanAir dataset not found at {self.root}.\n"
                "Expected layout: <root>/<env>/<level>/<traj>/{image_left,depth_left,flow}/...\n"
                "Download from https://theairlab.org/tartanair-dataset/."
            )
        self._items: List[Tuple[Path, Path, Path]] = []
        # Walk env / level / traj. Tolerant: any traj dir that has the three
        # expected sub-dirs is included.
        for traj_dir in sorted(self.root.glob("*/*/*")):
            img_dir = traj_dir / "image_left"
            depth_dir = traj_dir / "depth_left"
            flow_dir = traj_dir / "flow"
            if not (img_dir.is_dir() and depth_dir.is_dir() and flow_dir.is_dir()):
                continue
            frames = sorted(img_dir.glob("*_left.png"))
            for frame in frames:
                idx = frame.stem.split("_")[0]  # zero-padded frame number
                next_idx = f"{int(idx) + 1:0{len(idx)}d}"
                depth_path = depth_dir / f"{idx}_left_depth.npy"
                flow_path = flow_dir / f"{idx}_{next_idx}_flow.npy"
                if not (depth_path.exists() and flow_path.exists()):
                    continue
                self._items.append((frame, depth_path, flow_path))

        if not self._items:
            raise FileNotFoundError(
                f"TartanAir root {self.root} found but no (img, depth, flow) "
                "triples were discovered. Did the download finish? "
                "Expected files like image_left/000000_left.png + "
                "depth_left/000000_left_depth.npy + flow/000000_000001_flow.npy."
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> GaussianTrainingExample:
        frame_path, depth_path, flow_path = self._items[idx]
        try:
            hr = _load_png(frame_path)
            depth_hr = _load_npy_chw(depth_path, channels=1)
            flow_hr = _load_npy_chw(flow_path, channels=2)
        except (ValueError, OSError) as e:
            # Defensive: a single corrupt file in a 200 GB dataset
            # would otherwise kill the whole training run. Log + try
            # the next index. Caps at 10 retries to avoid an infinite
            # loop on systemic dataset corruption.
            import logging
            log = logging.getLogger("oss.gaussian.data.tartanair")
            log.warning(
                "skipping corrupt TartanAir sample idx=%d (%s): %s",
                idx, frame_path, e,
            )
            n = len(self._items)
            for offset in range(1, 11):
                next_idx = (idx + offset) % n
                next_frame, next_depth, next_flow = self._items[next_idx]
                try:
                    hr = _load_png(next_frame)
                    depth_hr = _load_npy_chw(next_depth, channels=1)
                    flow_hr = _load_npy_chw(next_flow, channels=2)
                    frame_path, depth_path, flow_path = (
                        next_frame, next_depth, next_flow,
                    )
                    break
                except (ValueError, OSError) as e2:
                    log.warning(
                        "skipping corrupt TartanAir sample idx=%d: %s",
                        next_idx, e2,
                    )
                    continue
            else:
                raise RuntimeError(
                    f"10 consecutive corrupt TartanAir samples starting at idx={idx} "
                    f"— dataset is broken, not a one-off; bail out."
                ) from e

        lr = (
            self.lr_synth.synthesize(hr, idx)
            if self.lr_synth is not None
            else self._box_downsample(hr, self.scale)
        )
        lr_h, lr_w = lr.shape[-2:]
        depth_lr = self._box_downsample(_normalize_depth_metres(depth_hr), self.scale)
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
                "source": "tartanair",
                "frame_path": str(frame_path),
                "scale": self.scale,
            },
        )


__all__ = ["TartanAirGaussianDataset"]
