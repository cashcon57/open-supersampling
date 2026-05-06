"""v6 dataset: Hypersim + TartanAir mix with 70/30 importance sampling.

Hypersim must be downloaded from https://github.com/apple/ml-hypersim.
The expected on-disk layout is the published Hypersim structure::

    root/
      ai_001_001/
        images/
          scene_cam_00_final_preview/
            frame.0000.tonemap.jpg     # HR sRGB color (8-bit)
            frame.0001.tonemap.jpg
            ...
          scene_cam_00_geometry_hdf5/
            frame.0000.depth_meters.hdf5
            frame.0000.normal_cam.hdf5
            ...
      ai_001_002/
      ...

Per the v6 architecture memo (``docs/superpowers/experiments/
2026-05-05-v6-architecture-canonical.md``) section 6, the v6 training mix is::

    TartanAir 60% + Hypersim 30% + held-out 10%

The held-out 10% is excluded from the training mix; this module yields
only the 60/30 training portion. Hypersim is single-frame photoreal
indoor data with real depth + normals and no flow — so motion and
canvas-hint channels are zeros. LR is synthesised from HR via the same
:class:`~oss.gaussian.data.lr_synthesis.EngineAliasedLRSynth` used for
TartanAir to keep the LR distribution consistent across the mix.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from oss.gaussian.data.lr_synthesis import EngineAliasedLRSynth
from oss.gaussian.data.tartanair import TartanAirGaussianDataset


# ---------------------------------------------------------------------------
# Hypersim file loaders
# ---------------------------------------------------------------------------


def _load_image(path: Path) -> torch.Tensor:
    from torchvision.io import read_image

    img = read_image(str(path)).float() / 255.0
    return img[:3]


def _load_hdf5_or_npy(path: Path) -> np.ndarray:
    """Load a Hypersim per-pixel file. Supports .hdf5 (canonical) and .npy
    (test fixtures)."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".hdf5", ".h5"}:
        try:
            import h5py
        except ImportError as e:  # pragma: no cover - h5py is in pyproject.toml
            raise ImportError(
                "h5py is required to read Hypersim .hdf5 files. Install with "
                "`pip install h5py>=3.10`."
            ) from e
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            key = "dataset" if "dataset" in keys else keys[0]
            return np.array(f[key])
    raise ValueError(f"unsupported Hypersim payload extension: {path}")


def _normalize_depth_metres(depth: torch.Tensor) -> torch.Tensor:
    """Match the TartanAir depth normalization in
    :mod:`oss.gaussian.data.tartanair`."""
    d = depth.clamp(min=0.0)
    p99 = torch.quantile(d.flatten(), 0.99).clamp(min=1e-6)
    return (d / p99).clamp(0.0, 1.0)


def _to_chw(arr: np.ndarray, expected_c: int) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if t.dim() == 2:
        t = t.unsqueeze(0)
    elif t.dim() == 3:
        if t.shape[0] == expected_c:
            pass  # already CHW
        elif t.shape[-1] == expected_c:
            t = t.permute(2, 0, 1)
        else:
            raise ValueError(
                f"could not coerce array of shape {tuple(arr.shape)} to "
                f"({expected_c}, H, W)"
            )
    else:
        raise ValueError(f"unsupported array rank {t.dim()}")
    return t.contiguous()


# ---------------------------------------------------------------------------
# HypersimDataset
# ---------------------------------------------------------------------------


class HypersimDataset(Dataset):
    """Photoreal indoor scenes from Blender Cycles. Real depth + normals,
    no flow (single-image / pair regime). 8-bit sRGB by default.

    Yields one item per frame: a dict with keys::

        {
            "lr_frame":     (3, H, W) float32 — LR RGB synthesised from HR
            "gt_hr_frame":  (3, H_hr, W_hr) float32 — HR sRGB
            "depth":        (1, H, W) float32 — normalised depth in [0, 1]
            "normals":      (3, H, W) float32 — surface normals in roughly [-1, 1]
            "motion":       (2, H, W) float32 — zeros (Hypersim is single-frame)
            "canvas_hint":  (3, H, W) float32 — zeros (no canvas in source)
            "metadata":     dict
        }
    """

    name = "hypersim"

    def __init__(
        self,
        root: Path | str,
        scale: float = 2.0,
        lr_synth: EngineAliasedLRSynth | None = None,
        held_out_scenes: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root)
        if scale < 1.0:
            raise ValueError(f"scale must be >=1.0; got {scale}")
        self.scale = float(scale)
        self.lr_synth = lr_synth
        self.held_out_scenes = set(held_out_scenes or [])

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Hypersim dataset not found at {self.root}.\n"
                "Expected layout: <root>/<scene>/images/scene_cam_NN_final_preview/*.jpg\n"
                "Download from https://github.com/apple/ml-hypersim."
            )

        # Items: (color_path, depth_path, normal_path_or_None, scene_name)
        self._items: List[tuple[Path, Path, Path | None, str]] = []
        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_name = scene_dir.name
            if scene_name in self.held_out_scenes:
                continue
            images_root = scene_dir / "images"
            if not images_root.is_dir():
                continue
            for cam_dir in sorted(images_root.iterdir()):
                if not cam_dir.is_dir() or "_final_preview" not in cam_dir.name:
                    continue
                geom_dir = images_root / cam_dir.name.replace(
                    "_final_preview", "_geometry_hdf5"
                )
                color_files = sorted(cam_dir.glob("frame.*.tonemap.jpg")) or sorted(
                    cam_dir.glob("frame.*.jpg")
                )
                for color in color_files:
                    parts = color.name.split(".")
                    if len(parts) < 3:
                        continue
                    fid = parts[1]
                    depth_path: Path | None = None
                    for ext in (".depth_meters.hdf5", ".depth_meters.npy"):
                        cand = geom_dir / f"frame.{fid}{ext}"
                        if cand.exists():
                            depth_path = cand
                            break
                    if depth_path is None:
                        continue
                    normal_path: Path | None = None
                    for ext in (".normal_cam.hdf5", ".normal_cam.npy"):
                        cand = geom_dir / f"frame.{fid}{ext}"
                        if cand.exists():
                            normal_path = cand
                            break
                    self._items.append((color, depth_path, normal_path, scene_name))

        if not self._items:
            raise FileNotFoundError(
                f"Hypersim root {self.root} found but no (color, depth) pairs "
                "were discovered. Did the download finish?"
            )

    def __len__(self) -> int:
        return len(self._items)

    def trajectory_key(self, idx: int) -> str:
        # Hypersim is per-frame static; treat the scene name as the
        # trajectory key so window datasets that gate on equal keys still
        # work (each frame is its own one-frame trajectory in practice).
        _, _, _, scene_name = self._items[idx]
        return scene_name

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        color_path, depth_path, normal_path, scene_name = self._items[idx]
        hr = _load_image(color_path)
        depth_arr = _load_hdf5_or_npy(depth_path)
        depth_hr = _to_chw(depth_arr, expected_c=1)

        if self.lr_synth is not None:
            lr = self.lr_synth.synthesize(hr, idx)
        else:
            lr = _box_downsample(hr, self.scale)

        lr_h, lr_w = lr.shape[-2:]
        depth_lr = _box_downsample(_normalize_depth_metres(depth_hr), self.scale)

        if normal_path is not None:
            normals_arr = _load_hdf5_or_npy(normal_path)
            normals_hr = _to_chw(normals_arr, expected_c=3)
            # Resample to LR. Use bilinear (normals are smooth-ish).
            normals_lr = F.interpolate(
                normals_hr.unsqueeze(0), size=(lr_h, lr_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            # Re-normalize per-pixel.
            n_norm = normals_lr.norm(dim=0, keepdim=True).clamp(min=1e-6)
            normals_lr = normals_lr / n_norm
        else:
            # Cheap normal-from-depth fallback (matches TartanAir).
            normals_lr = _depth_to_normals(depth_lr)

        motion_lr = torch.zeros((2, lr_h, lr_w), dtype=torch.float32)
        canvas = torch.zeros((3, lr_h, lr_w), dtype=torch.float32)

        return {
            "lr_frame": lr.contiguous().float(),
            "gt_hr_frame": hr.contiguous().float(),
            "depth": depth_lr.contiguous().float(),
            "normals": normals_lr.contiguous().float(),
            "motion": motion_lr,
            "canvas_hint": canvas,
            "metadata": {
                "source": "hypersim",
                "frame_path": str(color_path),
                "scene": scene_name,
                "scale": self.scale,
                "static": True,
            },
        }


# Local helpers — keep tartanair.py untouched.


def _box_downsample(hr: torch.Tensor, scale: float) -> torch.Tensor:
    if hr.dim() != 3:
        raise ValueError(f"expected (C,H,W); got {tuple(hr.shape)}")
    s = int(round(scale))
    if s < 1:
        raise ValueError(f"scale must be >=1; got {scale}")
    if s == 1:
        return hr.clone()
    C, H, W = hr.shape
    H2 = (H // s) * s
    W2 = (W // s) * s
    x = hr[:, :H2, :W2]
    x = x.view(C, H2 // s, s, W2 // s, s).mean(dim=(2, 4))
    return x


def _depth_to_normals(depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() != 3 or depth.shape[0] != 1:
        raise ValueError(f"expected (1,H,W); got {tuple(depth.shape)}")
    d = depth[0]
    dx = torch.zeros_like(d)
    dy = torch.zeros_like(d)
    dx[:, 1:-1] = d[:, 2:] - d[:, :-2]
    dy[1:-1, :] = d[2:, :] - d[:-2, :]
    nz = torch.ones_like(d)
    n = torch.stack([-dx, -dy, nz], dim=0)
    norm = n.norm(dim=0, keepdim=True).clamp(min=1e-6)
    return n / norm


# ---------------------------------------------------------------------------
# TartanAir adapter that yields the v6 dict shape
# ---------------------------------------------------------------------------


class _TartanAirV6Wrapper(Dataset):
    """Adapt :class:`TartanAirGaussianDataset` (which yields a
    GaussianTrainingExample) to the v6 dict shape."""

    name = "tartanair_v6"

    def __init__(
        self,
        base: TartanAirGaussianDataset,
        held_out_envs: Sequence[str] | None = None,
    ) -> None:
        self.base = base
        held = set(held_out_envs or [])
        if held:
            self.base._items = [
                t for t in self.base._items
                if not any(env in t[0].parts for env in held)
            ]

    def __len__(self) -> int:
        return len(self.base)

    def trajectory_key(self, idx: int) -> str:
        frame_path, _, _ = self.base._items[idx]
        # Use the trajectory directory as key.
        return str(frame_path.parent.parent)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        ex = self.base[idx]
        return {
            "lr_frame": ex.lr_frame,
            "gt_hr_frame": ex.gt_hr_frame,
            "depth": ex.depth,
            "normals": (
                ex.normals
                if ex.normals is not None
                else torch.zeros((3, *ex.lr_frame.shape[-2:]), dtype=torch.float32)
            ),
            "motion": ex.motion,
            "canvas_hint": ex.canvas_hint,
            "metadata": dict(ex.metadata),
        }


# ---------------------------------------------------------------------------
# MixedTartanAirHypersimDataset
# ---------------------------------------------------------------------------


class MixedTartanAirHypersimDataset(Dataset):
    """Interleaves TartanAir and Hypersim with the v6 60/30 ratio (held-out
    10% excluded). Only training data is yielded — eval is built separately.

    The two sources have different lengths; we expose a virtual length
    equal to the sum and pick a source per-index using the configured
    ratio with a deterministic per-index hash so that multiple workers
    see the same source assignment.

    Args:
        tartanair: a TartanAirGaussianDataset (already filtered for
            held-out envs) wrapped via _TartanAirV6Wrapper, or None.
        hypersim: a HypersimDataset (already filtered for held-out
            scenes), or None.
        tartanair_ratio: weight for TartanAir source. v6 default 0.667.
        hypersim_ratio: weight for Hypersim source. v6 default 0.333.
        seed: deterministic seed for the per-index source assignment.
    """

    def __init__(
        self,
        tartanair: Dataset | None,
        hypersim: Dataset | None,
        tartanair_ratio: float = 0.667,
        hypersim_ratio: float = 0.333,
        seed: int = 0,
    ) -> None:
        if tartanair is None and hypersim is None:
            raise ValueError(
                "MixedTartanAirHypersimDataset requires at least one source."
            )
        self.tartanair = tartanair
        self.hypersim = hypersim
        if tartanair_ratio < 0 or hypersim_ratio < 0:
            raise ValueError("ratios must be non-negative")
        total = tartanair_ratio + hypersim_ratio
        if total <= 0:
            raise ValueError("at least one ratio must be > 0")
        self.tartanair_ratio = float(tartanair_ratio) / total
        self.hypersim_ratio = float(hypersim_ratio) / total
        self.seed = int(seed)

        n_t = len(tartanair) if tartanair is not None else 0
        n_h = len(hypersim) if hypersim is not None else 0
        self._n_t = n_t
        self._n_h = n_h
        self._length = n_t + n_h

        # Override-only ratios: if one source is empty, force the other to 1.0
        if self._n_t == 0:
            self.tartanair_ratio = 0.0
            self.hypersim_ratio = 1.0
        if self._n_h == 0:
            self.tartanair_ratio = 1.0
            self.hypersim_ratio = 0.0

    def __len__(self) -> int:
        return self._length

    def _pick_source(self, idx: int) -> str:
        # Deterministic per-index pick driven by a Mersenne Twister seeded
        # by (self.seed, idx). Same idx → same source across workers/epochs.
        # random.Random doesn't accept tuples; combine seed + idx into a
        # single int via a stable mix.
        mixed_seed = (self.seed * 0x9E3779B97F4A7C15 + idx) & 0xFFFFFFFFFFFFFFFF
        rng = random.Random(mixed_seed)
        u = rng.random()
        return "tartanair" if u < self.tartanair_ratio else "hypersim"

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        source = self._pick_source(idx)
        if source == "tartanair" and self.tartanair is not None:
            sub_idx = idx % max(self._n_t, 1)
            return self.tartanair[sub_idx]
        if self.hypersim is not None:
            sub_idx = idx % max(self._n_h, 1)
            return self.hypersim[sub_idx]
        # fallback: only one source available
        if self.tartanair is not None:
            return self.tartanair[idx % self._n_t]
        raise RuntimeError("no source available")


# ---------------------------------------------------------------------------
# Trajectory datasets
# ---------------------------------------------------------------------------


class TrajectoryDataset(Dataset):
    """Wrap a frame dataset and yield fixed-length consecutive windows.

    Windows never cross ``trajectory_key`` boundaries when the wrapped dataset
    exposes that method. The motion tensor at trajectory frame 0 is zero; for
    frame ``t > 0`` it is copied from source frame ``t - 1`` so callers receive
    motion from the previous frame into the current frame.
    """

    def __init__(self, base: Dataset, trajectory_length: int) -> None:
        if trajectory_length < 1:
            raise ValueError("trajectory_length must be >= 1")
        self.base = base
        self.trajectory_length = int(trajectory_length)
        self._starts: list[int] = []

        n = len(base)
        for start in range(max(0, n - self.trajectory_length + 1)):
            if self._window_stays_in_trajectory(start):
                self._starts.append(start)

        if n > 0 and not self._starts:
            raise ValueError(
                f"no length-{self.trajectory_length} trajectories available in "
                f"{type(base).__name__} with {n} frames"
            )

    def _trajectory_key(self, idx: int) -> str:
        if hasattr(self.base, "trajectory_key"):
            return str(self.base.trajectory_key(idx))  # type: ignore[attr-defined]
        return "default"

    def _window_stays_in_trajectory(self, start: int) -> bool:
        key = self._trajectory_key(start)
        end = start + self.trajectory_length
        return all(self._trajectory_key(i) == key for i in range(start + 1, end))

    def __len__(self) -> int:
        return len(self._starts)

    def trajectory_key(self, idx: int) -> str:
        return self._trajectory_key(self._starts[idx])

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        start = self._starts[idx]
        frames = [self.base[start + t] for t in range(self.trajectory_length)]

        motion_frames = [torch.zeros_like(frames[0]["motion"])]
        for t in range(1, self.trajectory_length):
            motion_frames.append(frames[t - 1]["motion"])

        return {
            "lr_frame": torch.stack([f["lr_frame"] for f in frames], dim=0),
            "gt_hr_frame": torch.stack([f["gt_hr_frame"] for f in frames], dim=0),
            "depth": torch.stack([f["depth"] for f in frames], dim=0),
            "normals": torch.stack([f["normals"] for f in frames], dim=0),
            "motion": torch.stack(motion_frames, dim=0),
            "canvas_hint": torch.stack([f["canvas_hint"] for f in frames], dim=0),
            "metadata": [dict(f.get("metadata", {})) for f in frames],
        }


class TrajectoryMixedDataset(MixedTartanAirHypersimDataset):
    """Mixed v6 dataset whose sampler index is a trajectory window."""

    def __init__(
        self,
        tartanair: Dataset | None,
        hypersim: Dataset | None,
        trajectory_length: int,
        tartanair_ratio: float = 0.667,
        hypersim_ratio: float = 0.333,
        seed: int = 0,
    ) -> None:
        self.trajectory_length = int(trajectory_length)
        tartanair_traj = (
            TrajectoryDataset(tartanair, self.trajectory_length)
            if tartanair is not None else None
        )
        hypersim_traj = (
            TrajectoryDataset(hypersim, self.trajectory_length)
            if hypersim is not None else None
        )
        super().__init__(
            tartanair=tartanair_traj,
            hypersim=hypersim_traj,
            tartanair_ratio=tartanair_ratio,
            hypersim_ratio=hypersim_ratio,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Convenience builder used by the training script
# ---------------------------------------------------------------------------


def build_v6_training_dataset(
    tartanair_root: Path | str | None,
    hypersim_root: Path | str | None,
    *,
    scale: float = 2.0,
    held_out_envs: Sequence[str] | None = None,
    held_out_scenes: Sequence[str] | None = None,
    tartanair_ratio: float = 0.667,
    hypersim_ratio: float = 0.333,
    lr_synth: EngineAliasedLRSynth | None = None,
    seed: int = 0,
    trajectory_length: int | None = None,
) -> MixedTartanAirHypersimDataset:
    """Build the v6 training dataset (60% TartanAir + 30% Hypersim).

    Either root can be None to skip that source. Held-out filters apply.
    """
    if lr_synth is None:
        lr_synth = EngineAliasedLRSynth(scale=scale)

    tartanair_wrapped: Dataset | None = None
    if tartanair_root is not None:
        ds_t = TartanAirGaussianDataset(
            root=tartanair_root, scale=scale, lr_synth=lr_synth,
        )
        tartanair_wrapped = _TartanAirV6Wrapper(ds_t, held_out_envs=held_out_envs)

    hypersim_ds: HypersimDataset | None = None
    if hypersim_root is not None:
        hypersim_ds = HypersimDataset(
            root=hypersim_root, scale=scale,
            lr_synth=lr_synth, held_out_scenes=held_out_scenes,
        )

    if trajectory_length is not None and trajectory_length > 1:
        return TrajectoryMixedDataset(
            tartanair=tartanair_wrapped,
            hypersim=hypersim_ds,
            trajectory_length=trajectory_length,
            tartanair_ratio=tartanair_ratio,
            hypersim_ratio=hypersim_ratio,
            seed=seed,
        )

    return MixedTartanAirHypersimDataset(
        tartanair=tartanair_wrapped,
        hypersim=hypersim_ds,
        tartanair_ratio=tartanair_ratio,
        hypersim_ratio=hypersim_ratio,
        seed=seed,
    )


__all__ = [
    "HypersimDataset",
    "MixedTartanAirHypersimDataset",
    "TrajectoryDataset",
    "TrajectoryMixedDataset",
    "build_v6_training_dataset",
]
