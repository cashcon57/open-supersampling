"""ORSDataset — load EXR triplets emitted by T1's render pipeline.

File layout produced by ``ors.render.mitsuba_pipeline.render_pair`` is::

    <scene>_v<view:04d>_<key>.exr

with keys ``noisy, ground_truth, albedo, normal, depth, motion``. We group
files by the ``<scene>_v<view>`` prefix and load all six buffers per sample.

Synthetic placeholders for v0.1 (T5):
- ``history``: ``ground_truth + 0.05 * randn`` — stand-in until v0.2 wires
  real temporal rollouts via SVGF-style reprojection.
- ``roughness`` / ``spec_hit_distance``: zeros — T1's AOV integrator does
  not extract these channels yet (Mitsuba 3.7 limitation noted in
  ``mitsuba_pipeline.py``). v0.2 will populate them.

The 11-channel ``aux`` tensor is laid out to match ``ORD.forward``'s contract:
``[albedo(3), normal(3), depth(1), roughness(1), spec_hit(1), motion(2)]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


_REQUIRED_KEYS = ("noisy", "ground_truth", "albedo", "normal", "depth", "motion")


def _load_exr(path: Path) -> np.ndarray:
    """Load an EXR as HWC float32. pyexr returns HWC by default."""
    import pyexr
    arr = pyexr.read(str(path))
    return np.asarray(arr, dtype=np.float32)


def _to_chw(arr: np.ndarray, expected_c: Optional[int] = None) -> torch.Tensor:
    """HWC numpy → CHW torch float32, with an explicit channel-count guard."""
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim != 3:
        raise ValueError(f"expected HWC array, got shape {arr.shape}")
    if expected_c is not None and arr.shape[-1] != expected_c:
        # Some EXR writers pad single-channel images to 3 channels; trim.
        if expected_c < arr.shape[-1]:
            arr = arr[..., :expected_c]
        else:
            raise ValueError(
                f"expected {expected_c} channels, got {arr.shape[-1]} for "
                f"array shape {arr.shape}"
            )
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float()


class ORSDataset(Dataset):
    """EXR-backed dataset matching ``ors.render.mitsuba_pipeline`` output."""

    def __init__(
        self,
        root: str | Path,
        augment: bool = False,
        crop_size: Optional[int] = None,
    ):
        self.root = Path(root)
        self.augment = augment
        self.crop_size = crop_size

        # Discover all sample prefixes by finding *_noisy.exr files.
        prefixes: list[str] = []
        for p in sorted(self.root.glob("*_noisy.exr")):
            stem = p.name[: -len("_noisy.exr")]
            # Verify all six required EXRs exist for this prefix; skip otherwise.
            if all((self.root / f"{stem}_{k}.exr").exists() for k in _REQUIRED_KEYS):
                prefixes.append(stem)
        self.prefixes = prefixes

    def __len__(self) -> int:
        return len(self.prefixes)

    def _load_sample(self, prefix: str) -> dict[str, torch.Tensor]:
        d = self.root
        noisy = _to_chw(_load_exr(d / f"{prefix}_noisy.exr"), expected_c=3)
        gt    = _to_chw(_load_exr(d / f"{prefix}_ground_truth.exr"), expected_c=3)
        albedo = _to_chw(_load_exr(d / f"{prefix}_albedo.exr"), expected_c=3)
        normal = _to_chw(_load_exr(d / f"{prefix}_normal.exr"), expected_c=3)
        depth  = _to_chw(_load_exr(d / f"{prefix}_depth.exr"), expected_c=1)
        motion = _to_chw(_load_exr(d / f"{prefix}_motion.exr"), expected_c=2)

        _, H, W = noisy.shape
        # Synthetic placeholders (see module docstring).
        history    = gt + 0.05 * torch.randn_like(gt)
        roughness  = torch.zeros(1, H, W, dtype=torch.float32)
        spec_hit   = torch.zeros(1, H, W, dtype=torch.float32)

        # 11-ch aux: albedo(3) + normal(3) + depth(1) + roughness(1) + spec_hit(1) + motion(2)
        aux = torch.cat([albedo, normal, depth, roughness, spec_hit, motion], dim=0)

        sample = {
            "noisy": noisy,
            "ground_truth": gt,
            "aux": aux,
            "history": history,
            "depth": depth,
            "motion": motion,
        }

        if self.crop_size is not None:
            sample = self._crop(sample, self.crop_size)
        if self.augment:
            sample = self._augment(sample)
        return sample

    @staticmethod
    def _crop(sample: dict[str, torch.Tensor], size: int) -> dict[str, torch.Tensor]:
        _, H, W = sample["noisy"].shape
        if H < size or W < size:
            return sample
        y = torch.randint(0, H - size + 1, (1,)).item()
        x = torch.randint(0, W - size + 1, (1,)).item()
        return {k: v[:, y:y + size, x:x + size] for k, v in sample.items()}

    @staticmethod
    def _augment(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Horizontal flip only — vertical flip + 90° rotations would invalidate
        # motion-vector signs without compensating remapping, which we skip in v0.1.
        if torch.rand(1).item() < 0.5:
            sample = {k: torch.flip(v, dims=(-1,)) for k, v in sample.items()}
        return sample

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._load_sample(self.prefixes[idx])
