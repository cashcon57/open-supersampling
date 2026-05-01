"""Common types for OSS-Gaussian Sprint 4 training data.

The training tuple consumed by ``GaussianParamNetwork`` (see
``oss/gaussian/network/param_net.py``) is a 12-channel concatenation:

    LR(3) + depth(1) + motion(2) + normals(3) + canvas_hint(3) = 12 ch

plus a ground-truth HR frame for the supervised loss.

Each individual loader (sintel/tartanair/hypersim/srgd) returns a
``GaussianTrainingExample``. The :class:`GaussianDataset` abstract base just
nails down the interface so that ``MixedGaussianDataset`` can stack them.

The classes intentionally accept tensors of any spatial size — the trainer is
responsible for crop / resize. This module performs only minimal shape
validation.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.utils.data import Dataset


# Channel layout (reference) — keep in sync with param_net.py's 12-ch contract.
LR_CHANNELS = 3
DEPTH_CHANNELS = 1
MOTION_CHANNELS = 2
NORMAL_CHANNELS = 3
CANVAS_CHANNELS = 3
TOTAL_INPUT_CHANNELS = (
    LR_CHANNELS + DEPTH_CHANNELS + MOTION_CHANNELS + NORMAL_CHANNELS + CANVAS_CHANNELS
)


@dataclass
class GaussianTrainingExample:
    """A single training tuple for the Gaussian param network.

    Spatial conventions:
        - ``lr_frame``, ``depth``, ``motion``, ``normals``, ``canvas_hint`` are at
          the same LR resolution ``(H, W)``.
        - ``gt_hr_frame`` is at HR resolution ``(H_hr, W_hr)``.
        - All tensors are ``torch.float32`` on CPU. Trainer is responsible for
          batching / device transfer.

    Channel layout matches the param_net 12-ch input when concatenated:
        cat([lr, depth, motion, normals, canvas_hint], dim=0) -> (12, H, W)
    """

    lr_frame: Tensor          # (3, H, W)
    depth: Tensor             # (1, H, W) — disparity-or-depth proxy in [0,1]
    motion: Tensor            # (2, H, W) — screen-space motion in pixels
    canvas_hint: Tensor       # (3, H, W) — warped previous prediction or zeros
    gt_hr_frame: Tensor       # (3, H_hr, W_hr)
    normals: Tensor | None = None  # (3, H, W) optional surface normals in [-1,1]
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- Validation --------------------------------------------------------
    def __post_init__(self) -> None:
        _check_chw(self.lr_frame, "lr_frame", LR_CHANNELS)
        _check_chw(self.depth, "depth", DEPTH_CHANNELS)
        _check_chw(self.motion, "motion", MOTION_CHANNELS)
        _check_chw(self.canvas_hint, "canvas_hint", CANVAS_CHANNELS)
        _check_chw(self.gt_hr_frame, "gt_hr_frame", LR_CHANNELS)
        if self.normals is not None:
            _check_chw(self.normals, "normals", NORMAL_CHANNELS)

        H, W = self.lr_frame.shape[-2:]
        for name, t in [
            ("depth", self.depth),
            ("motion", self.motion),
            ("canvas_hint", self.canvas_hint),
        ]:
            if t.shape[-2:] != (H, W):
                raise ValueError(
                    f"{name} spatial shape {tuple(t.shape[-2:])} != lr_frame {(H, W)}"
                )
        if self.normals is not None and self.normals.shape[-2:] != (H, W):
            raise ValueError(
                f"normals spatial shape {tuple(self.normals.shape[-2:])} != lr_frame {(H, W)}"
            )

    # ---- Helpers -----------------------------------------------------------
    def stack_input(self) -> Tensor:
        """Concatenate the 12-ch input tensor expected by GaussianParamNetwork."""
        normals = self.normals
        if normals is None:
            normals = torch.zeros(
                (NORMAL_CHANNELS, *self.lr_frame.shape[-2:]),
                dtype=self.lr_frame.dtype,
            )
        return torch.cat(
            [self.lr_frame, self.depth, self.motion, normals, self.canvas_hint], dim=0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a dict view (used by torch dataloader collate + tests)."""
        d = asdict(self)
        # asdict will deep-copy tensors via dataclasses; keep the originals.
        d["lr_frame"] = self.lr_frame
        d["depth"] = self.depth
        d["motion"] = self.motion
        d["canvas_hint"] = self.canvas_hint
        d["gt_hr_frame"] = self.gt_hr_frame
        d["normals"] = self.normals
        d["metadata"] = dict(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GaussianTrainingExample":
        return cls(
            lr_frame=d["lr_frame"],
            depth=d["depth"],
            motion=d["motion"],
            canvas_hint=d["canvas_hint"],
            gt_hr_frame=d["gt_hr_frame"],
            normals=d.get("normals"),
            metadata=dict(d.get("metadata", {})),
        )


def _check_chw(t: Tensor, name: str, expected_c: int) -> None:
    if not isinstance(t, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(t)}")
    if t.dim() != 3:
        raise ValueError(f"{name} must be (C,H,W); got shape {tuple(t.shape)}")
    if t.shape[0] != expected_c:
        raise ValueError(
            f"{name} channel count {t.shape[0]} != expected {expected_c}"
        )


class GaussianDataset(Dataset, abc.ABC):
    """Abstract base for OSS-Gaussian training datasets.

    Subclasses load from a real on-disk dataset. They MUST:
      - Accept ``root: Path`` and ``scale: float`` (HR/LR ratio).
      - Provide ``__len__`` and ``__getitem__ -> GaussianTrainingExample``.
      - Raise ``FileNotFoundError`` with a helpful message when ``root`` is
        missing in real (non-fixture) mode.

    The optional ``synthetic`` constructor flag enables a unit-test path that
    builds a tiny in-memory fake dataset matching the on-disk layout. Loaders
    use this in tests so we don't require the real ~100GB datasets to run
    pytest. Production code never sets ``synthetic=True``.
    """

    name: str = "gaussian-dataset"

    def __init__(self, root: Path | str, scale: float = 2.0) -> None:
        super().__init__()
        if scale < 1.0:
            raise ValueError(f"scale must be >=1.0; got {scale}")
        self.root = Path(root)
        self.scale = float(scale)

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> GaussianTrainingExample: ...

    # ---- Helpers shared across loaders -------------------------------------
    @staticmethod
    def _box_downsample(hr: Tensor, scale: float) -> Tensor:
        """Box-average downsample HR (C, H, W) -> LR. Scale must be int-ish.

        We use a fixed-int-divisor box average rather than F.interpolate so
        the test fixtures behave deterministically and integer-pixel align.
        """
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

    @staticmethod
    def _depth_to_normals(depth: Tensor) -> Tensor:
        """Cheap normal-from-depth: cross product of finite differences.

        ``depth`` is (1, H, W). Output is (3, H, W) in roughly [-1, 1].
        """
        if depth.dim() != 3 or depth.shape[0] != 1:
            raise ValueError(f"expected (1,H,W); got {tuple(depth.shape)}")
        d = depth[0]
        # central differences with edge replication
        dx = torch.zeros_like(d)
        dy = torch.zeros_like(d)
        dx[:, 1:-1] = d[:, 2:] - d[:, :-2]
        dy[1:-1, :] = d[2:, :] - d[:-2, :]
        # normal = normalize((-dx, -dy, 1))
        nz = torch.ones_like(d)
        n = torch.stack([-dx, -dy, nz], dim=0)
        norm = n.norm(dim=0, keepdim=True).clamp(min=1e-6)
        return n / norm

    @staticmethod
    def _zero_canvas(lr_frame: Tensor) -> Tensor:
        return torch.zeros((CANVAS_CHANNELS, *lr_frame.shape[-2:]), dtype=lr_frame.dtype)


def collate_examples(batch: list[GaussianTrainingExample]) -> dict[str, Tensor]:
    """Default collate for a batch of GaussianTrainingExample.

    Stacks each tensor field. ``normals`` is replaced with zeros when missing
    on a per-example basis so the batch is always well-defined. Metadata
    becomes a list (un-stacked).
    """
    if not batch:
        raise ValueError("collate_examples received empty batch")
    lr = torch.stack([e.lr_frame for e in batch], dim=0)
    depth = torch.stack([e.depth for e in batch], dim=0)
    motion = torch.stack([e.motion for e in batch], dim=0)
    canvas = torch.stack([e.canvas_hint for e in batch], dim=0)
    gt_hr = torch.stack([e.gt_hr_frame for e in batch], dim=0)
    normals_list = []
    for e in batch:
        if e.normals is None:
            normals_list.append(torch.zeros((NORMAL_CHANNELS, *e.lr_frame.shape[-2:])))
        else:
            normals_list.append(e.normals)
    normals = torch.stack(normals_list, dim=0)
    return {
        "lr_frame": lr,
        "depth": depth,
        "motion": motion,
        "canvas_hint": canvas,
        "normals": normals,
        "gt_hr_frame": gt_hr,
        "metadata": [e.metadata for e in batch],
    }


__all__ = [
    "GaussianTrainingExample",
    "GaussianDataset",
    "collate_examples",
    "LR_CHANNELS",
    "DEPTH_CHANNELS",
    "MOTION_CHANNELS",
    "NORMAL_CHANNELS",
    "CANVAS_CHANNELS",
    "TOTAL_INPUT_CHANNELS",
]
