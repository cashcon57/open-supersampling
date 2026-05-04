"""Multi-frame trajectory window dataset for the v5 gaussian-temporal track.

Wraps any base dataset that exposes:
    - ``__len__()``
    - ``__getitem__(idx)`` -> mapping with per-frame fields (e.g. ``lr_frame``,
      ``depth``, ``motion``, ``normals``, ``canvas_hint``, ``gt_hr_frame``)
    - ``trajectory_key(idx)`` -> hashable identifier of the trajectory the
      frame belongs to. Windows only span equal trajectory keys.

Use ``oss.sr.temporal.dataset.adapt_tartanair`` / ``adapt_sintel`` to add the
``trajectory_key`` shim to TartanAir/Sintel base datasets — do **not**
re-implement those shims here.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Mapping

import torch
from torch.utils.data import Dataset


class TrajectoryWindowDataset(Dataset):
    """Emit a length-``window`` list of consecutive frames per ``__getitem__``.

    Windows that would cross a trajectory boundary are excluded.

    ``__getitem__(idx)`` returns::

        {
            "frames": [frame_0, frame_1, ..., frame_{window-1}],
            "trajectory_key": <hashable>,
        }

    where each ``frame_k`` is the raw mapping produced by ``base[i + k]``.
    """

    def __init__(self, base: Any, window: int = 5) -> None:
        if not hasattr(base, "trajectory_key"):
            raise TypeError(
                "Base dataset must expose `trajectory_key(idx) -> hashable`. "
                "Use oss.sr.temporal.dataset.adapt_tartanair / adapt_sintel "
                "to add it."
            )
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window}")
        self.base = base
        self.window = int(window)
        self._start_indices: List[int] = []
        n = len(base)
        for i in range(n):
            j = i + self.window - 1
            if j >= n:
                continue
            cur_key = base.trajectory_key(i)
            same_traj = all(
                base.trajectory_key(k) == cur_key for k in range(i + 1, j + 1)
            )
            if same_traj:
                self._start_indices.append(i)

    def __len__(self) -> int:
        return len(self._start_indices)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        start = self._start_indices[idx]
        frames = [self.base[start + k] for k in range(self.window)]
        return {
            "frames": frames,
            "trajectory_key": self.base.trajectory_key(start),
        }


def _stack(field: str, items: Iterable[Mapping[str, Any]]) -> torch.Tensor:
    return torch.stack([it[field] for it in items], dim=0)


def default_collate_window(
    samples: List[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Stack each frame field across the batch dimension.

    For a list of ``B`` samples each containing ``N = window`` frames, returns::

        {
            "frames": [
                {field: (B, ...) tensor for every field present in frame_0},
                ...  # length N
            ],
            "trajectory_key": [str, ...],  # length B
        }
    """
    if not samples:
        return {"frames": [], "trajectory_key": []}
    window = len(samples[0]["frames"])
    out_frames: List[Mapping[str, torch.Tensor]] = []
    for i in range(window):
        items = [s["frames"][i] for s in samples]
        # Use the first sample's keys as the canonical field set.
        fields = list(items[0].keys())
        out_frames.append({f: _stack(f, items) for f in fields})
    return {
        "frames": out_frames,
        "trajectory_key": [s["trajectory_key"] for s in samples],
    }


__all__ = [
    "TrajectoryWindowDataset",
    "default_collate_window",
]
