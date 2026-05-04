"""Sequential frame-pair wrapper for the v5 pixel temporal track.

Wraps any base dataset that exposes:
    - __len__()
    - __getitem__(idx) -> mapping with keys
        lr_frame, depth, motion, normals, canvas_hint, gt_hr_frame
    - trajectory_key(idx) -> hashable identifier of the trajectory/sequence
      that frame ``idx`` belongs to. Pairs only span equal trajectory keys.

For TartanAir/Sintel datasets that don't expose ``trajectory_key`` directly,
the caller is expected to add a thin shim. See ``adapt_*`` helpers below.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Mapping

import torch
from torch.utils.data import Dataset


class SequentialPairDataset(Dataset):
    def __init__(self, base: Any) -> None:
        if not hasattr(base, "trajectory_key"):
            raise TypeError(
                "Base dataset must expose `trajectory_key(idx) -> hashable`. "
                "Use adapt_tartanair / adapt_sintel to add it."
            )
        self.base = base
        self._pair_indices: List[int] = []
        prev_key = None
        for i in range(len(base)):
            cur_key = base.trajectory_key(i)
            if i + 1 < len(base) and base.trajectory_key(i + 1) == cur_key:
                self._pair_indices.append(i)
            prev_key = cur_key

    def __len__(self) -> int:
        return len(self._pair_indices)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        i = self._pair_indices[idx]
        prev_key = self.base.trajectory_key(i - 1) if i > 0 else None
        cur_key = self.base.trajectory_key(i)
        is_first_in_seq = (prev_key != cur_key)
        return {
            "t": self.base[i],
            "t_plus_1": self.base[i + 1],
            "is_first_in_seq": bool(is_first_in_seq),
        }


def _stack(field: str, items: Iterable[Mapping[str, Any]]) -> torch.Tensor:
    return torch.stack([it[field] for it in items], dim=0)


def default_collate_pair(samples: List[Mapping[str, Any]]) -> Mapping[str, torch.Tensor]:
    t_items = [s["t"] for s in samples]
    p_items = [s["t_plus_1"] for s in samples]
    out: dict[str, torch.Tensor] = {}
    for prefix, items in (("t_", t_items), ("tp1_", p_items)):
        out[f"{prefix}lr"] = _stack("lr_frame", items)
        out[f"{prefix}depth"] = _stack("depth", items)
        out[f"{prefix}motion"] = _stack("motion", items)
        out[f"{prefix}normals"] = _stack("normals", items)
        out[f"{prefix}canvas"] = _stack("canvas_hint", items)
        out[f"{prefix}gt_hr"] = _stack("gt_hr_frame", items)
    out["is_first_in_seq"] = torch.tensor(
        [bool(s["is_first_in_seq"]) for s in samples], dtype=torch.bool
    )
    return out


# ---------------------------------------------------------------------------
# Trajectory-key shims for TartanAir / Sintel.
# ---------------------------------------------------------------------------


def adapt_tartanair(ds) -> Any:
    """Add ``trajectory_key`` to a TartanAirGaussianDataset.

    TartanAir's ``_items`` contains tuples of ``(image_path, depth_path,
    flow_path)``. The trajectory dir is the parent of ``image_left/``.
    """
    items = list(ds._items)

    def trajectory_key(idx: int) -> str:
        img_path = items[idx][0]
        # .../<env>/<level>/<traj>/image_left/000000_left.png
        return str(img_path.parent.parent)

    ds.trajectory_key = trajectory_key  # type: ignore[attr-defined]
    return ds


def adapt_sintel(ds) -> Any:
    """Add ``trajectory_key`` to SintelGaussianDataset (one key per sequence)."""
    items = list(ds._items)

    def trajectory_key(idx: int) -> str:
        img_path = items[idx][0]
        # .../training/clean/<seq>/frame_NNNN.png
        return str(img_path.parent)

    ds.trajectory_key = trajectory_key  # type: ignore[attr-defined]
    return ds


__all__ = [
    "SequentialPairDataset",
    "default_collate_pair",
    "adapt_tartanair",
    "adapt_sintel",
]
