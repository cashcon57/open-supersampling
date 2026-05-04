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
    """Wraps a base dataset to emit consecutive frame pairs ``(t, t+pair_stride)``.

    ``pair_stride`` controls the frame gap between paired samples:
    - ``pair_stride=1`` (default): adjacent frames ``(i, i+1)``
    - ``pair_stride=k``: frames ``(i, i+k)`` from the same trajectory; frames
      whose trajectory key changes within ``[i, i+k]`` are excluded

    The default of 1 matches the spec; larger strides expose the model to
    longer-displacement motion during training when the dataset's flow is
    accumulated forward.
    """

    def __init__(self, base: Any, pair_stride: int = 1) -> None:
        if not hasattr(base, "trajectory_key"):
            raise TypeError(
                "Base dataset must expose `trajectory_key(idx) -> hashable`. "
                "Use adapt_tartanair / adapt_sintel to add it."
            )
        if pair_stride < 1:
            raise ValueError(f"pair_stride must be >= 1; got {pair_stride}")
        self.base = base
        self.pair_stride = int(pair_stride)
        self._pair_indices: List[int] = []
        for i in range(len(base)):
            j = i + self.pair_stride
            if j >= len(base):
                continue
            # All intermediate frames must share the same trajectory key.
            cur_key = base.trajectory_key(i)
            same_traj = all(
                base.trajectory_key(k) == cur_key for k in range(i + 1, j + 1)
            )
            if same_traj:
                self._pair_indices.append(i)

    def __len__(self) -> int:
        return len(self._pair_indices)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        i = self._pair_indices[idx]
        prev_key = self.base.trajectory_key(i - 1) if i > 0 else None
        cur_key = self.base.trajectory_key(i)
        is_first_in_seq = (prev_key != cur_key)
        return {
            "t": self.base[i],
            "t_plus_1": self.base[i + self.pair_stride],
            "is_first_in_seq": bool(is_first_in_seq),
        }


def _field(item: Any, field: str) -> Any:
    if isinstance(item, Mapping):
        return item[field]
    return getattr(item, field)


def _stack(field: str, items: Iterable[Any]) -> torch.Tensor:
    vals: list[torch.Tensor] = []
    for item in items:
        val = _field(item, field)
        if val is None and field == "normals":
            lr = _field(item, "lr_frame")
            val = torch.zeros((3, *lr.shape[-2:]), dtype=lr.dtype, device=lr.device)
        vals.append(val)
    return torch.stack(vals, dim=0)


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


class _TartanairTrajectoryKey:
    """Top-level callable so DataLoader workers can serialize it.

    Closures over local variables cannot be transported to spawn-based
    DataLoader workers on Windows. A bound method on a top-level class can
    be transported; the dataset reference is captured as an attribute.
    """

    __slots__ = ("ds",)

    def __init__(self, ds: Any) -> None:
        self.ds = ds

    def __call__(self, idx: int) -> str:
        # .../<env>/<level>/<traj>/image_left/000000_left.png
        return str(self.ds._items[idx][0].parent.parent)


class _SintelTrajectoryKey:
    """Top-level callable for Sintel; same worker-transport rationale."""

    __slots__ = ("ds",)

    def __init__(self, ds: Any) -> None:
        self.ds = ds

    def __call__(self, idx: int) -> str:
        # .../training/clean/<seq>/frame_NNNN.png
        return str(self.ds._items[idx][0].parent)


def adapt_tartanair(ds) -> Any:
    """Add ``trajectory_key`` to a TartanAirGaussianDataset.

    TartanAir's ``_items`` contains tuples of ``(image_path, depth_path,
    flow_path)``. The trajectory dir is the parent of ``image_left/``.
    """
    ds.trajectory_key = _TartanairTrajectoryKey(ds)  # type: ignore[attr-defined]
    return ds


def adapt_sintel(ds) -> Any:
    """Add ``trajectory_key`` to SintelGaussianDataset (one key per sequence)."""
    ds.trajectory_key = _SintelTrajectoryKey(ds)  # type: ignore[attr-defined]
    return ds


__all__ = [
    "SequentialPairDataset",
    "default_collate_pair",
    "adapt_tartanair",
    "adapt_sintel",
]
