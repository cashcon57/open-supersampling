"""Parent-child deferred-materialization spawner for v7.

Adapted from Diolatzis et al. 2024 (https://arxiv.org/abs/2405.20067).
Every active Gaussian carries a dormant child whose parameters are
expressed in the parent's reference frame. During training each
child's opacity/brightness drift; when either crosses a fixed
threshold the child "materializes" into a full top-level Gaussian
added to the canvas.

Thresholds (paper §3.3):
  opacity > 0.1     OR    brightness > 0.01

The mechanism is loss-adaptive density control without explicit
splitting heuristics -- capacity grows where the optimizer pushed it
through the child's parameters, not via per-frame disocclusion
masks.

Used by v7 model in place of v6 DisocclusionSpawner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from oss.sr.v7.nd_canvas_state import NDCanvasState


OPACITY_MATERIALIZE = 0.1
BRIGHTNESS_MATERIALIZE = 0.01


@dataclass
class ChildState:
    """Per-parent dormant child state. Allocated alongside the canvas;
    one child per active parent.

    All tensors are (capacity, ...) with the same per-Gaussian indexing
    as the parent NDCanvasState.
    """
    dpos: torch.Tensor       # (capacity, 3)  child position offset from parent
    dcov_raw: torch.Tensor   # (capacity, 6)  child cov raw offset
    dfeat: torch.Tensor      # (capacity, R)  child feature offset
    opacity: torch.Tensor    # (capacity,)    child opacity (independent)
    brightness: torch.Tensor # (capacity,)    proxy for visual prominence

    @classmethod
    def empty(cls, capacity: int, feature_dim: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> "ChildState":
        d = torch.device(device)
        return cls(
            dpos=torch.zeros((capacity, 3), device=d, dtype=dtype),
            dcov_raw=torch.zeros((capacity, 6), device=d, dtype=dtype),
            dfeat=torch.zeros((capacity, feature_dim), device=d, dtype=dtype),
            opacity=torch.full((capacity,), 1e-6, device=d, dtype=dtype),
            brightness=torch.full((capacity,), 1e-6, device=d, dtype=dtype),
        )

    def reset(self) -> "ChildState":
        self.dpos.zero_()
        self.dcov_raw.zero_()
        self.dfeat.zero_()
        self.opacity.fill_(1e-6)
        self.brightness.fill_(1e-6)
        return self


def materialize_mask(child: ChildState, n_live: int) -> torch.Tensor:
    """Returns (n_live,) bool mask of children ready to be promoted.

    n_live is the parent canvas's live count; only that prefix of the
    child arrays is considered.
    """
    op_pass = child.opacity[:n_live] > OPACITY_MATERIALIZE
    br_pass = child.brightness[:n_live] > BRIGHTNESS_MATERIALIZE
    return op_pass | br_pass


def materialize_to_canvas(
    canvas: NDCanvasState,
    child: ChildState,
) -> int:
    """For each parent whose child crosses threshold, append a new
    Gaussian to the canvas (parent + child offsets), and reset the
    materialized child slot to dormant.

    Returns the number of children materialized this round.
    """
    mask = materialize_mask(child, canvas.n_live)
    if not mask.any():
        return 0
    parent_idx = mask.nonzero(as_tuple=True)[0]

    # New Gaussian = parent + child offset
    new_positions = canvas.positions[parent_idx] + child.dpos[parent_idx]
    new_cov_raw = canvas.cov_raw[parent_idx] + child.dcov_raw[parent_idx]
    new_features = canvas.features[parent_idx] + child.dfeat[parent_idx]
    new_opacity = child.opacity[parent_idx]    # take child's opacity as the new Gaussian's

    canvas.add(
        positions=new_positions,
        cov_raw=new_cov_raw,
        features=new_features,
        opacity=new_opacity,
    )
    # Reset the materialized slots back to dormant (a fresh child will
    # be initialized on the next spawn-step for these now-active parents).
    child.dpos[parent_idx] = 0.0
    child.dcov_raw[parent_idx] = 0.0
    child.dfeat[parent_idx] = 0.0
    child.opacity[parent_idx] = 1e-6
    child.brightness[parent_idx] = 1e-6
    return int(mask.sum().item())


def initialize_children_for_new_parents(
    child: ChildState,
    parent_indices: torch.Tensor,
    init_dpos_std: float = 0.1,
    init_dfeat_std: float = 1e-3,
) -> None:
    """Allocate fresh dormant children for the given parent indices.
    Called periodically (every ~300 training steps) for all currently-
    live parents. Child parameters are NOT zero -- they get a tiny
    random offset to break symmetry so each child has a unique
    optimization trajectory."""
    n = parent_indices.shape[0]
    if n == 0:
        return
    g = torch.Generator(device=child.dpos.device).manual_seed(int(parent_indices.sum().item()))
    child.dpos[parent_indices] = (
        torch.randn((n, 3), generator=g, device=child.dpos.device,
                    dtype=child.dpos.dtype) * init_dpos_std
    )
    child.dcov_raw[parent_indices] = 0.0   # cov starts identical to parent
    child.dfeat[parent_indices] = (
        torch.randn((n, child.dfeat.shape[-1]), generator=g, device=child.dfeat.device,
                    dtype=child.dfeat.dtype) * init_dfeat_std
    )
    child.opacity[parent_indices] = 1e-6
    child.brightness[parent_indices] = 1e-6
