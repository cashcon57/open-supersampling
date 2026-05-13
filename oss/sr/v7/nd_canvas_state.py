"""N-D Gaussian canvas state for v7.

Each Gaussian carries:
  position  (x, y, t)        shape (3,)
  cov_raw   Cholesky factor   shape (6,)  -- L00, L10, L11, L20, L21, L22
                                            (lower-triangular packed)
  features                    shape (R,)
  opacity                     scalar

Implementation choice: store Cholesky params, not full V. PSD is then
guaranteed by construction (assuming positive diagonals). The full
covariance V = L L^T is computed on demand for the rasterizer.

Cap and live-pointer semantics match v6 CanvasState:
  - capacity is fixed; appending past capacity rolls or errors
  - `n_live` is the index of the last-spawned Gaussian + 1
  - `mask` (per-Gaussian active bool) allows pruning without compaction
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def cholesky_pack_to_L(raw: torch.Tensor) -> torch.Tensor:
    """raw: (..., 6)  ->  L: (..., 3, 3) lower-triangular Cholesky factor.

    Layout: raw = (l00, l10, l11, l20, l21, l22).
    Diagonal entries are forced positive via exp(), as in the paper.
    """
    L = torch.zeros(raw.shape[:-1] + (3, 3), device=raw.device, dtype=raw.dtype)
    L[..., 0, 0] = raw[..., 0].exp()
    L[..., 1, 0] = raw[..., 1]                  # unconstrained off-diag
    L[..., 1, 1] = raw[..., 2].exp()
    L[..., 2, 0] = raw[..., 3]                  # unconstrained off-diag
    L[..., 2, 1] = raw[..., 4]                  # unconstrained off-diag
    L[..., 2, 2] = raw[..., 5].exp()
    return L


def cholesky_pack_to_cov(raw: torch.Tensor) -> torch.Tensor:
    """raw: (..., 6)  ->  V: (..., 3, 3) full 3D covariance V = L L^T."""
    L = cholesky_pack_to_L(raw)
    return L @ L.transpose(-1, -2)


@dataclass
class NDCanvasState:
    """N-D Gaussian canvas state, persisted across frames in v7.

    All tensors live on a single device; the canvas is per-rank.
    Tensor shapes use a capacity-bounded pool with an explicit live mask.

    positions   (capacity, 3)   (x, y, t)
    cov_raw     (capacity, 6)   packed Cholesky factor
    features    (capacity, R)
    opacity     (capacity,)
    mask        (capacity,)     bool, True if Gaussian is live
    n_live      int             one past the largest occupied index
    """
    positions: torch.Tensor
    cov_raw: torch.Tensor
    features: torch.Tensor
    opacity: torch.Tensor
    mask: torch.Tensor
    n_live: int = 0

    @property
    def capacity(self) -> int:
        return int(self.positions.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.positions.device

    @property
    def count(self) -> int:
        """Number of currently-active Gaussians."""
        if self.n_live == 0:
            return 0
        return int(self.mask[: self.n_live].sum().item())

    @classmethod
    def empty(cls, capacity: int, feature_dim: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> "NDCanvasState":
        """Allocate an empty canvas with given capacity / feature dim."""
        d = torch.device(device)
        return cls(
            positions=torch.zeros((capacity, 3), device=d, dtype=dtype),
            cov_raw=torch.zeros((capacity, 6), device=d, dtype=dtype),
            features=torch.zeros((capacity, feature_dim), device=d, dtype=dtype),
            opacity=torch.zeros((capacity,), device=d, dtype=dtype),
            mask=torch.zeros((capacity,), device=d, dtype=torch.bool),
            n_live=0,
        )

    def add(
        self,
        positions: torch.Tensor,   # (K, 3)
        cov_raw: torch.Tensor,     # (K, 6)
        features: torch.Tensor,    # (K, R)
        opacity: torch.Tensor,     # (K,)
    ) -> "NDCanvasState":
        """Append K Gaussians to the canvas. Raises if exceeds capacity.

        Returns self for chaining; modifies in place.
        """
        k = int(positions.shape[0])
        if k == 0:
            return self
        if self.n_live + k > self.capacity:
            raise ValueError(
                f"NDCanvasState capacity {self.capacity} exceeded "
                f"(have {self.n_live} live + {k} new); enlarge capacity "
                f"or prune first"
            )
        start = self.n_live
        end = start + k
        self.positions[start:end] = positions.to(device=self.device, dtype=self.positions.dtype)
        self.cov_raw[start:end] = cov_raw.to(device=self.device, dtype=self.cov_raw.dtype)
        self.features[start:end] = features.to(device=self.device, dtype=self.features.dtype)
        self.opacity[start:end] = opacity.to(device=self.device, dtype=self.opacity.dtype)
        self.mask[start:end] = True
        self.n_live = end
        return self

    def prune(self, keep_mask: torch.Tensor) -> "NDCanvasState":
        """Apply a (n_live,) bool mask; Gaussians where keep_mask=False
        become dormant. Capacity is preserved; n_live is unchanged."""
        if keep_mask.shape[0] != self.n_live:
            raise ValueError(
                f"keep_mask must be (n_live={self.n_live},); got {keep_mask.shape}"
            )
        self.mask[: self.n_live] = keep_mask.to(device=self.device, dtype=torch.bool)
        return self

    def prune_to_count(self, target_count: int, strategy: str = "lowest_opacity") -> int:
        """Reduce active Gaussian count to <= target_count by dropping
        the lowest-opacity (or oldest, depending on strategy) actives.

        Returns the number of Gaussians dropped.
        """
        if target_count < 0:
            raise ValueError("target_count must be >= 0")
        n_active = self.count
        if n_active <= target_count:
            return 0
        n_to_drop = n_active - target_count

        live_mask = self.mask[: self.n_live]
        live_indices = live_mask.nonzero(as_tuple=True)[0]
        if strategy == "lowest_opacity":
            opacities = self.opacity[live_indices]
            # Take the n_to_drop smallest opacities
            _, smallest_idx = torch.topk(opacities, k=n_to_drop, largest=False)
            drop_idx_in_canvas = live_indices[smallest_idx]
        elif strategy == "oldest":
            # Oldest = earliest index (canvas is append-only). Take
            # the first n_to_drop live indices.
            drop_idx_in_canvas = live_indices[:n_to_drop]
        else:
            raise ValueError(f"unknown prune strategy {strategy!r}")
        self.mask[drop_idx_in_canvas] = False
        return int(n_to_drop)

    def compact(self) -> int:
        """Shift live Gaussians toward the front of the pool, dropping
        dormant ones. Returns the new n_live. Useful periodically to
        reclaim capacity headroom after pruning."""
        live_mask = self.mask[: self.n_live]
        if live_mask.all().item():
            return self.n_live
        live_idx = live_mask.nonzero(as_tuple=True)[0]
        n_new = int(live_idx.shape[0])
        # Move live entries to the front
        self.positions[:n_new] = self.positions[: self.n_live][live_idx]
        self.cov_raw[:n_new] = self.cov_raw[: self.n_live][live_idx]
        self.features[:n_new] = self.features[: self.n_live][live_idx]
        self.opacity[:n_new] = self.opacity[: self.n_live][live_idx]
        # Reset trailing entries
        if n_new < self.n_live:
            self.positions[n_new : self.n_live] = 0.0
            self.cov_raw[n_new : self.n_live] = 0.0
            self.features[n_new : self.n_live] = 0.0
            self.opacity[n_new : self.n_live] = 0.0
        self.mask.zero_()
        self.mask[:n_new] = True
        self.n_live = n_new
        return n_new

    def active_view(self):
        """Return (positions, cov, features, opacity) sliced to active
        Gaussians only -- ready to feed the rasterizer."""
        live_mask = self.mask[: self.n_live]
        idx = live_mask.nonzero(as_tuple=True)[0]
        positions = self.positions[: self.n_live][idx]
        cov_raw = self.cov_raw[: self.n_live][idx]
        features = self.features[: self.n_live][idx]
        opacity = self.opacity[: self.n_live][idx]
        cov = cholesky_pack_to_cov(cov_raw)   # (K, 3, 3)
        return positions, cov, features, opacity

    def detach_state(self) -> "NDCanvasState":
        """Detach the canvas tensors from any autograd graph by re-pointing
        them at fresh storage carrying the same values.

        This is required between training-step / per-sample boundaries
        because `add()` does in-place index assignment (`positions[a:b] = ...`),
        which increments the underlying storage's version counter. Without
        a fresh-storage reset, sample B's `add()` corrupts the version of
        the storage referenced by sample A's autograd CopySlices node, and
        backward fails with either "modified by inplace operation" or
        "backward through the graph a second time".

        `.detach().clone()` returns a new tensor with the same values but
        a fresh storage + no grad_fn, so the prior graph's tensor reference
        keeps its version stable and the new canvas can be mutated freely.
        """
        self.positions = self.positions.detach().clone()
        self.cov_raw = self.cov_raw.detach().clone()
        self.features = self.features.detach().clone()
        self.opacity = self.opacity.detach().clone()
        return self

    def reset(self) -> "NDCanvasState":
        """Clear all Gaussians; capacity preserved. Also detaches storage so
        the next `add()` cannot collide with the prior graph's version
        bookkeeping (see `detach_state`)."""
        self.detach_state()
        self.mask.zero_()
        self.n_live = 0
        return self
