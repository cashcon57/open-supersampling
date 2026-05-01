"""Prune + spawn policy and mechanism — Sprint 5 / T5.4 + T5.5.

Policy (which Gaussians retire) is deliberately separated from mechanism
(which slots they free, where the replacements come from). The split
exists so the prune rules can be tuned independently of the buffer-
management code.

Public surface:

- ``PrunePolicy`` — tuneable rule thresholds.
- ``select_for_pruning(...) → indices`` — pure function. No state writes.
- ``select_spawn_tiles(...) → tile (y, x) coords`` — pick which high-error
  tiles get fresh Gaussians.
- ``apply_prune_spawn(state, prune_idx, new_gaussians)`` — the only place
  alive/positions/scales/etc. mutate.

Design doc: ``docs/superpowers/gaussian-canvas-design.md`` §3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover — avoid circular import at runtime
    from oss.gaussian.canvas.canvas import PersistentCanvas
    from oss.gaussian.renderer import GaussianBatch


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrunePolicy:
    """Tuneable thresholds for the prune decision tree.

    All thresholds are picked deliberately conservative for Sprint 5 v1.
    Sprint 5 ablation tasks (post-merge) revisit the values once we have
    real Cyberpunk frame data.
    """

    age_max: int = 60
    """Above this many frames a Gaussian is eligible for the
    ``aged + low-contribution`` rule (R2)."""

    age_low_error_pct: float = 0.75
    """A Gaussian is "low contribution" if its error is below this
    quantile of the alive distribution (default 75th percentile = the
    bottom three quarters of error)."""

    tile_error_pct: float = 0.95
    """Tiles whose error is above this quantile fire rule R3 — Gaussians
    in them are candidates for replacement."""

    min_age_before_prune: int = 3
    """Gaussians younger than this never get pruned by R3 (prevents
    spawn↔prune oscillation)."""

    max_prune_per_frame_frac: float = 0.05
    """Cap on prunes per frame as a fraction of canvas capacity (5%)."""


# ---------------------------------------------------------------------------
# Pruning selection
# ---------------------------------------------------------------------------


def select_for_pruning(
    alive_mask: torch.Tensor,
    in_frame: torch.Tensor,
    age: torch.Tensor,
    g_error: torch.Tensor,
    tile_error: torch.Tensor,
    capacity: int,
    policy: Optional[PrunePolicy] = None,
) -> torch.Tensor:
    """Return long-tensor indices of Gaussians to retire this frame.

    Decision tree (rule R1 first, R3 last):

    R1. ``in_frame == False``                           → prune.
    R2. ``age > age_max`` AND ``g_error <= age-quantile`` → prune.
    R3. ``age >= min_age_before_prune`` AND
        ``g_error >= tile_error_quantile``                → prune.

    Total prunes are then clamped to
    ``floor(max_prune_per_frame_frac * capacity)``.

    Already-dead slots are never selected.
    """
    p = policy if policy is not None else PrunePolicy()

    n = alive_mask.shape[0]
    device = alive_mask.device
    if n == 0:
        return torch.zeros((0,), dtype=torch.long, device=device)

    selected = torch.zeros(n, dtype=torch.bool, device=device)

    # R1 — out-of-frame & alive.
    r1 = alive_mask & (~in_frame)
    selected = selected | r1

    # R2 — aged dead-weight.
    alive_err = g_error[alive_mask]
    if alive_err.numel() > 0:
        # ``quantile`` requires float and 1-D input; we already have that.
        age_q = torch.quantile(
            alive_err.float(),
            float(p.age_low_error_pct),
        )
    else:
        age_q = torch.tensor(0.0, device=device)
    r2 = alive_mask & (age > p.age_max) & (g_error <= age_q)
    selected = selected | r2

    # R3 — high tile-error replacement.
    if alive_err.numel() > 0:
        tile_q = torch.quantile(
            alive_err.float(),
            float(p.tile_error_pct),
        )
    else:
        tile_q = torch.tensor(float("inf"), device=device)
    r3 = (
        alive_mask
        & (age >= p.min_age_before_prune)
        & (g_error >= tile_q)
    )
    selected = selected | r3

    idx = selected.nonzero(as_tuple=False).flatten()

    # Cap the total. Prefer R1 hits (out-of-frame) first, then R2, then R3.
    cap = max(1, int(p.max_prune_per_frame_frac * capacity))
    if idx.numel() > cap:
        # Sort by descending priority: R1 > R2 > R3, then by descending error.
        priority = torch.zeros(n, dtype=torch.float32, device=device)
        priority[r3] = 1.0
        priority[r2] = 2.0
        priority[r1] = 3.0
        prio_for_idx = priority[idx]
        # Tie-break by error (high error wins inside same priority).
        # Replace +inf to keep argsort numerically stable.
        err_clipped = torch.where(
            torch.isfinite(g_error[idx]),
            g_error[idx],
            torch.full_like(g_error[idx], 1e30),
        )
        score = prio_for_idx * 1e6 + err_clipped
        order = torch.argsort(score, descending=True)
        idx = idx[order[:cap]]
    return idx


# ---------------------------------------------------------------------------
# Spawn selection (which tiles deserve fresh Gaussians)
# ---------------------------------------------------------------------------


def select_spawn_tiles(
    tile_error: torch.Tensor,
    n_tiles: int,
    classifier_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pick the top ``n_tiles`` highest-error tiles, restricted to tiles
    the classifier marked complex.

    Args:
        tile_error:      ``(h, w)`` per-tile error.
        n_tiles:         How many tiles to select. If ``classifier_mask``
                         narrows the candidate pool below ``n_tiles``,
                         returns however many are available.
        classifier_mask: ``(h, w)`` bool. Only ``True`` tiles eligible.
                         If ``None``, every tile is eligible.

    Returns:
        ``(M, 2)`` long tensor of (y, x) tile coordinates.
    """
    if tile_error.ndim != 2:
        raise ValueError(f"tile_error must be (h, w); got {tuple(tile_error.shape)}")
    h, w = tile_error.shape
    device = tile_error.device

    if classifier_mask is not None:
        if classifier_mask.shape != tile_error.shape:
            raise ValueError(
                f"classifier_mask {classifier_mask.shape} != tile_error {tile_error.shape}"
            )
        masked = torch.where(
            classifier_mask,
            tile_error,
            torch.full_like(tile_error, -float("inf")),
        )
    else:
        masked = tile_error

    flat = masked.flatten()
    eligible = flat.isfinite() & (flat > -float("inf"))
    n_eligible = int(eligible.sum().item())
    take = min(int(n_tiles), n_eligible)
    if take <= 0:
        return torch.zeros((0, 2), dtype=torch.long, device=device)

    _, idx = torch.topk(flat, take)
    ty = idx // w
    tx = idx % w
    return torch.stack([ty, tx], dim=-1)


# ---------------------------------------------------------------------------
# Apply (the only state-mutating function)
# ---------------------------------------------------------------------------


def apply_prune_spawn(
    canvas: "PersistentCanvas",
    prune_idx: torch.Tensor,
    new_gaussians: Optional["GaussianBatch"] = None,
) -> None:
    """Mark ``prune_idx`` slots dead, then write ``new_gaussians`` into
    free slots in order. Mutates ``canvas`` in place.

    ``new_gaussians`` may contain more Gaussians than ``prune_idx`` freed
    — only the first M (= number of free slots after prune) are written;
    the rest are dropped on the floor for this frame. Conversely, if
    fewer new Gaussians arrive than slots freed, the extra slots stay
    dead (alive count drops below capacity, by design).
    """
    # 1. Mark prune indices dead and reset their per-Gaussian state.
    if prune_idx.numel() > 0:
        canvas.alive[prune_idx] = False
        canvas.age[prune_idx] = 0
        canvas.error[prune_idx] = 0.0

    if new_gaussians is None:
        return

    # 2. Write new Gaussians into the first dead slots.
    free_idx = (~canvas.alive).nonzero(as_tuple=False).flatten()
    m = min(free_idx.numel(), new_gaussians.num_gaussians)
    if m == 0:
        return
    write = free_idx[:m]
    canvas.positions[write] = new_gaussians.xy[:m].to(canvas.positions.dtype)
    canvas.scales[write] = new_gaussians.scale[:m].to(canvas.scales.dtype)
    canvas.rotations[write] = new_gaussians.rot[:m].to(canvas.rotations.dtype)
    f = min(canvas.colors.shape[1], new_gaussians.feat.shape[1])
    canvas.colors[write, :f] = new_gaussians.feat[:m, :f].to(canvas.colors.dtype)
    canvas.alive[write] = True
    canvas.age[write] = 0
    canvas.error[write] = 0.0


__all__ = [
    "PrunePolicy",
    "select_for_pruning",
    "select_spawn_tiles",
    "apply_prune_spawn",
]
