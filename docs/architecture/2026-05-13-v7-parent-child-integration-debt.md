# v7 parent-child spawner — integration debt + design call

**Date:** 2026-05-13
**Status:** **LANDED** (commits 2026-05-13). Originally tagged as deferred; the user directed it be addressed since it's a definite future requirement, not a maybe. Mechanism is now wired through `V7Model` + `scripts/sr_train_v7.py` behind the `--enable-parent-child` flag (off by default).

This memo retains the original design discussion as historical context. See § "What landed" at the end for the current state.

## Why it matters

Per `docs/capture/v7-training-data-spec.md` discussion of fine-Gaussian density. The base uniform per-tile spawner places ~1 Gaussian per 64 HR pixels, which cannot represent sub-pixel features (thin wires, distant fences, hair). Parent-child is the existing-but-unused lever that lets density grow adaptively where SR loss is high.

## Original implementation (predates wiring)

`oss/sr/v7/parent_child_spawner.py` implements the Diolatzis-2024 deferred-materialization pattern:

```python
ChildState(dpos, dcov_raw, dfeat, opacity, brightness)  # capacity, capacity, ...
materialize_mask(child, n_live) -> bool[n_live]
materialize_to_canvas(canvas, child) -> int                # appends to canvas
initialize_children_for_new_parents(child, parent_indices)
```

Materialization triggers when `child.opacity > 0.1` OR `child.brightness > 0.01`. Once triggered, a new top-level Gaussian gets appended to the canvas at (parent_pos + child.dpos) with parent_cov + child.dcov_raw and the child slot is reset to dormant.

## Design call — answered

The Diolatzis paper drives child.opacity drift via **per-parent loss residual**, accumulated over many steps. It is **not** a gradient-through-Parameters mechanism. Three candidate signals were on the table:

- Per-parent loss splat via the rasterizer's alpha-blend weights (cleanest but custom backward hook)
- Per-parent position-grad-norm via lifting `canvas.positions` to a Parameter (interacts with autograd-isolation fix)
- **Per-parent rendered-position gradient via `retain_grad` on `active_view().positions`** ← picked

The third option is the least invasive: the spawner already produces `positions` with `requires_grad=True` (it's a function of backbone features), and `active_view()` slices those positions into a tensor that the rasterizer consumes. By calling `retain_grad()` on that slice during `render_canvas`, we make autograd populate `.grad` after the user calls `loss.backward()`. The grad norm per parent = "how much would moving this parent reduce loss." High norm = parent is in a high-loss region = its child should drift up.

## What landed

### `oss/sr/v7/model.py`

- `V7Config.enable_parent_child` (default `False`), `parent_child_drift_rate` (default 0.05), `parent_child_decay` (default 0.98).
- `V7Model._child: ChildState` allocated alongside the canvas when `enable_parent_child=True`. Persists across `reset_state` semantics:
  - `reset_state()` clears child too (trajectory boundary).
  - Inter-spawn (within a single sample): child accumulates.
- `V7Model.initialize_new_children(init_dpos_std)`: idempotent. Called after each spawn to attach children to any newly-live parents. Tracks `_n_children_initialized` so it doesn't re-init existing children.
- `V7Model.drift_children_from_grad()`: read `.grad` off the retained-grad active_view positions tensor; normalize per-parent; decay then drift child opacity/brightness on live indices. Returns per-parent grad norms for diagnostics.
- `V7Model.materialize_pending_children()`: thin wrapper around `materialize_to_canvas()` under `@torch.no_grad()`.

### `scripts/sr_train_v7.py`

- `--enable-parent-child` flag (off by default).
- `--parent-child-drift-rate` (default 0.05) and `--parent-child-decay` (default 0.98) CLI knobs.
- Inner loop calls `initialize_new_children` after each spawn, then `drift_children_from_grad` + `materialize_pending_children` between `backward()` and `optim.step()`.
- The per-batch `materialized` count gets recorded in `history.jsonl` for dashboard interpretation.

### `tests/sr/v7/test_parent_child_integration.py`

9 tests covering:

- ChildState allocation gated on the flag (both on and off paths)
- `reset_state` clears children
- `initialize_new_children` is idempotent and tracks the new-parent count
- Single drift step pushes the highest-gradient child past the materialization threshold
- Drift signal is rank-preserving wrt per-parent gradient norm (largest grad → largest opacity)
- Materialization fires when threshold is crossed and produces real canvas Gaussians
- Full backward → drift → materialize → optim.step loop survives the autograd-isolation fix
- Disabled path is zero-overhead (no allocations, no method invocations succeed)

111/111 v7 tests pass.

## Known limitations of the current wiring

These are not "the design is wrong" — they're the natural scope edges of a one-session implementation.

1. **Materialized Gaussians don't persist across samples within a step.**  The trainer calls `model.reset_state(device)` at the start of each sample (each triplet is treated as its own trajectory). Materialized Gaussians from sample N die before sample N+1 starts. The benefit shows up in the spawner's learned weights (the spawner sees over time which tile-locations have parents whose children materialized → adjusts its own spawn-bias) rather than in within-step accumulation.

2. **No cross-step canvas persistence.** Full Diolatzis-faithful behavior would have the canvas survive across many training steps within one trajectory, with materialized children stacking up over many frames. Doing that requires:
   - Dataset emitting consecutive (i, i+1, i+2), (i+2, i+3, i+4), … triplets in batch order
   - Trainer not calling `reset_state` between samples within a "trajectory"
   - A trajectory length CLI flag with the dataloader respecting it
   This is a meaningful refactor (~1 week) that should follow once pico-005 produces a first metric — if uniform-density-plus-Sobel-plus-within-step-materialization isn't enough, this is the next investment.

3. **Brightness drift is opacity-proportional.** Currently `child.brightness` drifts at `0.1 × child.opacity_drift_rate × normalized_grad`. A separate brightness signal (the parent's *rendered feature magnitude*, not its position gradient) would be more faithful to the paper, but requires a second hook. Deferred.

4. **No materialization quota.** Under a sustained high-loss regime, every parent might materialize a child each step → canvas fills up rapidly → `add()` raises overflow. Workaround: the canvas-pruning policy (`prune_to_count`) already exists; trainer could call it when overflow is imminent. Not wired yet. For pico-005 at TartanAir HR with cap=16384 and 4800 baseline actives this won't trip; flag for higher-res training.

## Phase 3 recommendation

Run pico-005 first with `--enable-parent-child` ON, drift_rate=0.05, decay=0.98. The default config (16384 cap, 4800 base actives, ~50% headroom) gives parent-child ~11000 slots for materializations before overflow. If materializations spike toward overflow in the first 5K steps, drop to drift_rate=0.02. Track `materialized` in history.jsonl to see whether the mechanism is firing meaningfully (>5 per step on average = working; 0 = dead).
