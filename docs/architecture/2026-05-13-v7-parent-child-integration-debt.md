# v7 parent-child spawner — integration debt + design call

**Date:** 2026-05-13
**Status:** **Implemented but unwired.** The mechanism is unit-tested in `tests/sr/v7/test_nd_canvas_state.py` (20 tests) but `scripts/sr_train_v7.py` never calls `initialize_children_for_new_parents` or `materialize_to_canvas`. As far as Phase 3 (pico-005) is concerned, parent-child is dead code.
**Why it matters:** see `docs/capture/v7-training-data-spec.md` discussion of fine-Gaussian density. The current trainer's uniform per-tile spawn rate cannot represent sub-pixel features (thin wires, distant fences, hair). Parent-child is the existing-but-unused lever that would let density grow adaptively where SR loss is high.

## What's there

`oss/sr/v7/parent_child_spawner.py` implements the Diolatzis-2024 deferred-materialization pattern:

```
ChildState(dpos, dcov_raw, dfeat, opacity, brightness)  # capacity, capacity, ...
materialize_mask(child, n_live) -> bool[n_live]
materialize_to_canvas(canvas, child) -> int                # appends to canvas
initialize_children_for_new_parents(child, parent_indices)
```

Materialization triggers when `child.opacity > 0.1` OR `child.brightness > 0.01`. Once triggered, a new top-level Gaussian gets appended to the canvas at (parent_pos + child.dpos) with parent_cov + child.dcov_raw and the child slot is reset to dormant.

## The integration question that has to be answered first

The Diolatzis paper drives child.opacity drift via **per-parent loss residual**, accumulated over many steps. It is **not** a gradient-through-Parameters mechanism. The current `ChildState` is a plain dataclass with regular tensors, which means:

1. If we make `child.opacity` a `nn.Parameter` and let AdamW push it via gradient, `materialize_to_canvas` and `initialize_children_for_new_parents` BOTH fail with `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation` because they do `child.dpos[parent_idx] = 0.0` etc. Confirmed empirically (`/tmp/bench_parent_child.py`).

2. The clean fix is to drive the drift via **manual residual accumulation under torch.no_grad():**

   ```python
   with torch.no_grad():
       # After backward, before optim.step:
       parent_grads = ...   # source still unresolved (see below)
       child.opacity[:n_live] += drift_rate * parent_grads
       child.opacity[:n_live] *= decay
   ```

   But this needs a per-parent gradient signal that the current model doesn't expose. Three candidate signals, none of which is implemented:

   - **Per-parent contribution to SR loss**, via splatting the SR loss back through the rasterizer's alpha-blend weights. Cleanest but requires a custom backward hook on the rasterizer.
   - **Per-parent position-gradient norm**, computed by wrapping `canvas.positions` in a Parameter with `requires_grad=True` so gradient accumulates on each parent. Currently positions is a buffer (no requires_grad). Would require lifting requires_grad on a tensor mutated by `canvas.add()` — interacts non-trivially with the autograd-isolation fix in commit `e302cc2`.
   - **Per-parent rendered-feature gradient**, captured by hooking the active-view slice. Indirect but doesn't touch the canvas storage.

3. The Diolatzis paper alternative — **per-pixel residual splatted back to parents via the rasterizer's footprint** — is the most architecturally honest version but is a bigger code change (~200 LOC including a new backward hook).

## What needs to be true before this lands

- A clear answer on which of the three signals to use (proposed: per-parent rendered-feature gradient — least invasive).
- A unit test that demonstrates `child.opacity` rising under realistic SR-loss gradient pressure within a step budget (< 50 steps for a synthesized high-frequency target).
- A correctness check that materialization doesn't break the autograd-isolation fix (the new Gaussians on the canvas should still get reset/detached between steps).
- Capacity-overflow handling once materialization is fanned in — under a sustained high-loss regime the canvas could fill up faster than `prune_to_count` reclaims. Needs a per-step cap on `materialize_to_canvas` calls.

## What's currently *unblocked* even without parent-child

- The base spawner produces `O(tile_size⁻²)` Gaussians per frame at uniform density. With `tile_size=16, k_per_tile=2` (the new defaults), TartanAir 480×640 HR gets 4800 actives after 2 spawns. That's enough density for v7-pico-005 to validate the OSS-FX math + show non-trivial alpha=0.5 PSNR over the bicubic-midpoint baseline.
- Sub-pixel features will be under-resolved. The Phase 3 pico-005 metric will measure how much that costs. If the alpha=1 SR PSNR is within +/- 0.5 dB of v6.2-pico-002, parent-child integration moves up the priority list for Phase 3.b.
- The Sobel high-frequency loss term added in this same commit gives the model *some* incentive to learn edge structure, which mitigates the sub-pixel-coverage gap until parent-child lands.

## Estimated work to land parent-child

- Design call + design doc: ~3 days
- Implementation (residual mechanism + trainer wiring + tests): ~1 week
- Regression validation on TartanAir smoke + 1080p HR resolutions: ~3 days
- **Total: ~2 weeks** if the signal choice goes smoothly; ~4 if we discover the per-parent attribution needs the custom rasterizer backward hook.

## Decision deferred to

After v7-pico-005 produces its first 20K-step checkpoint. The alpha=1 PSNR + alpha=0.5 PSNR there will tell us whether sub-pixel coverage is the actual bottleneck or whether other model issues dominate.
