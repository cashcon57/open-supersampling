# OSS-Gaussian — Sprint 5 Plan

**Spec:** `docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md`
**Master plan:** `docs/superpowers/plans/2026-05-01-gaussian-master-plan.md`
**Design doc (this sprint):** `docs/superpowers/gaussian-canvas-design.md`
**Branch:** `v0.2-dev`
**Estimate:** ~3 weeks (12 tasks; some parallelisable; 1 task blocked on Sprint 2).

Sprint 5 produces the **temporal heart** of OSS-Gaussian — a persistent GPU
buffer of N Gaussians that survives frame-to-frame, gets warped by motion
vectors, and is repaired in-place via per-tile reconstruction-error driven
prune+spawn. It depends on Sprints 1–4 (renderer, classifier, param network)
and feeds Sprint 6 (frame extrapolation) directly.

---

## Inputs from prior sprints

- **Sprint 1** — `oss.gaussian.renderer.Rasterizer` + `GaussianBatch` (CPU
  reference + CUDA gsplat path). The canvas calls this every frame.
- **Sprint 2** — Cyberpunk 2077 D3D12 hook producing per-frame
  (color, depth, motion). **T5.10 is gated on Sprint 2 landing.** The rest
  of Sprint 5 runs on synthetic data.
- **Sprint 3** — `TileClassifier` mask of complex/simple tiles. The canvas
  uses this to gate spawn invocations of the param network (simple tiles
  do not spawn new Gaussians).
- **Sprint 4** — `GaussianParamNetwork` + `OutputHead`. Spawn path calls
  the network on a sparse subset of tiles (the high-error ones) and
  decodes via `OutputHead.decode` → fresh Gaussians.

## Outputs Sprint 6 consumes

- `PersistentCanvas` with stable `update(...) → GaussianBatch` per frame.
- `temporal_stability` metric (frame-to-frame pixel delta in flat regions).
- Bench harness measuring per-frame canvas time on CPU and CUDA.

---

## Key design decisions (codified now to avoid drift)

1. **SoA (struct-of-arrays) layout.** Tensors `positions (N,2)`,
   `scales (N,2)`, `rotations (N,)`, `colors (N,F)`, `age (N,)`,
   `error (N,)`. Per-frame ops (warp, render, error scoring) touch one
   array end-to-end → cache friendly, vectorises trivially in PyTorch and
   later in a custom CUDA kernel. AoS would force gathers.
2. **Covariance frozen frame-to-frame.** GS-STVSR shows 0.99 frame-to-frame
   covariance correlation. Scale + rot are written **only on spawn**; warp
   only mutates positions. This is the core perf win and the reason a
   small param network suffices.
3. **Bilinear warp with sub-pixel sampling.** Each Gaussian's μ samples
   the motion-vector field at its (sub-pixel) position via bilinear
   interpolation, then μ ← μ + sampled_mv. Out-of-frame after warp →
   pruned.
4. **Per-tile error, not per-Gaussian error.** Compute per-tile MSE
   between the rendered canvas and the upsampled LR input. Each Gaussian
   inherits the error of the tile its (post-warp) center falls into.
   Per-Gaussian rendering attribution is a v2 perf-tune problem; per-tile
   is a strict superset of the information needed to drive prune/spawn.
5. **Prune ↔ spawn are paired** under a fixed Gaussian budget. The
   replacement Gaussians for a high-error tile come from a **single
   sparse call** to the Sprint-4 network on just that tile's input
   patch. Net N stays roughly constant frame-to-frame (a lifecycle
   ringbuffer, not a free-for-all).
6. **Policy/mechanism split.** `prune_spawn.py` exposes
   `select_for_pruning(state, frame, …) → indices` and
   `apply_prune_spawn(state, prune_idx, new_gaussians) → new_state`.
   Tuning the rules never touches the mechanism.

## Constraints

- **Pure PyTorch v1.** No custom CUDA kernels. Same code path runs on CPU
  and CUDA. Perf tuning and a fused kernel are post-Sprint-5 work.
- **No training in this sprint.** Sprint 5 reuses Sprint 4 network
  weights as-is. Param-net inference path on tiny tile patches is the
  only training-adjacent code introduced.
- **Sprint 1–4 code untouched.** Sprint 5 is purely additive under
  `oss/gaussian/canvas/` + `tests/gaussian/test_canvas.py`.

---

## Files

```
oss/gaussian/canvas/
  __init__.py            T5.1  — public API re-exports
  canvas.py              T5.1  — PersistentCanvas (SoA state + update/render)
  warp.py                T5.2  — bilinear motion-vector warp on positions
  error_detection.py     T5.3  — per-tile MSE; per-Gaussian error mapping
  prune_spawn.py         T5.4, T5.5 — prune policy + spawn mechanism
  metrics.py             T5.7  — temporal stability metric
  bench.py               T5.9  — perf bench harness (CPU + CUDA)
tests/gaussian/
  test_canvas.py         T5.1–T5.6, T5.8 — all unit + integration tests
docs/superpowers/
  gaussian-canvas-design.md  T5.0 — design notes (SoA, frozen Σ, decision tree)
oss/gaussian/canvas/cyberpunk_smoke.py
                         T5.10 — gated on Sprint 2 (live G-buffer feed)
```

---

## Tasks

### T5.0 — Design doc
**Goal:** One-page design rationale committed before code, so reviewers can
sanity-check decisions against the spec.

**Files:** `docs/superpowers/gaussian-canvas-design.md`

**Steps:**
1. Write SoA-vs-AoS rationale.
2. Document covariance-frozen invariant + cite GS-STVSR finding.
3. Pruning decision tree (out-of-frame → age + low contribution → tile
   error > τ).
4. Spawn cost analysis: one sparse param-net forward pass on a 16×16
   tile patch is ~K MACs cheap; budget table for 1K/8K/15K canvases.

**Verify:** Doc renders. Linked from this plan and from
`oss/gaussian/canvas/canvas.py` module docstring.

**Acceptance:** Design decisions explicit and traceable to spec.

**Time:** 0.5 day.

---

### T5.1 — Canvas data structure
**Goal:** `PersistentCanvas` class with SoA tensors, lifecycle ops, and a
`render()` that delegates to Sprint-1 `Rasterizer`.

**Files:** `oss/gaussian/canvas/canvas.py`, `oss/gaussian/canvas/__init__.py`

**API:**
```python
canvas = PersistentCanvas(
    capacity=8000,         # standard tier; 1K pico, 5K lite, 15K ultra
    feat_dim=3,
    device="cpu",          # cuda when available
    dtype=torch.float32,
    output_hw=(720, 1280),
)
canvas.initialize_random()                                # T5.1
canvas.initialize_from_network(decoded_params)            # T5.1 helper
gb: GaussianBatch = canvas.snapshot()                     # SoA → batch
img = canvas.render()                                     # (F, H, W)
```

**Steps:**
1. SoA tensors as `nn.Buffer`-style attributes (no autograd by default;
   the canvas is state, not parameters).
2. `alive` mask `(N,)` bool — pruning toggles to False; spawn fills the
   first False slot. Alive-count getter for tests.
3. `snapshot()` filters by `alive` and returns a `GaussianBatch`.
4. `render()` calls `Rasterizer(snapshot(), output_hw)`.
5. `to(device)` mirrors PyTorch idiom.

**Verify:** `pytest tests/gaussian/test_canvas.py::test_canvas_init -v`
passes (random + network-driven init, alive-count contract).

**Acceptance:** Capacity invariant — alive count never exceeds capacity.
SoA tensor shapes match the design doc.

**Time:** 2 days.

---

### T5.2 — Motion warp
**Goal:** Apply per-frame motion vectors to position tensor via bilinear
sample of the MV field at each Gaussian's (sub-pixel) center.

**Files:** `oss/gaussian/canvas/warp.py`

**API:**
```python
new_xy, in_frame = warp_positions(xy, motion, hw)
# xy:     (N, 2) sub-pixel
# motion: (2, H, W) per-pixel motion vectors
# hw:     (H, W) frame dimensions
# new_xy: (N, 2) shifted positions
# in_frame: (N,) bool — False if warped outside [0, W) × [0, H)
```

**Steps:**
1. Use `F.grid_sample` for bilinear sample (normalised coords).
2. Edge policy: `padding_mode='zeros'` returns 0 motion at edges; that's
   fine because the `in_frame` mask catches it.
3. CPU + CUDA validated by the same code path.

**Verify:**
- `test_warp_zero_motion`: still image with `motion=0` → positions
  unchanged.
- `test_warp_constant_motion`: uniform `(dx, dy)` field → every
  position shifts by exactly `(dx, dy)` (within 1e-5 absolute).
- `test_warp_out_of_frame`: positions warped past the frame edge get
  `in_frame=False`.

**Acceptance:** Errors well below sub-pixel accuracy. No silent NaNs at
edges.

**Time:** 1 day.

---

### T5.3 — Error detection
**Goal:** Per-tile MSE between rendered canvas and upsampled LR input,
plus per-Gaussian error lookup (each Gaussian inherits its tile's error).

**Files:** `oss/gaussian/canvas/error_detection.py`

**API:**
```python
tile_err = per_tile_mse(rendered, lr_upsampled, tile_size=16)  # (h, w)
g_err    = gaussians_error_from_tiles(xy, tile_err, tile_size, hw)  # (N,)
```

**Steps:**
1. `per_tile_mse`: subtract, square, `F.avg_pool2d(kernel=tile_size)`.
2. `gaussians_error_from_tiles`: integer-divide xy by tile_size to find
   the owning tile; clamp into `[0, h) × [0, w)`. Out-of-frame
   Gaussians get `+inf` so prune always chooses them first.

**Verify:**
- `test_error_blank_canvas`: zero canvas vs. random LR → high error
  everywhere.
- `test_error_matched_canvas`: render that already matches LR → near-zero
  error.
- `test_error_per_gaussian_lookup`: a Gaussian sitting in tile (1,2) gets
  exactly `tile_err[1,2]`.

**Acceptance:** Error scales linearly with intensity delta squared.
Out-of-frame Gaussians flagged with `+inf`.

**Time:** 1 day.

---

### T5.4 — Pruning policy
**Goal:** Pure-function selection of which Gaussians to retire this frame,
with explicit rule precedence.

**Files:** `oss/gaussian/canvas/prune_spawn.py`

**API:**
```python
prune_idx = select_for_pruning(
    alive_mask,           # (N,) bool
    in_frame,             # (N,) bool — False = out of frame after warp
    age,                  # (N,) int — frames alive
    g_error,              # (N,) float — per-Gaussian error
    tile_error,           # (h, w) per-tile MSE
    policy=PrunePolicy(...),
)
# prune_idx: (M,) long — indices to mark dead
```

**Decision tree (in order):**
1. Out-of-frame (`in_frame == False`) → prune.
2. `age > age_max` AND `g_error < age_low_error_pct` (75th-percentile)
   → prune (dead-weight long-lived Gaussian).
3. Top-k Gaussians whose tile error is in the upper `tile_error_pct`
   (default 95th percentile) → prune (replaced by spawn).

Total prune count clamped to `max_prune_per_frame` (default 5% of capacity).

**Verify:**
- `test_prune_out_of_frame`: only out-of-frame Gaussians selected when no
  other rule fires.
- `test_prune_aged_lowcontrib`: age > threshold + low error → pruned.
- `test_prune_high_tile_error`: high-tile-error Gaussians selected; others
  spared.
- `test_prune_budget`: never exceeds `max_prune_per_frame`.

**Acceptance:** Rule precedence is deterministic; pruning never touches
already-dead slots.

**Time:** 1.5 days.

---

### T5.5 — Spawn integration
**Goal:** Replace pruned Gaussians by calling the Sprint-4 network on the
tiny tile patches that fired the rule. Net Gaussian count holds steady.

**Files:** `oss/gaussian/canvas/prune_spawn.py` (continued)

**API:**
```python
new_g = spawn_for_tiles(
    network,              # GaussianParamNetwork
    output_head,          # OutputHead
    lr_frame,             # (B=1, C, H, W)
    g_buffers,            # depth/motion/normals
    canvas_render,        # (3, H, W) — current render fed back as canvas hint
    spawn_tiles,          # (M, 2) tile (y, x) coords
    tile_classifier_mask, # (h, w) — bypass simple tiles
)
# new_g: GaussianBatch with M*K_per_tile Gaussians

apply_prune_spawn(state, prune_idx, new_g)
```

**Steps:**
1. Filter `spawn_tiles` against the tile-classifier mask — never spawn
   on simple tiles.
2. Build a sparse-tile sub-input: gather only the tiles in question,
   stacked into a small (B', C, T, T) batch. Network is convolutional
   so it accepts the sub-batch directly.
3. Decode raw tensor via `OutputHead.decode` → fresh `(Δμ, scale, rot,
   color)` per Gaussian. Map decoded tile-local coords back to canvas
   pixel space.
4. `apply_prune_spawn` writes the new Gaussians into the slots the prune
   step freed (alive=False slots first).

**Verify:**
- `test_spawn_high_error_tiles`: feed a canvas with one obviously-wrong
  tile; spawn fires on that tile.
- `test_spawn_skips_simple_tiles`: classifier marks the bad tile simple;
  no spawn happens (system relies on the bilinear path for simple tiles).
- `test_spawn_preserves_capacity`: alive count after prune+spawn within
  `[capacity - max_prune_per_frame, capacity]`.

**Acceptance:** Spawn path is a single sparse forward pass. Total
Gaussian count stays inside ±5% of capacity over 100 sequential frames.

**Time:** 2 days.

---

### T5.6 — Canvas update loop
**Goal:** Tie T5.1–T5.5 together as one entry-point method:
```python
canvas.update(motion, lr_frame, g_buffers, tile_mask, network, head)
```

**Files:** `oss/gaussian/canvas/canvas.py`

**Per-frame sequence:**
1. `xy ← warp_positions(xy, motion)`; flag out-of-frame.
2. `age ← age + 1` for alive Gaussians.
3. `rendered ← render()`.
4. `tile_err ← per_tile_mse(rendered, upsample(lr_frame))`.
5. `g_err ← gaussians_error_from_tiles(xy, tile_err)`.
6. `prune_idx ← select_for_pruning(...)`.
7. `spawn_tiles ← top_error_tiles(tile_err, k=len(prune_idx) // K_per_tile)`.
8. `new_g ← spawn_for_tiles(network, head, ..., spawn_tiles, tile_mask)`.
9. `apply_prune_spawn(state, prune_idx, new_g)`.

**Verify:** `test_canvas_update_one_frame_lowers_error` — start from a
deliberately-wrong canvas; one update round lowers the per-tile MSE
mean by ≥20%.

**Acceptance:** No NaNs, no negative ages, alive count within budget.

**Time:** 1.5 days.

---

### T5.7 — Temporal stability metric
**Goal:** Quantify ghosting — frame-to-frame pixel delta in flat regions
(low-gradient tiles). Drives the graduation criterion in master plan §
Graduation Decision Point.

**Files:** `oss/gaussian/canvas/metrics.py`

**API:**
```python
ts = temporal_stability(
    rendered_t,           # (3, H, W)
    rendered_tplus1,      # (3, H, W)
    tile_size=16,
    flat_threshold=0.02,  # gradient mag below this = "flat"
)  # → scalar (lower = more stable)
```

**Steps:**
1. Compute per-tile gradient magnitude on `rendered_t` (Sobel + avg-pool).
2. Mask = tiles where gradient magnitude < `flat_threshold`.
3. Per-pixel delta = `|rendered_tplus1 - rendered_t|.mean(0)`; reduce
   over pixels in flat tiles. Return mean.

**Verify:**
- `test_temporal_stability_static`: identical frames → 0.
- `test_temporal_stability_uniform_shift`: small constant offset on a
  flat region → metric equals offset (within tolerance).

**Acceptance:** Pure function, no canvas state; reusable from Sprint 6.

**Time:** 0.5 day.

---

### T5.8 — Integration test: end-to-end one frame
**Goal:** Cover the full warp → render → error → prune+spawn → render
loop against a synthetic ground truth. This is the test that catches
inter-module bugs.

**Files:** `tests/gaussian/test_canvas.py`

**Steps:**
1. Build a synthetic `(H, W) = (128, 128)` LR frame: a single bright
   square on dark background.
2. Initialize canvas with **wrong** Gaussians (square in the wrong
   position). Render → high error.
3. Fake motion vectors that move the canvas square towards the LR
   square.
4. Stub network: returns Gaussians that match the LR square exactly.
5. Run one `canvas.update(...)`.
6. Assert post-update `per_tile_mse(canvas.render(), lr_upsampled).mean()`
   is < 50% of pre-update mean.
7. Assert alive count is within `[capacity - max_prune, capacity]`.

**Verify:** Test passes on CPU. Same test runs on CUDA when available
(skip-mark guarded).

**Acceptance:** Integration test is the canonical "did Sprint 5 work"
signal; always green on `main`.

**Time:** 1 day.

---

### T5.9 — Perf bench
**Goal:** Measure per-frame canvas time at the operating points OSS-G
will actually use. No optimisation yet — just a baseline so post-sprint
CUDA work has a number to beat.

**Files:** `oss/gaussian/canvas/bench.py`

**Configs:**
- Capacity: 1K, 5K, 8K, 15K
- Output HW: 720p, 1080p, 1440p
- Backends: CPU (reference rasterizer), CUDA gsplat (when available)

**Steps:**
1. 10 warm-up updates, 100 timed updates.
2. Report `mean / p50 / p95 / p99` ms per phase: warp / render / error /
   prune / spawn / total.
3. CSV at `oss/gaussian/canvas/bench/canvas_bench.csv`.

**Verify:** `python -m oss.gaussian.canvas.bench` produces CSV. Sanity
checks: warp+error+prune all < 1 ms at 8K on CUDA. Render dominates.

**Acceptance:** CSV checked in. Numbers documented in design doc.

**Time:** 1 day.

---

### T5.10 — Cyberpunk live integration smoke (gated on Sprint 2)
**Goal:** Drive `PersistentCanvas` from real Cyberpunk G-buffers captured
by the Sprint-2 D3D12 hook. **Do not start before Sprint 2 lands.**

**Files:** `oss/gaussian/canvas/cyberpunk_smoke.py`

**Steps:**
1. Read 60 sequential frames from a Sprint-2 dump.
2. Initialize canvas from frame 0 via the param network.
3. For frames 1..59: call `canvas.update(...)` with that frame's
   motion / LR / G-buffers.
4. Compute `temporal_stability` series; assert mean < OSSPico baseline
   on the same frame set (master-plan §5 graduation criterion).
5. Save side-by-side video for the human review checkpoint.

**Verify:** Smoke run completes without NaN or OOM on RTX 3080 Ti.

**Acceptance:** First real-data signal that Sprint 5 produces
ghosting-free upscale.

**Time:** 2 days. **BLOCKED on Sprint 2.**

---

### T5.11 — Comparison vs OSSPico on captured frames
**Goal:** Numerical comparison vs the existing pixel-based track on the
same Cyberpunk frame set (PSNR + SSIM + temporal stability). Feeds the
master-plan graduation gate.

**Files:** `oss/gaussian/canvas/cyberpunk_smoke.py` (extended), report at
`docs/superpowers/sprint-5-comparison-report.md`.

**Verify:** Report numbers match graduation criterion shape: PSNR + SSIM
+ temporal stability + latency rows.

**Acceptance:** Report committed; no decision yet — that's Sprint 7
graduation gate. **BLOCKED on T5.10.**

**Time:** 1 day.

---

### T5.12 — Sprint 5 code review checkpoint
**Goal:** Run review pipeline on Sprint 5 commits before Sprint 6 starts.

**Steps:**
1. `python -m oss.gaussian.review.run --sprint 5 --commit-range <base>..HEAD`.
2. Review artifacts saved to `oss/gaussian/review/artifacts/sprint-5/`.
3. Judge verdict APPROVE → mark sprint complete, proceed to Sprint 6.
4. REQUEST_CHANGES → iterate.
5. BLOCK → escalate to user.

**Verify:** Judge verdict file exists and is APPROVE.

**Time:** 0.5 day.

---

## Total time estimate

| Task | Days |
|---|---|
| T5.0  Design doc          | 0.5 |
| T5.1  Canvas data struct   | 2.0 |
| T5.2  Motion warp          | 1.0 |
| T5.3  Error detection      | 1.0 |
| T5.4  Prune policy         | 1.5 |
| T5.5  Spawn integration    | 2.0 |
| T5.6  Update loop          | 1.5 |
| T5.7  Temporal metric      | 0.5 |
| T5.8  Integration test     | 1.0 |
| T5.9  Perf bench           | 1.0 |
| T5.10 Cyberpunk smoke      | 2.0 (gated) |
| T5.11 vs OSSPico report    | 1.0 (gated) |
| T5.12 Review checkpoint    | 0.5 |
| **Total**                  | **15.5 days ≈ 3 weeks** |

T5.0–T5.9 + T5.12 are unblocked and form a complete Sprint 5 deliverable
on synthetic data. T5.10–T5.11 are the live-data validation tier and
slot in once Sprint 2 lands.

---

## Risks + mitigations

1. **Sparse network call is slow** (one tiny forward per frame per
   high-error tile). *Mitigation:* batch all spawn tiles into a single
   sub-input. If still slow, cache tile encodings frame-to-frame.
2. **Prune oscillation** — same Gaussians get pruned + respawned every
   frame. *Mitigation:* `min_age_before_prune` floor in `PrunePolicy`
   (default 3 frames). Already in T5.4 decision tree.
3. **Motion vector noise causes drift on flat regions.** *Mitigation:*
   the warp respects MV magnitude regardless of region; relying on the
   error-driven prune to repair drift, with the temporal-stability
   metric as the regression alarm.
4. **CUDA path differs subtly from CPU.** *Mitigation:* every test runs
   both backends with a tolerance-matched assertion.
5. **Sprint 2 slips.** *Mitigation:* T5.10/T5.11 are explicitly gated;
   T5.0–T5.9 ship Sprint 5 on synthetic data alone.
