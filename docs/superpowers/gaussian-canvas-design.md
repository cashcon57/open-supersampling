# OSS-Gaussian Persistent Canvas — Design Notes

**Sprint:** 5
**Plan:** `docs/superpowers/plans/2026-05-01-gaussian-sprint-5-plan.md`
**Spec:** `docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md`
**Status:** Authoritative for the Sprint 5 Python v1.

This doc records the four design decisions that bound the Sprint 5 module.
Tests reference these; reviewers gate against them.

---

## 1. SoA vs AoS

We chose **Struct of Arrays**: separate `(N, …)` tensors for `positions`,
`scales`, `rotations`, `colors`, `age`, `error`, plus a `(N,)` `alive`
mask.

**Why:**
- **Per-frame ops are columnar.** Warp touches positions only. Render
  reads positions/scales/rotations/colors. Error scoring writes one
  scalar per Gaussian. Each step is a single contiguous tensor sweep.
- **Vectorises trivially.** PyTorch ops on `(N, …)` tensors map directly
  to BLAS / CUDA without manual gather. AoS would force per-field
  strided access on every op.
- **Future CUDA kernel friendly.** Coalesced loads on positions become
  one warp per 32 Gaussians. AoS would block coalescing because the
  next-Gaussian's position lives one struct-stride away.
- **Capacity-stable.** Pruning toggles `alive[i] = False`; spawn fills
  the first `False` slot. No realloc, no compaction step. The buffer is
  a fixed-size ring with a free list implicit in `alive`.

**Trade-offs accepted:**
- A single Gaussian's full state requires reading from 6 tensors. We
  pay this cost only at debug / log boundaries, never in the hot path.
- The `GaussianBatch` snapshot is a lightweight view; we never mutate
  through it.

---

## 2. Why covariance is frozen frame-to-frame

GS-STVSR (2025) reports correlation 0.99 between adjacent frames'
covariance for the same scene Gaussian. The structural primitive's
*shape* is temporally stable; only its *position* and *colour* shift.

**Implication for Sprint 5:**
- Warp updates `positions` only. `scales` and `rotations` are read-only
  after spawn.
- The Sprint-4 param network is invoked only on **spawn events** (a
  Gaussian retiring + replacement appearing), not every frame.
- Network MAC budget drops from O(N) per frame to O(spawn_rate × tile
  area) per frame — typically 1–2% of the naive cost.

**Update path on spawn only:**
```
frame N:    Gaussian i dies (out-of-frame OR high tile error)
frame N:    network predicts a fresh Gaussian for tile T → fills slot i
frame N+1+: only positions[i] mutates; scales[i] / rotations[i] frozen
            until i dies again
```

This is the core perf win. The whole sprint is built around preserving
this invariant.

---

## 3. Pruning policy decision tree

Order is significant: each Gaussian is evaluated against rules **top
down**, the first match wins, total prune count is then clamped to the
per-frame budget.

```
                ┌─────────────────────────┐
                │  for each alive Gaussian │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │  in_frame == False?     │ → PRUNE (rule R1)
                └────────────┬────────────┘
                             │ no
                ┌────────────▼─────────────────────────────┐
                │  age > age_max  AND                       │
                │  g_error < 75th-percentile g_error?       │ → PRUNE (R2)
                └────────────┬─────────────────────────────┘
                             │ no
                ┌────────────▼─────────────────────────────┐
                │  tile_error[g_tile] > 95th-percentile     │
                │  AND age >= min_age_before_prune?         │ → PRUNE (R3)
                └────────────┬─────────────────────────────┘
                             │ no
                          KEEP
```

**Total prunes ≤ `max_prune_per_frame`** (default 5% of capacity). Each
prune triggers exactly one spawn slot, so the canvas is in a stable
lifecycle ringbuffer.

**Why these rules:**
- **R1** — out-of-frame Gaussians can never contribute again; cheap to
  detect via the warp's `in_frame` flag.
- **R2** — long-lived Gaussian that consistently scores low on its tile
  is dead weight; recycling it lets a different region get more budget.
  Threshold is age-conditioned to avoid churning brand-new Gaussians.
- **R3** — high tile error means the network can do better; replace.
  `min_age_before_prune` (3 frames default) prevents oscillation.

---

## 4. Spawn cost analysis (sparse network call)

A spawn pass does **one forward through the Sprint-4 network on a tiny
sub-input** containing only the high-error tiles.

**Anatomy of one spawn pass:**
- Inputs: `M` tiles × 16×16 px × 12 channels (LR + depth + motion +
  normals + canvas hint).
- Outputs: `M × K_per_tile × per_gauss_channels` — decoded by
  `OutputHead.decode` into a `GaussianBatch`.

**Tier budgets (assuming Sprint 5 max_prune = 5% capacity, K_per_tile
from Sprint 4 tier table):**

| Tier      | Capacity | Max prunes/frame | K_per_tile | Tiles spawned/frame |
|-----------|---------:|-----------------:|-----------:|--------------------:|
| Pico      | 1 000    | 50               | 3          | ~17                 |
| Lite      | 5 000    | 250              | 5          | 50                  |
| Standard  | 8 000    | 400              | 5          | 80                  |
| Ultra     | 15 000   | 750              | 8          | ~94                 |

The network is convolutional; 80 tiles batched into one (80, 12, 16, 16)
forward pass is a fraction of a millisecond on the 3080 Ti relative to
the dense per-frame inference path the canvas replaces.

**Why this is cheaper than naive every-frame inference:**
- Naive: every complex tile (~30% of frame, hundreds of tiles per LR
  frame) goes through the network every frame.
- Canvas: only tiles that **actually need new Gaussians** go through the
  network. Steady-state spawn rate is a few tens of tiles, not hundreds.

The expected speedup is the master-plan §5 latency margin — the source
of the "≤ 110% OSSPico" graduation criterion budget.

---

## Cross references

- Renderer API: `oss/gaussian/renderer/rasterizer.py`
- Decoder: `oss/gaussian/network/output_head.py`
- Tile mask: `oss/gaussian/classifier/classifier.py`
- Sprint plan: `docs/superpowers/plans/2026-05-01-gaussian-sprint-5-plan.md`
