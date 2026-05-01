# OSS-Gaussian — Sprint 3 Detailed Plan: Tile Classifier

**Spec:** `docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md` (§3.1, §3.2 row 3, §13 Q3)
**Master plan:** `docs/superpowers/plans/2026-05-01-gaussian-master-plan.md` (Sprint 3 outline)
**Branch:** `v0.2-dev`
**Estimate:** ~1 week (5 working days)
**Parallelism:** Runs alongside Sprint 2 (D3D12 hook). Depends only on Sprint 1 renderer being importable.

---

## Goal

Per-frame **16×16 tile mask** that classifies tiles as **complex** (Sprint 4 network must predict Gaussian params) or **simple** (bypass network — bilinear passthrough straight to render). Eliminates ~70% of Gaussian-param-network cost.

The classifier is a **pure heuristic** in v1: gradient magnitude + depth discontinuity + motion magnitude (+ optional normal variance), thresholded adaptively so a tunable fraction of tiles is "complex" each frame.

No training, no learned components. Sprint 4 network is gated by this mask but does not feed back into it.

---

## Inputs / Outputs

| Tensor   | Shape          | Dtype        | Notes                                        |
|----------|----------------|--------------|----------------------------------------------|
| `frame`  | `(B, 3, H, W)` | float32      | LR RGB (linear or sRGB; tolerated either way)|
| `depth`  | `(B, 1, H, W)` | float32      | LR linear depth (any units; ratio-based)     |
| `motion` | `(B, 2, H, W)` | float32      | LR per-pixel motion vectors (px or NDC)      |
| `normals`| `(B, 3, H, W)` | float32      | Optional. Unit-length world/view normals     |
| **`mask`** | `(B, H/16, W/16)` | bool      | True = complex, False = simple               |

Conventions match the rest of OSS (NCHW, B-major). H, W must be multiples of `tile_size` (default 16); otherwise the trailing partial tile is dropped from the mask (documented).

---

## Tasks

### T3.1 — Classifier API + scaffold (0.5d)

**Goal:** `oss.gaussian.classifier.TileClassifier` importable, callable, returns a mask of correct shape and dtype. Body is a stub returning all-False.

**Files:**
- `oss/gaussian/classifier/__init__.py` — re-export `TileClassifier`, `overlay_mask`.
- `oss/gaussian/classifier/classifier.py` — class scaffold + dataclass for config.

**Steps:**
1. Define `TileClassifier(tile_size=16, target_complex_fraction=0.30, weights=...)`.
2. `__call__(frame, depth, motion, normals=None) -> Tensor[bool]` shape `(B, H/T, W/T)`.
3. Validate input shapes (NCHW, B agreement, H%T==0, W%T==0). Raise `ValueError` with concrete dims.
4. Stub body returns `torch.zeros(..., dtype=torch.bool)`.

**Verify:** `python -c "from oss.gaussian.classifier import TileClassifier; c=TileClassifier(); print(c(torch.zeros(1,3,128,128), torch.zeros(1,1,128,128), torch.zeros(1,2,128,128)).shape)"` prints `torch.Size([1, 8, 8])`.

**Acceptance:** Module imports cleanly on CPU-only machine. Shape contract enforced.

---

### T3.2 — CPU PyTorch reference: gradient + depth + motion features (1d)

**Goal:** Vectorized PyTorch implementation of per-tile complexity score, runs on CPU and CUDA from one code path.

**Files:**
- `oss/gaussian/classifier/classifier.py` (extend).

**Steps:**
1. **RGB gradient magnitude:** Sobel (or simple `[-1,0,1]`) along H and W on the luma channel; `grad_mag = sqrt(gx² + gy²)`. Reduce per-tile via `mean` (use `F.avg_pool2d` over tiles, kernel=tile_size, stride=tile_size).
2. **Depth discontinuity:** abs gradient on `log(depth + eps)` (scale-invariant); per-tile `max` reduction (use `F.max_pool2d`, since a single hard edge inside a tile must trigger).
3. **Motion magnitude:** `||motion||_2` per pixel; per-tile `mean`.
4. **Normal variance (optional):** if `normals is not None`, compute per-tile angular variance — 1 − ||mean(normals)||₂ (with tile-mean of unit vectors). Skip cleanly when `normals=None`.
5. **Combine:** weighted sum with config weights (defaults `wg=1.0, wd=1.0, wm=0.5, wn=0.25`). Each feature normalized to its own per-frame max (avoid one feature dominating due to unit mismatch) before weighting.

**Verify:**
- Score tensor shape `(B, H/T, W/T)`, dtype `float32`, finite.
- On a uniform input → score is ~0 everywhere.
- On a single-pixel impulse → exactly one tile lights up.

**Acceptance:** Runs in <5 ms on a 1080p single-frame batch on CPU (laptop). No Python loops over pixels or tiles.

---

### T3.3 — Adaptive thresholding (0.5d)

**Goal:** Pick a threshold per frame (per batch element) such that ~`target_complex_fraction` of tiles end up classified complex.

**Files:** `oss/gaussian/classifier/classifier.py`.

**Steps:**
1. Per batch element, flatten the score map; compute `kth = (1 - target_complex_fraction) * num_tiles`.
2. Use `torch.kthvalue` (CPU + CUDA) to find threshold; mask = `score > threshold`.
3. Edge case: `target_complex_fraction == 0.0` → all-False mask. `== 1.0` → all-True mask. Bypass `kthvalue` for these.
4. Edge case: ties at the threshold may push the actual fraction slightly above target — document and accept (within ±5% in tests).

**Verify:** Random score maps → mask fraction within ±5% of target across many seeds.

**Acceptance:** `target_complex_fraction` is honored within ±5% on synthetic and real-shaped inputs.

---

### T3.4 — Visualization helper (0.25d)

**Goal:** `overlay_mask(frame, mask) -> Tensor` that produces a debug visualization for documentation, paper figures, and tile-debug screenshots from gameplay captures.

**Files:** `oss/gaussian/classifier/classifier.py` (or `viz.py` if it grows).

**Steps:**
1. Up-sample `mask` (nearest) from `(B, H/T, W/T)` → `(B, 1, H, W)`.
2. Tint complex tiles red at 50% opacity over the original `frame`. Leave simple tiles untouched.
3. Optional grid lines at tile boundaries (toggle, default off).
4. Return `(B, 3, H, W)` float in `[0, 1]`.

**Verify:** Shape and dtype contract. Visual inspection on a saved PNG smoke test in pytest.

**Acceptance:** A 256×256 frame + synthetic mask produces a recognizable overlay; spot-check passes.

---

### T3.5 — Test suite (1d)

**Goal:** `tests/gaussian/test_classifier.py` covering correctness, shapes, threshold honoring, all on CPU.

**Files:** `tests/gaussian/test_classifier.py`.

**Steps — required cases:**
1. **Smooth-vs-noisy:** smooth gradient image + noise patch in known tile range → noisy tiles must be in `mask=True` set.
2. **Depth discontinuity:** flat color, depth has step in known tile range → those tiles must be `True`.
3. **Motion edge:** static color, MV non-zero in known patch → that patch must be `True`.
4. **Threshold honor:** random score → `mask.float().mean()` within ±5% of `target_complex_fraction`.
5. **Shape correctness:** inputs of `(1, 3, 128, 128)`, `(2, 3, 256, 384)`, `(1, 3, 720, 1280)` → mask is `(B, H/16, W/16)` bool.
6. **Bad input rejection:** non-multiple-of-tile-size H/W raises `ValueError`. Mismatched B raises.
7. **Optional normals:** classifier works with and without `normals` (no shape mismatch).

**Verify:** `python -m pytest tests/gaussian/test_classifier.py -v` passes on the `venv-py312` env.

**Acceptance:** All seven cases green. No flakiness across 50 reruns (use fixed seeds).

---

### T3.6 — Threshold selection ablation (0.5d)

**Goal:** Pick a default `target_complex_fraction` and feature weights backed by data.

**Files:**
- `scripts/gaussian_classifier_ablation.py` (new) — runs the classifier on a small dataset of LR frames + depth/motion (Sintel subset is fine; can stub on synthetic noise + sinusoid until Sprint 2 dumps Cyberpunk data).
- `results/gaussian/classifier_ablation.csv` — output (gitignored if large; checked-in if small).

**Steps:**
1. Sweep `target_complex_fraction ∈ {0.20, 0.25, 0.30, 0.35, 0.40}`.
2. Sweep weight combinations (small grid, 3³ = 27 max).
3. For each combo: log mean tile score, std, fraction in spatially smooth regions (false-positive proxy).
4. Pick the combo that minimizes false-positives at target=0.30 → set as defaults.

**Verify:** CSV exists. Defaults in `TileClassifier.__init__` reflect the picked combo (with a comment linking to the CSV row).

**Acceptance:** Picked defaults documented inline. Re-runnable script.

**Note:** This is heuristic-tuning, not training. No model weights produced.

---

### T3.7 — Performance benchmark (0.5d)

**Goal:** Confirm classifier is cheap enough not to dominate the per-frame budget.

**Files:**
- `oss/gaussian/classifier/bench.py` — timing harness (mirrors `oss/gaussian/renderer/bench.py` style).
- `results/gaussian/classifier_bench.csv`.

**Configs to bench:**
- 540p, 720p, 1080p, 1440p LR inputs.
- B=1, B=4.
- CPU + CUDA (skip CUDA if unavailable).

**Steps:**
1. 100 iterations after 10 warm-up. Report mean / p50 / p95 / p99.
2. Reality-check budgets (3080 Ti, B=1):
   - 540p classifier ≤ 0.10 ms
   - 1080p classifier ≤ 0.30 ms
   - 1440p classifier ≤ 0.50 ms
3. If any config blows the budget by >2×: open an issue and queue a CUDA-kernel follow-up (deferred from this sprint per spec — pure PyTorch is fast enough).

**Verify:** CSV produced. Numbers within budget or follow-up issue filed.

**Acceptance:** Documented numbers in CSV + brief comment in PR.

---

### T3.8 — Integration documentation (0.25d)

**Goal:** Document how Sprint 4 (network) and Sprint 5 (canvas) consume the mask.

**Files:**
- `docs/superpowers/integration-points.md` — append a new section, do not rewrite existing content.

**Section content:**
1. Mask shape, dtype, alignment with renderer tile_size (TILE_SIZE=16 — same as `oss/gaussian/renderer/rasterizer.py`).
2. How Sprint 4 consumes: gather complex tiles into a packed batch for network inference; scatter outputs back into a full per-tile param tensor (with simple tiles set to "passthrough" sentinel values).
3. How Sprint 5 consumes: spawn-from-LR is gated to complex tiles only; per-tile error-detection threshold may be relaxed inside simple tiles.
4. Visualization usage: `overlay_mask` for debug recordings + Sprint 7 cross-platform port verification.

**Verify:** Section appears at the bottom of `integration-points.md` between an `## 8.` header and the existing summary.

**Acceptance:** Sprint 4 / 5 designers can read the section and write their consumer code without re-reading the classifier source.

---

### T3.9 — Sprint 3 integration smoke test (0.5d)

**Goal:** End-to-end pipe — classifier output is shape-compatible with renderer's `TILE_SIZE`. Renderer + classifier importable in the same process. No collisions with Sprint 1 modules.

**Files:** `tests/gaussian/test_classifier_integration.py`.

**Steps:**
1. Import both `from oss.gaussian.renderer import Rasterizer, TILE_SIZE` and `from oss.gaussian.classifier import TileClassifier`.
2. Assert `TileClassifier(tile_size=TILE_SIZE)` constructs cleanly.
3. Run classifier on a synthetic LR frame; confirm `mask.shape[-2:] == (H // TILE_SIZE, W // TILE_SIZE)`.
4. Confirm existing Sprint 1 tests still pass (`pytest tests/gaussian/ -v`).

**Verify:** `pytest tests/gaussian/ -v` passes (excluding GPU-only marked tests on CPU runners).

**Acceptance:** No import errors, no shape mismatches, no regressions in Sprint 1 tests.

---

### T3.10 — Sprint 3 code review checkpoint (0.5d)

**Goal:** Run the code review pipeline on Sprint 3 commits.

**Steps:**
1. `python -m oss.gaussian.review.run --sprint 3 --commit-range <sprint-2-merge>..HEAD`
2. Artifacts saved to `oss/gaussian/review/artifacts/sprint-3/`.
3. APPROVE → close sprint, unblock Sprint 4.
4. REQUEST_CHANGES → iterate.
5. BLOCK → escalate.

**Verify:** Judge verdict file exists and is APPROVE.

**Acceptance:** Sprint closed, Sprint 4 ready to start.

---

## Time budget

| Task  | Estimate |
|-------|----------|
| T3.1  | 0.5d     |
| T3.2  | 1.0d     |
| T3.3  | 0.5d     |
| T3.4  | 0.25d    |
| T3.5  | 1.0d     |
| T3.6  | 0.5d     |
| T3.7  | 0.5d     |
| T3.8  | 0.25d    |
| T3.9  | 0.5d     |
| T3.10 | 0.5d     |
| **Total** | **~5.5d (1 week)** |

---

## Out of scope (deferred)

- **Custom CUDA kernel** for the classifier — pure PyTorch is fast enough (T3.7 confirms). If a config blows budget, file a follow-up Sprint-3-perf task. Not in this sprint.
- **Learned classifier** — v1 is pure heuristic. A learned variant is a future research direction once Sprint 4 trained network exposes per-tile error data.
- **32×32 tile size investigation** (spec §13 Q3) — keep at 16 to match renderer; revisit during Sprint 4 if network throughput is bottlenecked by tile count.
- **Cyberpunk-real-frame validation** — depends on Sprint 2 G-buffer dumps. Until then, validate on Sintel + synthetic. Re-run T3.6 ablation once Cyberpunk data lands.

---

## Risks

1. **`kthvalue` on huge tensors is slow.** Mitigation: tile counts are tiny (1080p / 16² = 8100 tiles); no risk.
2. **Feature scale mismatch dominates score.** Mitigation: per-frame normalization in T3.2, ablation in T3.6.
3. **Per-tile reduction choice (mean vs max) wrong for one feature.** Mitigation: pick `max` for depth discontinuity (single hard edge matters), `mean` elsewhere; revisit if T3.6 shows false-negatives at depth steps.
4. **Sprint 2 not done in time → no real Cyberpunk frames for T3.6.** Mitigation: synthetic + Sintel is enough to ship Sprint 3 with a calibration round queued for after Sprint 2.
