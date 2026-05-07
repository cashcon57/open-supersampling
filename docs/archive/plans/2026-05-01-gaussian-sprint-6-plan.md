# OSS-Gaussian — Sprint 6 Detailed Plan: Frame Extrapolation

> **Note (2026-05-05):** Sprint 6 has been **absorbed into v6** as the natural byproduct of Gaussian-canvas SR (rendering the same canvas at α<1 = frame extrapolation, essentially free). The current Sprint 6 / v6 architecture is documented in `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md`. This original plan remains valid for the FX-specific deliverables (`oss/gaussian/extrapolation/extrapolator.py` is already built per this plan); the v6 work makes those deliverables consume the v6-trained canvas.

**Spec:** `docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md` (§3.1, §3.2 row 6, §4.1)
**Master plan:** `docs/superpowers/plans/2026-05-01-gaussian-master-plan.md` (Sprint 6 outline)
**Design doc:** `docs/superpowers/gaussian-frame-extrapolation.md`
**Branch:** `v0.2-dev`
**Estimate:** ~1 week (5 working days)
**Depends on:** Sprint 1 (renderer), Sprint 5 (persistent canvas + warp).
**Parallelism:** Sprint 6 scaffolding can land while Sprint 5 is in flight by
coding to the API contract declared in `oss/gaussian/canvas/__init__.py`.
End-to-end runs require Sprint 5 merged.

---

## Goal

Render the persistent Gaussian canvas at fractional time positions
``t-1 + alpha`` for ``alpha ∈ [0, 1]``, producing **synthesised intermediate
frames between real renders** without any additional model.

The killer property: **frame extrapolation is the same operation as the
Sprint 5 motion warp, parameterised by a smaller alpha.** No separate
frame-generation network. DLSS Frame Generation is a heavy additive pass
over a learned optical-flow net; OSS-Gaussian's frame gen is one scalar
multiplied into the warp step that the canvas already performs every
frame.

Targets (master plan §4.1): 60→120 FPS in Cyberpunk 2077 on RTX 3080 Ti.
Stretch: 60→90 (low-overhead path) and 60→144 (G-sync display).

---

## Inputs from prior sprints

- **Sprint 1:** `oss/gaussian/renderer/` — `Rasterizer`, `GaussianBatch`.
- **Sprint 5:** `oss/gaussian/canvas/` — `PersistentCanvas`, `warp_canvas`.
  Sprint 6 imports both unchanged.

## Outputs Sprint 7 consumes

- `FrameExtrapolator` class — drives intermediate-frame rendering on M3 Max
  (Metal renderer port) and Steam Deck (Vulkan/ncnn) without code changes.
- `AlphaSchedule` + presets — used by the cross-platform engine to select
  cadences appropriate to display refresh rates.

---

## Key design decisions

1. **No new model.** Sprint 6 trains nothing. The whole sprint is about
   driving Sprint 5's warp at fractional alpha and measuring quality.
2. **API contract first.** `oss/gaussian/canvas/__init__.py` declares the
   `PersistentCanvas` / `warp_canvas` surface that Sprint 6 consumes.
   Sprint 5 fills it in. Sprint 6 unit tests use a synthetic warp so they
   pass before Sprint 5 merges.
3. **Latency budget.** Cost above-and-beyond a normal canvas render is one
   in-place add on the (N, 2) position tensor. Empirically (test
   `test_warp_is_essentially_free_relative_to_render`) this lands within
   2× of the bare render time on CPU, and is dwarfed by the rasteriser
   on GPU. Hard target: ``extrapolate_time < 1.1 × render_time``.
4. **Alpha distribution.** Multi-intermediate cadences (60→144, 60→240)
   distribute alphas uniformly across the displayed period so the maximum
   per-step warp magnitude stays minimal — high-alpha samples accumulate
   the most non-linear-motion error.
5. **Quality scaling.** PSNR vs the actual ``t+1`` frame is reported per
   alpha bucket (0.0, 0.25, 0.5, 0.75, 1.0). The number we publish is
   PSNR at alpha=0.5 — this is the canonical 60→120 FPS-doubling case.

---

## Files

```
oss/gaussian/extrapolation/
  __init__.py             T6.1 — public re-exports
  extrapolator.py         T6.1 — FrameExtrapolator class
  alpha_scheduler.py      T6.1 — AlphaSchedule + presets
  bench.py                T6.2 — latency benchmark (3080 Ti)
  quality.py              T6.3 — PSNR vs actual t+1 evaluator
  vs_dlss_fg.py           T6.4 — comparison harness (3080 Ti only)
oss/gaussian/canvas/
  __init__.py             contract stub (Sprint 6 lands defensively)
docs/superpowers/
  gaussian-frame-extrapolation.md   T6.1 — design rationale + failure modes
  gaussian-extrapolation-results.md T6.5 — quality + latency report
tests/gaussian/
  test_extrapolation.py   T6.1 — alpha=0/1/0.5 + scheduler tests
  test_extrapolation_canvas_integration.py  T6.6 (post Sprint 5)
```

Approx scaffold size: ~250 lines Python + ~250 lines tests + ~150 lines
docs. The bench / quality / vs_dlss_fg modules are <train-host>-only and ship
incrementally from T6.2 onward.

---

## Tasks

### T6.1 — Alpha-conditioned warp + scaffold (Day 1)

**Goal:** `FrameExtrapolator` exists, takes (canvas, motion, alpha,
output_hw), returns the rendered intermediate frame. `AlphaSchedule`
exists with the three presets. All tests in `test_extrapolation.py` pass
locally on CPU using a synthetic warp double.

**Steps:**

1. Write `oss/gaussian/canvas/__init__.py` API contract (TODO comments
   for Sprint 5). This is the doc Sprint 5 builds against — keep it
   accurate and minimal.
2. Implement `FrameExtrapolator.extrapolate(...)`. Validate inputs:
   alpha ∈ [0, 1], motion shape (2, H, W), output_hw positive.
3. Implement lazy import of `oss.gaussian.canvas.warp_canvas` so the
   module imports on machines without Sprint 5 merged. Allow callers to
   override via `warp_fn=` constructor kwarg.
4. Implement `AlphaSchedule` + `schedule_for(source_fps, target_fps)` +
   the three master-plan presets. Distribute intermediates uniformly via
   gcd-reduced cadence.
5. Tests in `tests/gaussian/test_extrapolation.py`:
   - alpha=0 → output equals direct canvas render at t.
   - alpha=1 → Gaussians shifted by full motion delta; rendered peak
     lands at the predicted t+1 location.
   - alpha=0.5 → midpoint property: ``p(0.5) - p(0) == 0.5 * (p(1) - p(0))``.
   - alpha out of range → `ValueError`.
   - schedule presets emit the right number of intermediates with all
     alphas in (0, 1).

**Verify:**
```
cd <home>/open-reconstruction-suite
source venv-py312/bin/activate
python -m pytest tests/gaussian/test_extrapolation.py -v
```
All tests pass on CPU without Sprint 5 merged.

### T6.2 — Latency measurement (Day 2)

**Goal:** On RTX 3080 Ti with Sprint 5 merged, the extrapolated frame
must complete *before* the next real-frame swapchain present. We need
hard numbers, not guesswork.

**Configs:**
- Canvas sizes: 5K, 8K, 15K Gaussians (Lite, Standard, Ultra tiers).
- Resolutions: 1080p, 1440p (Cyberpunk targets).
- Alphas: {0.0, 0.25, 0.5, 0.75, 1.0}.

**Steps:**

1. `oss/gaussian/extrapolation/bench.py` — torch.cuda timed loop, 100
   iterations after 10 warm-ups, reports mean / p50 / p95 / p99 ms per
   config.
2. Two reference numbers per config:
   - `render_only` — `Rasterizer(canvas.gaussians, output_hw)`
   - `extrapolate(alpha)` — full path through `FrameExtrapolator`
3. Acceptance threshold per config: ``extrapolate_p95 < 1.1 × render_only_p95``.
4. Output: CSV at `oss/gaussian/extrapolation/bench/bench_results_3080ti.csv`.
5. If any config exceeds the threshold, root-cause before T6.3 ships
   (most likely culprit: an accidental clone in `warp_canvas`).

**Verify:** CSV exists, threshold met for 8K @ 1440p (the canonical
Cyberpunk config). Numbers logged in
`docs/superpowers/gaussian-extrapolation-results.md`.

### T6.3 — Quality measurement (Day 3)

**Goal:** PSNR / SSIM of the predicted intermediate frame vs the actual
ground-truth frame at that time, on captured Cyberpunk sequences.

**Steps:**

1. `oss/gaussian/extrapolation/quality.py`:
   - For each consecutive frame triple ``(t-1, t, t+1)`` in a captured
     sequence: feed the canvas at time t and the t-1→t motion field
     into `FrameExtrapolator` at alphas {0.25, 0.5, 0.75, 1.0}, evaluate
     against the *same alpha along the t→t+1 axis* (alpha=1 evaluates
     against actual frame t+1).
   - Report PSNR, SSIM, LPIPS per alpha bucket aggregated across the
     sequence.
2. Use Sprint 5's Cyberpunk capture set (≥500 frames; same one used in
   Sprint 5 T5.8 OSSPico comparison).
3. Add test `test_extrapolation_canvas_integration.py` that runs the
   first 10 frames end-to-end through the real Sprint 5 canvas. Marked
   `@pytest.mark.cuda` so CPU runs skip it.
4. Report numbers in `gaussian-extrapolation-results.md` keyed by
   alpha and capture-scene.

**Verify:** PSNR at alpha=0.5 ≥ 32 dB on Cyberpunk capture set
(parity-with-Windows minimum bar; below this, frame gen is visibly worse
than the real frame). Lower bound is enforced by `quality.py` `--strict`
flag for CI gating.

### T6.4 — Comparison vs DLSS Frame Generation (Day 4)

**Goal:** Side-by-side measurement of OSS-Gaussian frame gen vs
DLSS Frame Generation on the same Cyberpunk sequence at the same FPS
target. **3080 Ti only.** Local CPU machines cannot run DLSS-FG; this
task does not block on the Mac.

**Steps:**

1. `oss/gaussian/extrapolation/vs_dlss_fg.py`:
   - Capture two Cyberpunk runs at matching seeds: one with DLSS-FG on,
     one with OSS-Gaussian frame gen on. Same scene, same camera path.
   - Per-frame metrics: PSNR vs reference (real-render-only at 120 FPS),
     mean motion-disocclusion ghosting score (defined in Sprint 5
     T5.7 and reused here), GPU frame time, total system latency
     (NVIDIA Reflex SDK if available, else swapchain-level wall time).
2. Output: comparison report in `gaussian-extrapolation-results.md`
   §"vs DLSS-FG", side-by-side metric table + 3 captured video clips
   (low-motion, high-motion, disocclusion-heavy).
3. Acceptance: OSS-Gaussian frame gen latency ≤ DLSS-FG latency,
   quality within 2 dB PSNR. Either way, results are honest — see
   master plan §11 risk 5: "DLSS comparison shows OSS-Gaussian worse"
   is an acceptable outcome that informs the graduation decision.

**Verify:** Report is committed; videos linked from the report; numbers
reproduced by running `python -m oss.gaussian.extrapolation.vs_dlss_fg
--scene <name>`.

### T6.5 — Failure-mode catalogue (Day 4, parallel with T6.4)

**Goal:** Document where alpha-warped frame extrapolation fails —
non-linear motion, fast rotation, occluder pop-in, etc. — so users know
when to dial down the cadence.

**Steps:**

1. Identify failure cases in the Cyberpunk captures from T6.3 and the
   DLSS comparison from T6.4. Look for:
   - Camera whip (angular velocity > ~120 °/s) — predicted Gaussian
     positions miss; ghosting in the rotating direction.
   - Disocclusion at high alpha — the Sprint 5 spawn path runs at
     real-frame cadence only; a Gaussian newly spawned at t never
     existed at t-1, so its predicted intermediate is nonsensical.
     Mitigation: tag spawn-this-frame Gaussians and freeze them at
     alpha=0 position only.
   - Non-linear motion — projectile arcs, accelerating vehicles. Linear
     warp magnitude × alpha overshoots/undershoots. No fix in v1; quality
     degrades gracefully with alpha.
2. Write up findings in `docs/superpowers/gaussian-frame-extrapolation.md`
   §Failure modes.
3. Implement the spawn-this-frame freeze in
   `FrameExtrapolator.extrapolate(...)` if the Sprint 5 canvas exposes a
   spawn-flag tensor (Sprint 5 T5.5 is the upstream task — coordinate).

**Verify:** Docs updated. If spawn-freeze is implemented, a new test in
`test_extrapolation.py` asserts spawn-flagged Gaussians do not move
under any alpha.

### T6.6 — Sprint integration test (Day 5)

**Goal:** Full pipeline runs end-to-end in CI: Cyberpunk capture →
Sprint 5 canvas → Sprint 6 extrapolator → rendered intermediate frame.

**Steps:**

1. `tests/gaussian/test_extrapolation_canvas_integration.py`:
   - Load a 10-frame Cyberpunk capture stub (committed under
     `tests/gaussian/data/` if size permits, else fetched on demand).
   - Initialise the real `PersistentCanvas`, run 10 frames, request an
     intermediate at alpha=0.5 between frames 5 and 6, assert PSNR
     against the real frame 5.5 (interpolated reference) > 30 dB.
2. Mark `@pytest.mark.cuda`; CPU runs skip.
3. Add `pytest -m "not cuda"` invocation to local-dev README.

**Verify:** Test passes on 3080 Ti CI runner; skipped cleanly on
CPU-only macOS.

### T6.7 — Sprint 6 code review checkpoint (Day 5)

**Goal:** Run the cross-cutting code review pipeline on Sprint 6
commits before Sprint 7 starts.

**Steps:**

1. `python -m oss.gaussian.review.run --sprint 6 --commit-range <base>..HEAD`.
2. Review artifacts saved under `oss/gaussian/review/artifacts/sprint-6/`.
3. Judge verdict APPROVE → mark Sprint 6 complete, proceed to Sprint 7.
4. REQUEST_CHANGES → iterate.
5. BLOCK → escalate to user.

**Verify:** Judge verdict file exists and is APPROVE.

---

## Risks + mitigations

1. **Sprint 5 slips.** Sprint 6 scaffold lands defensively against the
   API contract; T6.2–T6.6 block on Sprint 5 merging but T6.1 is unblocked.
2. **Non-linear motion at alpha=1 looks bad.** Expected and documented
   in T6.5. Mitigation: cadences default to alpha ≤ 0.5 (60→120). 60→144
   only enabled for users who opt in.
3. **DLSS-FG comparison shows OSS worse.** Acceptable per spec §11 risk 5.
   Sprint 6 reports honest numbers; graduation criterion (spec §5)
   absorbs the verdict.
4. **CUDA path-only quality regression.** All quality numbers must be
   reproducible on the Sprint 1 reference renderer for tiny inputs. If
   CUDA / reference diverge by > 1 dB PSNR, root-cause before sign-off.
5. **CI without GPU.** All `@pytest.mark.cuda` tests skip on CPU; T6.1
   tests use the synthetic warp and run everywhere.
