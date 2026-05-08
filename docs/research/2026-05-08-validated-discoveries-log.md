# Validated Discoveries Log — OSS Project

**Date:** 2026-05-08
**Purpose:** Single source of truth for what OSS has actually MEASURED + VALIDATED, vs what is hypothesized. Not a hype document — strict criterion: data exists on disk, reproducibility script works, claim has bounded error or measured improvement.

For forward-looking hypotheses (untested claims from council), see `docs/research/hypotheses/H001-H005.md`.

---

## D-Series — Validated discoveries (data on disk, reproducible)

### D1 — v6.1 stippling artifact: λ=2px checkerboard from spawner integer-pixel bias
- **What:** v6.1-pico-001 produces a regular dotted/mosaic pattern across foliage / dark walls. FFT of `v6.1 − GT` residual at step 11000 shows peaks at λx=2px (mag 177,265) and λy=2px (mag 138,551). Mean abs error 9.09/255 (~3.6%).
- **Mechanism:** spawner regresses Δxy with mean fractional bias (0.67, 0.25), creating integer-pixel-aligned Gaussians → per-pixel checkerboard.
- **Memo:** `docs/superpowers/experiments/2026-05-08-v6.1-stippling-artifact-detection.md`
- **Status:** characterized + measured. Fix landed in pico-001 mid-flight (subpixel jitter, commit `221a325`); structural fix (disocclusion-pixel-center spawn) goes in pico-002.
- **Publishable:** YES — workshop paper class. Specific failure mode of integer-pixel-aligned Gaussian SR spawners with measured FFT signature + structural fix.

### D2 — v6.0 16-pixel grid artifact + architectural fix
- **What:** v6.0 produced a 16-pixel periodic grid artifact (different from D1's 2px pattern).
- **Fix:** randomized per-frame spawner tile offsets + overlapping rasterizer tiles with cosine feathering.
- **Memo:** `docs/superpowers/experiments/2026-05-07-v6.1-pico-grid-artifact-architectural-fix.md`
- **Status:** validated working. Different mechanism than D1 (tile-period vs sub-tile-period).

### D3 — v5-pixel-temporal completed baseline validation
- **What:** v5 pixel-temporal architecture trained to completion (80k steps), held-out evaluation finished 2026-05-06. Beats baselines.
- **Data:** `docs/superpowers/experiments/2026-05-06-v5-pixel-temporal-final-held-out-eval-results.json`, memo `2026-05-06-v5-pixel-temporal-final-held-out-eval.md`
- **Status:** baseline established. v6 architecture builds on this.

### D4 — v6.2 architecture runs end-to-end
- **What:** Per `RESEARCH.md` and commit `732166a`: HAT backbone → MV+covariance canvas warp → keyframe active mask → cross-attention pixel↔Gaussian fusion → V6Rasterizer → composite head → spawner writeback. 253 v6 tests pass.
- **Status:** structural property verified. Frame extrapolation via canvas warp (H005) is partial-validated at architecture level — not yet benchmarked vs DLSS-FG / FSR-FG.

### D5 — Pade [4/4] approximation of `expf(-q/2)` over q∈[0, 9]
- **What:** Diagonal Pade form, 4 multiply-add steps each for numerator + denominator + one reciprocal.
- **Coefficients:** num `[1, 1/2, 3/28, 1/84, 1/1680]`, den `[1, −1/2, 3/28, −1/84, 1/1680]`
- **Measured max abs error:** `5.83e-4` (worst at q=9.0). Below kernel atol=1e-5? No — barely above 1e-3 ship bar. Acceptable for non-critical paths.
- **Memo:** `docs/superpowers/experiments/2026-05-08-phase4-elegance-a.md`
- **Status:** mathematically validated. Ship-with-flag candidate for replacing SFU `expf` calls.

### D6 — Axis-aligned/isotropic Gaussian separable factoring is bit-exact
- **What:** At `rot=0`, `q = dx²/sx² + dy²/sy²`, so `exp(−q/2) = exp(−dx²/(2sx²)) · exp(−dy²/(2sy²))` factors exactly.
- **Cross-term bound at small rotation:** `|Δw| ≤ 9·|sin θ cos θ|·|sx/sy − sy/sx|`. At sx/sy=2, threshold rotation = 7.4e-7 rad.
- **Memo:** `docs/superpowers/experiments/2026-05-08-phase4-elegance-b.md`
- **Status:** validated. Ship axis-aligned/isotropic FAST path; reject broad small-rotation approximation until checkpoint rotation histograms exist.

### D7 — 256-entry LUT for `expf(-q/2)` over q∈[0, 9]
- **What:** LUT with linear interpolation, 512 bytes as fp16, Δq = 0.0353.
- **Measured max abs error:** `3.86e-5` — **below bf16/fp16 accumulation noise**.
- **Memo:** `docs/superpowers/experiments/2026-05-08-phase4-elegance-c.md`
- **Status:** validated. Ship if LUT resident in constant/shared memory. **Strong candidate** for narrow-Gaussian fast path.

### D8 — 3σ cull radius validated as the right default
- **What:** Mass loss at 2σ cull = 13.534%, at 2.5σ = 4.394%, at 3σ = 1.111%, at √12σ = 0.247%.
- **L∞ feature contribution at feat=3:** 2σ = 0.406, 2.5σ = 0.132, 3σ = 0.033.
- **Memo:** `docs/superpowers/experiments/2026-05-08-phase4-elegance-d.md`
- **Status:** **REJECTED 2σ** as default; 3σ is correct; 2.5σ is flag-only quality knob.

### D9 — `q > 12` far-field skip bound
- **What:** `exp(-6) = 2.48e-3`, per-Gaussian L∞ at feat=3 = 7.4e-3.
- **Memo:** `docs/superpowers/experiments/2026-05-08-phase4-elegance-e.md`
- **Status:** mathematically bounded. Not bit-exact but acceptable for mass-loss-bounded skip. Mostly redundant with 3σ AABB which corresponds to q=9.

### D10 — Quantized Gaussian state bounds (xy int16, rot uint8, scale fp16)
- **What:** xy int16 over 1920px → 0.0586 px step (0.029 px half-ULP). rot uint8 over 2π → 1.4° step. fp16 scale relative precision = 9.77e-4.
- **Caveat:** screen-wide fp16 xy NOT equivalent to int16 fixed-point. At 4K, fp16 pixel centers can exceed 1 px half-ULP near far edge. Prefer fp32 xy or tile-local fixed-point.
- **Memo:** `docs/superpowers/experiments/2026-05-08-phase4-elegance-k.md`
- **Status:** xy int16 + fp16 scale plausible for inference state; int8 rotation = flag-only until anisotropy bounds known.

### D11 — Redundant-computation audit (12 specific findings, line-cited)
- **What:** Static audit of `oss/sr/v6/`, `oss/cuda/src/`, `oss/gaussian/renderer/` found 12 specific redundancies:
  - M1: `topk_norm` accepted but discarded at `rasterizer_fwd.cu:463`
  - M2: Forward writes unused `aabb`/`pair_count` at `:505-517`
  - M3: Deterministic `gid_sorted`/`tile_offsets` materialized unnecessarily at `:525-537`
  - M4: `out` + `tile_offsets` zeroed before full overwrite at `:502`, `:527`
  - M5: Forward weight loop recomputes row-only pixel coords inside column loop at `:208-211`
  - M6: Backward recomputes `dx*dx`, `dx*dy`, `dy*dy` at `:84`, `:106-108`
  - M7: Forward repeats `c*c`, `s*s`, `c*s` at `:63-65` (backward already CSEs at `:140`)
  - M8: `d_rot` factorable at `:150-153`
  - M9: Canvas warp filters `in_frame` AFTER Jacobian/covariance work begins
  - M10-M12: Multiple cacheable constants recomputed (RoPE, tile centers, feather masks, Sobel)
- **Memo:** `docs/superpowers/experiments/2026-05-08-phase4-elegance-m.md`
- **Status:** **shippable engineering wins, line-cited**. Each is independently verifiable + fixable. Estimated cumulative gain: 5-15% kernel speedup.

### D12 — Plateau loss imbalance diagnosed in v6.1
- **What:** v6.1 training plateau attributed to specific loss-component imbalance.
- **Memo:** `docs/superpowers/experiments/2026-05-08-v6.1-plateau-loss-imbalance-diagnosis.md`
- **Status:** diagnosed; informs pico-002 loss schedule.

### D13 — Stippling artifact ARCHITECTURAL audit
- **What:** Architectural audit of v6.1 spawner identified the integer-pixel-bias mechanism (companion to D1's empirical FFT measurement).
- **Memo:** `docs/superpowers/experiments/2026-05-08-v6.1-stippling-artifact-architecture-audit.md`
- **Status:** architecture-level root cause confirmed. Fix path documented.

---

## Tier 3 — Deferred to frame-test (need real ckpts, not yet validated)

These elegance-audit questions REQUIRE held-out frame tests with trained checkpoints; closed-form math doesn't decide them:

- **F**: Top-K compositing — how many Gaussians per pixel for 99% mass? Synthetic fallback insufficient.
- **G**: Edge-only tile rendering — what fraction of LR tiles are edge-flagged? Synthetic fallback only.
- **H**: 2K→4K hierarchical pass — depends on learned decoder.
- **I**: Precomputed tile masks — needs consecutive-frame ckpt stats.
- **J**: Spatially varying budget — quality-coupled.
- **L**: Decoupled feature compression — information loss through learned head.

**Action:** these graduate to validation when pico-002 produces a usable checkpoint.

---

## What's NOT yet validated (claims still hypothetical)

- All H001-H005 (forward-looking novel formulae)
- "OSS beats DLSS at X ms" — no apples-to-apples integration yet
- "Frame-gen is essentially free" — partial (D4 confirms structural property; cost not benchmarked vs DLSS-FG)
- "L2-resident canvas hits target ms" — design target, not measured in shipping context
- "Student model can replace HAT-Tiny at <0.4M params" — needs HAT-Tiny actual ms measurement first

---

## Publication readiness summary

| Finding | Publishable today? | Venue |
|---------|-------------------|-------|
| D1 stippling artifact + fix | YES | Workshop paper / arxiv tech note |
| D5-D7 elegance math (Pade, separable, LUT) | YES | Tech note in arxiv preprint appendix |
| D8 cull radius validation | YES | Same as D5-D7 |
| D11 redundant-computation audit | YES (engineering note) | Repo blog post |
| Architecture preprint (H002, H005 + D4 system runs) | YES with caveats | arxiv preprint |
| Full DLSS comparison | NO | wait for pico-002 + integration |

---

## Update protocol

- New validated finding → add as next D-number, link the memo
- Hypothesis transitions `untested` → `validated` → reclassify as D-finding here, update hypothesis file's status
- Hypothesis transitions `untested` → `refuted` → keep in hypothesis files with refutation note; reference here in "What's NOT validated" section
- This log is the data backing for the dashboard's Hypothesis > Result > Lab Notes section
