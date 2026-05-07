# v6 Research Tracks Design — Sprint 5 Closeout Candidate

**Status: SUPERSEDED 2026-05-05.** Written before the 2026 Gaussian-temporal research synthesis (`docs/research/2026-05-05-gaussian-temporal-research-deep-dive.md`) was reckoned with. This memo framed v6 as a v5 race resolution into A/B scenarios; the actual v6 commits to **Gaussian-temporal as the architectural foundation** with HAT spatial backbone + cross-attention + covariance resampling + score-based active pruning + custom kernels per vendor + DLL-shim integration.

**Replaced by:** `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md`.

Memo retained for forensic value (decision context, scenario A/B framing).

---

**Author:** Claude (subagent), 2026-05-04.
**Branch:** `v0.2-dev`.
**Predecessors:**
- `docs/superpowers/specs/2026-05-04-v5-pixel-temporal-design.md`
- `docs/superpowers/specs/2026-05-04-v5-gaussian-temporal-design.md`
- `docs/superpowers/notes/2026-05-04-pico-distillation-design.md`
- `docs/superpowers/notes/2026-05-04-s7-game-integration-design.md`
- `docs/superpowers/notes/cuda-mega-kernel-design.md`
- `docs/superpowers/notes/vendor-optimization-audit.md`
- `docs/superpowers/notes/2026-05-04-v5-pixel-temporal-onnx-export-design.md`
- `docs/superpowers/oss-ray-retracing-track.md` (parked on NoiseBase)

---

## 0. Purpose & framing

Sprint 5 will close in one of two states:

- **Outcome A (most likely a priori):** v5-pixel-temporal ships as the v0.2 production model. v5-gaussian-temporal becomes parked research input.
- **Outcome B:** v5-gaussian-temporal explicitly beats pixel on the agreed gates (≥ 0.4 dB PSNR, ≥ 0.01 SSIM, no LPIPS regression > 0.005, latency budget within 1.4× pixel). Gaussian ships; pixel becomes the production fallback and a comparison reference.

This document **must work in either world**. v6 is structured as three buckets:

1. **Common-to-both productization** — work that lands regardless of who wins.
2. **Scenario-conditional research tracks** — A vs B branch.
3. **Cross-cutting deliverables** — distillation, OSS-FX, OSS-RG, design freeze.

The default sprint length is 6 months (2026-06-01 → 2026-11-30). Calendar slips are expected; the bring-to-implementation gate at month 6 is hard.

---

## 1. Race resolution & loser disposition

### 1.1 Decision criteria (re-stated, normative)

The pixel-vs-gaussian race resolves on the held-out Sintel + UE5-replay eval set defined in the v5 specs. Concretely:

- **Quality gate:** Gaussian wins iff `PSNR_g − PSNR_p ≥ 0.4 dB` **and** `SSIM_g − SSIM_p ≥ 0.01` **and** `LPIPS_g − LPIPS_p ≤ 0.005`.
- **Latency gate:** Gaussian must fit `latency_g ≤ 1.4 × latency_p` on the 3080 Ti reference at 1440p→4K.
- **Stability gate:** temporal variance on the v5 Gaussian "≤ pixel-track variance" target must hold on the held-out eval; ties go to pixel.

Any other outcome → pixel ships.

### 1.2 Loser disposition

- The v5 race **loser is preserved as an open-sourced comparison reference**, not deleted. Concretely: ONNX export, FP16 weights, eval-table row, and a tagged release `v0.2-{pixel|gaussian}-loser-ref`.
- Rationale: the loser is a strict-superset signal for v6 research and an external credibility artifact for the project. Cost is small (~50 MB artifact + ~0.5 day of CI plumbing).
- This resolves one of the open questions below.

---

## 2. Common productization work (lands in either scenario)

### 2.1 OSS-FX α-conditioned frame extrapolation

`docs/superpowers/notes/2026-05-04-s7-game-integration-design.md` sketched an α-conditioned extrapolator that synthesizes one future frame from the most recent SR output plus motion vectors, conditioned on a confidence/blend scalar α ∈ [0,1].

v6 productizes this with the following commitments:

- **Architecture:** small (≤ 200K params) U-Net taking `(SR_t, MV_t→t+1, α)` → `Δframe_t+1`. Reuse pixel-temporal recurrent state if Outcome A; reuse Gaussian field if Outcome B (see §3 / §4 conditional bindings).
- **Training data:** UE5 + Sintel replays at 60→120 FPS pairs. Synthesize MV ground truth from rendered velocity buffers when available; else estimate via RAFT-Lite at training time (offline only — never at inference).
- **Latency target:** ≤ 1.5 ms additive on 3080 Ti at 1080p output (≈ 0.6× SR forward pass).
- **API surface:** single `oss_fx::predict_next(handle, alpha)` call; α exposed as a clamped float to game devs with documented semantics (0 = full extrapolation, 1 = pure repeat). Default α schedule per content type lives in a JSON table shipped with the runtime.

### 2.2 NoiseBase + OSS-RG (ray-reconstruction track)

`docs/superpowers/oss-ray-retracing-track.md` is parked on a NoiseBase data download. v6 unparks it as soon as the dataset materializes:

- **Trigger:** NoiseBase or equivalent (≥ 50K paired noisy/clean ray-traced sequences with G-buffers) lands on the lab NAS.
- **Work item:** OSS-RG = ray-reconstruction denoiser, sibling to OSS-SR. Same 1440p→4K spatial scale but consuming 1 spp ray-traced + G-buffers instead of rasterized history.
- **Owner gate:** if NoiseBase is still unavailable by month 3, defer OSS-RG to v7. Do not synthesize a stand-in dataset — that path was already eliminated in earlier sprints.

### 2.3 Pico distillation (Steam Deck tier)

Per `docs/superpowers/notes/2026-05-04-pico-distillation-design.md`:

- Distill the **shipping v5 model** (whichever wins) to a Pico tier.
- ~6h training run on a single 3080 Ti per the design memo.
- **Param target:** see open question §6.1; default to **~150K params (Pico)** unless v5-shipping forward latency at 1080p output on Steam Deck APU exceeds 8 ms, in which case fall back to a more aggressive 80–100K Pico-Mini variant.
- Deliverable: ONNX FP16 + INT8 PTQ, both gated through the existing Sprint 4 quality gate (≤ 0.5 dB PSNR delta vs FP32 teacher).

### 2.4 Design freeze + bring-to-implementation gate (month 6)

A hard month-6 deliverable: `docs/superpowers/specs/v0.3-bring-to-implementation.md` with frozen API surface, ABI, ONNX op-set, and game-integration ABI for OSS-SR + OSS-FX (+ OSS-RG if it shipped). v6 does not slip past this.

---

## 3. Scenario A — Pixel ships, Gaussian becomes research input

### A.1 Multi-frame transformer scaling on a frozen pixel backbone

The v5 Gaussian transformer was 4 layers / 549K params / history = 5. Scenario A asks: does scaling the transformer **on top of a frozen v5-pixel backbone** give a meaningful quality bump?

**Experiment design:**

- Freeze v5-pixel-temporal weights end-to-end. Treat pixel features at the penultimate decoder stage as a sequence of per-frame tokens (history = h, h ∈ {5, 10}).
- Train a transformer head with `L ∈ {4, 8}` layers, width 256, on top of those frozen features. Output is a residual added to the pixel-temporal output.
- Sweep `(L, h) ∈ {(4,5), (4,10), (8,5), (8,10)}`. Each cell ≈ 8h of training on the existing rig.
- **Success bar:** ≥ 0.2 dB PSNR over frozen pixel baseline on Sintel eval **and** ≤ 1.3× latency. Anything below is a negative result and we say so.

**Risk:** transformer-on-frozen-backbone often plateaus; budget the experiment to fail cheaply.

### A.2 3D Gaussians + view-dependent effects

The v5 Gaussian spec deferred 4D Gaussian Splatting and view-dependent (SH or learned) terms. v6 explores whether they pay rent on real game content.

- Add per-Gaussian `(SH_l=2)` color terms (9 params/channel/Gaussian, ≈ 27 floats).
- Add a per-Gaussian temporal-decay scalar `τ` for 4D extension. Field grows from ~700 KB → ~2.8 MB at FP32 / ~1.4 MB at FP16. Compression discussion in §4.B.2.
- **Eval:** specular-heavy UE5 replays (chrome, water, polished floors). Success bar = visible reduction in temporal hue-flicker on those clips, scored both by an LPIPS-VGG temporal-consistency proxy and by a 5-rater human eval.
- **Cost:** training overhead ≈ 1.6× v5-Gaussian. Defer if we can't afford the GPU days.

### A.3 Cross-attention between Gaussians and pixel features

v5 deferred. v6 designs the experiment.

- Architecture: per-frame, project the active Gaussian field's top-K (K ≈ 1024) Gaussians into a token set; cross-attend pixel-temporal feature tokens against them at the bottleneck.
- Training: warm-start both backbones from v5 weights, fine-tune with cross-attn module unfrozen first, then full unfreeze for the final 20% of steps.
- **Hypothesis:** Gaussians provide spatially sparse but temporally stable priors that the pixel pipeline currently lacks. Cross-attn lets pixel "ask" Gaussians for stable colors at disocclusion boundaries.
- **Success bar:** ≥ 0.15 dB PSNR on disocclusion-heavy held-out subset (camera-cut clips); no regression on stable clips.

---

## 4. Scenario B — Gaussian ships, pixel becomes fallback

### B.1 Gaussian temporal stability under aggressive motion

v5 success bar was "≤ pixel-track variance." Scenario B pushes the bar:

- New target: temporal variance ≤ 0.7 × v5-pixel variance on the "fast-camera" subset (camera angular velocity > 90°/s) of the UE5 replay set.
- Method: introduce an explicit Gaussian-momentum term — predict per-Gaussian `(Δμ, Δσ)` from MVs and accumulate across history rather than re-sampling. Effectively shifts the field from "re-evaluate each frame" to "advect, then correct."
- **Risk:** advection drift on long clips. Mitigation: hard clamp on `||Δμ||` per step + every-N-frame full re-sample.

### B.2 Gaussian compression for Steam Deck

v5 Gaussian field is ~700 KB at FP32 and the Steam Deck VRAM budget for SR is ~8 MB total (per the integration notes). v6 must shrink the field.

- **INT8 quantization** of `(μ, σ, color)` channels with per-channel scale. Expected ~4× shrink with quality loss bounded by the existing PTQ gate.
- **Top-K pruning** — drop Gaussians whose contribution norm falls below threshold τ. Sweep `K ∈ {2k, 4k, 8k}` (current avg field is ~16k Gaussians).
- **Combined target:** ≤ 200 KB field + ≤ 1.2× forward latency vs v5-Gaussian, no PSNR regression > 0.3 dB.

### B.3 Hybrid pixel + Gaussian (background/foreground split)

Use Gaussians for stable persistent background; pixel-temporal for fast-moving foreground. Practical because v5 keeps the pixel weights live as the documented fallback.

- Mask source: depth + MV-magnitude classifier, threshold sweep on training data. Soft mask α ∈ [0,1] blended.
- Two forward passes share the same input pyramid; outputs blended at the final upsample stage.
- **Success bar:** beats pure-Gaussian on dynamic-character UE5 clips (`character_chase_*`) by ≥ 0.3 dB without regressing static clips by > 0.1 dB.
- Doubles as a graceful-degradation story: if hybrid ships, the pixel branch is no longer "fallback" but a load-bearing component, which de-risks Steam Deck where Gaussian alone may be too heavy.

---

## 5. Sequencing — 6-month sprint plan

```
Month:   1     2     3     4     5     6
─────────────────────────────────────────────
Pico   |█████████|                              (§2.3 — distill v5 winner)
OSS-FX |██████████████|                         (§2.1 — α-extrapolation)
OSS-RG       |████████████████|                 (§2.2 — gated on NoiseBase)
Research      |█████████████████|               (§3 if A, §4 if B)
Freeze                              |█████|     (§2.4 — bring-to-impl gate)
```

- **Months 1–2:** Pico distillation runs in background on the 3080 Ti while OSS-FX prototype starts. Both are low-risk, high-leverage.
- **Months 1–3:** OSS-FX α-extrapolation matures to a productizable prototype.
- **Months 2–4:** NoiseBase OSS-RG track if data lands by month 3; otherwise defer.
- **Months 3–5:** Scenario-conditional research bucket (§3 or §4).
- **Month 6:** v0.3 design freeze + bring-to-implementation gate. Hard deadline.

Slip policy: §3/§4 research is the first thing to drop if we're behind. Pico, OSS-FX, and the freeze gate are not optional.

---

## 6. Open questions

### 6.1 Distillation target params: 150K (Pico) vs 600K (Lite) vs 2M (Heavy)?

**Recommendation:** ship Pico (~150K) for Steam Deck tier, Lite (~600K) for 1660-class, Heavy = the v5 winner unmodified. This is three tiers, not one. The Pico distillation memo already targets the small end; Lite is a free byproduct of the same teacher with a wider student. Heavy is just the shipped v5.

Open: whether to publish all three or just Pico + Heavy. Defaulting to all three pending integration team feedback.

### 6.2 Should the v5 race loser be open-sourced as a comparison reference?

**Decision (§1.2):** yes. Tagged release, ONNX + FP16 weights, eval-table row.

### 6.3 OSS-FX latency budget vs SR latency: how to expose to game devs?

Two options on the table:

- **(a)** OSS-FX is a separate pass with its own latency line item. Devs choose to enable it. Budget contract: ≤ 1.5 ms at 1080p.
- **(b)** OSS-FX is fused into the SR forward graph behind α; setting α = 1 fully bypasses it.

**Recommendation:** (a). Keep the surface separable. Game devs already understand frame-extrapolation (DLSS 3, FSR 3) as a distinct toggle, and (b) makes A/B testing harder and complicates the ONNX export.

### 6.4 Research-bucket overflow

If both §3 and §4 turn out partially relevant (e.g. pixel ships but the Gaussian compression work in §4.B.2 is still useful for the loser-reference release), do we run partial scenario-B work? **Provisional answer:** yes for compression, no for hybrid; revisit at month 3.

### 6.5 NoiseBase contingency

If NoiseBase is still gone by end of month 3, do we (a) defer OSS-RG to v7, (b) attempt a smaller substitute dataset (FALCOR Orca, ReSTIR clips), or (c) generate synthetic data from UE5 path tracer? **Default:** (a), per §2.2. (c) was already ruled out earlier; (b) revisit only if a credible substitute appears.

---

## 7. Non-goals for v6

Explicitly NOT in scope, to keep the sprint honest:

- Mobile (iOS/Android) ports — v7+.
- DX12/Vulkan mega-kernel productization — `docs/superpowers/notes/cuda-mega-kernel-design.md` stays a design memo through v6. Vendor optimization audit (`vendor-optimization-audit.md`) findings inform but do not gate v6.
- New training data pipelines beyond NoiseBase. UE5 + Sintel remain the canonical sets.
- Any ML-architecture rewrite below the level of "swap the head" or "add a module." Deep rewrites are v7.

---

## 8. Exit criteria for v6

v6 is "done" when all of:

1. Pico tier shipped with quality-gate pass.
2. OSS-FX shipped behind a stable API with documented α semantics.
3. OSS-RG shipped **or** explicitly deferred-with-rationale to v7.
4. At least one §3 or §4 research-track experiment landed with a written negative-or-positive result memo in `docs/superpowers/experiments/`.
5. v5 race loser open-sourced as a tagged comparison reference.
6. v0.3 bring-to-implementation spec frozen and reviewed.

Anything less and v6 rolls into v6.1 cleanup before v7 starts.
