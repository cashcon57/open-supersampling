# SR-CNN Beats V0.5 and GSASR — Training Distribution Is the Unlock
**Date:** 2026-05-02
**Status:** complete
**Predecessors:** `2026-05-02-v05-pixel-residual-success.md`, `2026-05-02-splats-cannot-SR-definitive.md`, `2026-05-02-splats-SR-literature-delta.md`, `2026-05-02-gsasr-on-engine-aliased-lr.md`
**Hardware:** RTX 3080 Ti 12 GB
**Code commits:** `2d12c7d` (SR upsample init fix), `768fe91` (SR held-out probe), `68337e9` (GSASR memo)

The day's investigation closes with a single, simple architectural picture. We have an answer to the user's "are we sure splats are useless" question, and we have a v0 ship.

## TL;DR

| Architecture | Train distribution | ActionRPG | Held-out (3 scenes mean) | Beats |
|---|---|---:|---:|------|
| Bicubic | n/a | 29.20 | 25.55 | (baseline) |
| V0.5 (splat + 12K residual CNN) | engine-aliased SRGD | +1.47 dB | +1.39 dB | 56/56 |
| **SR-CNN simple (this work)** | **engine-aliased SRGD** | **+2.71 dB** | **+2.77 dB** | **64/64** |
| GSASR (released, EDSR backbone) | DIV2K bicubic-clean | n/a | −0.04 dB | 0/24 |
| GSASR (released) | DIV2K bicubic-clean (sanity) | n/a | −3.70 dB on bicubic-clean | n/a |

**Two pieces of one story:**

1. **SR-CNN is strictly better than V0.5.** Same data, same trainer, no splat path — and it wins by 1.0–2.0 dB extra margin across train + 3 held-out scenes. The splat path was net-negative for SR. The "splats are decorative" finding was a polite way of saying "splats were producing noise that the residual CNN had to learn around."
2. **GSASR (a published Gaussian-SR success) loses to bicubic on our LR by −0.04 dB.** It is not architecturally inferior to bicubic — its release benchmarks beat bicubic by ~1–3 dB on standard SR tests. It loses because it was trained on bicubic-clean LR (DIV2K) and our engine-aliased LR is out-of-distribution for it.

**The binding constraint is the training distribution, not the architecture.** Any model trained on engine-aliased LR — splats, CNN, transformer — appears to beat any model trained on bicubic-clean LR, on engine-aliased eval. This matches Real-ESRGAN §3.1's bicubic-LR-trap warning, scaled up.

## What we did today, in chronological order

1. Validated pipeline with pico tier (12 dB plateau).
2. Discovered + fixed dead-init bugs (K-symmetry breaking + log_scale=log(8) for gsplat backward).
3. Trained V0.5 (splat + 12K residual CNN). Beat bicubic +1.5 dB on training scene, +1.4 dB held-out.
4. **Splat-contribution probe:** `zero+residual` produced bit-identical output to `splat+residual`. Splats explicitly ignored by the trained CNN.
5. **Triple-checked splat-SR uselessness** across 5 paths (incl. direct 50K-Gaussian Image-GS optim losing −3.59 dB).
6. **Literature delta:** GSASR / GS-STVSR / GaussianSR all use Gaussians as HR FEATURES, not RGB — different thesis from ours. Architectural ranking favoured running the released GSASR before reimplementing.
7. **Built `oss/sr/`** as a clean CNN super-resolver. SRCNNSimple = head_conv + N res blocks + PixelShuffle 2× + bicubic skip. SRRRDB stretch backbone. Trainer gains `--model {gaussian, sr_cnn, sr_rrdb}` dispatch with full backward-compat for the Gaussian (RR) track.
8. **First SR-CNN run failed at 12 dB**, same plateau as splats. Diagnosed as Kaiming-normal init on `upsample_conv` producing residual std ~0.5, which when combined with `clamp(0, 1)` zeroed half the output and killed clamp gradients. Fixed with `N(0, 0.01)` weight init — residual stays in `[-0.01, 0.01]` at step 0, output ≈ bicubic, gradient flows.
9. **SR-CNN re-run after fix:** ActionRPG +2.71 dB at step 1000. CitySample +2.40, StylizedRendering +1.87, ArchVizInterior +4.05 — **64/64 samples beat bicubic**.
10. **GSASR run on our LR (parallel agent):** −0.04 dB on engine-aliased LR (0/24 beats), −3.70 dB on bicubic-clean (expected bicubic-LR-trap). Architecture not the bottleneck. Training distribution is.

## Decisions

1. **SR-CNN replaces V0.5 as the v0 deliverable.** Same trainer, same data adapters, same held-out probe. The splat path was actively hurting performance; CNN-only is cleaner and better.
2. **The Gaussian-as-features thesis remains untested at our LR distribution.** GSASR's pretrained weights don't transfer because they were trained on bicubic-clean. To test the thesis fairly we would need to retrain GSASR on engine-aliased LR — that's 2–4 weeks of work, not the cheap GSASR-inference test we hoped for.
3. **The "splats can't do SR" finding is now precise:**
   - **Falsified for our pipeline:** ✅ Gaussians-as-RGB at our resource budget cannot beat bicubic.
   - **Not falsified in general:** ❓ Gaussians-as-features (GSASR thesis) could work but would require training on engine-aliased data.
   - **Practically irrelevant for v0:** SR-CNN already wins.
4. **Sprint 5 (canvas) and Sprint 6 (extrapolation)** in the SR context are not just dead, they're unmotivated. The CNN-only model has no temporal state to warp.
5. **Multi-day SR-CNN production run is unblocked.** Standard tier (~306K params) on full multi-scene SRGD GameEngineData (18K samples) over 24–72 hours.

## Open questions

1. Does the standard-tier SR-CNN at 24–72 hours of multi-scene training push past +5 dB margin? Likely yes given the lite-tier numbers.
2. Does the RRDB backbone buy meaningful additional dB at the same training budget? Cheap to A/B once SR-CNN baseline is locked.
3. Does generalisation hold from synthesised engine-aliased LR to *real* game-engine LR (the OSSContribute capture data we don't yet have)? The engine-aliased synth was hand-tuned; real game LR may have different artifacts (TAA-specific patterns, screen-space effects, post-process bloom). Untestable until OSSContribute ships.
4. Would training GSASR's architecture on engine-aliased LR beat SR-CNN? Probably yes (it's a much bigger network), but that's a 2–4 week research investment vs. shipping SR-CNN now.

## What survives from the Gaussian work

The `oss/gaussian/` modules (renderer, network, output_head, prior_bank) are not deleted. They remain available for the OSS Ray Retracing (denoising) track per `docs/superpowers/oss-ray-retracing-track.md`, where the splat representation is known to fit (D1 result on synthetic noise beat OIDN PSNR).

The Sprint 4 trainer infrastructure — DataLoader path, engine-aliased LR synth, bicubic comparison, held-out probe, lab notebook discipline — was built once and serves both tracks unchanged.

## 2026-05-02 (later) — System-level reframe

After this memo landed, a system-level framing came up that none of the per-component analysis above considered. Restating the relevant numbers honestly:

- **GSASR margin of −0.04 dB on engine-aliased LR** is not statistically meaningful. With per-frame PSNR variance of ~1–2 dB, the standard error on a 24-sample mean is ~±0.4 dB — the observed margin is ≈10× smaller than the noise floor. The "0/24 wins" stat is the actually-load-bearing finding, but even that only says "GSASR sits consistently slightly below bicubic on this distribution," not "GSASR's architecture is incapable."
- **The strong claim "architecture is not the bottleneck"** in the GSASR memo is therefore overstated. We have evidence that GSASR's *pretrained-on-bicubic-clean weights* don't transfer to engine-aliased LR. We do *not* have evidence that GSASR's *architecture, retrained on engine-aliased LR*, would lose to SR-CNN.

**The system-level argument we missed:**

If splat-based SR achieves *parity* (or near-parity) with bicubic on single-image SR, but *also* makes frame extrapolation essentially free via Gaussian-position warping along motion vectors, then a slightly worse SR + much cheaper extrapolation could be a better total system than a much better SR + a separate CNN extrapolator.

| System | SR quality | Extrapolation cost | Extrapolation quality |
|--------|-----------|--------------------:|-----------------------|
| SR-CNN + (separate CNN extrapolator, TBD) | +2.7 dB vs bicubic | ~1 ms (separate model fwd) | learned, can ghost |
| GSASR-style + Gaussian warp | ≈ bicubic (parity) | ~tens of µs (just rasterize) | structurally coherent |
| GSASR-style + small CNN decoder + Gaussian warp | ?? (untested) | ~tens of µs | structurally coherent |

We have not measured rows 2 or 3. We measured *only* per-component SR quality and concluded splats were dead — but Sprint 5 (canvas) and Sprint 6 (extrapolation) were the entire reason to use splats in the first place. Killing them on a per-component SR result alone is the wrong frame.

## Updated decisions

1. **SR-CNN ships as v0** — that part stays. We have a working CNN super-resolver; ship it as the immediate deliverable.
2. **Sprint 5 (canvas) and Sprint 6 (extrapolation) are NOT cancelled.** They were the original motivation for the splat path; the per-component SR result alone does not justify killing them.
3. **The next decisive test is on extrapolation, not SR.** Take GSASR's pretrained weights → predict Gaussians for frame N → warp Gaussian centres along motion vectors → rasterize at frame N+1 → compare to ground truth. Compare to the trivial bicubic-warp baseline (bicubic upsample of frame N + flow warp). If the Gaussian-warp extrapolation beats the bicubic-warp extrapolation by a clear margin, the system-level case for splats is alive even if their per-frame SR is at parity. ~1–2 days to set up.
4. **The "splats are dead for SR" framing is too strong.** The accurate version: "splats-as-direct-RGB at our scale do not beat bicubic on per-frame SR, AND we have not yet measured whether they unlock cheap extrapolation that compensates."

This entry corrects the over-conclusion in this memo and the GSASR memo. The original numbers stand; the strong qualitative framing is what's been walked back.
