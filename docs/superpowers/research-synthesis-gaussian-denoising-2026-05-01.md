# Research synthesis — Gaussian denoising / Ray Reconstruction replacement

**Date:** 2026-05-01
**Source:** External research batch on Gaussian-based denoising + path-tracing reconstruction.
**Status:** Strategic project pivot identified; action items queued; existing v1 scope unchanged.

This doc consolidates a third batch of external research into the OSS
project. Where the first two batches validated the *upscaling*
architecture (research-synthesis-2026-05-01.md), this batch shows the
**same architecture is a slot-in for DLSS Ray Reconstruction (DLSS 3.5)
denoising**.

---

## 1. The strategic insight

The Gaussian temporal canvas pipeline we are building for upscaling
(LR → Gaussians → HR) is structurally identical to "Neural Gaussian
Denoising" (NGD) for path tracing (noisy 1spp → Gaussians → clean
HR). Same renderer, same canvas, same extrapolation. Different input
distribution and different training data.

This is a **massive scope multiplier** at minimal architectural cost:

| | DLSS Super Resolution | DLSS 3.5 Ray Reconstruction |
|---|---|---|
| Replaced by | OSS-Gaussian (Sprints 1–7) | OSS-Gaussian + denoising input (post-v1) |
| Input | LR (denoised) frame + G-buffer | Noisy 1spp + G-buffer |
| Network | GaussianParamNetwork (Sprint 4) | Same — different training data |
| Renderer | Image-GS rasterizer (Sprint 1) | Same |
| Canvas | PersistentCanvas (Sprint 5) | Same |
| Frame gen | Sprint 6 warp | Same |

OSS-Gaussian is therefore not just "a vendor-agnostic DLSS-SR
replacement" — it's "a vendor-agnostic DLSS+DLSS-RR replacement using
one shared model family." That's a stronger story.

---

## 2. The four research angles + their fit to our architecture

### 2.1 G-Buffer Driven Anisotropic Splatting

**Idea:** use surface normal + depth gradient to define each Gaussian's
covariance, so kernels stretch along edges and stay thin across them.
Per-pixel anisotropic kernels physically aware of scene topology.

**Fit to our arch:** Sprint 4 enhancement. Currently
`oss/gaussian/network/prior_bank.py` has a **fixed 16-entry covariance
bank**; the network predicts a softmax distribution over those fixed
shapes. The next iteration: condition the bank weights on the
G-buffer's normal + depth gradient at the tile's center, so the bank
selection follows surface orientation. That gives us anisotropy with
no extra model parameters — just a tile-level conditioning input.

**Action:** Add to Sprint 4 OutputHead refactor list (already had
"quick wins" from the earlier paper synthesis; this is another).

### 2.2 World-Space Splat Accumulation

**Idea:** persistent cloud of Gaussians in **world** space, fused over
multiple frames; re-rasterize from current camera. Perfect temporal
stability via geometric coherence.

**Fit to our arch:** Sprint 5's PersistentCanvas is currently
**screen-space** (2D Gaussians, positions in pixel coords). World-space
3D Gaussians is a v2 lift — bigger architectural change. The
research-validated approach for v1 is screen-space (matches Image-GS,
GS-STVSR, GSASR). World-space is the moonshot for v2.

**Action:** Document as v2 research direction; do not pull into v1 scope.

### 2.3 Neural Lifting for Monte Carlo Noise (the slot-in for RR)

**Idea:** lightweight network maps noisy 1spp → Gaussian parameter map
(Δμ, Σ, α, color) → splat → clean HR. Replaces the CNN denoiser slot
in the post-processing stack.

**Fit to our arch:** **This is exactly Sprint 4 + Sprint 1 with a
different training distribution.** The architecture stays the same;
only the dataset and conditioning change:
- Sprint 4 (current): trained on (LR, G-buffer) → HR pairs
- Sprint 4 (extension): same network, also trained on (noisy 1spp,
  G-buffer) → reference pairs

**Action:** Treat as a v1 stretch goal / v2 milestone. Train a separate
checkpoint of the Sprint 4 network on Monte Carlo noise data
(NoiseBase already has this — see existing OSSRG training pipeline)
and ship as `OSS Ray Retracing`. Same network class, different weights.

### 2.4 Multi-Scale Splat Pyramids

**Idea:** hierarchy of Gaussians at multiple scales — large Gaussians
capture low-freq lighting, small Gaussians capture high-freq detail.
Sort + cull "fireflies" by promoting variance up the pyramid.

**Fit to our arch:** v2 research direction. Our current single-scale
canvas works fine for upscaling; multi-scale is denoising-specific
(fireflies are a Monte Carlo noise artifact). Defer.

**Action:** Document; do not implement in v1.

### 2.5 Stochastic Ray-Traced 3DGS (the moonshot)

**Idea:** scene IS a 3D Gaussian cloud; rays bounce off Gaussians;
unbiased Monte Carlo estimator on the Gaussian density itself. No
separate denoiser — noise resolves through stochastic sampling.

**Fit to our arch:** Completely different architecture. Out of scope
for v1; possibly v3+ research. Mention in roadmap.

---

## 3. Concrete action items added to the project

### Action 1: Anisotropic G-buffer-conditioned bank (Sprint 4 extension)

Add a quick-win to Sprint 4's OutputHead refactor list (already in
papers-2407.18046-2501.06838-2503.14171-synthesis.md):

> Pass a per-tile (normal, depth_gradient) feature into the bank-weight
> head so the softmax distribution is physically conditioned on the
> tile's surface orientation. ~1 day implementation, expected quality
> win on geometric edges.

### Action 2: OSS Ray Retracing as v1 stretch / v2 milestone

Add to design spec § 12 "Open questions" (or a new § 13 "Future"):

> OSS Ray Retracing — apply the same Sprint 4 network architecture +
> Sprint 1 renderer to Monte Carlo path-traced denoising. Drop-in slot
> for DLSS 3.5 Ray Reconstruction. Training data: NoiseBase (existing
> dataset, used by OSSRG pixel track).

### Action 3: README pitch tightens

Update README compatibility section:

> OSS-Gaussian targets the universal DLSS API surface. v1 ships a
> drop-in replacement for **DLSS Super Resolution**. The architecture
> extends naturally to **DLSS Ray Reconstruction** by retraining on
> Monte Carlo noise data — same renderer, same canvas, different
> conditioning. Roadmap goal: one model family covers both DLSS
> products.

### Action 4: OSSRG vs OSS Ray Retracing positioning

Document decision: **keep OSSRG (pixel-based denoiser) as a parallel
track** until OSS Ray Retracing is trained and benchmarked. If
OSS Ray Retracing beats OSSRG on quality + iso-latency, archive OSSRG
to `oss/legacy/`. If not, both ship.

---

## 4. What does *not* change in v1 scope

- All 7 sprints proceed as planned. No scope additions to v1.
- Sprint 4's training data stays Sintel + TartanAir + HyperSim + SRGD
  (clean LR → HR).
- The Sprint 4 close-out gate (iso-latency vs FSR/DLSS Quality) is
  unchanged.
- The graduation criterion is unchanged.

The denoising direction is a **post-v1 amplifier**, not a v1 scope
addition.

---

## 5. Citations to add

When the README References section gets written:

- "Neural Lifting for Monte Carlo Reconstruction" — search arxiv for
  recent 2025 publications under this title; canonical citation TBD.
- "Stochastic Ray-Traced 3DGS" — early 2026 papers per external review;
  citation TBD.
- DLSS 3.5 Ray Reconstruction (NVIDIA blog post + technical overview).
- NVIDIA OptiX / OIDN — the denoiser stack we're competing with.
