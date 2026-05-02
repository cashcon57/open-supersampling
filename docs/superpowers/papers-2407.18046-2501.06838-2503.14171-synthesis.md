# Paper synthesis — informing the Sprint 4 OutputHead

**Date:** 2026-05-01
**Scope:** Three papers most adjacent to the OSS-Gaussian Sprint 4 param network.
**Target file:** `oss/gaussian/network/output_head.py` (`OutputHead.decode`).
**Sibling files referenced:** `oss/gaussian/network/param_net.py`,
`oss/gaussian/network/prior_bank.py`, `oss/gaussian/renderer/rasterizer.py`.

> **Access notes.** Full PDFs of 2503.14171 and 2407.18046 hit a 10 MB
> WebFetch ceiling and a binary-decode failure respectively, so the
> details below were sourced from the ar5iv HTML mirrors plus the arXiv
> abstract pages. GSASR (2501.06838) PDF parsed successfully. Section /
> page numbers below are from those rendered HTML versions; verify
> against the canonical PDF before quoting in the README.

---

## 1. Per-paper summaries

### 2503.14171 — Lightweight Gradient-Aware Upscaling of 3DGS

Renders a 3D Gaussian scene at low resolution then upscales using
**bicubic spline interpolation whose coefficients are derived from
analytical Gaussian gradients** rather than finite differences.
Computes ∂I/∂x, ∂I/∂y, ∂²I/∂x∂y per LR pixel directly from the rendered
Gaussians — the LR render already has these gradients because Gaussian
splatting is differentiable. The 4×4 spline coefficient matrix `A` is
solved per pixel via `A = C⁻¹ F (Cᵀ)⁻¹` (Section 5, "Image Gradient
Based Upscaling", Eq. 6–7).

**Numbers (Table 1 / Table 2, MipNeRF360, ×4):** 26.85 dB PSNR / 0.809
SSIM / 0.322 LPIPS. End-to-end at 4K: 15.6 ms render + 1.8 ms upscale,
4.2× faster than full-res 3DGS (72.4 ms). Scale factors evaluated 2× /
3× / 4× / 8×.

### 2407.18046 — GaussianSR

A learned arbitrary-scale image super-resolver that represents each
LR pixel as a continuous 2D Gaussian field. Architecture: encoder →
**Selective Gaussian Splatting (SGS)** classifier with a vocabulary of
**100 fixed kernel classes** assigned per-pixel via Gumbel-softmax
during training, hard argmax at inference → 2D Gaussian rendering at
the query coordinate → decoder. Per-Gaussian params: μ ∈ ℝ², covariance
Σ (σx, σy, ρ), opacity σ(ξ), feature vector v ∈ ℝᵏ. Uses **dual-stream
feature decoupling**: 8 channels go through Gaussian rendering, the
rest take a cheap bicubic path.

**Numbers:** 29.03 dB PSNR on DIV2K ×4 (vs LIIF 29.00 dB) with fewer
parameters, 41.70 ms at 256×256 vs LIIF 43.18 ms.

### 2501.06838 — GSASR (Gaussian Splatting Arbitrary-Scale SR)

Same family as GaussianSR but **feed-forward** prediction of
image-conditioned 2D Gaussians plus a **sampling-density vector** that
encodes the target scale, letting one trained model serve any scale
factor without retraining. Custom CUDA scale-aware rasterizer adjusts
Gaussian footprints to the target resolution. Per-Gaussian params:
position, full 2D covariance, opacity, RGB.

**Numbers (Method §3, Experiments §4):** ~32–35 dB PSNR @ ×2,
~28–31 dB @ ×4, ~24–28 dB @ ×8 on standard SR benchmarks, ~2.5 M
parameters total, 35–60 FPS at the lower scales on consumer GPUs.

---

## 2. What our OutputHead does today

`OutputHead.decode` (`output_head.py:86–155`) consumes the raw
`(B, K·per_g, Hₜ, Wₜ)` tensor from `GaussianParamNetwork`
(`param_net.py:175–202`) and emits a `DecodedParams` dataclass plus a
`GaussianBatch` for the renderer. Per Gaussian we predict:

- Δμx, Δμy with `tanh(·)·tile_size` envelope → centre stays within
  ±1 tile of its tile centre.
- `log_scale` clamped to ±ln(8) → multiplicative factor ∈ [1/8, 8] on
  the bank-selected (sx, sy).
- `rot_offset` clamped via `tanh(·)·π/4`, added to the bank-selected θ.
- `bank_logits[bank_size]` softmaxed and fed through `CovariancePriorBank`
  (`prior_bank.py:170–225`). The bank stores 16 hand-picked (sx, sy, θ)
  shapes; effective sx/sy is the geometric mean over weights, θ is a
  circular weighted mean.
- 3 colour channels with `sigmoid` (LDR) or `softplus` (HDR) activation.

The `GaussianBatch` interface (`rasterizer.py:45–90`) is **scale + rot,
not full covariance**, RGB feature, no opacity (it is implicit in the
top-K accumulator + `topk_norm`). `tile_size = 16` is hardcoded by the
upstream Image-GS CUDA kernel.

Our head therefore has no opacity prediction, no view-dependent
features, no scale-conditioning, no SR-style learned upsampling — the
LR→HR multiplier is implicit in the renderer's output resolution.

---

## 3. Recommendations, ordered by impact

### Quick wins (≤1 day)

| # | Change | Where | Paper | Expected effect |
|---|---|---|---|---|
| QW-1 | Add a per-Gaussian **opacity logit** channel; multiply into `feat` after `sigmoid`. Keeps `GaussianBatch` schema; rasterizer's top-K already does softmax-style normalisation but explicit α improves edge fidelity. | `param_net.py::_NON_BANK_CHANNELS` (4→5), `output_head.py::decode` slice indices, default init bias = +2 so initial α≈0.88 ⇒ training is stable. | GaussianSR §3.2; GSASR §3 (both list opacity as separate from colour). | +0.2–0.5 dB PSNR on thin-line content; lets the network kill Gaussians it doesn't need. |
| QW-2 | Add a **scale-conditioning input** to the head: a single scalar `s = HR_h / LR_h` broadcast as an extra channel onto `d1` before `tile_proj`. | New arg to `GaussianParamNetwork.forward`; concat onto decoder output. | GSASR §3.1 sampling-density vector. | Lets one checkpoint serve ×2/×3/×4 with no retraining. |
| QW-3 | **Per-tile log-scale prior** = log(target scale) instead of 0, so default Gaussian footprint matches the upsampling factor at step 0. | `output_head.py::decode` add `+ math.log(scale_factor_target)` before clamp. | 2503.14171 §5 (gradient-aware spline footprint should match the upsample ratio). | Faster convergence, fewer dead-Gaussian artifacts in early training. |
| QW-4 | **Centre offset envelope = 1.5 · tile_size** (not 1.0). 2503.14171 uses a 4×4 neighbourhood; centring strictly within ±1 tile starves boundary tiles of contributors. | `output_head.py:121–122` constant. | 2503.14171 §5, Fig. 3. | Reduces tile-boundary seams in the output. |
| QW-5 | Log the **bank-weight entropy** and **softmax temperature** as a metric. GaussianSR uses Gumbel-softmax with a temperature schedule (§3.2). Our softmax is plain. | Diagnostic only; no behaviour change. | GaussianSR §3.2. | Gives us a knob if we see bank-weight collapse during T4.4. |

### Bigger refactors (1–3 days)

| # | Change | Where | Paper | Expected effect |
|---|---|---|---|---|
| BR-1 | **Gumbel-softmax over the bank** during training, hard argmax at inference. Drop the geometric-mean blending in `CovariancePriorBank.forward` and instead return a single bank entry — discrete selection means each Gaussian commits to one shape, killing the "blurry mean shape" failure mode. | `prior_bank.py::forward` add `discrete: bool` mode; `output_head.py` selects between soft (training) / hard (inference). | GaussianSR §3.2 SGS classifier. | Eliminates a class of degenerate covariances; expected +0.1–0.3 dB and visibly sharper highlights. |
| BR-2 | **Gradient-aware loss term**: render the HR target via the Sprint 1 rasterizer, compare its analytical gradient (∂I/∂x, ∂I/∂y) to the GT image's Sobel gradient. Add as a 0.1× weight on top of L1+SSIM. | New `oss/gaussian/training/losses.py` (Sprint 4.4); not the OutputHead itself but it consumes our outputs. | 2503.14171 §5 — they show analytical Gaussian gradients are cheap and well-defined. | Sharper edges; matches the "gradient-aware" supervision the closest paper relies on. |
| BR-3 | **Decouple "wide" and "narrow" channel paths**: 3 colour channels go through Gaussian rasterisation; reserve 5 extra feature channels rendered cheaper or via bicubic for high-frequency detail, à la GaussianSR's dual-stream. Requires `GaussianBatch.feat` to widen and a tiny CNN that fuses wide-feat → RGB at HR. | `rasterizer.py::GaussianBatch.feat` already supports F>3; add a fusion conv post-render in a new `oss/gaussian/network/fusion_head.py`. | GaussianSR §3.3 dual-stream feature decoupling. | Keeps Gaussian count low while gaining detail; ~30% perf saving at iso-quality. |
| BR-4 | **Scale-aware rasterisation hint**: pass HR/LR ratio into the renderer so per-Gaussian footprint is automatically widened at higher upsample factors. | `rasterizer.py::Rasterizer.__call__` add `scale_factor`; multiplies `inv_scale` accordingly. | GSASR §3.2 scale-aware rasterizer. | Required if QW-2 lands; otherwise QW-2 just shifts work to the network. |
| BR-5 | **Neighbour-tile contribution**: at decode time include K Gaussians from each of the 8 neighbour tiles when rendering a target tile (still per-tile parameter prediction, just looser locality). | `output_head.py::to_gaussian_batch` and the renderer call site. | 2503.14171 §5 4×4 neighbourhood Fig. 3. | Removes block artifacts at tile boundaries which Sprint 1 benches already flagged. |

### Future research bets (post-v1)

| # | Bet | Paper | Why post-v1 |
|---|---|---|---|
| FR-1 | Replace the fixed bank with a **learned codebook** initialised from the current 16 entries, updated with EMA during training (VQ-VAE style). | GaussianSR's 100-class classifier outperforms hand-picked priors §4. | Risks instability before we have stable training; defer until T4.10 ablation completes. |
| FR-2 | **Arbitrary-scale single checkpoint** via sampling-density conditioning across the entire UNet, not just the head. | GSASR §3.1. | Requires retraining infra for variable-resolution batches; Sprint 7 territory. |
| FR-3 | **Differentiable mixed-partial gradient supervision** (∂²I/∂x∂y) à la 2503.14171 Eq. 6 — they show this regularises the spline coefficients. | 2503.14171 §5 Eq. 6. | Mixed-partial PyTorch autograd is expensive; only worth it if BR-2 underperforms. |
| FR-4 | **Continuous query coordinates** (sub-pixel sampling) so the renderer can produce HR pixels at arbitrary float coordinates, not just integer grid positions. | GaussianSR §3.4. | Requires a renderer rewrite — gsplat is integer-grid only. |

---

## 4. Things we should NOT adopt

- **Per-scene Gaussian optimisation loops** (the 3DGS base of 2503.14171
  expects 30 k+ iterations of gradient descent per scene). We need
  feed-forward prediction at frame rate; never reintroduce a
  per-frame optimisation loop.
- **View-dependent Spherical Harmonics colour** (3DGS, 2503.14171 base).
  Game upscaling already has the shaded LR colour; SH adds parameters
  and zero quality for our setting.
- **High Gaussian counts via densification / pruning heuristics**
  (2503.14171 trains with 1–5 M Gaussians). Our budget is 1 k–15 k
  per frame across tiers — densification is a post-v1 conversation
  at most.
- **Gumbel-softmax at inference** (GaussianSR §3.2 uses it for soft
  labels in training only). Always argmax at runtime; the stochastic
  noise is not free quality.
- **GaussianSR's 100-class kernel classifier verbatim** — 100 entries
  for our 16 k-Gaussian budget means the softmax head dominates
  bandwidth. Our 16-entry bank is already at the right size for
  Pico/Standard tiers.
- **Per-pixel Gaussian rendering** (GaussianSR / GSASR render every HR
  pixel with its own Gaussian). We render *tiles* of K Gaussians
  because we have G-buffers and a CNN giving us spatial structure for
  free. Per-pixel would 256× our parameter count.
- **Full 2×2 covariance regression** (GSASR §3). Our (sx, sy, θ) is
  what gsplat consumes; emitting Σ then re-deriving (sx, sy, θ)
  would cost a Cholesky and gain nothing the bank doesn't already give us.

---

## 5. Citations (drop-in for the README References section)

```
[GaussianSR] D. Hu, X. Cao, T. Liu et al., "GaussianSR: High Fidelity
2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution,"
arXiv:2407.18046, July 2024. https://arxiv.org/abs/2407.18046

[GSASR] J. Liang, Y. Li, R. Timofte et al., "GSASR: 2D Gaussian
Splatting for Arbitrary-Scale Super-Resolution," arXiv:2501.06838,
January 2025. https://arxiv.org/abs/2501.06838

[GradAware3DGS] S. Niedermayr, J. Stumpfegger, R. Westermann,
"Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting,"
arXiv:2503.14171, March 2025. https://arxiv.org/abs/2503.14171
```

Author lists are best-effort from the arXiv landing pages; verify
spelling against the canonical metadata before publishing the README.
