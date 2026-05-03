# Literature Delta: Did We Falsify General Gaussian SR, or Only Our Implementation?

**Date:** 2026-05-02
**Status:** complete — definitive
**Predecessor:** `2026-05-02-splats-cannot-SR-definitive.md`
**Hardware:** literature review only (no training run)
**Code commit:** N/A — analysis memo

---

## Hypothesis

Our five-experiment result (`2026-05-02-splats-cannot-SR-definitive.md`) shows that 2D Gaussian splats plateau at 11–13 dB PSNR vs. ~29 dB bicubic on engine-aliased SRGD data, and top out at −3.59 dB vs. bicubic even with 50 000 directly-optimised Gaussians on Sintel. Yet three published papers (GSASR, GS-STVSR, GaussianSR) claim to succeed at Gaussian SR. The question is whether the published successes falsify our general conclusion or merely expose architectural differences our implementation could close.

**Null hypothesis (what we want to test):** the 2D Gaussian splat representation is structurally incapable of single-image super-resolution at competitive quality, regardless of implementation. Our plateau is a property of the representation, not of our architecture.

**Alternative hypothesis:** one or more specific architectural choices we made differ from the published successes in a way that explains the entire gap. If the alternative is true, one of those differences is the unlock.

---

## Sources

| Paper | arXiv | Status | Reproduced independently? |
|-------|-------|--------|--------------------------|
| GSASR — Generalizable Image SR via 2D Gaussian Splatting | 2501.06838 | Accepted ICCV 2025 | Code released (github.com/ChrisDud0257/GSASR); not independently reproduced by us |
| GS-STVSR — Continuous Spatial-Temporal Video SR via 2D Gaussian Evolution | 2604.18047 | Preprint (Apr 2026) | No code release at time of writing |
| GaussianSR — High Fidelity 2D Gaussian SR | 2407.18046 | Published 2024 | No independent reproduction found |

**Access notes:** Full HTML versions retrieved via arXiv mirror. GSASR PDF binary-decoded successfully for cross-check. GaussianSR arXiv HTML (v2) 404'd; retrieved from the versioned v1 HTML. GS-STVSR HTML retrieved from v1. Section numbers below are from those HTML versions; verify against canonical PDFs before citing.

**Bicubic-LR-trap flag:** All three papers train against **bicubic-downsampled LR**. Our training uses engine-aliased LR (Halton jitter + area downsample + TAA blur σ=1.5 + JPEG q=85). This is a critical baseline difference documented in `lr_synthesis.py` and `2026-05-01-validation-decision-memo.md` Decision 3. PSNR numbers are NOT directly comparable. Paper PSNR numbers reflect a degradation domain where bicubic is near-optimal as a baseline; ours does not.

---

## Structured Architectural Deltas

### Paper 1: GSASR (arXiv:2501.06838)

**What they claim:** feed-forward per-image prediction of 2D Gaussians for arbitrary-scale SR. ~32 dB at ×2, ~29–31 dB at ×4 on standard SR benchmarks (Set14, Urban100, DIV2K). Accepted ICCV 2025.

| Aspect | Ours | GSASR | Impact rating |
|--------|------|-------|---------------|
| **Gaussian count** | 1 000–7 200 (pico–standard tier); 50 000 in direct-fit test | N = m × H × W, m=16 Gaussians per LR pixel → ~16× the pixel count. For a 128×128 LR patch: ~262 144 Gaussians | HIGH — GSASR uses 20–250× more Gaussians than our standard tier at any given resolution |
| **Covariance representation** | Fixed 16-entry bank; network predicts softmax weights over bank; final (sx,sy,θ) is geometric/circular weighted mean | Fully free per-Gaussian: five decoupled MLP heads predict (μx, μy, σx, σy, ρ). No discrete bank. "No activation function applied to o" (§3.2) | MID — Our bank prevents degenerate covariances, but weighted-mean of discrete entries cannot represent arbitrary continuous shapes. The direct-fit test used no bank constraint and still lost; this delta alone is not the unlock. |
| **Position constraint** | tanh(·)·tile_size envelope: Gaussians stay within ±1 tile of their tile centre | Truly free: reference positions at equal intervals, offsets predicted with no activation function (§3.2) | MID — Our constraint prevents inter-tile movement. At 2x SR, a Gaussian at a tile boundary cannot cover pixels in the adjacent tile's contribution zone. Combined with the tile-grid output structure, this creates seams. |
| **Compositing operator** | `rasterize_gaussians_sum` — additive SUM with topk_norm. No per-Gaussian opacity | Alpha-weighted sum: each Gaussian has explicit opacity α ∈ [0,1]. Contribution: G(x,y) = α·c·f(x,y). Final pixel: I = Σ Gᵢ (Eq. 3–4, §3.2). **Not** Porter-Duff OVER — both use additive compositing, but GSASR has explicit α. | MID — Opacity gives the network an "erase" degree of freedom. Our top-K accumulator normalises implicitly, but a dedicated α channel lets the network learn to suppress Gaussians in confident regions and amplify them in uncertain ones. Worth testing. |
| **Network backbone** | Small 4-level UNet (channels 16–48 for standard); tile-stride pooling to tile resolution; ~500K params total | EDSR-baseline or RDN encoder (~1.5–5M params) feeding Gaussian embeddings; 6× Swin Transformer-style Gaussian Interaction Blocks with cross-attention (§3.2) | HIGH — GSASR's encoder is 3–10× larger and uses attention-based interaction between Gaussian embeddings. Our UNet has no cross-Gaussian attention; each tile's parameters are predicted independently. |
| **Color / feature** | 3-channel RGB per Gaussian, sigmoid-activated | 3-channel RGB per Gaussian, plus opacity. Same sigmoid-activated range. | LOW |
| **Scale conditioning** | None. The network produces fixed output at the LR→HR ratio baked into the training regime. | Sampling-density vector encodes the target scale; one trained model serves arbitrary scale factors (§3.1). Scale-aware CUDA rasterizer adjusts Gaussian footprints to target resolution. | MID for our SR use case. We train at fixed 2x; this delta does not explain the plateau. |
| **Loss function** | L1 + 0.1·(1 − SSIM). HDR tonemap, LPIPS, temporal not yet wired. | L1 loss only. Same simple supervision. | LOW |
| **Training dataset / LR degradation** | SRGD GameEngineData + engine-aliased LR (Halton jitter, area downsample, σ=1.5 TAA blur, JPEG q=85). Bicubic is NOT near-optimal on our LR. | DIV2K, 48×48 LR patches, **bicubic downsampling only** (§3.2). Adam, 500K iterations, 4×A100 | HIGH for baseline comparability. Paper's bicubic baseline is near-perfect for their degradation; our engine-aliased LR produces a degradation where bicubic is genuinely harder to beat. |
| **Output architecture** | Network → raw tile tensor → OutputHead decode → GaussianBatch → rasterizer → final image (no refiner) | Network → Gaussian params → custom scale-aware CUDA rasterizer → final image (no post-rasterizer refiner stated in paper) | LOW — both are rasterizer-terminal |

**GSASR-specific concern — bicubic-LR trap:** GSASR trains and evaluates exclusively on bicubic-clean LR. On such data, bicubic upsampling is the near-inverse of bicubic downsampling — effectively the lower bound. Any learned SR network should beat it. On our engine-aliased LR (TAA blurred, JPEG compressed), bicubic is a more competitive baseline because it is the only off-the-shelf method that handles the blur correctly. The published GSASR numbers do not tell us whether GSASR would beat bicubic on our degradation domain.

---

### Paper 2: GS-STVSR (arXiv:2604.18047)

**What they claim:** continuous spatial-temporal video SR via 2D Gaussian evolution across frames. 26.04 dB on Vid4 at ×4 spatial (no direct bicubic comparison published). 12.67M total parameters. 0.64s per pair at 1280×720.

| Aspect | Ours | GS-STVSR | Impact rating |
|--------|------|-----------|---------------|
| **Gaussian count** | 1 000–7 200 per frame | One Gaussian per LR pixel per frame (N = pixel count). For 720p ÷ 4 = 180×320 LR → ~57 600 Gaussians | HIGH — GS-STVSR uses ~8–57× more Gaussians than our standard tier. Even at a 1080p LR (never our case), we'd need 10× more Gaussians to match density. |
| **Covariance representation** | Fixed 16-entry bank, softmax-weighted blend | Covariance Prior Bank (CPB) — discrete vocabulary similar in concept. Weighted recombination via softmax. Covariance stability ≈0.99 between frames noted. | LOW — conceptually similar to our bank. GS-STVSR drew its bank inspiration from the same prior (GaussianSR §3.2). Bank itself is not the differentiator. |
| **Position constraint** | tanh·tile_size (Gaussians pinned ±1 tile from tile centre) | LR pixel centre + learned offset Δμ via "Adaptive Position Drifting." Offset window spatially varying {1,2,...,10} pixels; learned weight maps control maximum offset. | MID — GS-STVSR allows larger and data-dependent offsets than our ±1-tile fixed envelope. |
| **Compositing operator** | Additive SUM with topk_norm; no per-Gaussian opacity | Additive SUM: fc(x,y,t) = Σ Gᵢ(x,y,t). No per-Gaussian opacity (Eq. 2, §3.1) | LOW — same compositing class |
| **Network backbone** | Small 4-level UNet, ~500K params | RVRT feature extractor (~30M params) + SpyNet optical flow + CPB covariance module + lightweight convolutional position/color heads. Total: 12.67M | HIGH — GS-STVSR's backbone is a video transformer (RVRT), not a small image CNN. 12.67M params vs. our 500K standard. The backbone has temporal attention and pre-trained optical flow. |
| **Temporal information** | Canvas hint (3 channels; single-frame canvas warp, not trained yet) | Explicit temporal evolution: optical flow-guided Gaussian position warping between input frames, "linear motion assumption" for intermediate steps, covariance resampling alignment to prevent drift | HIGH for video SR specifically. Our single-frame-at-a-time inference has no temporal signal. This delta does not apply to our single-image SR use case but is relevant for future video mode. |
| **CNN refiner on top of Gaussian output** | V0.5: PixelResidualHead (12K-param CNN) added post-rasterizer | No post-rasterization CNN refiner mentioned (Gaussian rendering is final output, §3.1) | MID (relevant) — GS-STVSR does NOT use a CNN refiner. This makes their claimed result harder to attribute to a cheap CNN trick. But: they have ~10x more Gaussians and a 25x larger backbone. |
| **Loss function** | L1 + 0.1·(1 − SSIM) | L1 + frequency loss (weight 0.05). 600K iterations, batch 32, patches 256×256 | LOW — similar setup |
| **Training dataset / LR degradation** | SRGD engine-aliased LR | Adobe240 videos (133 videos, 720P). Spatial scale sampled uniformly [4,8] during training. LR degradation not explicitly defined; likely native resolution downsampling. | MID — video SR from high-fps camera footage vs. game-engine render pipeline. Different distribution. |

**GS-STVSR-specific concern — unreproduced preprint:** Published April 2026, no code release at time of writing. The result cannot be independently verified. The RVRT backbone is a known-strong video SR prior (itself ~30 dB PSNR capable on its own). The 26.04 dB on Vid4 at ×4 may largely reflect RVRT's capability with the Gaussian parameterisation as a secondary effect.

---

### Paper 3: GaussianSR (arXiv:2407.18046)

**What they claim:** 2D Gaussian SR for arbitrary scale, fewer parameters than LIIF, ~29 dB on DIV2K ×4.

| Aspect | Ours | GaussianSR | Impact rating |
|--------|------|-----------|---------------|
| **Gaussian count** | 1 000–7 200 per image | One Gaussian per LR pixel; the SGS classifier assigns one kernel from a 100-class bank per pixel. N = H×W LR pixels | HIGH — one-per-pixel density, vs. our ~0.3–0.5 Gaussians per LR pixel at standard tier |
| **Covariance representation** | Fixed 16-entry bank, softmax-weighted blend producing a "mean shape" | 100-class vocabulary; Gumbel-softmax during training, hard argmax at inference. Each pixel selects a single discrete kernel — no blending, no "mean shape" | HIGH — Gumbel-softmax + argmax means each Gaussian commits to exactly one kernel shape. Our geometric weighted-mean can produce shapes not in the bank. Critically, the hard selection eliminates the "blurry mean shape" failure mode our diagnostics showed (bank_entropy_norm ≈1.0 = uniform = mean-of-all shapes). |
| **Position constraint** | tanh·tile_size (±1 tile from tile centre) | Free per-pixel position, predicted from encoder features | MID |
| **Compositing operator** | Additive SUM with topk_norm, no opacity | Alpha-weighted sum: cᵢ = σ(ξ)·vᵢ where ξ is the per-Gaussian opacity parameter. Rendering: C = Σ fᵢ(p\|μᵢ,Σᵢ)·cᵢ (Eq. 5, §3.2). Explicit opacity | MID |
| **Dual-stream feature decoupling** | No. All Gaussian output goes through a single rasterizer path. | 8 feature channels go through Gaussian rasterisation. Remaining channels take a cheap bicubic upsampling path. Both streams fused in decoder. (§3.3) | HIGH — GaussianSR does not ask the Gaussian representation to carry the full image. It carries only 8 feature channels (structure / detail), while the bicubic path carries the baseline colour. The Gaussian path only needs to encode what bicubic cannot. |
| **CNN decoder post-rasterizer** | V0.5 PixelResidualHead (12K params); but proven to learn zero weight on the splat channels | Multiple FC layers restore channel dimension and produce RGB after Gaussian rendering (§3.3, Fig. 2). The CNN decoder is architecturally integral, not a residual bolt-on. | HIGH — GaussianSR's Gaussian output is feature-space, not RGB. A mandatory CNN decoder converts rendered features to RGB. In our pipeline, the Gaussian directly produces RGB; the residual head has to overcome completely wrong RGB values rather than completing a partially-encoded feature. |
| **Network backbone** | Small UNet (standard: 16→24→32→48 channels, ~500K params) | Unspecified encoder; likely similar scale. Total params claimed "fewer than LIIF" | MID |
| **Loss function** | L1 + 0.1·(1 − SSIM) | L1 only | LOW |
| **Training dataset / LR degradation** | SRGD engine-aliased LR | DIV2K (800 images), bicubic-downsampled LR, scale U(1,4) | HIGH (bicubic-LR-trap, same issue as GSASR) |
| **PSNR vs. bicubic** | Splat-only: ~12 dB vs. bicubic ~29 dB on SRGD. V0.5: +1.3 dB above bicubic on SRGD. | 29.03 dB on DIV2K ×4 (vs. LIIF 29.00 dB, vs. bicubic 26.81 dB → ~2.2 dB above bicubic). But: bicubic-LR-trap. On clean bicubic LR, any learned SR method should beat bicubic. | Cannot directly compare; different degradation domains. |

---

## Cross-paper synthesis: the five differences most likely to matter

Ranking from highest to lowest estimated impact on the splat-SR quality gap:

### 1. Gaussian density (HIGH, all three papers)

All three papers use N ≥ H×W Gaussians (one or more per LR pixel). Our standard tier uses ~5 Gaussians per 16×16 tile = ~0.02 per LR pixel — 50–800× fewer than the papers' architectures.

**Why this matters for SR specifically:** SR requires Gaussians to represent sub-pixel features at HR resolution. A Gaussian at LR pixel (i,j) with a footprint of ~1 LR pixel projects to a ~4 LR-pixel footprint at 2x HR. With only 5 Gaussians per 16×16 tile (320 LR pixels), each Gaussian must "carry" ~64 LR pixels of content — far too coarse to encode the high-frequency edges that bicubic preserves through its anti-aliasing math.

**Counter-argument — direct-fit test:** Our direct-fit test used 50 000 Gaussians on a 512×218 frame — that is 50 000 / (512×218) ≈ 0.45 per LR pixel, much closer to the published papers' density — and still lost by 3.59 dB. This is the single most important fact in this analysis. **50K Gaussians per frame with 5 000 optimisation steps and no bank constraint is a superset of what the papers do at the representation level.** If 0.45 Gaussians/pixel with free covariance cannot beat bicubic, more density alone is not the unlock.

**Revised assessment:** Gaussian density is necessary but not sufficient. Something else the papers do — that the direct-fit test does not — is the actual unlock.

### 2. Dual-stream feature decoupling with mandatory CNN decoder (HIGH, GaussianSR)

GaussianSR never asks the Gaussian representation to produce correct RGB values directly. The Gaussians encode an 8-channel **feature** at HR resolution. A separate CNN decoder then converts those features to RGB. The bicubic path handles the remaining channels in parallel.

This is fundamentally different from our setup, where:
- Gaussians directly output RGB via `sigmoid(color)`.
- The residual head (V0.5) receives already-wrong RGB from the splat and tries to correct it.

**Why this is the likely unlock for us:** The 50K direct-fit test asked Gaussians to produce correct RGB values. The network training (splat-only V0) asked the same. Both failed. GaussianSR never asks this. The Gaussian only needs to produce a HR feature distribution that contains enough signal for the CNN decoder to complete the picture — a substantially easier task. The CNN decoder then does what CNNs are good at: dense feature-to-RGB mapping with spatial context.

**Analogy:** The difference between asking someone to reproduce a painting exactly (our setup) vs. asking them to produce a charcoal sketch that a skilled colourist will complete (GaussianSR's setup). The Gaussian representation is well-suited to the sketch task; it is not suited to the painting task.

### 3. Hard Gumbel-softmax bank selection vs. soft weighted mean (HIGH, GaussianSR)

GaussianSR uses 100 discrete kernels with Gumbel-softmax training and hard argmax at inference. Our 16-entry bank uses softmax-weighted geometric mean — every Gaussian gets the "average shape" of the whole bank weighted by confidence.

Our training diagnostics consistently show `bank_entropy_norm ≈ 1.0` even late in training. This means the bank weights are nearly uniform — every Gaussian is getting the mean of all 16 shapes. The geometric mean of all 16 entries is a near-circular blob. The network cannot escape this local optimum because all-equal-softmax-weights produce zero gradient through the bank with respect to the logits (the gradient of softmax output w.r.t. logits is proportional to the deviation from uniform; near-uniform → near-zero gradient → bank stays uniform).

**Why this matters:** A Gaussian representing a horizontal edge needs a single anisotropic entry (`(8,1,0°)`), not the average of 16 entries including circular and vertical-elongated shapes. With soft blending, the "committed" shape requires the softmax to concentrate mass on one entry, which requires large logits, which requires the network to fight against its own initialisation. Hard Gumbel-softmax during training provides a gradient signal that explicitly pushes toward the selected discrete shape.

**Note:** The direct-fit test had no bank; it used free (sx, sy, θ) via direct Adam optimisation. This test had access to any continuous covariance and still lost. So bank hardness alone is not the answer. But combined with the feed-forward network context (where the bank provides the covariance signal), hard selection may matter more — the network does not get 5 000 gradient steps to find the right covariance; it gets one forward pass.

### 4. Large attention-based backbone with cross-Gaussian interaction (HIGH, GSASR)

GSASR uses a Swin Transformer-based Gaussian Interaction Block (6 layers of window-based cross-attention) with EDSR or RDN encoder (~1.5–5M params). Our UNet has no cross-tile interaction at the Gaussian parameter level; each tile's K Gaussians are predicted independently from that tile's features.

For SR, cross-tile interaction is important because HR edges frequently span multiple tiles. A horizontal edge at tile row 3 and tile row 4 should produce Gaussians that form a coherent edge across the boundary. With independent tile prediction, each tile's Gaussians can only know their local LR content — they cannot coordinate with the adjacent tile's Gaussians to form a coherent HR feature.

**Counter-argument:** The UNet skip connections do provide spatial context at tile scale. At stride-16 in the bottleneck, each position sees a 128×128 receptive field. But the skip connections operate on spatial features, not on Gaussian parameter space.

### 5. Compositing: explicit per-Gaussian opacity (MID, GSASR + GaussianSR)

Both GSASR and GaussianSR predict a per-Gaussian opacity α ∈ [0,1] that modulates the Gaussian's contribution. Our rasterizer uses `topk_norm` (implicit normalisation) but no explicit α.

**Why this matters:** Opacity gives the network an "erase" degree of freedom — it can produce a Gaussian with well-defined position and shape but contribute with α≈0, effectively killing it without corrupting the output. Without α, a poorly-positioned Gaussian has no way to opt out; it contributes with full weight and corrupts nearby pixels.

**Impact estimate:** Probably +0.2–0.5 dB if implemented. This is a one-hour code change. The `_NON_BANK_CHANNELS` count increments from 7 to 8; the decode slice adds an α channel; and the feat multiplication becomes `feat = α * sigmoid(color)`. The rasterizer's `feat` dim already supports F>3.

**Counter-argument:** The direct-fit test had no α and had 5 000 gradient steps to zero out bad Gaussians via their position or scale. It still lost. Opacity alone is not the unlock.

---

## Honest assessment of the papers' claims

**GSASR (ICCV 2025, code released):** Strongest claim. The architecture is qualitatively different from ours in backbone scale (5–10x larger), Gaussian density (50x more), and free covariance prediction. The result is likely real at bicubic-LR. **Bicubic-LR-trap applies.** We cannot know from the paper whether GSASR beats bicubic on our engine-aliased degradation. The code is available; we could test it on our LR distribution. This is the most actionable open question.

**GaussianSR (arXiv 2024, no independent reproduction found):** Claims ~2.2 dB above bicubic on DIV2K ×4 under bicubic-LR. The dual-stream + CNN decoder architecture is the most architecturally informative for us — it explains why our "Gaussians output wrong RGB" setup fails while theirs can succeed. The 100-class Gumbel-softmax bank + mandatory CNN decoder is a genuinely different thesis from "Gaussians produce correct RGB." Moderate confidence in the claim; not independently reproduced but architecturally plausible.

**GS-STVSR (preprint Apr 2026, no code):** Weakest claim status — not peer-reviewed, no code, cannot verify. The RVRT backbone is itself a very strong video SR baseline (~30 dB at ×4 on clean video). The 26.04 dB on Vid4 may reflect RVRT capability more than the Gaussian evolution. The Covariance Prior Bank and no-CNN-refiner design is conceptually similar to our approach (minus the backbone gap and density gap). This is the paper most likely to be overstating the Gaussian contribution.

---

## Verdict

The central question: **did we falsify general Gaussian SR, or only our specific implementation?**

**The answer is: we falsified our specific implementation architecture, not the general possibility. However, the path from our implementation to a competitive one is substantially more than a few parameter tweaks — it requires architectural departures that are equivalent to a different system design.**

The most important evidence is the direct-fit test: 50 000 freely-optimised Gaussians with no bank constraint lost to bicubic by 3.59 dB on Sintel. This is the representation ceiling, and it is below bicubic. This result **does** falsify the naive thesis that "Gaussian rasterisation is itself a useful SR prior." It is not. Every dB above bicubic in the published papers comes from what the network adds before/around the Gaussian rasterisation — not from the rasterisation itself.

What the papers do that we do not:

1. **They use Gaussian-rendered features, not Gaussian-rendered RGB.** GaussianSR's dual-stream + CNN decoder is the critical structural difference. The Gaussian representation is used as a continuous HR feature extractor, not as an RGB image generator. A CNN decoder completes the image. In our V0.5, the residual head is doing the same thing — but it is trying to recover from completely wrong RGB values rather than complete an HR feature. The splat-contribution probe showed the splat channels learned weights ≈0, confirming we have a CNN with useless noise inputs, not a Gaussian-feature-fed CNN.

2. **They have 20–250× more Gaussians**, which makes the feature representation denser and higher-fidelity. However, our direct-fit test at 50K Gaussians (within the papers' density range) still failed — so density alone is insufficient.

3. **They use much larger, attention-based backbones** (EDSR/RDN/RVRT, 1.5M–30M params) vs. our 500K UNet. Cross-Gaussian interaction via transformer attention lets the network coordinate Gaussian parameters across tile boundaries.

4. **Hard Gumbel-softmax bank selection** (GaussianSR) vs. our soft weighted mean solves the "blurry mean shape" problem that our bank_entropy diagnostics expose. This is a medium-cost change.

**Is the gap closeable?** Yes — but the correct frame is "we need to redesign toward what GaussianSR does, not tune what we have." Specifically: the Gaussians should produce HR features for a CNN decoder to complete, not RGB values directly. This is a fork from our current architecture, not a fix to it.

---

## Engineering effort to test each unlock

| Unlock | Delta from current | Effort estimate | Risk |
|--------|-------------------|----------------|------|
| **Add explicit per-Gaussian opacity α** | `_NON_BANK_CHANNELS` 7→8; multiply into feat | 1–2 hours | Low. Likely +0.2–0.5 dB. Not the main unlock. |
| **Switch to Gumbel-softmax + hard argmax** for bank selection | `prior_bank.py::forward` add discrete mode; training toggle | 1–2 days | Medium. Requires verifying Gumbel gradient through bank. Expected to break bank entropy collapse. |
| **Redesign Gaussian output as feature → CNN decoder (GaussianSR architecture)** | Widen `feat` to 8+ channels; add feature-to-RGB CNN post-rasterizer; change training loss to apply on CNN output | 1–2 weeks | Medium-high. This is the most likely unlock. The 12K residual head is already doing this conceptually; the issue is it receives wrong RGB, not HR features. Redesign: render features, not RGB. |
| **Increase Gaussian density to 1/LR-pixel** | Change K_per_tile × tiles to cover all LR pixels (K≈256 at tile=16 for parity, or tile=4 with K=16) | 1 week (CUDA kernel may need tuning) | High. Memory footprint scales linearly. At 1 Gaussian/pixel on 1080p÷4 LR (270×480), need 129K Gaussians; the rasterizer has not been tested at this scale. |
| **Replace UNet with attention-based backbone (EDSR/Swin)** | New `param_net.py` with EDSR or Swin backbone, ~2M params | 2–4 weeks | High. Training cost increases proportionally. On 3080 Ti at batch=4 this may exceed VRAM. |
| **Reproduce GSASR on our LR distribution** | Run the released GSASR code against SRGD engine-aliased LR | 2–5 days (setup + eval) | Low risk, high information. This directly answers whether the architecture gap is the cause or whether the degradation domain is the real limit. |

**Recommended next action:** Before any architecture redesign, run GSASR (released code) on our engine-aliased LR distribution. If GSASR also plateaus below bicubic on engine-aliased LR, the degradation domain is the binding constraint, not our architecture. If GSASR beats bicubic on engine-aliased LR, we have a proof-of-concept for the "Gaussian features + CNN decoder" path and a concrete architecture to study.

---

## Decisions following from this analysis

1. **Do not declare the Gaussian SR representation "impossible."** The correct statement is: "Direct Gaussian-to-RGB SR is impossible at our Gaussian counts and network scale. Gaussian-to-feature-then-CNN SR may be viable if the feature path is properly designed and the backbone is adequately large."

2. **The V0.5 residual head is the right concept, wrong implementation.** The head should receive Gaussian-rendered HR features, not Gaussian-rendered wrong RGB + bicubic fallback. However, this requires widening the Gaussian feat dimension to 8+ channels and removing the sigmoid-to-RGB step before the rasterizer.

3. **The pivot to CNN-SR (`2026-05-02-splats-cannot-SR-definitive.md` Decision 5) remains correct** for the near-term deliverable. A standard CNN SR network with G-buffer conditioning produces +1.3–2.1 dB above bicubic in 1 000 steps. The architectural redesign described here is Sprint 5+ territory.

4. **Add opacity to the Gaussian batch.** It is a one-hour change with no downside. QW-1 from `papers-2407.18046-2501.06838-2503.14171-synthesis.md` is still valid.

5. **Do not spend time increasing Gaussian density** without first redesigning the Gaussian output to be features rather than RGB. More Gaussians producing wrong RGB produces more wrong RGB faster.

---

## Open questions (ordered by information value)

1. **Would GSASR beat bicubic on engine-aliased LR?** This is the single highest-value experiment — ~3 days of setup, answers whether the architectural gap or the degradation gap is the binding constraint.

2. **Does the "Gaussian features → CNN decoder" redesign (GaussianSR thesis) work at our Gaussian counts (7K) and backbone scale (500K)?** This requires building the redesigned architecture. Given the direct-fit test result, we expect it helps but not to 30+ dB — the density gap alone implies we are missing representational capacity.

3. **Does Gumbel-softmax selection break the bank entropy collapse?** Relatively cheap to test (1–2 days). Would give clean diagnostic evidence on whether the bank design is a confound.
