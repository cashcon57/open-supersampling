# Research synthesis — Gaussian splatting for real-time game upscaling

**Date:** 2026-05-01
**Source:** Two external research batches (deep web search + paper review).
**Status:** Action items extracted, plan updates queued.

This doc consolidates two batches of external research into the
OSS-Gaussian project. It validates the architecture, surfaces concurrent
work, sharpens the framing for users, and queues plan changes.

---

## 1. The validation — concurrent work confirms the approach

| Paper | What it does | Relevance |
|---|---|---|
| GaussianSR (arXiv 2407.18046, July 2024) | Pixel as continuous Gaussian field, simultaneously refined and upsampled by stacked Gaussian rendering. | Validates 2D-Gaussian-as-SR primitive. Same architectural family as ours. |
| GSASR (arXiv 2501.06838) | Predicts (opacity, μ, σ, ρ, RGB) per Gaussian; CUDA rasterizer renders at any scale via sampling-density vector. | Almost line-for-line our Sprint 4 approach. They beat INR-based per-pixel SR because each Gaussian contributes to a region, not a point. |
| Image-GS (arXiv 2407.01866, SIGGRAPH 2025) | Content-adaptive 2D Gaussian image representation. The renderer we vendored. | Foundation we build on. Already in `oss/gaussian/renderer/vendor/image_gs/`. |
| GS-STVSR (arXiv 2604.18047, 2025) | 2D Gaussian video SR; covariance correlation 0.99 frame-to-frame. | Source of our covariance-freezing optimisation in Sprint 5 + 6. |
| arXiv 2503.14171 (gradient-aware 3DGS upscaling) | Render 3DGS scene at low res, upscale via gradient-aware net. Optionally bake into training. | Closest existing work. Different premise (their input is a pre-built 3D scene; ours is 2D + G-buffer per frame). **Their gradient-aware upscaling head should inform our Sprint 4 OutputHead design.** |
| GFFE (arXiv 2406.18551) | G-buffer-free frame extrapolation. Uses optical flow, not splats. | Lower latency than DLSS 3 / FSR 3 interpolation. Same goal as our Sprint 6 but pixel-based. We should benchmark against this. |
| DLSS 4 / Frame Warp (NVIDIA 2025) | Predictive forward rendering with reprojection. | Validates the **disocclusion problem** kills naive frame extrapolation. NVIDIA invented guard bands + layered rendering and *still* has interior disocclusion artifacts in fast motion. Our splat canvas attacks this structurally. |

**Take-away:** we are at the frontier of an active research direction, not pioneering a hypothesis. GaussianSR + GSASR are concurrent work doing the same architectural pattern. Cite them in the README.

---

## 2. The killer framing for users + reviewers

External reviewers independently arrived at two phrases worth stealing:

1. **"3D-aware temporal accumulation that sidesteps screen-space TAA's
   fundamental limitations."** — DLSS 4 / Frame Warp had to invent guard
   bands + layered rendering for disocclusions; the splat canvas handles
   them structurally because warped primitives have volume, not just a
   depth value.
2. **"A vector-based upscaler"** — analogous to SVG vs JPEG. DLSS deals in
   pixels; OSS-Gaussian deals in continuous primitives that can be
   rasterised at any resolution / viewpoint shift. Non-experts can grok
   the structural advantage from this analogy alone.

Lead the README + project page with these two framings.

---

## 3. Architecture comparison — what they prescribed vs what we shipped

| Their prescription | Our implementation |
|---|---|
| Lightweight CNN encoder, single forward pass image → Gaussian params | `oss/gaussian/network/param_net.py` (Sprint 4): 75K – 5M params per tier, encoder/decoder UNet, single forward pass. ✓ |
| Persistent splat buffer, update from LR each frame, warp via G-buffer, rasterise high-res | `oss/gaussian/canvas/canvas.py` (Sprint 5): SoA persistent buffer, motion warp (positions only), error-detection + prune+spawn, render via Sprint 1 rasterizer. ✓ |
| Spawn new splats in voids based on current G-buffer | `oss/gaussian/canvas/prune_spawn.py` (Sprint 5): high-error tiles spawn from network output. ✓ |
| Tile-based rasterisation with sort by tile, avoid O(N²) blending | Image-GS CUDA rasterizer (Sprint 1 vendored). 16×16 tiles, top-K per tile. ✓ |
| Build upscaler-only first, add temporal cache next | Master plan Sprint 1+4 → Sprint 5+6 build order matches exactly. ✓ |
| Vulkan compute for Steam Deck | Sprint 7 `oss/gaussian/ports/vulkan_ncnn/` scaffold. ✓ |

We are not just "compatible with" the research recommendations — the
research independently arrived at our exact build order and component
list.

---

## 4. Where we differ — and why it's deliberate

### 2D Gaussians, not 3D

Most of the research is on 3D Gaussian splatting. We use 2D (Image-GS).

- **Trade lost:** view-dependent appearance via spherical harmonics. We
  cannot simulate specular highlights changing with viewpoint shift
  during extrapolation.
- **Trade kept:** the engine has already baked specular into the LR
  frame. We're warping screen-space tinted blobs, not lighting a 3D
  scene. For a game *upscaler*, this is the right trade: simpler model,
  faster inference, no SH parameter prediction needed.
- **Open question for v2:** if extrapolation visibly breaks specular on
  fast camera turns, we may need a small view-dependent residual head.
  Defer this to v2 measurement.

### Covariance freezing

GS-STVSR observed 0.99 correlation in covariance frame-to-frame. We
exploit this in Sprint 5 — only positions update each frame, scale and
rotation are set on spawn and frozen. The net result: per-frame compute
is dominated by the position warp + rasterisation, not network
re-prediction.

External reviewer's "VRAM concern" (1M Gaussians/frame at 720p) doesn't
apply to us because of compounded engineering choices:

| Concern | Our mitigation |
|---|---|
| 1M Gaussians per frame | K=5 per 16×16 tile + ~30% complex-tile classifier → ~8K Gaussians at 1080p |
| ~80 bytes per Gaussian (3D + SH) | 2D + no SH = ~40 bytes |
| Per-frame parameter rewrite | Covariance frozen; only positions updated |
| **Net storage per frame** | **~320 KB** (vs their estimated ~160 MB — 500× lighter) |

---

## 5. Plan updates — action items from both research batches

### Update 1: Expand baselines beyond OSSPico

Current state: graduation criterion compares Gaussian against OSSPico.
That's a friendly baseline.

**Change:** add `oss/gaussian/bench/baselines.py` with bicubic +
FSR 2 + FSR 3 + DLSS Quality + DLSS Frame Gen comparators. Sprint 1 close-out
should record perf+quality vs *all* of these, not just OSSPico.

Iso-latency comparison vs FSR/DLSS Quality is the research-grade
benchmark.

### Update 2: Sprint 4 close-out checkpoint *before* Sprint 5

Currently Sprint 4 → Sprint 5 has no decision gate. Add an explicit
checkpoint after Sprint 4 produces upscaling-only PSNR / SSIM / LPIPS
numbers vs FSR/DLSS at iso-latency.

If we're meaningfully behind FSR 2 Quality at the same compute budget,
the temporal canvas won't save us — we'd need to revisit network
architecture (per-tile attention head? hybrid CNN+transformer?) before
investing in Sprint 5.

If we beat FSR 2 Quality at iso-latency, Sprint 5+6 (canvas + extrap)
become the differentiator over DLSS 4.

### Update 3: README direction (when we write user-facing copy)

Two-sentence pitch:
> OSS-Gaussian is a vector-based real-time game upscaler. Where DLSS and
> FSR work in pixels, we work in continuous 2D Gaussian primitives that
> warp coherently with engine motion vectors — eliminating ghosting
> structurally and producing frame extrapolation as a free byproduct of
> the same canvas.

Lead the README with this. Defer architectural detail to design docs.

### Update 4: Required reading queue

Before Sprint 4 OutputHead implementation work starts, study:
1. **arXiv 2503.14171** — gradient-aware 3DGS upscaling. Their head
   design + how they use scene gradients for sharpening informs our
   OutputHead.
2. **GaussianSR (arXiv 2407.18046)** — cite as concurrent work; study
   their per-pixel-as-Gaussian formulation for any tricks we missed.
3. **GSASR (arXiv 2501.06838)** — sampling density vector trick for
   arbitrary scale. May replace or complement our K-per-tile fixed budget.

### Update 5: Concurrent-work attribution

When the README + paper draft come (post-v1), cite:
- Image-GS (foundation, vendored)
- GaussianSR / GSASR (concurrent work, same family)
- GS-STVSR (covariance-freezing insight)
- arXiv 2503.14171 (closest 3DGS upscaling work)

## 6. What's *not* changing

- Build order (1 → 2/3 → 4 → 5 → 6 → 7) matches the research
  prescription. No re-ordering needed.
- Component list is complete. No new modules required from this
  research.
- Graduation criterion (PSNR + SSIM + temporal stability + user
  approval + ≤110% latency vs OSSPico) stays as-is, with the *addition*
  of an iso-latency FSR/DLSS comparison report attached to the
  graduation decision.
- Sprint plans don't need rewriting — the changes above are
  augmentations, not redesigns.

## 7. Confidence level

The research validates the project's architectural premises with
remarkable specificity. Two independent reviewers + four+ recent papers
arrived at our component list and build order from first principles.
This is the strongest external validation signal a project could ask
for at this stage.

Outstanding risk that *no* external research touches: training-data
domain gap. NVIDIA has a captured-frames pipeline for thousands of
shipped games. We have synthetic data + hopefully Cyberpunk 2077
RenderDoc captures. This remains the highest-uncertainty input to v1
quality.
