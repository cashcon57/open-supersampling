# v7 — N-Dimensional Gaussian Canvas

**Filed:** 2026-05-12
**Status:** spec, queued behind v6.3 (which itself is queued behind v6.2-pico-002 finishing at step 100K)
**Driver:** the paper ["N-Dimensional Gaussians for Fitting of High Dimensional Functions"](https://arxiv.org/abs/2405.20067) (Diolatzis, Zirr, Kuznetsov, Kopanas, Kaplanyan, 2024) and the architectural observation that OSS-FX is **structurally the same operation as slicing an N-D Gaussian mixture at a chosen time coordinate.**

## One-line claim

**Extend the OSS Gaussian canvas from 2D (x, y) to 3D (x, y, t).** The time axis t is a native dimension of each Gaussian, not an external warp step. Rendering at any α∈[0, 1] becomes "slice the 3D mixture at t = N + α" — one tensor operation, no motion-vector warp at inference. OSS-FX stops being a separate primitive; it's *the natural mode of operation* of the rasterizer.

## Why this rewrites the OSS-FX story

The v6.x story for OSS-FX was: per-frame warp the canvas by α·motion, render. That requires the canvas to encode "what the scene looks like NOW" and rely on motion vectors to evolve forward. Two consequences:

1. **The canvas's job is hard** — it must compress all the scene's temporal context into a snapshot of "right now." Long-term content disappears as soon as motion vectors stop tracking it (occlusion).
2. **Motion vectors must be accurate** — non-geometric motion (reflections, particles, transparencies, shadows) doesn't follow engine motion, so the warp distorts them.

The N-D story rewrites both:

1. **The canvas encodes a trajectory.** A Gaussian at (x, y, t) with a t-extent of, say, σ_t = 5 frames is *valid* over a range of past + future frames around t. The mixture naturally encodes "this content existed at this location during these frames." Occlusion is handled by the Gaussian fading naturally as t leaves its support window.
2. **No inference-time motion warp needed.** To render frame N + α: slice the mixture at t = N + α. Each Gaussian's contribution is weighted by its t-Gaussian falloff. The motion field is *implicit in the spatial-temporal correlation structure of the Gaussians*, not an external buffer applied at inference.

Frame extrapolation, frame interpolation, and temporal supersampling collapse into the same primitive: choose t.

## Concrete architecture

### N-D Gaussian primitive

Each Gaussian in the v7 canvas carries:

- **Mean μ ∈ ℝ³**: position (x, y, t). x, y in HR pixel coordinates; t in frame units (e.g. 0.0 = frame 0, 1.5 = halfway between frames 1 and 2).
- **Covariance V ∈ ℝ^{3×3}** parameterized via unconstrained Cholesky `V = L · Lᵀ`. Six free parameters per Gaussian (L00, L10, L11, L20, L21, L22). Unconstrained variant validated in v6.3 work to cover full PSD space without representational loss.
- **Feature vector f ∈ ℝ^R**: same low-rank latent representation as v6.2 (R=16 default), splatted to the rasterizer's output channels.

Memory per Gaussian: 3 (μ) + 6 (L) + 16 (f) = **25 floats** at fp32 → 100 bytes. Pico tier 2K Gaussians = 200 KB total. Standard tier 5K = 500 KB. Heavy tier 15K = 1.5 MB. Trivial.

### Time-slice rasterization

To render at t = t*:

```text
For each Gaussian g_i with mean μ_i = (x_i, y_i, t_i) and covariance V_i:
    Compute the marginal 2D Gaussian on the (x, y) plane conditional on t = t*:
        μ_2D = (x_i, y_i) + V_xt^T · V_tt^{-1} · (t* - t_i)
        V_2D = V_xy - V_xt^T · V_tt^{-1} · V_xt
        weight = exp(-0.5 · (t* - t_i)^2 / V_tt) · alpha_t
    Splat μ_2D, V_2D, feature · weight into the 2D rasterizer.
```

This is the standard conditional-Gaussian formula. Cost per Gaussian is small constant arithmetic; the resulting 2D splat goes through the same gsplat-style rasterizer the v6 code already uses.

The **per-frame canvas warp** (canvas_warp.py) goes away. There's nothing to warp at inference — the canvas already encodes time, and the renderer just asks for a t-slice.

### LSH culling becomes worthwhile

The bench at 2D (`scripts/sr_v6_lsh_culling_bench.py`, 2026-05-12) showed LSH gives only 1.6× tighter per-tile culling vs AABB, at higher prefilter cost — net wash for v6.x 2D. At **3D**, AABB starts failing (a 3D bounding box around an oriented ellipsoid wastes more volume), and the paper's reported 2–3× LSH speedup becomes accessible. The v7 rasterizer should be written with LSH binning from the start; the existing 2D-AABB-via-gsplat path doesn't generalize cleanly.

### Spawner: loss-adaptive density control

The DisocclusionSpawner from v6.x doesn't generalize to t-aware Gaussians. v7 adopts the paper's **parent-child deferred-materialization** scheme:

- Each Gaussian has one dormant child per training step.
- Children's parameters are expressed in the parent's reference frame (a small offset in μ, small Cholesky perturbation in L, modulated feature).
- Children materialize (become full Gaussians) when their opacity or brightness crosses a threshold during optimization.
- New children spawn for every Gaussian on a fixed cadence (~300 steps).

This is the same fix planned for v6.3.1 (canvas-utilization H009 followup). v7 builds it in from day one.

## Training implications

### Loss recipe

Same SR losses as v6.2 plus an explicit OSS-FX term:

- L_sr = Charbonnier(out_{at α=1}, GT_frame_N) + LPIPS + GAN, as v6.2
- L_fg = Charbonnier(out_{at α<1}, GT_at_N+α) + perceptual, supervised on intermediate-frame data
- L_temp_consistency = warp(out_{at α=k_1}, motion) vs out_{at α=k_2} for nearby k_1, k_2 (smoothness across the t axis)

### Training data

Three sources, all already identified in the v6.3.1 OSS-FX plan:

1. **Subsampled TartanAir** — even-indexed frames in, odd-indexed as α=0.5 GT. Free, already on disk.
2. **Vimeo-90K triplets** — 73,171 frame triplets where the middle frame is the natural α=0.5 GT. ~80 GB download.
3. **OSS Capture Tool game footage** — captured at 120 Hz on supported titles; alternate frames hold out as α=0.25 / 0.5 / 0.75 GT. Future, depends on the capture tool shipping.

### Backbone choice: transformer at teacher, CNN at student (matches DLSS 4 direction)

Each v7 teacher uses a **transformer-class backbone** (HAT family). Each v7 shipping **student** is a small CNN distilled from its tier's teacher and exported to TensorRT FP8 with custom cross-vendor kernels:

| Tier | v7 Teacher (research) | v7 Student (ships) |
| --- | --- | --- |
| Pico | HAT-Tiny (~3M, transformer) -- the v7-pico-005 backbone | ≤0.4M nano-CNN |
| Standard | HAT-Small (~5M, transformer) | ≤1M CNN |
| Heavy | HAT-L-derived Heavy (~17M, transformer) | ≤2M CNN |

This matches DLSS 4's strategy: NVIDIA ships a transformer-class top-tier model AND a CNN-class fallback for older hardware. They did not replace CNN with transformer; they added the transformer for the high-end and kept the CNN distillation for broader hardware support.

Backbone ablations as part of the v7-pico-005 cycle:

| Run | Backbone | Purpose |
| --- | --- | --- |
| v7-pico-005 (main) | HAT-Tiny | Establish v7's quality at pico tier (apples-to-apples vs v6.2-pico-002) |
| v7-pico-005-no-canvas | HAT-Tiny + canvas disabled | Confirm canvas is load-bearing |
| v7-pico-005-no-spawner | HAT-Tiny + parent-child off | Measure parent-child spawner contribution |
| v7-pico-005-cnn-ablation | 4-layer CNN ~500K params | Research: does canvas compensate for transformer quality? If yes, simpler shipping path |

The CNN ablation is research-only -- the shipping path remains teacher (transformer) → student (CNN) distillation regardless of the ablation outcome.

### What v6.3 still delivers as the bridge

v6.3 stays in the plan as the **proving ground** for the components before they scale to N-D:

| v6.3 piece | Carries forward to v7 |
|---|---|
| Magnitude scaling on canvas_hr at fusion boundary | Same fix needed — N-D doesn't help if composite_head ignores canvas |
| Canvas-aware aux loss (refined_hr=0 path) | Same; forces canvas to carry SR signal |
| Cholesky covariance parameterization (unconstrained variant) | Required for 3D — sigmoid bound doesn't cover OSS shapes (test confirmed) |
| Parent-child deferred spawning (v6.3.1 followup) | Built into v7 from day one |

v6.3 is no longer the destination; it's the runway. We debug each component at 2D where ablation cost is one training cycle (~5 days on the 3080 Ti), then scale to N-D in v7 with the per-component bugs already shaken out.

## Sprint shape

| Phase | Scope | Duration |
| --- | --- | --- |
| **0** | Land v6.3 spec implementation (Cholesky + magnitude scale + aux loss flags) behind defaults. v6.3-pico-003 training run validates the canvas-utilization fixes at 2D. | 1 week build + 5 days train |
| **1** | v7 N-D rasterizer scaffold + tests. Pure-PyTorch ref implementation of 3D Gaussian time-slice. No model integration yet. | 1 week |
| **2** | v7 model wiring: canvas state extended to N-D, spawner replaced with parent-child, training loop accepts intermediate-frame supervision. Smoke-test on TartanAir-subsampled at 1K steps. | 1.5 weeks |
| **3** | v7-pico-005 training run on TartanAir-subsampled + Vimeo-90K. 100K steps, ~6 days on 3080 Ti. First OSS-FX metric. | 1 week |
| **4** | CUDA rasterizer port (Triton kernel) + LSH culling kernel. Targets 2–3× speedup at 3D vs naive Python ref. | 2–3 weeks |
| **5** | Cross-engine fine-tune cycle on captured game footage (Cyberpunk 2077, Alan Wake 2 via OSS Capture Tool). | 2 weeks + dataset wait |

Total time-to-first-OSS-FX-result: **~4 weeks of engineering + 11 days of training**, starting from v6.2 completion.

Compute estimate (v7-pico-005, pico-tier — the actual Phase 3 deliverable, not Heavy):

| Component | H100-hours | Spot ($1.50/hr) | On-demand ($3/hr) |
| --- | --- | --- | --- |
| v7-pico-005 single training run (100K steps, ~12 s/step on H100; pico-002 baseline 4.5 s/step × ~1.5–2× for N-D overhead) | ~70–110 | $100–$170 | $200–$330 |
| 3–5 ablations (canvas-on/off, spawner-on/off, α-curriculum) | 200–400 | $300–$600 | $600–$1.2K |
| Vimeo-90K fine-tune (50K steps) | 30–50 | $50–$80 | $100–$160 |
| Eval + held-out + slack 20% | ~80 | $120 | $240 |
| **Pico-tier v7 first cycle total** | **~400–650** | **$600–$1K spot** | **$1.1K–$1.9K on-demand** |

Heavy-tier v7 (the shipping target via distillation from pico-005) would scale to the cost-estimate memo's Heavy cycle range ($17K–$33K spot for 11K–22K H100-hours), but that's not the Phase 3 cost — that's later.

## Risks + open questions

- **Can a 3D Gaussian mixture actually represent useful temporal scene structure?** Not measured anywhere in this paper or in OSS. v7-pico-005 is the test.
- **Will the 2D backbone (HAT-Tiny) feature path still help, or does N-D canvas alone produce SR + FG?** Unknown. v7 model can be tested with and without backbone.
- **Memory at training time** — gradient through time-slice operation requires backward pass to flow through the conditional-Gaussian arithmetic. Standard autograd handles it, but training memory scales with R · N_gaussians.
- **Reflections / non-geometric motion** — engine motion vectors don't track these. v7 might still produce ghosting on them unless the captured data covers them during training.
- **Loss balance** — supervising α<1 + α=1 jointly might create training instability if the model can't satisfy both at the same time. Curriculum: train α=1 (pure SR) for first 20K steps, then add α=0.5, then add α=0.25 / 0.75.

## What this does NOT change

- The HAT-Tiny **teacher** model lineage. We still distill v7 to ≤1M-param student before shipping.
- The Apache-2.0 + CC-BY-4.0 licensing.
- The cross-vendor kernel commitment (CUDA + HIP + Metal + Level Zero + Vulkan).
- The dashboard run-name conventions (`srcnn-v7.0-pico-005` etc.).

## Filing companion docs

- `docs/research/2026-05-12-nd-gaussians-paper-relevance.md` — detailed analysis of the source paper (the LSH bench result, the Cholesky test outcomes, the parent-child spawner plan)
- This spec replaces the "park N-D Gaussians as v7 direction" line in `docs/architecture/2026-05-11-v63-pico-003-canvas-utilization-spec.md` §"What this does NOT fix". v7 is no longer parked — it is the architectural target.
