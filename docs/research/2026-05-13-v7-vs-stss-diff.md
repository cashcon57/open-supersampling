# v7 vs STSS — Architectural Diff

**Filed:** 2026-05-13
**Source:** Wu et al., "STSS: Space-Time Supersampling for Real-Time Rendering" (arXiv:2312.10890v1, AAAI 2024)
**Purpose:** locate the bar that OSS Heavy must beat or match for unified SR + frame extrapolation.

## 1. STSS architecture in 7 bullets

- **Backbone:** standard U-Net (Ronneberger 2015). No transformer, no attention except a local 5×5 window module. Total network ~0.4M params (Backbone 141K + History Embedding 121K + ERM 156K).
- **Temporal state representation:** "History Embedding" — a learned 2D feature volume encoding previous frames + their warping masks, concatenated with the current features before fusion (the same trick from Guo 2021 / ExtraNet). No explicit per-pixel history buffer, no point primitives, no Gaussian splats. Pure 2D feature maps throughout.
- **Disocclusion handling:** the **reshading mechanism** — treats both aliasing regions and warping holes as a single class of "invalid pixels" and re-derives their colors from G-buffer + light features. Augmented at training time with **Random Reshading Masking (RRM)** that drops random rectangles in the valid mask, forcing the network to reshade arbitrary regions, not just the warping-hole ones. Loss is L1-weighted ×2 inside reshading regions.
- **Shared feature cache between SR and extrapolation:** there is **no persistent feature cache** in the OSS sense. The "sharing" is that the *same* U-Net Φ processes both modes — for SR (SF mode) it takes warped LR frames at t-4, t-2, t; for extrapolation (EF mode) it takes warped LR frames at t-5, t-3, t-1. The History Embedding is per-frame, not a long-lived structure. Sharing = parameter sharing, not state sharing.
- **G-buffer channels used:** base color, normal, depth, metallic, roughness as direct network input; motion vector, stencil, world position, normal, NoV (dot(world normal, view vec)) used to *compute* the validity mask. Roughly 9–14 channels of G-buffer in the input stack.
- **No teacher-student.** Single model trained and deployed directly. No distillation pipeline.
- **Efficient Reshading Module (ERM):** local 5×5 ReLU-linear attention where BRDF embedding is Q, masked (light + G-buffer) features are K/V, invalid pixels zero out K/V. ~10% of backbone compute. This is the only "attention" in the model — heavily local.

## 2. Numbers (RTX 3090 fp16, from Tables 1, 3, 4, 5)

| Quantity | Value |
|---|---|
| Inference latency @ 720p | 3.86 ms |
| Inference latency @ 1080p | **4.35 ms** (9.0 ms with overhead) |
| Inference latency @ 2160p (4K) | 17.19 ms |
| 1440p | not reported |
| Parameter count | **0.4M** total |
| PSNR (SF / EF, Lewis scene) | 35.02 / 34.72 dB |
| LPIPS (SF / EF) | 0.018 / 0.020 |
| SSIM (SF / EF) | 0.957 / 0.957 |
| VMAF | 78.85 |
| Two-stage baseline (ExtraNet+NSRR) | 10.3 + 20.4 ≈ 30 ms @ 1080p, PSNR 34.29/33.87, LPIPS 0.080/0.089 |
| Throughput improvement | 3.3× FPS (26→87 FPS @ 1080p Lewis) |

**No quantitative DLSS/FSR comparison** — only qualitative figures (Fig. 8), because target metrics differ. STSS does not benchmark against DLSS3 frame-gen.

EF (extrapolation) quality is essentially equal to SF (supersampling) quality (Δ ≈ 0.3 dB PSNR). Their unified model loses only **0.63 dB vs separately-optimized specialist heads** — that's the headroom we get back by specializing per mode if we want it.

## 3. Things v7 should steal

1. **Reshading mechanism + Random Reshading Masking.** Treat aliasing + disocclusion + canvas-spawn holes as one class of "needs-reshade" pixels. Use a 2× loss weight inside that mask. RRM is a one-line training-time augmentation that gives free disocclusion robustness — drop in immediately for v7-pico-005. Cost: nothing.
2. **Stencil + NoV channel.** v7's 9-ch LR input has RGB + depth + motion + normals. **NoV** (dot of world normal and view vector) and **stencil** are cheap signals for "is this a silhouette / material boundary." Add them in the capture-tool spec before too much data is collected. Cost: 2 channels.
3. **ERM-style local linear attention as the student-CNN cross-attention.** The Pico student needs cross-attention to a feature cache; STSS shows a 5×5 ReLU-linear attention is ~10% of a U-Net's cost. That's a concrete budget target for our distilled student. Cost: replaces the placeholder softmax cross-attn.
4. **Single-network parameter sharing for SF and EF modes.** v7 already has this for free via "OSS-FX is just rendering at t = N+α" — but STSS's *training-time* trick of feeding both SF inputs (t-4, t-2, t) and EF inputs (t-5, t-3, t-1) on alternating batches is a curriculum we should mirror.
5. **Per-mode separate evaluation reporting.** STSS reports SF and EF metrics separately. v7 dashboard should split α=1 and α<1 PSNR/LPIPS so we don't hide EF degradation under SF averages.

## 4. Things v7 explicitly does differently — and why those are advantages

- **N-D Gaussian primitive with V_xt cross-correlation.** STSS has **nothing analogous.** Their "temporal state" is a 2D learned feature map (History Embedding). v7's Gaussians live in (x, y, t) with full 3×3 Cholesky-packed covariance — V_xt encodes how a primitive's spatial position correlates with t, i.e. *motion direction is a learned property of the primitive itself.* This is the differentiator. Be loud.
- **No motion-vector warp at inference.** STSS warps every input frame with engine motion vectors before the network sees them. v7's time-slice rasterizer does no inference-time warp — the motion field is implicit in V_xt. **STSS will ghost on reflections, transparencies, particles, and shadows** (non-geometric motion the MV buffer can't track). v7 should not, *if training data covers those cases*.
- **Transformer teacher → CNN student distillation.** STSS is a single 0.4M U-Net. v7's Pico tier ships a ≤0.4M CNN distilled from a HAT-Tiny teacher. Same shipping size; better quality ceiling. Matches DLSS 4 strategy (transformer up, CNN out).
- **Loss-adaptive parent-child density control.** STSS has no notion of allocating capacity to hard regions. v7's parent-child spawner (Diolatzis 2024) lets the canvas grow Gaussians where loss is high.
- **Single primitive for SR + FG-interpolation + FG-extrapolation.** STSS handles SF and EF but does not do mid-frame interpolation (α=0.5 between input frames). v7's slice-at-t framework gives interp for free.

## 5. Open questions STSS does not answer for OSS

- **Can N-D Gaussians actually represent useful temporal scene structure at 100K-step budgets?** STSS shows pure 2D is enough for ~4 ms inference. We are betting Gaussians do better long-tail quality; that bet is unvalidated.
- **Cross-engine generalization.** STSS reports a single dataset (UE-rendered scenes Lewis / Medieval Docks / Bunker). Robustness across engines / art styles unmeasured. Our cross-engine capture data path is still our problem.
- **Per-pixel temporal stability over long sequences.** STSS evaluates frame quality, not temporal flicker over 100+ frames. Our temporal-consistency loss is on us.
- **TensorRT FP8 deployment numbers.** STSS reports fp16 on RTX 3090. We need INT8/FP8 numbers for ≤2 ms on 4060-class hardware.
- **Reflection/transparency/particle handling without MV warp.** STSS punts via reshading from G-buffer; our claim is V_xt avoids the punt, but we have not validated.

## 6. v7 vs STSS architecture diff table

| Component | v7 | STSS | Verdict (one-line rationale) |
|---|---|---|---|
| Backbone (research/teacher) | HAT-Tiny transformer (~3M) | U-Net CNN (~0.14M) | **v7** — higher quality ceiling, distilled to similar deploy size. |
| Backbone (shipped) | ≤0.4M nano-CNN (distilled) | 0.4M U-Net (direct) | **Tie on size**, v7 wins on quality if distillation gap < transformer gain. |
| Canvas / state representation | 3D Gaussian mixture in (x,y,t), N×25 floats | 2D History Embedding feature map | **v7** — V_xt encodes motion as primitive property, no MV-warp dependency. |
| Temporal alignment | Time-slice marginalization at t* | Per-frame MV warp into target frame | **v7** — handles non-geometric motion natively; STSS ghosts on reflections. |
| Frame extrapolation mechanism | Same rasterizer, t = N+α | Same U-Net with shifted input frames (t-5,t-3,t-1) | **v7** — single primitive for SR + interp + extrap; STSS lacks interp. |
| Disocclusion handling | Parent-child spawner + Gaussian t-falloff | Reshading mask + RRM augmentation + ERM | **STSS today** — proven recipe; v7 should steal RRM + reshading-weighted loss. |
| G-buffer channels | RGB + depth + MV + normals (9ch) | RGB + depth + MV + normals + metallic + roughness + NoV + stencil | **STSS** — add NoV + stencil to v7 capture spec. |
| Loss recipe | Charbonnier + LPIPS-VGG + foreground-aux + temporal-consistency + Sobel HF | L1 (2× in reshading regions) + 0.01·VGG perceptual | **v7** — richer, but borrow the 2× mask-weighted term. |
| Density control | Loss-adaptive parent-child (Diolatzis 2024) | Fixed-capacity feature map | **v7** — adaptive capacity to hard regions. |
| Teacher-student | Yes (HAT teacher → CNN student, per tier) | No | **v7** — DLSS-4-style strategy. |
| Latency claim @ 1080p (deploy) | TBD; target ≤2 ms Pico | 4.35 ms fp16 RTX 3090 | **STSS proven, v7 aspirational** — must hit the number to count. |
| Quality SF/EF separation | Reports α-curve; same model | Reports SF/EF separately; same model | **Tie.** |
| Cross-vendor kernels | CUDA + HIP + Metal + L0 + Vulkan committed | CUDA only (implied) | **v7** — broader deployment. |

**Bottom line:** STSS sets the bar at 0.4M params, 4.35 ms @ 1080p, PSNR ~35, LPIPS ~0.02, unified SF+EF. v7's architectural differentiators (N-D Gaussian primitive, V_xt motion encoding, transformer teacher, parent-child spawner) are real and unprecedented in this space — but they need to land *under* STSS's latency and *at or above* its quality numbers to matter. The RRM + reshading-weighted-loss + NoV/stencil channels are free wins; take them now.
