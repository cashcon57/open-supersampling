# OpenSuperSampling: Covariance-Resampled Online Gaussian-Temporal Super-Resolution for Real-Time Game Rendering

**Cash Conway**
*OpenSuperSampling Project*
`cashcon57@gmail.com`

**Status:** pre-alpha. v6 architecture locked 2026-05-05. v5 dual-track validation in flight. Implementation, training, and integration runtime all incomplete. This document describes the design under development; reported numbers are scoped to what has actually been measured.

---

## Abstract

OpenSuperSampling (OSS) is an open, vendor-agnostic alternative to proprietary real-time super-resolution stacks (DLSS, FSR, XeSS). The canonical v6 architecture is an online Gaussian-temporal super-resolver: a HAT-Base spatial backbone produces coarse high-resolution features from the current low-resolution frame plus engine G-buffers; these features cross-attend to a persistent canvas of 5K–15K 2D Gaussians, accumulated across frames and warped per-frame by an analytical sub-pixel transform with explicit covariance resampling in the manner of Zhou et al. (GS-STVSR, 2026). A Spatial-Temporal Variation Score (Yuan et al., NeurIPS 2025) drives score-based active pruning. Because the canvas is rendered through the same rasterizer at fractional time positions $\alpha \in (0, 1)$, frame extrapolation falls out as a one-tensor-add byproduct rather than a separate ML network. Three model tiers (Pico, Standard, Heavy) share one architecture, distilled, and target every major GPU vendor through per-vendor custom kernels. Integration is planned through DLL-shim drop-in for any title already supporting DLSS, FSR, or XeSS — no game-developer cooperation required. The project is pre-alpha; v4 single-frame baseline is trained, v5 temporal validation tracks are in training/queued, v6 implementation is not yet started.

---

## 1. Motivation

The dominant real-time super-resolution stacks are vendor-coupled. DLSS is restricted to NVIDIA RTX hardware and requires the proprietary NGX runtime. XeSS XMX path is restricted to Intel Arc; its dp4a fallback exists but is bounded in quality. FSR 2/3 is open and cross-vendor but is implemented as hand-tuned shader passes without learned components and is bounded above by what shader-grade temporal accumulation can express on a fixed sampling grid. FSR 4 (the announced ML path) brings a learned model but remains an AMD project under AMD release control.

The result is that the ~60% of installed gaming GPUs outside of current-flagship NVIDIA — Steam Deck, mid-range AMD, Intel Arc, Apple Silicon, older NVIDIA — does not have an open ML upscaler with quality competitive with DLSS-class output. Two architectural observations follow.

**Observation 1.** The published advantage of DLSS over FSR 2/3 is largely temporal stability and disocclusion handling — properties that depend on the choice of *what* gets accumulated across frames, not only on raw network capacity. Pixel-grid temporal accumulators must resample previous output through a bilinear or bicubic warp per frame and gate on heuristic disocclusion masks; ML helps but cannot remove the resampling-blur compounding fundamental to the grid.

**Observation 2.** Frame extrapolation in current production stacks (DLSS Frame Generation, FSR 3 Frame Generation) is a separate ML pass that synthesizes intermediate frames from a pair of rendered frames plus motion. It carries its own training cost, its own latency, and its own hallucination mode. There is no architectural reason these two functions — super-resolution and frame extrapolation — should be different networks if the underlying scene representation is continuous in space and time.

OSS targets both observations from the same starting point: replace the implicit pixel grid with an explicit, persistent, primitive-based scene representation amenable to analytical temporal warping.

---

## 2. Contributions

This paper-style document describes three architectural commitments that, to our knowledge, are not jointly present in any shipped or published real-time super-resolver:

1. **Persistent Gaussian canvas with engine-motion-vector-driven analytical sub-pixel warp for streaming game super-resolution.** Existing 4D Gaussian methods (Wu et al. 2024, 4D-Rotor, Spacetime GS, Yuan et al. 2025) target offline reconstruction and require full bidirectional temporal context. OSS v6 maintains a streaming canvas of 5K–15K 2D Gaussians per scene, updated frame-by-frame from past frames only, warped analytically per Gaussian by the engine's motion vector field. There is no resample-blur compounding across the temporal trail, and disocclusion is handled by exact densification rather than heuristic gate-and-fill.

2. **Application of GS-STVSR-style covariance resampling to streaming temporal SR.** Following Zhou et al. (2026), each warped Gaussian's screen-space covariance is updated by

$$\Sigma'_{\text{output}} = J_t\, \Sigma_t\, J_t^\top + \Sigma_{\text{recon}}$$

where $J_t$ is the warp Jacobian and $\Sigma_{\text{recon}}$ is an EWA-style low-pass reconstruction filter matched to the target output resolution. This is anti-shimmering by mathematical construction: aliasing energy is filtered before rasterization, not after. Pixel-grid methods (DLSS, FSR, XeSS) can apply post-hoc temporal anti-aliasing but cannot pre-emptively reshape the reconstruction kernel of an underlying continuous primitive — they have no continuous primitive.

3. **Frame extrapolation as a direct byproduct of $\alpha$-conditioned canvas rendering.** The same trained canvas, rendered at $\alpha = 1$, produces super-resolved current-frame output; rendered at $\alpha \in (0, 1)$ along the motion field, it produces an extrapolated intermediate frame. The architectural cost above and beyond the SR render is one in-place addition to the $(N, 2)$ Gaussian-position tensor. The trained weights are shared. This contrasts with DLSS Frame Generation, which is a separately trained network with its own latency budget and failure modes.

In support of these three, OSS v6 also commits to: a cross-attention bridge between dense pixel features (HAT-Base backbone) and sparse Gaussian tokens (canvas K/V); three deployment tiers (Pico ~1M params, Standard ~5M, Heavy ~15M) sharing one architecture and trained by Heavy → Standard → Pico distillation; per-vendor custom inference kernels (CUDA/CUTLASS, HIP/rocWMMA, Metal/MPS, Level Zero/XMX, Vulkan compute) following the same engineering pattern used internally by every shipped ML upscaler; and a DLL-shim integration path that requires no game-developer cooperation in any title that already supports DLSS, FSR, or XeSS.

---

## 3. Architecture (v6 canonical)

```text
                                      ┌──────────────────────────┐
  current LR + G-buffers ────────────►│ HAT-Base spatial backbone│──► coarse SR features
  (RGB, depth, motion, normals)       └──────────────────────────┘                │
                                                                                  ▼
                                                       ┌────────────────────────────┐
   persistent Gaussian canvas ──► analytical warp ────►│ cross-attention            │──► refined HR features
   (5K-15K Gaussians per scene,    by engine MVs +     │ (pixel queries × Gaussian  │
    accumulated across frames)     covariance          │  keys/values)              │
                                   resampling          └────────────────────────────┘
                                                                                  │
   key-frame active mask (every K=10 frames) ───────────► tile-based rasterizer ──► HR output
                                                                                  │
   Spatial-Temporal Variation Score pruning ◄─── canvas update ◄──────────────────┘

   Frame extrapolation (OSS-FX): rasterize canvas at α ∈ (0, 1) instead of α = 1.
   Above-and-beyond cost: one in-place add to the (N,2) position tensor.
```

### 3.1 HAT-Base spatial backbone

A Hybrid Attention Transformer (Chen et al., 2023) at the Base scale (~17M params) provides the dense feature path. Window self-attention plus channel attention, configured to consume the 9-channel input stack (RGB + depth + motion + normals) at low resolution, producing coarse high-resolution features.

### 3.2 Persistent Gaussian canvas

A canvas of $N \in [5\mathrm{K}, 15\mathrm{K}]$ 2D Gaussians, each parameterized by mean $\mu \in \mathbb{R}^2$ in HR pixel coordinates, covariance $\Sigma \in \mathbb{R}^{2 \times 2}$ stored via scale + rotation, opacity $o \in [0, 1]$, and color $c \in \mathbb{R}^3$. Persists across frames; updated by warp + transformer + densify + prune at each step.

### 3.3 Analytical warp with covariance resampling

Per-Gaussian:

$$\mu' = \mu + v(\mu),\qquad \Sigma' = J_v(\mu)\, \Sigma\, J_v(\mu)^\top + \Sigma_{\text{recon}}$$

where $v(\mu)$ is the engine motion vector sampled at $\mu$ and $J_v$ its Jacobian, computed by finite differences on the engine MV field. The reconstruction kernel $\Sigma_{\text{recon}}$ is a screen-space low-pass matched to the rasterizer's HR pixel pitch. No bilinear or bicubic resampling is involved at any step.

### 3.4 Cross-attention bridge

Dense backbone features serve as queries; the warped Gaussian token set serves as keys and values. The cross-attention output refines the HR feature map with structured information drawn from the persistent canvas. Multi-head (4 heads at Standard, 8 at Heavy), with rotary positional encoding indexed by Gaussian mean.

### 3.5 Spatial-Temporal Variation Score pruning

Following Yuan et al. (NeurIPS 2025), each Gaussian is scored by

$$\mathcal{S}_i = \mathrm{SS}_i \cdot \mathrm{TS}_i$$

where $\mathrm{SS}_i$ is a spatial contribution score (sum of $\alpha_i \cdot T_i$ across recent frames) and $\mathrm{TS}_i$ a temporal-stability score. Bottom 60–80% are pruned in a periodic pass. Yuan et al. report 14–34× rendering speedup at <0.3 dB quality cost on Plenoptic Video; the architecture is portable to streaming SR settings with the same scoring criterion.

### 3.6 Key-frame active-Gaussian mask

Every $K = 10$ frames a binary active mask is precomputed identifying Gaussians with non-negligible contribution; intermediate frames inherit the nearest key-frame mask. The rasterizer skips inactive Gaussians. This exploits the empirical 85–95% active-set overlap between adjacent frames reported by Yuan et al.

### 3.7 Tile-based rasterizer

A standard 16×16-tile screen-space rasterizer with per-tile depth sort and front-to-back alpha compositing, derived from gsplat (Kerbl et al., SIGGRAPH 2023). The same rasterizer is used at $\alpha = 1$ for SR and at $\alpha < 1$ for OSS-FX frame extrapolation.

---

## 4. Method

### 4.1 Inputs

Per frame the network consumes 9 channels at low resolution: RGB (3) + depth (1) + 2D motion vectors (2) + surface normals (3). The earlier SRGD-era "canvas hint" channels are dropped because TartanAir and Hypersim do not provide them and zero-padding produced an observed train-time/eval-time distribution mismatch in v3/v4.

### 4.2 Loss

The combined loss carried forward from v6 design work is

$$\mathcal{L} = \mathcal{L}_{\text{Char}} + \mathcal{L}_{\text{LPIPS}} + \mathcal{L}_{\text{VGG-MS}} + 0.5\,\mathcal{L}_{\text{wavelet}} + 0.05\,\mathcal{L}_{\text{GAN}} + 0.2\,\mathcal{L}_{\text{edge}} + 0.5\,\mathcal{L}_{\text{TC}} + 0.01\,\mathcal{L}_{\text{Greg}}$$

Components: Charbonnier (smooth L1) for pixel fidelity (~0.1–0.2 dB over plain L1 in our internal screening); LPIPS-VGG for perceptual quality; multi-scale VGG L1 over relu1_1/2_1/3_1/4_1/5_1 with weights {0.1, 0.1, 1.0, 1.0, 1.0}; wavelet L1 for explicit high-frequency supervision (targets thin-feature ghosting on power lines, foliage edges); UNetD hinge GAN (after Wang et al., Real-ESRGAN, 2021) starting at step 20K to avoid early-training instability; Sobel-edge L1 as a sharpness regularizer; warp-then-diff temporal consistency; and a mild Gaussian-canvas anti-collapse regularizer.

### 4.3 Training

AdamW with $\beta = (0.9, 0.99)$ and weight decay $10^{-4}$. Cosine schedule with three warm restarts ($T_0 = 50\mathrm{K}$, $T_{\text{mult}} = 1$) at base LR $2 \times 10^{-4}$. Mixed precision in bf16 (more stable than fp16 under the hinge GAN loss on Ampere). Effective batch 16 (batch=4, grad-accum=4) for the Heavy teacher at patch size $256^2$. Patch sampling is 70% importance-weighted by per-tile variance + 30% uniform. EMA on the teacher only ($\beta = 0.999$). Teacher target: 300K steps; per-student tier: 80K steps with KD + GT loss.

### 4.4 Datasets

The v6 training mix is 60% TartanAir Easy (real depth + flow, photoreal outdoor) and 30% Hypersim (real depth + normals, photoreal indoor); the remaining 10% is held out for evaluation. SRGD is excluded from v6: it has zero-valued G-buffers (depth/motion/normals were placeholder zeros at capture time) and mixing it in produced a measured distribution mismatch in earlier runs. The held-out environment for TartanAir is the `oldtown` env exclusively; this closes the data-leak gap from a v5-pixel launch that initially iterated the full Easy split.

---

## 5. Evaluation protocol

All comparisons are run on a fixed-batch held-out manifest derived from environments excluded from training. The current v5 manifest is 64 paired frames drawn entirely from TartanAir `oldtown` (`<train-host-data>/checkpoints/v5_held_out_manifest.json`); a Sintel held-out subset is queued behind dataset-fetch completion.

Reported metrics:

- **PSNR** (dB), per-frame, mean and per-frame distribution.
- **LPIPS-VGG**, per-frame, mean and distribution. Note that LPIPS implementations vary across the literature; we use the canonical `lpips` Python package weights, and direct cross-paper comparison is therefore approximate.
- **Temporal stability**, defined as the warp-then-diff L1 between $\hat{y}_t$ warped by motion $t \to t+1$ and $\hat{y}_{t+1}$, averaged over the held-out clip.
- **Latency**, measured end-to-end on RTX 3080 Ti via TensorRT FP16 with a narrow optimization profile per resolution.

External comparisons against DLSS, FSR, XeSS are *not* currently apples-to-apples: those numbers are published on different content (game captures vs TartanAir/Hypersim/Sintel), at different resolutions, with different LR-synthesis assumptions (real engine LR vs our `EngineAliasedLRSynth` recipe), and against possibly different LPIPS implementations. A principled comparison requires the DLL-shim integration runtime to capture matching engine LR + G-buffers in the same titles those vendors evaluate on. That runtime is in design, not built.

---

## 6. Current results (honest)

This section describes only what has been measured. Numbers are scoped to the test set and configuration on which they were measured. We do not claim parity with or superiority over any vendor stack at this time.

### 6.1 v4 single-frame baseline

Trained on SRGD with engine-aliased LR synthesis (Halton jitter + TAA blur + JPEG q=85). Production checkpoint `srcnn-prod-v4-lpips/step-00385000.pt` is a fork from v3 step-240K with `L1 + 0.1·SSIM + 0.1·LPIPS-VGG` for ~12 hours.

On a fixed-batch CitySample held-out A/B against v3:

- **v4 vs v3:** −22% LPIPS, −1.5 dB PSNR, win on perceptual A/B 64/64 frames. This is a Real-ESRGAN-style perceptual/fidelity trade and matches the published direction of that paper.
- **v4 vs bicubic:** ~+2.5–3 dB PSNR, LPIPS preferred 8/8 on the rolling fixed batch.

These numbers are valid only on the SRGD-distribution held-out set. They are not directly comparable to vendor-published numbers on game captures.

### 6.2 v5-pixel-temporal (control track)

Implementation complete (`oss/sr/temporal/`); training in flight on `<train-host>` warm-started from v4 step-385K; held-out gate scheduled. No final-checkpoint numbers exist yet. Live-training metrics are not appropriate input to a quality claim and are deliberately omitted here.

### 6.3 v5-gaussian-temporal (research track)

Implementation complete (`oss/sr/gaussian_temporal/`, 113 tests passing 1 skipped as of 2026-05-04); staged Stage-0/1/2 validation queued sequentially behind the pixel run on the shared RTX 3080 Ti. No convergence numbers.

### 6.4 v6

Architecture locked 2026-05-05 (`docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md`). Implementation queued behind v5 staged validation. No training runs. No numbers.

### 6.5 Charts with vendor reference lines

The live in-flight training viz at `http://<tailnet-ip>:8080/` annotates v5-pixel PSNR/LPIPS curves with reference lines drawn from published benchmarks of bicubic, FSR 1–4, and DLSS 2–4. These reference lines are visualization aids on different content distributions and resolutions; the apparent margin of any in-flight v5 curve over a vendor reference line is **not** evidence that v5 beats that vendor. A valid comparison requires identical content, identical LR synthesis, and identical metric implementations. We do not yet have any of those.

---

## 7. Limitations

We list the known weaknesses of the current design, in decreasing order of expected impact on shippability.

- **No production shipping precedent for Gaussian-temporal SR.** Every component cited (3DGS rasterization, 4DGS-1K pruning, GS-STVSR covariance resampling, HAT backbone) has been published and is reproducible, but no game has shipped with a Gaussian-temporal upscaler. Stage 0/1/2 validation gates are designed to de-risk the convergence question cheaply (~7 hours total) before paying for full training.
- **Per-frame fitter cost may exceed the real-time budget.** A canvas of 5K–15K Gaussians warped + cross-attended + rasterized + S-T-scored per frame is a heavier per-frame workload than a pixel-grid temporal pass. Score-based pruning (4DGS-1K reports 14–34× speedup) and key-frame active masking are the principal mitigations; whether they suffice on the v6 architecture has not yet been measured.
- **Cross-attention from pixel queries to Gaussian K/V has no production precedent.** Reference implementations exist in research papers and the layer is straightforward in PyTorch; the engineering risk is in the per-vendor fused-kernel implementation, where the irregular sparse K/V access pattern is harder than pure dense attention.
- **GAN training instability.** Real-ESRGAN-style hinge UNetD loss is well-trodden but still requires careful warmup and discriminator-loss monitoring. The bf16 + 20K-step pixel-only warmup mitigation is standard but not foolproof.
- **DLL-shim integration is unbuilt.** The S7 design memo (`docs/superpowers/notes/2026-05-04-s7-game-integration-design.md`) describes the NGX/FSR/XeSS shim path; no shim has been compiled, no game has been intercepted. Comparative evaluation against vendor stacks on real game content depends on this runtime.
- **Training corpus is currently TartanAir-easy-mode plus Hypersim.** AAA volumetric and transparency content (smoke, foliage, particle systems, hair, glass) is not well-represented. Generalization to that content will require the OSS Capture Tool community pipeline (Sprint 7-data) to accumulate real game captures, and is gated on contributor opt-in volume.
- **Steam Deck is not yet viable.** RDNA 2 has no matrix accelerator. The v6 Pico tier targets Steam Deck through hand-tuned Vulkan compute, which is the same engineering pattern FSR 2 uses, but Pico is not yet implemented or trained.
- **Six-month timeline-to-demo is optimistic.** The honest expected-case slip is to twelve months; the design memo acknowledges this explicitly.
- **HDR support is partial.** As of commit `694a0f3` the gaussian-temporal model uses a softplus output activation, so HDR input/output flows through architecturally without clipping (sigmoid-clamping was removed). However, the entire training corpus (TartanAir, Hypersim, SRGD) is 8-bit sRGB. HDR-specific patterns — sun discs, neon, specular highlights, BT.2020 wide-gamut colors — are not well-represented in what the model has seen. Expected HDR quality is approximately 70–80% of SDR quality on the same content class: noticeably better than bicubic, behind DLSS HDR. v6.1 schedules retraining on HDR-encoded data via INSANE-mode capture of HDR-rendered games plus re-rendered Hypersim in linear scRGB to close this gap.

---

## 8. Related work

**3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023, [1]).** The rasterization foundation. OSS uses the same EWA-projection + tile-based front-to-back alpha-composite pipeline at 2D for the canvas.

**4D-GS (Wu et al., CVPR 2024, [2]).** Canonical Gaussians plus a HexPlane-decomposed deformation MLP for offline dynamic scenes. OSS is distinct: streaming use case, past frames only, no canonical reference, online densification on disocclusion rather than smooth deformation.

**4D-Rotor / Spacetime GS [3, 4].** Native 4D primitives with conditional-marginalization to a 3D slice at render time. Handles topology change naturally but at 1.5–3× more primitives per equal quality. OSS instead handles topology change via online densification on disocclusion, which is cheaper for the streaming case.

**4DGS-1K (Yuan et al., NeurIPS 2025, [5]).** Spatial-Temporal Variation Score pruning + key-frame active mask. OSS adopts both directly. Yuan et al. report 1029 FPS rendering on Plenoptic Video at <0.3 dB cost; we expect different absolute numbers in the streaming-SR setting but the same direction.

**GS-STVSR (Zhou et al., 2026, [6]).** The covariance-resampling formulation for spatio-temporal SR on Gaussian primitives. OSS lifts the resampling formula directly into the streaming-SR canvas.

**HAT (Chen et al., 2023, [7]).** Hybrid Attention Transformer for image SR. OSS uses HAT-Base as the spatial backbone of the dense path and HAT-Tiny/Small at distilled tiers.

**DLSS / FSR / XeSS [proprietary].** The targets. DLSS uses a per-vendor fused CUDA kernel with tensor-core MMA over a learned temporal-aware network and a separate Frame Generation network. FSR 2/3 is a hand-tuned shader; FSR 4 (announced) adds a learned model. XeSS uses XMX matrix instructions on Arc with a dp4a fallback. OSS is positioned as cross-vendor by design with one trained architecture and per-vendor specialist kernels — the same engineering pattern, applied openly across all backends rather than within one.

**Real-ESRGAN (Wang et al., CVPR 2021, [8]).** Source of the UNetD hinge GAN training discipline, the perceptual/fidelity trade pattern, and the LR-synthesis fidelity awareness that v3→v4 used.

---

## 9. References

[1] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* SIGGRAPH 2023.

[2] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, X. Wang. *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering.* CVPR 2024.

[3] Y. Duan, F. Wei, Q. Dai, Y. He, W. Chen, B. Chen. *4D-Rotor Gaussian Splatting: Towards Efficient Novel-View Synthesis for Dynamic Scenes.* SIGGRAPH 2024.

[4] Z. Li, Z. Chen, Z. Li, Y. Xu. *Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis.* CVPR 2024.

[5] Y. Yuan et al. *1000+ FPS 4D Gaussian Splatting for Dynamic Scene Rendering.* NeurIPS 2025. arXiv:2503.16422.

[6] H. Zhou et al. *GS-STVSR: Spatio-Temporal Video Super-Resolution via Gaussian Splatting.* 2026.

[7] X. Chen, X. Wang, J. Zhou, Y. Qiao, C. Dong. *Activating More Pixels in Image Super-Resolution Transformer (HAT).* CVPR 2023. arXiv:2205.04437.

[8] X. Wang, L. Xie, C. Dong, Y. Shan. *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.* ICCV Workshops 2021.

[9] A. Guédon, V. Lepetit. *Gaussian Frosting: Editable Complex Radiance Fields with Real-Time Rendering.* ECCV 2024.

[10] *3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting.* NVIDIA. CVPR 2025. arXiv:2412.12507.

[11] *GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering.* HPCA 2026.

[12] *4DGC: Rate-Aware 4D Gaussian Compression.* CVPR 2025. arXiv:2503.18421.

[13] V. Ye, R. Li, J. Tancik, A. Kanazawa et al. *gsplat: An Open-Source Library for Gaussian Splatting.* arXiv:2409.06765 (mathematical supplement: arXiv:2312.02121).

[14] Khronos Group. *glTF KHR_gaussian_splatting Extension.* (Ratification target Q2 2026.)

[15] P. Heckbert. *Fundamentals of Texture Mapping and Image Warping.* M.S. Thesis, UC Berkeley, 1989. (EWA filter origin.)

---

*Project repository:* `https://github.com/<owner>/OpenSuperSampling` (active branch: `v0.2-dev`).
*Companion documents:* the v6 canonical architecture memo (`docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md`) and the Gaussian-temporal research deep-dive (`docs/research/2026-05-05-gaussian-temporal-research-deep-dive.md`) provide implementation-level detail and worked-out math for every component referenced above.
