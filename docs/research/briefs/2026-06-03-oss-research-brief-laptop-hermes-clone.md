# OSS Research Brief — 2026-06-03

## Project Status Summary
- **Latest commits:** No new commits since 2026-05-18 (`0660f55` — wipe-prep: preserve v7-pico-005 step 5000 snapshot). 16-day commit gap.
- **Training dashboard:** `srcnn-v7.0-pico-005` at step 9,100/100K, trainer degraded — GPU host offline since ~May 31. Loss 0.12692 at last update.
- **Open issues:** 10 open (no changes since yesterday). #14 VRAM measure, #13 Nsight L2 profile, #12 packed canvas, #10 decoupled G-buffer encoder, #9 RepVGG reparameterization, #8 cross-vendor kernels, #7 FSR 4 distillation, #6 launch v6.2-pico-002, #5 no-ML shader, #4 HAT-Tiny distillation.
- **Key concern:** v7-pico-005 training host has been offline ~3 days. The recurrence of GPU reclaim mid-run (v6.2-pico-002 killed at step 74K for same reason) suggests a systemic environment stability issue, not a one-off.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[GS-STVSR]** Shi et al. *Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting.* arXiv:2604.18047, Apr 20 2026. (Previously covered — still the most relevant recent paper.) Covariance resampling alignment, near-constant inference ×2–×8, >3× speedup at ×32. Direct architectural precedent for v7's Cholesky canvas warp.
- **[ContinuousSR / Pixel-to-Gaussian]** Peng et al. *Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling.* ICLR 2026. arXiv:2503.06617. Single-pass Gaussian field construction, ~1ms arbitrary-scale rendering. USTC + Huawei. **Direct competitor** — the pixel→Gaussian→render pipeline mirrors OSS's feed-forward canvas construction. Already in BIBLIOGRAPHY.md (2026-05-31), but warrants increased attention: if inference speed claims hold at game resolution, this could ship before OSS.
- **[GSASR]** Hu et al. *Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution.* ICCV 2025. arXiv:2501.06838. SoTA arbitrary-scale SR with HAT-L encoder + custom CUDA scale-aware rasterizer. Public weights and demo available. **Primary benchmarking target** — OSS v6 must beat GSASR on SR quality to justify the added complexity of temporal canvas.

### Super-Resolution / Upscaling
- **[Efficient INT8 SISR]** Nguyen et al. *Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training.* CVPR 2026 MAI Workshop. arXiv:2604.20291. Three-stage: reconstruction → Mamba-teacher distillation (Charbonnier + DCT supervision) → QAT on fused deploy graph. 29.79 dB PSNR / 0.863 SSIM at INT8. Already in BIBLIOGRAPHY.md, but the three-stage pipeline is a production-grade template for OSS student export. **Key finding:** training directly on the fused deployment graph (not the training graph) with weight clipping + BatchNorm recalibration is critical for INT8 quality — OSS should adopt this for FP8 student export.
- **[GameSR]** Mercier et al. *GameSR: Real-Time Super-Resolution for Interactive Gaming.* ICLR 2026. Engine-agnostic, RGB-only, rep-parameterized convs + PixelUnshuffle + lightweight ConvLSTM. 30–60% bandwidth savings in cloud gaming, up to 240 FPS, near-parity with DLSS/FSR on no-reference metrics. **Competitor to monitor:** works without engine hooks — if quality is genuinely close to DLSS/FSR, it challenges OSS's thesis that exact MVs are necessary for competitive SR quality. However, their no-reference metric claims need verification (no-reference metrics are unreliable for frame-to-frame consistency).
- **[NTIRE 2026 Efficient SR Challenge]** Ren et al. arXiv:2604.03198. 95 registered, 15 valid submissions. Key trends: re-parameterization widely adopted, affinity-based teacher-student distillation common. Validates OSS #9 (RepVGG ablation) and overall distillation strategy.

### Frame Extrapolation / Generation
- **[STSS]** He et al. *Low-latency Space-time Supersampling for Real-time Rendering.* AAAI 2024. arXiv:2312.10890. Unified SR + frame extrapolation in 4ms (75% faster than two-stage 17ms). Random Reshading Masking + Efficient Reshading Module. Already covered but worth re-flagging: this is the **closest conceptual prior** to OSS's unified SR+FG thesis. Published at AAAI — validates the approach is academically recognized.
- **[GFFE]** Wu et al. (Intel/NVIDIA). *GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering.* ACM TOG 2024. arXiv:2406.18551. Eliminates G-buffer dependency, only needs color history. Lightweight shading correction network. Intel-authored — strongly suggests Intel's frame-gen approach will be G-buffer-free, unlike NVIDIA's DLSS-FG which requires MVs + depth. This is the direction OSS-FX should benchmark against: can a canvas-based approach (which uses MVs) beat a G-buffer-free approach convincingly enough to justify the engine integration cost?
- **[Decoupled Motion GFFE]** Wu et al. *Decoupled Motion Prediction for Real-time G-buffer Free Frame Extrapolation.* SIGGRAPH 2026. ACM 10.1145/3746027.3754747. Extends GFFE: decouples shading motion from geometry motion. Already in BIBLIOGRAPHY.md. The decoupled approach is conceptually similar to OSS's canvas: geometry warps with MVs, appearance is re-used from the persistent canvas.

### Inference Optimization
- **[DeepFusionKernel]** Zhang et al. *Deep Kernel Fusion for Transformers.* arXiv:2602.11808, Feb 2026. Deeply fused kernel for SwiGLU MLP blocks, 13.2% speedup on H100, 9.7% on A100 over SGLang. LLM-focused but the pattern of fusing entire sub-networks into single kernels to reduce HBM traffic is directly applicable to OSS's HAT-Tiny backbone. The 2nd-gen DLSS 4.5 transformer (below) likely uses similar kernel fusion internally.

## Archive.org Findings
- No new archived papers found beyond what's already captured. GS-STVSR, GSASR, and GRAPE are all on arXiv (open access) — no takedown risk requiring archive.org backup at this point.

## GitHub / Open Source
- **GSASR** (github.com/ChrisDud0257/GSASR): ICCV 2025, weights + demo released. CUDA scale-aware 2D rasterizer. Active repo — the most directly comparable open-source Gaussian-SR implementation.
- **ContinuousSR / Pixel-to-Gaussian** (associated with ICLR 2026): USTC + Huawei. No public repo yet but paper claims ~1ms arbitrary-scale rendering.
- **GameSR**: No public repo found. ICLR 2026 paper but code not released — limits ability to benchmark directly.
- **gaussian-splatting ecosystem**: The May 2026 Radiance Fields newsletter shows explosive DCC tool integration (Photoshop, Houdini, Blender, Corona, Postshot) but no new game-SR focused repos. The ecosystem is maturing around creative tools and geospatial, not real-time game upscaling — OSS remains in an under-explored niche.
- **NVIDIA vk_gaussian_splatting**: NVIDIA's Vulkan testbed with 3DGRT + 3DGUT + DLSS-RR integration. Closest existing reference for "DLSS-on-splats." Monitor for updates.

## News

### NVIDIA DLSS 4.5 / 5.0
- **DLSS 4.5 Ray Reconstruction announced at Computex (May 31):** 2nd-gen transformer model unifying denoising + SR into single transformer. 35% more compute, 20% more parameters, comparable performance. Deeper spatial awareness, better motion data utilization. Available **August 2026** for all RTX GPUs. Blender 5.3 will integrate as denoiser. **Key takeaway:** NVIDIA is collapsing denoising + SR into one model — validates OSS's "one architecture for multiple tasks" thesis. The 2nd-gen transformer model is the target OSS must beat on quality.
- **DLSS 4.5 Super Resolution (already shipping):** 2nd-gen transformer model, replaces prior CNN-based SR. Shipped March 2026. Dynamic Multi Frame Generation up to 6X on RTX 50-series.
- **DLSS 5 (GTC March 2026):** Real-time neural rendering model — photorealistic lighting and materials infused at inference. End-to-end scene semantic understanding (subsurface scattering, fabric sheen, hair). Artist-controlled via intensity/color-grading/masking. Launching Fall 2026. **Massive competitive escalation:** DLSS 5 is not "better upscaling" — it's a different product category (neural rendering). If it ships at competitive quality, it moves the goalposts from "can OSS match DLSS SR quality?" to "can OSS match neural rendering quality?" This is a 3-5 year gap for OSS to close.

### AMD FSR 4.1
- **FSR 4.1 officially coming to RDNA 3 (July 2026) and RDNA 2 (early 2027):** AMD extends ML-powered FSR 4 upscaling to older GPUs. RDNA 3 gets it in July (next month), RDNA 2 gets it Q1 2027. **Implication:** FSR 4.1's ML model will run on a much wider GPU base, making it the most accessible ML upscaler. OSS's cross-vendor pitch must compete with FSR 4.1 being "free + ML-powered + works on RDNA 2/3." The value prop shifts from "OSS is the only open ML upscaler" to "OSS is the only vendor-neutral unified SR+FG architecture."
- FSR 4.1 currently in 300+ games (up from 30 at RDNA 4 launch). Rapid adoption momentum.

### Intel XeSS 3
- **XeSS 3 with Multi-Frame Generation:** Intel has entered the multi-frame generation race. XeSS 3 MFG inserts up to 3 generated frames between rendered frames (3:1 ratio). Runs on Alchemist GPUs and newer, plus Core Ultra CPUs. Specific release date TBD but driver support confirmed. **Key takeaway:** All three major vendors now offer some form of MFG. OSS-FX's frame extrapolation (rather than interpolation) remains the unique angle — zero latency vs. MFG's inherent 1-frame delay.

### Qualcomm Snapdragon GSR 2
- Qualcomm expanding GSR 2 (temporal anti-aliasing + upscaling for mobile) at GDC 2026. Neural SR + frame generation for Snapdragon mobile devices in active development. Mobile gaming is becoming a major upscaling battleground — validates OSS's Pico tier targeting Steam Deck and mobile-class hardware.

### NVIDIA enters PC CPU market
- First Windows AI PCs with NVIDIA chips as main processor expected first week of June 2026 (Surface and Dell). If NVIDIA controls both CPU and GPU in future gaming PCs, DLSS integration becomes even tighter — potential for proprietary hardware-level upscaling acceleration that open-source alternatives can't match.

## Action Items / Follow-ups
- [ ] **CRITICAL: Restore v7-pico-005 training.** GPU host appears offline since ~May 31. If the 3080 Ti is unrecoverable, need a plan B: resume from step-5000 checkpoint on alternative hardware or cloud.
- [ ] **Benchmark GameSR quality claims.** Their "near-parity with DLSS/FSR" is measured on no-reference metrics — needs controlled evaluation. If RGB-only engine-agnostic SR can genuinely match DLSS/FSR, OSS's MV-dependent architecture loses its main differentiation argument. Find GameSR code or reproduce architecture from paper.
- [ ] **Study DLSS 4.5 Ray Reconstruction architecture (August launch).** If NVIDIA is collapsing SR + denoising into a single transformer, OSS's "Gaussian canvas does SR and FG in one" thesis is validated but the competitive bar is now higher: OSS needs to collapse SR + FG + denoising. The v6 canvas already supports this (OSS-RR is designed on the same primitives) but it's not yet implemented.
- [ ] **DLSS 5 threat assessment.** Fall 2026 launch. Neural rendering is a generational leap beyond pixel upscaling. OSS should NOT try to compete on "photorealistic material infusion" at this stage — focus on SR+FG parity first, neural rendering is v10+ territory. Document the gap honestly in README.
- [ ] **FSR 4.1 RDNA 3 launch (July 2026) — update competitive analysis.** OSS's cross-vendor advantage shrinks if FSR 4.1 is ML-powered, open-to-integrate, and runs on all AMD GPUs back to RDNA 2. The remaining OSS differentiator is: unified SR+FG architecture, Apache 2.0 license (FSR is MIT but AMD controls the ecosystem), and Gaussian-canvas anti-aliasing as a property rather than a post-process.
- [ ] **DeepFusionKernel approach for HAT-Tiny.** The "fuse entire sub-networks into single kernels" pattern could accelerate OSS's transformer backbone. Study whether gsplat's CUDA rasterizer + fused HAT blocks could share a kernel launch to reduce overhead.
- [ ] **Add Decoupled Motion GFFE (SIGGRAPH 2026) to BIBLIOGRAPHY.md.** The decoupled shading/geometry approach could simplify OSS-FX canvas warp.
