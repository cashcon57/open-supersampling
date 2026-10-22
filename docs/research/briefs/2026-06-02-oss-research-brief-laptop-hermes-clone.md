# OSS Research Brief — 2026-06-02

## Project Status Summary
- **Latest commits:** No new commits since 2026-05-18 (`0660f55` — wipe-prep: preserve v7-pico-005 step 5000 snapshot). 9-day commit gap.
- **Training dashboard:** `srcnn-v7.0-pico-005` at step 9,100/100K, stale **59h**. Trainer degraded — GPU host appears offline since ~May 31. Loss was 0.12692 at last update. $0.23 GPU cost so far.
- **Open issues:** 10 open — #14 (VRAM measure v6.2), #13 (Nsight L2 profile), #12 (packed low-rank canvas), #10 (decoupled G-buffer encoder), #9 (RepVGG reparameterization ablation), #8 (cross-vendor kernel ports), #7 (FSR 4 distillation), #6 (launch v6.2-pico-002), #5 (no-ML shader variant), #4 (HAT-Tiny student distillation), #3 (counting-sort tile bin), #1 (R=4-8 latent rasterizer)
- **Key concern:** v7-pico-005 training host has been offline ~2 days. If the 3080 Ti is unrecoverable, the run may need restart on alternative hardware. Step 74K v6.2-pico-002 precedent shows the GPU was reclaimed mid-run — pattern repeating.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[GS-STVSR]** Shi et al. *Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting.* arXiv:2604.18047, Apr 20 2026. (Previously covered in detail June 1 brief, still the most relevant recent paper.) Models spatiotemporal evolution of Gaussian kernels via continuous motion modeling. Covariance resampling alignment prevents drift. Near-constant inference time ×2–×8, >3× speedup at ×32. **OSS relevance:** Direct architectural precedent for v7's canvas-warp-by-Cholesky. Their optical-flow motion module is the offline analog of OSS's exact-engine-MV pipeline.

- **[AA-2DGS]** Younes & Boukhayma. *Anti-Aliased 2D Gaussian Splatting.* NeurIPS 2025. arXiv:2506.11252. World-space flat smoothing kernel constrains Gaussian frequency from training-view max sampling frequency. Object-space Mip filter via affine ray-splat intersection approximation. Code available. **OSS relevance:** Directly validates v7's Mip-Splatting anti-aliasing filters. The object-space Mip filter approach could refine OSS's current rasterizer anti-aliasing — this is a concrete implementation reference.

- **[SR3R]** *Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting.* CVPR 2026. arXiv:2602.24020. Feed-forward GS for super-resolution 3D reconstruction. **OSS relevance:** Feed-forward GS SR shows the architecture is viable for single-pass inference, validating OSS's real-time ambitions.

- **[MoVieS]** Lin et al. *Motion-Aware 4D Dynamic View Synthesis in One Second.* CVPR 2026. arXiv:2507.10065. One-second 4D dynamic view synthesis with motion-aware Gaussians. **OSS relevance:** The "one second" claim on 4D content is evidence that 4D Gaussian representations are becoming tractable — v7's N-D canvas is on the right trajectory.

- **[PhysGM]** Lv et al. *Large Physical Gaussian Model for Feed-Forward 4D Synthesis.* CVPR 2026. arXiv:2508.13911. Feed-forward 4D Gaussian synthesis. Li Auto / BIT / HIT. **OSS relevance:** Industry (autonomous driving) investing in feed-forward 4D Gaussians — validates the direction. The "physical" constraint model may be relevant for OSS's canvas physics.

### Super-Resolution / Upscaling
- **[EfRLFN / StreamSR]** Bogatyrev et al. *Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content.* arXiv:2602.11339, Feb 2026. Comprehensive benchmark of 11 real-time SR models. New StreamSR dataset from YouTube. Proposes EfRLFN with tanh activation + efficient channel attention. Fine-tuning on StreamSR generalizes well. Code + dataset available. **OSS relevance:** StreamSR dataset could augment OSS training data (real-world streaming content with compression artifacts). The tanh activation is novel for real-time SR and worth testing in OSS's student backbone.

- **[NTIRE 2026 Efficient SR Challenge]** Ren et al. arXiv:2604.03198, Apr 2026. 95 registered, 15 valid submissions. Target: maintain ~26.90 dB PSNR on DIV2K_LSDIR while reducing runtime/params/FLOPs. **OSS relevance:** Key findings from the challenge report: re-parameterization is commonly adopted (validates OSS #9 RepVGG ablation). Affinity-based distillation between teacher and student networks is a trend — relevant to OSS's HAT-Tiny → nano-CNN distillation plan.

- **[INT8 SR]** Nguyen et al. *Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training.* CVPR 2026 MAI Workshop. arXiv:2604.20291. Three-stage: basic reconstruction → teacher distillation with Charbonnier + DCT supervision → QAT on fused deploy graph. 29.79 dB PSNR / 0.863 SSIM on 4K SISR at INT8. **OSS relevance:** The three-stage pipeline (train → distill → quantize) is a template for OSS's student deployment path. Confidence-weighted output-level distillation from Mamba teacher is directly applicable to OSS's HAT-Tiny → nano-CNN distillation. Weight clipping + BatchNorm recalibration for QAT stability are implementation details OSS should adopt.

### Frame Extrapolation / Generation
- **[GFFE]** Wu et al. (Intel/NVIDIA). *GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering.* ACM TOG 2024. arXiv:2406.18551. Eliminates G-buffer dependency — only needs color history. Novel heuristic framework handles disocclusions, lightweight shading correction network for final quality. Easier game integration than G-buffer-dependent methods. **OSS relevance:** GFFE is the closest academic precedent to OSS-FX's "no extra input" philosophy. Their disocclusion handling via motion fragment analysis could inform OSS-FX's spawner when canvas Gaussians go out of view. Intel-authored — may foreshadow XeSS frame gen direction.

- **[Decoupled Motion Prediction for GFFE]** Wu et al. *Decoupled Motion Prediction for Real-time G-buffer Free Frame Extrapolation.* SIGGRAPH 2026. ACM DOI: 10.1145/3746027.3754747. Extends GFFE by decoupling shading motion from geometry motion — shading variations propagated from historical frames. **OSS relevance:** Decoupled shading/geometry motion could simplify OSS-FX's canvas warp — warp geometry with MVs, but reuse shading from previous composite. Intel continues leading academic frame extrapolation work.

- **[X4Rate]** Dixit & Sarangi. *X4Rate: A Scene-Adaptive Extrapolation Framework for 4× Frame Rate Upsampling.* VISIGRAPP 2026. Generates 4× frames (e.g. 60→240 FPS) via scene-adaptive extrapolation. **OSS relevance:** 4× extrapolation is aggressive — if the technique works acceptably, it validates OSS-FX's ambition for multi-frame continuous α rendering without a frame-gen network.

- **[STSS]** He et al. *Low-latency Space-time Supersampling for Real-time Rendering.* AAAI 2024. arXiv:2312.10890. Unified SR + frame extrapolation in 4ms — 75% faster than two-stage 17ms pipeline. Random Reshading Masking + Efficient Reshading Module handles aliasing and warping holes uniformly. **OSS relevance:** This is the closest conceptual prior to OSS's "one architecture for SR+FG." Demonstrates that unified spatiotemporal supersampling is architecturally viable and was published at AAAI — validates OSS's core thesis that SR and FG should share primitives.

- **[ExtraSS]** Wu et al. *ExtraSS: A Framework for Joint Spatial Super Sampling and Frame Extrapolation.* SIGGRAPH Asia 2023. Another early joint SR+extrapolation framework from the same Intel lab. Uses G-buffers + motion vectors. **OSS relevance:** Paired with STSS and GFFE, shows the industry (Intel) is actively researching unified SR+extrapolation architectures — OSS is tracking the right direction.

- **[Consecutive Frame Extrapolation with Predictive Sparse Shading]** Wu et al. *SIGGRAPH Asia 2025 (TOG).* Neural network predicts which pixels need shading vs. can be extrapolated. Sparse shading reduces render load, neural extrapolation fills the rest. Code available. **OSS relevance:** The predictive sparse shading approach is the inverse of OSS (which renders everything and extrapolates motion). Complementary — OSS could potentially skip rendering static regions by leveraging canvas persistence, achieving similar render-load reduction.

### Super-Resolution / Upscaling (continued)
- **[FlashVSR]** Zhuang et al. *FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution.* CVPR 2026. arXiv:2510.12747. One-step diffusion VSR with block-sparse attention + Tiny Conditional Decoder. ~17 FPS at 768p, 12× speedup over prior diffusion VSR. Code + demo available. **OSS relevance:** FlashVSR proves diffusion-based VSR can approach real-time. The block-sparse attention pattern could inform OSS student backbone design. The Tiny Conditional Decoder architecture is a candidate distillation target structure.

- **[GameSR]** Mercier et al. *GameSR: Real-Time Super-Resolution for Game Streaming.* OpenReview. 1080p→4K at <16ms, up to 39.25 dB PSNR, SSIM >0.999, VMAF >93. Operates directly on rendered frames without source code access. **OSS relevance:** Validates that <16ms SR at 4K is achievable with lightweight models on game content. The "no source code access" design means it works from RGB frames only — same constraint as OSS's deployment path. PSNR numbers are a benchmark target.

### Inference Optimization
- **[FlashFormer]** *Whole-Model Kernels for Efficient Low-Batch Inference.* arXiv:2505.22758. Single fused kernel for entire transformer forward pass — eliminates intermediate tensor materialization. **OSS relevance:** While focused on LLMs, the whole-model-kernel approach could theoretically apply to OSS's HAT-Tiny backbone, reducing the number of kernel launches in the critical path. Worth watching for generalization to vision transformers.

## Archive.org Findings
- No significant archived versions of papers relevant to OSS were found. GS-STVSR, GSASR, GRAPE, and STSS papers remain accessible on arXiv directly — no takedowns detected.

## GitHub / Open Source
- **[OptiScaler](https://github.com/optiscaler/OptiScaler):** Active development — bridges upscaling/frame gen across GPUs. Can swap FSR 4 in place of DLSS/XeSS in unsupported games. Growing community. **OSS relevance:** OptiScaler proves the DLL-shim pattern works at scale — validates OSS's planned S7 integration path. Also means OSS could be tested in real games via OptiScaler once a DLL shim exists.

- **[GSASR](https://github.com/chrisdud0257/gsasr):** ICCV 2025. SoTA arbitrary-scale SR via 2D Gaussians. Code available. **OSS relevance:** Reference implementation for 2D-GS-based SR rasterization.

- **[fast-gaussian-rasterization](https://github.com/dendenxu/fast-gaussian-rasterization):** Geometry-shader-based global CUDA sorted high-performance 3DGS rasterizer, 5-10× speedup. **OSS relevance:** Could inform OSS's custom CUDA rasterizer optimization (currently using gsplat).

- **[gaussian-splatting (graphdeco-inria)]:** RTX 50-series (CUDA 12.8) training setup documented — relevant if OSS upgrades GPU hardware.

- **[NVIDIA RTXNTC](https://github.com/NVIDIA-RTX/RTXNTC):** Neural Texture Compression SDK v0.9.2. 85% VRAM reduction for PBR textures. **OSS relevance:** NTC + OSS upscaling could compound VRAM savings — render low-res textures (NTC-decompressed) → OSS-upscale. Synergistic with v6.2's VRAM budget analysis (#14).

- **[CVPR 2026 GS Papers](https://github.com/Awesome3DGS/3D-Gaussian-Splatting-Papers/blob/main/2026/CVPR.md):** 80 papers accepted at CVPR 2026 using 3DGS. Growth from ~30 at CVPR 2025. Gaussian splatting is now a major conference subfield — competitive pressure increasing.

## News
- **[CRITICAL] NVIDIA DLSS 5 announced at GTC 2026 (Mar 2026):** Arriving Fall 2026. Real-time neural rendering model that infuses pixels with photorealistic lighting and materials — not just another upscaler. Runs on single RTX 50 GPU at up to 4K. "Bridges the gap between rendering and reality." Uses learned models for lighting, material responses, and textures directly on the GPU. **OSS relevance:** This is a paradigm shift beyond SR+FG — DLSS 5 adds neural material/lighting enhancement as a third dimension. OSS's canvas architecture (engine-agnostic accumulation) is well-positioned to eventually incorporate neural material/lighting data as additional Gaussian properties. The Fall 2026 ship date means OSS needs v7-pico results before then for competitive positioning.

- **NVIDIA DLSS 4.5:** Launched Jan 2026 (SR) + Mar 31 2026 (Dynamic MFG). 2nd-gen transformer SR (linear color space, reduced ghosting, better AA). 6X Dynamic Multi Frame Generation (Blackwell only) — generates up to 5 frames per rendered frame with dynamic multiplier shifting. Now in 250+ games. **OSS relevance:** The dynamic MFG model (auto-shifting multipliers like a transmission) validates OSS-FX's continuous α∈(0,1) approach conceptually. DLSS 4.5's move to linear color space is architecturally interesting — OSS already operates in linear space via softplus activation. The quality bar is moving up; DLSS 4.5's 2nd-gen transformer will be the comparison target, not DLSS 4.

- **Intel XeSS 3 with Multi-Frame Generation:** Officially released (May 2026). XeSS 3 adds MFG across all Intel Arc GPUs (not just latest gen). Up to 3 additional AI-generated frames per rendered frame. Driver 32.0.101.8509 (WHQL). **OSS relevance:** Intel's strategy of bringing MFG to older Arc GPUs contrasts with NVIDIA's Blackwell-only MFG — signals that the frame-gen compute budget is manageable on older hardware. Supports OSS's thesis that frame extrapolation can be lightweight. Intel is aggressively building the full DLSS suite (XeSS-SR + XeSS-FG + XeSS-MFG).

- **Helldivers 2 "Optimizing Liberty" patch (May 28, 2026):** Simultaneously ships DLSS 4.5, FSR 4.0.3, XeSS 3.0, NVIDIA Reflex, and AMD Anti-Lag 2 — all in one title. **OSS relevance:** First major title to ship all three vendor upscalers simultaneously at current-gen versions. Normalizes multi-upscaler support as a standard game feature — strengthens the case for vendor-neutral solutions like OSS.

- **AMD FSR 4.1 expansion:** After community backlash, AMD announced FSR 4.1 (INT8 model) coming to RX 7000 series in July 2026 and RX 6000 series in early 2027. Previously RDNA 4 exclusive. **OSS relevance:** AMD capitulating on exclusivity validates the cross-vendor approach. The INT8 vs FP8 model distinction matters for OSS's deployment — if INT8 is "good enough" for AMD, OSS's student can target INT8 without quality concerns. OptiScaler already enables FSR 4 on older GPUs via INT8 — the community is bypassing vendor locks.

- **Apple Metal 4 + MetalFX:** WWDC 2025 (June 2025) — MetalFX now includes Frame Interpolation and Denoising, not just upscaling. Metal 4 adds tensor resources and ML encoder for GPU-timeline ML inference. **OSS relevance:** Apple is building the full DLSS suite (SR + FG + RR) into MetalFX. This makes Apple Silicon a legitimate fourth platform for OSS (NVIDIA, AMD, Intel, Apple). The Metal tensor resource + shader ML support is the API surface OSS's Metal backend would target.

- **[NEW] ARM Neural Frame Rate Upscaling (NFRU):** Announced at GDC 2026. Neural frame generation for mobile GPUs — claims 540p→1080p at 4ms per frame. Part of ARM's Neural Graphics Development Kit. **OSS relevance:** Mobile gaming is the next frontier for neural upscaling/frame gen. ARM's 4ms budget at mobile power is aggressive — validates OSS's <2ms desktop target. NFRU + Vulkan ML could make OSS viable on ARM laptops and Steam Deck-class handhelds via Vulkan compute backend.

- **AMD FSR Redstone (FSR 4.x):** SDK 2.2 brings sharper upscaling, better RT denoising, smoother FG — all ML-based on RDNA4 matrix cores. FSR 4 coming to RDNA 3/3.5/PS5 Pro by Q1 2026 (per MLiD rumors). **OSS relevance:** If FSR 4 reaches RDNA3, the cross-vendor ML-upscaling playing field expands — OSS's cross-vendor bet becomes more valuable, not less. FSR Radiance Caching (2026) for neural GI is a separate feature track.

- **NVIDIA Neural Texture Compression (NTC):** GTC 2026 session — 85% VRAM reduction for PBR textures. NTC + DLSS upscaling compound. **OSS relevance:** Could reduce the VRAM headroom OSS needs to coexist with game rendering in the same GPU memory budget.

- **CVPR 2026:** 4,090 accepted papers, 25.42% acceptance rate, +42.5% YoY. 3D Gaussian Splatting is now a major subfield (80 papers). **OSS relevance:** Academic attention to GS is accelerating, which means more competition but also more techniques to borrow.

## Action Items / Follow-ups
- [ ] **CRITICAL: Restart v7-pico-005 training.** GPU host offline ~2 days (now ~59h stale). Check 3080 Ti status. If unrecoverable, resume from step-9100 checkpoint on alternative hardware. This mirrors the v6.2-pico-002 GPU-reclamation pattern.
- [ ] **[NEW] DLSS 5 competitive threat assessment.** DLSS 5's neural material/lighting model (Fall 2026) goes beyond SR+FG. OSS's canvas architecture can theoretically absorb this (additional Gaussian properties for lighting/material), but we need to articulate how. The v7 canvas should be future-proof against this dimension — document the path from SR+FG → SR+FG+neural materials.
- [ ] **[NEW] Study Intel's frame extrapolation papers (GFFE + ExtraSS + Decoupled Motion Prediction + Predictive Sparse Shading).** Intel is the clear academic leader in game frame extrapolation. Decoupled shading/geometry motion (GFFE follow-up) could simplify OSS-FX's canvas warp. The predictive sparse shading approach (SIGGRAPH Asia 2025) is a complementary render-load-reduction technique.
- [ ] **[NEW] Evaluate STSS architecture against v7.** STSS (AAAI 2024) unified SR+extrapolation in 4ms — the closest prior art to OSS's thesis. Read deep, understand their Random Reshading Masking + Efficient Reshading Module. If STSS can do unified SR+FG in 4ms without Gaussians, why do Gaussians give OSS a better ceiling?
- [ ] **[NEW] ARM NFRU as a mobile target.** ARM's 540p→1080p at 4ms on mobile GPU validates <2ms desktop OSS target. Evaluate if OSS's student model could run on ARM Mali/Immortalis GPUs via Vulkan compute — this opens mobile gaming as an eventual platform.
- [ ] **Deep-read AA-2DGS (arXiv:2506.11252).** The object-space Mip filter could refine v7's rasterizer anti-aliasing. Code available — consider integrating the affine ray-splat Mip filter into OSS's rasterizer.
- [ ] **Test EfRLFN's tanh activation** in OSS student backbone candidates. Novel in real-time SR context. The StreamSR dataset could augment OSS training (real-world compression artifacts).
- [ ] **Adopt the 3-stage distillation-to-quantization pipeline** from arXiv:2604.20291 for the student deployment path: train → Charbonnier + DCT teacher distillation → QAT on fused graph with weight clipping + BN recalibration.
- [ ] **Track DLSS 4.5 quality numbers.** The 2nd-gen transformer SR in linear color space raises the quality bar. Also track DLSS 5 developments — paradigm shift beyond SR+FG.
- [ ] **Evaluate NTC + OSS compound.** If Neural Texture Compression ships widely, the VRAM budget for the OSS canvas + model becomes more generous.
- [ ] **Monitor CVPR 2026 GS papers** for techniques applicable to OSS's canvas/rasterizer. 80 GS papers — priority scan: SR3R, MoVieS, PhysGM, FlashVSR for 4D/feed-forward GS patterns and efficient diffusion VSR.
- [ ] **[NEW] Test FlashVSR's Tiny Conditional Decoder** as a candidate student backbone architecture for OSS's distillation target. Block-sparse attention pattern may be applicable.
