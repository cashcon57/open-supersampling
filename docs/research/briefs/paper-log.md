# Research paper log (auto-generated)

Entries the Hermes "OSS Research & Status" job appended to its local `BIBLIOGRAPHY.md` copies between 2026-05-28 and 2026-10-22. They were never committed until the 2026-10-22 repo consolidation. They are **unreviewed**: authors, venues and arXiv IDs were not checked, and some entries say so themselves. Promote an entry to the curated `BIBLIOGRAPHY.md` only after verifying it.

## From the desktop Hermes clone, May–Jun

## 7. New Additions (2026-05-31 Research Brief)

**[STGV]** Lin et al. (2026). *STGV: Spatio-Temporal Hash Encoding for Gaussian-based Video Representation*. arXiv:2604.10910
> Decomposes video features into 2D spatial + 3D temporal hash encodings, +0.98 dB PSNR over other GS video methods. Relevant to OSS's static-canvas + dynamic-spawner split — hash encoding could improve canvas memory efficiency.

**[TRiGS]** Yeom et al. (2026). *TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting*. arXiv:2604.00538
> SE(3) continuous motion + hierarchical Bézier residuals scales 4DGS to 600-1200 frames without memory explosion. Reference for v7+ canvas motion parameterization improvements for long game sessions.

**[RetimeGS]** Wang et al. (2026). *RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting*. CVPR 2026. arXiv:2603.13783
> Diagnoses 4DGS interpolation ghosting as "temporal aliasing"; optical-flow-guided initialization + triple-rendering supervision. Validates OSS's covariance-resampling approach to temporal anti-aliasing.

**[StreamSTGS]** Ke et al. (2026). *StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video*. AAAI 2026. arXiv:2511.06046
> Canonical GS as 2D image + temporal features as video stream, 170 KB/frame, adaptive bitrate. Reference for OSS canvas compression in cloud/streaming deployment scenarios.

**[ContinuousSR]** Peng et al. (2026). *Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling*. ICLR 2026. arXiv:2503.06617
> Pixel-to-Gaussian paradigm: single-pass Gaussian field construction, ~1ms arbitrary-scale rendering. **Direct competitor to OSS Gaussian canvas — requires benchmarking.** USTC + Huawei.

**[FlashVSR]** Zhuang et al. (2026). *FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution*. CVPR 2026. arXiv:2510.12747
> First one-step streaming diffusion VSR at ~17 FPS (768×1408, A100). Three-stage distillation + block-sparse attention + tiny decoder. Distillation pipeline + VSR-120K dataset relevant to OSS temporal training.

**[DUO-VSR]** Lv et al. (2026). *DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution*. CVPR 2026. arXiv:2603.22271
> Three-stage distillation: DMD + RFS-GAN dual-stream → preference-guided refinement. Dual-stream strategy directly applicable to OSS v6.2.1 distillation pipeline.

**[INT8-SISR]** Nguyen et al. (2026). *Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training*. CVPR 2026 MAI Workshop. arXiv:2604.20291
> Extract-refine-upsample + Mamba teacher → re-parameterizable INT8 student. QAT-on-fused-graph + weight clipping + BN recalibration recipe. **Directly actionable for OSS ≤1M-param student FP8/INT8 export.**

**[VFF-CSTVSR]** Becker et al. (2026). *Continuous Space-Time Video Super-Resolution with 3D Fourier Fields*. ICLR 2026. arXiv:2509.26325
> Unified 3D Video Fourier Field for spatio-temporal SR with analytical anti-aliasing PSF. Theoretical competitor to OSS N-D Gaussian canvas — both avoid decoupling spatial/temporal but use different representations (implicit Fourier vs. explicit Gaussian).

**[EfRLFN]** Bogatyrev et al. (2026). *Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content*. arXiv:2602.11339
> Efficient Channel Attention + tanh activation for real-time SR on compressed streaming content. StreamSR YouTube dataset. Low direct relevance but useful for degradation robustness testing.

**[GameSR]** (2026). *GameSR: Real-Time Super-Resolution for Interactive Gaming*. ICLR 2026.
> Engine-agnostic SR on RGB frames only, no engine hooks required. Competitor to monitor — if competitive quality without MVs, it undercuts OSS's "exact engine MVs are the advantage" thesis.

## 8. New Additions (2026-06-01 Research Brief)

**[3DGS³]** Zhao et al. (2026). *3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering*. arXiv:2605.11489.
> Unifies SR and frame interpolation as a post-rendering framework on 3DGS output; Gradient-Aware Super Sampling (GASS) leverages GS differentiability for GRU-based refinement. Closest conceptual analog to OSS's unified SR+FG thesis — validates the "single architecture" approach but as post-processing rather than native canvas.

**[VidSplat]** Tang et al. (2026). *VidSplat: Gaussian Splatting Reconstruction with Geometry-Guided Video Diffusion Priors*. SIGGRAPH 2026. arXiv:2605.11424.
> Training-free generative reconstruction using video diffusion priors for sparse-view completion. Geometry-guided, stage-wise denoising. The diffusion-prior-for-missing-data pattern is applicable to OSS's disocclusion hallucination and data augmentation problems.

**[SRSplat]** Hu et al. (2025). *SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images*. AAAI 2026 Oral. arXiv:2511.12040.
> Feed-forward Gaussian prediction with Reference-Guided Feature Enhancement (MLLM + diffusion external knowledge) + Texture-Aware Density Control (TADC). TADC's adaptive Gaussian density based on texture complexity is a candidate refinement for OSS's parent-child loss-adaptive density logic.

**[IFRVP]** Hirose et al. (2025). *IFRVP: Real-time Video Prediction With Fast Video Interpolation Model and Prediction Training*. ICIP 2024. arXiv:2503.23185.
> Extends frame interpolation models (IFRNet, convolution-only) for video prediction; ELAN residual blocks for speed. Direct competitor for OSS-FX's quality/speed extrapolation claims. The "extend interpolation for prediction" approach is an alternative to α-conditioned canvas rendering.

**[AccelVFI]** (2026). *Accelerator-Native Video Interpolation via Codec Motion Vector Priors*. arXiv:2603.26835.
> Real-time 30→60 FPS frame interpolation on mobile NPUs using codec MVs. Validates that even approximate motion vectors are sufficient for real-time frame interpolation on constrained hardware — OSS's exact engine MVs are a stronger signal.

**[3DGS³]** Zhao et al. (2026). *3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering*. arXiv:2605.11489.
> Unified post-rendering framework performing super sampling AND frame interpolation from low-resolution 3DGS renders. Gradient-Aware Super Sampling (GASS) + Lightweight Temporal Frame Interpolation (LTFI). Closest conceptual analog to OSS's "one architecture for SR+FG" thesis.

**[CDA-VSR]** Wang et al. (2026). *Compressed-Domain-Aware Online Video Super-Resolution*. CVPR 2026. arXiv:2603.07694.
> Motion-vector-guided deformable alignment for online VSR. Coarse warp from compressed MVs + local residual offsets. Residual-map-gated fusion suppresses mismatched regions. Directly applicable to OSS's canvas-warp design.

**[Mobile-GS]** Du et al. (2026). *Mobile-GS: Real-time Gaussian Splatting for Mobile Devices*. ICLR 2026. arXiv:2603.11531.
> First real-time GS on mobile: 116 FPS at 1600×1063 on Snapdragon 8 Gen 3. Proves GS rasterization is viable on mobile-class hardware — OSS's student model could target Steam Deck and mobile dGPUs.

**[INT8-SISR]** (2026). *Efficient INT8 Single-Image Super-Resolution via Deployment-Optimized Quantization*. MAI 2026 Challenge. arXiv:2604.20291.
> 29.79 dB PSNR / 0.8634 SSIM on 4K SR with INT8 quantization. Directly informs OSS's student model INT8/FP8 export strategy. Proves INT8 quality ceiling is viable for shipping upscaling.

**[NTIRE-2026-Efficient-SR]** Ren et al. (2026). *The Eleventh NTIRE 2026 Efficient Super-Resolution Challenge Report*. CVPR 2026 Workshop. arXiv:2604.03198.
> 95 registered, 15 valid submissions targeting ~26.90 dB PSNR on DIV2K_LSDIR with reduced runtime/params/FLOPs. Key trends: re-parameterization widely adopted, affinity-based teacher-student distillation common. Validates OSS's RepVGG ablation (#9) and HAT-Tiny → nano-CNN distillation path.

**[MoVieS]** Lin et al. (2026). *MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second*. CVPR 2026. arXiv:2507.10065.
> One-second 4D dynamic view synthesis with motion-aware Gaussians. Proves 4D Gaussian representations are becoming computationally tractable — validates OSS v7's N-D canvas trajectory. The motion-awareness in the Gaussian representation is analogous to v7's V_xt cross-correlation.

**[PhysGM]** Lv et al. (2026). *PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis*. CVPR 2026. arXiv:2508.13911.
> Feed-forward 4D Gaussian synthesis with physical constraints. Industry (Li Auto, autonomous driving) investment in 4D Gaussians validates the direction. The "physical" constraint model may inform OSS canvas physics for realistic motion.

**[STSS]** He et al. (2024). *Low-latency Space-time Supersampling for Real-time Rendering*. AAAI 2024. arXiv:2312.10890.
> Unified SR + frame extrapolation in 4ms — 75% faster than two-stage pipeline. Random Reshading Masking + Efficient Reshading Module handles aliasing and warping holes uniformly. Closest conceptual prior to OSS's "one architecture for SR+FG" thesis. Validates that unified spatiotemporal supersampling is architecturally viable.

**[GFFE]** Wu et al. (Intel/NVIDIA) (2024). *GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering*. ACM TOG 2024. arXiv:2406.18551.
> G-buffer-free frame extrapolation — only needs color history, novel disocclusion handling via motion fragment analysis, lightweight shading correction network. Closest academic precedent to OSS-FX's "no extra input" philosophy. Intel-authored — may foreshadow XeSS frame-gen approach.

**[ExtraSS]** Wu et al. (Intel) (2023). *ExtraSS: A Framework for Joint Spatial Super Sampling and Frame Extrapolation*. SIGGRAPH Asia 2023.
> Early joint SR+extrapolation framework from Intel using G-buffers + motion vectors. Paired with STSS and GFFE, establishes Intel as the academic leader in unified SR+extrapolation for games.

**[Predictive Sparse Shading]** Wu et al. (2025). *Consecutive Frame Extrapolation with Predictive Sparse Shading*. SIGGRAPH Asia 2025 (TOG).
> Neural network predicts which pixels need shading vs. can be extrapolated — reduces render load, neural extrapolation fills the rest. Complementary to OSS: OSS could skip rendering static regions by leveraging canvas persistence.

**[FlashVSR]** Zhuang et al. (2026). *FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution*. CVPR 2026. arXiv:2510.12747.
> One-step diffusion VSR, ~17 FPS at 768p, block-sparse attention + Tiny Conditional Decoder. Proves diffusion-based VSR can approach real-time. Tiny Conditional Decoder is a candidate distillation target for OSS's student backbone.

**[GameSR]** Mercier et al. (2024). *GameSR: Real-Time Super-Resolution for Game Streaming*. OpenReview.
> 1080p→4K at <16ms, up to 39.25 dB PSNR. Operates from RGB frames only — same constraint as OSS's deployment path. Validates <16ms 4K SR is achievable with lightweight models on game content.

**[X4Rate]** Dixit & Sarangi (2026). *X4Rate: A Scene-Adaptive Extrapolation Framework for 4× Frame Rate Upsampling*. VISIGRAPP 2026.
> 4× frame rate extrapolation (60→240 FPS) via scene-adaptive approach. Validates OSS-FX's continuous α rendering for multi-frame extrapolation.

**[Decoupled Motion GFFE]** Wu et al. (2026). *Decoupled Motion Prediction for Real-time G-buffer Free Frame Extrapolation*. SIGGRAPH 2026. ACM 10.1145/3746027.3754747.
> Extends GFFE by decoupling shading motion from geometry motion. Shading variations propagated from historical frames. Could simplify OSS-FX canvas warp: warp geometry with MVs, reuse shading from previous composite.

**[FlashFormer]** (2025). *FlashFormer: Whole-Model Kernels for Efficient Low-Batch Inference*. arXiv:2505.22758.
> Single fused kernel for entire transformer forward pass — eliminates intermediate tensor materialization. While LLM-focused, the whole-model-kernel approach could reduce kernel launch overhead in OSS's HAT-Tiny backbone.

## 9. New Additions (2026-06-03 Research Brief)

**[DeepFusionKernel]** Zhang et al. (2026). *Deep Kernel Fusion for Transformers*. arXiv:2602.11808.
> Deeply fused kernel for SwiGLU MLP blocks, 13.2% speedup on H100 over SGLang via HBM traffic reduction and cache reuse. LLM-focused but the pattern of fusing entire sub-networks into single kernels to reduce memory bandwidth is directly applicable to OSS's HAT-Tiny backbone optimization.

**[DLSS 4.5 Ray Reconstruction]** NVIDIA (2026). *DLSS 4.5 Ray Reconstruction — 2nd Gen Transformer Model*. Computex announcement, May 31 2026. Ships August 2026.
> Unifies denoising + Super Resolution into a single 2nd-gen transformer — 35% more compute, 20% more params, comparable performance. Deeper spatial awareness, better motion data utilization, expanded training dataset. Collapsing multiple tasks into one model validates OSS's "one architecture for SR+FG+denoising" thesis. Ships on all RTX GPUs.

**[DLSS 5]** NVIDIA (2026). *NVIDIA DLSS 5 — AI-Powered Breakthrough in Visual Fidelity for Games*. GTC March 2026, ships Fall 2026.
> Real-time neural rendering model infusing pixels with photorealistic lighting and materials — not just upscaling but a new product category (neural rendering). End-to-end scene semantic understanding (subsurface scattering, fabric sheen, hair interaction). Sets the 3-5 year quality target for OSS beyond SR+FG parity.

**[FSR 4.1 RDNA 3/2]** AMD (2026). *FSR 4.1 Upscaling for RDNA 3 (July 2026) and RDNA 2 (Q1 2027).*
> ML-powered FSR 4 upscaling extended to older AMD GPUs. Makes ML upscaling accessible on the widest GPU base. Shifts OSS's competitive positioning from "only open ML upscaler" to "only vendor-neutral unified SR+FG architecture with Apache 2.0 license."

**[XeSS 3 Multi-Frame Generation]** Intel (2026). *XeSS 3 with Multi-Frame Generation.*
> Intel enters the MFG race — up to 3 generated frames between rendered frames (3:1 ratio). Runs on Alchemist+ GPUs and Core Ultra CPUs. All three major vendors now offer MFG, making OSS-FX's zero-latency frame extrapolation the unique angle vs. interpolation-based approaches.

**[Snapdragon GSR 2 Neural]** Qualcomm (2026). *Neural Super Resolution and Frame Generation for Snapdragon Mobile Devices.* GDC 2026.
> Qualcomm developing neural SR + FG for mobile. Mobile is becoming a major upscaling battleground — validates OSS Pico tier targeting Steam Deck and mobile-class hardware.

## From the desktop research clone BIBLIOGRAPHY.md, May–Aug

## Research monitor additions — 2026-06-08
- 2026-06-08: **SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images** (arXiv:2511.12040) — OSS-relevant for feed-forward Gaussian SR and runtime student design patterns.
- 2026-06-08: **DiffST: Spatiotemporal-Aware Diffusion for Real-World Space-Time Video Super-Resolution** (arXiv:2605.13182) — OSS-relevant as a space-time SR + frame-interpolation reference for unified SR/FG training.
- 2026-06-08: **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483) — OSS-relevant game supersampling baseline/dataset reference.

**[ConFi-GS]** Li et al. (2026). *ConFi-GS: Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution*. arXiv:2605.24964
> Reliability-aware high-frequency injection for GS-SR; candidate for OSS confidence-gated splat spawning/densification and pseudo-label filtering.

**[FlowGS]** Mo, Lu, Wu (2026). *Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution*. arXiv:2605.22147
> Uses 2D Gaussian splatting as a continuous feature field for arbitrary-scale SR; supports OSS's 2D-GS representation choice, though domain is remote sensing.

**[RiGS]** Wu et al. (2026). *RiGS: Rigid-aware 4D Gaussian Splatting from a Single Monocular Video*. arXiv:2605.23672
> Static/rigid/transient Gaussian decomposition with temporal-scale-aware motion; reference for v7 persistent vs transient canvas primitives.

**[CDA-VSR]** Wang et al. (2026). *Compressed-Domain-Aware Online Video Super-Resolution*. arXiv:2603.07694
> Uses supplied motion vectors, residual maps, and frame types for fast online VSR; strong prior for OSS exact-engine-motion-vector compute reuse and gating.

**[ReFrame]** Liu and Aamodt (2025). *ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering*. arXiv:2506.13814
> Temporal feature caching for neural rendering workloads; candidate optimization for OSS teacher/student inference under high frame coherence.

**[Liquid-D3DGS]** Li, Pal, Tan (2026). *Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting*. arXiv:2606.07670
> Continuous-time CfC/LNN deformation field for dynamic 3DGS; potential residual-motion prior for v7 t-axis interpolation/extrapolation.

**[EvoGS]** Shi, Gasparini, Morin, Ooi (2026). *EvoGS: Constructing Continuous-Layered Gaussian Splatting with Evolution Tree for Scalable 3D Streaming*. arXiv:2606.07179
> Wavelet-inspired parent-child evolution tree that reduces Gaussian redundancy/VRAM; direct comparison target for v7 parent-child density and packed-canvas work.

**[3DGS3]** Zhao, Gao, Cai, Liu (2026). *3DGS$^3$: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering*. arXiv:2605.11489
> Joint 3DGS post-rendering super sampling + frame interpolation; closest new competitive paper to OSS's unified SR+FG positioning after GS-STVSR.

**[LEGS]** Guo, Huo, Sun, Gong (2026). *LEGS: Laplacian-Enhanced Gaussian Splatting with a Nonlinear Weighted Loss*. arXiv:2606.07932
> Loss-only second-order edge/detail weighting for Gaussian optimization; cheap candidate ablation for OSS raster/detail recovery.
---

## Research monitor additions — 2026-06-09

**[LiteVSR]** Cao et al. (2026). *LiteVSR: Lightweight Adaptation of Frozen Diffusion Transformers for Video Super-Resolution*. arXiv:2606.09250
> Frozen-DiT VSR with a lightweight state-aware adapter and one-step sampling compatibility; relevant to OSS teacher/student distillation and efficient temporal SR adaptation, though too diffusion-heavy for the shipping path.

**[REFINE]** Chen et al. (2026). *REFINE: Super-efficient 3D Gaussian Splatting Pruning via Rendering-Free Primitive Importance*. arXiv:2606.09074
> Rendering-free Hessian/proxy primitive-importance metric for 3DGS pruning; relevant to v7 parent-child density/pruning cost control and a candidate replacement/ablation against current loss-adaptive density heuristics.

**[NGram-MoSE]** Huang, Bui, Chuang (2026). *NGram-MoSE: Efficient Remote Sensing Super-Resolution via N-Gram Context and Mixture-of-Experts*. arXiv:2606.08535
> Lightweight transformer SR with N-gram context injection for window-boundary artifacts and sparse MoE FFNs; useful for OSS HAT-window artifact mitigation and student efficiency ideas despite non-game remote-sensing domain.

**[UltraFlash]** Luxury et al. (2026). *Ultra Flash: Scaling Real-Time Streaming Video Generation to High Resolutions*. arXiv:2606.09150
> Cascaded streaming video-generation SR with causal latent upsampler, sparse causalization, single-step distillation, and dynamic cache management; mostly generative-video rather than game SR, but relevant as evidence that cache-aware temporal upsampling is moving toward real-time high-res.

**[Envision4D]** Song et al. (2026). *Envision4D: Envisioning Visual Futures via Feed-forward 4D Gaussian Splatting for Autonomous Driving*. arXiv:2606.10656
> Feed-forward 4D-GS future extrapolation with temporal attention and pose prediction; relevant to OSS-FX as external evidence that Gaussian primitives are becoming a future-frame prediction substrate, though it targets autonomous-driving scenes rather than game-engine exact motion vectors.
## Research brief 2026-06-10
- Jiaxiang Li et al. **ConFi-GS Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution**. arXiv:2605.24964, 2026-05-24. Relevance: direct Gaussian-splatting SR detail recovery / confidence-guided high-frequency injection candidate for OSS canvas detail. http://arxiv.org/abs/2605.24964v1
- Jiangwei Mo, Xi Lu, Hanlin Wu. **Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution**. arXiv:2605.22147, 2026-05-21. Relevance: continuous-scale GS SR; compare arbitrary-scale representation and rasterization costs against OSS v7. http://arxiv.org/abs/2605.22147v1
- Shibo Liu. **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors**. arXiv:2603.26835, 2026-03-27. Relevance: motion-vector-prior interpolation for accelerator-native VFI; strong OSS-FX design analogue. http://arxiv.org/abs/2603.26835v3
- Ce Wang, Zhenyu Hu, Wanjie Sun. **SlimDiffSR: Toward Lightweight and Efficient Remote Sensing Image Super-Resolution via Diffusion Model Distillation**. arXiv:2605.02198, 2026-05-04. Relevance: lightweight SR distillation patterns for teacher→student compression, despite non-game domain. http://arxiv.org/abs/2605.02198v2
- Jędrzej Maczan. **Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers**. arXiv:2604.02344, 2026-02-09. Relevance: cross-vendor dispatch-overhead data relevant to OSS browser/WebGPU or Vulkan-style backend decisions. http://arxiv.org/abs/2604.02344v1
- Yong Liu et al. **Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution**. arXiv:2308.02794, 2023-08-05. Relevance: deployment-friendly SR transformer block for possible teacher simplification / student distillation reference. http://arxiv.org/abs/2308.02794v1
- Antoine Mercier et al. **Efficient neural supersampling on a novel gaming dataset**. arXiv:2308.01483, 2023-08-03. Relevance: game-specific neural supersampling dataset/baseline for competitive comparison. http://arxiv.org/abs/2308.01483v1
- Jaber Jaber, Osama Jaber. **AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis**. arXiv:2606.09682, 2026-06-08. Relevance: persistent/megakernel fusion reference for reducing dispatch overhead in OSS student+raster/composite inference. http://arxiv.org/abs/2606.09682v1
- Haotong Qin et al. **QuantSR+: Pushing the Limit of Quantized Image Super-Resolution Networks**. arXiv:2605.22351, 2026-05-21. Relevance: low-bit SR quantization reference for OSS FP8/INT8 student deployment and calibration. http://arxiv.org/abs/2605.22351v1
- Yongfei Guo et al. **LEGS: Laplacian-Enhanced Gaussian Splatting with a Nonlinear Weighted Loss**. arXiv:2606.07932, 2026-06-06. Relevance: cheap Gaussian detail/edge loss candidate for v7 teacher training. http://arxiv.org/abs/2606.07932v1
## Research brief 2026-06-11

- Adrian Ramlal, John S. Zelek. **PersistGS: Differentiable Physics for Object Permanence in 4D Gaussian Splatting**. arXiv:2606.03479, 2026-06-02. Relevance: differentiable rigid-body/object-permanence prior for persistent Gaussian canvases under occlusion; candidate for OSS disocclusion handling. http://arxiv.org/abs/2606.03479v1
- Denis Gridusov, Maxim Popov, Sergey Kolyubin. **R5DGS: Semantic-Aware 4D Gaussian Splatting with Rigid Body Constraints for Efficient Dynamic Scene Reconstruction**. arXiv:2605.25909, 2026-05-25. Relevance: semantic identity + rigid-body constrained dynamic Gaussians for efficient future extrapolation; candidate reference for static/rigid/transient canvas decomposition. http://arxiv.org/abs/2605.25909v1
- Pham Phuong Nam Nguyen et al. **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training**. arXiv:2604.20291, 2026-04-22. Relevance: deployment-aware quantized SR student training with reparameterizable backbone, DCT supervision, confidence-weighted loss; direct reference for OSS ≤1M-param INT8/FP8 student distillation. http://arxiv.org/abs/2604.20291v1
- Christopher Thirgood, Dipon Kumar Ghosh, Simon Hadfield. **TIDES: Time-Derivative Event Simulation via Deformable Reconstruction**. arXiv:2606.02058, 2026-06-01. Relevance: continuous-time dynamics from dynamic Gaussian splatting; possible alpha-time derivative/math reference for OSS-FX supervision. http://arxiv.org/abs/2606.02058v1

## Added by research brief 2026-06-12

**[VeloGauss]** Lu and Zhao (2026). *VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos*. arXiv:2605.10567
> Per-Gaussian velocity fields overlap OSS v7 native time/cross-covariance design and may inform OSS-FX velocity losses.

**[RDNU]** Adam-AtlasSoftware (2026). *Radeon Decoupled Neural Upscaler*. GitHub: Adam-AtlasSoftware/RDNU
> Open-source RDNA3/G-buffer-guided neural upscaler to monitor as a potential cross-vendor competitor/baseline.

## Research brief 2026-06-16

- Zhenhua Wu et al. **RealityBridge: Bridging Editable 3D Gaussian Splatting Driving Simulations and Real-World Videos**. arXiv:2606.16278, 2026-06-15. Relevance: 3DGS-specific artifact repair and temporal-flicker reduction for synthetic driving videos; useful for OSS capture realism and visual-QA failure modes. http://arxiv.org/abs/2606.16278v1
- Mijeong Kim, Jungtaek Kim, Bohyung Han. **GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes**. arXiv:2604.02915, 2026-04-03. Relevance: uncertainty-aware Gaussian motion/extrapolation reference for OSS-FX confidence gating and sparse/disoccluded regions. http://arxiv.org/abs/2604.02915v1
- Ole Beisswenger, Jan-Niklas Dihlmann, Hendrik P. A. Lensch. **FrameDiffuser: G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering**. arXiv:2512.16670, 2025-12-18. Relevance: G-buffer-conditioned future-frame synthesis is competitive prior art/teacher-data inspiration, though too slow/stochastic for OSS shipping inference. http://arxiv.org/abs/2512.16670v1

## Research brief 2026-06-17

- Faruk Alpay, Levent Sarioglu, Yaser Hadri. **Renderable Partial Representations for Dynamic Gaussian Splatting under Incomplete Delivery**. arXiv:2606.17212, 2026-06-15. Relevance: partial dynamic-Gaussian delivery/recovery model for robust canvas subsets, degraded rendering, and watcher/checkpoint failure modes. http://arxiv.org/abs/2606.17212v1
- Zheng Dong, Daifei Qiu, Pinxuan Dai, Ke Xu, Jiamin Xu. **TurboGS: Accelerating 3D Gaussian Splatting via Error-Guided Sparse Pixel Sampling and Optimization**. arXiv:2606.15924, 2026-06-14. Relevance: error-guided sparse sampling/optimization could inform v7 teacher training speed and loss-adaptive density ablations. http://arxiv.org/abs/2606.15924v1
- Xiang Gao et al. **teasr: training-efficient any-step diffusion transformer for real-world image super-resolution**. arXiv:2606.16188, 2026-06-15. Relevance: efficient any-step diffusion SR is not a shipping runtime path, but may be useful as a stronger teacher/pseudo-target source for student distillation. http://arxiv.org/abs/2606.16188v1

## Research brief 2026-06-18

- Jeongmin Bae et al. **Hand-4DGS: Feed-Forward 3D Gaussian Splatting for 4D Hand Reconstruction from Egocentric Videos**. arXiv:2606.19156, 2026-06-17. Relevance: feed-forward ~60 FPS dynamic Gaussian reconstruction; useful temporal-conv/structural-prior reference for lightweight OSS motion/canvas updates. https://arxiv.org/abs/2606.19156v1
- Fangyi Li et al. **GSPan: A Continuous Gaussian Primitive Representation for Arbitrary-Scale Pansharpening**. arXiv:2606.17722, 2026-06-16. Relevance: rendered 2D-Gaussian residual detail fields for arbitrary-scale SR-like reconstruction; compare against OSS canvas residual/detail injection. https://arxiv.org/abs/2606.17722v1
- Jia Li et al. **Neural Super-Resolution for Real-time Rendering with Radiance Demodulation**. arXiv:2308.06699, 2023-08-13. Relevance: real-time rendering SR using lighting/material demodulation for temporal stability and sharp texture; candidate G-buffer/material decomposition baseline for OSS. https://arxiv.org/abs/2308.06699v2
- Dachun Kai et al. **EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution**. arXiv:2606.13580, 2026-06-11. Relevance: separates motion alignment from high-frequency texture enhancement; useful analogy for OSS teacher/student texture/detail losses, despite event-camera domain. https://arxiv.org/abs/2606.13580v1

## Research brief 2026-06-19
- Yuhan Chen, Kunyang Huang, Fuchen Li. **AIGS-Net: Compact Illumination Field Modeling via 2D Gaussian Splatting for Fast Low-Light Image Enhancement**. arXiv:2606.17998, 2026-06-16. Relevance: another 2D-GS-as-image-operator datapoint; opacity/compact Gaussian basis design may transfer to low-cost canvas correction layers. https://arxiv.org/abs/2606.17998v1
- Mufan Liu, Qi Yang, Miaoran Zhao. **Progressively Deformable 2D Gaussian Splatting for Video Representation at Arbitrary Resolutions**. arXiv:2503.05600, 2025-03-07. Relevance: close representation-level precedent for persistent 2D Gaussian video/canvas state and progressive density ordering. https://arxiv.org/abs/2503.05600v2
- Zhengyao Lv, Menghan Xia, Xintao Wang. **DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution**. arXiv:2603.22271, 2026-03-23. Relevance: useful teacher→student training pattern for compressing high-quality temporal restoration into a cheap student. https://arxiv.org/abs/2603.22271v1

## Research brief 2026-06-20
- Hasan Yazar et al. **Intrinsic 4D Gaussian Segmentation from Scene Cues**. arXiv:2606.18623, 2026-06-17. Relevance: mask-free primitive affinity graph over appearance/orientation/scale/deformation/rendered-boundary cues; candidate for OSS canvas culling, object-aware stability, and adaptive quality masks. https://arxiv.org/abs/2606.18623v1
- Ruiqing Wang et al. **MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution**. arXiv:2603.20690, 2026-03-21. Relevance: one-step flow/diffusion SR distillation pattern to compare against HAT-Tiny teacher → ≤1M-param student compression. https://arxiv.org/abs/2603.20690v1
- Nengbo Lu, Minghua Pan. **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations**. arXiv:2605.10586, 2026-05-11. Relevance: Gaussian-state future prediction/extrapolation reference; possible long-horizon OSS-FX regularization source beyond single-frame alpha interpolation. https://arxiv.org/abs/2605.10586v1
- Ashim Dhor, Rasel Mondal, Pin Yu Chen. **ITNet: A Learnable Integral Transform That Subsumes Convolution, Attention, and Recurrence**. arXiv:2606.19538, 2026-06-17. Relevance: speculative unified/fused operator reference for future canvas-update kernels that combine convolution, attention, and temporal recurrence. https://arxiv.org/abs/2606.19538v1

## Research brief 2026-06-21
- Max Defez, Filippo Quarenghi, Mathieu Vrac, Stephan Mandt, Tom Beucler. **A Scale-Adaptive Framework for Joint Spatiotemporal Super-Resolution with Diffusion Models**. arXiv:2604.21903, 2026-04-23. Relevance: scale-conditioned joint spatial+temporal SR reference for OSS unified SR+FX training; diffusion/climate domain means teacher/conditioning ideas only, not shipping runtime. https://arxiv.org/abs/2604.21903v1
- Stylianos I. Venieris, Mario Almeida, Royson Lee, Nicholas D. Lane. **NAWQ-SR: A Hybrid-Precision NPU Engine for Efficient On-Device Super-Resolution**. arXiv:2212.09501, 2022-12-15. Relevance: older but still useful hybrid-precision/quantized SR deployment reference for OSS INT8/FP8 student planning. https://arxiv.org/abs/2212.09501v3

## Research brief 2026-06-22
- Prateek Poudel, Prashant Aryal, Kirtan Kunwar, Navin Nepal, Dinesh Baniya Kshatri. **Wavelet-Space Representations for Neural Super-Resolution in Rendering Pipelines**. arXiv:2508.16024, 2025-08-22. Relevance: G-buffer/history-aware stationary-wavelet SR for rendering pipelines; candidate frequency/detail head or loss for OSS HAT teacher and ≤1M-param student shimmer control. https://arxiv.org/abs/2508.16024v3

## Research brief 2026-06-23

- Yehonathan Litman et al. **Lift4D: Harmonizing Single-View 3D Estimation for 4D Reconstruction In-the-Wild**. arXiv:2606.23688, 2026-06-22. Relevance: occlusion-aware deformable 3DGS recovery with temporally consistent single-view priors; candidate teacher/offline reference for OSS persistent-canvas disocclusion handling. https://arxiv.org/abs/2606.23688v1
- Jiale Shi et al. **GaussianZoom: Progressive Zoom-in Generative 3D Gaussian Splatting with Geometric and Semantic Guidance**. arXiv:2605.18252, 2026-05-18. Relevance: continuous LOD and visibility modulation for alias-free cross-scale 3DGS rendering; potential reference for v7 parent-child density / packed-canvas budgets. https://arxiv.org/abs/2605.18252v1
- Emanuele Artioli et al. **Thin-Client Interactive Gaussian Adaptive Streaming over HTTP/3**. arXiv:2605.08699, 2026-05-09. Relevance: QUIC/HTTP3 streaming of view-dependent 2D Gaussian projections; useful if OSS explores canvas/history serialization or thin-client deployment. https://arxiv.org/abs/2605.08699v1
- Hidir Yesiltepe et al. **DTG-Restore: Training-Free Diffusion Refinement for Generative Video Super-Resolution**. arXiv:2605.30431, 2026-05-28. Relevance: diffusion-based temporal VSR refinement with geometry-preserving guidance; possible offline teacher/pseudo-target source, not shipping runtime. https://arxiv.org/abs/2605.30431v1

## Research brief 2026-06-24

- Songyin Wu, Deepak Vembar, Anton Sochenov, Selvakumar Panneer. **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering**. arXiv:2406.18551, 2024-05-23. Relevance: low-latency neural frame-extrapolation baseline/competitor for OSS-FX positioning and artifact/metric comparison; OSS differs by using exact engine motion vectors plus persistent Gaussian canvas instead of a standalone extrapolator. https://arxiv.org/abs/2406.18551v1

## Research brief 2026-06-25
- Vikram Sandu, Mayurdeep Pathak, Rajiv Soundararajan. **Temporally Aware Densification for Dynamic 3D Gaussian Splatting**. arXiv:2606.23212v1, 2026-06-22. Relevance: temporally aware densification for dynamic 3DGS; direct comparison target for OSS v7 parent-child/loss-adaptive density and pruning heuristics. https://arxiv.org/abs/2606.23212v1
- Rui Wang, Quentin Lohmeyer, Siyu Tang et al.. **Multi4D: High-Fidelity Dynamic Gaussian Splatting via Multi-Level Competitive Allocation**. arXiv:2606.22197v1, 2026-06-20. Relevance: multi-level competitive allocation for dynamic Gaussian splatting; useful for balancing motion consistency against high-frequency fidelity in v7 canvas updates. https://arxiv.org/abs/2606.22197v1
- Yu-Syuan Xu, Hao-Lun Sun, Hao-Wei Chen et al.. **Fidelity- and Perception-Aware Local Implicit Attention for Arbitrary-Scale Image Super-Resolution**. arXiv:2606.21910v1, 2026-06-20. Relevance: arbitrary-scale SR with fidelity/perception-aware local implicit attention; possible teacher-loss/head reference for continuous-scale OSS SR. https://arxiv.org/abs/2606.21910v1
- Yinuo Zhang, Guangshun Wei, Yuanfeng Zhou et al.. **UniRED: Unified RGB-D Video Frame Interpolation with Event Guidance**. arXiv:2606.24282v1, 2026-06-23. Relevance: RGB-D/event-guided frame interpolation; useful external artifact/conditioning reference for OSS-FX even though game inputs differ. https://arxiv.org/abs/2606.24282v1
- Spencer H. Bryngelson. **ANEForge: Python for direct computation on the Apple Neural Engine**. arXiv:2606.17090v1, 2026-06-12. Relevance: direct Apple Neural Engine programming; informs feasibility of future Apple/Metal/ANE student inference path beyond CoreML. https://arxiv.org/abs/2606.17090v1
- Ali Asaria, Tony Salomone, Deep Gandhi. **Realizing Native INT8 Compute for Diffusion Transformers on Consumer GPUs: A Fused INT8 GEMM Kernel for Ideogram 4.0**. arXiv:2606.14598v1, 2026-06-12. Relevance: fused INT8 GEMM lessons on consumer GPUs; warns OSS student quantization needs real fused kernels, not just nominal INT8 export. https://arxiv.org/abs/2606.14598v1

## Research brief 2026-06-27
- Youyang Du, Junqiu Zhu, Zheng Zeng, Lu Wang, Lingqi Yan. **Neural Image Space Tessellation effect**. arXiv:2602.23754v2, 2026-05-27. Relevance: real-time screen-space neural G-buffer post-process using depth and geometric/shading normals; candidate artifact/detail/silhouette cleanup reference for OSS capture channels, losses, or lightweight post-heads. https://arxiv.org/abs/2602.23754v2

## Research brief 2026-06-28
- Laurent Belcour. **Neural Texture Compression using Hypernetworks**. arXiv:2606.26913, 2026-06-25. Relevance: real-time neural texture decoder/hypernetwork patterns, including learned super-resolution decoders, may inform tiny material-aware OSS student heads or G-buffer-conditioned detail recovery. https://arxiv.org/abs/2606.26913
- Yitong Li et al. **Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation**. arXiv:2606.23743, 2026-06-21. Relevance: full-stack cache/sparse-attention/token-pruning/quantization/kernel-fusion co-optimization checklist for OSS student+raster deployment. https://arxiv.org/abs/2606.23743
- Haoqian Meng et al. **SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference**. arXiv:2606.26587, 2026-06-25. Relevance: outlier-aware sparse+dense low-bit activation path and fused preparation kernels are useful references for future OSS FP8/FP4 student inference kernels. https://arxiv.org/abs/2606.26587

## Research brief 2026-06-29
- Paweł Katarzyński. **Fewer, Better Frames: A Compute-Normalized Proof of Concept for Coherence-First World-Model Rendering with Model-Guided FSR4 Frame Generation**. arXiv:2606.02586, 2026-05-11. Relevance: competitive-positioning lead for FSR4-guided frame generation in world-model/rendering loops; useful counterpoint for OSS-FX's exact-motion-vector Gaussian-canvas extrapolation. https://arxiv.org/abs/2606.02586v1

## Research brief 2026-06-30
- Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto. **Resonant Brane Splatting for Arbitrary-Scale Super-Resolution**. arXiv:2606.29453v1, 2026-06-28. Relevance: higher-order Gaussian-Hermite/Brane primitives encode high-frequency detail with fewer overlaps; candidate primitive/raster ablation for OSS v7 edge/detail recovery. https://arxiv.org/abs/2606.29453v1
- Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto. **Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution**. arXiv:2606.29400v1, 2026-06-28. Relevance: feed-forward adaptive 2D-Gaussian allocation/densification for arbitrary-scale SR; direct comparison target for OSS v7 parent-child density and packed-canvas budgeting. https://arxiv.org/abs/2606.29400v1
- S. M. A. Sharif, Abdur Rehman, Seongwan Kim, Jaeho Lee. **Decoder-Free Distillation for Quantized Image Restoration**. arXiv:2603.09624v1, 2026-03-10. Relevance: quantized low-level-vision student training with bottleneck-only distillation and learnable gradient/loss balancing; useful for OSS INT8/FP8 student distillation. https://arxiv.org/abs/2603.09624v1

## Research brief 2026-07-01
- Yujin Tang, Tian Zhou, Xin Lin, Cheng Tan, Yifan Hu. **Learning Video Dynamics with Predictive Differentiable Rendering**. arXiv:2606.31050v1, 2026-06-30. Relevance: future-frame prediction with internal 3DGS state and differentiable rendering; direct comparison for OSS-FX Gaussian-state video dynamics. https://arxiv.org/abs/2606.31050v1
- Shengjun Zhang, Jinzhao Li, Xin Fei, Yueqi Duan. **Learning Efficient 4D Gaussian Representations from Monocular Videos with Flow Splatting**. arXiv:2606.29976v1, 2026-06-29. Relevance: efficient flow-splatting 4D Gaussian motion representation; compare against v7 native time-axis/cross-covariance canvas motion encoding. https://arxiv.org/abs/2606.29976v1
- Christos Anagnostopoulos, Alexandros Gkillas, Nikos Piperigkos, Aris S. Lalos. **Fast and Accurate Outlier-Aware LiDAR Super-Resolution for SLAM Applications**. arXiv:2606.28607v1, 2026-06-26. Relevance: robust outlier-aware real-time SR update pattern; possible analogue for OSS disocclusion/noisy-G-buffer gating despite LiDAR domain mismatch. https://arxiv.org/abs/2606.28607v1

## Research brief 2026-07-03
- Geunhyuk Youk, Jeonghyeok Do, Dayeon Kim, Jihyong Oh, Munchurl Kim. **AVSR-Diff: Scale-Agnostic Diffusion Priors for Temporally Consistent Arbitrary-Scale Video Super-Resolution**. arXiv:2607.00987v1, 2026-07-01. Relevance: temporal arbitrary-scale VSR with temporally-gated recurrence and scale-aware Fourier refinement; useful teacher/pseudo-target reference for OSS v7 training and student distillation, not a shipping runtime path. https://arxiv.org/abs/2607.00987v1
- Zhicheng Geng, Luming Liang, Tianyu Ding, Ilya Zharkov. **RSTT: Real-time Spatial Temporal Transformer for Space-Time Video Super-Resolution**. arXiv:2203.14186v1, 2022-03-27. Relevance: older single-model STVSR baseline for unified spatial upscaling + temporal interpolation positioning; far above OSS target budget but useful competitive/history reference. https://arxiv.org/abs/2203.14186v1

## Research brief 2026-07-04
- Simon Pochinda, Momen K. Tageldeen, Mark Thompson, Tony Rinaldi, Troy Giorshev, Keith Lee, Jie Zhou, Frederick Walls. **LCS: An AI-based Low-Complexity Scaler for Power-Efficient Super-Resolution of Game Content**. arXiv:2507.22873v1, 2025-07-30. Relevance: low-complexity, quantized/reparameterized game-content scaler for NPU/low-power deployment; useful baseline for OSS Pico/handheld ≤1M-param student positioning. https://arxiv.org/abs/2507.22873v1
- Alexander Becker, Julius Erbach, Dominik Narnhofer, Konrad Schindler. **Continuous Space-Time Video Super-Resolution with 3D Fourier Fields**. arXiv:2509.26325v2, 2025-09-30. Relevance: continuous spatial+temporal video representation with analytic Gaussian point-spread anti-aliasing; strong comparison point for OSS v7 continuous time-axis rendering and anti-aliasing design. https://arxiv.org/abs/2509.26325v2
- Huimin Zeng, Yue Bai, Yun Fu. **Arbitrary-Scale 3D Gaussian Super-Resolution**. arXiv:2508.16467v2, 2025-08-22. Relevance: scale-aware 3DGS rendering plus progressive arbitrary-scale SR; useful if OSS extends beyond fixed integer upscales. https://arxiv.org/abs/2508.16467v2

## Research brief 2026-07-05
- Jianwei Hu, Tingxuan Huang, Hengyu Zhou, Ningna Wang, Xiaohu Guo, Jinshan Lai, Bin Wang. **MVFusion-GS: Motion-Variance Guided Temporal Attention for High-Quality Dynamic Gaussian Splatting**. arXiv:2607.01578v1, 2026-07-02. Relevance: per-Gaussian motion-variance refinement and temporal attention for dynamic/static separation; direct comparison target for OSS v7 `V_xt` motion encoding, temporal coherence, and loss-adaptive density/pruning heuristics. https://arxiv.org/abs/2607.01578v1

## Research brief 2026-07-06
- Yuhongze Zhou, Zihao Yang, Xinxin Zuo. **TemporalGS: Training-Free Plug-and-Play Acceleration for 3D Gaussian Splatting Rendering via Temporal Priors**. arXiv:2607.03390v1, 2026-07-03. Relevance: training-free temporal-prior acceleration for 3DGS rendering; direct study target for OSS v7 canvas rendering cost, temporal cache reuse, and active-mask pruning. https://arxiv.org/abs/2607.03390v1

## Research brief 2026-07-07
- Lucas Yunkyu Lee, Soonho Kim, Youngwook Kim, Sangmin Kim, Jaesik Park. **FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles**. arXiv:2605.03337v3, 2026-05-05 / updated 2026-07-01. Relevance: analyzes temporal partitioning, Gaussian durations, and photometric-vs-motion decoupling in dynamic 4DGS; direct reference for OSS v7 native time-axis, density/pruning, and motion-loss design. https://arxiv.org/abs/2605.03337v3

## Research brief 2026-07-08
- Keito Abe, Kaede Shiohara, Takashi Otonari, Toshihiko Yamasaki. **GUSH3R: Everyone Everywhere All at Once as Gaussians**. arXiv:2607.05243v1, 2026-07-06. Relevance: feed-forward dynamic human-scene Gaussian reconstruction; off-domain but useful for transient/dynamic-object canvas separation ideas in OSS v7/OSS-FX. https://arxiv.org/abs/2607.05243v1

## Research brief 2026-07-09
- Yimo Wang, Bin Kang, Shuojue Yang, Yueming Jin. **DeGenseGS: Geometrically and Semantically Decoupled Surgical Scene Understanding in 4D Gaussian Splatting**. arXiv:2607.04761v1, 2026-07-06. Relevance: off-domain surgical 4DGS, but geometry/semantic decoupling may inform future OSS object-aware/transient canvas channels. https://arxiv.org/abs/2607.04761v1
- Lihua Wei, Huatong Gao, Jia Gong, Zhiyu Tan, Hao Li, Jun Liu, Zhih Ren. **PhyMRI-SR: Toward Physics-Aware MRI Image Super-Resolution**. arXiv:2607.06238v1, 2026-07-07. Relevance: off-domain MRI SR using 2D Gaussian Splatting / coordinate reconstruction; useful only as a degradation-physics-aware SR reference, not a game-runtime lead. https://arxiv.org/abs/2607.06238v1

## Research brief 2026-07-10
- Zhihui Ke, Yuyang Liu, Xiaobo Zhou, Tie Qiu. **DLGStream: Dynamic Language-embedded Guassian Splatting for Open-vocabulary Enabled Free-viewpoint Video Streaming**. arXiv:2606.28840v1, 2026-06-27. Relevance: dynamic Gaussian-streaming representation with temporal redundancy reduction and 4D frame interpolation; useful for OSS canvas serialization, active-mask streaming, and thin-client/streaming variants. https://arxiv.org/abs/2606.28840v1

## Research brief 2026-07-11
- In-Hwan Jin, Hyeongju Mun, Joonsoo Kim, Kugjin Yun. **On the Design of Mixture-of-Experts for Dynamic Gaussian Splatting**. arXiv:2607.08250v1, 2026-07-09. Relevance: multi-deformation MoE design for dynamic Gaussian scenes; direct study target if OSS v7 native-time `V_xt` motion encoding needs expert routing for heterogeneous static/rigid/articulated/transient motion. https://arxiv.org/abs/2607.08250v1

## Research brief 2026-07-12
- Jun-Sang Yoo, Seung-Won Jung. **MASS: Motion-Aligned Selective Scan for Refinement in Flow-Based Video Frame Interpolation**. arXiv:2606.27718v1, 2026-06-26. Relevance: motion-aligned selective-scan refinement for VFI under nonlinear motion/occlusion; useful external baseline for OSS-FX disocclusion and temporal artifact handling versus alpha-time Gaussian-canvas rendering. https://arxiv.org/abs/2606.27718v1

## Research brief 2026-07-13
- Kaizhe Zhang, Shinan Chen, Qian Zhao, Weizhan Zhang, Caixia Yan, Yudeng Xin. **MVGSR: Multi-View Consistent 3D Gaussian Super-Resolution via Epipolar Guidance**. arXiv:2512.15048v1, 2025-12-17. Relevance: epipolar-constrained multi-view attention for 3DGS SR; useful analogue for exact-MV-aligned temporal-history/canvas consistency checks in OSS. https://arxiv.org/abs/2512.15048v1
- Xiang Feng et al. **IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution**. arXiv:2511.22233v1, 2025-11-27. Relevance: mask-guided fusion of external 2D SR/depth priors with internal multi-scale 3DGS features; useful for OSS teacher/pseudo-target mixing without injecting inconsistent hallucinated detail into the canvas. https://arxiv.org/abs/2511.22233v1

## Research brief 2026-07-14
- Mingchao Sun, Luyang Tang, Yu Liu, Xu Yan, Zhan Li, Yunwei Zhang et al.. **ABot-3DWorld 0: A Universal World Model to Explore Any 3D Space**. arXiv:2607.11673v1, 2026-07-13. Relevance: Potentially relevant to v7 native time-axis canvas, temporal covariance, and dynamic-scene splat management. https://arxiv.org/abs/2607.11673v1
- Fatimah Zohra, Chen Zhao, Shuming Liu, Yahya Al Malallah, Bernard Ghanem. **HyperGS: Fast and Generalizable Gaussian Video Representation**. arXiv:2607.11500v1, 2026-07-13. Relevance: Potentially relevant to v7 native time-axis canvas, temporal covariance, and dynamic-scene splat management. https://arxiv.org/abs/2607.11500v1
- Tianyu Xiong, Rui Li, Suning Ge, Jiaqi Yang. **SalientGS: Unified SfM-to-3DGS with Importance-Guided MCMC Gaussian Allocation**. arXiv:2607.11285v1, 2026-07-13. Relevance: Potentially relevant to v7 native time-axis canvas, temporal covariance, and dynamic-scene splat management. https://arxiv.org/abs/2607.11285v1
- Yanjin Zhu, Shaofan Liu, Jianke Zhu. **Incremental Online Scene Reconstruction by 3D Gaussian Triangulation**. arXiv:2607.10690v1, 2026-07-12. Relevance: Potentially relevant to v7 native time-axis canvas, temporal covariance, and dynamic-scene splat management. https://arxiv.org/abs/2607.10690v1
- Aaron Maurice Berman, Shantanu Dave. **Grassmannian Splatting I: Moving rank-2 Spacetime Surfels for Dynamic Scene Rendering**. arXiv:2607.10489v1, 2026-07-11. Relevance: Potentially relevant to v7 native time-axis canvas, temporal covariance, and dynamic-scene splat management. https://arxiv.org/abs/2607.10489v1

## Research brief 2026-07-15
- Seung-gyeom Kim, Areum Kim, Yongjae Yoo, Sukmin Yun. **Implicit 4D Gaussian Splatting for Fast Motion with Large Inter-Frame Displacements**. arXiv:2607.12362v1, 2026-07-14. Relevance: SPIN-4DGS predicts Gaussian attributes from explicit spatiotemporal positions to preserve fast-moving objects under large inter-frame displacement; direct study target for OSS v7 native-time attributes, `V_xt` motion encoding, and OSS-FX fast-motion failures. https://arxiv.org/abs/2607.12362v1
- Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies. **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices**. arXiv:2607.10495v1, 2026-07-11. Relevance: fully convolutional, structurally reparameterized VSR that collapses to standard inference convolutions; direct reference for OSS ≤1M-param student design and RepVGG/ECBSR/RepSR deployment ablations. https://arxiv.org/abs/2607.10495v1

## Research brief 2026-07-16
- Cheng-Tai Hsieh, Jiwei Shan, Han Fang, Jianshu Hu, Tao Ni, Lijun Han, Yutong Ban, Shing Shin Cheng, Hesheng Wang. **ExtraGS: Enhancing Endoscopic View Extrapolation via Diffusion-Guided 3D Gaussian Splatting**. arXiv:2607.12785v2, 2026-07-14. Relevance: off-domain 3DGS view extrapolation with diffusion guidance; watch only for OSS-FX disocclusion/unknown-region priors, not as a real-time shipping architecture. http://arxiv.org/abs/2607.12785v2
- Zeyi Ren, Wencheng Yan, Jiawen Zhang, Jintao Yan, Sheng Zhou, Zhisheng Niu. **GeoFovea-GS: Geometry-Aware Cross-Layer Gaussian Splatting for Wireless Aerial VR**. arXiv:2607.12641v1, 2026-07-14. Relevance: foveated/wireless-VR Gaussian rendering and geometry-aware cross-layer allocation; potential future input for OSS foveated rendering, streaming, or canvas-budget policies. http://arxiv.org/abs/2607.12641v1

## Research brief 2026-07-17
- Amir Said, Randall Rauwendaal. **Compression of 3D Gaussian Splatting Data Using GPU-friendly Graphics Texture Coding**. arXiv:2607.14513v1, 2026-07-16. Relevance: compresses 3DGS color coefficients with GPU-native texture-compression schemes; direct reference for OSS packed Gaussian canvas layout, memory bandwidth, and cross-vendor hardware decode/raster paths. https://arxiv.org/abs/2607.14513v1
- Junteng Zhang, Tong Chen, Yuxin Zhao, Yibo Shi, Jing Wang, Zhan Ma. **SpeedyGS: Content-Aware 3D Gaussian Splatting Compression via Two-Stage Optimization**. arXiv:2607.12656v1, 2026-07-14. Relevance: rate-distortion-guided 3DGS quantization/pruning with faster decoding; useful for OSS active-density, packed-canvas, and interactive compression trade-offs. https://arxiv.org/abs/2607.12656v1

## Research brief 2026-07-19
- Jiahui Huang, Jiawei Ren, Michal Tyszkiewicz, Bjoern Haefner, Michael Shelley, Xin Kang, Seung Wook Kim, Ning Xu, Qi Wu, Janick Martinez Esturo, Shengyu Huang, Nick Schneider, Laura Leal-Taixe, Zan Gojcic, Sanja Fidler. **Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation**. arXiv:2607.14203v1, 2026-07-15. Relevance: NVIDIA feed-forward static/dynamic 3DGS-layer reconstruction from calibrated multi-camera driving data; a reference for v7 canvas partitioning, future capture/multi-camera work, and 3DGUT/non-pinhole compatibility—not a real-time SR/FG runtime. https://arxiv.org/abs/2607.14203v1

## Research brief 2026-07-20
- Damani Mguni-Coker. **ImprovedVBGS: Real-time Continual Variational Bayes Gaussian Splatting**. arXiv:2607.15542v1, 2026-07-17. Relevance: spatially truncated variational inference and reassignment are a direct reference for bounded-memory persistent-canvas updates, active-set pruning, and packed-layout experiments. https://arxiv.org/abs/2607.15542v1

## Research brief 2026-07-21
- Zhang Chen, Shuai Wan, Fuzheng Yang, Jiazhi Xia, Weiyao Lin, Junhui Hou. **SPARE-GS: Structural Parsimony and Resource Efficiency for 3D Gaussian Splatting**. arXiv:2607.16624v1, 2026-07-18. Relevance: global budget-constrained regional quotas coordinate densification, pruning, and termination; direct ablation target for v7 loss-adaptive parent-child density and issue #12 packed-canvas budgeting. https://arxiv.org/abs/2607.16624v1
- Zihao Zhang, Haoyu Zhao, Siqian Yang, Yidi Wu, Yudong Jiang, Zuxuan Wu. **SPEED: One-Step Pixel Diffusion for High-quality Video Frame Interpolation**. arXiv:2607.15585v1, 2026-07-17. Relevance: one-step pixel diffusion VFI is an offline high-motion/disocclusion perceptual-teacher and benchmark reference for OSS-FX, not a <2ms shipping path. https://arxiv.org/abs/2607.15585v1

## Research brief 2026-07-23
- Tingjia Zhang, Bo Chen, Shengzhong Liu, Fan Wu, Guihai Chen. **CaT-GS: Efficient 3DGS Rendering for Large Scale Scenes via Inter-frame Caching and Tile Scheduling**. arXiv:2607.17842v1, 2026-07-20. Relevance: inter-frame preprocessing/occlusion cache plus tile-load scheduling is a direct profiling/ablation target for OSS v7 persistent-canvas rasterization, #12 packed layout, and future fused cross-vendor kernels. https://arxiv.org/abs/2607.17842

## Research brief 2026-07-24
- Jiahao He, Yihua Shao, Zhengkai Zhao, Pan Gao, Fei Ma, Jingcai Guo, Hao Tang, Nicu Sebe, Qi Tian. **GrainGS: Gradient-Decoupled Gaussian Splatting for Efficient Dynamic Novel View Synthesis**. arXiv:2607.21448v1, 2026-07-23. Relevance: hierarchical anchor scaffold plus local temporal offsets, canonical/residual appearance, and stop-gradient isolation of canonical positions are direct ablation references for OSS v7 persistent-canvas stability, `V_xt` motion handling, and loss-adaptive density. https://arxiv.org/abs/2607.21448v1

## Research brief 2026-07-25
- Yukun Shi, Minglun Gong. **Future Rendering ≠ Future Surface: A Benchmark and Dataset for Dynamic Surface Reconstruction Beyond the Observed Window**. arXiv:2607.21471v1, 2026-07-23. Relevance: held-out-future geometry benchmark and future/observed error-gap protocol; direct evaluation template for OSS-FX beyond visually plausible frame metrics. https://arxiv.org/abs/2607.21471
- Karima Ma, Andrew Adams, Jonathan Ragan-Kelley. **Finding Fast Filters**. arXiv:2607.20634v1, 2026-07-22. Relevance: automatic quality/performance search over multirate, recurrent, summed, and cascaded filters with fused vectorized lowering; candidate methodology for OSS EWA/Mip-splat filtering. https://arxiv.org/abs/2607.20634
- Nicholas Fry, Ignacio Alzugaray, Mark Pupilli, Paul H. J. Kelly, Andrew J. Davison. **Rendering 3D Gaussians on a Graph Processor**. arXiv:2607.15951v1, 2026-07-17. Relevance: SRAM/tile-owned 3DGS renderer exposes local routing bandwidth, capacity, and density imbalance; direct reference for #12 packed canvas and #13 cache profiling. https://arxiv.org/abs/2607.15951

## Research brief 2026-07-28
- Jiun Lee, Jaekwang Kim, Sangmin Lee. **SubSplat: High-Resolution Pixel-aligned 3DGS via Sub-pixel Gaussian Reparameterization**. arXiv:2607.20813v1, 2026-07-23. Relevance: low-resolution feature-driven Gaussian subdivision with high-frequency aggregation; a direct parent-child/loss-adaptive density and packed-canvas (#12) ablation reference for OSS v7. https://arxiv.org/abs/2607.20813

## Research brief 2026-07-29
- Muhammad Talha, William Gordon, Sajid Umair, Anique Akhtar, Joel Jung. **SplatStream: Fine Granular Scalable Gaussian Splatting for Adaptive 3D Scene Streaming**. arXiv:2607.25971v1, 2026-07-28. Relevance: importance-packetized quality/resolution layers, temporal B-frame coding, and inter-layer prediction provide direct controls for OSS #12 packed-canvas budget, progressive quality, and temporal delta-layout experiments. https://arxiv.org/abs/2607.25971
- Sicheng Gao, Zhuyun Zhou, Yixuan Liu, Tong Shen, Zongwei Wu, Radu Timofte. **TRaM-VSR: Importance-Aware Token Routing and Merging for One-Step Diffusion Video Super-Resolution**. arXiv:2607.22231v1, 2026-07-24. Relevance: motion/boundary-aware adaptive quality allocation is a teacher-side comparison for v7 loss-adaptive Gaussian density; it is not a ≤2ms shipping student architecture. https://arxiv.org/abs/2607.22231

## Research brief 2026-07-30
- ByungHyun Kim, Jinwoo Jeon, Woontack Woo. **AtlasLC: Fast Codec-Ready Compression of Object-Centric 3D Gaussian Splatting**. arXiv:2607.26525v1, 2026-07-29. Relevance: training-free local-competition pruning and deterministic atlas packing target deployment-aware payload/decode/FPS trade-offs; direct reference for OSS #12 packed-canvas layout, GPU-friendly decode, and pruning measurements. https://arxiv.org/abs/2607.26525

## Research brief 2026-07-31
- Shady Gmira, Evangelos Alexiou, Emmanouil Potetsianakis, Emmanuel Thomas. **TSOG: A Format For Temporally And Spatially Ordered Gaussians**. arXiv:2607.28049v1, 2026-07-30. Relevance: temporally and spatially ordered 4DGS attributes stored as index-aligned image data; direct study target for OSS #12 temporal delta organization, packed-canvas serialization/bandwidth, and GPU texture/decode experiments. https://arxiv.org/abs/2607.28049
- Qiang Hu, Zhenlong Wu, Lei Huang, Zihan Zheng, Xiaoyun Zhang, Wenjun Zhang. **GenSplatCodec: Feed-Forward Gaussian Splatting Compression via One-Step Diffusion**. arXiv:2607.24403v1, 2026-07-27. Relevance: compact structural/reference appearance streams with geometry-guided generative decoding; offline rate-distortion and high-frequency canvas reference only, not a ≤2ms OSS shipping path. https://arxiv.org/abs/2607.24403

## Research brief 2026-08-03
- Shizeng Jiang, Hao Zhang, Xuerui Ma, Ying Hu, Tao Zhang. **QIRF: Quantum-Inspired Non-Orthogonal Function-Space Compression for 3D Gaussian Splatting**. arXiv:2607.18067v1, 2026-07-20. Relevance: overlap-aware function-space compression reports 71.7% fewer Gaussians/storage and 34.3% faster 3DGS rendering; direct ablation reference for OSS #12 packed canvas, loss-adaptive density/pruning, and #13 memory/raster profiling. https://arxiv.org/abs/2607.18067

## Research brief 2026-08-04
- Hantang Li, Qiang Zhu, Xiandong Meng, Debin Zhao, Xiaopeng Fan. **CLEAR: Conflict-aware Learning via Evidence-guided Adaptive Routing for Unified Sparse-View 3D Gaussian Super-Resolution**. arXiv:2608.02206v1, 2026-08-03. Relevance: confidence-gated routing of high-frequency gradients and Gaussian densification gives a direct ablation for OSS v7 parent-child density when LR engine evidence conflicts with teacher/perceptual priors. https://arxiv.org/abs/2608.02206
- Minnan Pei, Gang Li, Zeyu Zhu, Siting Wang, Junwen Si, Zhuoran Song, Yu Feng, Fangxin Liu, Xiaoyao Liang, Jian Cheng. **DeGS: A Scalable 3DGS Architecture via Decoupled Workload Parsing and Reorganization**. arXiv:2608.02099v1, 2026-08-03. Relevance: decoupled alpha/transmittance parsing, workload reorganization, and blending are direct profiling/prototyping references for OSS #3 tile bin, #12 packed canvas, and #13 raster memory behavior. https://arxiv.org/abs/2608.02099
- Zhengyang Zhang, Ziyu Lu, PengCheng Li, Hongbo Duan, Yi Liu, Pengting Luo, Peiyu Zhuang, Xinghui Li, Shaohua Ma. **FAST-GS: Frequency Aware Space-time Gaussian Splatting for Photorealistic Dynamic Novel View Synthesis**. arXiv:2608.01958v1, 2026-08-03. Relevance: Fourier motion components plus frequency-aware temporal regularization are a direct hard-motion ablation for OSS v7 `V_xt` time-axis motion encoding and OSS-FX coherence. https://arxiv.org/abs/2608.01958

## Research brief 2026-08-06
- Seunghyeon Song, Joo Chan Lee, Chanung Park, Jun Young Jeong, Minseo Lee, Eunbyung Park, Jong Hwan Ko. **ACA-GS: Adaptive-Capacity Anchored Gaussian Splatting for Compact Dynamic Radiance Fields**. arXiv:2608.04581v1, 2026-08-05. Relevance: adaptive per-anchor Gaussian count and feature-channel allocation reports up to 1.5× higher compression at comparable quality; **HIGH RELEVANCE** for v7 parent-child density, #12 packed-canvas budgets, and motion/detail-aware capacity allocation. https://arxiv.org/abs/2608.04581

## Research brief 2026-08-09
- Long Peng, Anran Wu, Wenbo Li, Peizhe Xia, Xueyuan Dai, Xinjie Zhang, Xin Di, Haoze Sun, Renjing Pei, Yang Wang, Yang Cao, Zheng-Jun Zha. **Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling**. arXiv:2503.06617v1, 2025-03-09; ICLR 2026. Relevance: direct 2D-Gaussian arbitrary-scale SR/Pico-student baseline; builds a Gaussian field from LR pixels in one pass and reports 1ms per-scale rendering. Research-only comparison: source is CC-BY-NC-4.0/non-commercial, incompatible with Apache-2.0 OSS reuse. https://arxiv.org/abs/2503.06617

## Research brief 2026-08-15
- Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg. **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading**. arXiv:2608.10423v1, 2026-08-11. Relevance: sparse tiled layered image-space cache with predictive shading, adaptive tile lifetimes, and explicit disocclusion handling is a direct non-neural OSS-FX baseline and candidate deterministic fallback/visibility-cache companion; evaluate it for multi-frame extrapolation latency and disocclusion error. https://arxiv.org/abs/2608.10423

## Research brief 2026-08-16
- Nuo Chen, Kangqi Ni, Lulin Liu, Joga Ivatury, Ying Ding, Farshid Alambeigi, Tianlong Chen, Zhiwen Fan. **EndoMD-SLAM: Endoscopic Gaussian Splatting SLAM under Optical Degradation with Memory and Static-Transient Decomposition**. arXiv:2608.08949v1, 2026-08-09. Relevance: memory-gated map updates plus an explicit static/transient split are direct ablations for OSS v7 persistent-canvas update confidence, disocclusion, and contamination handling. https://arxiv.org/abs/2608.08949
- Jisoo Jeong, Hong Cai, Jamie Menjay Lin, Hanno Ackermann, Hyeonjun Sim, Yinhao Zhu, Yunxiao Shi, Fatih Porikli. **SNM-VFI: Symmetric Nonlinear Motion-Guided Generative Video Frame Interpolation**. arXiv:2608.13460v1, 2026-08-13. Relevance: confidence-gated fusion of correspondence-supported prediction and uncertain-region detail provides a direct OSS-FX reliability/loss-evaluation template; diffusion itself is teacher-only, not a ≤2ms runtime path. https://arxiv.org/abs/2608.13460

## Research brief 2026-08-17
- Yanming Yang, Chenxi Song, Ping Wang, Xin Yuan, Chi Zhang. **GS²CI: Robust Gaussian Splatting For Snapshot Compressive Imaging via Large Vision Model Priors**. arXiv:2608.13502v1, 2026-08-13. Relevance: opacity-guided split candidates plus explicit candidate-ratio, mean-opacity, and total-Gaussian growth constraints are a direct bounded parent-child-density ablation for v7; its snapshot-compressive setting is not an OSS benchmark. https://arxiv.org/abs/2608.13502

## Research brief 2026-08-19
- Yang Luo, Yan Gong, Yongsheng Gao, Jie Zhao. **RoofGS: Roofline-Guided End-to-End Acceleration of 3D Gaussian Splatting**. arXiv:2608.15785v1, 2026-08-16. Relevance: stage-specific roofline analysis separating memory-bound front-end work from instruction-throughput-bound rasterization is a direct measurement framework for OSS #12 packed canvas, #13 Nsight L2/DRAM profiling, and tile-bin/raster kernel priorities; adapt methodology rather than assuming its 3DGS kernels transfer directly. https://arxiv.org/abs/2608.15785

## Research brief 2026-08-20
- Vu Trung Nghia Nguyen, Yuchen Wang, Kyung Chul Lee, et al. **QuARC-GS: Quantized Anchored Residual Coding for Compact Dynamic Scene Streaming with Gaussian Splatting**. arXiv:2608.18285v1, 2026-08-18. Relevance: anchored quantization and residual coding for dynamic Gaussian streams are a direct study target for OSS #12 packed low-rank canvas layout, temporal delta encoding, and cross-vendor decode-bandwidth budgeting. https://arxiv.org/abs/2608.18285
- Jiaming Fan, Jian Lu, Jinling Jia, et al. **Depth Anything V4: Dynamic 4D Scene Reconstruction via Riemannian Flow Matching on 4D Gaussian Splatting**. arXiv:2608.18388v1, 2026-08-18. Relevance: a teacher-side temporal-state/motion-loss reference for OSS v7 `V_xt`, update confidence, and disocclusion; evaluate as a bounded ablation, not a ≤2ms student runtime. https://arxiv.org/abs/2608.18388v1

## Research brief 2026-08-21
- Yuanhao Ban, Jiaqi Feng, Hengguang Zhou, Xiaohuan Pei, Justin Cui, Cho-Jui Hsieh. **Stream4D: 4D-Consistency for Streaming Autoregressive Diffusion Video Models**. arXiv:2608.19556v1, 2026-08-20. Relevance: feed-forward dynamic-4D reconstruction consistency plus scene-flow magnitude/jitter losses are a bounded teacher-side reference for OSS v7 native-time `V_xt` trajectories and OSS-FX temporal stability; diffusion is not a shipping runtime path. https://arxiv.org/abs/2608.19556v1

## Research brief 2026-08-22
- Yiwei Li, Jiannong Cao, Weixun Gao, Rui Cao, Songye Zhu, Yinfeng Cao, Mingjin Zhang. **S²GS: Structured Sparse Gaussian Streaming for Efficient Free-Viewpoint Video Reconstruction on Edge-IoT Devices**. arXiv:2608.19639v1, 2026-08-20. Relevance: octree-organized canonical-state residuals plus temporal sparse-update gating are a direct packed-canvas (#12) and temporal-delta-layout ablation; public code is MIT-licensed. https://arxiv.org/abs/2608.19639
- Yudong Jin, Tao Xie, Qihang Zhang, Zehong Shen, Zhen Xu, Yujun Shen, Hujun Bao, Xiaowei Zhou, Yinghao Xu. **4DAnyone: Create Anyone in 4D from a Casual Monocular Video**. arXiv:2608.20335v1, 2026-08-20. Relevance: bounded reference-context packing is a useful conceptual precedent for fixed-budget temporal/context state, not a shipping runtime. https://arxiv.org/abs/2608.20335
- Laurent Vit, Oliver Batchelor, Richard Green. **3D Gaussian Accelerated Ray Tracing: Fast Training through Particle-Based Backward Propagation**. arXiv:2608.17298v1, 2026-08-18. Relevance: tile–primitive gradient accumulation avoids atomic contention; direct OSS-RR and potential canvas-gradient-kernel training reference. https://arxiv.org/abs/2608.17298
- Tim-Felix Fassch, Jochen Kall, Cyrill Stachniss. **Compact Feed-Forward 3D Gaussians via Saliency-Guided Primitive Merging**. arXiv:2608.10712v1, 2026-08-11. Relevance: saliency-guided primitive merging is a candidate post-training compaction stage for v7 parent-child density and packed-canvas budgets. https://arxiv.org/abs/2608.10712
- Wei Zhang, Shengkai Yu, Shiqiang Gong, Qi Zhang, Qiang Li, Qi Wang. **HiCo-GS: Hierarchical Context Aggregation and Geometric Consistency for Octree Gaussian Splatting**. arXiv:2608.14136v1, 2026-08-14. Relevance: parent/self/child context aggregation and alpha-weighted geometric consistency are density-hierarchy/regularization ablation ideas; public code license is unasserted. https://arxiv.org/abs/2608.14136
- Rong Fu, Chunlei Meng, Yangchen Zeng, Xiaowen Ma, Yongtai Liu, Wangyu Wu, Shuo Yin, Zijian Zhang, Sicheng Li, Yingrui Ji, Chenhao Wang, Simon Fong. **MotionCraft: Latent World Modeling with Sparse Attention for Visual Upscaling**. arXiv:2608.08553v1, 2026-08-09. Relevance: teacher-side reference for controllable streaming VSR and temporal-state quality/performance modes, not a ≤1M student runtime. https://arxiv.org/abs/2608.08553

## From the desktop research clone docs/BIBLIOGRAPHY.md

## Core Architecture Foundations

### Gaussian Splatting & Temporal Representations
- **3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** — Zhao, Gao, Cai, Liu (2026-05-12) — arXiv:2605.11489 — Unified post-rendering super-sampling + frame interpolation for 3DGS. GASS gradient-aware refinement + LTFI lightweight temporal fusion. Directly validates OSS's one-canvas-for-both thesis.
- **R5DGS: Semantic-Aware 4D Gaussian Splatting with Rigid Body Constraints** — Gridusov, Popov, Kolyubin (2026-05-25) — arXiv:2605.25909 — 4D Gaussians with CLIP identity encoding + rigid-body dynamics for object-level tracking and frame extrapolation.
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes** — Lu, Pan (2026-05-11) — arXiv:2605.10586 — Inverse physics inference from multi-view video without explicit priors. Self-supervised motion dynamics.
- **EMAG: Differentiable 4D Gaussian Mixture Splatting for EEG Spatial Super-Resolution** — Lazarovich et al. (2026-05-28) — arXiv:2605.29731 — Full 4×4 precision matrix with spatial-temporal coupling via 4D Gaussians. Mathematical primitive confirmation for OSS v7's V_xt.
- **Accelerating 3D Gaussian Splatting using Tensor Cores** — Li, Sui, Wu, Song, Yuan, Tang, Dai (2026-05-18) — arXiv:2605.17855 — FP16 3DGS rasterization on Tensor Cores with negligible quality degradation. Directly applicable to OSS inference optimization.
- **ContinuousSR: Pixel to Gaussian — Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling** — Peng, Wu et al. (ICLR 2026) — arXiv:2503.06617 — Pixel-to-Gaussian paradigm for arbitrary-scale SR. Open source. Code: github.com/peylnog/ContinuousSR.
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** — Shi, Di, Peng et al. (2026-04-27) — arXiv:2604.18047 — First 2D-GS framework for continuous STVSR. Deformable 2D Gaussians with learned temporal kernel evolution. Already classified in docs/research/2026-05-13-gs-stvsr-classification.md.
- **GaussianZoom: Progressive Zoom-in Generative 3D Gaussian Splatting** (2026-05) — arXiv:2605.18252 — Iterative progressive GS for zoom-in reconstruction.

### Super-Resolution
- **DiffST: Spatiotemporal-Aware Diffusion for Real-World Space-Time Video Super-Resolution** — Chen, Yang, Han et al. (2026-05-13) — arXiv:2605.13182 — One-step diffusion STVSR with cross-frame attention. Architecture patterns relevant, but diffusion latency unsuitable for real-time games.
- **LCS: An AI-based Low-Complexity Scaler for Power-Efficient Super-Resolution of Game Content** — Pochinda, Tageldeen, Thompson et al. (2025-07-30) — arXiv:2507.22873 — AMD-adjacent game scaler targeting NPU offload. GameIR dataset, adversarial training + reparameterization.
- **CDA-VSR: Compressed-Domain-Aware Online Video Super-Resolution** — CVPR 2026 Highlight — arXiv:2603.07694 — Motion-vector-guided deformable alignment for video SR. Directly applicable to OSS canvas warping. Code: github.com/sspBIT/CDA-VSR.
- **OSDEnhancer: Taming Real-World Space-Time Video Super-Resolution with One-Step Diffusion** — Wei, Bai et al. (2026-01) — arXiv:2601.20308 — Unified SR+FG via one-step diffusion with LoRA-based task specialization.
- **FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution** — CVPR 2026 — arXiv:2510.12747 — Block-sparse attention + tiny conditional decoder. ~17 FPS at 768×1408 on A100. Code: github.com/OpenImagingLab/FlashVSR.
- **Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Gaming Content** (2026-02) — arXiv:2602.11339 — Benchmark of real-time SR methods on gaming content. Game-specific failure modes.

### Frame Extrapolation
- **Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos** (2026-05-18) — arXiv:2605.18233 — Train-free infinite frame generation approach.
- **Consecutive Frame Extrapolation with Predictive Sparse Shading** — Yuan et al. (ACM SIGGRAPH 2025 / TOG) — Neural frame extrapolation using sparse shading in real-time rendering. Architecturally aligned with OSS-FX.

### Video Quality Assessment
- **How Accurate are Video Quality Models for Diffusion-Based Video Super-Resolution?** — Herb et al. (2026-05-25) — arXiv:2605.25940 — VQA model evaluation on diffusion VSR. Artifact profile characterization.

### World Models / Real-Time Frameworks
- **minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models** — Zhao, Zhu, Yan et al. (2026-05-28) — arXiv:2605.30263 — Full pipeline for converting diffusion models to few-step distilled autoregressive world models. Teacher→student distillation patterns.

### Inference & Hardware
- **TC-MIS: Maximal Independent Set on Tensor-cores** (2026-05-28) — arXiv:2605.29604 — Non-ML algorithms on Tensor Cores. Validates broader Tensor Core applicability trend.
- **Range, Not Precision: Block-Floating-Point Half-Precision FFT and SAR Imaging on Apple Silicon** (2026-05-27) — arXiv:2605.28451 — Block-FP16 approaches for Apple Silicon, relevant to Metal backend.
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization** (2026-04) — arXiv:2604.20291 — QAT on fused deploy graph with BN recalibration. Won MAI 2026 Quantized 4K SR challenge. Directly applicable to OSS student model TensorRT FP8 export.

- **Mobile / Edge Super-Resolution**
- **NTIRE 2026 Challenge on Mobile Real-World Image Super-Resolution** (2026-04) — arXiv:2604.17306 — SoTA mobile SR solutions. Relevant to OSS Pico tier.
- **NTIRE 2026 Efficient Super-Resolution Challenge** (2026-04) — arXiv:2604.03198 — Efficiency techniques for image/video SR.
- **NTIRE 2026 Image Super-Resolution (×4) Challenge** — Samsung SRC-B won with cascaded framework. arXiv:2604.14558.

### Distillation for Super-Resolution
- **DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution** — CVPR 2026 — arXiv:2603.22271 — Distills 1B-param 50-step diffusion VSR to single step via dual-stream distribution matching + adversarial supervision. Code: github.com/cszy98/DUO-VSR.
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization** — MAI 2026 @ CVPR 2026 — arXiv:2604.20291 — Mamba teacher→QAT student with BN recalibration. Won MAI 2026 Quantized 4K SR challenge. Three-stage pipeline: basic→fidelity→QAT on fused graph.
- **Accelerating Diffusion Models with One-Step Distillation for Image and Video Super-Resolution** — MMSys 2026 — One-step diffusion distillation for SR.

### Integrated SR + Frame Generation
- **Mob-FGSR: Frame Generation and Super Resolution for Mobile Real-Time Rendering** — Yang, Zhu, Zhuge et al. (SIGGRAPH 2024) — Integrated SR+FG framework with no neural networks at runtime. Splat-based MV reconstruction, depth-aware atomic splatting, quadratic motion model. Deployed on OnePlus phones (120 FPS Genshin Impact). Code: github.com/Mob-FGSR/MobFGSR. Strongest validation of one-canvas SR+FG thesis.
- **Super-Resolution with Structured Motion** — Litterio, Lizarazo-Ferro, Felzenszwalb, Zia (ICCP 2025) — arXiv:2505.15961 — Motion blur as helpful cue for SR. Pseudo-random motion enables HR reconstruction from single LR image. Validates OSS's motion-vector-first approach.
- **EfRLFN: Efficient Real-Time SR with Hyperbolic Tangent Activation** — Bogatyrev, Abud, Molodetskikh et al. (2026-02) — arXiv:2602.11339 — StreamSR dataset + tanh activation for real-time SR. Code: github.com/EvgeneyBogatyrev/EfRLFN.

*Last updated: 2026-06-03. DUO-VSR, INT8 QAT, NTIRE ×4 SR, one-step diffusion distillation, and EfRLFN added. GS-STVSR, ContinuousSR, and earlier papers documented in individual deep-read notes under docs/research/.* DLSS 4.5 shipping, FSR 4.1 coming to RX 7000 in July, GFFE author moved to NVIDIA.*

## From the desktop research clone docs/research/BIBLIOGRAPHY.md

## Core References (OSS Architecture Foundations)

- **3D Gaussian Splatting for Real-Time Radiance Field Rendering** — Kerbl et al. (SIGGRAPH 2023). Original 3DGS paper. Foundation for OSS's Gaussian primitive.
- **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering** — Wu et al. (CVPR 2024). Introduced temporal dimension to 3DGS. Source of S-T variation score used in v7.
- **Mip-Splatting: Alias-free 3D Gaussian Splatting** — Yu et al. (CVPR 2024). Anti-aliasing via 3D smoothing filter. Adopted in v7 rasterizer.
- **HAT: Hybrid Attention Transformer for Image Restoration** — Chen et al. (CVPR 2023). HAT-Tiny backbone used in v6.x/v7 teacher models.

## Gaussian Splatting + Temporal

- **[GS-STVSR]** Shi et al. (arXiv:2604.18047, Apr 2026). Ultra-Efficient Continuous Spatio-Temporal VSR via 2DGS. Covariance resampling alignment. **Closest academic work to OSS.**
- **[RetimeGS]** Wang et al. (arXiv:2603.13783, CVPR 2026). Continuous-time 4DGS with temporal anti-aliasing via triple-rendering supervision.
- **[SpeeDe3DGS]** CVPR 2026 Poster #433. Temporal pruning + motion grouping for efficient 4DGS.
- **[Adaptive Spatio-Temporal 3DGS for Oscillatory Motion]** — INRIA (2025).
- **[Spacetime Gaussian Feature Splatting]** — OPPO US Research (2024). Dynamic scene representation.
- **[GaussianHSI]** Zhang et al. (arXiv:2604.17727, Apr 2026). Voronoi-guided Bilateral 2DGS for Arbitrary-Scale Hyperspectral SR. Voronoi pixel-to-Gaussian association + reference-aware bilateral weighting. **Novel rasterization approach for OSS tile-replacement consideration.**
- **[GaussianZoom]** (arXiv:2605.18252, May 2026). Progressive zoom-in generative 3DGS for multi-scale reconstruction. Progressive refinement strategy relevant to OSS training curriculum.

## Super-Resolution / Upscaling

- **[ContinuousSR (Pixel-to-Gaussian)]** Peng et al. (ICLR 2026 Poster). 2D Gaussian arbitrary-scale SR, 1ms per scale, Deep Gaussian Prior.
- **[GameSR]** Rehman et al. (ICLR 2026). Engine-independent real-time upscaling for cloud gaming. 240 FPS, 30-60% bandwidth savings. **Direct competitor validation.**
- **[GSASR]** Chen et al. (ICCV 2025). Feed-forward 2D Gaussian ASR with HAT-L encoder. SoTA arbitrary-scale SR.
- **[StreamSR / EfRLFN]** Bogatyrev et al. (arXiv:2602.11339, Feb 2026). Streaming-specific real-time SR benchmark + Efficient Channel Attention + tanh activation model. **Unusual design choices for student model consideration.**
- **[NTIRE 2026 Efficient SR Challenge Report]** Ren et al. (arXiv:2604.03198, CVPRW 2026). 95 registrants, target 26.90 dB PSNR while minimizing runtime/params/FLOPs. **Student architecture reference.**
- **[NTIRE 2026 Mobile Real-World Image SR Challenge]** Li et al. (arXiv:2604.17306, CVPRW 2026). ×4 mobile SR with unknown degradations. **Pico-tier relevance.**
- **[INT8-SR]** Nguyen et al. (arXiv:2604.20291, MAI Workshop @ CVPR 2026). Deployment-aware INT8 quantization with teacher guidance. **OSS TensorRT FP8 student export recipe.**
- **[OFTSR]** ICLR 2026. One-Step Flow SR with Tunable Fidelity. Flow matching as training objective.
- **[NTIRE 2026 3D Content SR Challenge / WaveletSR]** Wang et al. (CVPRW 2026). Blind SR on noisy 3DGS. Winner: WaveletSR. **Validates 3DGS SR as a growing subfield.**

## Frame Extrapolation / Generation

- **[Mob-FGSR]** SIGGRAPH 2024. Unified frame generation + super-resolution for mobile via splat-based motion vectors. **Closest unified SR+FG approach.**
- **[GFFE]** Wu et al. (arXiv:2406.18551, 2024). G-buffer Free Frame Extrapolation for real-time rendering. **Closest non-Gaussian FG competitor.**
- **[Consecutive Frame Extrapolation with Predictive Sparse Shading]** ACM 2025. Sparse shading for extrapolated frames.
- **[X4Rate]** Dixit & Sarangi (VISAPP 2026). Scene-adaptive extrapolation for 4× frame rate upsampling.
- **[Adaptive Frameless Rendering]** arXiv:2510.15876. Interleaved rendering + extrapolation.
- **[G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering]** arXiv:2512.16670. Diffusion-based future frame rendering.
- **[Surface-Aware Feed-Forward Quadratic Gaussian for Frame Interpolation]** NeurIPS 2025 Poster. Quadratic motion model for frame interpolation.

## Continuous Space-Time Representations

- **[Continuous Space-Time VSR with 3D Fourier Fields (VFF)]** Becker et al. (arXiv:2509.26325, ICLR 2026). Implicit Fourier representation — opposite architectural bet from OSS's explicit Gaussians. **Benchmark target.**

## Inference Optimization

- **[CUTLASS v4.5.2]** NVIDIA (May 2026). Blackwell-optimized GEMM with blockscaled FP8 quantization.
- **[TensorRT]** NVIDIA. FP8/INT8 inference optimization for OSS student model export target.

## Competitive Landscape

- **DLSS 4.5 (NVIDIA)** — 2nd gen transformer SR (CES 2026). Linear-space training, Model M/L, 6× Dynamic MFG. DLSS 4.5 Ray Reconstruction (2nd gen transformer, Aug 2026).
- **DLSS 5 (NVIDIA)** — Real-time neural rendering, "GPT moment for graphics" (GTC 2026, Fall 2026 launch).
- **FSR 4 / Redstone (AMD)** — ML-based upscaling, Frame Gen, Ray Regeneration, Radiance Caching. FSR 4.1 extends to RX 7000 (July 2026) and RX 6000 (2027). 300+ games. Cross-vendor openness planned.
- **XeSS 3 (Intel)** — Multi-Frame Generation, Arc A/B-series support.
- **MetalFX (Apple)** — Temporal Upscaler + Frame Interpolation, Metal 4 ray tracing.

## Tools & Ecosystem

- **[OptiScaler]** — Open-source upscaler swap tool for PC games. Validates OSS DLL-shim path.
- **[Arm Accuracy Super Resolution (ASR)]** — Open-source mobile temporal upscaler. Validates open-source upscaler viability.
- **[Upscayl]** — Open-source image upscaler (30k+ stars). No game-specific features.

*Last updated: 2026-06-02*

## From the laptop research clone BIBLIOGRAPHY.md, Aug–Oct

**[M³ISR]** Liu, Liu, Chen, Zhou, Wang, and Jiang (2026). *M³ISR: A Multi-Modal Multi-View Benchmark for 3D/4D Gaussian Splatting and Feedforward Compression*. arXiv:2608.22465.
> Controlled 1080p synthetic benchmark with RGB, depth, segmentation, static/dynamic masks, and 3D/4DGS streaming/compression tracks. Useful external validation for v7 canvas storage, update cost, and rate-distortion work; it does not replace engine-motion-vector game-data evaluation.

**[Native-4DGS]** Yang, Pan, Zhu, Zhang, Feng, Jiang, and Torr (2025). *4D Gaussian Splatting: Modeling Dynamic Scenes with Native 4D Primitives*. arXiv:2412.20720.
> **HIGH RELEVANCE.** Native 4D Gaussian primitives and time-sliced rendering provide an independent, interactive-rate precedent for v7's `(x,y,t)` canvas and covariance cross-terms; its memory-reduction variants are a useful control for the packed-canvas work.

**[3DGS-Particle-Backward]** Vit, Batchelor, and Green (2026). *3D Gaussian Accelerated Ray Tracing: Fast training through particle-based backward propagation*. arXiv:2608.17298.
> Newly material training-kernel reference: particle-centric, tile-mapped backward propagation is a control for v7 temporal-Gaussian gradient accumulation and atomic contention; it is not a replacement for OSS's rasterized shipping path.

**[RetimeGS]** Wang, Ma, Shen, Wang, and Sander (2026). *RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting*. CVPR 2026. arXiv:2603.13783
> Explicit continuous-time Gaussian behavior plus flow/triple-rendering supervision reduces interpolation ghosting. Closest external test of v7's native time-axis premise; source for anti-aliasing and α-time supervision ablations.

**[TSOG]** Gmira et al. (2026). *TSOG: A Format For Temporally And Spatially Ordered Gaussians*. ICIP 2026. arXiv:2607.28049
> Model-agnostic, index-aligned temporal/spatial Gaussian attribute encoding reports >90% storage reduction with small PSNR change. Direct reference for v7 canvas serialization, cache ordering, and capture-upload bandwidth.

**[Compact-FFGS]** Fassch, Kall, and Stachniss (2026). *Compact Feed-Forward 3D Gaussians via Saliency-Guided Primitive Merging*. arXiv:2608.10712
> Saliency-aware merging and controllable LoD reduce redundant feed-forward Gaussian sets to roughly one-twentieth of their primitives. Candidate post-spawn compaction mechanism, conditional on temporal-identity validation.

**[DeGS]** Pei et al. (2026). *DeGS: A Scalable 3DGS Architecture via Decoupled Workload Parsing and Reorganization*. MICRO 2026. arXiv:2608.02099
> Separates coverage/transmittance checking from blending and compacts irregular work before parallel rasterization. Direct dataflow reference for OSS custom tile bins and fused CUDA/HIP raster kernels.

**[DAV4]** Fan, Lu, Jia, and Zhang (2026). *Depth Anything V4: Dynamic 4D Scene Reconstruction via Riemannian Flow Matching on 4D Gaussian Splatting*. arXiv:2608.18388.
> Riemannian flow matching keeps scale, rotation, and opacity updates on valid 4DGS manifolds. Teacher-side reference for stable v7 covariance/opacity/time updates; not a real-time or shipping-student architecture.

**[RoofGS]** Luo, Gong, Gao, and Zhao (2026). *Roofline-Guided End-to-End Acceleration of 3D Gaussian Splatting*. arXiv:2608.15785.
> Roofline-guided split between memory-bound front-end work and instruction-bound rasterization, with compact attributes, 32-bit sort keys, approximate exponentials, culling, and kernel fusion. Primary implementation/ablation reference for #3 tile bin, #12 packed canvas, and #13 Nsight profiling; its reported throughput is not directly transferable to OSS.

**[TileGS]** Tan, Turkulainen, Ilola, Tavakoli, and Kannala (2026). *TileGS: Tile-Local Depth Binning for Gaussian Splatting Rasterization*. arXiv:2609.03613.
> **Direct #12/#13 rasterizer ablation.** Reorganizes globally sorted splat streams into tile-local depth ranges with selective ordering repair; reports 1.44× mean raster-kernel speedup over gsplat on RTX 4090 while matching output within numerical noise. Test on the 2D temporal canvas with geometry-attribute traffic, instruction count, compositing error, and end-to-end frame time—not its 3DGS benchmark alone.

**[MotionAware-4DGS]** Dhiman, Kathare, Vignesh, Boregowda, and Radhakrishnan (2026). *Towards Alias-Free 4D Gaussian Representations with Motion-Aware Filtering*. arXiv:2608.21828.
> Local-motion-conditioned smoothing for dynamic 4D Gaussians, rather than static Mip-style filtering. Direct v7 anti-aliasing ablation: compare motion-conditioned covariance/filter selection against the current static filter for shimmer, blur, and disocclusions.

**[ACA-GS]** Song, Lee, Park, Jeong, Lee, Park, and Ko (2026). *ACA-GS: Adaptive-Capacity Anchored Gaussian Splatting for Compact Dynamic Radiance Fields*. ACM Multimedia 2026. arXiv:2608.04581.
> Dynamically assigns Gaussian cardinality and feature channels per anchor from local spatiotemporal complexity. Direct v7 reference for #12's bounded canvas: test adaptive parent/child capacity and feature-width budgets with temporal-identity and L2-residency gates.

**[GS²CI]** Yang, Song, Wang, Yuan, and Zhang (2026). *GS²CI: Robust Gaussian Splatting For Snapshot Compressive Imaging via Large Vision Model Priors*. arXiv:2608.13502.
> Uses opacity-guided splitting, mean-opacity regulation, and explicit candidate/count caps under ambiguous supervision. A bounded-densification and unreliable-observation ablation source for v7 disocclusion spawning and bad engine-MV/capture inputs.

**[CLEAR]** Li, Zhu, Meng, Zhao, and Fan (2026). *CLEAR: Conflict-aware Learning via Evidence-guided Adaptive Routing for Unified Sparse-View 3D Gaussian Super-Resolution*. arXiv:2608.02206.
> Evidence-conditioned routing treats observed LR gradients as a reliable anchor and selectively routes high-frequency/densification gradients. Candidate confidence gate for OSS spawning and imperfect-teacher mixing; must pass temporal-identity validation.

**[RBS]** Federico, Amato, Gennaro, Carrara, and Di Benedetto (2026). *Resonant Brane Splatting for Arbitrary-Scale Super-Resolution*. arXiv:2606.29453.
> Replaces flat 2D Gaussians with Gaussian-Hermite color modes and culling-aware rasterization, providing a direct quality-per-primitive alternative to test before expanding v7 canvas state.

**[QuADA-GS]** Federico, Amato, Gennaro, Carrara, and Di Benedetto (2026). *Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution*. arXiv:2606.29400.
> Global-budgeted local routing and irregular feature densification are direct references for OSS's bounded canvas/tile scheduler and loss-adaptive spawning policy.

**[Amulet]** Künzel et al. (2026). *Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading*. arXiv:2608.10423.
> Sparse layered image-space cache, predictive shading, and adaptive tile refresh for multi-frame extrapolation. Primary competitive/systems reference for OSS-FX disocclusion and refresh policy; compare against exact-engine-MV Gaussian-canvas rendering.

**[SNM-VFI]** Jeong, Cai, Lin, Ackermann, Sim, Zhu, Shi, and Porikli (2026). *SNM-VFI: Symmetric Nonlinear Motion-Guided Generative Video Frame Interpolation*. arXiv:2608.13460.
> Fuses flow-generated intermediate candidates with confidence maps to contain occlusion/boundary failures. Not a latency-compatible OSS-FX architecture, but a control reference for invalid-engine-MV and hole-confidence evaluation.

**[3DGS³]** Zhao, Gao, Cai, and Liu (2026). *3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering*. arXiv:2605.11489.
> A post-render GASS super-sampling module plus compact temporal interpolation network validates the joint SR+temporal-rendering product surface, but retains separate modules. Direct competitive baseline for OSS's stronger shared-canvas/fractional-time rendering claim.

**[TRiGS]** Yeom et al. (2026). *TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting*. arXiv:2604.00538.
> Continuous SE(3) transforms, hierarchical Bézier residuals, and local anchors preserve primitive identity over long sequences rather than respawning Gaussians. Candidate control for v7 canvas memory growth and rigid-region temporal stability.

**[STGV]** Lin et al. (2026). *STGV: Spatio-Temporal Hash Encoding for Gaussian-based Video Representation*. arXiv:2604.10910.
> Separates static 2D spatial and dynamic 3D temporal hash encodings with canonical key-frame initialization; a compact reference for decoupling v7 persistent canvas state from motion residuals.

**[QuARC-GS]** Nguyen, Wang, Lee, and Zhou (2026). *QuARC-GS: Quantized Anchored Residual Coding for Compact Dynamic Scene Streaming with Gaussian Splatting*. arXiv:2608.18285.
> Quantization-aware anchored residual coding for dynamic Gaussian streams. Newly material input to #12’s packed canvas: evaluate its anchor/residual layout against L2 residency, temporal primitive identity, and reconstruction before adopting any streaming-specific machinery.

**[EfRLFN/StreamSR]** Bogatyrev et al. (2026). *Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content*. arXiv:2602.11339.
> StreamSR benchmarks 11 real-time SR models on compressed streaming content and releases EfRLFN/code/data. External efficiency and degradation-robustness reference, not evidence for OSS's game-G-buffer or <2 ms targets.

**[NanoVSR]** Pawlicki, Kańduła, Pucek, and Dobies (2026). *NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices*. ECCV 2026. arXiv:2607.10495.
> Fully convolutional VSR with structural reparameterization, a standard-convolution deploy graph, and progressive temporal training; a direct student-backbone candidate for #9/#4, subject to OSS-resolution latency measurements.

**[Efficient-INT8-SISR]** Nguyen, Le, Vo, and Nguyen (2026). *Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training*. MAI 2026 Workshop at CVPR. arXiv:2604.20291.
> LR-domain compute, re-parameterizable blocks, teacher-guided reconstruction, and QAT on the fused deploy graph provide a concrete post-distillation recipe for the ≤1M OSS student; its mobile x3 SISR results do not establish game-VSR latency.

**[DUO-VSR]** Lv, Xia, Wang, and Wong (2026). *DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution*. CVPR 2026. arXiv:2603.22271.
> One-step diffusion-VSR distillation combines trajectory-preserving initialization, distribution matching, adversarial score features, and preference refinement. Teacher-side perceptual-distillation reference only; not a candidate for the ≤1M-parameter real-time shipping graph.

**[VFF-STVSR]** Becker, Erbach, Narnhofer, and Schindler (2025). *Continuous Space-Time Video Super-Resolution with 3D Fourier Fields*. arXiv:2509.26325.
> A continuous 3D video field jointly samples arbitrary space-time points with an analytical Gaussian PSF. Key non-Gaussian baseline for v7: compare quality, temporal consistency, and raster cost against the persistent N-D Gaussian canvas.

**[CDA-VSR]** Wang, Li, Hou, Dong, and Yang (2026). *Compressed-Domain-Aware Online Video Super-Resolution*. CVPR 2026. arXiv:2603.07694.
> Coarse motion-vector warping, local residual alignment, residual-map reliability gating, and frame-type-aware compute achieve a reported >2× speedup over TMP. Direct student ablation for exact-engine-MV alignment and disocclusion confidence, not a replacement for the persistent Gaussian canvas.

**[MotionCraft]** Fu et al. (2026). *MotionCraft: Latent World Modeling with Sparse Attention for Visual Upscaling*. arXiv:2608.08553.
> Motion-aware latent-state prediction with sparse nonlocal attention and a compact decoder; teacher-side comparator for persistent state and controllable temporal smoothness, not evidence of sub-2 ms deployment.

**[S²GS]** Li, Cao, Gao, Cao, Zhu, Cao, and Zhang (2026). *S²GS: Structured Sparse Gaussian Streaming for Efficient Free-Viewpoint Video Reconstruction on Edge-IoT Devices*. arXiv:2608.19639.
> Octree-organized residuals, hierarchical temporal gating, and discrete sparse updates are a direct compact-update reference for v7 canvas residency; evaluate against temporal-identity and L2-residency gates rather than importing the free-viewpoint system.

**[MeanSR]** Niu, Kou, Zhang, Yan, Sun, and Zhang (2026). *MeanSR: Restoration Trajectory Learning for One-Step Perceptual Super-Resolution*. arXiv:2608.09405.
> LR-conditioned restoration-velocity learning is a teacher-free one-step perceptual-SR control for OSS student training; it does not establish game-temporal quality or a sub-2 ms deployment path.

**[QuantSR+]** Qin et al. (2026). *QuantSR+: Pushing the Limit of Quantized Image Super-Resolution Networks*. arXiv:2605.22351.
> Progressive slimming and function-local distillation for 2–4-bit SR; a later-stage quantization reference after OSS has established its fused FP8/INT8 student baseline.

**[FoRM]** He and Liu (2026). *Flow-Map Distillation on Relation Manifolds for Image Restoration*. arXiv:2608.05769.
> Compositional relation-map distillation with semigroup consistency is a concrete teacher-to-student ablation for the ≤1M OSS model; validate only against game-temporal metrics and the fused deployment graph.

**[VeloGauss]** Lu and Zhao (2026). *VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos*. arXiv:2605.10567 (ICME 2026).
> Per-Gaussian velocity learning plus global physical constraints improves future-frame extrapolation from video. The exact-engine-MV OSS setting should use this as an ablation source for motion/velocity consistency, not as a replacement for observed engine motion.

**[ANVIL]** Liu (2026). *ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors*. arXiv:2603.26835.
> Reuses exposed codec motion vectors, removes learned flow and sampling from the accelerator graph, and confines residual refinement to INT8-friendly convolutional operators. A concrete deployment control for an exact-engine-MV-prealigned OSS student, not a substitute for the persistent Gaussian canvas or OSS-FX extrapolation.

**[MGMVFI]** Park and Cho (2026). *Following Motion for Sequential Modeling in Video Frame Interpolation*. arXiv:2608.22861.
> Motion-guided serialization aligns state-space updates with optical-flow trajectories and falls back to contextual synthesis around unreliable flow. A residual-refinement/confidence-gating control after OSS exact engine-MV warp, not a replacement for the persistent canvas or fractional-time renderer.

## Research brief 2026-08-31

**[ABCD]** Shiu and Subr (2026). *ABCD: Alpha-Composited Block Coordinate Descent: Constant-VRAM Training for Large Radiance Fields*. SIGGRAPH 2026 Posters. arXiv:2608.27735.
> Out-of-core spatial block-coordinate training pre-composites frozen regions and bounds peak VRAM for fixed block/image sizes; a teacher-training ablation for v7 canvas scale, subject to temporal-block and disocclusion validation.

**[NUQ-3DGS]** Van Hauwermeiren, Rondao Alface, and Munteanu (2026). *Non-Uniform Quantisation for 3DGS Compression*. arXiv:2608.28272.
> Importance-weighted quantisation and merging are a concrete packed-canvas/serialization ablation for #12, gated on temporal primitive identity, quality, and L2-residency measurements.

## Research brief 2026-09-01

**[ATGS]** Wu, Liang, Hu, Yang, Xiong, Li, Zheng, Wang, and Wang (2026). *ATGS: Anchored Temporal Gaussian Splatting for Long Volumetric Video Representation*. SIGGRAPH 2026. arXiv:2608.30184.
> Time-conditioned anchors plus query-time temporal windows are a direct v7 canvas-residency/scalability ablation; adopt only if they preserve exact-MV temporal identity and disocclusion quality while reducing active Gaussians/tile and raster cost.

**[SMG]** Yu, Yang, Luo, Sun, and Hariharan (2026). *SMG: Semantic Motion Graph for Monocular Dynamic Gaussian Splatting*. ECCV 2026. arXiv:2608.31023.
> Low-rank graph motion with reliable nodes guiding uncertain regions is a teacher-side confidence/disocclusion ablation after the OSS exact-engine-MV warp, not a learned-motion replacement for game inputs.

## Research brief 2026-09-02

**[EvoGS]** Dong, Shirani, Chen, and Zhou (2026). *EvoGS: Modeling Deformation Evolution for Dynamic Gaussian Splatting*. arXiv:2609.00994.
> Persistent per-Gaussian deformation state, residual-memory correction, and uncertainty-aware densification are a direct v7 ablation for exact-MV residual confidence and spawn gating; retain engine motion vectors as the primary motion signal.

## Research brief 2026-09-03

**[CC-4DGS]** Park and Rhee (2026). *CC-4DGS: Computational Deformation and Point-Cloud Compression for Storage-Efficient Dynamic Gaussian Splatting*. arXiv:2609.02184.
> Compact deformation features and Gaussian-attribute compression are a direct packed temporal-canvas reference for #12; adopt only if temporal primitive identity, disocclusion quality, and L2 residency survive the OSS exact-MV workload.

**[Atlas]** Zhu, Liu, Li, Wu, Liu, Li, Wei, Zhao, Leng, Guo, and Feng (2026). *Atlas: Algorithm-Hardware Co-Design for On-Device City-Scale 3D Gaussian Splatting in VR*. arXiv:2609.02352.
> Pose/LoD-aware streaming and stereo rasterization are a runtime reference for active Gaussian selection and canvas residency; validate on OSS’s 2D temporal tiles before generalizing the reported VR gains.

**[PixelIR]** Qiao, Shi, Guo, Zhang, and Cao (2026). *PixelIR: Fidelity-Perception Decoupling via Pixel-Space Image-Residual Flow Matching for Efficient One-Step Real-World Super-Resolution*. arXiv:2608.30782.
> Fidelity/residual target separation is a student-distillation ablation for #4, but its 32.9M deployment model is far above OSS’s ≤1M shipping budget.

## Research brief 2026-09-04

**[M3ISR]** Liu, Liu, Chen, Zhou, Wang, and Jiang (2026). *M³ISR: A Multi-Modal Multi-View Benchmark for 3D/4D Gaussian Splatting and Feedforward Compression*. arXiv:2608.22465.
> Controlled 3D/4DGS streaming and compression tracks with synchronized RGB/depth/semantic data are a supplementary evaluation reference for #12; retain OSS-specific engine-MV, temporal-identity, disocclusion, VRAM, and L2-residency gates.

## Research brief 2026-09-09 (evening update)

**[HyperGS]** Zohra, Zhao, Liu, Al Malallah, and Ghanem (2026). *HyperGS: Fast and Generalizable Gaussian Video Representation*. arXiv:2607.11500.
> Feedforward per-frame 8-parameter Gaussian prediction with a rank-based degeneration regularizer is the nearest published relative of the OSS student head; transplant the regularizer into v7 canvas training if opacity/shape collapse appears at scale.

**[GameSR]** Rehman, Shirmohammadi, and Hefeeda (2026). *GameSR: Real-Time Super-Resolution for Interactive Gaming*. ICLR 2026 submission, OpenReview:wnJkdo5Gu9.
> Engine-independent reparam-CNN + ConvLSTM SR on encoded game frames at 240 FPS and near-DLSS/FSR parity is the decode-only competitor control; quantify the engine-MV quality gap before defending OSS's engine-MV premise.

## Research brief 2026-09-10

**[TruncGradGS]** Morales, Le-Pham, Atkins, and Hua (2026). *TruncGradGS: Improved 3D Gaussian Splatting via Truncated Gradient Updates*. arXiv:2609.03534.
> Truncated gradient schedules stabilize GS training dynamics; cheap candidate ablation for v7 canvas training instability (D12/D13 lineage), to be tested alongside the HyperGS rank regularizer.

**[CompactAppearanceGS]** Hahlbohm, Condor, Franke, Eisemann, and Magnor (2026). *Compact Neural Appearance Models for Efficient Gaussian Splatting*. arXiv:2609.05255.
> Compact per-Gaussian appearance in place of SH is a direct attribute-compaction reference for #12 (packed low-rank canvas layout); mine before freezing the 32B/Gaussian claim.

**[MedGSSR]** Wang, Hong, Zhao, Wang, Feng, Qin, Kuang, Yin, Bashashati, and Zhu (2026). *MedGSSR: Generalizable Medical Image Super-Resolution 3D Reconstruction via Hierarchical Feed-forward Gaussian Splatting*. arXiv:2609.06874.
> Third independent group (after HyperGS, SR3R) converging on feedforward generalizable Gaussian SR; trend evidence for the student-side bet, not a technique donor.

## Research brief 2026-09-11

**[EdMCGS]** Wang, Zhang, and Zhu (2026). *EdMCGS: Event-Driven Markov Chain Gaussian Splatting for Extreme-Low-Frame-Rate Dynamic Scene Reconstruction*. arXiv:2609.08332.
> Markov-chain Gaussian state evolution with event-stream evidence, rendered at any intermediate timestamp; nearest published "fill in missing time with splats" formulation — deep-read the temporal-identity/update mechanics against the v7 canvas update rule.

**[VSCP-GS]** Chu, Li, and Huang (2026). *View-Structured Conformal Prediction for 3D Gaussian Splatting*. arXiv:2609.10307.
> Certified prediction-coverage guarantees over rendered views instead of uncertainty heatmaps; candidate certified-coverage gate for the OSS eval harness (disocclusion/hole coverage) — a differentiator no DLSS/FSR/XeSS competitor offers.

**[LinearMask-GS]** Ryu and Lee (2026). *LinearMask-GS: Stable-Mask Importance Pruning for Compact 3D Gaussian Splatting*. arXiv:2609.10095.
> Stabilized learned-mask (Gumbel-Sigmoid) pruning without quality cliffs; Gaussian-count-control candidate for #12 alongside Compact-FFGS merging — decide the mechanism before freezing packed-canvas capacity.

**[EFQ-Softmax]** Han et al. (2026). *EFQ-Softmax: Exp-Free Quantization for Softmax*. arXiv:2609.09721.
> Exp-free low-bit softmax for FP8/FP4 tensor-core attention; only relevant if the ≤1M student ships attention blocks in the FP8 TensorRT path (flag for #12→fused-TRT sequencing).

## Research brief 2026-09-11 (midday update)

**[RouteBridge]** Han et al. (2026). *RouteBridge: Reliability-Routed Bidirectional Distillation Between Neural Radiance Fields and 3D Gaussian Splatting*. arXiv:2609.09606.
> Per-ray routed bidirectional teacher selection by reliability; concept donor for #4/#7 — e.g. route HAT-Tiny vs FSR 4-teacher supervision per region (edges/occlusions) instead of a single global loss mix.

**[Splats2Silicon]** (2026). *From Splats to Silicon: Rethinking Computational Efficiency of 3DGS*. arXiv:2609.06157.
> Workload-centric framework connecting representation/runtime/hardware optimizations to end-to-end frame time; methodology reference for #13's Nsight L2 profiling and any future canvas efficiency claims.

**[FreeTransformSR]** (2026). *FreeTransformSR: Efficient Lightweight Image Super-Resolution via Free Low-Rank Learnable Transform*. arXiv:2609.05912.
> Lightweight SR via free low-rank learnable transforms; candidate architecture donor for the #9 reparameterized student backbone comparison set (RepVGG/ECBSR/RepSR lineage).

## Research brief 2026-09-12 (midday update)

**[Mob-FGSR]** Yang, Zhu, Zhuge, Qiu, Li, Yan, Yan, Yan, and Jin (2024). *Mob-FGSR: Frame Generation and Super Resolution for Mobile Real-Time Rendering*. SIGGRAPH 2024. https://mob-fgsr.github.io/ (code: github.com/Mob-FGSR/MobFGSR).
> Shipped joint interpolation/extrapolation + SR from color/depth/engine-MVs with zero runtime NNs (MV splatting + LUT resampling; 2.2–2.3 ms on Snapdragon 8 Gen 3; OnePlus Ace 3 Pro/13). Newly surfaced as material: the closest shipped precedent for OSS's "one pipeline, FG + SR from engine MVs" thesis — use as the #5 No-ML ablation baseline and cite in positioning (OSS deltas: continuous `t_query`, learned quality, vendor-neutral).

**[D2GV-AR]** Liu, Yang, Zhao, Huang, Yang, Li, and Xu (2026, v2). *Progressively Deformable 2D Gaussian Splatting for Video Representation at Arbitrary Resolutions*. arXiv:2503.05600.
> Canonical per-GOP 2D Gaussian set with neural-ODE temporal evolution, Nyquist scale-grouping for arbitrary-scale rendering, and D-optimal pruning for progressive coding at 250+ FPS. Nearest video-codec-side cousin of the v7 canvas; ablation donor for #12 (packed canvas, progressive capture-upload).

**[GS-NFS]** Ghosh, Wang, Hong, Pavez, Chaudhuri, Pang, Madhyastha, Ortega, and Govindan (2026). *GS-NFS: Bandwidth-adaptive Streaming of Dynamic Gaussian Splats and Point Clouds*. arXiv:2606.05650.
> GPU-parallel full-frame-rate encode/decode of dynamic Gaussian frames (1–2 orders faster than prior codecs at competitive rate). #12 serialization reference: mine the GPU-side position/attribute parallelization before writing custom canvas-streaming code.

**[MultiCam-FrameSyn]** Daly and Kokaram (2025). *A new dataset and comparison for multi-camera frame synthesis*. SPIE DIP XLVIII. arXiv:2508.09068.
> Controlled array-camera comparison: 3DGS beats frame interpolators by +5 dB PSNR on synthetic scenes but loses up to 3.5 dB on real footage. Direct eval-plan citation for OSS's domain bet — Gaussian canvases win in geometry-rich synthetic rendering (game capture), degrade on noisy real footage.

## Research brief 2026-09-13

**[P-GSVC]** Wang, Shi, and Ooi (2026). *P-GSVC: Layered Progressive 2D Gaussian Splatting for Scalable Image and Video*. arXiv:2603.10551.
> Layered base+enhancement 2D Gaussian splats with joint cross-layer optimization give a unified progressive representation for images and video. Newly surfaced as material (NUS, GS-STVSR lineage): direct donor for the #12 layered/progressive canvas — progressive capture-upload, bandwidth-adaptive streaming, canvas LoD — and the cross-layer tracking question is exactly the temporal-identity problem a layered v7 canvas must answer.

**[CVT-GS]** Li, Li, Peng, Wang, Zhu, Li, Liu, Wang, and Pang (2026). *CVT-GS: Learning to Simplify 3D Gaussian Splatting with Centroidal Voronoi Tessellation*. arXiv:2609.08730.
> Optimization-free post-hoc simplification of trained 3DGS scenes (no retraining or fine-tuning) via CVT-based primitive consolidation. Completes the #12 mechanism three-way (LinearMask-GS train-time pruning / Compact-FFGS merging / CVT-GS post-hoc); uniquely applies to already-trained checkpoints, gated on temporal identity, disocclusion quality, and L2 residency.

## Research brief 2026-09-13 (midday update)

**[GSReuse]** Tao, Sun, Huang, Zhu, Guo, Zhang, Wang, and Guo (2026). *GSReuse: Temporally Adaptive Screen-Space Reuse for Accelerating 3D Gaussian Splatting*. IEEE TVCG 33(5), 2026. IEEE Xplore 11460301 (not on arXiv). DOI 10.1109/TVCG.2026.3679889.
> Drop-in 3DGS render accelerator: estimates reliable per-pixel Gaussian-splatting motion vectors from the rendered image + depth + camera, then warps reusable contents to the new view. Newly surfaced: independent precedent for the OSS canvas-warp thesis argued from the pure-runtime side; its reliability gate for warp-reuse vs re-render maps directly onto the v7 disocclusion/spawner decision. #13 methodology reference. **Evening update: author PDF acquired (letianhuang.github.io/assets/pdf/GSReuse.pdf, mirrored in docs/research/papers/, sha256 eaed57be...) and deep-read — purely geometric non-learned pipeline: alpha-depth GS-MVs + conservative morphological-closing pixel mask (cross kernel) + 16×16 tile-validity gate + max-depth fill + depth-guided bilateral color fill + occlusion-aware atomicMin composition; incremental-mode speedup S≈(1/a+n)/(1+n/a); 2.24× avg (4.5→2.0 ms/frame) on stock 3DGS, ~1.8× combined with compressors, 1.6× on CityGaussian-v2, 2.64 ms per generated frame vs RIFE 9.89 ms, wins PSNR/SSIM/LPIPS in 8/9 Mip-NeRF360 scenes (RTX 4090, 1600×1000). Stage split: compositing ~70% / sorting ~20% of baseline frame time. Limitations (depth sensitivity — 2DGS depth better than 3DGS; fast-motion reuse collapse; dense tiles resist reuse) map onto the OSS engine-MV + spawner design rationale.**

**[LapFH-GS]** Yang, Zhang, Yan, Shi, and Wang (2026). *Laplacian Frequency Hierarchies for Efficient 3D Gaussian Splatting Training*. arXiv:2609.03334.
> Frequency-staged training: fit low frequencies, archive that Gaussian field, fit residuals in new fields, compose at inference. TruncGradGS-class control for v7 canvas training growth, plus a hot/cold archived-field residency alternative for #12 if tile-level archival compositing is acceptable.

## Research brief 2026-09-15 (midday update)

**[GameSR]** Ur Rehman, Shirmohammadi, Amer, and Hefeeda (2026). *GameSR: Real-Time Super-Resolution for Interactive Gaming*. IEEE Transactions on Games, accepted July 2026 (not on arXiv). Author PDF: www2.cs.sfu.ca/~mhefeeda/Papers/tg26_GameSR.pdf (mirrored in docs/research/papers/, sha256 2bbfe650…).
> Engine-independent SR operating directly on encoded game frames (cloud-gaming lane): reparameterized conv blocks (1×1→3×3→1×1 merged at inference) + PixelUnshuffle down-and-up + lightweight ConvLSTM temporal context, Charbonnier loss, Torch-TensorRT FP16 with kernel fusion. 138K parameters, ~4 ms/frame at 2×→1080p (up to 240 FPS), 7–15 ms at 4K, 35–56% cloud-gaming bandwidth reduction, competitive no-reference perceptual quality vs DLSS/FSR with no engine data. Strongest published existence proof for the OSS ≤1M student tier (#4/#9): reparameterization pattern feeds the #9 student-backbone ablation, and the ConvLSTM ablation (−5 dB PSNR on CS2 without it) quantifies cheap recurrent temporal context as the highest-leverage component in a tiny game-SR model.

**[D2GS-4K]** Zhang and Zhu (2026). *Deformable 2D Gaussian Splatting for Efficient 4K Video Compression*. arXiv:2609.14129.
> Real-time UHD video codec representing each GOP as a coarse-to-fine multi-scale 2DGS structure with a lightweight deformation network; beats H.265 on LPIPS with real-time decoding. Closest external analogue to the v7 persistent 2D Gaussian canvas to date — persistent 2D-Gaussian field warped across frames by a deformation network — and external validation that 2D-GS temporal representations hit real-time decode budgets. Multi-scale hierarchy is a candidate donor for #12 packed-canvas capacity allocation. Mirrored (sha256 a5dd950c…).

**[TFE]** Pu, Tian, Zhang, Zhang, Guo, Zhu, and Guo (2026). *Tile-based Frame Extrapolation*. SIGGRAPH Asia 2026 Conference Papers, DOI 10.1145/3829340.3842212 (not on arXiv). Code: NJUCG/tile-based-frame-extrapolation; project site thdlrt.github.io/tile-based-frame-extrapolation-site/ (supplement mirrored, sha256 7df716bc…).
> Frame extrapolation for deferred-rendered scenes: two-path adaptive warp guided by target-time G-buffers, geometric mask selects unreliable-reprojection regions, sparse tiles refined by a lightweight TRNet with residuals applied inside the mask, plus a shadow-buffer-guided reshade pass for moving shadows. 2.70 ms complete pipeline at 720p (RTX 5070 Ti, TensorRT FP16; ExtraSS 3.67, ExtraNet 6.10, Mob-FGSR 0.11 at much lower quality) and PSNR 30.43 / SSIM 0.9365 / LPIPS 0.0614 avg over six UE5 scenes — beats both CNN-extrapolation baselines on quality. New direct OSS-FX comparison row; third independent instance (with GSReuse and VS-Splat) of spend-compute-only-where-reprojection-fails tile gating for the spawner design note. Requires target-time G-buffers (added latency) — the exact cost OSS-FX's α∈(0,1) single-endpoint rendering avoids.

## Research brief 2026-09-16

**[DecoGS]** Sulo, Supikov, Demir, and Liu (2026). *DecoGS: Adaptive Static-Dynamic Decoupling of 3D Gaussians for Free-Viewpoint Video Streaming*. arXiv:2609.17230.
> Online training of 3D Gaussians from streaming video that selectively updates only spatiotemporal regions exhibiting motion or photometric change, with gradient gating and visibility filtering; 34.55/31.60 dB PSNR on N3DV/MeetRoom, 261 FPS rendering, and **70× lower temporal flicker** than the best prior streaming method, no large-scale pretraining. Training-time member of the spend-compute-only-where-things-move family (GSReuse, TFE, VS-Splat are runtime-side) and the first with a quantified flicker number — the metric dimension the v7 canvas must defend under bad engine-MVs or spurious spawns. Donor for the spawner activity-gate design note and candidate control for #12's bounded-canvas update cost; "no large-scale pretraining" supports the small-data teacher tier.

## Research brief 2026-09-17

**[FastFlowGS]** Mallick, Cherdchusakulchai, Ruiz Olle, Mosella-Montoro, Ribeiro-Gomes, Carrasco, and De la Torre (2026). *Racing in Volume with Flow Ensembles* (FastFlowGS + Monaco4D). arXiv:2609.16310.
> Streaming 4D Gaussian Splatting of fast-moving subjects from a sparse ring of fixed external cameras (F1 sequences; photorealistic UE5 benchmark Monaco4D), fusing sparse matches, semi-dense tracks, and dense optical flow — each lifted to 3D with geometric uncertainty and combined through a Kalman-style temporal update; +12.6% VMAF at 35% greater efficiency (CMU-Panoptic), up to +18.6% dynamic-region PSNR with 28.3% lower per-frame optimization time (Monaco4D). The uncertainty-weighted multi-signal motion fusion is the closest published analog to the v7 spawner's evidence problem — engine MVs, rendered-image motion, and depth as signals of differing reliability, fused recursively under per-signal uncertainty rather than trusted singly. Multi-signal-fusion member of the spend-compute-only-where-things-move family (GSReuse, TFE, DecoGS, VS-Splat). ⚠ Published Sep 14; missed by every prior brief including the Mon-14 listing scan — first confirmed series coverage gap; listing-scan coverage audit queued.

**[M3-Ultra-Opt]** Ochiai (2026). *Systematic Optimization of Real-Time Diffusion Model Inference on Apple M3 Ultra*. arXiv:2605.16259.
> 10-phase optimization study for real-time img2img diffusion on M3 Ultra (60-core GPU, 512 GB unified): CoreML conversion, quantization, Token Merging, Neural Engine, compact distilled models (SDXS-512), interpolation, flow-based frame skipping, distillation → 22.7 FPS at 512². Headline finding: the Apple Silicon optimization landscape is fundamentally different from CUDA — quantization gives no speedup, parallel inference is ineffective, the Neural Engine is unsuitable for large-scale models. Planning input for #8's Metal lane: the ≤1M student should bet on CoreML + compact reparameterized convs + pipelining, with quantization demoted from presumed win to measured experiment. Complements MLX-DLSS as the Apple-lane evidence base.

## Research brief 2026-09-26

**[AdaTex4D]** Jiang, Wang, Yuan, and Ma (2026). *AdaTex4D: Adaptive Texture Capacity Allocation for 4D Gaussian Splatting*. arXiv:2609.29963.
> Frozen dynamic-Gaussian backbones allocate anisotropic per-primitive texture capacity from visibility-normalized gradients, temporal peak demand, and deformed scales; reports 47.3% of uniform texture bytes at comparable reconstruction quality. Candidate #12 ablation for visibility/temporal-demand-conditioned canvas feature capacity; dense texture materialization currently limits throughput gains, so test packed storage and end-to-end latency before claiming a runtime benefit.

## Research brief 2026-10-07

**[Pixel-to-Gaussian ContinuousSR]** Peng et al. (2025). *Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling*. arXiv:2503.06617. One-pass pixel-to-Gaussian arbitrary-scale SR reports ~1 ms rendering per scale; relevant continuous-reconstruction and rasterization-cost comparator for the OSS canvas, though not temporal/game inference evidence.

## Research brief 2026-10-02

**[GS-STVSR]** Shi et al. (2026). *GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting*. arXiv:2604.18047. Continuous time-conditioned 2D Gaussians, flow-guided motion, covariance resampling, adaptive offsets; directly relevant precedent for OSS's native time/covariance axis, though video VSR and game-rendered extrapolation differ.

**[NanoVSR]** Pawlicki, Kańduła, Pucek, and Dobies (2026). *NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices*. arXiv:2607.10495. Structural reparameterization collapses a compact spatiotemporal CNN into standard convolutions for accelerator deployment; useful ≤1M student design candidate, with reported 644k-param variant.

**[SPEED]** Wu et al. (2026). *SPEED: One-Step Pixel Diffusion for High-quality Video Frame Interpolation*. arXiv:2607.15585. One-step pixel-space VFI and Noise-Update-Only Attention offer a frame-generation comparison/control; likely too compute-heavy for OSS's <2 ms target but worth tracking as quality frontier.

## Research brief 2026-10-11

**[GSCV]** Yang, Xia, Yang, Van Der Auwera, and Li (2026). *Efficient Gaussian Splatting Sequence Compression with Standard Video Codecs*. arXiv:2610.07795. Inter-PLAS improves inter-frame correlation before standard video-codec compression of Gaussian sequences; adjacent option for packed-canvas/checkpoint storage if profiling confirms bandwidth pressure, not a replacement for the real-time renderer.

## Research brief 2026-10-12

**[Amulet]** Künzel et al. (2026). *Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading*. arXiv:2608.10423. Predictively filled sparse layered image-space cache for future-view extrapolation with adaptive tile shading; direct disocclusion/cache-scheduler comparator for OSS-FX. https://arxiv.org/abs/2608.10423

**[NanoVSR]** Pawlicki, Kańduła, Pucek, and Dobies (2026). *NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices*. arXiv:2607.10495. Structural reparameterization fuses a compact temporal VSR model into standard inference convolutions; direct candidate deployment/export baseline for OSS student distillation.

## Research brief 2026-10-13

**[SparseView4DGS]** Wang et al. (2026). *Sparse-View 4D Gaussian Splatting via Spatiotemporal Priors and Generative Assistance*. arXiv:2610.04606 (SIGGRAPH Asia 2026 Workshops). Region-adaptive initialization/densification and intermediate-time motion supervision are candidate v7 training ablations; its multi-view reconstruction and diffusion pseudo-supervision differ from real-time game SR.

## Research brief 2026-10-14 (evening)

**[Mobile-4DGS]** Du, Fang, Zhu, Hartley, and Yu (2026). *Mobile-4DGS: Unified Static-Dynamic Real-time Mobile Gaussian Splatting*. arXiv:2610.05289. Compact explicit 4D representation with second-order Gaussian motion for real-time mobile rendering; candidate for non-linear OSS-FX motion extrapolation and Pico-tier budget comparison.

**[TETRIS-4DGS]** Wu, Liu, Li, Yu, and Liao (2026). *TETRIS-4DGS: Efficient Post-Training Compression of 4D Gaussian Splatting*. arXiv:2610.14663. Importance-based adaptive quantization and codec-friendly packing (>40× compression); reference for packed canvas layout (issue #12) and quantized Gaussian state (D10).

**[DynStream]** Xian, Zhou, Xiong, Wang, and Yang (2026). *DynStream: Online Streaming 4D Gaussian Reconstruction of Dynamic Worlds from Unposed Video*. arXiv:2610.09720. Online incremental 4D Gaussian updating; conceptual comparator for persistent-canvas spawn/retire policy.

## Research brief 2026-10-15

**[Frozen History]** Yu and Zhao (2026). *Frozen History: When Temporal Losses Stop Accumulation in Learned Real-Time Supersampling*. arXiv:2610.15752. Shows temporal-consistency losses can suppress history accumulation under accurate registration; audit OSS canvas registration/jitter/depth handling and temporal-loss weighting against its four error modes.

## Research brief 2026-10-15 (evening)

**[ReFrame]** Liu and Aamodt (2025). *ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering*. arXiv:2506.13814 (ICML 2025). Cross-frame feature caching for encoder-decoder rendering networks (denoising, supersampling, frame extrapolation), ~1.4× speedup with negligible quality loss; candidate inference-budget lever for the ≤1M student.

**[Frozen History — detail]** Yu and Zhao (2026), arXiv:2610.15752. Correction/extension of the morning entry: ×4 model 2.03 ms/1080p on H200; +0.34–0.62 dB over best published QRISP; temporal weight 0.25 costs 0.7–0.9 dB under exact registration. Code: github.com/CV-AC/Frozen_history.

**[MihawkGen]** mihawk-99 (2026). *PS5 MihawkGen* (software, GitHub). 0.43M-param two-frame neural interpolation, ~2.2 ms at 4K on base PS5; size/latency comparator for the Pico tier and the OSS-FX path.

## Research brief 2026-10-16

**[ResStreamGS]** Liu, Wu, Luo, Na, Hu, and Quan (2026). *ResStreamGS: Residual-Sensitive Dual-Stream Gaussian Splatting for Mobile Volumetric Video Streaming*. arXiv:2610.21121. Splits Gaussians into temporally stable (sent once per GOP) and residual-sensitive (per-frame) sets; +7.5–8.4 dB post-codec, −56.5% payload, 30.9 FPS on Snapdragon 778G; reference for canvas stable/dynamic refresh policy (H004, issue #12).

**[QuantSR+]** (2026). *QuantSR+: Pushing the Limit of Quantized Image Super-Resolution Networks*. arXiv:2605.22351. Unified quantization operators, network design, and training optimization for low-bit SR; candidate recipe for student quantization (#9/#4).

**[OptiScaler-DLSS5 edit-cache]** Skynizz (2026). *optiscaler-dlss5 v1.2.1* (software, github.com/Skynizz/optiscaler-dlss5). Community fork that runs DLSS 5 neural rendering every N frames and reprojects the cached per-pixel edit with motion vectors (per-pixel depth/colour validation, anti-ghost fingerprint); 13.2 ms → ~0.9 ms/frame on RTX 4070. Prior art for MV-reprojected cached neural state / canvas refresh cadence (H004, H009).

## Research brief 2026-10-17

**[OpenDLSS-NR]** maanHimself (2026). *OpenDLSS-NR* (software, github.com/maanHimself/OpenDLSS-NR, MIT). Bit-exact Vulkan reimplementation of NVIDIA's DLSS 5 neural-rendering network (71-block Swin/ViT, FP8 E4M3 via cooperative matrix; user-supplied weights); reference for Vulkan/FP8 kernel structure (#8) and the FP8-on-non-Blackwell cost (39–50% fps), and a legal-differentiation foil for clean-room OSS.

**[Integer-Only Training]** (2026). *Memory-Efficient Integer-Only Training of Deep Neural Networks*. arXiv:2610.16298. Integer-only training algorithm; low priority, only relevant if training VRAM on the 3080 Ti becomes binding.

## Research brief 2026-10-18

**[Sony FGSS patent]** Sony Interactive Entertainment (2026). *Frame Generation Super Sampling* (patent application filed/published 2026-10-15; reported by Patentlyze, Wccftech 2026-10-16). Encoder/decoder that predicts a future synthetic frame from a rendered frame, optional motion vectors, and controller input (multiple candidate futures; non-ML motion-vector fallback); prior-art/IP-landscape reference for the OSS-FX α<1 extrapolation path and the no-ML variant (#5). Patent-level only; claims not yet read.

**[DLSS-NR-on-AMD 0.8.0]** Daniel Blanco / danielblnc (2026). *DLSS-NR-on-AMD* (software, github.com/danielblnc/DLSS-NR-on-AMD; license unstated/NOASSERTION; v0.8.0 reported by VideoCardz 2026-10-18). HIP + DX12 compute-kernel port that runs NVIDIA DLSS SR, DLSS 5 Neural Rendering, Ray Reconstruction (Model F) and Frame Generation/MFG up to 4x on Radeon RX 7000/9000 with user-supplied DLLs; reference for HIP kernel structure (#8) and evidence of community cross-vendor demand OSS targets cleanly.

## Research brief 2026-10-19

**[dGS/dBS]** Gao, Z., Planche, B., Nguyen, V. N., Zheng, M., Choudhuri, A., Innanje, A. (2026). *Direct Conditional Parameterization for N-Dimensional Splatting*. arXiv:2610.22778 (NeurIPS 2026). Replaces covariance-derived conditional slicing in N-D Gaussian/Beta splatting (view + time) with explicit per-effect parameters; 6.9–7.7× faster slicing, up to +1.26 dB; closest published relative of the v7 N-D canvas with a time-axis slice (α<1 extrapolation) and a candidate way to cheapen it (H002/H004).

**[AMD lightweight FSR 4]** Huynh, J. / AMD (2026). *FSR 4 for APUs, laptops and handhelds by end of 2026 via a dedicated lightweight model* (TheElec interview 2026-10-08; reported by Digital Foundry, The Verge, TechPowerUp). Confirms a vendor small non-FP8 SR model for non-matrix-hardware APUs; benchmark target for the Pico/student tier (#4, #7, #8). No latency numbers yet.

**[DLSS FG Preset C]** NVIDIA (2026). *DLSS Frame Generation Performance Mode "Preset C"* (NVIDIA app beta 2026-10-13; VideoCardz). Swaps FG's second (optical-flow) AI model for a lighter algorithm, up to 35% faster on RTX 5060 at slight quality cost; reference for no-ML/cheap-extrapolation design (#5, OSS-FX).

## Research brief 2026-10-20 (evening)

**[FSR 4.1 INT8 expansion]** AMD (2026). *FSR Upscaling 4.1 expands to Radeon RX 6000, Ryzen AI Max/AI 9, Ryzen Z2 and Xbox Series X|S* (AMD newsroom + Adrenalin 26.10.1 release notes, 2026-10-20; AMD dev article *Upscale Everything*). Same ML upscaler retuned for INT8 on shader hardware (~13% frame cost vs FSR 3 at 4K Balanced on RX 6900 XT; 4K Performance ≈1.3 ms on RX 9070 XT), plus a handheld model <½ the size of the INT8 model; benchmark/teacher-source reference for the INT8 student path (#4, #7, #8, #9).

## Research brief 2026-10-21

**[KoalaGS]** Beche, R., de Charette, R. (2026). *KoalaGS: Complexity-Aware Continuous Level of Detail for Gaussian Splatting*. arXiv:2610.28615. One Gaussian set stays effective when only a prefix is rendered, allocating capacity by image complexity; reference for the H004 budgeted tile scheduler / #12 packed canvas and for serving Pico/Standard/Heavy tiers from one canvas ordering (3DGS scene-level, idea transfer only).

**[RLSplat]** Kim, I., Joo, M., Park, J., Kim, S., et al. (2026). *RLSplat: Resource-Linear Budget Control for Feed-forward 3D Gaussian Splatting*. arXiv:2610.27712. Budget-controllable feed-forward 3DGS whose inference cost scales linearly with the Gaussian budget (no predict-dense-then-prune); relevant to spawner design under a <2 ms budget (H004, #4).

**[fsr4-shader-performance-overrides]** zgauthier2000 (2026). *fsr4-shader-performance-overrides* (software, github.com/zgauthier2000/fsr4-shader-performance-overrides; release dll-2026-10-19). Community replacement shaders that speed up FSR 4.1.1 on RX 6000/7000/iGPU; datapoint that RDNA2/3 INT8 upscaling is shader-bound and hand-tunable (#8, #9) and a candidate teacher-capture source for #7 (legal/ToS check pending).

## Research brief 2026-10-21 (evening)

**[Wavelet-Space SR]** (2025). *Wavelet-Space Super-Resolution for Real-Time Rendering*. arXiv:2508.16024. Wavelet-space feature decomposition inside a neural SR network for real-time rendering; surfaced in this run's sweep and not previously in the bibliography. Candidate cheap-backbone idea for #9/#10 (read before the RepVGG/ECBSR ablation). Authors/venue not verified this run.
