# OSS Research Brief — 2026-05-31

## Project Status Summary
- **Latest commits (2 weeks):**
  - `0660f55` wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc
  - `56a9a65` trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)
  - `d1255c8` trainer: WindowsPath shim on Linux for cross-OS checkpoint resume
  - `e1ab7fe` hat.py: use flash-attn 2 directly when available, fallback to SDPA
  - `e9b4b65` trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)
  - `26cd9da` dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB
  - `83b4f07` v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast
  - `baf4b02` worker: apex .com → .org 301 redirect
  - `eb0ec2e` trainer: bf16 autocast around forward + loss
  - `5c2f2e7` trainer: add throughput baselines (cudnn.benchmark, TF32, torch.compile)
- **Open issues:** #14 (VRAM footprint), #13 (Nsight L2 profiling), #12 (packed low-rank canvas), #10–5 (v6.2.1 ablations & cross-vendor ports), #4 (HAT-Tiny student distillation)
- **Training dashboard:** v7.0 Pico-005 (`srcnn-v7.0-pico-005`) stale at step 9,100 — trainer degraded, stuck for 7 minutes, now 11h stale. Loss 0.12692. ETA ~1.6 days on 3080 Ti if resumed.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047, Apr 2026) — Mingyu Shi et al. 2D-GS framework for continuous spatio-temporal VSR at arbitrary scale factors. Replaces dense INR grid queries with Gaussian kernel evolution via optical-flow-guided motion module + covariance resampling alignment. Near-constant inference time at ×2–×8, 3× speedup at ×32. **Directly relevant: GS-STVSR's covariance resampling is already OSS v6 core upsampler; this paper validates the 2D-GS-as-SR-operator design direction. Their flow-guided motion module is the offline analog of OSS's exact-engine-MV pipeline — OSS has stronger input signal (engine MVs vs. estimated flow).**

- **STGV: Spatio-Temporal Hash Encoding for Gaussian-based Video Representation** (arXiv:2604.10910, Apr 2026) — Lin et al. Decouples video features into 2D spatial hash encodings (static) + 3D temporal hash encodings (dynamic). +0.98 dB PSNR over other GS-based video methods. **Relevant: the explicit static/dynamic decomposition principle is similar to OSS's canvas (persistent static) + spawner (dynamic disocclusion) split. Their hash encoding approach could inform a more memory-efficient canvas representation.**

- **TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting** (arXiv:2604.00538, Apr 2026) — Yeom et al. SE(3) transformations + hierarchical Bézier residuals for continuous rigid motion, beating piecewise-linear 4DGS approaches. Scales to 600-1200 frame sequences without memory explosion. **Relevant: their continuous SE(3) formulation is a more principled motion model than piecewise linear — could inspire v7+ canvas motion parameterization improvements, especially for long game sessions.**

- **RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting** (CVPR 2026, arXiv:2603.13783) — Wang et al. Optical-flow-guided initialization + triple-rendering supervision to prevent temporal aliasing in 4DGS interpolation. **Relevant: the temporal anti-aliasing problem is exactly what OSS's covariance resampling addresses — their diagnosis of ghosting as "temporal aliasing" validates our approach.**

- **StreamSTGS: Streaming Spatial and Temporal Gaussian Grids** (AAAI 2026, arXiv:2511.06046) — Ke et al. Compacts dynamic GS to 170 KB/frame via 2D image encoding + video temporal features, adaptive bitrate without retraining. **Relevant: the compression scheme (canonical Gaussians as 2D image, temporal features as video) is a potential canvas serialization path for cloud/streaming OSS deployment.**

- **VidSplat: Gaussian Splatting Reconstruction with Geometry-Guided Video Diffusion Priors** (arXiv:2605.11424, May 2026) — **Relevant: video diffusion priors for GS reconstruction; potential teacher signal for OSS if geometry-guidance generalizes.**

- **Rigid-aware 4D Gaussian Splatting (RiGS)** (arXiv:2605.23672, May 2026) — Single monocular video to 4DGS with multi-scale temporal motions. **Adjacent: monocular-to-4D is not OSS's problem (we have stereo engine data), but multi-scale motion decomposition is interesting.**

- **ContinuousSR: Pixel to Gaussian — Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling** (ICLR 2026, arXiv:2503.06617) — Long Peng et al. (USTC + Huawei). Pixel-to-Gaussian paradigm: builds 2D continuous HR signal from LR in single pass, then ~1ms arbitrary-scale rendering. ICLR 2026 accepted. **Direct competitor to OSS's Gaussian canvas concept for the single-frame SR path. Their 1ms rendering claim is extremely aggressive — needs benchmarking against OSS rasterizer. Key risk: if ContinuousSR generalizes to temporal + game content before OSS ships, it preempts the core technical claim.**

### Super-Resolution / Upscaling

- **FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution** (CVPR 2026, arXiv:2510.12747) — Zhuang et al. One-step streaming diffusion VSR at ~17 FPS (768×1408, A100). Three-stage distillation + block-sparse attention + tiny conditional decoder. 12× speedup over prior one-step diffusion VSR. **Significant: first diffusion VSR approaching real-time. Not game-ready (17 FPS on A100 is still ~60ms — well above OSS's <2ms target), but the distillation pipeline (3-stage, train-friendly) is reference material for OSS's own teacher→student path. Their VSR-120K dataset (120K videos) could be useful for OSS temporal training.**

- **DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution** (CVPR 2026, arXiv:2603.22271) — Lv et al. Three-stage distillation: progressive guided initialization → dual-stream (DMD + RFS-GAN) → preference-guided refinement. **Relevant: dual-stream distillation strategy could inform OSS's v6.2.1 distillation pipeline where distribution matching + adversarial signals compete.**

- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization** (CVPR 2026 MAI Workshop, arXiv:2604.20291) — Nguyen et al. Extract-refine-upsample design, Mamba teacher → re-parameterizable student, INT8 TFLite. 29.79 dB PSNR at ×3 SR. Three-stage pipeline with QAT on fused deploy graph. **Directly actionable for OSS student: the re-parameterizable backbone + QAT-on-fused-graph approach is exactly what we need for the ≤1M-param student exported to FP8/INT8. Their weight clipping + BatchNorm recalibration for quantization stability is a concrete recipe.**

- **NTIRE 2026 Efficient Super-Resolution Challenge Report** (CVPRW 2026, arXiv:2604.03198) — Comprehensive benchmark of real-time SR models. Teams widely adopted re-parameterization for inference speed without capacity sacrifice. **Confirms re-parameterization is the community standard for efficient SR — validates OSS's v6.2.1 RepVGG/ECBSR ablation direction (issue #9).**

- **EfRLFN: Efficient Real-Time Super-Resolution for Streaming** (arXiv:2602.11339, Feb 2026) — Bogatyrev et al. (MSU). Efficient Channel Attention + tanh activation, StreamSR YouTube dataset. **Low relevance for game SR, but their StreamSR dataset (compressed streaming content) could be a robustness test for OSS degradation pipeline.**

- **GameSR: Real-Time Super-Resolution for Interactive Gaming** (ICLR 2026) — Engine-agnostic, runs on RGB frames only, no engine hooks. **Competitor watch: engine-agnostic approach that doesn't need MVs — if they achieve competitive quality, it undercuts OSS's "exact engine MVs are the advantage" thesis.**

- **Continuous Space-Time Video Super-Resolution with 3D Fourier Fields** (ICLR 2026, arXiv:2509.26325) — ETH Zurich. VFF representation: unified 3D Fourier field for spatio-temporal SR with built-in analytical anti-aliasing PSF. **Theoretical competitor: their unified spatio-temporal representation + analytical PSF is philosophically similar to OSS's N-D Gaussian canvas — both avoid decoupling spatial/temporal. Difference: Fourier fields are implicit (query-based), OSS is explicit (canvas-based). The VFF's analytical PSF is a cleaner AA solution than Gaussian EWA — worth studying.**

### Frame Extrapolation / Generation

- **FramePack: Packing Input Frame Context in Next-Frame Prediction** (arXiv:2504.12626, Apr 2025) — Neural network structure for training next-frame prediction models for video generation. **Adjacent: frame prediction for video generation, not real-time game extrapolation. Their "frame packing" concept may have parallels to OSS's canvas accumulation.**

### Inference Optimization

- **Deep-learning Evaluation of Execution Parallelism in GPU Kernels** (arXiv:2604.14552, Apr 2026) — Systematic INT8 TensorRT profiling across GPU tiers. L4 achieves 58× speedup for ResNet via INT8 TensorRT. **Confirms INT8/FP8 is the right inference target for OSS student. Their profiling methodology is a reference for OSS's own kernel benchmarking.**

- **Systematic Optimization of Real-Time Diffusion Model Inference on GPUs** (arXiv:2605.16259, May 2026) — Quantization + kernel fusion survey for real-time diffusion. **Reference for OSS kernel optimization strategy. Their finding that AWQ/GPTQ + TensorRT INT8 is the current Pareto frontier for GPU inference aligns with OSS's planned FP8 TRT export.**

- **DNN Compilation with Virtual Tensors for Data Movement Elimination** (arXiv:2604.09558, Apr 2026) — Operator fusion for eliminating data movement. **Reference: OSS's planned fused-op kernels could benefit from virtual tensor approaches to reduce memory bandwidth.**

## Archive.org Findings
- No novel archived papers found that aren't already accessible on arXiv. The GS-STVSR, GSASR, GRAPE, and STSS papers are all live on arXiv with no take-down history detected. No need to archive at this time.

## GitHub / Open Source

- **ContinuousSR** (ICLR 2026, [github.com/peylnog/ContinuousSR](https://github.com/peylnog/ContinuousSR)) — Official code released. USTC + Huawei. 2D Gaussian modeling for arbitrary-scale SR with "1ms per scale" rendering claim. **HIGH PRIORITY to benchmark against OSS rasterizer.**
- **FlashVSR** (CVPR 2026, [github.com/OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR)) — Code, pretrained models, and VSR-120K dataset released. **Dataset useful for OSS temporal training data augmentation.**
- **SR3R** (CVPR 2026, [github.com/XiangFeng66/SR3R](https://github.com/XiangFeng66/SR3R)) — Feed-forward GS for 3D SR from sparse views. **Confirms feed-forward GS prediction across scenes is viable (already in BIBLIOGRAPHY).**
- **GSASR** ([github.com/ChrisDud0257/GSASR](https://github.com/ChrisDud0257/GSASR)) — ICCV 2025. SoTA arbitrary-scale 2D-GS SR. Active, weights released. **Already tracked in BIBLIOGRAPHY as primary prior-art baseline.**
- **Awesome-CVPR2026-Low-Level-Vision** ([github.com/Kobaayyy/Awesome-CVPR2026-...](https://github.com/Kobaayyy/Awesome-CVPR2026-CVPR2025-CVPR2024-CVPR2021-CVPR2020-Low-Level-Vision)) — Curated list including SR3R, FlashVSR, and other relevant GS+SR papers from CVPR 2026.
- **NTIRE 2026 ESR** ([github.com/Amazingren/NTIRE2026_ESR](https://github.com/Amazingren/NTIRE2026_ESR)) — Solution code for NTIRE 2026 Efficient SR challenge, SPAN baseline. Rep-parameterization demonstrated.
- **Upscayl** ([github.com/upscayl/upscayl](https://github.com/upscayl/upscayl)) — Open-source AI image upscaler. Still the most starred SR tool on GitHub.
- **NVIDIA/DLSS** ([github.com/NVIDIA/DLSS](https://github.com/NVIDIA/DLSS)) — Official DLSS SDK. No major updates observed recently.

## News

### NVIDIA / DLSS
- **DLSS 4.5 launched at CES 2026 (Jan):** Dynamic Multi Frame Generation up to 6× ("automatic transmission for GPU"), cleaner upscaling. Spring 2026 rollout for RTX 50-series.
- **DLSS 5 announced at GTC 2026 (Mar):** "GPT moment for graphics." Real-time neural rendering model infusing pixels with photoreal lighting/materials. Fall 2026 release. Called "3D-Guided Neural Rendering" — combining structured 3D engine data with generative AI. **Major competitive signal: DLSS is pivoting from pure upscaling to neural re-rendering. This makes OSS's unified canvas approach (SR+FG from one model) even more prescient — NVIDIA is heading the same direction but with a bigger model. OSS's pitch of "same quality, smaller model, open source" becomes more compelling as the proprietary ceiling rises.**
- **DLSS 4.5 games arriving:** 007 First Light, Aniimo, Barkour all launching May 2026 with DLSS 4.5 support.

### AMD / FSR
- **FSR 4.1 confirmed for RDNA 3 (July 2026) and RDNA 2 (2027):** AMD backporting ML-based FSR upscaling to older GPUs using INT8 quantization to maintain performance. **Significant: FSR 4.1 on RDNA 2/3 dramatically expands the installed base for ML upscaling — OSS's cross-vendor kernel story becomes more relevant as AMD pushes ML-SR to older hardware.**
- **FSR 4 backward compatibility:** 60+ games upgradeable from FSR 3.1 to FSR 4 on RX 9000 series.
- **AMD rebranded:** "FSR Upscaling" (formerly FSR 4), "FSR Frame Generation" — simplifying naming.

### Intel / XeSS
- **XeSS 3 Multi-Frame Generation now live:** Driver 32.0.101.8509 (WHQL) enables XeSS 3 MFG across all Arc GPUs including Arc B580. Inserting up to 3 generated frames between rendered frames. MFG can be enabled in any title with XeSS 2 support. **Significant: Intel is the first to backport MFG to existing hardware — driver-level, not game-level. This pressures NVIDIA's RTX 50-series exclusivity and validates the frame-gen-on-existing-hardware approach OSS is building.**

### Apple / Metal
- **Metal 4:** First-class ML tensor support in API + shading language. Neural rendering techniques showcased at WWDC 2025. M5 chip with 16-core Neural Engine and 10-core GPU with per-core Neural Accelerator. **Metal 4 ML tensors + neural rendering support makes the Apple Silicon backend for OSS more viable — the Metal Compute path planned for v6.2 (issue #8) should target Metal 4 APIs.**

### Unity
- **Unity 6.6:** Planned new upscaling framework with "latest DLSS4" support. **Engine-level upscaling abstraction becoming standard — OSS DLL-shim integration path should be compatible with Unity's upcoming framework.**

## Action Items / Follow-ups

- [ ] **URGENT: Resume v7-pico-005 training.** Trainer stuck at step 9,100 for 11h. Check if 3080 Ti is still available or if another process reclaimed it. If killed, resume from checkpoint 00005000.pt per `wipe-prep` commit.
- [ ] **Benchmark ContinuousSR against OSS rasterizer.** Their "1ms arbitrary-scale rendering" claim needs validation — if it holds, it's a significant competitive threat to OSS's core Gaussian-canvas-for-SR thesis.
- [ ] **Deep-read ContinuousSR (ICLR 2026).** Pixel-to-Gaussian paradigm is the closest published work to OSS's approach. Understand exactly how their Gaussian field is built, what MLP/encoder they use, and whether their approach could be extended to temporal.
- [ ] **Deep-read DUO-VSR distillation pipeline.** The three-stage DMD+RFS-GAN approach may directly inform OSS's v6.2.1 distillation (issue #7) — dual-stream distribution matching + adversarial could solve training instability issues.
- [ ] **Evaluate VSR-120K dataset from FlashVSR.** 120K videos could augment TartanAir temporal training — especially for real-world degradation robustness.
- [ ] **Incorporate INT8 deployment recipe** (arXiv:2604.20291) into student export pipeline. Their QAT-on-fused-graph + weight clipping + BN recalibration is a proven approach for deployment-quality INT8 SR.
- [ ] **FSR 4.1 INT8 on RDNA 3 (July 2026):** This is a direct competitor benchmark. Once FSR 4.1 lands on RDNA 3, OSS should measure against it on TartanAir oldtown for apples-to-apples.
- [ ] **XeSS 3 MFG driver-level approach:** Study whether OSS's DLL-shim could similarly enable MFG-like behavior across any game — driver-level frame generation is a compelling feature.
- [ ] **Metal 4 backend scoping:** Apple's ML tensor support in Metal 4 + M5 Neural Engine means a Metal Compute path for OSS is more viable than 6 months ago. Scope effort for v8+.
- [ ] **DLSS 5 neural rendering pivot:** Track closely. If DLSS 5 delivers "3D-guided neural rendering" in Fall 2026, OSS needs to show its unified canvas approach achieves competitive visual quality at a fraction of the model size before DLSS 5 ships. The window to demonstrate OSS's value proposition is ~4-5 months.
