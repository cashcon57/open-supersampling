# OSS Research Brief — 2026-06-10

## Project Status Summary
- Latest commits:
  - `a97a180 feat: add tailnet-gated training controls to dashboard`
  - `fcf8084 feat: tailnet-gated training control API + dashboard panel`
  - `0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc`
  - `56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)`
  - `d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume`
  - `e1ab7fe hat.py: use flash-attn 2 directly when available, fallback to SDPA`
  - `e9b4b65 trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)`
  - `26cd9da dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB`
  - `83b4f07 v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast`
  - `baf4b02 worker: apex .com → .org 301 redirect`
- Open issues:
  - #14: D3: measure v6.2-pico-002 VRAM footprint after v6.2 wiring and checkpoint exist
  - #13: H004: run Nsight L2 hit-rate profile on 3080 Ti CUDA host
  - #12: H004: implement packed low-rank canvas layout before claiming 32B/Gaussian
  - #10: v6.2.1 ablation: decoupled G-buffer encoder before concat-fusion
  - #9: v6.2.1 ablation: RepVGG/ECBSR/RepSR reparameterization for student backbone
  - #8: v6.2: cross-vendor kernel ports (HIP / Metal / Vulkan / SYCL)
  - #7: v6.2.1: distillation pipeline with FSR 4 binary as teacher signal
  - #6: v6.2-pico-002: launch on 3080 Ti once codex T1-T5 land
  - #5: v6.2: No-ML shader variant (ships alongside v6.2)
  - #4: v6.2: HAT-Tiny student distillation (≤1M params)
- README snapshot: public status remains **pre-alpha**. Active architecture is **v7 / `srcnn-v7.0-pico-005`**: N-D Gaussian canvas with native time axis, 3D Cholesky covariance with `V_xt`, HAT-Tiny teacher, parent-child loss-adaptive density, Mip-Splatting AA, and `alpha=0.5` OSS-FX supervision. Shipping student is still not trained; target remains ≤1M-param CNN/FP8 and <2ms RTX-class inference.
- Training dashboard: browser-rendered dashboard reports **v7.0 Pico OSS-FX live training** at **20,950 / 100,000 steps** (21%), loss **0.1087**, viz @ step 20,000, ETA ~1.4d, but service tooltip says **Trainer offline/stuck at step 20,950 for 46h**. Watcher healthy (published 1s ago), CF Worker healthy (HTTP 200 in 87ms), R2 origin healthy, GPU stats unavailable / host offline or rsync failed. This is the main ops blocker.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Envision4D: Envisioning Visual Futures via Feed-forward 4D Gaussian Splatting for Autonomous Driving** (arXiv:2606.10656, 2026-06-09; Qi Song, Yifei He, Chi Zhang, Zheng Fu, Xuhe Zhao et al.) — Feed-forward 4D-GS future-scene extrapolation for autonomous driving, aimed at ghosting and large-displacement failures in future prediction. Why it matters to OSS: direct conceptual overlap with v7's native time axis and OSS-FX, but OSS has the stronger input contract via exact engine motion vectors. http://arxiv.org/abs/2606.10656v1
- **ManiSplat: Manipulation Trajectory Synthesis from Monocular Video via Decoupled 3D Gaussian Splatting** (arXiv:2606.10645, 2026-06-09; Wenhao Hu, Haonan Zhou, Liu Liu, Yun Du, Xinjie Wang et al.) — Dynamic/interactive 3DGS decomposition for articulated manipulation scenes from monocular video. Why it matters to OSS: lower priority than Envision4D, but decoupling static/dynamic/contact-like components may inform persistent vs transient canvas channels. http://arxiv.org/abs/2606.10645v1
- **LEGS: Laplacian-Enhanced Gaussian Splatting with a Nonlinear Weighted Loss** (arXiv:2606.07932, 2026-06-06; Yongfei Guo, Qizhou Huo, Xuan Sun, Yuanhao Gong) — Adds Laplacian/detail-aware nonlinear weighting to Gaussian optimization to sharpen contours and fine structure. Why it matters to OSS: cheap loss-side ablation candidate for v7 detail recovery without changing runtime architecture. http://arxiv.org/abs/2606.07932v1
- **EvoGS: Constructing Continuous-Layered Gaussian Splatting with Evolution Tree for Scalable 3D Streaming** (arXiv:2606.07179, 2026-06-05; Yuang Shi, Simone Gasparini, Géraldine Morin, Wei Tsang Ooi) — Continuous-layered/progressive GS representation using an evolution tree to reduce redundancy and quality discontinuities. Why it matters to OSS: compare against v7 parent-child loss-adaptive density and issue #12 packed low-rank canvas work. http://arxiv.org/abs/2606.07179v1
- **Scene-Adaptive Nonlinear Tone Curves for Pseudo Ground-Truth Generation in Low-Light 3D Gaussian Splatting** (arXiv:2606.11841, 2026-06-10; Mingzhe Lyu, Jinqiang Cui, Hong Zhang) — Generates scene-adaptive pseudo-GT for low-light 3DGS instead of uniform linear gain. Why it matters to OSS: not core SR/FG, but the pseudo-label reliability idea is relevant if OSS mixes real captures, synthetic GT, and teacher outputs.

### Super-Resolution / Upscaling
- **LiteVSR: Lightweight Adaptation of Frozen Diffusion Transformers for Video Super-Resolution** (arXiv:2606.09250, 2026-06-08; Yu Cao, Ziquan Liu, Zhensong Zhang, Jiankang Deng, Shaogang Gong) — Frozen video-DiT VSR with lightweight state-aware adapters and one-step sampling compatibility. Why it matters to OSS: teacher-side adaptation/distillation reference; too heavy for shipping student.
- **QuantSR+: Pushing the Limit of Quantized Image Super-Resolution Networks** (arXiv:2605.22351, 2026-05-21; Haotong Qin, Xudong Ma, Xianglong Liu, Jie Luo, Jinyang Guo) — Low-bit quantization framework for SR networks targeting 2–4 bit precision without severe detail loss. Why it matters to OSS: directly relevant to ≤1M-param FP8/INT8 student deployment and could inform quantization-aware training or post-training calibration. http://arxiv.org/abs/2605.22351v1
- **Fast Image Super-Resolution via Consistency Rectified Flow** (arXiv:2605.12377, 2026-05-12; Jiaqi Xu, Wenbo Li, Haoze Sun, Fan Li, Zhixin Wang) — Few-/single-step generative SR via consistency/rectified-flow style training. Why it matters to OSS: possible high-quality teacher or perceptual prior, not a runtime path.
- **FoA-SR: Faithful or Aesthetic? Profile-Aware Preference Optimization for Real-World Image Super-Resolution** (arXiv:2606.10275, 2026-06-09; Amjad Mahdi Alqarni, Peizhong Ju) — Uses profile-specific preference optimization for faithful vs aesthetic SR outputs. Why it matters to OSS: useful warning for eval design — game upscaling should bias faithful/temporal consistency, not hallucinated aesthetic detail.

### Frame Extrapolation / Generation
- **Ultra Flash: Scaling Real-Time Streaming Video Generation to High Resolutions** (arXiv:2606.09150, 2026-06-08; Luxury, Jie Huang, Zihao Fan, Xiaoxiao Ma, Yuming Li et al.) — Cascaded streaming video-generation pipeline claiming ~30 FPS at 1K and ~18 FPS at 2K via causal latent upsampling, sparse causalization, dynamic cache management, and single-step distillation. Why it matters to OSS: not game-engine MV-based, but a strong signal that cache-aware temporal generation is moving toward real-time; mine for cache scheduling/distillation ideas.
- **DiffST: Spatiotemporal-Aware Diffusion for Real-World Space-Time Video Super-Resolution** (arXiv:2605.13182, 2026-05-13; Zheng Chen, Ruofan Yang, Jin Han, Dehua Song, Zichen Zou) — Diffusion STVSR combines VSR and VFI while trying to address efficiency and spatiotemporal feature use. Why it matters to OSS: direct task-level baseline for unified SR+FG, but diffusion-heavy and likely too slow for runtime.
- **R5DGS: Semantic-Aware 4D Gaussian Splatting with Rigid Body Constraints for Efficient Dynamic Scene Reconstruction** (arXiv:2605.25909, 2026-05-25; Denis Gridusov, Maxim Popov, Sergey Kolyubin) — Adds semantic and rigid-body constraints to 4DGS for dynamic reconstruction and future extrapolation. Why it matters to OSS: suggests rigid/semantic constraints as an optional canvas regularizer for stable moving objects.
- **SKIP: Sparse Keyframe Interpolation Paradigm for Efficient Embodied World Models** (arXiv:2606.00664, 2026-05-30; Ziheng He, Yixiang Chen, Ning Yang, Zhanqian Wu, Qisen Ma) — Generates/interpolates sparse keyframes to reduce rollout cost in embodied world models. Why it matters to OSS: mostly out-of-domain, but supports keyframe/active-frame thinking similar to canvas active-set updates.

### Inference Optimization
- **AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis** (arXiv:2606.09682, 2026-06-08; Jaber Jaber, Osama Jaber) — Compiles a model forward pass into a single persistent cooperative CUDA kernel with static schedule validation. Why it matters to OSS: not SR-specific, but relevant to <2ms inference: persistent/megakernel fusion could reduce dispatch overhead in the student+raster/composite path. http://arxiv.org/abs/2606.09682v1
- **CFRNet: Cycle-Consistent Fixed-Point Training for Real-Time Blind Face Restoration on Consumer Embedded NPUs** (arXiv:2606.06850, 2026-06-05; Fuchen Li, Xinyang Wang, Yahui Zhang, Yuhan Chen, Jiahong Guo) — Fixed-point training for restoration on constrained NPUs, avoiding hard-to-compile operators. Why it matters to OSS: supports the student-design bias toward simple CNN ops and quantization-friendly activations.
- **AI Level of Detail: Distance-Aware ML Model Precision Selection for Real-Time Human Motion Prediction in Games** (arXiv:2606.06565, 2026-06-04; Mathew Varghese) — Applies game-style LOD to ML inference precision based on perceptual distance. Why it matters to OSS: candidate runtime scheduling principle for foveated/region-adaptive precision or splat budget selection.
- **Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers** (arXiv:2604.02344, 2026-02-09; Jędrzej Maczan) — Measures dispatch overhead across vendors/backends. Why it matters to OSS: dispatch overhead is likely fatal for many tiny kernels; supports fused kernels/persistent command buffers for WebGPU/Vulkan-style ports.

### Game Engine Integration
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694, 2026-03-08; Yuhang Wang, Hai Li, Shujuan Hou, Zhetao Dong, Xiaoyao Yang et al.) — Uses codec-side motion vectors/residuals/frame types to avoid expensive alignment and redundant work in online VSR. Why it matters to OSS: strong external validation that supplied motion metadata is valuable; OSS's exact engine MVs should be strictly better than codec MVs.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483, 2023-08-03; Antoine Mercier, Ruan Erasmus, Yashesh Savani, Manik Dhingra, Fatih Porikli et al.) — Game-specific neural supersampling dataset and method. Why it matters to OSS: relevant baseline/dataset reference for positioning OSS against existing game neural supersampling.

## Archive.org Findings
- Query `"gaussian splatting" super-resolution`: only broad GitHub-star-list archive items surfaced; no paper-specific GS-SR capture found.
- Query `"frame generation" "deep learning"`: found **MoCha-LD** OSF registration plus archived FluidFrames/DLSS Swapper builds; useful as historical/software artifacts, not OSS architecture inputs.
- Query `"real-time upscaling" game`: no results returned.
- Query `GS-STVSR OR GSASR OR GRAPE OR STSS`: results were noisy/unrelated (`grape` false positives, generic GitHub archive dumps). No verified archived GS-STVSR/GSASR/GRAPE/STSS paper/project capture this run.
- Follow-up: broad Archive.org full-text is low signal for acronym-heavy paper names; use direct Wayback CDX checks against known project/paper URLs next time.

## GitHub / Open Source
- **Adam-AtlasSoftware/RDNU — Radeon Decoupled Neural Upscaler** (`https://github.com/Adam-AtlasSoftware/RDNU`, pushed 2026-06-09, ★15, GPL-3.0) — RDNA3-focused neural upscaler using decoupled G-buffer guidance, pretrained on M3VIR/GameIR and fine-tuned on vKITTI2/Sintel. **Threat/opportunity:** closest open-source competitor signal; vendor-specific/RDNA-first and recurrent pixel-grid rather than unified Gaussian SR+FG, but useful for datasets, G-buffer contract, AMD deployment assumptions, and benchmark comparison.
- **hvcl/3DCT_SR_DDNM_NAB-GS** (`https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS`, pushed 2026-05-28, Apache-2.0) — Medical CT SR with diffusion projection priors and negative-alpha-blending GS. Low direct game relevance; mine only if NAB-GS raster/density tricks look transferable.
- **shepherdscientific/neural-spatial-upscaler** (`https://github.com/shepherdscientific/neural-spatial-upscaler`, pushed 2026-06-02) — FPGA/TMDS ternary CNN 240p→1080p upscaler. Not a direct game competitor, but an edge-hardware signal for tiny CNN/ternary kernels.
- **nova-studios-hw/novagpu-ts1t** (`https://github.com/nova-studios-hw/novagpu-ts1t`, pushed 2026-06-11, MIT, ★11) — Open GPU RTL mentioning hardware ray tracing/frame generation. Mostly hardware/RTL; watch only as a long-range open GPU ecosystem signal.
- GitHub searches for `DLSS FSR analysis`, `FSR 4 open source upscaler`, and `XeSS frame generation` returned no relevant new analysis/tool repos this run.
- Vendor/developer feeds: AMD GPUOpen currently has an RDNA4 WMMA guide series; relevant to future HIP/RDNA kernel work. NVIDIA DLSS feed returned no parseable RSS items from this environment. Intel community page was 403. Apple developer RSS shows Xcode 27 beta / platform releases, but no direct Metal upscaling item in the fetched slice.

## News
- **DLSS 4 / 4.5:** Google News RSS shows NVIDIA/press coverage around **DLSS 4.5 Ray Reconstruction** and >1000 RTX games/apps dated 2026-05-31/06-01, plus Helldivers 2 DLSS 4.5/FSR 4 update delay. Competitive signal: NVIDIA keeps improving transformer/ray-reconstruction quality; OSS positioning should emphasize unified SR+FG and open/vendor-neutral path rather than matching RR scope immediately.
- **FSR 4 / 4.1:** RSS shows AMD FSR 4.1 RDNA 3.5/iGPU support dispute and possible Multi Frame Generation expansion rumors dated 2026-06-04/06. Competitive signal: FSR support matrix uncertainty is an OSS opportunity for vendor-neutral messaging, but RDNA-targeted open projects like RDNU could crowd that narrative.
- **Intel Arc / XeSS:** RSS shows Intel Arc G-Series handheld push and **XeSS 3 Multi Frame Generation** coverage in Death Stranding 2 around 2026-05-31. Competitive signal: multi-frame generation is becoming table stakes across vendors; OSS's one-architecture SR+FG needs measured OSS-FX quality soon.
- **Apple Silicon / Metal:** WWDC/platform-state coverage and Cyberpunk 2077-on-Mac articles surfaced; no primary Metal upscaling API change verified. Watch Xcode 27 / Metal docs for shader/kernel features relevant to the planned Metal backend.
- **Game engine updates:** no recent Unreal/Unity upscaling-specific RSS hits returned.

## Action Items / Follow-ups
- [ ] **Fix/inspect trainer stall first**: dashboard says trainer offline/stuck at step 20,950 for 46h while publishing stack is healthy; no further research matters if v7 gating is blocked.
- [ ] **Deep-read RDNU** for architecture, license constraints, model size, runtime path, datasets, G-buffer contract, weights, and AMD-specific assumptions.
- [ ] **Deep-read Envision4D** for 4D-GS future extrapolation mechanics and ghosting mitigation; compare to OSS exact-MV native-time canvas.
- [ ] **Deep-read AutoMegaKernel / QuantSR+** for fused-kernel and quantized-SR deployment ideas applicable to the ≤1M-param student.
- [ ] **Ablate LEGS-style Laplacian/detail loss** in v7 teacher training if it can be added without changing runtime.
- [ ] **Investigate AI-LOD-style runtime scheduling** for splat budget / precision / foveated quality knobs.
- [ ] Retry Archive.org via direct Wayback CDX on known URLs for GS-STVSR, GSASR, GRAPE, STSS, 3DGS³ instead of broad search.
