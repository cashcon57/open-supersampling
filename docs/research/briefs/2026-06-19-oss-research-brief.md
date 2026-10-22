# OSS Research Brief — 2026-06-19

## Project Status Summary
- Latest commits: a97a180 feat: add tailnet-gated training controls to dashboard; then fcf8084 feat: tailnet-gated training control API + dashboard panel, 0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc, 56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock). Full checked set: `a97a180 feat: add tailnet-gated training controls to dashboard; fcf8084 feat: tailnet-gated training control API + dashboard panel; 0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc; 56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock); d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume; e1ab7fe hat.py: use flash-attn 2 directly when available, fallback to SDPA; e9b4b65 trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput); 26cd9da dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB; 83b4f07 v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast; baf4b02 worker: apex .com → .org 301 redirect`
- Open issues: #14: D3: measure v6.2-pico-002 VRAM footprint after v6.2 wiring and checkpoint exist; #13: H004: run Nsight L2 hit-rate profile on 3080 Ti CUDA host; #12: H004: implement packed low-rank canvas layout before claiming 32B/Gaussian; #10: v6.2.1 ablation: decoupled G-buffer encoder before concat-fusion; #9: v6.2.1 ablation: RepVGG/ECBSR/RepSR reparameterization for student backbone; #8: v6.2: cross-vendor kernel ports (HIP / Metal / Vulkan / SYCL); #7: v6.2.1: distillation pipeline with FSR 4 binary as teacher signal; #6: v6.2-pico-002: launch on 3080 Ti once codex T1-T5 land; #5: v6.2: No-ML shader variant (ships alongside v6.2); #4: v6.2: HAT-Tiny student distillation (≤1M params)
- README snapshot: v5-pixel-temporal (the in-distribution validation step before v6) measures **PSNR 25.703 / LPIPS 0.1666 / temporal ratio 0.337×** on TartanAir `oldtown` held-out (single dataset; no cross-dataset eval yet). v6 — the canvas + covariance + cross-attention + spawner architecture — is in training; v6.1 (active) adds randomized spawner offsets and feathered overlapping rasterizer tiles to eliminate a structural 16-pixel grid artifact diagnosed at v6-Pico-001 step-20K. OSS-FX (the α<1 extrapolation path) is one rasterizer call away from the v6 forward, since the canvas + motion field it depends on are already wired; integration is the next sprint. Cross-vendor kernels (CUDA/HIP/Metal/Level Zero/Vulkan) and the DLL-shim integration path are designed but not built. Solo maintainer, AI-augmented development. Looking for compute, hardware loaners, contract or full-time engineering work to take v6 
- Training dashboard: rendered `https://opensupersampling.com` redirected to `.org`. Visible state says **v7.0 Pico OSS-FX** / `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz @ step **26,000**, cost **$0.76**, GPU stats unavailable. Contradiction to track: header/card says "live" and watcher/R2/CF are healthy, but Trainer tooltip says **offline, stuck at step 30300 for 129h**.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **AIGS-Net: Compact Illumination Field Modeling via 2D Gaussian Splatting for Fast Low-Light Image Enhancement** (arXiv:2606.17998, 2026-06-16; Yuhan Chen, Kunyang Huang, Fuchen Li, Zhuohan Qin et al.) — Ultra-light 2D-GS illumination field for fast low-light enhancement. Why it matters to OSS: another 2D-GS-as-image-operator datapoint; opacity/compact Gaussian basis design may transfer to low-cost canvas correction layers. [https://arxiv.org/abs/2606.17998v1]
- **GSPan: A Continuous Gaussian Primitive Representation for Arbitrary-Scale Pansharpening** (arXiv:2606.17722, 2026-06-16; Fangyi Li, Xiaoyuan Yang, Yixiao Li, Zongyang Sui et al.) — Represents arbitrary-scale residual detail as continuous 2D Gaussian primitives. Why it matters to OSS: direct support for Gaussian residual/detail fields rather than pixel-grid heads; compare residual injection and raster costs. [https://arxiv.org/abs/2606.17722v1]
- **Progressively Deformable 2D Gaussian Splatting for Video Representation at Arbitrary Resolutions** (arXiv:2503.05600, 2025-03-07; Mufan Liu, Qi Yang, Miaoran Zhao, He Huang et al.) — Deformable 2D Gaussian video representation with progressive/arbitrary-resolution decoding. Why it matters to OSS: close representation-level precedent for persistent 2D Gaussian video/canvas state and progressive density ordering. [https://arxiv.org/abs/2503.05600v2]
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047, 2026-04-20; Mingyu Shi, Xin Di, Long Peng, Boxiang Cao et al.) — Continuous spatio-temporal VSR using 2D Gaussian splatting. Why it matters to OSS: remains the closest academic analogue for covariance-resampled 2D-GS SR plus time interpolation; already core v6/v7 prior. [https://arxiv.org/abs/2604.18047v1]

### Super-Resolution / Upscaling
- **DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution** (arXiv:2603.22271, 2026-03-23; Zhengyao Lv, Menghan Xia, Xintao Wang, Kwan-Yee K. Wong) — Dual-stream distillation framework for one-step video SR from expensive diffusion teachers. Why it matters to OSS: useful teacher→student training pattern for compressing high-quality temporal restoration into a cheap student. [https://arxiv.org/abs/2603.22271v1]
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (arXiv:2604.20291, 2026-04-22; Pham Phuong Nam Nguyen, Nam Tien Le, Thi Kim Trang Vo, Nhu Tinh Anh Nguyen) — Deployment-aware INT8 SISR with teacher-guided training and a compact reparameterizable graph. Why it matters to OSS: direct reference for ≤1M-param quantized student training and calibration. [https://arxiv.org/abs/2604.20291v1]
- **A Scale-Adaptive Framework for Joint Spatiotemporal Super-Resolution with Diffusion Models** (arXiv:2604.21903, 2026-04-23; Max Defez, Filippo Quarenghi, Mathieu Vrac, Stephan Mandt et al.) — Scale-adaptive joint spatiotemporal SR with diffusion models. Why it matters to OSS: not real-time, but useful teacher/pseudo-label reference for arbitrary spatial+temporal scale supervision. [https://arxiv.org/abs/2604.21903v1]
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483, 2023-08-03; Antoine Mercier, Ruan Erasmus, Yashesh Savani, Manik Dhingra et al.) — Game neural supersampling algorithm/dataset with motion vectors and auxiliary modalities. Why it matters to OSS: baseline dataset/paper for game-specific neural upscaling comparisons. [https://arxiv.org/abs/2308.01483v1]

### Frame Extrapolation / Generation
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567, 2026-05-11; Nengbo Lu, Bin Zhao) — Learns physically consistent Gaussian velocity fields from videos. Why it matters to OSS: relevant to native time-axis/cross-covariance losses and alpha-time extrapolation priors. [https://arxiv.org/abs/2605.10567v1]
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586, 2026-05-11; Nengbo Lu, Minghua Pan) — Uses Gaussian scene representations to learn physical causality and future trajectories. Why it matters to OSS: candidate conceptual reference for disocclusion/future-frame failure cases beyond exact motion-vector warps. [https://arxiv.org/abs/2605.10586v1]
- **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (arXiv:2406.18551, 2024-05-23; Songyin Wu, Deepak Vembar, Anton Sochenov, Selvakumar Panneer et al.) — G-buffer-free frame extrapolation for low-latency real-time rendering. Why it matters to OSS: competitive baseline for extrapolation latency/quality tradeoffs; OSS has stronger engine inputs but should compare against its no-G-buffer positioning. [https://arxiv.org/abs/2406.18551v1]

### Inference Optimization
- **ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering** (arXiv:2506.13814, 2025-06-14; Lufei Liu, Tor M. Aamodt) — Caches neural-network layers across temporally coherent real-time rendering frames. Why it matters to OSS: optimization idea for teacher/student inference if feature/canvas updates are coherent frame-to-frame. [https://arxiv.org/abs/2506.13814v1]
- **Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers** (arXiv:2604.02344, 2026-02-09; Jędrzej Maczan) — Measures WebGPU dispatch overhead across NVIDIA/AMD/Apple/Intel and browser/native backends. Why it matters to OSS: informs cross-vendor backend/kernel-fusion decisions, especially if WebGPU/Vulkan-like paths are considered. [https://arxiv.org/abs/2604.02344v1]

### Game / Motion-Vector Integration
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694, 2026-03-08; Yuhang Wang, Hai Li, Shujuan Hou, Zhetao Dong et al.) — Online VSR using codec motion vectors/residual maps to avoid expensive motion estimation. Why it matters to OSS: strong analogue for exploiting exact engine motion vectors and residual/gating signals. [https://arxiv.org/abs/2603.07694v1]

## Archive.org Findings
- `"gaussian splatting" "super-resolution"` — 3 hit(s), mostly broad Archive.org/GitHub/software mirrors: github.com-Awesome-Image-Registration-Organization-awesome-image-registration_-_2025-08-20_11-05-40; github.com-pbsds-starred_-_2025-10-03_04-36-09; github.com-pbsds-starred_-_2025-08-25_04-40-27. No confirmed archived GS-STVSR/GSASR/GRAPE/STSS paper copy found in this pass.
- `"frame generation" "deep learning"` — 4 hit(s), mostly broad Archive.org/GitHub/software mirrors: DLSS Swapper builds for Windows; MoCha-LD: Temporal-Chunked Latent Diffusion with Motion-Aware Consistency Regularization for Long Video Generation; Minecraft-Radiance - Radiance 2026-06-13. No confirmed archived GS-STVSR/GSASR/GRAPE/STSS paper copy found in this pass.
- `"real-time upscaling" game` — no useful hits returned.
- `GS-STVSR` — no useful hits returned.
- `GSASR` — 1 hit(s), mostly broad Archive.org/GitHub/software mirrors: CIA Reading Room cia-rdp80b01676r003500070011-7: LETTER TO (SANITIZED)FROM ALLEN W. DULLES. No confirmed archived GS-STVSR/GSASR/GRAPE/STSS paper copy found in this pass.
- `GRAPE "super-resolution"` — no useful hits returned.
- `STSS "super-resolution"` — no useful hits returned.

## GitHub / Open Source
- `pranav-asthana/SplatSuRe` (78★, pushed 2026-06-16, NOASSERTION) — Official code for SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting (CVPR 2026) https://github.com/pranav-asthana/SplatSuRe
- `hvcl/3DCT_SR_DDNM_NAB-GS` (1★, pushed 2026-06-12, Apache-2.0) — Zero-shot 3D CT Super-Resolution using Diffusion-based 2D Projection Priors and Negative Alpha Blending Gaussian Splatting (NAB-GS) https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS
- `rankcardinalbay/lossless-scaling-frame-generation` (0★, pushed 2026-06-18, NOASSERTION) — Lsfg Git | Supercharge your gaming setup with this advanced Lossless Scaling Frame Generation utility. Smooth out stuttering, unlock ultra-high framerates, and enjoy flawless, AI-driven motion interpolation on any PC. The ultimate open-source tool for fluid gameplay. https://github.com/rankcardinalbay/lossless-scaling-frame-generation
- `AdvaitPalve11/Jojo-Player` (0★, pushed 2026-06-18, MIT) — A modern open-source video player built in Java with FFmpeg, resolution upscaling, frame generation, and hardware-accelerated playback. https://github.com/AdvaitPalve11/Jojo-Player
- `ShrekDino/FreeGen` (0★, pushed 2026-06-16, Apache-2.0) — Open-source upscaling and frame generation — Vulkan, FSR 3.1. https://github.com/ShrekDino/FreeGen
- `gokusayayin641-ai/neural-upscaler-docs-page` (0★, pushed 2026-06-10, None) — Neural Upscaler X4 documentation https://github.com/gokusayayin641-ai/neural-upscaler-docs-page
- `Adam-AtlasSoftware/RDNU` (16★, pushed 2026-06-09, GPL-3.0) — Radeon Decoupled Neural Upscaler - RDNA3-focused ML-based uUpscaler using sunny2109/RDG (Decoupled G-buffer Guidance). Pretrained on guluthemonster/M3VIR and LLLebin/GameIR, fine-tuned on vKITTI2 and MPI-Sintel. https://github.com/Adam-AtlasSoftware/RDNU
- `dlss_fsr_analysis` — no repos found by GitHub REST search for the last ~30d query.

Notes: `pranav-asthana/SplatSuRe` is the strongest GS-SR code lead in this run (CVPR 2026, 78★). `Adam-AtlasSoftware/RDNU` remains the most OSS-adjacent open neural-upscaler competitor lead, but it was already tracked in the bibliography from prior runs. `ShrekDino/FreeGen` claims Vulkan + FSR 3.1 open upscaling/frame-gen; low signal (0★) but worth a quick clone/read if it grows.

## News
- Tue, 09 Jun 2026 — CONTROL Resonant Launching September 24th With DLSS 4.5 & Path Tracing, Plus New DLSS Games Released - NVIDIA (Google News RSS lead; source link via RSS).
- Wed, 17 Jun 2026 — Apple's Game Porting Toolkit 4 is a big improvement for modern game coders - AppleInsider (Google News RSS lead; source link via RSS).
- Tue, 16 Jun 2026 — Zenless Zone Zero Version 3.0 Launches June 17th, Adding DLSS 4.5 & Ray Tracing - NVIDIA (Google News RSS lead; source link via RSS).
- Mon, 15 Jun 2026 — AMD Explains FSR 4.1 Upscaling for RDNA 3, Promises Quality Parity with RDNA 4 - TechPowerUp (Google News RSS lead; source link via RSS).
- Sat, 13 Jun 2026 — AMD FSR 4 technology support discovered in The Outlast Trials - GameGPU (Google News RSS lead; source link via RSS).
- Sat, 06 Jun 2026 — AMD may implement FSR 4 Multi Frame Generation for RDNA 3 and RDNA 4 - GameGPU (Google News RSS lead; source link via RSS).
- Sat, 06 Jun 2026 — Intel Arc G3 Extreme Performance Benchmarks Show Clear Disruption In The Handled Segment, Offers Double The Battery At Same Performance As Z2 Extreme - Wccftech (Google News RSS lead; source link via RSS).
- Thu, 11 Jun 2026 — Intel Arc GPU Graphics Drivers 101.8826 WHQL Released - TechPowerUp (Google News RSS lead; source link via RSS).
- Mon, 08 Jun 2026 — Update CPU Z 2.20.2 adds support for Intel Arc G3 chips Extreme - GameGPU (Google News RSS lead; source link via RSS).
- Mon, 08 Jun 2026 — Apple aids app development with new intelligence frameworks and advanced tools - Apple (Google News RSS lead; source link via RSS).
- Tue, 09 Jun 2026 — Apple Outlines Major AI and Developer Tool Updates at 2026 Platforms State of the Union - MacRumors (Google News RSS lead; source link via RSS).
- Wed, 17 Jun 2026 — Apple's Game Porting Toolkit 4 is a big improvement for modern game coders - AppleInsider (Google News RSS lead; source link via RSS).
- Tue, 16 Jun 2026 — Upcoming Hardware Launches 2026 (Updated Jun 2026) - TechPowerUp (Google News RSS lead; source link via RSS).
- Mon, 08 Jun 2026 — Nvidia Is Doubling Down on the CPU Market. That's Bad News for AMD and Intel Stock Investors - The Motley Fool (Google News RSS lead; source link via RSS).
- Tue, 09 Jun 2026 — Best gaming GPU deals 2026 — deals on Nvidia, AMD, and Intel gaming graphics cards - Tom's Hardware (Google News RSS lead; source link via RSS).
- `Unreal Engine Unity upscaling update when:14d` — RSS error/no results: []

### Vendor / primary-page checks
- nvidia_dlss: fetched https://www.nvidia.com/en-us/geforce/news/; title scrape mostly returned page/feed metadata (GeForce Articles, Guides, Gaming News, Featured Stories | NVIDIA, NVIDIA Home, Menu, Menu).
- amd_gpuopen: fetched https://gpuopen.com/news/; title scrape mostly returned page/feed metadata (News - AMD GPUOpen, News - AMD GPUOpen).
- intel_xess: direct fetch failed (<HTTPError 403: 'Forbidden'>); relied on news/RSS leads where available.
- unreal: direct fetch failed (<HTTPError 403: 'Forbidden'>); relied on news/RSS leads where available.
- unity: direct fetch failed (<HTTPError 404: 'Not Found'>); relied on news/RSS leads where available.
- apple_metal: fetched https://developer.apple.com/news/rss/news.rss; title scrape mostly returned page/feed metadata (Latest News - Apple Developer, Updated Apple Developer Program License Agreement now available, Changes to iOS in Brazil, New domain for Sign in with Apple and iCloud+ Hide My Email).

## Action Items / Follow-ups
- [ ] Ops: investigate dashboard trainer state — public page shows "live" but Trainer is offline/stuck at step 30,300 for 129h and GPU stats are unavailable.
- [ ] Deep read: `AIGS-Net` and `GSPan` for compact/adaptive 2D-GS image operators and residual-detail representations.
- [ ] Deep read / compare: `D2GV-AR` progressive deformable 2D Gaussian video representation vs OSS v7 persistent canvas and parent/child density ordering.
- [ ] Student path: revisit `Efficient INT8 SISR` + `DUO-VSR` for deployment-aware quantization and one-step temporal distillation recipes.
- [ ] Competitive watch: `SplatSuRe` repo and `3DGS^3`/GS-STVSR line for converging GS-based SR+FI claims.
- [ ] Backend watch: WebGPU dispatch-overhead paper and ReFrame layer caching for cross-vendor/kernel-fusion roadmap.
- [ ] Market watch: FSR 4.1/RDNA3 parity headlines and DLSS 4.5 game rollouts; if confirmed by primary vendor posts, update competitive-positioning docs.

_Source scan artifact: `/tmp/oss-research-scan-2026-06-19.json` (temporary, not committed)._
