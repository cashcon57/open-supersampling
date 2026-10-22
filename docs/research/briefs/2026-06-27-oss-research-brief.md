# OSS Research Brief — 2026-06-27

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
- Open issues (first 10 from GitHub API):
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
- README snapshot: project remains pre-alpha; current architecture is v7 / `srcnn-v7.0-pico-005`, an N-D Gaussian canvas with native time axis, HAT-Tiny teacher, parent-child loss-adaptive density, and Mip-Splatting filters. Shipping ≤1M-param student is still not trained; cross-vendor kernels and DLL-shim runtime remain designed-not-built.
- Training dashboard: rendered dashboard says `v7.0 Pico OSS-FX is training right now`, active run `srcnn-v7.0-pico-005`, **30,300 / 100,000** steps, loss **0.18269**, viz at **26,000**, cost **$0.76**, **12.63 GPU-hours**. Service health tooltip says **Trainer — offline, stuck at step 30300 for 4h**, while Watcher is healthy/published 0s ago, CF Worker healthy (HTTP 200 in 88ms), and R2 origin healthy (HTTP 200). GPU stats unavailable / host offline or rsync failed. Compared with yesterday's brief, the stuck-duration counter reset from 47h to 4h without step progress; treat as dashboard/metrics-state inconsistency, not confirmed training recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047; Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha; 2026-04-20) — Reconfirmed as the central external prior: 2D-GS continuous spatial+temporal SR with covariance-resampling alignment and nearly constant inference time across temporal scales. Why it matters to OSS: this is the closest published analogue to the v7 Gaussian canvas SR+FX thesis; keep it as primary baseline, especially covariance drift handling.
- **Progressively Deformable 2D Gaussian Splatting for Video Representation at Arbitrary Resolutions** (arXiv:2503.05600; Mufan Liu, Qi Yang, Miaoran Zhao; 2025-03-07) — Deformable 2D-GS video representation at arbitrary resolutions. Why it matters to OSS: supports persistent 2D Gaussian video/canvas state and progressive density ordering; already in bibliography but still a relevant architecture check.
- **Temporally Aware Densification for Dynamic 3D Gaussian Splatting** (arXiv:2606.23212; Vikram Sandu, Mayurdeep Pathak, Rajiv Soundararajan; 2026-06-22) — Temporal densification for dynamic Gaussian scenes. Why it matters to OSS: direct comparison target for v7 parent-child/loss-adaptive density and pruning heuristics; already captured yesterday.

### Super-Resolution / Upscaling
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (arXiv:2604.20291; Pham Phuong Nam Nguyen, Nam Tien Le, Thi Kim Trang Vo, Nhu Tinh Anh Nguyen; 2026-04-22) — Compact extract-refine-upsample student, low-resolution compute, DCT supervision, confidence-weighted teacher distillation, and QAT on the fused deploy graph. Why it matters to OSS: directly maps to the ≤1M-param INT8/FP8 student path; already in bibliography, but this is still the cleanest deploy-aware student-training reference.
- **Neural Super-Resolution for Real-time Rendering with Radiance Demodulation** (arXiv:2308.06699; Jia Li, Ziling Chen, Xiaolong Wu, Lu Wang, Beibei Wang, Lei Zhang; 2023/2024) — Real-time rendering SR splits lighting/material, uses occlusion-aware warping and recurrent temporal loss. Why it matters to OSS: G-buffer/material decomposition may reduce texture blur/ghosting in the teacher or student; already tracked.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483; Antoine Mercier, Ruan Erasmus, Yashesh Savani, Manik Dhingra, Fatih Porikli, Guillaume Berger; 2023-08-03) — Game-content neural supersampling plus motion vectors/depth/jitter/mipmap-bias modalities. Why it matters to OSS: still the key public gaming-content dataset/baseline for cross-engine claims; already tracked.

### Frame Extrapolation / Generation
- **Neural Image Space Tessellation effect** (arXiv:2602.23754v2; Youyang Du, Junqiu Zhu, Zheng Zeng, Lu Wang, Lingqi Yan; 2026-05-27) — Screen-space neural post-process that uses depth + geometric/shading normals to smooth low-poly silhouettes without changing geometry. Why it matters to OSS: not SR/FG directly, but it is a real-time neural G-buffer post-process with near-constant frame overhead; useful for OSS capture/input-channel design and edge/silhouette artifact handling. Newly material, not previously in bibliography.
- **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (arXiv:2406.18551; Songyin Wu, Deepak Vembar, Anton Sochenov, Selvakumar Panneer; 2024-05-23) — Low-latency frame extrapolation baseline. Why it matters to OSS: competitive positioning foil; OSS should contrast exact engine motion vectors + persistent Gaussian canvas against standalone G-buffer-free extrapolation. Already tracked.
- **UniRED: Unified RGB-D Video Frame Interpolation with Event Guidance** (arXiv:2606.24282; Yinuo Zhang, Guangshun Wei, Yuanfeng Zhou et al.; 2026-06-23) — RGB-D/event-guided interpolation. Why it matters to OSS: artifact/conditioning reference for OSS-FX, though game inputs differ; already captured yesterday.

### Inference Optimization
- **Realizing Native INT8 Compute for Diffusion Transformers on Consumer GPUs: A Fused INT8 GEMM Kernel for Ideogram 4.0** (arXiv:2606.14598; Ali Asaria, Tony Salomone, Deep Gandhi; 2026-06-12) — Fused INT8 GEMM on consumer GPUs. Why it matters to OSS: reinforces that nominal INT8/FP8 export is insufficient without real fused kernels; already captured yesterday.
- **ANEForge: Python for direct computation on the Apple Neural Engine** (arXiv:2606.17090; Spencer H. Bryngelson; 2026-06-12) — Direct ANE programming path. Why it matters to OSS: potential Apple/Metal/ANE student-inference route beyond CoreML; already captured yesterday.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 broad Archive.org hits, all GitHub-list/archive aggregator snapshots; no paper takedown/update signal.
- `"frame generation" "deep learning"`: 4 broad hits including FluidFrames/DLSS Swapper builds and an OSF registration; no material OSS paper update signal.
- `"real-time upscaling" game`: 0 broad Archive.org hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16,593 hits, dominated by unrelated `grape` media/items; broad acronym query is too noisy.
- Wayback CDX: GSASR GitHub has confirmed snapshots (2025-03-13, 2025-08-01). GRAPE GitHub URL checked returned no snapshots for the guessed repo path. GS-STVSR arXiv CDX request timed out this run; retry targeted arXiv CDX later.

## GitHub / Open Source
- **optiscaler/OptiScaler** (GPL-3.0, ~9k stars, latest commit shown 2026-06-27, latest release v0.9.3 on 2026-06-18) — Mature middleware that intercepts DLSS2+/FSR2+/XeSS inputs and redirects to chosen upscaling/FG outputs; includes FSR4, XeFG/FSR-FG paths, Fakenvapi, Nukem DLSSG-to-FSR3 integration, and OptiFG. Threat/positioning: it owns the practical DLL-shim/user-mod integration surface OSS eventually targets, even though it swaps existing vendors rather than shipping a new OSS model.
- **PancakeTAS/lsfg-vk** (~4.5k stars, updated 2026-06-27) — Lossless Scaling Frame Generation on Linux via Vulkan. Opportunity/threat: Linux/Steam Deck FG demand is active; OSS should watch for Vulkan presentation/hook lessons.
- **4BH15H3K-SB/optiscaler-fsr4-render-harness** (~69 stars, updated 2026-06-28 per GitHub topic page) — Guide/harness around OptiScaler + FSR4 upscaling/frame-gen. Likely SEO/noisy, but it shows FSR4 modding content is attracting attention.
- **ind4skylivey/0ptiscaler4linux** (~73 stars) and **Spexxl/OptiTux-GUI** (~44 stars) — Linux OptiScaler automation/GUI ecosystem. Opportunity: integration tooling expectations are forming around per-game profiles, rollback, and GPU-specific presets.
- **GPUOpen-LibrariesAndSDKs/FidelityFX-SDK** — FSR SDK v2.3.0 (`Redstone`) released 2026-06-24 with FSR Upscaling ML-Upscaler 4.1.1, FSR Frame Generation ML 4.0.1, Ray Regeneration 1.2.0, and DirectX 12/Vulkan samples. Threat/opportunity: AMD is moving ML upscaling/FG into a public SDK cadence; OSS needs benchmark positioning and can mine sample integration patterns.

## News
- NVIDIA primary page/news: DLSS 4.5 messaging now emphasizes 2nd-generation transformer Super Resolution, Dynamic Multi Frame Generation, 6X MFG, enhanced Frame Generation, and Ray Reconstruction launching August 2026 for all RTX GPUs. NVIDIA says Ray Reconstruction unifies denoising and Super Resolution using temporal/spatial engine data. Competitive pressure: DLSS keeps expanding from SR/FG into denoise/reconstruction; OSS's “single canvas for SR+FG, later RR” pitch remains credible but must show measured quality soon.
- NVIDIA DLSS technology page lists DLSS Super Resolution as using multiple lower-res samples, motion data, and previous-frame feedback; this reinforces OSS’s exact-engine-MV + persistent-canvas framing as a comparable but vendor-neutral design surface.
- AMD GPUOpen/FidelityFX SDK: v2.3.0 / Redstone release on 2026-06-24 with ML upscaler, ML frame generation, Ray Regeneration, and radiance caching preview in one SDK. Competitive pressure: AMD now has a coherent public ML reconstruction stack spanning SR+FG+denoise.
- Google/news search led to RSS/headline-level chatter about broader DLSS/FSR/XeSS comparisons and OptiScaler/FSR4 usage. Treat as leads only; primary-source-confirmed items above are NVIDIA + AMD pages/repos.

## Action Items / Follow-ups
- [ ] Ops: investigate `srcnn-v7.0-pico-005` stuck at step 30,300; dashboard stuck-duration reset from 47h yesterday to 4h today without step progress.
- [ ] Deep-read GS-STVSR covariance-resampling alignment vs current v7 covariance drift handling.
- [ ] Add a lightweight experiment note for NIST-style G-buffer/normal-guided silhouette/detail cleanup as a possible artifact head or loss term.
- [ ] Track OptiScaler/lsfg-vk as integration competitors; ensure OSS DLL-shim plan covers per-game profiles, rollback, anti-cheat warnings, and Linux/Proton path.
- [ ] Inspect AMD FSR SDK 2.3.0 samples for ML upscaler/FG API shape, motion-vector contracts, and Vulkan/DX12 sample integration patterns.
- [ ] Retry targeted Wayback CDX for GS-STVSR arXiv and exact GRAPE repo/project URLs; broad Archive.org acronym search remains unusable.
