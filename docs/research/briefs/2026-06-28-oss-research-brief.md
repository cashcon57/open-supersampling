# OSS Research Brief — 2026-06-28

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
- README snapshot: repo still positions OSS as pre-alpha v7 / `srcnn-v7.0-pico-005`: N-D Gaussian canvas with native time axis, HAT-Tiny teacher, parent-child loss-adaptive density, Mip-Splatting filters, and planned ≤1M-param distilled student. Student, cross-vendor kernels, and DLL-shim runtime remain not shipped.
- Training dashboard: rendered `opensupersampling.com` redirects to `.org` and shows `v7.0 Pico OSS-FX` at **30,300 / 100,000** steps, loss **0.18269**, viz @ **26,000**, cost **$0.76**, **12.63 GPU-hours**. Contradiction: headline says live/updated just now, but service tooltip says **Trainer — offline, stuck at step 30300 for 21h**; Watcher healthy (published 1s ago), CF Worker healthy (HTTP 200 in 88ms), R2 origin healthy (HTTP 200), GPU stats unavailable / host offline or rsync failed. Compared with 2026-06-27, step/loss did not advance and stuck-duration changed from 4h to 21h; treat as ongoing trainer outage, not recovered training.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Lift4D: Harmonizing Single-View 3D Estimation for 4D Reconstruction In-the-Wild** (arXiv:2606.23688; Yehonathan Litman, Xiaoxuan Ma, Manan Shah et al.; 2026-06-22) — Test-time optimization initializes deformable 3DGS from temporally consistent single-view 3D predictions and sculpts it with occlusion-aware video evidence. Why it matters to OSS: occlusion-aware deformable 3DGS with temporally consistent priors remains relevant to v7 disocclusion and persistent-canvas recovery; already tracked in bibliography, not newly appended.
- Recent broad GS query was otherwise dominated by MRI/robotics/sign-language GS papers; no new close GS-STVSR/OSS-FX analogue beyond already-tracked Lift4D / temporally-aware densification class.

### Super-Resolution / Upscaling
- **Neural Texture Compression using Hypernetworks** (arXiv:2606.26913; Belcour Laurent; 2026-06-25) — Trains a hypernetwork to emit latent texture features plus MLP decoder weights, extending neural texture compression to decoders that can learn super-resolution. Why it matters to OSS: real-time shader-side hypernetwork/latent texture decode with optional learned SR decoders is adjacent to OSS material/G-buffer-aware reconstruction and may inspire ultra-small per-material or per-surface student heads.
- **teasr: training-efficient any-step diffusion transformer for real-world image super-resolution** (arXiv:2606.16188; Xiang Gao, Chenxin Zhu, Yushun Fang et al.; 2026-06-15) — Any-step diffusion SR via self-adversarial distillation inside one model, removing auxiliary teacher/discriminator memory overhead. Why it matters to OSS: already tracked as a possible high-quality offline teacher/pseudo-target source; still too diffusion-heavy for shipping but useful for any-step/self-distillation training ideas.
- **Ultra Flash: Scaling Real-Time Streaming Video Generation to High Resolutions** (arXiv:2606.09150;  Luxury, Jie Huang, Zihao Fan et al.; 2026-06-08) — Cascaded streaming high-resolution video generation reports ~30 FPS at 1K and ~18 FPS at 2K on one GPU. Why it matters to OSS: already in bibliography; its causal latent upsampler, sparse causalization, single-step distillation, and dynamic cache management reinforce the importance of cache-aware temporal upsampling.

### Frame Extrapolation / Generation
- **TIDES: Time-Derivative Event Simulation via Deformable Reconstruction** (arXiv:2606.02058; Christopher Thirgood, Dipon Kumar Ghosh, Simon Hadfield; 2026-06-01) — Continuous-time event simulator built on dynamic Gaussian splatting derives per-pixel intensity dynamics directly from the scene rather than differencing rendered frames. Why it matters to OSS: already tracked; dynamic-GS continuous-time intensity derivatives are a strong math reference for OSS-FX alpha-time supervision and avoiding discrete-frame timestamp batching.
- **SKIP: Sparse Keyframe Interpolation Paradigm for Efficient Embodied World Models** (arXiv:2606.00664; Ziheng He, Yixiang Chen, Ning Yang et al.; 2026-05-30) — Generates sparse task-relevant keyframes, then reconstructs missing intervals with a gap predictor and action-conditioned interpolator for 4.16× faster dense rollouts. Why it matters to OSS: sparse-keyframe then learned interpolation is a competing efficiency pattern; OSS should be able to argue its exact-MV Gaussian canvas avoids a separate interpolator while retaining event/task-relevant transitions.

### Inference Optimization
- **Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation** (arXiv:2606.23743; Yitong Li, Junsong Chen, Haopeng Li et al.; 2026-06-21) — Agentic acceleration stack composes cache, sparse attention, token pruning, quantization, and kernel fusion for video diffusion inference, reaching >2× end-to-end speedups. Why it matters to OSS: the five-technique stack maps directly onto OSS student/raster/composite deployment work: cache, sparse attention, token/primitive pruning, quantization, and kernel fusion should be co-tuned for target GPU rather than optimized independently.
- **SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference** (arXiv:2606.26587; Haoqian Meng, Yilun Luo, Yafei Zhao et al.; 2026-06-25) — Training-free inference method splits activations into sparse FP4 backbone plus dense residual compensation with fused preparation kernels. Why it matters to OSS: although LLM-focused, the sparse-dense FP4 decomposition plus fused prep kernel is relevant to future FP4/FP8 student kernels and warns that activation outliers need a separate path.

### Game Engine / Motion-Vector Integration
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694; Yuhang Wang, Hai Li, Shujuan Hou et al.; 2026-03-08) — Online VSR uses codec motion vectors, residual maps, and frame types to reduce alignment overhead and more than double inference speed vs a strong baseline. Why it matters to OSS: already tracked and directly relevant: compressed-domain MVs/residuals/frame types are analogous to engine MVs/G-buffers for reducing alignment cost and gating compute.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483; Antoine Mercier, Ruan Erasmus, Yashesh Savani et al.; 2023-08-03) — Game supersampling dataset and method using rendering auxiliary modalities such as motion vectors and depth. Why it matters to OSS: already tracked as the public gaming supersampling dataset/baseline; remains the key external citation for motion vectors/depth/jitter/mipmap-bias modalities in rendered content.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 broad hits, all GitHub/archive-list style snapshots; no takedown/update signal.
- `"frame generation" "deep learning"`: 4 broad hits including DLSS Swapper / FluidFrames builds and OSF registration noise; no material OSS paper update.
- `"real-time upscaling" game`: 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: broad query remains unusable (16,594 hits, dominated by unrelated `grape` items).
- Wayback CDX confirmed archived arXiv snapshots for `2403.17888` and `2406.04358`; CDX timed out for guessed `2504.16423` and `2503.18498` URLs this run.

## GitHub / Open Source
### Gaussian splatting + super-resolution
- **Azusa309/MedGSSR** (1★, pushed 2026-06-22, license MIT) — MedGSSR: Generalizable Medical Image Super-Resolution 3D Reconstruction via Hierarchical Feed-forward Gaussian Splatting (ECCV 2026) https://github.com/Azusa309/MedGSSR
- **pranav-asthana/SplatSuRe** (81★, pushed 2026-06-16, license NOASSERTION) — Official code for SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting (CVPR 2026) https://github.com/pranav-asthana/SplatSuRe
  - Relevance: **SplatSuRe** is already in bibliography and remains the more material repo (selective 3DGS SR, 81★, updated 2026-06-16). **MedGSSR** is medical-domain but worth a skim for feed-forward hierarchical GS-SR implementation patterns.
### Frame generation + open source
- **AdvaitPalve11/Jojo-Player** (0★, pushed 2026-06-18, license MIT) — A modern open-source video player built in Java with FFmpeg, resolution upscaling, frame generation, and hardware-accelerated playback. https://github.com/AdvaitPalve11/Jojo-Player
- **ShrekDino/FreeGen** (0★, pushed 2026-06-16, license Apache-2.0) — Open-source upscaling and frame generation — Vulkan, FSR 3.1. https://github.com/ShrekDino/FreeGen
- **nova-studios-hw/novagpu-ts1t** (12★, pushed 2026-06-14, license MIT) — N.E.O.N. Architecture — Open source GPU RTL in Verilog. Token dataflow, hardware ray tracing, frame generation. First silicon target: 28nm. https://github.com/nova-studios-hw/novagpu-ts1t
  - Relevance: `FreeGen` claims Vulkan + FSR 3.1 open-source upscaling/FG but has 0★; monitor only if code/weights become real. `Jojo-Player` is app-level video playback, not OSS architecture competition.
### Neural upscaler / analysis tools
- **gokusayayin641-ai/neural-upscaler-x4-page** (0★, pushed 2026-06-26, license unknown) — Neural Upscaler X4 - AI Image Upscaling for Blender https://github.com/gokusayayin641-ai/neural-upscaler-x4-page
- No new DLSS/FSR/XeSS analysis repos surfaced via GitHub Search API this run. Prior watch targets from 2026-06-27 (OptiScaler, lsfg-vk, FidelityFX-SDK Redstone) remain more material than today's new repos.

## News
- **NVIDIA / DLSS**: Google News RSS includes NVIDIA's Zenless Zone Zero Version 3.0 item adding `DLSS 4.5 & Ray Tracing` and multiple DLSS 4/4.5 comparison headlines. Vendor pages were reachable and still heavily mention DLSS/frame-generation, but this run did not identify a new primary DLSS SDK/architecture release beyond yesterday's DLSS 4.5 competitive pressure.
- **AMD / FSR**: RSS headlines report FSR 4.1/4.1.1 INT8 support expanding to RX 7000 / RDNA 3-class hardware and Valve testing FSR 4 for Steam Machine. Treat as headline-level until primary AMD docs are checked; strategic implication is clear: AMD is pushing ML upscaling down-stack, directly pressuring OSS's cross-vendor/student positioning.
- **Intel / XeSS**: RSS was mostly Arc G3 / MSI Claw handheld coverage plus noisy comparison headlines; no primary XeSS technical update was confirmed. Intel docs page returned HTTP 403.
- **Apple / Metal**: RSS mentions Game Porting Toolkit 4 improvement and M4 GPU benchmark/spec pieces; no OSS-specific Metal upscaling update confirmed.
- **Game engines**: Unreal primary page returned HTTP 403; RSS results were mostly game PC specs/reviews rather than engine upscaling SDK news. Unity blog reachable but no DLSS/FSR/XeSS/upscaling mentions found in fetched page.
- Source caveat: several Google News results are SEO/noisy mirrors; primary-source-confirmed signals this run are limited to reachable NVIDIA/AMD/Unity pages plus GitHub/repo APIs.

## Action Items / Follow-ups
- [ ] Ops: investigate `srcnn-v7.0-pico-005` still stuck at step 30,300; dashboard now reports trainer offline/stuck 21h with watcher/origin healthy.
- [ ] Deep-read **Neural Texture Compression using Hypernetworks** for tiny decoder/hypernetwork patterns that could map to material-aware OSS student heads.
- [ ] Deep-read **Sol Video Inference Engine** for a deployment checklist: cache, sparse attention, primitive/token pruning, quantization, kernel fusion, and quality validation as one co-optimized stack.
- [ ] Skim **SharQ** for outlier-aware sparse+dense low-bit activation handling before committing to FP8/FP4 student kernels.
- [ ] Verify primary AMD FSR 4.1/4.1.1 docs/release notes; RSS suggests INT8 support expansion to RDNA 3/RX 7000 and Steam Machine relevance.
- [ ] Monitor `pranav-asthana/SplatSuRe`, `Azusa309/MedGSSR`, and `ShrekDino/FreeGen`; only SplatSuRe looks material today.
