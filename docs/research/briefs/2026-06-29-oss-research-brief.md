# OSS Research Brief — 2026-06-29

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
- README snapshot: still pre-alpha, with v7 / `srcnn-v7.0-pico-005` described as the active research-teacher path: N-D Gaussian canvas with native time axis, HAT-Tiny teacher, parent-child loss-adaptive density, Mip-Splatting filters, alpha-time OSS-FX supervision, and planned ≤1M-param distilled students. Student, cross-vendor kernels, and DLL-shim runtime remain not shipped.
- Training dashboard: rendered `opensupersampling.com` redirects to `.org` and shows **v7.0 Pico OSS-FX** at **30,300 / 100,000** steps, loss **0.18269**, viz @ **26,000**, cost **$0.76**, **12.63 GPU-hours**, **5.05 kWh**. Dashboard contradiction persists: headline says `live` / `updated just now`, but service tooltip says **Trainer — offline, stuck at step 30300 for 45h**. Watcher healthy (published 1s ago), CF Worker healthy (HTTP 200 in 94ms), R2 origin healthy (HTTP 200), GPU stats unavailable / host offline or rsync failed. Compared with 2026-06-28, step/loss/viz did not advance and stuck-duration increased from 21h to 45h.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Lift4D: Harmonizing Single-View 3D Estimation for 4D Reconstruction In-the-Wild** (arXiv:2606.23688; Yehonathan Litman, Xiaoxuan Ma, Manan Shah, Nicolas Ugrinovic et al.; 2026-06-22) — Test-time optimization initializes deformable 3DGS from temporally consistent single-view 3D predictions and refines with occlusion-aware video evidence. Why it matters to OSS: still the strongest recent disocclusion/persistent-canvas recovery reference surfaced by the GS temporal scans; already tracked in bibliography.
- **Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution** (arXiv:2605.22147; Jiangwei Mo, Xi Lu, Hanlin Wu; 2026-05-21) — Uses 2D Gaussian splatting as a continuous-scale remote-sensing SR representation. Why it matters to OSS: reinforces 2D-GS-as-continuous-image-operator, but already tracked and non-game-domain.
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047; Mingyu Shi, Xin Di, Long Peng, Boxiang Cao; 2026-04-20) — Continuous space-time VSR using 2D Gaussian splatting and covariance-style resampling. Why it matters to OSS: still the primary external math precedent for covariance-resampled temporal SR; no new version or archive signal found today.
- Other fresh GS hits were MRI novel-view rendering, text-to-3D scene generation, RGB-D digital twins, sign-language data, and thermomechanical scene editing. No new close OSS-FX / GS-STVSR analogue beyond already-tracked items.

### Super-Resolution / Upscaling
- **Neural Texture Compression using Hypernetworks** (arXiv:2606.26913; Laurent Belcour; 2026-06-25) — Hypernetwork emits latent texture features plus decoder weights, and the method explicitly supports real-time neural texture decoding with learned super-resolution decoders. Why it matters to OSS: already appended 2026-06-28; still worth a deep read for material/G-buffer-aware tiny student heads.
- **Fast Image Super-Resolution via Consistency Rectified Flow** (arXiv:2605.12377; Jiaqi Xu, Wenbo Li, Haoze Sun, Fan Li; 2026-05-12, v2) — Few/one-step flow-style SR target from the diffusion family. Why it matters to OSS: possible offline teacher/pseudo-target reference, but not a shipping runtime path and less material than the already-tracked one-step/distillation SR papers.
- **SlimDiffSR: Toward Lightweight and Efficient Remote Sensing Image Super-Resolution via Diffusion Model Distillation** (arXiv:2605.02198; Ce Wang, Zhenyu Hu, Wanjie Sun; 2026-05-04, v2) — Distills diffusion SR into a lighter remote-sensing model. Why it matters to OSS: already tracked as a teacher→student compression reference; non-game domain.

### Frame Extrapolation / Generation
- **Fewer, Better Frames: A Compute-Normalized Proof of Concept for Coherence-First World-Model Rendering with Model-Guided FSR4 Frame Generation** (arXiv:2606.02586; Paweł Katarzyński; 2026-05-11) — Argues that lower native cadence plus model-guided FSR4 frame generation can trade nominal frame rate for longer-horizon scene coherence. Why it matters to OSS: competitive-positioning lead, not a direct architecture paper; useful because it frames FSR4 FG as part of world-model rendering and gives OSS a benchmark narrative to counter with exact-MV Gaussian canvas extrapolation. Newly appended to bibliography.
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586; Nengbo Lu, Minghua Pan; 2026-05-11) — Gaussian-state future prediction with physical causality priors. Why it matters to OSS: already tracked as a longer-horizon OSS-FX regularization reference.
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567; Nengbo Lu, Bin Zhao; 2026-05-11) — Learns per-Gaussian velocity fields for dynamic scenes. Why it matters to OSS: already tracked; overlaps v7 native time / cross-covariance motion representation.
- **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (arXiv:2406.18551; Songyin Wu, Deepak Vembar, Anton Sochenov, Selvakumar Panneer; 2024-05-23) — Low-latency neural frame extrapolation without G-buffers. Why it matters to OSS: already tracked as a direct OSS-FX baseline/competitor; OSS should contrast exact engine MVs + persistent canvas vs standalone extrapolator.

### Inference Optimization
- **Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation** (arXiv:2606.23743; Yitong Li, Junsong Chen, Haopeng Li et al.; 2026-06-21) — Combines cache, sparse attention, token pruning, quantization, and kernel fusion for >2× video diffusion inference speedup. Why it matters to OSS: already appended 2026-06-28; use as a deployment checklist for co-tuning student/raster/composite inference rather than optimizing each layer in isolation.
- **SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference** (arXiv:2606.26587; Haoqian Meng, Yilun Luo, Yafei Zhao et al.; 2026-06-25) — Sparse FP4 backbone plus dense residual compensation with fused prep kernels. Why it matters to OSS: already appended 2026-06-28; relevant to future FP8/FP4 student kernels and activation-outlier handling.
- **Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers** (arXiv:2604.02344; Jędrzej Maczan; 2026-02-09) — Measures dispatch overhead across WebGPU backends. Why it matters to OSS: already tracked; reinforces that cross-vendor inference needs fused dispatch/megakernel thinking, not just portable shader correctness.

### Game Engine / Motion-Vector Integration
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694; Yuhang Wang, Hai Li, Shujuan Hou, Zhetao Dong; 2026-03-08) — Uses codec motion vectors, residual maps, and frame type metadata to reduce online VSR alignment overhead. Why it matters to OSS: already tracked and directly analogous to exact engine MVs/G-buffers as free alignment/compute-gating signals.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483; Antoine Mercier, Ruan Erasmus, Yashesh Savani, Manik Dhingra; 2023-08-03) — Game supersampling baseline/dataset with rendered auxiliary modalities. Why it matters to OSS: already tracked as the key public gaming supersampling reference.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 broad hits, all GitHub/archive-list style snapshots; no material paper takedown/update signal.
- `"frame generation" "deep learning"`: 4 broad hits including FluidFrames older builds, an OSF registration for MoCha-LD, and unrelated Minecraft content; no material OSS paper update.
- `"real-time upscaling" game`: 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16,596 hits, dominated by unrelated `grape` items; still too noisy for broad conclusions.
- Wayback CDX: confirmed archived snapshots for `arxiv.org/abs/2403.17888`, `arxiv.org/abs/2512.02172` (SplatSuRe), and `github.com/ChrisDud0257/GSASR`; no snapshots returned for guessed `arxiv.org/abs/2604.18047` or `github.com/yifita/GS-STVSR` this run.

## GitHub / Open Source
### Gaussian splatting + super-resolution
- **Azusa309/MedGSSR** (1★, pushed 2026-06-22, MIT) — MedGSSR: Generalizable Medical Image Super-Resolution 3D Reconstruction via Hierarchical Feed-forward Gaussian Splatting. https://github.com/Azusa309/MedGSSR
- **pranav-asthana/SplatSuRe** (81★, pushed 2026-06-16, license unknown) — Official code for SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting. https://github.com/pranav-asthana/SplatSuRe
  - Relevance: SplatSuRe remains the material repo; MedGSSR is medical-domain but could still be skimmed for hierarchical feed-forward GS-SR implementation patterns.

### Frame generation + open source
- **AdvaitPalve11/Jojo-Player** (0★, pushed 2026-06-18, MIT) — Java/FFmpeg video player claiming resolution upscaling and frame generation. https://github.com/AdvaitPalve11/Jojo-Player
- **ShrekDino/FreeGen** (0★, pushed 2026-06-16, Apache-2.0) — Claims open-source upscaling and frame generation with Vulkan + FSR 3.1. https://github.com/ShrekDino/FreeGen
  - Relevance: monitor `FreeGen` only if code/weights become substantive; current star/activity profile is not a serious OSS competitor signal.

### Neural upscaler / analysis tools
- **gokusayayin641-ai/neural-upscaler-x4-page** (0★, pushed 2026-06-26, no license detected) — Blender-facing “Neural Upscaler X4” page. https://github.com/gokusayayin641-ai/neural-upscaler-x4-page
- GitHub Search API returned no new DLSS/FSR/XeSS analysis repos for `DLSS OR FSR OR XeSS analysis upscaler pushed:>=2026-06-15`. Prior watch targets (OptiScaler, lsfg-vk, FidelityFX-SDK Redstone/FSR 4 lineage) remain more material.

## News
- **NVIDIA / DLSS**: NVIDIA developer page confirms current positioning: DLSS 4 introduced Multi Frame Generation and a transformer SR model; DLSS 4.5 adds Dynamic Multi Frame Generation and a second-generation transformer. It also states DLSS Super Resolution uses motion data + prior-frame feedback and that the second-gen transformer improves stability, anti-aliasing, and visual clarity. OSS implication: NVIDIA is explicitly pushing the transformer + dynamic-FG narrative; OSS needs student latency proof plus “one architecture for SR+FG” evidence to avoid being dismissed as research-only.
- **NVIDIA Linux/open-source lead**: Google News RSS surfaced a Tom’s Hardware headline that NVK gained experimental DLSS support via imported CUDA binaries. The article URL returned moved/removed during extraction, so treat as headline-level only. If real, it matters as another sign that “open-source wrapper around closed vendor upscaler” pressure is rising on Linux.
- **AMD / FSR**: RSS headlines from TechPowerUp/Tom’s Hardware/CG Channel claim FSR 4.1 / SDK 2.3 expanded official INT8 support to Radeon RX 7000 / older GPUs and that Valve is testing FSR 4 ahead of Steam Machine. Primary AMD pages were unreachable or timed out in this run, and TechPowerUp bot-check blocked extraction, so this remains headline-level. Strategic implication: AMD appears to be pushing ML upscaling down-stack, directly pressuring OSS’s cross-vendor/student positioning.
- **Intel / XeSS**: RSS was mostly Arc G3 / handheld coverage and noisy comparison headlines; no primary XeSS technical update confirmed. Intel XeSS URL resolved to a generic Intel products page in the fetch, not useful technical content.
- **Apple / Metal**: RSS mentions Game Porting Toolkit 4 and Metal gaming performance commentary; no new primary Metal/ANE upscaling technical update confirmed. Keep ANEForge from 2026-06-25 on the Apple inference watchlist.
- **Game engines**: Unreal blog returned HTTP 403; Unity blog was reachable but the fetched page exposed no obvious DLSS/FSR/XeSS/upscaling update. Google News had one CG Channel headline tying Unreal/upscaling to AMD FSR SDK 2.3, but direct page fetch returned not found.
- Source caveat: news section is mixed-quality RSS plus partial primary-page checks. Treat AMD FSR 4.1/RX7000 and NVK DLSS as follow-up leads, not confirmed facts, until primary release notes or repo commits are inspected.

## Action Items / Follow-ups
- [ ] Ops: investigate `srcnn-v7.0-pico-005`; dashboard reports trainer offline/stuck at step 30,300 for 45h while watcher/origin remain healthy.
- [ ] Deep-read **Neural Texture Compression using Hypernetworks** for tiny material-aware decoder/head patterns.
- [ ] Deep-read **Fewer, Better Frames** to understand the FSR4/world-model FG framing and write the OSS counter-positioning: exact engine MVs + persistent Gaussian canvas vs model-guided external FG.
- [ ] Verify primary AMD FSR 4.1 / SDK 2.3 release notes and whether INT8/RX7000 expansion is real.
- [ ] Verify NVK experimental DLSS support from primary Mesa/NVK commits or merge requests; if true, track as Linux integration pressure.
- [ ] Monitor `SplatSuRe`, `MedGSSR`, and `FreeGen`; only SplatSuRe is material today.
