# OSS Research Brief — 2026-07-05

## Project Status Summary
- Latest commits: `a97a180` tailnet-gated training controls on dashboard; `fcf8084` tailnet-gated training control API/dashboard panel; `0660f55` preserved v7-pico-005 step-5000 snapshot + recovery doc; `56a9a65` reverted DataLoader to `num_workers=0` for WSL2 deadlock; `d1255c8` Linux `WindowsPath` checkpoint-resume shim; plus HAT flash-attn fallback, TF32/DataLoader prefetch, live PSNR proxy fix, fp32 raster smoothing, and apex `.com`→`.org` redirect.
- Open issues: #14 VRAM footprint; #13 Nsight L2 hit-rate profile; #12 packed low-rank canvas layout; #10 G-buffer encoder ablation; #9 RepVGG/ECBSR/RepSR backbone ablation; #8 cross-vendor kernel ports; #7 FSR4 teacher distillation; #6 v6.2-pico launch; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- README status snapshot: pre-alpha; current teacher architecture is v7 / `srcnn-v7.0-pico-005`, with N-D Gaussian canvas, native time axis, `V_xt` cross-correlation, parent-child loss-adaptive density, Mip-Splatting AA filters, and alpha/time-query OSS-FX supervision. Shipping ≤1M-param TensorRT/FP8 student remains untrained.
- Training dashboard: headline still says `v7.0 Pico OSS-FX is training right now`; active run `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz step **26,000**, updated just now/live. Health tooltip contradicts this: **Trainer — offline, stuck at step 30,300 for 189h**. Watcher healthy/published 0s ago, CF Worker healthy HTTP 200 in 86ms, R2 origin healthy HTTP 200. GPU stats unavailable / host offline or rsync failed. Same step as 2026-07-04; stuck duration increased from 165h to 189h.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **MVFusion-GS: Motion-Variance Guided Temporal Attention for High-Quality Dynamic Gaussian Splatting** (arXiv:2607.01578v1, Jianwei Hu, Tingxuan Huang, Hengyu Zhou, Ningna Wang, Xiaohu Guo, Jinshan Lai, Bin Wang, 2026-07-02) — Adds per-Gaussian motion-variance refinement and temporal attention over neighboring timesteps for dynamic/static separation and temporal coherence. **Why it matters to OSS:** direct v7 comparison point for motion-aware Gaussian state; motion-variance statistics could be an ablation against `V_xt` cross-covariance and parent-child density/pruning heuristics.
- **DL-SLAM: Enabling High-Fidelity Gaussian Splatting SLAM in Dynamic Environments based on Dual-Level Probability** (arXiv:2607.01860v1, Ziheng Xu et al., 2026-07-02) — Uses pixel-level and object-level dynamic probabilities to prune dynamic Gaussians from a static GS map. **Why it matters to OSS:** less direct than MVFusion-GS, but the dual-level dynamic-probability idea may transfer to disocclusion/transient-object gating in a persistent canvas.
- Already-tracked papers resurfaced: **Learning Efficient 4D Gaussian Representations from Monocular Videos with Flow Splatting** (arXiv:2606.29976), **Resonant Brane Splatting** (arXiv:2606.29453), and **Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution** (arXiv:2606.29400). No new version materially changed the OSS read this run.

### Super-Resolution / Upscaling
- **AVSR-Diff: Scale-Agnostic Diffusion Priors for Temporally Consistent Arbitrary-Scale Video Super-Resolution** (arXiv:2607.00987v1, Geunhyuk Youk et al., 2026-07-01; already appended 2026-07-03) — Decouples scale-agnostic latent denoising from continuous coordinate rendering with temporally gated recurrence and scale-aware Fourier refinement. **Why it matters to OSS:** strong teacher/pseudo-target candidate for arbitrary-scale temporal SR; not a shipping runtime path.
- **Neural Texture Compression using Hypernetworks** (arXiv:2606.26913v1, Laurent Belcour, 2026-06-25; already appended) — Hypernetwork-generated latent texture decoders, including learned super-resolution decoders. **Why it matters to OSS:** possible tiny material/detail-head reference for G-buffer-conditioned student detail recovery.
- Prior material additions still most actionable for SR/student work: **LCS: An AI-based Low-Complexity Scaler for Power-Efficient Super-Resolution of Game Content** (arXiv:2507.22873), **Continuous Space-Time Video Super-Resolution with 3D Fourier Fields** (arXiv:2509.26325), and **Arbitrary-Scale 3D Gaussian Super-Resolution** (arXiv:2508.16467). No newer low-complexity game-upscaler paper surfaced today.

### Frame Extrapolation / Generation
- Current arXiv broad frame-generation query was noisy: mostly autonomous-driving/video-generation papers, not real-time game frame extrapolation. No new OSS-FX-relevant paper stronger than the already-tracked **Learning Video Dynamics with Predictive Differentiable Rendering** (arXiv:2606.31050), **CausalGS** (arXiv:2605.10586), **VeloGauss** (arXiv:2605.10567), and **RSTT** (arXiv:2203.14186).
- **ForgeDrive** (arXiv:2606.31226) and **LWDrive** (arXiv:2606.29879v2) mention future-frame generation/world-model supervision for autonomous driving, but are too far from real-time reconstruction to merit bibliography append.

### Inference Optimization
- **FlexViT: A Flexible FPGA-based Accelerator for Edge Vision Transformers** (arXiv:2606.31938v1, 2026-06-30) — INT8 FPGA ViT accelerator with dual-mode dataflow and depth-first tiling. **Why it matters to OSS:** indirect reminder that student/teacher deployment needs shape-aware tiling and bandwidth control; not a direct GPU/TensorRT SR reference.
- **MINT: Dynamic-Precision CNN Inference with MSDF Digit-Serial Arithmetic on FPGA** (arXiv:2606.31514v1, 2026-06-30) — Dynamic per-layer precision for CNN inference. **Why it matters to OSS:** indirect low-power quantization lead, but FPGA-specific and not enough for bibliography.
- No new TensorRT/INT8/kernel-fusion SR paper stronger than the already-tracked **Efficient INT8 Single-Image Super-Resolution** and **Decoder-Free Distillation for Quantized Image Restoration**.

### Game Engine / Motion-Vector Integration
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694; already tracked) remained the top motion-vector SR hit: motion-vector-guided deformable alignment, residual-map gating, and frame-type-aware compute allocation. **Why it matters to OSS:** continues to support exact-engine-MV reuse and residual/disocclusion gating.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483; already tracked) remains the most directly game-specific academic baseline from the query.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 hits, still dominated by GitHub/star-list captures; no useful primary paper mirror.
- `"frame generation" "deep learning"`: 4 broad hits, including FluidFrames older builds, DLSS Swapper builds, Minecraft/Radiance capture, and an OSF record; ecosystem archaeology only.
- `"real-time upscaling" game`: 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16,607 hits but polluted by unrelated “grape” media/items; broad search remains unusable. Targeted Wayback CDX checks for GS-STVSR, GSASR, and GRAPE timed out from the cron environment; `STSS` query returned no captures.

## GitHub / Open Source
- **GiulioFede/ResonantBraneSplatting** — pushed 2026-07-01, 0 stars, no license; matches RBS arXiv. **Opportunity:** primitive/raster idea for v7 edge/detail ablations. **Blocker:** no license, read-only until clarified. <https://github.com/GiulioFede/ResonantBraneSplatting>
- **Azusa309/MedGSSR** — pushed 2026-06-22, MIT, 2 stars; hierarchical feed-forward Gaussian splatting for medical image SR / 3D reconstruction. **Relevance:** architecture-only; domain mismatch large. <https://github.com/Azusa309/MedGSSR>
- **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** — still public, MIT, 3 stars, pushed 2026-07-02; static structural analysis of AMD FSR 4.1.0 neural upscaler. **Threat/opportunity:** useful lead for issue #7 FSR4 teacher/operator-layout reconnaissance; verify claims before relying on them. <https://github.com/Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling>
- `"frame generation" "open source"` returned no useful current GitHub repos. `"neural upscaler"` mostly returned `cloudy-sky-gold/neural-upscaler-*` HTML landing/docs pages, not an implementation.
- Vendor direct probes: NVIDIA GeForce News, AMD GPUOpen News, AMD FidelityFX SDK releases, Apple Metal, and Unity releases loaded; Intel XeSS and Unreal blog returned 403. No primary vendor page scrape exposed a new technical upscaling architecture disclosure.

## News
- **DLSS 4:** Google News RSS returned no useful DLSS 4 items in this run. NVIDIA GeForce News loaded, but the lightweight scrape did not expose a new architecture/SDK disclosure.
- **FSR 4:** RSS continues to show competitive momentum: Valve testing AMD FSR 4 ahead of Steam Machine launch (PCMag), Proton Experimental adding FSR 4 support for RDNA 3 on Linux (TechPowerUp), AMD FSR 4.1/backward-support headlines, and AMD FSR SDK v2.3 / FSR 4.1.1 headline. **Threat:** Linux/Steam/handheld availability overlaps OSS’s target positioning; confirm with AMD/Valve primary pages before treating headline details as fact.
- **Intel Arc / XeSS:** RSS was mostly Arc G3/handheld chip performance headlines and low-quality comparison spam. No primary XeSS technical update verified; Intel page returned 403.
- **Apple / Metal:** RSS showed Metal gaming/Game Porting Toolkit commentary and Apple M5 GPU benchmark chatter. Apple Metal overview loaded; no concrete MetalFX/upscaler API change found.
- **GPU announcements:** mostly pricing/market commentary and low-quality aggregator items; no actionable NVIDIA/AMD/Intel architecture disclosure for OSS.
- **Game engines:** Google News RSS produced no useful Unreal/Unity upscaling items. Unity releases page loaded; Unreal blog returned 403.

## Action Items / Follow-ups
- [ ] CRITICAL ops: dashboard still says live/training while trainer is offline/stuck at step 30,300 for **189h** and GPU stats are unavailable; reconcile trainer process, public copy, and health semantics.
- [ ] Add/deep-read **MVFusion-GS** for motion-variance-guided temporal attention and dynamic/static separation; compare to v7 `V_xt` motion encoding and loss-adaptive density.
- [ ] Skim **DL-SLAM** only for dynamic-probability pruning/gating ideas; likely not core prior art.
- [ ] Continue deep-read queue from 2026-07-04: **LCS**, **Continuous Space-Time Video Super-Resolution with 3D Fourier Fields**, and **Arbitrary-Scale 3D Gaussian Super-Resolution**.
- [ ] Track AMD/Valve FSR 4 on Steam/Proton/handhelds and FSR 4.1.x Linux/older-RDNA claims as competitive positioning threats; seek primary vendor links.
- [ ] Keep **RBS / ResonantBraneSplatting** idea-only until license is explicit.
