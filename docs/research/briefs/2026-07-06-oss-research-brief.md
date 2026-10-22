# OSS Research Brief — 2026-07-06

## Project Status Summary
- Latest commits: `a97a180` tailnet-gated training controls on dashboard; `fcf8084` tailnet-gated training control API + dashboard panel; `0660f55` preserved v7-pico-005 step-5000 snapshot + recovery doc; `56a9a65` reverted DataLoader to `num_workers=0` for WSL2 deadlock; `d1255c8` Linux `WindowsPath` checkpoint-resume shim; plus HAT flash-attn fallback, TF32/DataLoader prefetch, Live PSNR proxy fix, fp32 raster smoothing, and apex `.com`→`.org` redirect.
- Open issues: #14 VRAM footprint; #13 Nsight L2 hit-rate profile; #12 packed low-rank canvas layout; #10 G-buffer encoder ablation; #9 RepVGG/ECBSR/RepSR backbone ablation; #8 cross-vendor kernel ports; #7 FSR4 teacher distillation; #6 v6.2-pico launch; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- README status snapshot: pre-alpha; current architecture is v7 / `srcnn-v7.0-pico-005` teacher with N-D Gaussian canvas, native time axis, `V_xt` cross-correlation, parent-child loss-adaptive density, Mip-Splatting AA filters, and alpha/time-query OSS-FX supervision. Shipping ≤1M-param TensorRT/FP8 student remains untrained.
- Training dashboard: headline says `v7.0 Pico OSS-FX is training right now`; active run `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz step **26,000**, updated just now/live. Health tooltip contradicts this: **Trainer — offline, stuck at step 30,300 for 20h**. Watcher healthy/published 1s ago, CF Worker healthy HTTP 200 in 90ms, R2 origin healthy HTTP 200. GPU stats unavailable / host offline or rsync failed. Compared with yesterday's brief, step/loss still did not advance; the stuck-duration counter reset from 189h yesterday to single-digit hours earlier today and is now 20h, so treat this as dashboard health-state reset/inconsistency, not training recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **TemporalGS: Training-Free Plug-and-Play Acceleration for 3D Gaussian Splatting Rendering via Temporal Priors** (Yuhongze Zhou, Zihao Yang, Xinxin Zuo; arXiv:2607.03390v1; 2026-07-03) — Uses temporal priors to accelerate 3DGS rendering without retraining. Why it matters to OSS: directly relevant to v7 canvas rendering cost and temporal coherence; worth checking whether its reuse/prior scheme maps to OSS's exact-engine-motion persistent canvas or active-mask pruning.
- **MVFusion-GS: Motion-Variance Guided Temporal Attention for High-Quality Dynamic Gaussian Splatting** (Jianwei Hu, Tingxuan Huang, Hengyu Zhou, Ningna Wang, Xiaohu Guo, Jinshan Lai, Bin Wang; arXiv:2607.01578v1; 2026-07-02; already tracked) — Per-Gaussian motion-variance refinement and temporal attention for dynamic/static separation. Why it matters to OSS: direct comparison target for v7 `V_xt` motion encoding, temporal coherence, and loss-adaptive density/pruning heuristics.
- **DL-SLAM: Enabling High-Fidelity Gaussian Splatting SLAM in Dynamic Environments based on Dual-Level Probability** (Ziheng Xu, Qingfeng Li, Xuefeng Liu; arXiv:2607.01860v2; 2026-07-02) — Dynamic-probability pruning in Gaussian maps. Why it matters to OSS: possible disocclusion/transient-object gating idea, but SLAM domain makes it lower priority.

### Super-Resolution / Upscaling
- No newly published, directly game-SR paper surfaced today. Broad arXiv hits were mostly older or off-domain: **Deep Unfolding Real-Time Super-Resolution Using Subpixel-Shift Twin Image and Convex Self-Similarity Prior** (arXiv:2602.21513; satellite/twin-image MISR) and **Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content** (arXiv:2602.11339; already surfaced in June, useful streaming/compression robustness reference).
- Already-tracked current SR queue remains **AVSR-Diff** for arbitrary-scale temporally consistent VSR teacher/pseudo-targets; **LCS** for low-complexity game-content scaler/student positioning; **Continuous Space-Time Video Super-Resolution with 3D Fourier Fields** for continuous-time SR representation and analytic anti-aliasing; **Arbitrary-Scale 3D Gaussian Super-Resolution** for scale-aware Gaussian SR.

### Frame Extrapolation / Generation
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586v1; 2026-05-11; already tracked) — Gaussian-state future prediction / physical-causality reference. Why it matters to OSS: long-horizon OSS-FX regularization idea beyond single alpha interpolation.
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567v1; 2026-05-11; already in bibliography) — Per-Gaussian velocity fields with physical consistency. Why it matters to OSS: overlaps v7 native time/cross-covariance motion representation; still a high-priority motion-loss read.
- **Continuous Space-Time Video Super-Resolution with 3D Fourier Fields** (arXiv:2509.26325v2; already appended 2026-07-04) resurfaced in the frame-generation query; keep it as the closest continuous space-time SR baseline, not a new paper.

### Inference Optimization
- No new TensorRT-specific SR paper. The relevant arXiv hits were already known quantized/mobile SR references: **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (arXiv:2604.20291), **Towards Clip-Free Quantized Super-Resolution Networks** (arXiv:2308.11365), **Unfolding Once is Enough** (arXiv:2308.02794), **NAWQ-SR** (arXiv:2212.09501), and mobile quantized SR challenge reports. Why it matters to OSS: reinforces that the student path needs deployment-aware QAT plus real fused kernels; no architecture-changing new result today.

### Game Engine / Motion-Vector Integration
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483; already tracked) remains the strongest game-specific academic baseline from today's query.
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694; already tracked) resurfaced as the closest motion-vector-guided analogue. Why it matters to OSS: codec motion vectors are weaker than engine MVs, but the online gating/alignment structure is useful for OSS exact-MV reuse.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 Archive.org hits, all GitHub/star-list or broad awesome-list captures; no useful primary paper mirror.
- `"frame generation" "deep learning"`: 4 broad hits, including an OSF MoCha-LD registration, FluidFrames older builds, DLSS Swapper builds, and unrelated Minecraft/Radiance captures; ecosystem archaeology only.
- `"real-time upscaling" game`: 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: broad query returned 16,609 noisy hits dominated by unrelated “grape” media/items; not useful without targeted URLs.
- Targeted Wayback CDX: **GSASR GitHub** has captures (`20250313053211`, `20250801002017`, `20250804114800`); targeted GRAPE openaccess URL returned no captures; GS-STVSR arXiv CDX timed out this run.

## GitHub / Open Source
- **GiulioFede/ResonantBraneSplatting** — pushed 2026-07-01, 0 stars, Python, no license. Claims higher-order “Branes” for arbitrary-scale SR. Opportunity: idea-only primitive/raster ablation for v7 edge/detail quality. Blocker: no license; do not use code. <https://github.com/GiulioFede/ResonantBraneSplatting>
- **Azusa309/MedGSSR** — pushed 2026-06-22, MIT, 2 stars. Hierarchical feed-forward Gaussian splatting for medical image SR / 3D reconstruction. Relevance: architecture pattern only; domain mismatch large. <https://github.com/Azusa309/MedGSSR>
- **pranav-asthana/SplatSuRe** — pushed 2026-06-16, 82 stars, license unclear. Selective super-resolution for multi-view-consistent 3DGS. Opportunity: reinforces selective/region-gated expensive refinement instead of uniform dense work. <https://github.com/pranav-asthana/SplatSuRe>
- **hvcl/3DCT_SR_DDNM_NAB-GS** — pushed 2026-06-12, Apache-2.0, 2 stars. Medical CT SR with diffusion-based 2D projection priors and Negative Alpha Blending Gaussian Splatting. Relevance: low priority/domain-specific, but alpha blending / projection-prior details may be skim-worthy if OSS needs pseudo-target generation ideas.
- **ShrekDino/FreeGen** — pushed 2026-06-16, Apache-2.0, 0 stars. Claims open-source upscaling/frame generation via Vulkan/FSR 3.1. Threat/opportunity: likely thin wrapper, but it occupies “open-source frame generation” search space; worth a quick repo skim for positioning.
- **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** — pushed 2026-07-02, MIT, 4 stars. Static structural analysis of AMD FSR 4.1.0 neural upscaler. Opportunity for issue #7 FSR4 teacher/operator-layout reconnaissance; verify claims before relying on them.
- **Adam-AtlasSoftware/RDNU** — pushed 2026-06-09, GPL-3.0, 16 stars. RDNA3-focused decoupled neural upscaler using G-buffer guidance and game-like datasets. Relevance: competitive open-source neural-upscaler lead; GPL code is not directly reusable for Apache OSS, but architecture/dataset claims remain worth skimming.

## News
- **FSR 4 / FSR 4.1:** Google News RSS surfaced CG Channel, TechSpot, PCMag Middle East, DLCompare, and Tech Times headlines around FSR 4.1/SDK 2.3, RX 7000 access, possible older-GPU/INT8 path, and Valve/Steam Machine testing. Treat these as headline-level leads unless confirmed from AMD/Valve primary pages. Threat: AMD is pushing ML upscaling down into older/handheld-relevant Radeon hardware, overlapping OSS Pico/Steam-Deck-style positioning. Opportunity: INT8-vs-FP8 public discussion supports OSS deployment-aware student quantization narrative.
- **NVIDIA / DLSS 4:** Direct NVIDIA GeForce news page was reachable and contains DLSS/frame-generation mentions, but no newly verified architecture/SDK disclosure was extracted today. RSS returned no useful DLSS 4 item in the last-14-days query.
- **Intel Arc / XeSS:** RSS mostly showed Arc G3 handheld/performance chatter and low-quality comparison spam. No primary XeSS technical update verified.
- **Apple Silicon / Metal:** RSS showed Metal gaming/Game Porting Toolkit 4 commentary, MacStories/Snazzy Labs coverage, and M5 GPU benchmark chatter. No concrete MetalFX/upscaler API change verified.
- **Game engines:** Unity releases page showed no DLSS/FSR/XeSS/upscaling mentions in lightweight fetch; Unreal blog returned HTTP 403 to the collector. No actionable Unreal/Unity upscaling API change found.

## Action Items / Follow-ups
- [ ] CRITICAL ops: dashboard remains contradictory — public copy says live/training at step 30,300 while trainer health says offline/stuck and GPU stats are unavailable. Step/loss have not advanced since yesterday; stuck timer reset then climbed to 20h, so check trainer process + health-state reset semantics.
- [ ] Deep-read **TemporalGS** for training-free temporal-prior 3DGS acceleration; compare against v7 active-mask pruning, temporal cache reuse, and exact-engine-MV canvas updates.
- [ ] Deep-read **MVFusion-GS** and compare motion-variance temporal attention against v7 `V_xt`, dynamic/static separation, and loss-adaptive density.
- [ ] Skim **RDNU** and **RE-of-FSR-4.1.0-Upscaling** for dataset/operator-layout signals relevant to issue #7; avoid code reuse if licensing/verification is weak.
- [ ] Track AMD FSR 4.1 INT8-on-RDNA3/RDNA2 rollout as a competitive threat to OSS’s “vendor-neutral + older hardware” positioning; confirm via AMD/Valve primary pages.
- [ ] Keep **RBS / ResonantBraneSplatting** idea-only until license is explicit.
