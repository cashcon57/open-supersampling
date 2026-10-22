# OSS Research Brief — 2026-06-16

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
- README snapshot: v7 remains current: N-D Gaussian canvas with native time axis, `V_xt` cross-correlation in 3D covariance, parent-child loss-adaptive density, Mip-Splatting-style anti-aliasing, α=0.5 OSS-FX supervision, and a future ≤1M-param distilled student. README still emphasizes that dashboard runs are research/teacher models, not the shipping inference model.
- Training dashboard: rendered `https://opensupersampling.com` redirects to `.org`. Visible state is **v7.0 Pico OSS-FX** at step **30,300 / 100,000**, loss **0.18269**, viz @ step **26,000**, cost **$0.76 / 12.63 GPU-hours / 5.05 kWh**, ETA ~1.2d. Contradiction persists/worsened: headline and run card say `v7 is training` / `live`, but service-health tooltip says **Trainer — offline, stuck at step 30,300 for 57h**. Watcher, CF Worker, and R2 are healthy; GPU stats unavailable (`host offline or rsync failed`).

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **RealityBridge: Bridging Editable 3D Gaussian Splatting Driving Simulations and Real-World Videos** (arXiv:2606.16278v1, 2026-06-15; Zhenhua Wu, Yun Pang, Mingkun Chang, Yuwei Ning, Liangzhi Wang, Yi Xiao) — Restores edited 3DGS driving-simulation videos to reduce sim-to-real artifacts, illumination mismatch, and temporal flicker. Why it matters to OSS: direct adjacent evidence that 3DGS render artifacts and temporal flicker need explicit restoration/consistency treatment; useful for OSS capture/synthetic-data realism and dashboard visual QA, but not a real-time SR method.
- **Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting** (arXiv:2606.07670v1, 2026-06-04; Mingzhao Li, Arghya Pal, Guan Yuan Tan) — Replaces frame-index MLP deformation in dynamic 3DGS with continuous-time CfC/LNN cells for smoother time interpolation. Why it matters to OSS: already tracked, but remains a strong v7 fallback if native `V_xt` covariance needs a residual continuous-time motion prior.
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586v1, 2026-05-11; Nengbo Lu, Minghua Pan) — Learns causal dynamics of dynamic 3D scenes from multi-view video using Gaussian representations. Why it matters to OSS: indirect but relevant to persistent-canvas disocclusion/future-state reasoning; likely too heavy for shipping path.
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567v1, 2026-05-11; Nengbo Lu, Bin Zhao) — Learns per-Gaussian velocity fields and physical consistency from dynamic videos. Why it matters to OSS: already in bibliography; still the closest motion-field prior for v7 native-time/OSS-FX losses.
- **GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes** (arXiv:2604.02915v1, 2026-04-03; Mijeong Kim, Jungtaek Kim, Bohyung Han) — Adds GP uncertainty, sparse-region motion estimation, and temporal extrapolation to 4DGS. Why it matters to OSS: opportunity for confidence/uncertainty signals in extrapolated canvas regions; threat only if competitors use uncertainty-gated FG quality controls.

### Super-Resolution / Upscaling
- **EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution** (arXiv:2606.13580v1, 2026-06-11; Dachun Kai, Jiayao Lu, Yueyi Zhang, Xiaoyan Sun) — Uses event-signal high-frequency detail for VSR texture enhancement instead of only motion refinement. Why it matters to OSS: no event input in games, but its alignment/detail split maps cleanly to engine-MV alignment plus confidence-gated detail injection.
- **ConFi-GS: Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution** (arXiv:2605.24964v1, 2026-05-24; Jiaxiang Li, Zongtan Zhou, Zhen Tan, Yadong Liu, Dewen Hu) — Reliability-aware high-frequency injection for GS-SR from low-res multi-view observations. Why it matters to OSS: already tracked; candidate confidence gate for splat spawning, densification, pseudo-label filtering, and detail hallucination restraint.
- **AtlasGS: Brain MRI Spatial Resolution Harmonization With Shared Gaussian Geometry** (arXiv:2606.02961v1, 2026-06-01; Yifan Gao, Peiran Xu, Yimeng He, Haoran Li, Ziyang Long, Yufeng Wang) — Learns subject-specific Gaussian geometry once and reuses it for modality-specific sparse-slice SR. Why it matters to OSS: domain mismatch, but the “shared Gaussian geometry + appearance adaptation” pattern is relevant to separating engine geometry/motion priors from RGB detail heads.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483v1, 2023-08-03; Antoine Mercier, Ruan Erasmus, Yashesh Savani, Manik Dhingra, Fatih Porikli, Guillaume Berger) — Game-rendering supersampling paper with auxiliary motion/depth modalities and a novel dataset. Why it matters to OSS: already tracked; still a primary baseline/dataset reference for game-specific claims.
- **Neural Super-Resolution for Real-time Rendering with Radiance Demodulation** (arXiv:2308.06699v2, 2023-08-13; Jia Li, Ziling Chen, Xiaolong Wu, Lu Wang, Beibei Wang, Lei Zhang) — Separates lighting/material components to improve temporal stability and texture sharpness for real-time rendering SR. Why it matters to OSS: useful reference for G-buffer/radiance decomposition; potential ablation for OSS input channels or losses.

### Frame Extrapolation / Generation
- **STAR-NT: Spatiotemporal Acceleration of Real-Time Neural Transparency Rendering** (arXiv:2606.16747v1, 2026-06-15; Grigoris Tsopouridis et al.) — Uses adaptive screen-space subdivision and temporal reprojection/reuse to reduce neural OIT rendering overhead. Why it matters to OSS: not SR/FG, but reinforces exploiting frame coherence and depth reprojection instead of recomputing every pixel every frame.
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (arXiv:2603.26835v3, 2026-03-27; Shibo Liu) — Deletes learned optical flow from mobile VFI by reusing codec motion vectors; designed around INT8/NPU deployment limits. Why it matters to OSS: already tracked; direct analogue for engine-MV-first OSS-FX and a strong argument against separate heavy FG networks.
- **FrameDiffuser: G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering** (arXiv:2512.16670v1, 2025-12-18; Ole Beisswenger, Jan-Niklas Dihlmann, Hendrik P. A. Lensch) — Causal/interactive G-buffer-conditioned diffusion forward rendering for future frames. Why it matters to OSS: threat/positioning reference: diffusion can synthesize future lighting/detail, but likely too slow and stochastic for OSS’s <2ms deterministic runtime; useful teacher-data idea.
- **A Scale-Adaptive Framework for Joint Spatiotemporal Super-Resolution with Diffusion Models** (arXiv:2604.21903v1, 2026-04-23; Max Defez et al.) — Reuses one architecture across spatial and temporal SR factors via conditional-mean prediction plus residual diffusion. Why it matters to OSS: supports single-model multi-scale/time conditioning, but diffusion runtime is non-starter for the shipping path.
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047v1, 2026-04-20; Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu) — Continuous spatial+temporal VSR using 2DGS rather than dense pixel queries. Why it matters to OSS: already core prior art; still the strongest external validation of OSS’s canvas-as-SR+time-rendering thesis.

### Inference Optimization
- **Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution** (arXiv:2308.02794v1, 2023-08-05; Yong Liu, Hang Dong, Boyang Liang, Songwei Liu, Qingji Dong, Kai Chen) — Reworks SR transformer units to avoid TensorRT-hostile redundant ops. Why it matters to OSS: already tracked; relevant if HAT-style teacher blocks need deployment-aware simplification for distillation.
- **NAWQ-SR: A Hybrid-Precision NPU Engine for Efficient On-Device Super-Resolution** (arXiv:2212.09501v3, 2022-12-15; Stylianos I. Venieris, Mario Almeida, Royson Lee, Nicholas D. Lane) — Hybrid-precision SR execution for constrained NPUs. Why it matters to OSS: older but still useful for per-layer precision allocation and hardware-aware student deployment.
- **QuickSRNet: Plain Single-Image Super-Resolution Architecture for Faster Inference on Mobile Platforms** (arXiv:2303.04336v2, 2023-03-08; Guillaume Berger et al.) — Very simple real-time mobile SR architecture. Why it matters to OSS: useful floor/baseline for Pico-tier student simplicity; reinforces that the shipping model should be boring and compiler-friendly.

## Archive.org Findings
- `"gaussian splatting" super-resolution`: 3 broad/noisy hits, mostly archived GitHub/star-list/index pages; no primary GS-SR paper snapshot surfaced.
- `"frame generation" deep learning`: 4 hits: FluidFrames Windows builds, DLSS Swapper Windows builds, MoCha-LD OSF registration, and a Minecraft/Radiance archive. Ecosystem breadcrumbs only.
- `"real-time upscaling" game`: 0 hits.
- Targeted acronym searches remain noisy: `GS-STVSR` 0, `STSS super resolution` 0, `GSASR` returned an unrelated CIA document, `GRAPE super resolution` returned unrelated media. Use targeted Wayback CDX URLs next run rather than broad Archive search for acronyms.

## GitHub / Open Source
- **ShrekDino/FreeGen** — new lead, pushed 2026-06-16; 0 stars; Apache-2.0; describes itself as “Open-source upscaling and frame generation — Vulkan, FSR 3.1.” Why it matters: potential direct OSS-adjacent open-source competitor/integration reference, but needs code/license/build seriousness review before treating as real. https://github.com/ShrekDino/FreeGen
- **optiscaler/OptiScaler** — active, pushed 2026-06-16; ~8.7k stars; GPL-3.0; bridges DLSS/XeSS/FSR inputs and enables FSR-FG/XeFG paths. Why it matters: still the practical integration ecosystem to watch for DLL-shim surfaces and user expectations. https://github.com/optiscaler/OptiScaler
- **Adam-AtlasSoftware/RDNU** — pushed 2026-06-09; 16 stars; GPL-3.0; RDNA3-focused G-buffer-guided neural upscaler trained on public/game-adjacent datasets. Why it matters: possible cross-vendor open neural upscaler competitor/baseline; still needs runnable-code/weights verification. https://github.com/Adam-AtlasSoftware/RDNU
- **hvcl/3DCT_SR_DDNM_NAB-GS** — pushed 2026-06-12; 1 star; zero-shot 3D CT SR with diffusion 2D projection priors and Negative Alpha Blending GS. Why it matters: medical/no direct game threat, but NAB-style reliability blending could inform splat detail injection. https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS
- GitHub search returned no new high-signal DLSS/FSR analysis-tool repos. `ShiinaManatsu/dlss-compositor` from the prior brief now returns GitHub API 404, so treat it as removed/private/unverified.

## News
- Primary scrape via `web_extract` failed with Firecrawl `Payment Required`; news section uses Google News RSS plus lightweight vendor-page HTML checks. Treat RSS-only items as leads unless primary vendor page is named.
- **NVIDIA / DLSS:** NVIDIA developer blog tag page lists **“NVIDIA DLSS 4.5 Delivers Super Resolution Upgrades and New Dynamic Multi Frame Generation”**, **“Build AI-Powered Games with NVIDIA DLSS 4.5, RTX, and Unreal Engine 5”**, and related RTX game-dev posts. Google News RSS also surfaced `CONTROL Resonant` DLSS 4.5/path-tracing launch coverage. **Threat:** NVIDIA’s story is no longer just SR quality; it is SR + dynamic multi-frame generation + UE tooling + path tracing/RR as a bundle.
- **AMD / FSR:** Google News RSS surfaced AMD FSR 4.1 RDNA3/quality-parity coverage and `The Outlast Trials` FSR 4 support discovery. AMD GPUOpen news page is reachable but top-level HTML parse only exposed general FSR SDK/Ray Regeneration/Radiance Caching navigation, not a fresh dated article. **Threat/opportunity:** FSR ML upscaling expansion beyond RDNA4 narrows OSS’s “vendor-neutral ML SR” differentiation; openness and unified SR+FG need proof.
- **Intel / XeSS / Arc:** RSS surfaced Arc G3 handheld/driver coverage and low-signal DLSS/XeSS/FSR comparison posts. **Signal:** handheld and midrange GPU upscaling benchmarks stay visible; OSS Pico/Steam Deck claims should remain measurement-gated.
- **Apple / Metal:** Apple Metal page reachable; RSS surfaced Apple developer-tool announcements, Mac/Cyberpunk performance commentary, and Game Porting Toolkit-adjacent coverage. **Opportunity:** Mac gaming keeps gaining mindshare; Metal backend remains designed-only and should be framed as future work until measured.
- **Game engines:** No strong Unreal/Unity upscaling release-note headline surfaced in this pass beyond NVIDIA’s DLSS 4.5/UE5 messaging.

## Action Items / Follow-ups
- [ ] Ops: investigate v7 trainer immediately. Dashboard shows step **30,300** and reports **Trainer offline/stuck for 57h** while publishing infra is healthy.
- [ ] Inspect `ShrekDino/FreeGen` for actual code, build path, Vulkan/FSR 3.1 integration, and whether it is a real competitor or repo-shell.
- [ ] Deep-read `RealityBridge` for 3DGS artifact/flicker restoration ideas applicable to synthetic capture and dashboard visual QA.
- [ ] Deep-read `GP-4DGS` for uncertainty/confidence signals in extrapolated Gaussian motion; compare against current OSS v7 confidence/density heuristics.
- [ ] Keep `VeloGauss`, `ANVIL`, and `GS-STVSR` as top references for v7 native-time, privileged-motion, and unified SR+FG positioning.
- [ ] Retry targeted Wayback CDX checks for known primary URLs (GS-STVSR, GSASR, GRAPE, 3DGS³) with per-URL backoff instead of broad acronym search.
- [ ] Treat DLSS 4.5 dynamic multi-frame generation as a competitive-positioning update: OSS should explicitly contrast one-canvas SR+FG with vendor multi-frame FG stacks once measured.

## Bibliography Update
- Appended concise entries for `RealityBridge`, `GP-4DGS`, and `FrameDiffuser` because they add new OSS-relevant angles: 3DGS artifact/flicker restoration, uncertainty-aware 4DGS extrapolation, and G-buffer-conditioned future-frame diffusion.
