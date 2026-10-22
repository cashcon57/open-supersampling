# OSS Research Brief — 2026-07-07

## Project Status Summary
- Latest commits: `a97a180` tailnet-gated training controls on dashboard; `fcf8084` tailnet-gated training control API + dashboard panel; `0660f55` preserved v7-pico-005 step-5000 snapshot + recovery doc; `56a9a65` reverted DataLoader to `num_workers=0` for WSL2 deadlock; `d1255c8` Linux `WindowsPath` checkpoint-resume shim; plus HAT flash-attn fallback, TF32/DataLoader prefetch, Live PSNR proxy fix, fp32 raster smoothing, and apex `.com`→`.org` redirect.
- Open issues: #14 VRAM footprint; #13 Nsight L2 hit-rate profile; #12 packed low-rank canvas layout; #10 G-buffer encoder ablation; #9 RepVGG/ECBSR/RepSR backbone ablation; #8 cross-vendor kernel ports; #7 FSR4 teacher distillation; #6 v6.2-pico launch; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- README status snapshot: pre-alpha; README still pitches unified SR + frame extrapolation via persistent 2D Gaussian canvas, with current public text partly lagging v7 naming. Shipping target remains a ≤1M-param TensorRT/FP8 student distilled from HAT-Tiny-class teachers; the student is not trained yet.
- Training dashboard: headline says `v7.0 Pico OSS-FX is training right now`; active run `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz step **26,000**, updated just now/live. Health tooltip contradicts this: **Trainer — offline, stuck at step 30,300 for 44h**. Watcher healthy/published 1s ago, CF Worker healthy HTTP 200 in 83ms, R2 origin healthy HTTP 200. GPU stats unavailable / host offline or rsync failed. Compared with 2026-07-06, step/loss still did not advance while the stuck-duration counter moved from 20h to 44h; this confirms a continuing trainer outage, not recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **DeGenseGS: Geometrically and Semantically Decoupled Surgical Scene Understanding in 4D Gaussian Splatting** (Yimo Wang, Bin Kang, Shuojue Yang, Yueming Jin; arXiv:2607.04761v1; 2026-07-06) — Decouples semantic evolution from geometric deformation in 4DGS for surgical video. Why it matters to OSS: mostly off-domain, but the semantic/geometric disentanglement is a useful design reference for keeping OSS canvas motion, appearance, and future object/region masks from over-coupling.
- **FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles** (Lucas Yunkyu Lee, Soonho Kim, Youngwook Kim, Sangmin Kim, Jaesik Park; arXiv:2605.03337v3; updated 2026-07-01) — Systematically analyzes dynamic-4DGS heuristics, including temporal partitioning from Gaussian durations and decoupling photometric fidelity from motion behavior. Why it matters to OSS: directly relevant to v7 native time-axis/cross-covariance design and should be deep-read before locking density/pruning and motion-loss choices.
- **Learning Efficient 4D Gaussian Representations from Monocular Videos with Flow Splatting** (Shengjun Zhang, Jinzhao Li, Xin Fei, Yueqi Duan; arXiv:2606.29976v1; 2026-06-29; already tracked) — Builds 4D Gaussian velocity fields and splats optical flow to supervise dynamics. Why it matters to OSS: close comparator for v7 exact-MV-driven motion encoding and canvas warping.
- **Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution / QuADA-GS** (Giulio Federico et al.; arXiv:2606.29400v1; 2026-06-28; already tracked) — Feed-forward adaptive densification / routing for 2D-GS ASR. Why it matters to OSS: direct comparison target for parent-child density and packed-canvas budgeting.
- **Resonant Brane Splatting for Arbitrary-Scale Super-Resolution** (Giulio Federico et al.; arXiv:2606.29453v2; updated 2026-07-01; already tracked) — Replaces flat Gaussians with higher-order Gaussian-Hermite “Branes” to encode high-frequency detail with fewer overlaps. Why it matters to OSS: possible primitive/raster ablation for edge/detail recovery, but code remains no-license in GitHub search.

### Super-Resolution / Upscaling
- **FMA-Net++: Motion- and Exposure-Aware Joint Video Super-Resolution and Deblurring** (Geunhyuk Youk, Jihyong Oh, Munchurl Kim; arXiv:2512.04390v2; updated 2026-06-30) — Non-recurrent, sequence-level VSR+deblurring with exposure-aware degradation priors. Why it matters to OSS: useful teacher-side degradation modeling reference if OSS adds exposure/motion-blur augmentation, but not a shipping-runtime model.
- **AVSR-Diff: Scale-Agnostic Diffusion Priors for Temporally Consistent Arbitrary-Scale Video Super-Resolution** (Geunhyuk Youk et al.; arXiv:2607.00987v1; 2026-07-01; already tracked) — Diffusion-prior arbitrary-scale VSR with temporally-gated recurrence and scale-aware Fourier refinement. Why it matters to OSS: possible offline teacher/pseudo-target source for v7/student training.
- **SlimDiffSR: Toward Lightweight and Efficient Remote Sensing Image Super-Resolution via Diffusion Model Distillation** (Ce Wang, Zhenyu Hu, Wanjie Sun; arXiv:2605.02198v2; updated 2026-05-17) — Single-step diffusion teacher plus pruning and MMD feature alignment for a much smaller SR student. Why it matters to OSS: off-domain but relevant to HAT/diffusion-teacher → ≤1M-param student compression.
- **OFTSR: One-Step Flow for Image Super-Resolution with Tunable Fidelity-Realism Trade-offs** (Yuanzhi Zhu et al.; arXiv:2412.09465v3; updated 2026-03-19) — Distills a conditional-flow SR teacher into one-step student outputs with controllable fidelity/realism. Why it matters to OSS: useful distillation/loss-shaping idea if student outputs need a runtime quality knob.

### Frame Extrapolation / Generation
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (Shibo Liu; arXiv:2603.26835v3; updated 2026-04-01; already tracked) — Uses codec motion vectors to prealign frames and leave a convolution-heavy residual net that quantizes well on mobile NPUs. Why it matters to OSS: strong analogue for exact-engine-MV frame extrapolation and for designing the student graph around accelerator-friendly ops.
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (Nengbo Lu, Minghua Pan; arXiv:2605.10586v1; already tracked) — Gaussian-state future prediction with physical causal factors. Why it matters to OSS: long-horizon OSS-FX regularization reference beyond simple alpha interpolation.
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (Nengbo Lu, Bin Zhao; arXiv:2605.10567v1; already tracked) — Per-Gaussian velocity fields with global physical consistency. Why it matters to OSS: overlaps v7 time/cross-covariance motion representation; good motion-loss comparator.

### Inference Optimization
- No fresh TensorRT-specific SR paper surfaced in primary arXiv queries. Relevant rediscoveries remain deployment-aware quantized SR, one-step flow/diffusion distillation, and ANVIL-style accelerator-native graph design. Why it matters to OSS: reinforces that the ≤1M-param student needs QAT/FP8-INT8 validation plus fused kernels, not just ONNX/TensorRT export.

### Game Engine / Motion-Vector Integration
- **Compressed-Domain-Aware Online Video Super-Resolution** (Yuhang Wang et al.; arXiv:2603.07694v1; already tracked) — Online VSR using codec motion vectors, residual maps, and frame types to reduce alignment cost. Why it matters to OSS: codec MVs are weaker than engine MVs, but the gating/alignment structure is useful for exact-MV reuse and unreliable-region handling.
- No new game-engine upscaler paper surfaced today. The closest current OSS-relevant academic baselines remain ANVIL, CDA-VSR, LCS, and older game-content neural supersampling work.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 Archive.org hits, all broad GitHub/star-list captures; no useful primary paper mirror.
- `"frame generation" "deep learning"`: 4 broad hits: OSF MoCha-LD registration, FluidFrames older Windows builds, Minecraft/Radiance capture, and DLSS Swapper builds. Useful only as ecosystem archaeology.
- `"real-time upscaling" game`: 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16,609 noisy hits dominated by unrelated “grape” results; broad acronym query is unusable without targeted URLs.
- Targeted Wayback CDX: `https://arxiv.org/abs/2501.19396` has captures from 2025-06-12 and 2026-06-05. Targeted checks for `2503.16342`, `hongyuanyu/GSASR`, and `2403.17152` timed out in this run, so absence is not evidence.

## GitHub / Open Source
- **GiulioFede/ResonantBraneSplatting** — pushed 2026-07-01, Python, 0 stars, no license. Opportunity: higher-order primitive/raster idea for v7 edge/detail quality; blocker: no license, do not reuse code. <https://github.com/GiulioFede/ResonantBraneSplatting>
- **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** — pushed 2026-07-02, Python, MIT, 4 stars. Static structural analysis of AMD FSR 4.1.0 neural upscaler. Opportunity: useful issue #7 reconnaissance for operator layout / teacher-signal extraction if claims verify. <https://github.com/Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling>
- **cloudy-sky-gold/neural-upscaler-x4-page** and **neural-upscaler-docs-page** — HTML landing/docs pages only, no evident technical payload from metadata; likely ignore unless the actual model/tool appears.
- `"frame generation" "open source" pushed:>=2026-06-23` and `"DLSS" "analysis" "FSR" pushed:>=2026-06-23` returned no GitHub repository hits.

## News
- **FSR 4 / FSR 4.1:** Google News RSS produced a PCMag Middle East headline claiming Valve is testing FSR 4 ahead of Steam Machine launch, plus low-quality/spammy duplicate leads. Treat as headline-level until confirmed by Valve/AMD primary sources. Competitive watch: FSR 4 moving toward handheld/older hardware would pressure OSS’s vendor-neutral + low-cost positioning.
- **NVIDIA / DLSS 4.5:** Primary NVIDIA developer page now describes **DLSS 4.5** with Dynamic Multi Frame Generation, 6x Multi Frame Generation, and a second-generation transformer model for Super Resolution/Ray Reconstruction/DLAA. RSS also surfaced an NVIDIA 2026-07-07 item for DOOM: The Dark Ages/Revelations adding DLSS Multi Frame Generation + path tracing. Threat: NVIDIA is widening both frame-generation multiplier and transformer-quality messaging while OSS still has no trained student/runtime.
- **Intel Arc / XeSS:** RSS mostly showed Arc G3 handheld/performance chatter and comparison spam; Intel XeSS page extraction fell through to a generic Intel product page, so no verified XeSS technical update.
- **Apple Silicon / Metal:** Apple Metal page confirms Metal 4 ML+graphics integration messaging and MetalFX support for Upscaling, Frame Interpolation, and Denoising; RSS surfaced generic Metal gaming/M5 GPU commentary. No new MetalFX API delta verified, but Apple continues to position ML inference as a first-class graphics path.
- **Game engines:** Unity releases page reachable but no upscaling-specific signal in lightweight fetch; Unreal blog returned HTTP 403. No actionable Unreal/Unity upscaling API update found.

## Action Items / Follow-ups
- [ ] CRITICAL ops: dashboard still claims live/training but trainer is offline/stuck at step 30,300 for 44h; check trainer process and dashboard health-state semantics.
- [ ] Deep-read **FreeTimeGS++** before finalizing v7 temporal-duration, density/pruning, and motion-loss design.
- [ ] Revisit **ANVIL** + **CDA-VSR** together for an exact-engine-MV ablation plan: MV prealign, residual-only student, and unreliable-region gating.
- [ ] Skim **FMA-Net++**, **SlimDiffSR**, and **OFTSR** for teacher/student degradation and one-step distillation ideas that can transfer despite off-domain datasets.
- [ ] Track **DLSS 4.5** positioning (Dynamic/6x MFG + second-gen transformer) and confirm the FSR 4 / Valve handheld lead from primary sources before treating it as a competitive threat.
- [ ] Keep **ResonantBraneSplatting** idea-only unless a usable license appears.
