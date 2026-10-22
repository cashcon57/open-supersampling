# OSS Research Brief — 2026-07-08

## Project Status Summary
- Latest commits: `a97a180` tailnet-gated training controls on dashboard; `fcf8084` tailnet-gated training control API + dashboard panel; `0660f55` preserved v7-pico-005 step-5000 snapshot + recovery doc; `56a9a65` reverted DataLoader to `num_workers=0` for WSL2 deadlock; `d1255c8` Linux `WindowsPath` checkpoint-resume shim; plus HAT flash-attn fallback, TF32/DataLoader prefetch, Live PSNR proxy fix, fp32 raster smoothing, and apex `.com`→`.org` redirect.
- Open issues: #14 VRAM footprint; #13 Nsight L2 hit-rate profile; #12 packed low-rank canvas layout; #10 G-buffer encoder ablation; #9 RepVGG/ECBSR/RepSR backbone ablation; #8 cross-vendor kernel ports; #7 FSR4 teacher distillation; #6 v6.2-pico launch; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- README status snapshot: pre-alpha; README now names v7 as the active teacher run (`srcnn-v7.0-pico-005`) with N-D Gaussian canvas/native time axis, parent-child density, Mip-Splatting filters, and α=0.5 OSS-FX supervision. Shipping model remains a not-yet-trained ≤1M-param student, targeting TensorRT FP8/custom cross-vendor kernels and <2ms-class inference.
- Training dashboard: headline says `v7.0 Pico OSS-FX is training right now`; active run `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz step **26,000**, updated 1 min ago/live. Health tooltips contradict this: **Trainer — offline, stuck at step 30,300 for 56h**; **Watcher — healthy, published 1s ago**; **CF Worker — offline, URLError**; **R2 origin — healthy, HTTP 200**. GPU stats unavailable / host offline or rsync failed. Compared with 2026-07-07, step/loss still did not advance and stuck-duration increased from 44h→56h, plus CF Worker regressed from healthy to offline/URLError.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **GUSH3R: Everyone Everywhere All at Once as Gaussians** (Keito Abe, Kaede Shiohara, Takashi Otonari, Toshihiko Yamasaki; arXiv:2607.05243v1; 2026-07-06) — Feed-forward dynamic human-scene reconstruction from monocular video using Gaussian representations. Why it matters to OSS: off-domain, but relevant to dynamic/transient object handling and feed-forward Gaussian state prediction; possible reference if OSS v7 needs object/transient-canvas separation beyond exact engine MVs.
- **PRISM3D: Probabilistic Refinement and Robust Initialization for Physically Consistent Scene Modeling under Extreme Motion Blur** (Gopi Raju Matta, Reddypalli Trisha, Vemunuri Divya Madhuri, Kaushik Mitra; arXiv:2607.03855v1; 2026-07-04) — Reconstructs scenes under severe motion blur with probabilistic refinement. Why it matters to OSS: useful only as a blur/degradation robustness lead for teacher augmentation; not directly a game upscaling architecture paper.
- **TemporalGS: Training-Free Plug-and-Play Acceleration for 3D Gaussian Splatting Rendering via Temporal Priors** (Yuhongze Zhou, Zihao Yang, Xinxin Zuo, Juwei Lu; arXiv:2607.03390v1; 2026-07-03; already tracked) — Uses temporal coherence to skip/reuse 3DGS rendering work without retraining. Why it matters to OSS: direct study target for v7 canvas rendering cost, temporal cache reuse, and active-mask pruning.
- **MVFusion-GS: Motion-Variance Guided Temporal Attention for High-Quality Dynamic Gaussian Splatting** (Jianwei Hu et al.; arXiv:2607.01578v1; 2026-07-02; already tracked) — Adds motion-variance temporal attention for dynamic GS. Why it matters to OSS: comparison target for `V_xt` motion encoding and density/pruning heuristics.
- **DeGenseGS: Geometrically and Semantically Decoupled Surgical Scene Understanding in 4D Gaussian Splatting** (Yimo Wang, Bin Kang, Shuojue Yang, Yueming Jin; arXiv:2607.04761v1; 2026-07-06; already in 2026-07-07 brief) — Decouples semantic and geometric evolution in 4DGS. Why it matters to OSS: weak/off-domain but relevant to avoiding over-coupled appearance/motion/mask state.

### Super-Resolution / Upscaling
- **Perceiving Better Moments: Cover Frame Reselection and Enhancement for Live Photos with the Live2K Dataset** (Junyu Lou, Kai Chen, Weiyi You, Hui Zeng, Lei Zhang, Shuhang Gu; arXiv:2607.04151v1; 2026-07-05) — Addresses quality mismatch between high-quality still-photo and constrained video pipelines in live-photo bursts. Why it matters to OSS: off-domain, but the photo-vs-video quality gap resembles teacher/pseudo-target construction from richer offline frames; possible data-curriculum inspiration, not a runtime path.
- **Neural Texture Compression using Hypernetworks** (Laurent Belcour; arXiv:2606.26913v1; 2026-06-25; already tracked) — Real-time neural texture decoders via hypernetworks. Why it matters to OSS: possible material/detail-head or G-buffer-conditioned student idea.
- **Ultra Flash: Scaling Real-Time Streaming Video Generation to High Resolutions** (Luxury et al.; arXiv:2606.09150v2; updated 2026-06-08; already tracked) — Cache/sparse causalization for real-time high-res streaming video generation. Why it matters to OSS: deployment checklist for cache-aware temporal upsampling, but generative-video domain is far from OSS shipping constraints.
- No fresh, clearly game-SR paper appeared in today's primary arXiv queries. The most relevant SR baselines remain LCS, game neural supersampling, AVSR-Diff, QuADA-GS, and Resonant Brane Splatting from prior briefs.

### Frame Extrapolation / Generation
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (Nengbo Lu, Minghua Pan; arXiv:2605.10586v1; already tracked) — Gaussian-state future prediction with physical causal factors. Why it matters to OSS: long-horizon OSS-FX regularization reference beyond α interpolation.
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (Nengbo Lu, Bin Zhao; arXiv:2605.10567v1; already tracked) — Per-Gaussian velocity fields with global physical consistency. Why it matters to OSS: comparator for v7 time-axis/cross-covariance motion representation and motion losses.
- **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (Songyin Wu et al.; arXiv:2406.18551v1; already tracked) — Standalone low-latency future-frame predictor without G-buffers. Why it matters to OSS: baseline/competitor for OSS-FX positioning; OSS should stress exact engine MVs + persistent Gaussian canvas versus separate extrapolator.
- No genuinely new frame-extrapolation paper surfaced today beyond already-tracked leads.

### Inference Optimization
- **ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering** (Lufei Liu, Tor M. Aamodt; arXiv:2506.13814v1; already tracked) — Reuses intermediate NN features across temporally coherent rendering frames. Why it matters to OSS: candidate optimization for teacher/student inference under high frame coherence.
- **Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution** (Yong Liu et al.; arXiv:2308.02794v1; rediscovered) — Efficient SR transformer unit for deployment. Why it matters to OSS: old but relevant for simplifying HAT-like teacher blocks or designing a student-friendly distillation target.
- **A High-Performance Accelerator for Super-Resolution Processing on Embedded GPU** (Wenqian Zhao et al.; arXiv:2303.08999v1; rediscovered) — Full-stack SR acceleration on embedded GPU. Why it matters to OSS: useful embedded/handheld prior for Pico-tier budgets, though model assumptions likely differ.
- No fresh TensorRT/INT8/kernel-fusion SR paper appeared today. Inference lesson is unchanged: the ≤1M-param student needs QAT/FP8-INT8 validation and fused custom kernels, not nominal export-only performance claims.

### Game Engine / Motion-Vector Integration
- **Compressed-Domain-Aware Online Video Super-Resolution** (Yuhang Wang et al.; arXiv:2603.07694v1; already tracked) — Uses codec motion vectors, residuals, and frame types to reduce online VSR alignment cost. Why it matters to OSS: codec MVs are weaker than engine MVs, but the gating/alignment pattern transfers to exact-MV reuse and unreliable-region handling.
- **Efficient neural supersampling on a novel gaming dataset** (Antoine Mercier et al.; arXiv:2308.01483v1; already tracked) — Game-content neural supersampling dataset/baseline. Why it matters to OSS: still one of the core non-vendor game-SR comparison points.
- No new game-engine integration or motion-vector game-upscaler paper surfaced in primary arXiv today.

## Archive.org Findings
- `"gaussian splatting" super-resolution`: 3 Archive.org hits, all broad GitHub/star-list captures; no useful primary paper mirror.
- `"frame generation" "deep learning"`: 4 broad hits including OSF MoCha-LD registration, DLSS Swapper Windows builds, FluidFrames older builds, and unrelated/game captures. Useful ecosystem archaeology only.
- `"real-time upscaling" game`: 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16,611 hits but polluted by unrelated “grape” content; broad acronym search remains unusable.
- Targeted Wayback CDX checks for GS-STVSR/GSASR/GRAPE URLs failed with Archive.org 503/timeouts this run, so absence is not evidence.

## GitHub / Open Source
- **GiulioFede/ResonantBraneSplatting** — pushed 2026-07-01, Python, 0 stars, no license. Opportunity: higher-order primitive/raster idea for v7 edge/detail quality; blocker: no license, treat as paper/idea only. <https://github.com/GiulioFede/ResonantBraneSplatting>
- **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** — pushed 2026-07-02, Python, MIT, 4 stars. Static structural analysis of AMD FSR 4.1.0 neural upscaler: DXIL entrypoints, weight extraction, blob layout, rebuild comparison. Opportunity: issue #7 reconnaissance for FSR4 teacher-signal/operator layout if claims verify. <https://github.com/Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling>
- **optiscaler/OptiScaler** — 9,119 stars, pushed 2026-07-07. Bridges DLSS2+/XeSS/FSR2+ inputs and enables FSR-FG/XeFG paths on non-FG titles. Threat/opportunity: integration-layer momentum is strong; OSS's DLL-shim path should expect users to compare it against OptiScaler-style substitution UX. <https://github.com/optiscaler/OptiScaler>
- **interceptballast918/optiscaler-client** and **pvjkoigdhi/Optiscaler-Client** — small Windows client repos pushed today; likely wrappers around OptiScaler, not core upscaling technology.
- **cloudy-sky-gold/neural-upscaler-x4-page/docs-page** — landing/docs pages only from metadata; ignore unless a real model/tool repository appears.
- `"frame generation" "open source" pushed:>=2026-06-24` returned no repository hits.

## News
- **NVIDIA / DLSS 4.5:** Primary NVIDIA developer page says DLSS 4 introduced Multi Frame Generation and transformer SR; **DLSS 4.5** adds **Dynamic Multi Frame Generation** and a **second-generation transformer model**. It also says DLSS MFG can generate up to five frames per rendered frame on RTX 50/Blackwell-class hardware and DLSS 4.5 is available through the Unreal Engine 5 plugin. Threat: NVIDIA is extending both quality messaging and FG multiplier while OSS is still pre-student.
- **AMD / FSR 4.1:** Primary GPUOpen page and SDK 2.3 article say **FSR Upscaling 4.1.1** now supports **RDNA 3 / Radeon RX 7000** discrete GPUs, previously RDNA 4-only; release also includes **FSR Frame Generation 4.0.1** and **Ray Regeneration 1.2.0**. Threat/opportunity: AMD is broadening ML-upscaler reach beyond newest GPUs, increasing pressure on OSS Pico/Standard positioning, but validating the cross-vendor ML reconstruction market.
- **AMD Redstone:** GPUOpen front page positions Redstone as a neural rendering suite: ML upscaling, ML frame generation, ray regeneration, and radiance caching. Competitive implication: vendor stacks are converging toward multi-feature neural-rendering bundles; OSS's “one architecture for SR+FX” needs clear scope control and measured quality/runtime evidence.
- **Intel / XeSS:** Google News RSS yielded mostly unrelated Arc/benchmark chatter; Intel XeSS primary page fetch returned HTTP 403. No verified XeSS technical update today.
- **Apple / Metal:** Apple Metal page still advertises MetalFX Upscaling, Frame Interpolation, and Denoising. No verified new API delta, but Apple continues to make ML graphics first-class; keep ANE/Metal student inference on the long-range backend watchlist.
- **Game engines:** NVIDIA page explicitly states DLSS 4.5 is in Unreal Engine through the official DLSS UE plugin. No actionable Unity upscaling API update surfaced; Unreal/Unity RSS hits were mostly benchmark/article noise.

## Action Items / Follow-ups
- [ ] CRITICAL ops: trainer remains offline/stuck at step 30,300 for 56h and CF Worker now reports offline/URLError; check trainer process, upload worker, and dashboard health-state semantics.
- [ ] Deep-read **TemporalGS** and **FreeTimeGS++** together for v7 temporal cache/active-mask/density-duration choices.
- [ ] Skim **GUSH3R** only if v7 needs transient-object/person dynamic-Gaussian separation; otherwise keep it low priority.
- [ ] Verify **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** claims before using it for issue #7; do not assume extracted FSR4 structure is accurate without reproducing.
- [ ] Watch AMD FSR SDK 2.3 / FSR 4.1.1 RDNA3 adoption as a competitor-positioning threat for OSS Standard and handheld/older-GPU messaging.
- [ ] Track DLSS 4.5 Dynamic MFG and UE5 plugin messaging; OSS public docs should avoid overclaiming until student + OSS-FX runtime evidence exists.
