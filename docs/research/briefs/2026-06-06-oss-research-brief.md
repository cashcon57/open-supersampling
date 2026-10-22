# OSS Research Brief — 2026-06-06

## Project Status Summary
- Latest commits:
- `a97a180` feat: add tailnet-gated training controls to dashboard
- `fcf8084` feat: tailnet-gated training control API + dashboard panel
- `0660f55` wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc
- `56a9a65` trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)
- `d1255c8` trainer: WindowsPath shim on Linux for cross-OS checkpoint resume
- `e1ab7fe` hat.py: use flash-attn 2 directly when available, fallback to SDPA
- `e9b4b65` trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)
- `26cd9da` dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB
- `83b4f07` v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast
- `baf4b02` worker: apex .com → .org 301 redirect
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
- README snapshot: repo still presents OSS as a vendor-neutral, Apache-2.0 SR + frame-extrapolation system using one persistent Gaussian canvas. Current public status is v7 research/teacher training (`srcnn-v7.0-pico-005`), with ≤1M-param distilled student not yet trained.
- Training dashboard: `srcnn-v7.0-pico-005` / **v7.0 Pico OSS-FX** is at **17,700 / 100,000** steps, loss **0.10913**, SR Charbonnier **0.01502**. Notable issue: trainer is **offline/stuck at step 17,700 for ~19h**; watcher, CF Worker, and R2 origin are healthy. Dashboard is still waiting for `score_log_v7.json`, so the +1 dB OSS-FX-vs-bicubic gate is not measured yet.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047, Shi et al., 2026-04-20) — 2D-GS C-STVSR with optical-flow-guided Gaussian motion, covariance resampling alignment, and near-constant inference across X2-X8 temporal scales. Why it matters to OSS: this remains the closest public prior for OSS-FX’s single Gaussian representation doing both spatial SR and temporal synthesis; their covariance-drift handling is the thing to re-check against v7’s native time-axis Cholesky.
- **ConFi-GS: Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution** (arXiv:2605.24964v1, Li et al., 2026-05-24) — separates “detail demand” from “detail reliability” before injecting high-frequency SR supervision into 3DGS. Why it matters to OSS: direct opportunity for v7 parent-child densification / loss-adaptive density — gate teacher/perceptual details by reliability instead of letting hallucinated high-frequency supervision become persistent canvas state.
- **SuperGS: Consistent and Detailed 3D Super-Resolution Scene Reconstruction via Gaussian Splatting** (arXiv:2505.18649, Xie et al., 2025-05-24) — two-stage LR→HR 3DGS with latent feature field, multi-view voting, uncertainty-aware pseudo-label weighting, and error-map-guided densification. Why it matters to OSS: useful uncertainty/densification recipe; less direct than GS-STVSR because it is scene-reconstruction, not streaming game SR.
- **SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting** (arXiv:2512.02172, Asthana et al., CVPR 2026) — selectively applies SR only in geometrically undersampled regions to avoid multi-view inconsistent hallucinations. Why it matters to OSS: same principle should apply to expensive v7 refinement / teacher losses — spend compute only where canvas evidence is weak.

### Super-Resolution / Upscaling
- **MaCo-GAN: Manifold-Contrastive Adversarial Learning for Single Image Super-Resolution** (arXiv:2606.05068, Han/Hwang/Jeon, 2026-06-03) — replaces vanilla GAN loss with a supervised contrastive minimax objective using LR-correspondent fake samples. Why it matters to OSS: possible perceptual-stage loss to reduce hallucinated textures while preserving LR faithfulness; relevant if v7/v8 adds GAN/perceptual fine-tuning after Charbonnier/LPIPS stabilization.
- **Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution / GSASR** (arXiv:2501.06838, Chen et al., ICCV 2025, revised 2025-07-30) — feed-forward prediction of image-conditioned 2D Gaussians plus custom CUDA scale-aware rasterizer. Why it matters to OSS: primary static-image GS SR baseline; repo activity continued into late May 2026.
- **Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling / ContinuousSR** (arXiv:2503.06617, Peng et al., 2025-03-09) — builds a 2D Gaussian field once, then renders arbitrary scales in ~1 ms per scale. Why it matters to OSS: strong student-model clue; compare its point-wise Gaussian prediction/rasterization path against OSS ≤1M-param student design.

### Frame Extrapolation / Generation
- **FramePack / FramePack-F1** (project/GitHub, not arXiv in this pass) — open-source next-frame / next-frame-section video diffusion system with a forward-only variant. Why it matters to OSS: not real-time game inference, but it is a public open-source “future-frame from history” competitor in the broader frame-generation narrative.
- No clearly new arXiv paper this run matched “single network SR + extrapolation for games” beyond GS-STVSR. That absence strengthens OSS positioning, but only if v7 gets measured.

### Inference Optimization
- No new OSS-direct TensorRT/INT8/kernel-fusion paper surfaced in this pass. The strongest optimization-relevant paper remains **ContinuousSR** for its “build field once, render per scale” runtime pattern.
- Watch AMD FSR 4 SDK docs/release notes instead of arXiv for practical ML inference details: public docs continue to mention ML upscaling, FSR 4.1 SDK, FP/INT paths, UE plugin, and fallback behavior.

## Archive.org Findings
- `"gaussian splatting" super-resolution` — github.com-pbsds-starred_-_2025-08-25_04-40-27 (`github.com-pbsds-starred_-_2025-08-25_04-40-27`, 2025-08-25T00:00:00Z); github.com-pbsds-starred_-_2025-10-03_04-36-09 (`github.com-pbsds-starred_-_2025-10-03_04-36-09`, 2025-10-03T00:00:00Z); github.com-Awesome-Image-Registration-Organization-awesome-image-registration_-_2025-08-20_11-05-40 (`github.com-Awesome-Image-Registration-Organization-awesome-image-registration_-_2025-08-20_11-05-40`, 2025-08-20T00:00:00Z)
- `"frame generation" deep learning` — [Must be compiled - Freemium] FluidFrames older builds for Windows (`fluidframeswin`, date n/a); DLSS Swapper builds for Windows (`dlssswapperbuildswin`, date n/a); MoCha-LD: Temporal-Chunked Latent Diffusion with Motion-Aware Consistency Regularization for Long Video Generation (`osf-registrations-nuxth-v1`, 2026-04-20T00:00:00Z)
- `"real-time upscaling" game` — no useful Archive.org records found.
- `GS-STVSR OR GSASR OR GRAPE OR STSS` — EPISODE44 - 'Will that be One Grape or Two?' (`dsxvszypiivixezxmwajbqduk1201o2mty4mjcwr`, 2012-06-26T00:00:00Z); Nazir 35 (`mfnfpgwdinwyqccvqgmi7thx2ilabhbem8h0pn2r`, 2023-02-26T00:00:00Z); Being visual : raising a generation of innovative thinkers (`beingvisualraisi0000fett`, 2012-01-01T00:00:00Z); This Is Mr. Captain Pa Grape Speaking (`this-is-mr.-captain-pa-grape-speaking`, date n/a); Oliver and Friends' Great Grape Adventure (`oliverfriendsgre0000nydi`, 2013-01-01T00:00:00Z)
- No credible archived/taken-down copy of GS-STVSR, GSASR, GRAPE, or STSS was found via Archive.org search. Current arXiv/GitHub sources are still available, so no recovery action needed.

## GitHub / Open Source
- `gaussian splatting super-resolution`: [hvcl/3DCT_SR_DDNM_NAB-GS](https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS) (0★, updated 2026-05-28) — Zero-shot 3D CT Super-Resolution using Diffusion-based 2D Projection Priors and Negative Alpha Blending Gaussian Splatting (NAB-GS); [NagasaiManda/SplatSuRe](https://github.com/NagasaiManda/SplatSuRe) (0★, updated 2026-04-26) — Official code for SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting; [ConnorBaker/blender-temp](https://github.com/ConnorBaker/blender-temp) (0★, updated 2026-03-18) — Messing around with Blender scenes, gaussian splatting, and super-resolution; [pranav-asthana/SplatSuRe](https://github.com/pranav-asthana/SplatSuRe) (75★, updated 2026-06-04) — Official code for SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting; [ChrisDud0257/GSASR](https://github.com/ChrisDud0257/GSASR) (98★, updated 2026-05-28) — Official PyTorch Code for our ICCV25 paper- Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution
- `frame generation open source`: [touhidsiddiqueeraj-bit/An-Open-Python-Framework-for-Battery-Operational-Reliability-Estimation](https://github.com/touhidsiddiqueeraj-bit/An-Open-Python-Framework-for-Battery-Operational-Reliability-Estimation) (0★, updated 2026-06-04) — Open-source Python implementation of multihorizon discrete-time hazard learning for battery energy storage systems. Reproduces and extends Shikdar & Laaksonen (2026) with modular code, synthetic data generation, and quantitative scaling guidelines.; [nova-studios-hw/novagpu-ts1t](https://github.com/nova-studios-hw/novagpu-ts1t) (11★, updated 2026-06-04) — N.E.O.N. Architecture — Open source GPU RTL in Verilog. Token dataflow, hardware ray tracing, frame generation. First silicon target: 28nm.; [khushboo1622/llm-evaluation-benchmarking-framework](https://github.com/khushboo1622/llm-evaluation-benchmarking-framework) (0★, updated 2026-05-29) — Automated LLM benchmarking and evaluation framework for testing open-source models across reasoning, coding, multilingual, safety, and structured generation tasks with performance, cost, and LLM-as-a-Judge analysis.; [nmako/human-ai-framework](https://github.com/nmako/human-ai-framework) (0★, updated 2026-05-22) — An open-source diagnostic and policy generation tool for integrating AI governance with Human Rights Due Diligence; [ShiinaManatsu/dlss-compositor](https://github.com/ShiinaManatsu/dlss-compositor) (3★, updated 2026-05-16) — First open-source tool for processing offline Blender Cycles renders through NVIDIA DLSS Ray Reconstruction (DLSS-RR) and DLSS Frame Generation (DLSS-FG)
- `neural upscaler`: [annitapetrocchi/Neural-Network-Upscaler](https://github.com/annitapetrocchi/Neural-Network-Upscaler) (0★, updated 2026-06-06) — AI-driven image upscaling utility using Generative Adversarial Networks (GANs).; [acnaninha572/Neural-Network-Upscaler](https://github.com/acnaninha572/Neural-Network-Upscaler) (0★, updated 2026-06-02) — AI-driven image upscaling utility using Generative Adversarial Networks (GANs).; [shepherdscientific/neural-spatial-upscaler](https://github.com/shepherdscientific/neural-spatial-upscaler) (0★, updated 2026-06-02) — TernaryCore Neural Spatial Upscaler Module — real-time 240p→1080p TMDS upscaling via ternary CNN kernels on Gowin Tang Nano 9K/20K FPGA fabric; [drabhichat/Neural-Network-Upscaler](https://github.com/drabhichat/Neural-Network-Upscaler) (0★, updated 2026-06-02) — AI-driven image upscaling utility using Generative Adversarial Networks (GANs).; [TobyDunn/Neural-Network-Upscaler](https://github.com/TobyDunn/Neural-Network-Upscaler) (0★, updated 2026-06-02) — AI-driven image upscaling utility using Generative Adversarial Networks (GANs).
- `DLSS FSR analysis tool` — no notable repos in API result.
- **OptiScaler** ([optiscaler/OptiScaler](https://github.com/optiscaler/OptiScaler)) is the most important open-source integration/analysis repo this run: it bridges DLSS2+/FSR2+/XeSS inputs, can replace native upscalers, and tracks FSR4 injection compatibility. Why it matters to OSS: validates the DLL-shim/modding route and is a practical compatibility-test target/threat.
- **GSASR** ([ChrisDud0257/GSASR](https://github.com/ChrisDud0257/GSASR)) is active enough to keep as a benchmark target (98★, updated 2026-05-28 in GitHub API result).
- **SplatSuRe** ([pranav-asthana/SplatSuRe](https://github.com/pranav-asthana/SplatSuRe)) is active and now CVPR 2026; the selective-SR idea is relevant to canvas reliability gating.
- **dlss-compositor** ([ShiinaManatsu/dlss-compositor](https://github.com/ShiinaManatsu/dlss-compositor)) is niche but interesting: offline Blender Cycles through DLSS-RR/DLSS-FG. Why it matters: public tooling around proprietary frame-gen paths is expanding.

## News
- **NVIDIA DLSS 4 developments past 14 days:** [Nvidia introduces DLSS 4.5 and Multi Frame Generation 6X at CES ...](https://www.tomshardware.com/pc-components/cpus/nvidia-introduces-dlss-4-5-and-multi-frame-generation-6x-at-ces-2026-updated-models-can-generate-higher-quality-upscaled-frames-and-more-of-them-dynamically) — Nvidia introduces DLSS 4.5 and Multi Frame Generation 6X at CES 2026 — updated models can generate higher-quality upscaled frames and more of ...
- **AMD FSR 4 developments past 14 days:** [AMD Caves, Extends FSR 4 to RX 7000 & 6000 GPUs ... - YouTube](https://www.youtube.com/watch?v=Jq9q5HQ3azs) — AMD finally announces FSR 4.1 coming to RX 7000 and RX 6000 series GPUs. Are these now the best used GPU in 2026? We look at FSR 4.1 Int8 ...
- **Intel XeSS Arc updates:** [Intel confirms XeSS 3 multi-frame generation support for Arc B580 ...](https://www.dlcompare.com/gaming-news/intel-confirms-xess-3-multi-frame-generation-support-for-arc-b580-gpu-soon-71092) — Intel has now confirmed that XeSS 3 with Multi-Frame Generation will arrive on the Intel Arc B580 graphics card this month.
- **Apple Metal performance updates:** [Apple unleashes M5, the next big leap in AI performance for Apple ...](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/) — M5 unleashes next-level AI performance with its next-generation GPU architecture, powerful CPU, faster Neural Engine, and higher unified memory ...
- **GPU announcements:** [CES 2026: 8 Big Chip Announcements By Intel, Nvidia, AMD And ...](https://www.crn.com/news/components-peripherals/2026/ces-2026-8-big-chip-announcements-by-intel-nvidia-amd-and-qualcomm) — CES 2026: 8 Big Chip Announcements By Intel, Nvidia, AMD And Qualcomm · Intel Core Ultra Series 3 · AMD Ryzen AI 400 Series · AMD Ryzen AI Max ...
- **Unreal Unity upscaling updates:** [List of games that support high-fidelity upscaling - PCGamingWiki](https://www.pcgamingwiki.com/wiki/List_of_games_that_support_high-fidelity_upscaling) — is limited to FSR and XeSS, DLSS 4 likely updated on 1.7 / 10 April Update ... NVIDIA DLSS plugin for Unreal Engine 5 is now available, also coming to Unity ...
- **Threat:** NVIDIA DLSS 4.5 claims a 2nd-gen transformer SR model, linear-space inference/training, 6X Dynamic Multi Frame Generation, and broad override support. Competitive bar moved upward on quality and frame-gen multiplicity.
- **Threat/opportunity:** AMD FSR 4/4.1 is public on GPUOpen with UE plugin support and ML upscaling docs. This is both a benchmark threat and a source of integration/API lessons.
- **Threat:** Intel XeSS 3 developer docs now frame XeSS as SR + FG + low-latency + Multi Frame Generation, including up to 3 generated frames on Intel hardware. Vendor stacks are converging on bundled SR/FG/latency suites.
- **Opportunity:** OptiScaler/compatibility-list ecosystem shows developers and modders tolerate drop-in upscaler replacement; OSS should target that path early.

## Action Items / Follow-ups
- [ ] Deep-read **ConFi-GS** and extract a concrete reliability-gated densification/loss recipe for v7 parent-child density.
- [ ] Compare **ContinuousSR / GRAPE / GSASR** against the ≤1M-param student target; identify whether point-wise Gaussian prediction can replace or simplify the planned nano-CNN student.
- [ ] Add an automated dashboard alert: trainer stuck/offline for >30 min should be surfaced separately from watcher/R2 health.
- [ ] Benchmark/inspect **OptiScaler** compatibility and DLL-shim assumptions against OSS runtime plans.
- [ ] Watch NVIDIA DLSS 4.5 linear-space model claims and AMD FSR 4.1 SDK details; both may affect OSS training color-space and inference-kernel decisions.
