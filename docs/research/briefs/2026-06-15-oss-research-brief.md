# OSS Research Brief — 2026-06-15

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
- README snapshot: v7 remains the current architecture: N-D Gaussian canvas with native time axis, `V_xt`/time-aware covariance, parent-child loss-adaptive density, Mip-Splatting-style anti-aliasing, α=0.5 OSS-FX supervision, and a future ≤1M-param distilled student. README still warns that live dashboard runs are research/teacher models, not the shipping inference path.
- Training dashboard: rendered `https://opensupersampling.com` redirects to `.org`. Visible state is **v7.0 Pico OSS-FX** at step **30,300 / 100,000**, loss **0.18269**, viz @ step **26,000**, ETA ~1.2d, cost **$0.76 / 12.63 GPU-hours / 5.05 kWh**. Important contradiction: headline and run card still say `v7 is training` / `live`, but service-health tooltip says **Trainer — offline, stuck at step 30,300 for 33h**. Watcher/CF Worker/R2 are healthy; GPU stats unavailable (`host offline or rsync failed`).

## New Papers (arXiv)

Primary arXiv API status: a first structured pass completed but returned no new high-signal records beyond already-tracked items; a second broader pass hit arXiv **429/timeouts** on all follow-up queries. Treat today’s arXiv section as **incremental monitoring**, not exhaustive proof of absence. No `BIBLIOGRAPHY.md` append is warranted from today’s paper scan.

### Gaussian Splatting / Temporal
- **Flex4DHuman: Flexible Multi-view Video Diffusion for 4D Human Reconstruction** (arXiv:2606.13655v1, 2026-06-11; Jen-Hao Cheng, Yipeng Wang, Hao Zhang, Gengshan Yang, Jenq-Neng Hwang) — Sparse/monocular video → dense multi-view video → 4D human reconstruction. Why it matters to OSS: indirect evidence that 4D Gaussian/video pipelines continue to standardize around dense temporal/view conditioning, but it is reconstruction/generation scale, not real-time game SR.
- **Envision4D: Envisioning Visual Futures via Feed-forward 4D Gaussian Splatting for Autonomous Driving** (arXiv:2606.10656v1, 2026-06-09; Qi Song et al.) — Feed-forward 4DGS predicts future dynamic scenes with temporal attention and pose prediction. Why it matters to OSS: already tracked; supports Gaussian primitives as future-frame substrates, while OSS has stronger privileged motion via engine vectors.
- **Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting** (arXiv:2606.07670v1, 2026-06-04; Mingzhao Li, Arghya Pal, Guan Yuan Tan) — Continuous-time liquid/CfC deformation field for dynamic 3DGS. Why it matters to OSS: already tracked; possible residual-motion prior if direct `V_xt` covariance underfits.
- **PersistGS: Differentiable Physics for Object Permanence in 4D Gaussian Splatting** (arXiv:2606.03479v1, 2026-06-02; Adrian Ramlal, John S. Zelek) — Rigid-body simulation for plausible occluded object Gaussians. Why it matters to OSS: already tracked; relevant to stale-splat handling and persistent-canvas object permanence.

### Super-Resolution / Upscaling
- **EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution** (arXiv:2606.13580v1, 2026-06-11; Dachun Kai, Jiayao Lu, Yueyi Zhang, Xiaoyan Sun) — Uses event signals for high-frequency texture enhancement in VSR. Why it matters to OSS: no event-camera input in games, but its alignment-vs-texture split maps to engine-MV/G-buffer alignment plus confidence-gated detail recovery.
- **ConFi-GS: Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution** (arXiv:2605.24964v1, 2026-05-24; Jiaxiang Li, Zongtan Zhou, Zhen Tan, Yadong Liu, Dewen Hu) — Reliability-aware high-frequency injection for GS-SR. Why it matters to OSS: already tracked; candidate confidence gate for splat spawning/densification and pseudo-label filtering.
- **Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution** (arXiv:2605.22147v1, 2026-05-21; Jiangwei Mo, Xi Lu, Hanlin Wu) — 2DGS continuous feature field for arbitrary-scale SR. Why it matters to OSS: already tracked; supports continuous-scale 2D-GS representation despite remote-sensing domain mismatch.
- **SlimDiffSR: Toward Lightweight and Efficient Remote Sensing Image Super-Resolution via Diffusion Model Distillation** (arXiv:2605.02198v2, 2026-05-04; Ce Wang, Zhenyu Hu, Wanjie Sun) — Distills diffusion SR into lightweight remote-sensing SR. Why it matters to OSS: already tracked; useful teacher→student compression pattern, not a shipping-game runtime design.

### Frame Extrapolation / Generation
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567v1, 2026-05-11; Nengbo Lu, Bin Zhao) — Learns per-Gaussian velocity fields and physics constraints from videos. Why it matters to OSS: already tracked; top deep-read candidate for v7 `V_xt`, `t_query`, velocity losses, and OSS-FX supervision.
- **3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** (arXiv:2605.11489v1, 2026-05-12; Zhao et al.) — Joint post-render 3DGS supersampling + frame interpolation. Why it matters to OSS: already tracked; closest competitive-paper framing to OSS’s unified SR+FG claim after GS-STVSR.
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (arXiv:2603.26835v3, 2026-03-27; Shibo Liu) — Uses codec motion vectors to remove learned optical flow from mobile VFI. Why it matters to OSS: already tracked; strong analogue for deleting expensive flow/FG networks using privileged motion vectors.

### Inference Optimization
- No new high-signal TensorRT/INT8/kernel-fusion SR papers surfaced in the completed scan.
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (arXiv:2604.20291v1; Pham Phuong Nam Nguyen et al.) remains the most direct student-deployment recipe: LR-space student, reparameterizable deploy graph, DCT supervision, and confidence-weighted QAT.
- **QuantSR+** and clip-free representative calibration remain relevant for treating low-bit deployment as a training-time constraint rather than a post-export patch.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 hits, all broad/noisy archived GitHub/star-list or index pages; no primary paper snapshot found.
- `"frame generation" "deep learning"`: 4 hits: FluidFrames Windows builds, DLSS Swapper Windows builds, MoCha-LD OSF registration, and a Minecraft/Radiance archive. Ecosystem breadcrumbs only.
- `"real-time upscaling" game`: 0 hits.
- Targeted Wayback checks:
  - `https://arxiv.org/abs/2604.18047` (GS-STVSR): timed out today.
  - `https://arxiv.org/abs/2602.24020` (SR3R): no 200 captures returned.
  - `https://arxiv.org/abs/2511.12040` (SRSplat): 2 captures found (`20251118043846`, `20260130074754`).
  - `https://arxiv.org/abs/2605.11489` (3DGS³): 1 capture found (`20260607145837`).
  - `https://github.com/ChrisDud0257/GSASR` and `https://github.com/jyeon009/GRAPE`: CDX timed out; retry later with per-URL backoff.

## GitHub / Open Source
- `ShiinaManatsu/dlss-compositor` — **new lead**; describes itself as an open-source C++ tool for processing offline Blender Cycles renders through NVIDIA DLSS Ray Reconstruction and DLSS Frame Generation; pushed 2026-06-15; 3 stars. Why it matters: not a game-runtime upscaler, but potentially useful for understanding DLSS-FG/RR integration surfaces and artifact behavior in controlled offline render sequences. https://github.com/ShiinaManatsu/dlss-compositor
- `optiscaler/OptiScaler` — active cross-upscaler bridge; pushed 2026-06-15; 8.7k stars. Supports DLSS2+/XeSS/FSR2+ input replacement, FSR-FG/XeFG on non-FG titles, Nukem DLSSG-to-FSR3 FG path. Why it matters: integration ecosystem is the competitive and eventual OSS-shim surface; monitor APIs/hooks. https://github.com/optiscaler/OptiScaler
- `hvcl/3DCT_SR_DDNM_NAB-GS` — zero-shot 3D CT SR with diffusion 2D projection priors and Negative Alpha Blending Gaussian Splatting; pushed 2026-06-12; 1 star. Medical domain; NAB/confidence blending may be conceptually useful for unreliable detail injection. https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS
- `Adam-AtlasSoftware/RDNU` — Radeon Decoupled Neural Upscaler; last pushed 2026-06-09; 16 stars. **Threat/opportunity:** RDNA3-focused, G-buffer-guided neural upscaler with public training-data claims; still needs code/weights/license seriousness check. https://github.com/Adam-AtlasSoftware/RDNU
- `senjul07/optiscaler-client-fsr4`, `interceptballast918/optiscaler-client`, `Streampogrow/lsfg-git` — SEO/noisy wrapper repos around OptiScaler/FSR4/Lossless Scaling FG. Monitor only for real code; do not treat as competitors.
- No new high-signal DLSS/FSR/XeSS analysis-tool repos found beyond OptiScaler and the `dlss-compositor` lead.

## News
- Web search/extraction backend failed with Firecrawl `Payment Required`, so this section uses Google News RSS headline monitoring and should be treated as lead-level unless a primary page is cited.
- **NVIDIA / DLSS:** Google News RSS surfaced NVIDIA’s `CONTROL Resonant` launch messaging with **DLSS 4.5 + path tracing**, plus Subnautica 2 DLSS 4.5 support coverage. **Threat:** NVIDIA’s marketing/quality story continues moving toward transformer/Ray Reconstruction/plugin-led integration; OSS needs visible demos/metrics, not only architecture claims.
- **AMD / FSR:** RSS surfaced **AMD FSR 4.1** coverage claiming RDNA 3 support / quality-parity messaging with RDNA 4, plus `The Outlast Trials` FSR 4 support discovery. **Threat/opportunity:** if FSR 4-class ML upscaling expands beyond RDNA4, vendor-neutral OSS must differentiate on openness, unified SR+FG, and cross-engine training data rather than “AMD lacks ML SR.”
- **Intel / XeSS / Arc:** RSS surfaced Arc G3 handheld benchmark/processor coverage and low-signal DLSS/XeSS/FSR comparison content. **Signal:** handheld/multi-vendor upscaling benchmarks remain consumer-visible; OSS Pico/Steam Deck claims need measured numbers before public use.
- **Apple / Metal:** RSS surfaced Apple’s development-tool announcements, Game Porting Toolkit 4 coverage, and Cyberpunk 2077-on-Mac performance commentary. **Opportunity:** Metal backend credibility matters as Mac gaming grows; OSS currently has Metal designed-only and should not imply parity.
- **Game engines:** RSS returned no strong Unreal/Unity upscaling headlines today. Continue monitoring UE/Unity release notes and vendor plugins.

## Action Items / Follow-ups
- [ ] Ops: investigate v7 trainer. Dashboard shows **no visible progress since yesterday** and now reports trainer offline/stuck at step **30,300 for 33h** while Watcher/CF/R2 are healthy.
- [ ] Deep-read `VeloGauss`, `PersistGS`, `R5DGS`, `Envision4D`, and `3DGS³` for v7 native-time, velocity, object permanence, rigid/transient decomposition, and unified SR+FG positioning.
- [ ] Inspect `ShiinaManatsu/dlss-compositor` for actual DLSS-FG/RR invocation path and whether it gives useful offline artifact/test harness ideas.
- [ ] Inspect `RDNU` for runnable code, weights, license, datasets, and whether it deserves a formal baseline/competitor note.
- [ ] Retry arXiv broad queries after cooldown; today’s follow-up pass hit 429/timeouts.
- [ ] Retry targeted Wayback CDX checks for GS-STVSR, GSASR, and GRAPE with per-URL backoff.
- [ ] Keep QAT/deploy-graph work front-loaded: `Efficient INT8 SISR`, `QuantSR+`, and clip-free PTQ all argue against treating precision as an afterthought.

## Bibliography Update
- No `BIBLIOGRAPHY.md` append this run. Today surfaced no new high-impact paper not already present or already identified in the 2026-06-08 through 2026-06-14 briefs.
