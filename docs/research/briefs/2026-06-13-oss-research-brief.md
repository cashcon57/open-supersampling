# OSS Research Brief — 2026-06-13

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
- README snapshot: v7 remains the current architecture/run: `srcnn-v7.0-pico-005`, N-D Gaussian canvas with native time axis and OSS-FX `t_query`; ≤1M-param distilled student is not trained; cross-vendor kernels remain designed-only.
- Training dashboard: rendered `https://opensupersampling.com` redirects to `.org`. Visible state: **v7.0 Pico OSS-FX** at step **25,750 / 100,000**, loss **0.17147**, viz @ step 25,000, ETA ~1.3d. Service health says Watcher/CF Worker/R2 healthy, but Trainer is **degraded / stuck at step 25,750 for 5m** and GPU stats unavailable (`host offline or rsync failed`). Compared with 2026-06-12, the run advanced +2,500 steps but still shows a short trainer-stuck warning.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **Flex4DHuman: Flexible Multi-view Video Diffusion for 4D Human Reconstruction** (arXiv:2606.13655v1, 2026-06-11; authors not fully expanded in brief pass) — Multi-view video diffusion for 4D human reconstruction. Why it matters to OSS: low direct game-SR relevance, but another signal that diffusion+4D reconstruction pipelines are leaning into temporal consistency and dynamic scene priors; not a bibliography candidate unless it contains reusable 4D temporal regularization.
- **WorldOlympiad: Can Your World Model Survive a Triathlon?** (arXiv:2606.11129v1, 2026-06-09) — Dynamic/world-model benchmark surfaced by 4D/dynamic queries. Why it matters to OSS: mostly not relevant to architecture; maybe useful only as a stress-test framing for future OSS-FX temporal prediction claims.
- **ManiSplat: Manipulation Trajectory Synthesis from Monocular Video via Decoupled 3D Gaussian Splatting** (arXiv:2606.10645v1, 2026-06-09; Wenhao Hu, Haonan Zhou, Liu Liu, Yun Du, Xinjie Wang et al.) — Decouples dynamic manipulation trajectories with 3DGS. Why it matters to OSS: already surfaced on 2026-06-10; still relevant as weak support for separating static/dynamic/transient Gaussian channels.
- **Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting** (arXiv:2606.07670v1, 2026-06-04; Mingzhao Li, Arghya Pal, Guan Yuan Tan) — Continuous-time/Liquid-cell deformation field for dynamic 3DGS. Why it matters to OSS: already in recent bibliography; high-value v7 `t_query` / residual-motion reference.
- **ConFi-GS: Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution** (arXiv:2605.24964v1, 2026-05-24; Jiaxiang Li et al.) — Confidence-guided high-frequency injection for GS-SR. Why it matters to OSS: candidate confidence gate for detail injection/spawning; already tracked.
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047v1, 2026-04-20; Shi et al.) — Core 2D-GS spatio-temporal VSR prior. Why it matters to OSS: still the closest primary covariance-resampling precedent; no new version found in this run.

### Super-Resolution / Upscaling
- **EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution** (arXiv:2606.13580v1, 2026-06-11; Dachun Kai, Jiayao Lu, Yueyi Zhang, Xiaoyan Sun) — Event-driven texture enhancement for VSR. Why it matters to OSS: not deployable in games without event input, but it supports the idea of explicitly separating high-frequency temporal texture recovery from bulk motion alignment; already surfaced 2026-06-12.
- **SlimDiffSR: Toward Lightweight and Efficient Remote Sensing Image Super-Resolution via Diffusion Model Distillation** (arXiv:2605.02198v2, 2026-05-04; Ce Wang, Zhenyu Hu, Wanjie Sun) — Lightweight SR diffusion distillation. Why it matters to OSS: teacher/student distillation reference only; remote-sensing domain and diffusion runtime are not a shipping path.
- **MambaLiteSR: Image Super-Resolution with Low-Rank Mamba using Knowledge Distillation** (arXiv:2502.14090v1, 2025-02-19) — Low-rank state-space SR with KD. Why it matters to OSS: possible student-backbone comparison if CNN/RepVGG-lite underperforms, but lower priority than INT8/QAT and reparameterized CNN references.
- **Neural Super-Resolution for Real-time Rendering with Radiance Demodulation** (arXiv:2308.06699v2, 2023-08-13; Jia Li et al.) — Real-time rendering SR using radiance demodulation. Why it matters to OSS: reinforces G-buffer/material-channel conditioning instead of RGB-only reconstruction.

### Frame Extrapolation / Generation
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567v1, 2026-05-11; Nengbo Lu, Bin Zhao) — Per-Gaussian velocity fields with physical consistency. Why it matters to OSS: already added 2026-06-12; remains a high-priority deep read for v7 `V_xt` / velocity loss design.
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586v1, 2026-05-11; Nengbo Lu, Minghua Pan) — Gaussian dynamics with causal prediction. Why it matters to OSS: supports per-Gaussian future-state prediction but is less directly tied to exact engine motion vectors.
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (arXiv:2603.26835v3, 2026-03-27; Shibo Liu) — Frame interpolation using codec MV priors. Why it matters to OSS: strong analogue to OSS using engine MVs as privileged low-cost temporal signal; already tracked.
- **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (arXiv:2406.18551v1, 2024-05-23) — Low-latency frame extrapolation without G-buffers. Why it matters to OSS: useful competitor/baseline framing; OSS can contrast against stronger engine-supplied G-buffer/MV inputs.
- **RSTT: Real-time Spatial Temporal Transformer for Space-Time Video Super-Resolution** (arXiv:2203.14186v1, 2022-03-27) — Older single-model space-time VSR. Why it matters to OSS: historical baseline for unified spatial+temporal upsampling, without Gaussian canvas economics.

### Inference Optimization
- **Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution** (arXiv:2308.02794v1, 2023-08-05; Yong Liu et al.) — TensorRT/deployment-friendly transformer unit. Why it matters to OSS: relevant to HAT teacher simplification and export-aware student architecture.
- **NAWQ-SR: A Hybrid-Precision NPU Engine for Efficient On-Device Super-Resolution** (arXiv:2212.09501v3, 2022-12-15; Stylianos I. Venieris et al.) — Hybrid-precision SR engine. Why it matters to OSS: reinforces precision policy as an architecture constraint, not a post-training afterthought.
- **Efficient Integer-Arithmetic-Only Convolutional Neural Networks** (arXiv:2006.11735v1, 2020-06-21; Pramod Gopalakrishnan, Michael Andersch) — Integer-only CNN inference. Why it matters to OSS: fallback mental model for non-FP8 targets and cross-vendor INT8 student deployment.
- **Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers** (arXiv:2604.02344v1, 2026-02-09; Jędrzej Maczan) — Dispatch overhead study across vendors/backends. Why it matters to OSS: <2ms runtime can be destroyed by dispatch count; strengthens the kernel-fusion/megakernel action item.

### Game Engine / Motion Vectors
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694v1, 2026-03-08; Yuhang Wang et al.) — Online VSR using codec MVs/residual/frame-type signals. Why it matters to OSS: public analogue for reusing external motion/residual metadata instead of estimating optical flow.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483v1, 2023-08-03; Antoine Mercier et al.) — Game-specific neural supersampling dataset/baseline. Why it matters to OSS: still one of the most relevant public game-SR references for eval and baseline framing.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: Archive.org advanced search returned 3 broad/noisy archived GitHub/star-list pages; no primary paper snapshot found.
- `"frame generation" "deep learning"`: 3 hits: MoCha-LD OSF registration plus FluidFrames/DLSS Swapper Windows build archives. Ecosystem breadcrumbs only, not OSS architecture inputs.
- `"real-time upscaling" game`: 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16,566 hits, dominated by acronym/common-word noise (especially `GRAPE`); not useful as broad query evidence.
- Targeted Wayback CDX follow-up attempted for known arXiv/GitHub URLs (`GS-STVSR`, `GSASR`, `GRAPE`, `4DGS-1K`, `RDNU`) but timed out at the Archive.org API layer. Treat as inconclusive, not absence of archived copies.

## GitHub / Open Source
- `Sungpyokim97/SR3R` — unofficial SR3R repo updated/pushed 2026-06-13; 1 star. SR3R is already OSS-relevant feed-forward Gaussian SR, but this repo still looks like a low-signal unofficial mirror. https://github.com/Sungpyokim97/SR3R
- `hvcl/3DCT_SR_DDNM_NAB-GS` — zero-shot 3D CT SR using diffusion 2D projection priors + Negative Alpha Blending Gaussian Splatting; pushed 2026-06-12. Medical-domain, but NAB/confidence blending may be conceptually useful for unreliable detail injection. https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS
- `Adam-AtlasSoftware/RDNU` — Radeon Decoupled Neural Upscaler; pushed 2026-06-09, now 16 stars. **Threat/opportunity:** remains the most relevant open-source competitor signal: RDNA3-focused, G-buffer-guided neural upscaling, datasets overlapping game/virtual data. Needs code/weights/license seriousness check. https://github.com/Adam-AtlasSoftware/RDNU
- `nova-studios-hw/novagpu-ts1t` — open-source GPU RTL mentioning frame generation; low direct OSS software relevance.
- No new high-signal DLSS/FSR analysis-tool repos found in the GitHub REST search window.
- Generic `web_search` failed with Firecrawl payment/credit exhaustion; GitHub REST, arXiv Atom, Archive.org APIs, Google News RSS, and browser rendering were used instead.

## News
- **NVIDIA / DLSS 4.5:** Google News RSS surfaced NVIDIA items for DLSS 4.5 Ray Reconstruction, over-1000 RTX games/apps messaging, and `CONTROL Resonant` launching with DLSS 4.5/path tracing. **Threat:** NVIDIA continues to move the competitor bar on transformer reconstruction, plugin integration, and marketing cadence.
- **AMD / FSR 4:** RSS surfaced `The Outlast Trials` FSR 4 support discovery and Pax Dei UE 5.6 coverage. **Opportunity/threat:** FSR 4 support expansion appears incremental; OSS can still position around openness/vendor neutrality, but RDNU-like AMD-focused open efforts may claim some of that mindshare.
- **Intel / XeSS:** RSS again surfaced XeSS 3 Multi Frame Generation in Death Stranding 2 coverage plus Arc G-series/handheld GPU coverage. **Threat:** multi-frame generation is becoming standard vendor positioning; OSS-FX needs measured demo output, not just architectural narrative.
- **Apple / Metal:** RSS showed M4 GPU benchmark/performance and Cyberpunk-on-Mac coverage. **Opportunity:** Metal backend value continues rising for game upscaling credibility; OSS still has no Metal kernels.
- **Game engines:** No new high-signal Unreal/Unity upscaling-specific primary source found this run beyond existing DLSS/FSR plugin-level momentum.

## Action Items / Follow-ups
- [ ] Ops: check trainer if dashboard remains stuck/degraded at step 25,750; it advanced since yesterday but still emits a stuck warning.
- [ ] Deep-read `VeloGauss`, `Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field`, and `PersistGS` for v7 time-axis / velocity / occlusion priors.
- [ ] Inspect `Adam-AtlasSoftware/RDNU` for runnable code/weights, license, DirectML/RDNA assumptions, and whether it should become a formal baseline.
- [ ] Add a targeted Archive.org/Wayback retry with longer timeout or batched URLs; broad acronym searches are too noisy.
- [ ] Keep kernel-fusion / dispatch-count planning front-loaded for the ≤1M-param student; dispatch overhead may dominate even if math ops fit the <2ms budget.

## Bibliography Update
- No new `BIBLIOGRAPHY.md` append this run. The materially relevant findings are already present in recent bibliography additions (`Liquid-D3DGS`, `VeloGauss`, `ConFi-GS`, `ANVIL`, `RDNU`, INT8/QAT refs). Newly surfaced `Flex4DHuman`/`WorldOlympiad` are too indirect for the core OSS bibliography.
