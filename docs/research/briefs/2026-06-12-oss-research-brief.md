# OSS Research Brief — 2026-06-12

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
- README snapshot: v7 is current: `srcnn-v7.0-pico-005`, N-D Gaussian canvas with native time axis and OSS-FX `t_query`; ≤1M-param distilled student is not trained; cross-vendor kernels remain designed-only.
- Training dashboard: rendered `https://opensupersampling.com` redirects to `.org`. Visible state: **v7.0 Pico OSS-FX** at step **23,250 / 100,000**, loss **0.15409**, viz @ step 23,000, ETA ~1.3d. Service health says Watcher/CF Worker/R2 healthy, but Trainer is **degraded / stuck at step 23250 for 7m** and GPU stats unavailable. Treat as short ops warning, not a training conclusion.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting** (arXiv:2606.07670v1, 2026-06-04; Mingzhao Li, Arghya Pal, Guan Yuan Tan) — Replaces discrete-time deformation MLP behavior with closed-form continuous-time/Liquid cells. Why it matters to OSS: directly relevant to v7’s native time axis and `t_query` smoothness if exact engine MVs need a learned residual.
- **Envision4D: Envisioning Visual Futures via Feed-forward 4D Gaussian Splatting for Autonomous Driving** (arXiv:2606.10656v1, 2026-06-09; Qi Song, Yifei He, Chi Zhang, Zheng Fu, et al.) — Feed-forward 4DGS future extrapolation for dynamic scenes. Why it matters to OSS: external validation that Gaussian primitives are becoming a future-frame substrate; compare against OSS-FX’s stronger engine-motion input.
- **PersistGS: Differentiable Physics for Object Permanence in 4D Gaussian Splatting** (arXiv:2606.03479v1, 2026-06-02; Adrian Ramlal, John S. Zelek) — Adds differentiable physics/object permanence during occlusion. Why it matters to OSS: potential prior for disocclusion handling and preventing canvas decay behind occluders.
- **RiGS: Rigid-aware 4D Gaussian Splatting from a Single Monocular Video** (arXiv:2605.23672v1, 2026-05-22; Chenyu Wu, Wanhua Li, Zhu-Tian Chen, Hanspeter Pfister) — Separates static, rigid, and transient Gaussian primitives. Why it matters to OSS: suggests a v7 canvas taxonomy for static world / rigid movers / particles instead of one homogeneous Gaussian pool.
- **ConFi-GS: Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution** (arXiv:2605.24964v1, 2026-05-24; Jiaxiang Li, Zongtan Zhou, Zhen Tan, Yadong Liu, et al.) — Confidence-guided high-frequency injection for low-res 3DGS SR. Why it matters to OSS: candidate confidence gate for splat spawning, detail injection, and pseudo-label filtering.
- **3DGS$^3$: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** (arXiv:2605.11489v1, 2026-05-12; Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu) — Unified post-rendering super sampling + frame interpolation for 3DGS. Why it matters to OSS: closest recent competitor to the “one system for SR+FG” positioning, though it post-processes 3DGS renders rather than persistent game canvas state.

### Super-Resolution / Upscaling
- **EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution** (arXiv:2606.13580v1, 2026-06-11; Dachun Kai, Jiayao Lu, Yueyi Zhang, Xiaoyan Sun) — Uses event signals for texture enhancement in VSR. Why it matters to OSS: not directly deployable for games, but supports separating high-frequency temporal texture signal from bulk motion alignment.
- **DTG-Restore: Training-Free Diffusion Refinement for Generative Video Super-Resolution** (arXiv:2605.30431v1, 2026-05-28; Hidir Yesiltepe, Koutilya PNVR, Gaurav Pathak, Navaneeth Bodla, et al.) — Decoupled time guidance for diffusion VSR refinement. Why it matters to OSS: teacher-side reference only; too heavy for runtime, but may help inspect temporal hallucination/replication failure modes.
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (arXiv:2604.20291v1, 2026-04-22; Pham Phuong Nam Nguyen, Nam Tien Le, Thi Kim Trang Vo, Nhu Tinh Anh Nguyen) — INT8 SISR with deployment-aware student training. Why it matters to OSS: direct precedent for ≤1M-param student distillation and quantization-aware architecture design.
- **DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution** (arXiv:2603.22271v1, 2026-03-23; Zhengyao Lv, Menghan Xia, Xintao Wang, Kwan-Yee K. Wong) — One-step VSR distillation combining distribution matching and adversarial supervision. Why it matters to OSS: useful teacher→student training pattern, even if diffusion/VSR setup differs.
- **Neural Super-Resolution for Real-time Rendering with Radiance Demodulation** (arXiv:2308.06699v2, 2023-08-13; Jia Li, Ziling Chen, Xiaolong Wu, Lu Wang, et al.) — Real-time rendering SR separates lighting/radiance components. Why it matters to OSS: reinforces using engine G-buffer/material channels rather than forcing the student to infer all structure from RGB.

### Frame Extrapolation / Generation
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567v1, 2026-05-11; Nengbo Lu, Bin Zhao) — Learns per-Gaussian velocity fields with physical consistency. Why it matters to OSS: high-priority deep read for v7 `V_xt` / velocity regularization in OSS-FX.
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586v1, 2026-05-11; Nengbo Lu, Minghua Pan) — Gaussian representation with learned causal dynamics. Why it matters to OSS: less game-specific, but supports per-Gaussian future-state prediction as a serious extrapolation primitive.
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (arXiv:2603.26835v3, 2026-03-27; Shibo Liu) — Reuses codec motion vectors to avoid unsupported/mobile-hostile flow refinement. Why it matters to OSS: strong analogue to using engine motion vectors as privileged low-cost alignment/extrapolation signal.
- **ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering** (arXiv:2506.13814v1, 2025-06-14; Lufei Liu, Tor M. Aamodt) — Caches neural network layers across temporally coherent rendering frames. Why it matters to OSS: candidate optimization if HAT/student features can be safely reused across adjacent game frames.
- **RSTT: Real-time Spatial Temporal Transformer for Space-Time Video Super-Resolution** (arXiv:2203.14186, 2022-03-27; Zhicheng Geng, Luming Liang, Tianyu Ding, Ilya Zharkov) — Single model for space-time VSR. Why it matters to OSS: older baseline for “one architecture handles spatial + temporal upsampling,” but without Gaussian canvas/exact engine MVs.

### Inference Optimization
- **Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution** (arXiv:2308.02794v1, 2023-08-05; Yong Liu, Hang Dong, Boyang Liang, Songwei Liu, et al.) — TensorRT-friendly SR transformer unit. Why it matters to OSS: HAT teacher/export constraints and student search should consider deployment graph shape early.
- **NAWQ-SR: A Hybrid-Precision NPU Engine for Efficient On-Device Super-Resolution** (arXiv:2212.09501v3, 2022-12-15; Stylianos I. Venieris, Mario Almeida, Royson Lee, Nicholas D. Lane) — Hybrid-precision SR engine. Why it matters to OSS: precision policy must be co-designed with architecture, not bolted onto a trained model.
- **Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers** (arXiv:2604.02344v1, 2026-02-09; Jędrzej Maczan) — Cross-vendor dispatch overhead study. Why it matters to OSS: <2ms budgets can be lost to many tiny dispatches; argues for fused kernels/megakernel planning in Vulkan/WebGPU-like backends.
- **Efficient Integer-Arithmetic-Only Convolutional Neural Networks** (arXiv:2006.11735v1, 2020-06-21; Pramod Gopalakrishnan, Michael Andersch) — Integer-only CNN inference. Why it matters to OSS: background reference for true low-bit student deployment when FP8 is not universally available.

### Game Engine / Motion Vectors
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694v1, 2026-03-08; Yuhang Wang, Hai Li, Shujuan Hou, Zhetao Dong, Xiaoyao Yang) — Online VSR using codec MVs/residuals/frame types. Why it matters to OSS: closest public analogue to reusing engine MVs and residual/disocclusion cues instead of estimating flow.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483v1, 2023-08-03; Antoine Mercier, Ruan Erasmus, Yashesh Savani, Manik Dhingra, Fatih Porikli, Guillaume Berger) — Game supersampling dataset/baseline with rendering metadata. Why it matters to OSS: still one of the most relevant public dataset/baseline references for cross-game eval.
- **Conditional Neural Video Coding with Spatial-Temporal Super-Resolution** (arXiv:2401.13959v1, 2024-01-25; Wei Jiang, Zipeng Wang, Feng Yang, Jiaying Liu) — Uses spatial-temporal SR in a conditioned video coding setting. Why it matters to OSS: less direct, but useful for thinking about motion/residual-conditioned reconstruction.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 broad hits, all noisy archived GitHub/star-list pages; no primary paper snapshot found.
- `"frame generation" "deep learning"`: 3 hits, including FluidFrames builds, DLSS Swapper builds, and an OSF registration for MoCha-LD; useful ecosystem breadcrumbs only.
- `"real-time upscaling" game`: 0 hits.
- `GS-STVSR`: 0 hits.
- `GSASR`: 1 false positive from CIA Reading Room (`gsasr` string collision), not relevant.
- `GRAPE "gaussian splatting"`: 0 hits.
- Targeted follow-up still needed: Wayback CDX against known arXiv/project/GitHub URLs for GS-STVSR, GSASR, GRAPE, and STSS. Broad Archive.org keyword search is too noisy for acronym-heavy paper names.

## GitHub / Open Source
- `Sungpyokim97/SR3R` — unofficial/private-looking SR3R repo updated 2026-06-12. Low stars, but SR3R is already OSS-relevant feed-forward Gaussian SR; monitor for real code/weights. https://github.com/Sungpyokim97/SR3R
- `hvcl/3DCT_SR_DDNM_NAB-GS` — zero-shot 3D CT SR using diffusion 2D projection priors + Negative Alpha Blending Gaussian Splatting. Medical-domain, but alpha/confidence blending may transfer to unreliable detail injection. https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS
- `Adam-AtlasSoftware/RDNU` — Radeon Decoupled Neural Upscaler, RDNA3-focused ML upscaler using Decoupled G-buffer Guidance and game/virtual datasets. **Threat/opportunity:** overlaps OSS cross-vendor and G-buffer-guided positioning; inspect license, actual model code, and whether claims are real. https://github.com/Adam-AtlasSoftware/RDNU
- `nova-studios-hw/novagpu-ts1t` — open-source GPU RTL mentioning frame generation; likely not immediately relevant to OSS software path.
- No high-signal new DLSS/FSR analysis-tool repos found via GitHub REST search in the `pushed:>=2026-05-29` window.
- Vendor/dev feeds checked: NVIDIA technical blog has a 2026-06-11 TensorRT FP8 checkpoint→engine quantization post; AMD GPUOpen has RDNA4 WMMA guide parts 1–3 (2026-06-02) and MiniDXNN DX12 MLP library (2026-04-09); Intel feed returned HTTP 403.

## News
- NVIDIA: Google News RSS surfaces NVIDIA items for **DLSS 4.5 Ray Reconstruction**, UE5 plugin availability, and new DLSS games. **Threat:** vendor stack keeps improving plugin-level RR/upscaling faster than OSS has runtime kernels.
- AMD: high-signal FSR 4 news was sparse in the 14-day RSS pass; AMD developer content is more about RDNA4 WMMA/tooling than FSR-specific releases. **Opportunity:** OSS can use open RDNA WMMA guidance for future HIP/DX12/compute kernels.
- Intel: XeSS 3 Multi Frame Generation appears in Death Stranding 2 coverage. **Threat:** Intel is entering the multi-frame-gen narrative; OSS-FX messaging must be crisp about one-canvas / no-second-network economics.
- Apple/Metal: M4 GPU benchmark/developer-tool coverage and Mac gaming visibility continue. **Opportunity:** Metal backend value is rising, but OSS still has zero Metal kernel code.
- Unreal/Unity: no high-signal upscaling-specific RSS hits this run.
- Web search backend note: generic `web_search` failed with Firecrawl payment/credit exhaustion; structured arXiv API, GitHub REST, Archive.org API, vendor RSS, Google News RSS, and rendered browser dashboard were used instead.

## Action Items / Follow-ups
- [ ] Ops: check trainer if it remains stuck/degraded at step 23,250; distinguish transient 7m stall from actual dead training process.
- [ ] Deep-read `VeloGauss`, `Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field`, and `PersistGS` for v7 time-axis, velocity, and occlusion priors.
- [ ] Inspect `Adam-AtlasSoftware/RDNU` for architecture, license, dataset provenance, and seriousness as an OSS competitor/baseline.
- [ ] Add targeted Wayback CDX checks for known GS-STVSR / GSASR / GRAPE / STSS URLs; do not rely on broad Archive.org acronym searches.
- [ ] Revisit student design constraints against INT8/QAT papers, TensorRT FP8 checkpoint conversion, AMD RDNA4 WMMA guidance, and dispatch-fusion needs before freezing ≤1M-param architecture.

## Bibliography Update
- No new `BIBLIOGRAPHY.md` append this run: the materially relevant findings surfaced today (`VeloGauss`, `RDNU`, `ConFi-GS`, `FlowGS`, `ANVIL`, `3DGS^3`, `PersistGS`, INT8 SR) are already present in the current bibliography/recent monitor additions.
