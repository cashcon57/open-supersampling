# OSS Research Brief — 2026-06-26

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
- Open issues (12 total, first 10):
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
- README snapshot: pre-alpha; v7 architecture is current (`srcnn-v7.0-pico-005`), with N-D Gaussian canvas + native time axis and HAT-Tiny teacher; shipping ≤1M-param student not trained yet. README still describes v7 as training, v6.2 step-74K as baseline-to-beat, and cross-vendor kernels / DLL-shim runtime as designed-not-built.
- Training dashboard: rendered state says `v7.0 Pico OSS-FX is training right now`, active run `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz at step **26,000**, cost **$0.76**, **12.63 GPU-hours**, but service health tooltip says **Trainer — offline, stuck at step 30300 for 47h**. Watcher healthy (published ~0-1s ago), CF Worker healthy (HTTP 200 ~79-91ms), R2 origin healthy (HTTP 200), GPU stats unavailable / host offline or rsync failed. This is a dashboard contradiction worth treating as an ops follow-up.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Envision4D: Envisioning Visual Futures via Feed-forward 4D Gaussian Splatting for Autonomous Driving** (arXiv:2606.10656, 2026-06-09; Qi Song, Yifei He, Chi Zhang et al.) — Feed-forward 4D Gaussian future prediction for autonomous driving; matters as external validation that Gaussian primitives are becoming a future-frame substrate. Already tracked in bibliography; still relevant to OSS-FX positioning.
- **GSPan: A Continuous Gaussian Primitive Representation for Arbitrary-Scale Pansharpening** (arXiv:2606.17722, 2026-06-16; Fangyi Li, Xiaoyuan Yang, Yixiao Li et al.) — Continuous Gaussian primitive representation for arbitrary-scale pansharpening; remote-sensing domain, but the rendered Gaussian residual/detail-field idea maps to OSS canvas detail injection. Already tracked.
- **Intrinsic 4D Gaussian Segmentation from Scene Cues** (arXiv:2606.18623, 2026-06-17; Hasan Yazar, Mohamed Rayan Barhdadi, Erchin Serpedin et al.) — 4DGS segmentation from intrinsic scene cues; potential object/primitive-affinity signal for canvas culling and temporal stability. Already tracked.

### Super-Resolution / Upscaling
- **EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution** (arXiv:2606.13580, 2026-06-11; Dachun Kai, Jiayao Lu, Yueyi Zhang et al.) — Event-driven VSR separates motion alignment from texture enhancement; useful analogy for OSS detail/texture losses even though event-camera inputs are not game inputs. Already tracked.
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (arXiv:2604.20291, 2026-04-22; Pham Phuong Nam Nguyen, Nam Tien Le, Thi Kim Trang Vo et al.) — Deployment-aware INT8 SISR with teacher-guided training; directly relevant to OSS ≤1M-param INT8/FP8 student distillation. Already tracked.
- **DTG-Restore: Training-Free Diffusion Refinement for Generative Video Super-Resolution** (arXiv:2605.30431, 2026-05-28; Hidir Yesiltepe, Koutilya PNVR, Gaurav Pathak et al.) — Training-free diffusion refinement for generative VSR; not a shipping path, but a possible offline pseudo-target / teacher-quality enhancer. Already tracked.

### Frame Extrapolation / Generation
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (arXiv:2603.26835, 2026-03-27; Shibo Liu) — ANVIL uses codec motion-vector priors for accelerator-native video interpolation; strong design analogue for OSS exact-engine-MV frame extrapolation. Already tracked.
- **ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering** (arXiv:2506.13814, 2025-06-14; Lufei Liu, Tor M. Aamodt) — Layer caching for neural real-time rendering exploits temporal coherence; candidate optimization for OSS teacher/student inference. Already tracked.
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586, 2026-05-11; Nengbo Lu, Minghua Pan) — Gaussian-state physical causality / future prediction; possible long-horizon regularizer beyond single α=0.5 interpolation. Already tracked.

### Inference Optimization
- **Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution** (arXiv:2308.02794, 2023-08-05; Yong Liu, Hang Dong, Boyang Liang et al.) — Deployment-friendly SR transformer unit; potential simplification reference for HAT-style teacher/student blocks. Already tracked.
- **NAWQ-SR: A Hybrid-Precision NPU Engine for Efficient On-Device Super-Resolution** (arXiv:2212.09501, 2022-12-15; Stylianos I. Venieris, Mario Almeida, Royson Lee et al.) — Hybrid-precision NPU SR engine; older but useful for quantization/accelerator partitioning. Already tracked.

### Game Engine Integration
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483, 2023-08-03; Antoine Mercier, Ruan Erasmus, Yashesh Savani et al.) — Game-specific neural supersampling dataset/baseline; still a direct comparison target for OSS cross-engine claims. Already tracked.
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694, 2026-03-08; Yuhang Wang, Hai Li, Shujuan Hou et al.) — Compressed-domain VSR using motion vectors/residual/frame-type metadata; reinforces the value of reusing engine/codec side information. Already tracked.

Note: earlier same-day arXiv scan returned 52 raw entries / 33 locally relevant-looking entries. A later refresh attempt hit arXiv API 429/timeouts across the broad query groups, so the memo preserves the verified earlier same-day metadata rather than weakening the report. Most material papers above were already captured in `BIBLIOGRAPHY.md`; today produced no genuinely new high-confidence bibliography append.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"` returned 3 Archive.org hits; first identifiers: github.com-Awesome-Image-Registration-Organization-awesome-image-registration_-_2025-08-20_11-05-40, github.com-pbsds-starred_-_2025-10-03_04-36-09. No paper takedown/update signal surfaced.
- `"frame generation" "deep learning"` returned 4 Archive.org hits; first identifiers: fluidframeswin, Minecraft-Radiance-Radiance-20260613122233. No paper takedown/update signal surfaced.
- `"real-time upscaling" game` returned 0 broad Archive.org hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS` returned 16593 noisy hits dominated by unrelated `grape` media/items; not useful as evidence without targeted CDX checks.
- Wayback CDX for `https://arxiv.org/abs/2404.19702` returned snapshots; verified archive path exists.
- Archive check warning: https://arxiv.org/abs/2403.10267 — TimeoutError('The read operation timed out').
- Archive check warning: https://arxiv.org/abs/2403.12535 — <HTTPError 503: 'Service Unavailable'>.

## GitHub / Open Source
- `Azusa309/MedGSSR` (1★, pushed 2026-06-22, Python, license MIT) — MedGSSR: Generalizable Medical Image Super-Resolution 3D Reconstruction via Hierarchical Feed-forward Gaussian Splatting (ECCV 2026) Low-priority: medical GS-SR feed-forward reconstruction, MIT, new but domain-distant.
- `pranav-asthana/SplatSuRe` (81★, pushed 2026-06-16, Python, license NOASSERTION) — Official code for SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting (CVPR 2026) Opportunity: CVPR 2026 selective GS super-resolution code; useful baseline for multi-view consistency / GS-SR detail handling.
- `ShrekDino/FreeGen` (0★, pushed 2026-06-16, C++, license Apache-2.0) — Open-source upscaling and frame generation — Vulkan, FSR 3.1. Watch: claimed Vulkan + FSR 3.1 open upscaling/frame-gen stack, but zero stars and unverified maturity.
- `bygalacos/OptiScalerBuilder` (18★, pushed 2026-06-26, n/a, license unknown) — Build Latest Version of OptiScaler 0.10.0-pre1 bundled with all required DLL files for SL/DLSS/DLSSG/FSR/FSRFG/XeSS/XeFG Ecosystem signal: tooling around upscaler DLL swapping/management remains active.
- `optiscaler/OptiScaler` (8915★, pushed 2026-06-26, C++, license GPL-3.0) — OptiScaler bridges upscaling/frame gen across GPUs. Supports DLSS2+/XeSS/FSR2+ inputs, replaces native upscalers, enables FSR-FG/XeFG on non-FG titles. Supports Nukem mod for DLSSG-to-FSR3 FG. Threat/positioning: mature DLL/upscaler bridge in the exact integration surface OSS plans, though it swaps existing tech rather than training a new vendor-neutral model.
- `Devanshu-dev/optiscaler-client-fsr4` (21★, pushed 2026-06-25, C#, license MIT) — optiscaler client optiscaler manager github download modern gui installer desktop utility pc gaming fps booster upscaling mod manager avalonia ui windows 11 linux gaming steam deck auto scanner game library manager profile manager ini configuration tool dlss to fsr xess fsr 4 int8 nukemfg fakenvapi optipatcher frame generation Ecosystem signal: tooling around upscaler DLL swapping/management remains active.
- `osyka-yuri/renderpilot` (3★, pushed 2026-06-25, Rust, license GPL-3.0) — Update and swap DLSS, FSR, XeSS and other rendering libraries across your games. Full DLSS control via NVAPI included. Ecosystem signal: tooling around upscaler DLL swapping/management remains active.

## News
- NVIDIA GeForce page (2026-06-23) lists new DLSS integrations for Mortal Shell II / The Sinking City 2 demos and notes Subnautica 2 upgraded to DLSS 4.5 Super Resolution; competitive pressure is continued rapid title adoption.
- NVIDIA GeForce page still surfaces DLSS 4.5 Ray Reconstruction + 2nd-gen transformer SR + Dynamic Multi Frame Generation as the current high-end positioning; OSS should keep the “single architecture for SR+FG” contrast explicit.
- AMD GPUOpen lists FSR SDK 2.3 / FSR Upscaling 4.1.1 for Radeon RX 7000 plus RDNA4 Frame Generation 4.0.1 and Ray Regeneration 1.2; opportunity/threat: FSR4 quality moving down-stack to RDNA3 narrows OSS’s “open + cross-vendor” pitch unless OSS can win on architecture/licensing.
- AMD GPUOpen WMMA guides for RDNA4 show FP8/INT8 128-bit-load and fused-WMMA patterns; directly relevant to future HIP/RDNA student kernels.
- AMD MiniDXNN is a native HLSL/D3D12 MLP inference library targeting Radeon matrix cores; possible reference for Windows/D3D12 inference without heavy framework interop.
- Google News RSS leads: Valve/Proton testing FSR4 for Steam Machine/Linux and Apple Game Porting Toolkit 4 improvements; headline-level only, not primary-source-verified in this run.

## Action Items / Follow-ups
- [ ] Ops: investigate `srcnn-v7.0-pico-005` trainer offline/stuck at step 30,300 for 47h while dashboard badge still says live/training.
- [ ] Deep-read `SplatSuRe` and compare selective GS-SR detail handling against OSS v7 parent-child density / detail recovery.
- [ ] Track `OptiScaler` and surrounding GUI/build tooling as the de facto open integration layer; OSS DLL-shim design should interoperate with or clearly differentiate from this ecosystem.
- [ ] Investigate AMD FSR SDK 2.3 / FSR 4.1.1 RDNA3 support and RDNA4 WMMA/FP8-INT8 guides for future HIP/RDNA student-kernel planning.
- [ ] Re-run targeted Wayback CDX checks for GS-STVSR/GSASR/GRAPE/STSS URLs; broad Archive.org acronym search was too noisy and two CDX requests timed out/503'd.
