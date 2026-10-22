# OSS Research Brief — 2026-07-17

_Collected 2026-07-18T01:05:44Z. Sources: local git/README, GitHub REST, arXiv Atom, Archive.org APIs, direct vendor pages, Google News RSS, rendered dashboard, targeted web extraction._

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
- README snapshot: v7 `srcnn-v7.0-pico-005` is the current research/teacher architecture: native time-axis N-D Gaussian canvas, HAT-Tiny, parent-child loss-adaptive density, Mip-Splatting filters, alpha=0.5 OSS-FX supervision. Distilled shipping student is not trained; cross-vendor kernels remain designed-only.
- Training dashboard: active run `srcnn-v7.0-pico-005`, step **35,200 / 100,000**, loss **0.10166**, viz @ **35,077**. Public page says `live` / updated just now, but service health tooltip says **Trainer offline, stuck at step 35,200 for 24h**; Watcher healthy/published 1s ago, CF Worker HTTP 200 in 108ms, R2 origin HTTP 200. GPU stats unavailable (`host offline or rsync failed`). Earlier same-day brief saw the same step/loss at 12h stuck; current read confirms the trainer stall is continuing while the watcher/CDN remain live.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Compression of 3D Gaussian Splatting Data Using GPU-friendly Graphics Texture Coding** (arXiv:2607.14513v1, 2026-07-16; Amir Said, Randall Rauwendaal) — Compresses 3DGS spherical-harmonic color coefficients with GPU-native texture-compression formats so decoding can happen through hardware-supported parallel texture paths. Why it matters to OSS: directly relevant to packed Gaussian canvas layout, memory bandwidth, and cross-vendor GPU decode paths before raster/composite.
- **Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation** (arXiv:2607.14203v1, 2026-07-15; NVIDIA et al.) — Turns short calibrated multi-view driving logs into layered static/dynamic 3DGS worlds in one forward pass. Why it matters to OSS: off-task, but NVIDIA moving feed-forward Gaussian scene reconstruction into driving simulation is a competitive signal for Gaussian primitives in real-time-ish rendering stacks.
- **HyperGS: Fast and Generalizable Gaussian Video Representation** (arXiv:2607.11500v1, 2026-07-13; Fatimah Zohra et al.) — Feed-forward, optimization-free Gaussian video representation with factorized spatiotemporal transformers and query-predicted per-frame Gaussians. Why it matters to OSS: already in `BIBLIOGRAPHY.md`; relevant to v7 native-time canvas management and avoiding per-scene/per-video optimization.
- **Implicit 4D Gaussian Splatting for Fast Motion with Large Inter-Frame Displacements** (arXiv:2607.12362v1, 2026-07-14; Seung-gyeom Kim, Areum Kim, Yongjae Yoo, Sukmin Yun) — Predicts Gaussian attributes from explicit spatiotemporal positions to preserve fast-moving objects under large displacements. Why it matters to OSS: direct study target for v7 `V_xt` motion encoding and OSS-FX fast-motion failure modes; already tracked.

### Super-Resolution / Upscaling
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (arXiv:2607.10495v1, 2026-07-11; Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies) — Fully convolutional VSR with structural reparameterization into standard inference convolutions and no explicit flow/custom ops. Why it matters to OSS: strongest current student-model lead for ≤1M-param TensorRT/HIP/Metal-friendly deployment and issue #9 RepVGG/ECBSR/RepSR ablations; already tracked.
- **CUST: Clustered Unit-level Similarity Transformer for Lightweight Image Super-Resolution** (arXiv:2607.11088v1, 2026-07-13; Jeongsoo Kim) — Lightweight SR transformer with clustered long-range patch similarity and explicit high-frequency extraction. Why it matters to OSS: possible HAT/global-context or student detail-head idea; still needs latency/params/kernel-read before bibliography.
- **RFMSR: Residual Flow Matching for Image Super-Resolution** (arXiv:2607.12753v1, 2026-07-14; Shuwei Huang et al.) — Residual flow matching for image SR centered on the LQ latent. Why it matters to OSS: teacher/pseudo-target only; likely too generative/heavy for shipping inference.

### Frame Extrapolation / Generation
- **MASS: Motion-Aligned Selective Scan for Refinement in Flow-Based Video Frame Interpolation** (arXiv:2606.27718v1, 2026-06-26; Jun-Sang Yoo, Seung-Won Jung) — Builds selective-scan feature sequences along flow-guided trajectories for nonlinear motion and occlusion refinement. Why it matters to OSS: external VFI baseline for OSS-FX disocclusion/artifact comparisons; already tracked.
- **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (arXiv:2406.18551, 2024; Songyin Wu, Deepak Vembar, Anton Sochenov, Selvakumar Panneer) — NVIDIA low-latency frame extrapolation without G-buffers. Why it matters to OSS: competitive baseline; OSS should contrast exact engine-MV + persistent Gaussian canvas against standalone extrapolator designs.
- **URVC: A Unified Real-Time Neural Video Coding Model with Temporal, Spatial, and Perceptual Adaptivity** (arXiv:2607.15033v1, 2026-07-16; Xihua Sheng, Chang Wen Chen) — Real-time neural video coding with adaptive temporal/spatial/perceptual choices. Why it matters to OSS: weak direct fit, but watch for runtime temporal-adaptivity patterns and bit/cost allocation analogies.

### Inference Optimization
- **SpeedyGS: Content-Aware 3D Gaussian Splatting Compression via Two-Stage Optimization** (arXiv:2607.12656v1, 2026-07-14; Junteng Zhang et al.) — Compresses 3DGS with rate-distortion-guided quantization/pruning and faster decoding than advanced codecs that still take seconds. Why it matters to OSS: relevant to packed-canvas density, active-mask pruning, and avoiding compression schemes that save VRAM but break interactivity.
- **Compression of 3D Gaussian Splatting Data Using GPU-friendly Graphics Texture Coding** is the most directly actionable optimization find this run: dedicated texture-decode hardware is closer to OSS's cross-vendor real-time constraints than heavyweight learned codecs.
- No new TensorRT/INT8/kernel-fusion SR paper stronger than already tracked NanoVSR / QuantSR+ / deployment-aware quantized SR surfaced.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"` — Archive.org advanced search returned 3 broad/noisy inventory hits, mostly archived GitHub-list pages; no confirmed paper takedown/update signal.
- `"frame generation" "deep learning"` — returned FluidFrames/DLSS Swapper archive items plus one OSF-style record; useful as software-history inventory only, not architecture evidence.
- `"real-time upscaling" game` — no useful Archive.org item hits.
- Targeted Wayback checks: GS-STVSR arXiv returned no 2025+ HTTP-200 rows; GSASR GitHub CDX timed out on refresh; GRAPE WACV and STSS search checks returned no useful 2026 rows. No evidence of removed/changed primary papers from Archive.org this run.

## GitHub / Open Source
- `gaussian splatting super-resolution pushed:>=2026-07-03` — no GitHub REST repo hits in the refresh window. Earlier same-day/30-day scan had low-star `ResonantBraneSplatting` and `MedGSSR`; no new stronger competitor appeared.
- `frame generation open source pushed:>=2026-07-03` — one hit, `OULDRABAH95/AllPen-framework-`, unrelated pentest/agent framework; no OSS action.
- `neural upscaler pushed:>=2026-07-03` — mostly toy/site repos (`bananascaler`, multiple generic `Neural-Network-Upscaler` repos, `neural-image-upscaler`); no threat to OSS architecture.
- Open-source tooling: **OptiScaler** remains the strongest ecosystem signal. Its README says it intercepts DLSS2+/FSR2+/XeSS inputs, supports FSR4, separates FG source/output since v0.9.0, and has experimental DX12 `OptiFG`. Why it matters to OSS: proves demand for cross-vendor upscaler/FG middleware and suggests the DLL-shim compatibility surface OSS should eventually target.
- Vendor pages checked: NVIDIA DLSS pages confirm DLSS 4.5, Dynamic Multi Frame Generation, second-generation transformer SR; AMD GPUOpen news is reachable and keyword-hits FSR/FidelityFX; Intel Arc page confirms XeSS 3 Multi-Frame Generation; Apple WWDC26 Games guide confirms Metal 4.1 / MetalFX temporal-upscaler changes.

## News
- **NVIDIA / DLSS 4.5:** NVIDIA primary pages confirm DLSS 4.5 with Dynamic Multi Frame Generation, up to five generated frames per rendered frame, and a second-generation transformer Super Resolution model. Threat: the competitive quality bar is now transformer SR + dynamic 6X MFG, not just DLSS 4 baseline.
- **AMD / FSR 4:** Google News/RSS and TechPowerUp report AMD is testing FSR Multi-Frame Generation up to 8x, but primary AMD confirmation was not found in this run. Treat as a high-priority rumor/lead, not verified spec.
- **Intel / XeSS:** Intel primary Arc Gaming Technologies page now presents XeSS 3 with Multi-Frame Generation inserting up to 3 generated frames between rendered frames, plus XeLL. Competitive note: all major vendors are converging on multi-frame generation; OSS's “one architecture SR+FG” positioning needs explicit latency/artifact comparisons.
- **Apple / Metal:** Apple WWDC26 Games guide says Metal 4.1 for games includes a redesigned MetalFX temporal upscaler using Neural Engine / Neural Accelerators on M5 Pro and M5 Max, improved API compatibility with leading upscalers, and command-line Metal debug/profile tools. Opportunity: Metal backend planning should account for Neural Accelerator/MetalFX coexistence and scriptable profiling.
- **Game engines:** No primary Unreal/Unity upscaler-specific change verified. Unreal RSS/web leads were mostly NVIDIA title support; Unity blog had no relevant upscaling hits.

## Action Items / Follow-ups
- [ ] CRITICAL: investigate trainer/dashboard mismatch — step/loss/viz remain stuck at 35,200 / 0.10166 / 35,077, trainer health now says offline for 24h, watcher/CDN still report live.
- [ ] Deep-read **Compression of 3D Gaussian Splatting Data Using GPU-friendly Graphics Texture Coding** for GPU-native packed-canvas/color-coefficient storage ideas.
- [ ] Deep-read **NanoVSR** for structural reparameterization patterns feeding issue #9 and the ≤1M-param student plan.
- [ ] Skim **SpeedyGS** against H004 packed low-rank canvas layout and active-density/pruning plans.
- [ ] Verify AMD FSR 8x/MFG rumor against primary AMD driver/release notes before changing competitive positioning.
- [ ] Use Apple Metal 4.1 / command-line profiling update as a future backend-planning input once Metal work starts.

## Bibliography Update
- Added `Compression of 3D Gaussian Splatting Data Using GPU-friendly Graphics Texture Coding` (arXiv:2607.14513v1) and `SpeedyGS` (arXiv:2607.12656v1) as material new entries for packed-canvas memory/compression and interactive 3DGS compression. No duplicate append for NanoVSR, SPIN-4DGS, HyperGS, ExtraGS, or GeoFovea-GS; all were already present.
