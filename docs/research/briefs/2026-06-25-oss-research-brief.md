# OSS Research Brief — 2026-06-25

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
- README snapshot: current public story is v7 `srcnn-v7.0-pico-005` training as the first N-D Gaussian canvas / native-time-axis teacher. Student models are still untrained; cross-vendor kernels and DLL shim remain designed-only.
- Training dashboard: `opensupersampling.com` redirects to `.org`. Visible state: `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz at step **26,000**, 30.3% complete, ETA ~1.2d. Health badges conflict with the “live” badge: Trainer tooltip says **offline, stuck at step 30300 for 23h**; Watcher healthy, published 1s ago; CF Worker healthy HTTP 200 in 92ms; R2 origin healthy HTTP 200. GPU stats unavailable (`host offline or rsync failed`). New tailnet-only training controls are visible.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Temporally Aware Densification for Dynamic 3D Gaussian Splatting** (arXiv:2606.23212v1, 2026-06-22; Vikram Sandu, Mayurdeep Pathak, Rajiv Soundararajan) — Despite modeling temporal motion, dynamic 3D Gaussian Splatting (3DGS) methods still inherit a static densification strategy that is ill-suited for dynamic scenes. This neglect of temporal behavior leads to under-reconstructed and blurry dynamic regions, as sh... Why it matters to OSS: direct candidate ablation for v7 parent-child/loss-adaptive density: densification should use temporal behavior, not static 3DGS heuristics.
- **Multi4D: High-Fidelity Dynamic Gaussian Splatting via Multi-Level Competitive Allocation** (arXiv:2606.22197v1, 2026-06-20; Rui Wang, Quentin Lohmeyer, Siyu Tang, Mirko Meboldt) — Dynamic 3D Gaussian splatting faces a fundamental tension between motion consistency and visual fidelity. Deformation-based approaches preserve temporal correspondence but suffer from motion over-factorization, oversmoothing high-frequency dynamics. In contras... Why it matters to OSS: reinforces the need to separate motion consistency from visual fidelity in dynamic Gaussian canvases; compare its competitive-allocation policy with v7 density/pruning.

### Super-Resolution / Upscaling
- **Fidelity- and Perception-Aware Local Implicit Attention for Arbitrary-Scale Image Super-Resolution** (arXiv:2606.21910v1, 2026-06-20; Yu-Syuan Xu, Hao-Lun Sun, Hao-Wei Chen, Hsien-Kai Kuo et al.) — Arbitrary-scale image super-resolution (ASISR) aims to reconstruct high-resolution images from low-resolution inputs over a continuous range of upscaling factors. While traditional pixel-regression approaches often produce overly smooth results that lack reali... Why it matters to OSS: arbitrary-scale local implicit attention is not game-specific, but the fidelity/perception split could inform teacher losses and continuous-scale SR heads.
- **No new game-specific neural supersampling arXiv paper surfaced in this run.** The search mostly rediscovered already-tracked items (GFFE, ANVIL, 3DGS³, DiffST, G-buffer/radiance demodulation).

### Frame Extrapolation / Generation
- **UniRED: Unified RGB-D Video Frame Interpolation with Event Guidance** (arXiv:2606.24282v1, 2026-06-23; Yinuo Zhang, Guangshun Wei, Yuanfeng Zhou, Yiran Shen) — High frame-rate RGB-D videos are crucial for a variety of downstream tasks, including motion analysis, dynamic scene understanding, and 3D reconstruction. However, due to hardware and sensing constraints, practical RGB-D cameras are typically limited to low fr... Why it matters to OSS: RGB-D + event-guided interpolation is a depth/motion-conditioned VFI reference; useful for OSS-FX artifact taxonomy even though event cameras are not the game input path.
- Already-tracked close competitors remain **GFFE** (arXiv:2406.18551), **ANVIL** (arXiv:2603.26835), and **3DGS³** (arXiv:2605.11489); no stronger same-week OSS-FX analog appeared.

### Inference Optimization
- **ANEForge: Python for direct computation on the Apple Neural Engine** (arXiv:2606.17090v1, 2026-06-12; Spencer H. Bryngelson) — ANEForge is a Python package that programs the Apple Neural Engine (ANE), the fixed-function neural accelerator on every recent Apple device, directly and without CoreML. In production the engine is reachable only through CoreML, which treats it as a schedulin... Why it matters to OSS: Apple ANE direct compute is relevant to a future Metal/Apple path, especially if CoreML operator support blocks custom student/raster fusion.
- **Realizing Native INT8 Compute for Diffusion Transformers on Consumer GPUs: A Fused INT8 GEMM Kernel for Ideogram 4.0** (arXiv:2606.14598v1, 2026-06-12; Ali Asaria, Tony Salomone, Deep Gandhi) — Post-training INT8 (W8A8) quantization of diffusion transformers is widely deployed as a speed optimization, yet on consumer Ampere GPUs it is frequently slower than the FP8 and NF4 alternatives it is meant to beat. We trace this to a software artifact: the pr... Why it matters to OSS: native INT8 fused GEMM on consumer GPUs is a warning for OSS INT8/FP8 planning: vendor/library paths can underperform without true hardware-level fused kernels.

## Archive.org Findings
- Archive.org keyword search for `"gaussian splatting" super-resolution` returned mostly generic GitHub collection snapshots, not primary paper mirrors.
- `"frame generation" deep learning` returned **DLSS Swapper builds for Windows**, **FluidFrames older builds**, and unrelated media/repo captures. Useful as tooling/archive leads, not OSS architecture evidence.
- `"real-time upscaling" game` returned no useful Archive.org item in this run.
- Wayback CDX confirmed captures for `https://github.com/ChrisDud0257/GSASR` (first listed 2025-03-13). CDX requests for GS-STVSR/arXiv URLs timed out; do not treat as absence.

## GitHub / Open Source
- **Azusa309/MedGSSR** — `MedGSSR: Generalizable Medical Image Super-Resolution 3D Reconstruction via Hierarchical Feed-forward Gaussian Splatting`, MIT, pushed 2026-06-22, 1 star. Medical-domain but another feed-forward Gaussian SR/reconstruction signal.
- **pranav-asthana/SplatSuRe** — official CVPR 2026 selective GS super-resolution repo, pushed 2026-06-16, 81 stars. Already in bibliography; selective-SR principle remains relevant to “spend expensive ops only where undersampled.”
- **ShrekDino/FreeGen** — open-source upscaling/frame-generation repo claiming Vulkan + FSR 3.1, Apache-2.0, pushed 2026-06-16, 0 stars. Low-signal but worth monitoring as an OSS/free FG competitor surface.
- **Adam-AtlasSoftware/RDNU** — Radeon Decoupled Neural Upscaler, GPL-3.0, pushed 2026-06-09, 16 stars. Already tracked; still the most relevant open-source RDNA/G-buffer neural-upscaler watch item.
- GitHub search found no fresh dedicated DLSS/FSR analysis tool repos in the last 30 days.

## News
- **NVIDIA DLSS 4.5 UE plugin** (NVIDIA blog, 2026-05-27): Dynamic Multi Frame Generation, 6x mode, and a second-generation transformer SR model are now packaged for Unreal Engine. Threat: competitor integration story is getting easier and more UE-native while OSS still lacks DLL shim/runtime.
- **AMD FSR SDK 2.3 / FSR Upscaling 4.1.1** (GPUOpen, 2026-06-24): ML-based FSR 4.1.1 expands to RDNA 3 / Radeon RX 7000 discrete GPUs; Frame Generation 4.0.1 improves motion-vector and camera-parameter input handling; Ray Regeneration 1.2 adds denoising/checkerboard modes and ABI-breaking changes. Threat/opportunity: AMD is broadening ML upscaling hardware support, but its motion-vector/camera-parameter notes validate OSS’s engine-signal-first positioning.
- **Intel Arc G-Series handheld processors** (Intel Newsroom): Arc G3/G3 Extreme handheld chips include XeSS 3, Multi-Frame Generation, and Xe Low Latency. Threat: handheld target space is increasingly crowded by first-party upscalers; opportunity: OSS Pico/Steam Deck story should emphasize vendor-neutral quality and non-NVIDIA/AMD/Intel portability.
- **Apple Game Porting Toolkit 4** (Apple Developer): adds Metal 4 evaluation, command-line Metal capture/debug/profiling, and explicit upscaling/denoising/frame-generation mentions in the porting workflow. Opportunity: improves the eventual Metal backend development loop; pair with ANEForge/Metal profiling research before committing to Apple runtime strategy.
- Google News RSS also surfaced headline-level reports of FSR 4.1 on older GPUs and Valve testing FSR 4 ahead of Steam Machine launch; treat as leads unless primary pages are checked.

## Action Items / Follow-ups
- [ ] Deep-read **Temporally Aware Densification for Dynamic 3D Gaussian Splatting** and compare its temporal densification signal against v7 parent-child loss-adaptive density.
- [ ] Add an ops follow-up for the dashboard contradiction: public copy says live, but trainer health says offline/stuck 23h and GPU stats unavailable.
- [ ] Track AMD FSR SDK 2.3 details for student distillation/benchmarking: RDNA 3 ML upscaler support changes the competitive baseline for midrange GPUs.
- [ ] Watch DLSS 4.5 UE plugin integration path; OSS runtime pitch needs a credible UE/DLL-shim milestone soon.
- [ ] Investigate whether ANEForge or Metal 4 tooling can support a realistic Apple/Metal inference backend plan, or whether OSS should stay GPU-compute-only on Apple initially.
