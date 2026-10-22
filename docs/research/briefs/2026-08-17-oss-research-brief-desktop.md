# OSS Research Brief — 2026-08-17

## Project Status Summary
- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated training controls and dashboard UI. Recent trainer changes preserve the v7 step-5K recovery snapshot, repair cross-OS checkpoint resume, add FlashAttention/TF32/prefetch throughput work, and retain smoothing filters in FP32 under BF16.
- **Open issues:** 12. The immediate architecture/release blockers remain student distillation (#4), cross-vendor ports (#8), reparameterized student experiments (#9), packed canvas (#12), L2 profiling (#13), and VRAM measurement (#14); #10 also tracks a decoupled G-buffer-encoder ablation.[3]
- **README snapshot:** v7 is a research teacher built around an N-D `(x,y,t)` Gaussian canvas. The shipping ≤1M CNN and non-CUDA backends remain unbuilt.
- **Training dashboard:** `srcnn-v7.0-pico-005` still presents as “live” at **36,650 / 100,000**, loss **0.19268**, viz @36K, but its Trainer health says **offline, stuck at step 36,650 for 195h**; GPU stats are unavailable. Watcher, CF Worker, and R2 are healthy. Step/loss remain unchanged while the outage counter grew, so this is not training progress.[1]

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **GS²CI: Robust Gaussian Splatting For Snapshot Compressive Imaging via Large Vision Model Priors** — Yanming Yang, Chenxi Song, Ping Wang, Xin Yuan, Chi Zhang; 2026-08-13; arXiv:2608.13502. It uses opacity-guided splitting plus explicit candidate-ratio, mean-opacity, and total-Gaussian growth bounds to stabilize ambiguous Gaussian optimization. **Why it matters to OSS:** directly actionable bounded parent-child spawning ablation for v7; port the growth-control mechanics, not the snapshot-compressive task.[10]
- **EndoMD-SLAM** (arXiv:2608.08949) and **FAST-GS** (arXiv:2608.01958) remain the strongest recently surfaced temporal-canvas leads, but were already added to prior briefs/bibliography. The former motivates static/transient update gating; the latter is a hard-motion `V_xt` regularization reference. No newer material temporal-GS paper displaced them in this run.

### Super-Resolution / Upscaling
- **MeanSR: Restoration Trajectory Learning for One-Step Perceptual Super-Resolution** — Axi Niu, Jiawei Kou, Kang Zhang, Qingsen Yan, Jinqiu Sun, Yanning Zhang; 2026-08-10; arXiv:2608.09405. It learns an LR-conditioned average velocity field to reach perceptual SR in one step. **Why it matters:** a possible offline teacher/pseudo-target comparison; its diffusion/trajectory machinery is not a plausible ≤2ms student path.[11]
- No newly published real-time game-SR or student-distillation result cleared the materiality bar. NanoVSR, FoRM, GRAPE, GSASR, and GS-STVSR remain the relevant existing baselines.

### Frame Extrapolation / Generation
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** — Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11; arXiv:2608.10423. It uses a sparse tiled layered image-space cache with predictive shading and localized refreshes, reporting multi-frame extrapolation with explicit disocclusion handling. **Why it matters:** this remains the clearest non-neural competitive baseline/threat to OSS-FX; benchmark its visibility-cache approach against the Gaussian canvas before claiming a unique one-architecture advantage.[12]
- No new material single-network neural frame-extrapolation paper was found after the previous brief. SNM-VFI remains the useful confidence-gated OSS-FX evaluation reference, not a shipping-runtime architecture.

### Inference Optimization
- No new material TensorRT/INT8/fused-kernel paper cleared the gate. AMD’s RDNA 4 WMMA guidance remains directly actionable: fuse adjacent GEMMs to retain the intermediate in registers, use the documented layout transformation, and validate against hipBLAS.[7]

## Archive.org Findings
- Internet Archive Advanced Search is still noisy/non-authoritative here: exact Gaussian-SR returns three archived GitHub-list artifacts; exact GS-STVSR returns zero; GSASR, GRAPE, and STSS acronym searches resolve to unrelated material. No archived/taken-down version altered provenance for GS-STVSR, GSASR, GRAPE, or STSS.[20]

## GitHub / Open Source
- **RDNU** is the only adjacent upscaler worth a technical read this run: a GPL-3.0, RDNA3-focused decoupled-G-buffer neural upscaler with FP16/INT8 paths; its latest visible commit is a WMMA-accelerated INT8 1×1 convolution validated against a reference kernel. **Opportunity:** inspect its build/runtime and measurement harness for the HIP student; **not reusable in OSS** because GPL-3.0 and no demonstrated game integration/quality parity.[13]
- **ZeroFG** remains a low-risk watch item: Apache-2.0 Vulkan midpoint-only frame-generation V1 POC, host-owned presentation/sync/frame pacing, and no independently validated standalone build. It is an integration/portability signal, not a quality competitor.[14]
- No credible new open DLSS/FSR/XeSS analysis tool appeared in the current GitHub searches. Other hits were generic video tooling, unrelated medical/3DGS work, or zero-star WIPs.[17]

## News
- **AMD (confirmed):** FSR SDK 2.3 makes ML FSR Upscaling 4.1.1 available on RDNA 3 RX 7000 discrete GPUs; it also updates FSR Frame Generation 4.0.1 input processing for motion-vector/camera corner cases. This widens the practical baseline OSS must beat and reinforces keeping MV/camera metadata in the vendor-neutral input contract.[6]
- **Intel (confirmed, Aug. 10):** Intel announced launch enablement of XeSS 3 SR, Multi-frame Generation, and Xe Low Latency for *STAR WARS: Galactic Racer* and *Tomb Raider: Legacy of Atlantis*. Competitive positioning now needs a credible plugin/DLL-shim deployment story, not only a model-quality claim.[9]
- **Apple (confirmed current platform direction):** MetalFX’s redesigned temporal upscaler uses Neural Engine/Neural Accelerators on M5 Pro/Max; the framework now supports dynamic-resolution subrectangles, supplied motion vectors, and distortion fields. This is a concrete contract to map against when the Metal student backend starts.[8]
- **NVIDIA / DLSS:** the reviewed NVIDIA technical feed had no new DLSS architecture disclosure. Google News RSS showed game-level DLSS 4.5/FSR 4 integration headlines only; treat those as unverified title-support leads, not platform changes.[18][19]

## Action Items / Follow-ups
- [ ] **P0 operational:** repair/restart the v7 trainer. The public dashboard remains internally contradictory (“live” while trainer is offline) and frozen at 36,650 / 0.19268 for 195h.
- [ ] Add GS²CI-style parent-child spawn controls: candidate ratio, mean-opacity guard, and hard canvas-count budget. Measure PSNR/LPIPS, spawn rate, canvas bytes, and raster time against the existing loss-adaptive density policy.
- [ ] Deep-read/benchmark Amulet as the explicit-visibility OSS-FX baseline; measure disocclusion error and multi-frame latency against direct `t_query` Gaussian rendering.
- [ ] Audit RDNU’s INT8 WMMA harness for test methodology only; build OSS’s own Apache-compatible HIP microbenchmark after #12/#13 establish packed-layout/L2 budgets.
- [ ] Define a Metal input-adapter contract now: dynamic-resolution subrects, exact MVs, camera parameters, and distortion/reactive-mask-equivalent data.

## Sources

[1] https://opensupersampling.com
[3] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open
[6] https://gpuopen.com/learn/amd-fsr-sdk-2-3-blog
[7] https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1
[8] https://developer.apple.com/metal/whats-new
[9] https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships
[10] https://arxiv.org/abs/2608.13502
[11] https://arxiv.org/abs/2608.09405
[12] https://arxiv.org/abs/2608.10423
[13] https://github.com/Adam-AtlasSoftware/RDNU
[14] https://github.com/hy300leosquizz-ctrl/ZeroFG
[17] https://api.github.com/search/repositories?q=neural+upscaler+pushed%3A%3E%3D2026-07-18&sort=updated&order=desc&per_page=10
[18] https://developer.nvidia.com/blog
[19] https://news.google.com/rss/search?q=DLSS+4+when%3A14d%26hl=en-US%26gl=US%26ceid=US%3Aen
[20] https://archive.org/advancedsearch.php
