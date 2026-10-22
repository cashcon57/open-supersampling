# OSS Research Brief — 2026-08-18

## Project Status Summary
- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated training controls and dashboard UI. The next recent changes preserve the v7 step-5K recovery snapshot, fix cross-OS checkpoint resume, enable FlashAttention/TF32/prefetch throughput work, and keep smoothing filters FP32 under BF16.
- **Open issues:** 12. Immediate blockers remain distilled student training (#4), cross-vendor kernels (#8), reparameterized student experiments (#9), packed canvas layout (#12), L2 profiling (#13), and v6.2 VRAM measurement (#14); #10 is the decoupled G-buffer encoder ablation.[2]
- **README snapshot:** v7 is explicitly still a HAT-Tiny teacher built on an N-D `(x,y,t)` Gaussian canvas. The ≤1M shipping CNN, OSS-FX integration, cross-vendor kernels, and DLL shim are not built.
- **Training dashboard:** the public page reports `srcnn-v7.0-pico-005`, step **36,650 / 100,000**, loss **0.19268**, and viz at step 36,000, yet service health reports the trainer **offline, stuck at step 36,650 for 219h**; its ETA (~1.1d) and GPU statistics are therefore stale. The teacher table also retains an old `srcnn-v6.2-pico-002` label. Watcher (published 0s ago), CF Worker (HTTP 200/148ms), and R2 origin (HTTP 200) are healthy. Treat the trainer as down and dashboard training/ETA metadata as stale, not as training progress.[1]

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- No newly material paper cleared the OSS relevance gate today. The targeted arXiv sweep did return **SplatGuide: Geometric Priors from 3D Gaussians for Pose-Free Novel View Synthesis** — Yejun Zhang, Zihan Wang, Xu Ji; 2026-08-17; arXiv:2608.16863 — but it is pose-free NVS, not temporal Gaussian reconstruction/SR; it should not enter the bibliography.
- The previous brief's GS²CI (bounded Gaussian-growth mechanics), FAST-GS (hard-motion temporal reference), and EndoMD-SLAM (static/transient update gating) remain the actionable leads; no newer result displaced them.[8]

### Super-Resolution / Upscaling
- No newly material real-time game-SR, video-SR student-distillation, or arbitrary-scale Gaussian-SR paper surfaced. MeanSR remains a teacher-side one-step perceptual-SR comparison, not a plausible ≤2ms student runtime.

### Frame Extrapolation / Generation
- No new material single-network neural frame-extrapolation result surfaced. Amulet remains the explicit visibility/disocclusion baseline to benchmark against OSS-FX's direct `t_query` render.

### Inference Optimization
- No new material TensorRT/INT8/kernel-fusion paper surfaced. This does **not** change the immediate implementation priority: resolve packed canvas/L2 behavior (#12/#13), then measure the student kernel path rather than extending teacher research.

## Archive.org Findings
- Internet Archive Advanced Search remains noisy for all three requested queries: Gaussian-SR returned three archived GitHub-list artifacts; frame-generation/deep-learning returned no results; real-time game upscaling returned 107 mostly unrelated emulator/API artifacts. Targeted provenance checks for GS-STVSR, GSASR, GRAPE, and STSS produced no usable archived/taken-down paper version. No bibliography change.[3]

## GitHub / Open Source
- **ZeroFG** is the only newly active adjacent repository worth retaining as a watch: Apache-2.0, Vulkan, host-supplied command buffer/resources, and a fixed midpoint (`phase=0.5`) pipeline. Its own README calls it a functional experimental POC with severe ghosting/smearing, poor disocclusion, meaningful GPU overhead, and no independently validated standalone build. **Opportunity:** borrow only its explicit Vulkan integration boundary; it is not a quality or architecture threat to the Gaussian canvas.[4]
- Other recent GitHub hits were generic video tooling, latent/video-generation upscalers, or zero-star WIPs. No credible new open DLSS/FSR/XeSS analysis tool appeared.

## News

### NVIDIA / DLSS
- No primary-source DLSS platform disclosure in the last 7–14 days. The relevant current competitive baseline remains DLSS 4.5: second-generation transformer ray reconstruction, a claimed 35% more compute capability and 20% more parameters at similar performance, deeper spatial awareness, and developer control over temporal accumulation. **Why it matters:** the quality target is moving toward learned joint spatial-temporal reconstruction; OSS needs measured temporal/disocclusion comparisons, not only the one-network argument.[5]

### AMD / FSR and Unreal
- AMD's current FSR SDK 2.3 exposes ML FSR Upscaling 4.1.1, ML Frame Generation 4.0.1, and Ray Regeneration 1.2.0; it permits driver-delivered upgrades for games already using the compatible API. The documented FG fixes explicitly cover motion-vector preprocessing and camera-information binding.[6]
- The UE plugin supports UE 5.2–5.7 and documents that missing WPO velocities produce blur/ghosting. **Why it matters:** OSS's capture/DLL contract must validate exact MVs, camera inputs, velocity coverage, resolution rectangles, and reactive/disocclusion semantics before model evaluation; this is a competitive integration requirement, not polish.[7]

### Intel / Apple / Engines
- No primary Intel XeSS/Arc, Apple MetalFX/Apple Silicon, Unreal, or Unity update inside the requested window survived source-quality filtering. Continue tracking them, but do not turn search snippets into architecture claims.

## Action Items / Follow-ups
- [ ] **P0 operational:** recover or formally retire the trainer state. The 219h offline report plus stale teacher-table/ETA metadata make the public dashboard unreliable.
- [ ] **P0 architecture:** execute #12 then #13: packed low-rank layout first, followed by Nsight L2 profiling; do not claim the 32B/Gaussian target before both measurements.
- [ ] Benchmark Amulet-style explicit visibility handling against native-time Gaussian rendering on disocclusions and multi-frame extrapolation.
- [ ] Deep-read GS²CI only for bounded parent-child spawn controls (candidate ratio, opacity guards, hard count budgets); keep its snapshot-imaging task out of OSS claims.
- [ ] Turn AMD/UE integration failure modes into capture acceptance tests: WPO/velocity coverage, camera binding, render/generation rectangles, and resize/present transitions.

## Sources

[1] https://opensupersampling.com
[2] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open
[3] https://archive.org/advancedsearch.php
[4] https://github.com/hy300leosquizz-ctrl/ZeroFG
[5] https://www.nvidia.com/en-us/geforce/news/dlss-4-5-ray-reconstruction-1000-rtx-games-apps-out-now
[6] https://gpuopen.com/amd-fsr-sdk
[7] https://gpuopen.com/learn/ue-fsr
[8] https://export.arxiv.org/api/query
