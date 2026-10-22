# OSS Research Brief — 2026-08-15

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke` (the local checkout exposes no other reachable recent commit).
- **Open issues:** 12. Critical sequencing remains #12 packed low-rank canvas layout → #13 Nsight L2 profile → #14 VRAM measurement; student distillation (#4/#9), FSR 4 teacher signal (#7), cross-vendor kernels (#8), and the G-buffer ablation (#10) are open.
- **README snapshot:** v7 remains the active research teacher: a persistent N-D `(x,y,t)` Gaussian canvas with `V_xt` correlation. The ≤1M-parameter shipping student and non-CUDA kernels are not yet built.
- **Training dashboard:** `data.json` labels `srcnn-v7.0-pico-005` active, but its latest metric remains step **36,650 / 100,000**, total loss **0.19268**, with no GPU telemetry. This is unchanged from the prior brief’s trainer-offline / 134h-stuck observation; the dashboard’s “training now” copy conflicts with its frozen metric state.[1]

## New Papers (arXiv)

Screened the requested Gaussian/temporal, VSR/interpolation, deployment, and game/MV terms. Broad Atom queries were noisy (many photonics, medical, and generic-generation false positives), so only primary abstract pages that affect the architecture or deployment path are retained below. Already-tracked papers are not duplicated in `BIBLIOGRAPHY.md`.

### Gaussian Splatting / Temporal
- **[RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](https://arxiv.org/abs/2603.13783)** (arXiv:2603.13783; Xuezhen Wang, Li Ma, Yulin Shen, Zeyu Wang, Pedro V. Sander; 2026-03-14; **already tracked**) — explicitly models each 3D Gaussian’s continuous-time behavior and uses flow-guided initialization/supervision plus triple-rendering supervision to reduce interpolation ghosting. **Why it matters to OSS:** closest external validation of v7’s native time-axis premise; add its time-supervision and large-motion/occlusion ablations as controls, while preserving OSS’s engine-MV input and extrapolation objective.[2]
- **[GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting](https://arxiv.org/abs/2604.18047)** (arXiv:2604.18047; Mingyu Shi et al.; 2026-04-20; **already tracked**) — continuous-time 2D Gaussian VSR with flow alignment and covariance resampling. **Why it matters to OSS:** direct precedent for the raster-space covariance operator, but it does not supply a persistent native-time canvas or future extrapolation beyond its observed pair.

### Super-Resolution / Upscaling
- **[Compressed-Domain-Aware Online Video Super-Resolution](https://arxiv.org/abs/2603.07694)** (arXiv:2603.07694; Yuhang Wang, Hai Li, Shujuan Hou, Zhetao Dong, Xiaoyao Yang; 2026-03-08; newly material) — CDA-VSR coarse-warps with supplied motion vectors, learns only local residual offsets, gates fusion from residual maps, and allocates compute by frame type; its abstract reports >2× inference speed over TMP with +0.13 dB PSNR on REDS4. **Why it matters to OSS:** high-value student control: exact engine MVs should eliminate much of its residual-alignment burden, while its reliability gating is a concrete design for invalid-MV/disocclusion handling.[3]
- **[NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices](https://arxiv.org/abs/2607.10495)** (arXiv:2607.10495; Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11; **already tracked**) — fully convolutional VSR structurally reparameterized to standard inference convolutions; the 644K model reports 27.2 FPS on Jetson Orin NX 16GB/25W. **Why it matters to OSS:** strongest deployable student-backbone comparator; its lack of explicit flow is a useful contrast to an engine-MV-prealigned OSS student.[4]

### Frame Extrapolation / Generation
- **[ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors](https://arxiv.org/abs/2603.26835)** (arXiv:2603.26835; Shibo Liu; 2026-03-27; **already tracked**) — pre-aligns with exposed motion vectors, removes learned flow and sampling, and confines residual refinement to INT8-friendly convolutional operations. **Why it matters to OSS:** validates treating engine MVs as a hardware-friendly first-class prior; it is interpolation, not proof of OSS-FX’s <2ms extrapolation path.

### Inference Optimization
- No new primary TensorRT, INT8-SR, or neural kernel-fusion paper cleared the materiality gate. The decisive work is empirical: implement #12’s packed state, measure tile-bin/L2 behavior in #13, then quantify #14’s VRAM and choose fusion/quantization from occupancy and bandwidth rather than paper claims.

## Archive.org Findings
- Advanced Search returned only noisy secondary GitHub snapshots for `"gaussian splatting" super-resolution`, generic DLSS/FluidFrames artifacts for `"frame generation" deep learning`, and zero results for `"real-time upscaling" game`.[5]
- Direct Archive.org keyword inventory returned no GS-STVSR item; `GSASR`, `GRAPE`, and `STSS` were acronym-polluted and did not yield a verified primary paper artifact. Wayback CDX returned no archived capture for either GS-STVSR or RetimeGS abstract URLs at query time. Treat this as **no verified archive copy**, not evidence of removal.

## GitHub / Open Source
- No newly pushed (since 2026-08-01) game-focused Gaussian-SR repo, open frame-generation implementation, or DLSS/FSR/XeSS analysis tool was verified by GitHub REST search. The only fresh `neural upscaler` lead, [`julesklord/bananascaler`](https://github.com/julesklord/bananascaler), is a zero-star MIT WIP TUI wrapper—not competitive technical signal.[6]
- Relevant baselines remain available: [ContinuousSR / Pixel-to-Gaussian](https://github.com/peylnog/ContinuousSR) (892 stars, last commit January 2026) and [GSASR](https://github.com/ChrisDud0257/GSASR) (Apache-2.0, 102 stars). Neither represents new activity this run; both remain reference implementations for 2D Gaussian SR.[7][8]
- AMD FidelityFX SDK was last pushed 2026-06-24 and NVIDIA/DLSS 2026-06-23; their recent metadata refreshes do not themselves establish a new runtime capability.[9][10]

## News
- **Threat / integration bar:** AMD’s current FSR SDK 2.3 packages ML FSR Upscaling **4.1.1**, Frame Generation **4.0.1**, and Ray Regeneration **1.2.0**; it adds FSR 4.1 support for RDNA 3 and permits eligible FSR 3.1+ DX12 games to receive driver-provided ML-upscaler upgrades. This raises the practical integration/maintenance bar for OSS’s proposed DLL-shim path.[11]
- Intel’s Arc G-Series material positions **XeSS 3** as SR + multi-frame generation + low-latency on next-generation handhelds. It is not a fresh 7–14 day announcement, but it reinforces that OSS Pico must be benchmarked on handheld-class hardware, not inferred from desktop TensorRT.[12]
- No new primary NVIDIA, Apple/MetalFX, Unreal, or Unity upscaling announcement dated in the prior 14 days cleared verification. DLSS 4.5’s second-generation transformer / 6× Dynamic MFG remains the published NVIDIA baseline, not a new run delta.[13]

## Action Items / Follow-ups
- [ ] **Critical:** inspect/restart the `srcnn-v7.0-pico-005` trainer on the 3080 Ti host. Treat the active dashboard label plus five-day frozen step and null GPU telemetry as a metrics-state contradiction, not evidence of training.
- [ ] Add CDA-VSR as a student ablation: exact-MV coarse warp + tiny local residual refinement + confidence/disocclusion gate; benchmark against NanoVSR-style reparameterized convolution while retaining the OSS Gaussian rasterizer.
- [ ] Keep #12 → #13 → #14 ordered; no paper substitutes for measured packed-state, cache, and VRAM decisions.
- [ ] Preserve the v7 distinction against RetimeGS/GS-STVSR: evaluate long-lived primitive identity and **future extrapolation** under disocclusions, not only continuous in-between rendering.
- [ ] Maintain a compatibility matrix for AMD FSR SDK 2.3’s input contract (linear color, depth, screen-space MVs, jitter) against the planned OSS capture/DLL contract before integration work begins.

## Sources

[1] https://opensupersampling.com/data.json — OSS training dashboard data
[2] https://arxiv.org/abs/2603.13783 — RetimeGS
[3] https://arxiv.org/abs/2603.07694 — Compressed-Domain-Aware Online Video Super-Resolution
[4] https://arxiv.org/abs/2607.10495 — NanoVSR
[5] https://archive.org/advancedsearch.php?q=%22gaussian%20splatting%22%20super-resolution&fl%5B%5D=identifier&fl%5B%5D=title&output=json&rows=8 — Internet Archive advanced search
[6] https://api.github.com/repos/julesklord/bananascaler — bananascaler repository metadata
[7] https://github.com/peylnog/ContinuousSR — ContinuousSR repository
[8] https://github.com/ChrisDud0257/GSASR — GSASR repository
[9] https://api.github.com/repos/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK — FidelityFX SDK repository metadata
[10] https://api.github.com/repos/NVIDIA/DLSS — NVIDIA DLSS repository metadata
[11] https://gpuopen.com/amd-fsr-sdk/ — AMD FSR SDK 2.3
[12] https://newsroom.intel.com/client-computing/intel-arc-g-series-processors-set-a-new-standard-for-handheld-pc-gaming — Intel Arc G-Series / XeSS 3
[13] https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-gen-6x-2nd-gen-transformer-super-res/ — NVIDIA DLSS 4.5
