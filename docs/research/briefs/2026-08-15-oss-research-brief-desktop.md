# OSS Research Brief — 2026-08-15

## Project Status Summary
- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated training controls and the dashboard panel. Recent trainer work preserved the v7 recovery snapshot, fixed cross-OS resume, enabled FlashAttention/TF32/prefetch, and kept BF16 smoothing in FP32.
- **Open issues:** 11 open: low-rank rasterizer (#1), tile bin (#3), ≤1M student distillation (#4), cross-vendor ports (#8), student reparameterization (#9), and packed-canvas/L2/VRAM work (#12–14).[2]
- **README snapshot:** v7 is an N-D `(x,y,t)` Gaussian teacher. The shipping artifact remains an untrained ≤1M CNN student with a <2 ms target; HIP/Metal/Level Zero/Vulkan kernels and DLL shim remain designed-only.
- **Training dashboard:** `srcnn-v7.0-pico-005` shows step **36,650 / 100,000**, loss **0.19268**, viz @36K, and “live”/ETA ~1.1d. But Trainer is **offline, stuck at 36,650 for 147h**; GPU stats are unavailable. Watcher, CF Worker, and R2 are healthy. This is a trainer-host outage, not training progress.[1]

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **GS²CI: Robust Gaussian Splatting For Snapshot Compressive Imaging via Large Vision Model Priors** — Yanming Yang, Chenxi Song, Ping Wang, Xin Yuan, Chi Zhang; submitted Aug. 13, 2026; arXiv:2608.13502. It uses opacity-guided splitting/growth regulation, explicit candidate-ratio limits, and Gaussian-count constraints under ambiguous temporal measurements. **Why it matters:** off-domain and not a core baseline, but its bounded-density controls are a useful v7 parent-child spawning ablation for weak motion/disocclusion evidence.[4]
- The broad arXiv Atom search timed out/returned HTTP 429 after two requests; direct arXiv abstract pages were used for the verified candidates. No other newly verified 2D/4D Gaussian SR paper displaced the existing ACA-GS, CLEAR, FAST-GS, or DeGS leads.

### Super-Resolution / Upscaling
- No new material student-scale/game-SR paper. The current decision pressure remains issue #12/#13 measurement: packed-layout bandwidth/L2 and student latency before choosing FP8 vs INT8 or a fusion topology.

### Frame Extrapolation / Generation
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** — Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; submitted Aug. 11, 2026; arXiv:2608.10423. A sparse tiled, layered image-space cache predictively stores potentially visible geometry/shading, refreshes stale tiles, and explicitly covers disocclusion; the abstract reports 60 Hz shading → 240 Hz display and up to 250 Hz at 4K. **Why it matters:** direct non-neural OSS-FX baseline and a candidate deterministic visibility-cache companion to alpha-time canvas rendering. It challenges a pure “one neural architecture” story but may improve multi-frame extrapolation without a second neural network.[3]

### Inference Optimization
- No new material TensorRT/INT8/kernel-fusion paper was verified this run.

## Archive.org Findings
- Advanced Search found only three Gaussian-SR records, all archived GitHub list artifacts; the frame-generation query returned consumer tool builds plus an unrelated diffusion registration. “Real-time upscaling game” and GS-STVSR returned zero; GSASR, GRAPE, and STSS acronym results were unrelated/noisy. No archival version or takedown changes the bibliography.[6]

## GitHub / Open Source
- **Watch, not a competitor:** [ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG), updated Aug. 15, is an Apache-2.0 C++ experimental Vulkan/mobile frame-generation POC (one star). It validates interest in portable Vulkan FG but has no demonstrated reconstruction/canvas architecture, quality, or latency evidence.[7]
- The other current GitHub hits are generic video tooling, empty repos, or WIP neural upscalers. No credible new open DLSS/FSR/XeSS analysis tool surfaced.
- **Positioning:** no new open project combines exact engine MVs, a persistent Gaussian SR canvas, and single-architecture extrapolation. Amulet is the adjacent technical threat; OSS execution and validation are the greater risks.

## News
- **Verified primary Intel signal:** Intel says *STAR WARS: Galactic Racer* and *Tomb Raider: Legacy of Atlantis* will launch with XeSS 3 Super Resolution, Multi-frame Generation, and Xe Low Latency. Preserve a vendor-neutral input contract that can consume comparable MV/depth/reactive-mask-class inputs and target a practical Unreal/plugin integration path.[5]
- Google News RSS produced DLSS 4.5/FSR 4 game-integration and XeSS Unreal-plugin headlines, but no new AMD, NVIDIA, Apple/MetalFX, Unreal, or Unity primary release with an OSS architectural consequence was verified. Treat them as leads only.

## Action Items / Follow-ups
- [ ] **P0 operational:** repair v7 trainer-host execution. Step/loss are frozen at 36,650 / 0.19268; dashboard health now reports 147h offline.
- [ ] Deep-read/prototype Amulet-style layered visibility cache as deterministic OSS-FX fallback or evaluation baseline; measure disocclusion error and latency beyond midpoint extrapolation.
- [ ] Test GS²CI-style candidate-ratio cap plus opacity/count regularization against current parent-child spawning; report quality, canvas count, and #12 packed-layout bandwidth.
- [ ] Baseline the ≤1M student against NanoVSR/reparameterized-CNN controls with exact engine MV on/off and end-to-end raster+composite latency for TensorRT FP8/INT8.
- [ ] Track ZeroFG only for Vulkan/mobile integration choices; do not treat its POC as a technical competitor without source, quality, and latency evaluation.

## Sources

[1] https://opensupersampling.com — OpenSuperSampling training dashboard
[2] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open — OpenSuperSampling open issues API
[3] https://arxiv.org/abs/2608.10423 — Amulet arXiv
[4] https://arxiv.org/abs/2608.13502 — GS2CI arXiv
[5] https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships — Intel Gamer Days 2026
[6] https://archive.org/advancedsearch.php?q=%22gaussian+splatting%22+super-resolution%26fl%5B%5D=identifier%26fl%5B%5D=title%26rows=5%26page=1%26output=json — Internet Archive Gaussian splatting super-resolution search
[7] https://github.com/hy300leosquizz-ctrl/ZeroFG — ZeroFG repository
