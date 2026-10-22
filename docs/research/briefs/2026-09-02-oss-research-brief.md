# OSS Research Brief — 2026-09-02

## Project Status Summary
- **Latest commits:** the local checkout exposes `b44278a test: add local Chrome Playwright dashboard smoke` as its reachable HEAD; `git log -10` returns that one commit.[unverified]
- **Open issues:** 12 open issues. The immediate performance gates are #12 packed low-rank canvas layout, #13 Nsight L2 profiling, #14 VRAM measurement, and #3 tile binning; #4/#7/#9 cover teacher→student distillation.[2]
- **README snapshot:** pre-alpha v7 is the N-D `(x,y,t)` Gaussian-canvas *teacher* architecture. The ≤1M shipping student and CUDA/HIP/Metal/Level Zero/Vulkan production paths remain unbuilt.[3]
- **Training dashboard:** Trainer is **offline**, stuck at **step 36,650 for 77h**; loss remains **0.19268**. Watcher (published 1s ago), Cloudflare Worker (HTTP 200 in 2138ms), and R2 origin are healthy.[1] No step/loss movement occurred since Sep. 1; this is a continuing outage, not a reset. The dashboard shell still contains stale v6.1/v6.2 copy, so treat the trainer-health row—not its static tier copy—as operational truth.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[EvoGS: Modeling Deformation Evolution for Dynamic Gaussian Splatting](https://arxiv.org/abs/2609.00994)** — Wei Dong, Shahram Shirani, Jun Chen, Han Zhou; **2026-09-01**; arXiv:2609.00994. It maintains a per-Gaussian deformation state, extrapolates from history, corrects from observations using temporal-residual statistics, and makes densification deformation/uncertainty aware.[4] **Why it matters to OSS:** direct v7 ablation for persistent primitive state: compare a cheap engine-MV prediction plus residual-memory confidence/gated spawning against the current native-time representation; never replace exact engine MVs with its learned motion path.
- **Already tracked / no duplicate bibliography entry:** ATGS (2608.30184), M³ISR (2608.22465), QuARC-GS (2608.18285), DeGS (2608.02099), MotionAware-4DGS (2608.21828), and GS-STVSR (2604.18047) were rediscovered by the focused arXiv sweep but are already in `BIBLIOGRAPHY.md`.

### Super-Resolution / Upscaling
- The focused arXiv Atom query returned HTTP 503 today, so no additional primary paper metadata was promoted from broad search. The direct game-SR/≤1M-student relevance gate therefore remains unchanged: GS-STVSR, GSASR, GRAPE, NanoVSR, CDA-VSR, and deployment-aware INT8 work are the controls; do not substitute their reported latencies for a fused OSS 1080p-LR profile.[unverified]

### Frame Extrapolation / Generation
- **[Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading](https://arxiv.org/abs/2608.10423)** — Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; **2026-08-11**; arXiv:2608.10423. A layered image-space cache is predictively shaded and selectively refreshed to extrapolate from a 60 Hz shading rate to a claimed up-to-250 Hz 4K display rate with explicit disocclusion handling.[5] **Why it matters to OSS:** it is the closest non-neural systems competitor to OSS-FX. Benchmark its refresh/disocclusion policy against Gaussian-canvas fractional-time rendering; threat is product-surface overlap, opportunity is its cache-lifetime/tile scheduler design.
- Existing MGMVFI (2608.22861), SPEED (2607.15585), SNM-VFI (2608.13460), and ANVIL remain useful residual/refinement controls, but they are interpolation or substantially heavier models—not evidence against OSS’s one-canvas extrapolation claim.[unverified]

### Inference Optimization
- No new direct TensorRT/INT8/kernel-fusion paper materially changes the near-term order: first measure #12 bytes/Gaussian, #3 bin/sort cost, #13 L2 hit rate, and end-to-end raster latency. DeGS/RoofGS remain the relevant 3DGS dataflow/roofline references already in the bibliography.[unverified]

## Archive.org Findings
- Advanced Search found three noisy GitHub snapshot/reading-list items for `"gaussian splatting" super-resolution`, mostly unrelated results for `"frame generation" deep learning`, and zero for `"real-time upscaling" game`; none is a useful primary paper archive.[6]
- Targeted Wayback CDX confirms one June 7 capture of **3DGS³** (`arXiv:2605.11489`) and three historical captures of the **GSASR** GitHub repository. It found no capture for GS-STVSR or GRAPE. This is archive coverage—not evidence of withdrawal/takedown.[6]

## GitHub / Open Source
- **[IE-SRGS](https://github.com/ZheXues/IE-SRGS)** is a newly surfaced AAAI 2026 3DGS-SR implementation: public, last updated Aug. 29, one star, and no declared SPDX license.[7] **Watch only:** inspect its internal/external knowledge fusion as a teacher-side quality control; its pipeline is offline 3D reconstruction, not an OSS game-runtime competitor.
- **[OptiScaler](https://github.com/optiscaler/OptiScaler)** is actively updated (Sep. 2; 10.2k stars; GPL-3.0) and bridges DLSS/XeSS/FSR inputs and several frame-generation paths across hardware.[9] **Integration signal:** it is the strongest open compatibility/shim reference, but its injection approach is not a substitute for OSS’s own controlled DLL-shim contract and capture validation.
- Recent GitHub REST searches otherwise produced mostly zero-star demos/course work; no production-grade open DLSS/FSR/XeSS analysis tool or OSS-equivalent engine-input Gaussian-SR/FG implementation was verified.

## News
- **NVIDIA baseline / threat:** DLSS 4.5’s second-generation transformer is trained and inferred in linear space; NVIDIA says it uses 5× the original transformer’s compute and couples it with Dynamic/6× Multi Frame Generation.[8] OSS should keep fractional-time quality, lighting/HDR stability, motion clarity, and latency as explicit comparative axes—not only PSNR.
- **AMD baseline / threat:** Redstone ships ML upscaling, ML FG, ray regeneration, and radiance-caching preview; its FG consumes current/previous rendered frames plus depth and motion vectors, estimates optical flow, and predicts per-pixel motion/appearance.[10] OSS’s opportunity is a smaller single-canvas FG surface, but its integration benchmark must include the same inputs and Anti-Lag/latency accounting.
- **Apple opportunity:** the current MetalFX API exposes temporal scaling from color+depth+motion and frame-interpolator interfaces.[11] This reinforces retaining a clean Metal input/raster boundary even before an Apple backend exists.
- Google News RSS (headline-level only) reported Sep. 1 DLSS 5 game launch coverage and Gamescom-era FSR/XeSS additions; no further primary Intel, Unreal, or Unity release note was verified today. Treat these as monitoring leads, not architecture facts.[unverified]

## Action Items / Follow-ups
- [ ] **P0 ops:** recover, terminate, or document the trainer. It is still stuck at 36,650/loss 0.19268 for 77h while the publishing stack is healthy.[1]
- [ ] **EvoGS ablation:** add residual-memory confidence and deformation-aware spawn gating after exact engine-MV warp; measure temporal identity survival, disocclusion quality, active primitives/tile, and raster cost.[4]
- [ ] **Amulet competitive test:** build a sparse-layer/cache baseline or at minimum adopt its stale-tile/disocclusion metrics for OSS-FX; compare latency, hole rate, and quality at 2×/4× display rates.[5]
- [ ] **Vendor-input contract:** test HDR-linear color, depth, MVs, camera cuts, WPO/forced velocities, dynamic resolution, and frame-generation latency against the AMD/XeSS/MetalFX input models before beginning cross-vendor kernel work.[10][11]
- [ ] **Performance gates first:** close #12/#13/#14 before committing a canvas packing, quantization, or fused-kernel layout.[2]
- [ ] **Watch IE-SRGS and OptiScaler:** inspect IE-SRGS’s license/training graph; separately use OptiScaler only as an integration-compatibility reference, not code to import.[7][9]

## Sources

[1] https://opensupersampling.com
[2] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[3] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[4] https://arxiv.org/abs/2609.00994
[5] https://arxiv.org/abs/2608.10423
[6] https://archive.org/advancedsearch.php
[7] https://github.com/ZheXues/IE-SRGS
[8] https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-gen-6x-2nd-gen-transformer-super-res/
[9] https://api.github.com/repos/optiscaler/OptiScaler
[10] https://gpuopen.com/learn/amd-fsr-redstone-developers-neural-rendering/
[11] https://developer.apple.com/documentation/metalfx
