# OSS Research Brief — 2026-09-06

## Project Status Summary

- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke` (local checkout).
- **Open issues:** 12. The immediate architecture/deployment queue remains #4 (≤1M student), #7 (FSR 4 teacher), #8 (vendor kernels), #12 (packed low-rank canvas), #13 (Nsight L2), and #14 (VRAM footprint).[3]
- **README snapshot:** OSS is pre-alpha v7 research: native `(x,y,t)` Gaussian canvas and fractional-time OSS-FX; the shipping student and non-CUDA backends remain unbuilt.[4]
- **Training dashboard:** `srcnn-v7.0-pico-005` remains at step **36,650**/100K with total loss **0.1926767** (SR Charbonnier 0.0235991; FG Charbonnier 0.0764263). The dashboard still labels it “training now,” but its health endpoint says the trainer is **offline, stuck for 173h**; watcher, Worker, and R2 are healthy.[1][2]
- **Operational conclusion:** there has been no metric progress since the prior brief; the dashboard’s live-copy/health contradiction should not be read as recovery. Trainer recovery is P0.

## New Papers (arXiv)

Targeted Atom searches were rate-limited (HTTP 429); this run fell back to the current arXiv category listings and primary abstract pages. No new paper passed the materiality gate relative to the already-catalogued daily-brief/bibliography set.

### Gaussian Splatting / Temporal

- **TileGS: Tile-Local Depth Binning for Gaussian Splatting Rasterization** — Wei Tan, Matias Turkulainen, Lauri Ilola, Hamed Rezazadegan Tavakoli, Juho Kannala; 2026-09-03; arXiv:2609.03613. Tile-local front-to-back depth ranges plus selective repair report a 1.44× mean raster-kernel speedup over gsplat on RTX 4090 while matching its output to numerical noise. **Why it matters:** it is a concrete alternative to OSS’s planned tile sorting/binning work (#3); replicate its workload accounting on OSS’s 2D temporal tiles before adopting it, because its remaining bottleneck is geometry-attribute traffic rather than reduced bytes.[5]
- **Already tracked, no new bibliography entry:** CC-4DGS, EvoGS, and Atlas remain relevant controls for packed temporal-canvas state, residual confidence/spawn gating, and active-Gaussian residency respectively. None displaces exact engine motion vectors as OSS’s primary motion signal.

### Super-Resolution / Upscaling

- No newly material real-time SR, ≤1M student-distillation, or engine-MV-conditioned result surfaced. Keep NanoVSR / reparameterized CNN work as the student control; the blocker is deployment measurement, not paper selection.

### Frame Extrapolation / Generation

- No new material single-network extrapolation result surfaced. OSS-FX’s architectural distinction remains: render the same time-aware canvas at a different `t_query`, rather than adding a second generation network.

### Inference Optimization

- TileGS is the only material fresh/reconfirmed optimization lead. Its profiling makes a useful caution: a speedup may come from less traversal work even when DRAM traffic increases. For OSS, benchmark full-frame time, instruction count, L2 hit rate, DRAM traffic, and compositing error—not occupancy alone.[5]

## Archive.org Findings

- Archive.org advanced search returned only three noisy GitHub-archive items for `"gaussian splatting" super-resolution`; `"frame generation" "deep learning"` returned an unrelated video-generation registration and archived tool builds. Targeted terms for GS-STVSR, GRAPE, and STSS returned no usable matching mirror; GSASR was a false CIA-record match. This is an inventory limitation, not evidence that preserved copies do not exist.[6][7]

## GitHub / Open Source

- **Integration-market signal:** `KillerPixelCrew/ReScaleFrame` was created 2026-09-05, is public GPL-3.0 C++, and describes reverse-engineering games to add modern upscaling/frame generation; its topics include XeSS and game modding. It is not an OSS dependency or a validated competitor yet (0 stars), but it validates the DLL/shim integration surface as an active open-source area.[8]
- **Watch, not adopt:** MAKO continues to present a Vulkan/Linux graphics layer around Lossless Scaling frame generation. It reinforces demand for vendor-neutral runtimes, but its optical-flow/overlay framing is architecturally different from an engine-buffer-conditioned Gaussian canvas.[9]
- Searches found no credible new open-source DLSS/FSR/XeSS analysis tool that changes OSS architecture or positioning.

## News

- **Confirmed competitive bar:** NVIDIA’s August 25 Gamescom update says DLSS 4.5 now includes a second-generation Super Resolution transformer, Dynamic/6× Multi Frame Generation, enhanced Frame Generation, and a second-generation Ray Reconstruction transformer.[10]
- No primary AMD FSR, Intel XeSS/Arc, Apple Metal/Apple Silicon, Unreal, or Unity upscaling announcement was verified inside the 7–14-day window. RSS supplied secondary and low-quality claims only; they are leads, not findings.[11]
- **Positioning implication:** do not claim feature parity with DLSS 4.5. Emphasize OSS’s vendor-neutral, inspectable engine-buffer path and single canvas for SR plus fractional-time extrapolation.

## Action Items / Follow-ups

- [ ] **P0:** recover/diagnose `srcnn-v7.0-pico-005`; dashboard status says 173h stuck at step 36,650 while publication infrastructure is healthy.
- [ ] **#3 / #12 / #13:** run a TileGS-style local-depth-binning ablation only after baseline packed-canvas metrics exist; record frame time, L2/DRAM counters, instruction count, and output error.
- [ ] **#4 / #9:** benchmark NanoVSR-style reparameterized CNN against RepVGG/ECBSR/RepSR in the fused ≤1M TensorRT graph.
- [ ] **Integration watch:** inspect ReScaleFrame’s game-input interception only if it demonstrates reusable capture/shim engineering; GPL-3.0 precludes direct incorporation into Apache-2.0 OSS.
- [ ] **Competitor watch:** track DLSS 4.5’s transformer/RR deployment, but keep OSS claims scoped to measured output and architecture—not proprietary quality parity.

## Sources

[1] https://opensupersampling.com/data.json
[2] https://opensupersampling.com/status.json
[3] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[4] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[5] https://arxiv.org/abs/2609.03613
[6] https://archive.org/advancedsearch.php?q=%22gaussian+splatting%22+super-resolution&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=year&rows=10&output=json
[7] https://archive.org/advancedsearch.php?q=%22frame+generation%22+%22deep+learning%22&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=year&rows=10&output=json
[8] https://api.github.com/repos/KillerPixelCrew/ReScaleFrame
[9] https://api.github.com/repos/eugeniosegala/MAKO
[10] https://www.nvidia.com/en-us/geforce/news/gamescom-2026-nvidia-geforce-rtx-dlss-4-5-announcements
[11] https://news.google.com/rss/search?q=DLSS+4+OR+FSR+4+OR+XeSS+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen
