# OSS Research Brief — 2026-09-05

## Project Status Summary

- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` is the newest visible local commit.
- **Open issues:** 12. The material queue is unchanged: #4 ≤1M student distillation, #7 FSR 4 teacher signal, #8 cross-vendor kernels, #12 packed low-rank canvas, #13 Nsight L2 profile, and #14 VRAM measurement.
- **README snapshot:** pre-alpha v7 teacher research; v7 uses an N-D `(x,y,t)` Gaussian canvas. The shipping student and HIP/Metal/Level Zero/Vulkan paths are still unbuilt.
- **Training dashboard:** `srcnn-v7.0-pico-005` remains at **36,650 / 100,000**, total loss **0.19268** (SR Charbonnier **0.02360**, FG Charbonnier **0.07643**), while the health row says **trainer offline / stuck 149h**; watcher, Worker, and R2 are healthy.[1] This is no step or loss movement from 2026-09-04 (125h); the regenerated-at timestamp advanced, so the publisher is healthy while the trainer is not. The live/ETA copy is stale; trainer recovery remains P0.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **TileGS: Tile-Local Depth Binning for Gaussian Splatting Rasterization** — Wei Tan, Matias Turkulainen, Lauri Ilola, Hamed Rezazadegan Tavakoli, Juho Kannala; submitted 2026-09-03; arXiv:2609.03613. It reorganizes long globally sorted tile streams into depth-local ranges with selective repair, reporting 1.44× mean raster-kernel speedup over gsplat on RTX 4090 while matching output within numerical noise.[8] **Why it matters to OSS:** direct custom-rasterizer ablation for #12/#13—test tile-local bins against the 2D temporal canvas; its profiling says geometry attributes remain the dominant traffic, strengthening the packed-canvas priority.

- **TruncGradGS: Improved 3D Gaussian Splatting via Truncated Gradient Updates** — Theo Morales, Nhat-Quynh Le-Pham, Robin Atkins, Binh-Son Hua; submitted 2026-09-03; arXiv:2609.03534. It uses a piecewise truncated-gradient formulation to counter distant-pixel gradient vanishing and reports stability gains for static and dynamic 3DGS.[9] **Why it matters to OSS:** teacher-side ablation lead for loss-adaptive spawning/covariance updates; not a shipping-rasterizer technique and not yet bibliography-worthy without a v7-specific validation.

### Super-Resolution / Upscaling

- No material new real-time/student-SR paper. The scan re-confirmed **NanoVSR** (Pawlicki, Kańduła, Pucek, Dobies; 2026-07-11; arXiv:2607.10495), a 644K fully-convolutional VSR model that structurally reparameterizes to standard convolutions; it is already tracked.[2] **Why it matters to OSS:** it remains a strong deployment control for the ≤1M student, but its lack of explicit engine-MV alignment makes it a control, not an OSS canvas replacement.

### Frame Extrapolation / Generation

- No newly submitted material paper. **Amulet** (Künzel et al.; 2026-08-11; arXiv:2608.10423) remains the closest system-level comparator: a sparse layered image-space cache with predictive tile lifetimes and adaptive shading, reporting a 60 Hz shading-to-240 Hz extrapolation use case.[3] **Why it matters to OSS:** directly test its explicit disocclusion/cache-validity strategy against v7's Gaussian canvas plus exact engine motion vectors. Already tracked; no bibliography append.

### Inference Optimization

- No new material TensorRT/INT8/kernel-fusion paper. Current actionable control remains NanoVSR's fused-standard-convolution deploy graph.[2]

## Archive.org Findings

- Archive.org advanced-search checks for `gaussian splatting super-resolution`, `frame generation deep learning`, and `real-time upscaling game` were noisy: respectively 3 GitHub-snapshot-like records, 228 broad/unrelated records, and 107 broad/unrelated records. Exact `GS-STVSR` returned zero; `GSASR` returned one unrelated CIA item.[4]
- No trustworthy archived alternate/versioned copy of GS-STVSR, GSASR, GRAPE, or STSS was found in this run. This is an Archive.org discovery limitation, not proof of absence.

## GitHub / Open Source

- **dlss-unlocked** is now 42 stars, MIT-licensed, created 2026-09-04 and pushed today; it claims DX12 DLSS upscaler, frame generation, and neural-rendering feature unlocking for RTX games with native DLSS2/3.[5] **Threat/opportunity:** market evidence for a drop-in shim, but no quality, title-coverage, security, or anti-cheat claim was independently validated. Track; do not use as an OSS dependency.
- **neural-upstream** is a tiny (2-star, MIT) C++ project claiming to run DLSS 5 neural rendering upstream of the upscaler.[6] Treat as an unvalidated integration lead, not competitive evidence.
- **ReScaleFrame** is a newly created GPL-3.0 framework targeting Ace Combat 7 with XeSS SR, MFG, and XeLL. Its native scaffold and game fingerprint/status UI build, but injection, renderer hooks, SR, FG, and in-game rendering are explicitly unimplemented.[10] **Opportunity:** its separation of game plugin, shared orchestrator, vendor backends, loader, and overlay is a useful DLL-shim boundary reference; it is not a working competitor or dependency.
- No credible newly found open-source Gaussian-SR or DLSS/FSR analysis tool displaces OptiScaler or the already tracked research baselines.

## News

- **Competitive baseline (primary, 11 days old):** NVIDIA says DLSS 4.5 Ray Reconstruction is available through its NVIDIA-app early-access channel; the update extends the DLSS 4.5 stack’s second-generation SR transformer, Dynamic/6× MFG, and enhanced FG model.[11] Treat this as a higher quality/feature bar, not proof of equivalent latency or an OSS architectural replacement.
- **Apple opportunity (not a 7–14-day announcement):** WWDC26’s Metal 4 material adds quantized tensor formats/scale factors and says the redesigned MetalFX temporal upscaler can accept motion vectors, process dynamic-resolution subrectangles, and use distortion fields; Apple also documents custom small networks directly in Metal shaders via TensorOps.[12] This is the concrete design target for a future Metal student/raster boundary—retain linear HDR, depth, motion vectors, dynamic-resolution, and postprocess-distortion contracts.
- No primary AMD FSR/Redstone, Intel XeSS/Arc, Unreal, or Unity **new** update within the last 14 days was verified; secondary results were not promoted to findings.

## Action Items / Follow-ups

- [ ] **P0:** recover/diagnose `srcnn-v7.0-pico-005`; reconcile stale live/ETA UI with the **149h** offline trainer state.
- [ ] **Deep read:** Amulet's layered-cache invalidation, predictive tile-lifetime scheduler, disocclusion behavior, and equal-budget latency methodology; implement an OSS-FX comparison plan.
- [ ] **#4 student:** reproduce NanoVSR-style structural reparameterization alongside RepVGG/ECBSR/RepSR under the ≤1M, fused TensorRT deployment graph.
- [ ] **#12/#13:** use existing 4DGS compression/streaming references only after L2-residency, temporal-primitive-identity, disocclusion, VRAM, and raster-cost gates exist.
- [ ] **Raster deep read:** reproduce TileGS’s tile-local binning with the 2D v7 canvas; capture instruction count, geometry-attribute traffic, compositing error, and frame time before adopting the reorganization.
- [ ] **Metal backend contract:** turn the WWDC26 input/scheduling guidance into a minimal Metal adapter spec before committing student/kernel interfaces.
- [ ] **Competitor watch:** follow DLSS 5 as proprietary neural rendering, while keeping OSS's public claim scoped to deterministic, engine-buffer-conditioned SR + extrapolation.

## Sources

[1] https://opensupersampling.com
[2] https://arxiv.org/abs/2607.10495
[3] https://arxiv.org/abs/2608.10423
[4] https://archive.org/advancedsearch.php?q=gaussian+splatting+super-resolution&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=date&fl%5B%5D=mediatype&rows=10&output=json
[5] https://api.github.com/repos/ShyVortex/dlss-unlocked
[6] https://api.github.com/repos/matiasLombo/neural-upstream
[7] https://news.google.com/rss/search?q=%28DLSS+OR+FSR+OR+XeSS+OR+%22game+upscaling%22+OR+%22MetalFX%22%29+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen
[8] https://arxiv.org/abs/2609.03613
[9] https://arxiv.org/abs/2609.03534
[10] https://github.com/KillerPixelCrew/ReScaleFrame
[11] https://www.nvidia.com/en-us/geforce/news/gamescom-2026-nvidia-geforce-rtx-dlss-4-5-announcements/
[12] https://developer.apple.com/wwdc26/guides/metal/
