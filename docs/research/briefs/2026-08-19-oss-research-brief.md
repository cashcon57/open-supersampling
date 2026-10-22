# OSS Research Brief — 2026-08-19

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke`; this checkout exposes only that commit. The working tree already contains the Aug. 11–18 research briefs plus a modified bibliography.[unverified]
- **Open issues:** 12. The performance-critical chain remains **#12 packed low-rank canvas → #13 Nsight L2 profile → #14 VRAM measurement**; #1 low-rank rasterizer and #3 counting-sort tile bin are also open. Student distillation (#4/#7/#9), non-CUDA ports (#8), and the G-buffer ablation (#10) remain unstarted.[3]
- **README snapshot:** v7 is the active *teacher* architecture: native `(x,y,t)` Gaussians, `V_xt` covariance cross-correlation, and time-slice OSS-FX. The ≤1M shipping student, DLL shim, and HIP/Metal/Vulkan/Level Zero kernels remain unbuilt.[4]
- **Training dashboard:** `srcnn-v7.0-pico-005` is still marked active/training in `data.json`, at **step 36,650/100K** with latest total loss **0.19268**; `status.json` independently reports the trainer **offline/stuck for 242h** while watcher, CF Worker, and R2 are healthy.[1][2] The step has not advanced since yesterday; treat “training now” as stale metadata, not recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- No newly material Gaussian-temporal or 2D-GS SR paper was verified today. Targeted Atom API queries returned HTTP 429/503, so the scan fell back to current `cs.CV`/`cs.GR` listings; the fresh lists did not surface a direct GS-STVSR/GRAPE/v7 competitor.[5]

### Super-Resolution / Upscaling
- **[SFMformer: A Spatial-Frequency Modulation Transformer for Lightweight Image Super-Resolution](https://arxiv.org/abs/2608.17966)** (arXiv:2608.17966; Chih-Hsiang Yang, Chia-Min Lin, Ching-Yu Tsai, Yung-Che Wang, Jen-Shiun Chiang; 2026-08-19) — newly listed lightweight image-SR transformer; the abstract was unavailable during this run after arXiv rate limiting, so this is a title-level lead rather than a validated result. **Why it matters:** deep-read only if it reports a sub-1M / real-time graph or deployable frequency module for the student; otherwise it is teacher-side only.[5][6]

### Frame Extrapolation / Generation
- No newly material extrapolation/single-network paper was verified in the rate-limited scan.[5] **Amulet** remains the direct external systems comparator for OSS-FX disocclusion/refresh policy; **3DGS³** remains the closest product-surface comparator but still uses separate SR and interpolation modules (both already tracked).[unverified]

### Inference Optimization
- No new verified TensorRT, INT8, or kernel-fusion result.[5] This remains a measurement problem, not a paper-selection problem: land #12, profile L2 in #13, then measure #14 before choosing packing, fusion, or quantization boundaries.[3]

## Archive.org Findings
- Broad Archive.org inventory across the requested terms returned six mostly irrelevant/generic items (including GitHub snapshots and unrelated media), not usable preserved paper artifacts.[7]
- Targeted Wayback CDX lookups for GS-STVSR, QuADA-GS, and RBS timed out in this run. That is incomplete archive coverage, **not** evidence that the papers lack captures.

## GitHub / Open Source
- The GitHub searches found no new material Gaussian-SR or DLSS/FSR-analysis implementation. The sole fresh `gaussian splatting super-resolution` hit was an empty course-project repository; exclude it from competitive tracking.[9]
- **[ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG)** remains worth a bounded integration read, but is not new since yesterday: Apache-2.0 Vulkan midpoint interpolation POC (created Aug. 10; last push Aug. 13), with host-owned synchronization/presentation and explicitly severe artifacts/no standalone validated build. **Opportunity:** borrow its narrow Vulkan host-resource contract only; it is not a quality or architecture threat.[8]

## News
- **AMD:** GPUOpen now lists FSR “Redstone” SDK 2.3 and FSR Upscaling 4.1 RDNA 3 support, alongside Frame Generation 4.0.1 and Ray Regeneration 1.2. **Threat/opportunity:** FSR’s expanding non-RDNA4 reach makes OSS’s cross-vendor claim more competitive-sensitive; preserve strict per-vendor latency/quality reporting rather than treating “open” as parity.[10]
- **Apple / Metal:** Metal 4 adds quantized tensor formats and scale factors in Metal Performance Primitives; its redesigned MetalFX temporal upscaler uses Neural Engine plus M5 Pro/Max Neural Accelerators, accepts dynamic-resolution subrectangles, motion vectors, and distortion fields. **Actionable implication:** a future Metal backend must treat dynamic-resolution regions and engine MVs as first-class ABI inputs.[11]
- **Intel:** Intel says its upcoming launch partners will support the full XeSS 3 suite—Super Resolution, Multi-frame Generation, and Xe Low Latency—on launch day. **Positioning:** OSS telemetry/compatibility needs independently observable SR, FG, and latency state, not one generic upscaler flag.[12]
- Google News surfaced an Aug. 19 report that *Alan Wake 2* added FSR 4 support; this is RSS/third-party lead only, not yet an architecture input pending AMD or game-publisher confirmation.[13]

## Action Items / Follow-ups
- [ ] **Critical:** restore/restart `srcnn-v7.0-pico-005` or its metric publication; the dashboard has held at step 36,650 for 242h while advertising it as active.[1][2]
- [ ] Land and benchmark #12 → #13 → #14 before committing to fused kernels, 32B/Gaussian claims, or student quantization.
- [ ] Deep-read SFMformer only if its reported parameter count, MACs, and deployment graph make it a real ≤1M student candidate; otherwise close as non-material.[5][6]
- [ ] Define Metal runtime inputs now: dynamic-resolution subrect, exact MV convention, distortion/post-process masks, and a quantized-weight fallback path.[11]
- [ ] Keep ZeroFG tracked as a Vulkan ABI reference; do not treat its POC quality, motion estimator, or integration as a baseline.[8]

## Sources

[1] https://opensupersampling.com/data.json
[2] https://opensupersampling.com/status.json
[3] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[4] https://raw.githubusercontent.com/cashcon57/open-supersampling/main/README.md
[5] https://arxiv.org/list/cs.CV/recent
[6] https://arxiv.org/abs/2608.17966
[7] https://archive.org/advancedsearch.php?q=%28%22gaussian+splatting%22+AND+%22super-resolution%22%29+OR+%28%22frame+generation%22+AND+%22deep+learning%22%29+OR+%28%22real-time+upscaling%22+AND+game%29&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=date&rows=30&output=json
[8] https://github.com/hy300leosquizz-ctrl/ZeroFG
[9] https://api.github.com/search/repositories?q=gaussian+splatting+super-resolution+pushed%3A%3E%3D2026-08-05&sort=updated&order=desc&per_page=10
[10] https://gpuopen.com/news
[11] https://developer.apple.com/metal/whats-new
[12] https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships
[13] https://news.google.com/rss/search?q=FSR+4+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen
