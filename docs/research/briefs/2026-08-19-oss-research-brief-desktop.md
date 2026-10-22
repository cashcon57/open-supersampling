# OSS Research Brief — 2026-08-19

## Project Status Summary
- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated training controls and dashboard UI; the preceding work preserves the v7 Pico recovery point, fixes WSL2 worker/checkpoint-resume issues, enables FlashAttention-2/TF32 throughput paths, and keeps rasterizer smoothing in FP32 under bf16.[unverified]
- **Open issues:** 12. The immediate architecture/performance queue remains #1 low-rank rasterizer, #3 counting-sort tile bin, #4 student distillation, #7 FSR 4 teacher-signal experiment, #8 cross-vendor ports, #9 reparameterized student backbone, #12 packed low-rank canvas, and #13 Nsight L2 profile.[2]
- **Training dashboard:** headline says v7 Pico OSS-FX is live at step **36,650 / 100,000**, loss **0.19268**, viz step 36,000. The service-health telemetry simultaneously marks the trainer **offline, stuck at step 36,650 for 243h**; watcher, CF Worker, and R2 origin are healthy. Treat the “live” headline/ETA as stale metrics, not evidence of active training. Current best remains v5 Pixel Temporal at **25.70 dB PSNR / 0.1666 LPIPS** on held-out TartanAir oldtown.[1]

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **RoofGS: Roofline-Guided End-to-End Acceleration of 3D Gaussian Splatting** (arXiv:2608.15785v1; Yang Luo, Yan Gong, Yongsheng Gao, Jie Zhao; 2026-08-16) — Uses a stage-wise roofline analysis and different fixes for memory-bound front-end work versus instruction-throughput-bound rasterization. **Why it matters to OSS:** direct methodology for #12/#13: profile the canvas update/bin/raster stages independently before choosing packed layout, tile binning, or kernel fusion; do not assume the rasterizer is the only bottleneck.[3]
- **Already tracked, no bibliography update:** *Amulet* (arXiv:2608.10423) is still the most directly relevant fresh OSS-FX baseline: sparse tiled layered image-space cache, adaptive shading, explicit visibility/disocclusion. It should remain the deterministic fallback comparison for the α<1 path rather than a neural-runtime candidate.[4]
- No newly published 2D-Gaussian arbitrary-scale or GS-STVSR/GSASR-class paper materially passed the relevance gate in this run. Broad 3DGS results were overwhelmingly reconstruction/SLAM/medical and not architecture-transferable to a ≤2 ms game pipeline.[unverified]

### Super-Resolution / Upscaling
- No new materially relevant paper found. The queried recent VSR results are predominantly diffusion/video-generation systems, not viable shipping students. *NanoVSR* (arXiv:2607.10495), already in the bibliography, remains the deployment-relevant reference: fully convolutional, structural reparameterization, standard-convolution TensorRT path.[unverified]

### Frame Extrapolation / Generation
- No new OSS-relevant extrapolation paper beyond already tracked *Amulet* and *SNM-VFI*. The new VFI stream is diffusion- or event-camera-oriented; it is useful for teacher/loss ideas but not the runtime budget.[unverified]

### Inference Optimization
- **RoofGS** is the only new material item. Its stage-specific roofline framing is a stronger near-term action than adopting paper-specific 3DGS kernels: capture per-stage arithmetic intensity, DRAM/L2 traffic, occupancy, and sort/bin cost on the 3080 Ti before changing #12’s layout.[3]

## Archive.org Findings
- Archive.org’s exact `"gaussian splatting" super-resolution` query returned only three GitHub snapshot artifacts, not an archived paper copy; `"frame generation" deep learning` returned a 2026 MoCha-LD registration plus old Windows utility builds; `"real-time upscaling" game` returned zero. The combined GS-STVSR/GSASR/GRAPE/STSS query is unusably noisy (16,903 hits, dominated by the ordinary word “grape”). No credible missing/taken-down copy was found.[5]
- Existing local bibliography coverage for GSASR, GS-STVSR, GRAPE, and Amulet is intact. Use targeted Wayback CDX lookups against known paper/project URLs next time rather than acronym-wide Archive searches.[unverified]

## GitHub / Open Source
- **MAKO** (`eugeniosegala/MAKO`) is an active Vulkan-powered Linux gaming layer targeting Lossless Scaling frame generation/scaling on Steam Deck, Steam Machine, and desktop Linux. It is a **competitive-positioning watch item**, not proof of DLSS/FSR-equivalent reconstruction quality: inspect its capture/interposition and Vulkan presentation strategy for OSS’s future Linux/Vulkan path.[6]
- **RDNU** (`Adam-AtlasSoftware/RDNU`) is a Radeon/RDNA3-focused ML upscaler using decoupled G-buffer guidance, with pretrained/fine-tuned model claims. This is a useful open AMD-side integration/training comparator, and potentially a source of benchmark protocol ideas; validate its license, measured latency, and reproducibility before treating it as a threat.[7]
- GitHub’s exact recent-push searches returned no useful hits; unquoted broad search was noisy. No new open-source DLSS/FSR binary-analysis tool passed verification.[unverified]

## News
- **Headline-level only:** Google News RSS reports recent game-level adoption claims: *Beast of Reincarnation* update 1.0.7 added DLSS 4.5/FSR 4; RSS also carries an Alan Wake 2 FSR 4 report and Intel Gamer Days/XeSS 3-supported-title coverage.[8][9][10]
- These are not primary vendor confirmations. No direct NVIDIA, AMD GPUOpen, Intel, Apple, Unreal, or Unity announcement was verified in the 14-day window; do not make roadmap claims from them. The practical signal is continuing game-side demand for vendor stacks, which reinforces OSS’s cross-vendor/DLL-shim differentiation rather than changing the v7 design.[unverified]

## Action Items / Follow-ups
- [ ] **Ops blocker:** restore or explicitly stop/relabel `srcnn-v7.0-pico-005`; dashboard claims “live” while trainer health has been stuck at 36,650 for 243h.[1]
- [ ] **#13 before #12:** run an Nsight Systems/Compute roofline-style stage breakdown of canvas update, cull/bin/sort, raster, and composite; record DRAM/L2/occupancy and stage time.[unverified]
- [ ] Deep-read RoofGS for measurement design only; prototype only techniques that survive OSS’s 2D, dynamic-canvas, cross-vendor constraints.[3]
- [ ] Benchmark Amulet-style layered-cache visibility/disocclusion against the planned α<1 canvas render on the same motion-vector inputs.[4]
- [ ] Audit MAKO and RDNU: license, capture/interposition, G-buffer contract, real latency, reproducible quality, and whether either can supply cross-vendor test fixtures.[6][7]

## Sources

[1] https://opensupersampling.org — OSS dashboard
[2] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=30 — OSS open issues API
[3] https://arxiv.org/abs/2608.15785 — RoofGS
[4] https://arxiv.org/abs/2608.10423 — Amulet
[5] https://archive.org/advancedsearch.php?q=%22gaussian+splatting%22+super-resolution&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=date&rows=10&output=json — Archive Gaussian SR search
[6] https://github.com/eugeniosegala/MAKO — MAKO
[7] https://github.com/Adam-AtlasSoftware/RDNU — RDNU
[8] https://news.google.com/rss/search?q=DLSS+4+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen — Google News DLSS 4 RSS
[9] https://news.google.com/rss/search?q=FSR+4+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen — Google News FSR 4 RSS
[10] https://news.google.com/rss/search?q=Intel+Arc+XeSS+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen — Google News Intel Arc XeSS RSS
