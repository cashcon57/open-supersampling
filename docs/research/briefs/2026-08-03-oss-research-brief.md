# OSS Research Brief — 2026-08-03

_Collection: local Git/README, GitHub REST, rendered public dashboard, targeted arXiv Atom queries, Archive.org Advanced Search and Wayback CDX, GitHub repository APIs, primary vendor pages, and Google News RSS. RSS-only claims are labelled as leads._

## Project Status Summary

- **Latest commits:** `a97a180` and `fcf8084` add tailnet-gated training controls to the public dashboard/API. The immediately preceding work preserves the v7 step-5K recovery snapshot; repairs cross-OS checkpoint resume and the WSL2 DataLoader deadlock; and adds FlashAttention 2, TF32 convolution, and prefetch throughput work.
- **Open issues:** 12 open issues: #14 v6.2 VRAM measurement, #13 Nsight L2 profile, #12 packed low-rank canvas, #10 decoupled G-buffer encoder ablation, #9 reparameterized student, #8 HIP/Metal/Vulkan/SYCL ports, #7 FSR 4 teacher signal, #6 Pico launch, #5 no-ML shader, #4 <=1M student distillation, #3 custom counting-sort tile bin, and #1 R=4–8 low-rank latent rasterizer.
- **README snapshot:** `srcnn-v7.0-pico-005` is the active HAT-Tiny research teacher: N-D `(x,y,t)` Gaussians, `V_xt` motion covariance, loss-adaptive density, Mip-Splat filters, and alpha=0.5 OSS-FX supervision. The <=1M TensorRT-FP8 shipping student and all non-CUDA backends remain unbuilt. The published held-out quality remains v5-only: 25.703 dB PSNR / 0.1666 LPIPS on TartanAir `oldtown`.
- **Training dashboard (rendered 2026-08-03):** headline says **v7 training**, `36,650 / 100,000`, loss `0.19268`, viz `36,000`, “live,” ETA ~1.1d. Service health says **Trainer offline, stuck at step 36,650 for 47h**; watcher is healthy (published 1s ago), CF Worker is HTTP 200 in 104ms, and R2 origin is HTTP 200. GPU telemetry is unavailable. Step/loss remain unchanged from the 2026-08-02 brief while the stuck counter advanced from 23h to 47h: no recovery evidence; the trainer/metrics emitter remains blocked despite healthy publishing infrastructure. The dashboard’s “live” copy conflicts with its trainer-health signal.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **FillGS: Filling Observation Gaps in 4D Gaussian Splatting via Viewpoint-Time Selection and Generative Refinement** — Takashi Otonari, Toshihiko Yamasaki; 2026-07-31; [arXiv:2607.29284](https://arxiv.org/abs/2607.29284). It selects virtual viewpoint-time samples by rendering sensitivity and motion-aware observation density, then filters conflicting/generated regions before 4DGS fine-tuning. **Why it matters to OSS:** an offline teacher/data-curation reference for disocclusion and fast-motion coverage; do not put generative refinement on the <=2ms runtime path.
- The remaining new temporal-GS hits were avatar/human reconstruction, robotics, or general world-model work, not a direct real-time 2D reconstruction or OSS-FX competitor. Already-tracked TSOG remains the stronger direct packed temporal-layout reference.

### Super-Resolution / Upscaling

- No newly published material game SR, 2D-GS arbitrary-scale SR, or student-distillation paper cleared the architecture gate. The scan re-found already-tracked **NanoVSR** ([arXiv:2607.10495](https://arxiv.org/abs/2607.10495)) and **TRaM-VSR** ([arXiv:2607.22231](https://arxiv.org/abs/2607.22231)). NanoVSR’s structurally reparameterized CNN remains a #9/#4 student control, not a new bibliography entry.

### Frame Extrapolation / Generation

- No new material single-network/exact-motion-vector extrapolation baseline. Recent candidates were diffusion video, event-video reconstruction, or coding work; none challenges the OSS claim only after OSS-FX is actually evaluated on held-out future frames.

### Inference Optimization

- **QIRF: Quantum-Inspired Non-Orthogonal Function-Space Compression for 3D Gaussian Splatting** — Shizeng Jiang, Hao Zhang, Xuerui Ma, Ying Hu, Tao Zhang; 2026-07-20; [arXiv:2607.18067](https://arxiv.org/abs/2607.18067). It uses local Gaussian-overlap/radiance-response matrices and generalized eigendecomposition to eliminate functionally redundant overlapping splats; reports 71.7% average count/storage reduction and 34.3% faster rendering on its 3DGS benchmarks. **Why it matters to OSS:** directly suggests an ablation for #12 loss-adaptive density/pruning beyond per-splat heuristics; benchmark its quality, update cost, bytes/Gaussian, L2/DRAM behavior, and raster time against the low-rank packed canvas before adopting it.
- No new material TensorRT, INT8-SR, or kernel-fusion paper. #13 profiling remains prerequisite to choosing a runtime layout.

## Archive.org Findings

- Advanced Search returned three archived GitHub/star-list snapshots for `"gaussian splatting" super-resolution`; `"frame generation" deep learning` returned FluidFrames/DLSS Swapper builds plus an unrelated diffusion record; `"real-time upscaling" game` returned zero results.
- Known-work terms remain noisy: GS-STVSR 0, GSASR one unrelated 1961 CIA item, and no validated archive record for the requested real-time-upscaling query. Wayback CDX has one 2026-07-24 capture of QIRF’s arXiv abstract page; it has no result for FillGS, NanoVSR, or the CT repo. This does **not** establish absence of other archived versions.

## GitHub / Open Source

- **No new credible direct OSS competitor or DLSS/FSR analysis tool.** The only fresh Gaussian-SR repo hit, [3DCT_SR_DDNM_NAB-GS](https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS) (Apache-2.0, 4 stars, pushed 2026-07-28), is zero-shot medical CT SR—not a game/canvas baseline.
- [RDNU](https://github.com/Adam-AtlasSoftware/RDNU) remains a data-contract lead: RDNA3-focused ML upscaler with decoupled G-buffer guidance, 15 stars, GPL-3.0, last pushed 2026-07-24. Inspect input/evaluation semantics only; do not reuse GPL code.
- **Integration/positioning watch:** [OptiScaler](https://github.com/optiscaler/OptiScaler) is active (9,621 stars / 349 forks; pushed 2026-08-01, metadata updated 2026-08-03). Its DLSS2+/XeSS/FSR2+ bridge plus FSR-FG/XeFG replacement behavior is the compatibility UX benchmark. GPL-3.0 requires strict isolation.

## News

- **AMD — primary verified:** GPUOpen lists **FSR Upscaling 4.1.1** (June 2026) and says FSR 4.1 supports RX 7000/RDNA3; its UE plugin pairs ML upscaling with ML frame generation. **Threat:** AMD’s cross-generation coverage continues improving, though it still presents separate SR and FG paths. **Opportunity:** OSS can validate its one-canvas claim against the same motion-vector/camera-input edge cases.
- **NVIDIA — primary verified:** NVIDIA’s [SIGGRAPH 2026 update](https://blogs.nvidia.com/blog/siggraph-news-2026/) highlights neural rendering, world models, simulation, and 21 accepted technical papers. It does not verify a shipping “DLSS 5”; RSS headlines claiming one remain unverified and must not drive roadmap decisions.
- **Intel — primary verified:** Intel’s developer page now markets **XeSS 3** with Multi Frame Generation (up to three AI-generated frames). This reinforces the competitor direction toward separated/upscale-plus-MFG stacks; it is not an OSS-equivalent unified canvas architecture.
- **Apple — primary verified:** Apple’s WWDC26 Metal guide says its redesigned MetalFX temporal upscaler uses the Neural Engine and Neural Accelerators on M5 Pro/Max. This raises the bar for an eventual Metal backend and supports keeping a compact CNN/student path independent of CUDA-specific assumptions.
- No qualifying new primary UE5/Unity upscaling announcement was found. Google News RSS leads about FSR 4.1/Proton and XeSS SDK 3.0.2 were not promoted without primary source confirmation.

## Action Items / Follow-ups

- [ ] **Critical:** restore/restart the 3080 Ti trainer or metrics emitter; require a higher step, fresh GPU telemetry, and a new viz before calling v7 live.
- [ ] Deep-read/implement a **QIRF-style overlap-aware pruning** offline ablation after #13; compare it with #12’s low-rank layout and present loss-adaptive parent-child density.
- [ ] Use **FillGS** only as a teacher-side/capture-data disocclusion experiment: motion-aware observation-density masks, reliable-region filtering, and future-frame holdouts.
- [ ] Keep NanoVSR as the #9/#4 reparameterized student control; prioritize TensorRT-friendly fused CNN graphs over a transformer-like student.
- [ ] Track OptiScaler input/resource/presentation behavior as the DLL-shim UX baseline while maintaining GPL isolation.
- [ ] Define Metal student constraints early: runtime code should be expressible as portable compact-CNN + raster kernels, with no CUDA-only dependency in the learned representation.

## Bibliography Update

- Added QIRF: it is a newly surfaced, materially relevant packed-canvas compression/pruning reference. FillGS remains in this memo only because its generative multi-view refinement is not a shipping architecture.
