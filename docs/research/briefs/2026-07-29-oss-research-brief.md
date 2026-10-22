# OSS Research Brief — 2026-07-29

_Collection: local Git/README, GitHub REST, rendered dashboard, arXiv API attempt plus direct arXiv pages, Archive.org Advanced Search, GitHub repository search, vendor primary pages, and Google News RSS. RSS-only claims are labelled as leads._

## Project Status Summary
- **Latest commits:** `a97a180` and `fcf8084` add tailnet-gated training controls/API and the dashboard panel. Recent supporting work preserves the v7 Pico checkpoint/recovery path, cross-OS resume, FlashAttention fallback, TF32/prefetch, and bf16 raster filtering.
- **Open issues:** **12**. Priority architecture/performance gates remain #14 (v6.2 VRAM footprint), #13 (3080 Ti Nsight L2 profile), #12 (packed low-rank canvas), #9 (reparameterized student), #8 (cross-vendor kernels), #7 (FSR 4 teacher signal), and #4 (≤1M distillation).
- **README snapshot:** `srcnn-v7.0-pico-005` is the HAT-Tiny v7 research teacher: native `(x,y,t)` Gaussians, `V_xt` motion covariance, parent-child density, Mip-Splat filtering, and fractional-time OSS-FX supervision. No shipping student or non-CUDA backend is trained/implemented. Published held-out metrics remain v5-only: **25.703 dB PSNR / 0.1666 LPIPS** on TartanAir `oldtown`.
- **Training dashboard (rendered 2026-07-29):** headline says **36,650 / 100,000**, loss **0.19268**, viz **36,000**, ETA ~1.1d; health says **Trainer offline, stuck at 36,650 for 12h**. Watcher is healthy (published 1s ago), CF Worker HTTP 200 in 92ms, R2 origin HTTP 200; GPU stats unavailable. The simultaneous “live/ETA” copy is contradicted by unchanged metrics and offline health; treat this as a real trainer/metrics outage, not recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[SplatStream: Fine Granular Scalable Gaussian Splatting for Adaptive 3D Scene Streaming](https://arxiv.org/abs/2607.25971)** (arXiv:2607.25971v2; Muhammad Talha, William Gordon, Sajid Umair, Zhu Li, Anique Akhtar, Joel Jung; submitted 2026-07-28, revised 2026-07-29) — Layered dynamic 3DGS with inter-layer prediction, temporal B-frames, and volume-opacity importance packetization for progressive delivery. **Why it matters to OSS:** direct #12 input: benchmark importance ordering, temporal delta coding, and quality-layer budgets in the persistent canvas; it is a streaming representation, not an SR/FG runtime competitor.
- No other newly published direct Gaussian-temporal paper cleared the relevance gate; already-tracked **SubSplat**, **GrainGS**, **CaT-GS**, and **SPARE-GS** remain higher-priority v7 density/cache controls.

### Super-Resolution / Upscaling
- **[TRaM-VSR: Importance-Aware Token Routing and Merging for One-Step Diffusion Video Super-Resolution](https://arxiv.org/abs/2607.22231)** (arXiv:2607.22231v1; Sicheng Gao, Zhuyun Zhou, Yixuan Liu, Tong Shen, Zongwei Wu, Radu Timofte; 2026-07-24) — Uses motion-sensitive and structural importance to route critical tokens through a high-fidelity local stream while merging less-important tokens globally. **Why it matters to OSS:** teacher/offline control for v7 loss-adaptive Gaussian density; do not import diffusion runtime into the ≤2ms student.
- **NanoVSR** (arXiv:2607.10495, already tracked) remains the strongest shipping-student comparison: structurally reparameterized convolutions collapsing to ordinary inference convolutions. It remains the direct control for #9/#4.

### Frame Extrapolation / Generation
- No newly published, material single-network game frame-extrapolation paper was verified. **ANVIL** (arXiv:2603.26835, already tracked) remains a useful codec-MV + convolutional-residual baseline, but its codec-input result must not be substituted for game-engine validation.
- **Future Rendering ≠ Future Surface** (arXiv:2607.21471, already tracked) remains the evaluation lead: add observed-vs-future error-gap reporting to OSS-FX, not only rendered-frame PSNR/LPIPS.

### Inference Optimization
- No new TensorRT/INT8 paper cleared the relevance bar. Existing direct controls remain deployment-aware INT8 SR (fused deploy graph + distillation + QAT), **Finding Fast Filters** for EWA/Mip filter search, and **Rendering 3D Gaussians on a Graph Processor** for tile-local SRAM/bandwidth and density imbalance.
- The arXiv Atom API returned HTTP 429 for all three targeted compound queries. Direct arXiv pages verified the two material candidates above; no claim of exhaustive fresh arXiv coverage is made.

## Archive.org Findings
- Advanced Search returned **3** noisy GitHub-list snapshots for `"gaussian splatting" super-resolution`, **3** mixed items for `"frame generation" deep learning` (DLSS Swapper/FluidFrames builds plus unrelated video generation), and **0** for `"real-time upscaling" game`.
- Known-paper checks produced no archival delta: GS-STVSR 0; GSASR one unrelated CIA item; GRAPE/STSS 0 or unrelated. No archived/taken-down paper version or meaningful GS-STVSR/GSASR/GRAPE/STSS update was verified.

## GitHub / Open Source
- GitHub REST search since 2026-07-15 found no credible new open DLSS/FSR/XeSS analysis tool. [`hvcl/3DCT_SR_DDNM_NAB-GS`](https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS) is active Apache-2.0 but a 4-star zero-shot CT SR project: not an OSS runtime lead.
- [`Adam-AtlasSoftware/RDNU`](https://github.com/Adam-AtlasSoftware/RDNU) is the closest open game-upscaler lead: GPL-3.0, 15 stars, pushed 2026-07-24, RDNA3-focused and G-buffer guided. **Opportunity:** inspect its data contract and measurement method. **Constraint:** GPL prevents Apache-2.0 code reuse.
- [`intel/xess`](https://github.com/intel/xess/releases/tag/v3.0.2) released XeSS SDK 3.0.2 on 2026-07-24: XeLL 1.3.2.10 corrects `minimumIntervalUs` from prior nanosecond reporting, fixes a non-Intel marker leak when `Present` stops, and unwraps the Streamline D3D12 proxy. Its native-D3D12 recommendation is concrete DLL-shim test input.

## News
- **AMD — verified primary update:** FSR SDK 2.3 / “Redstone” exposes ML **FSR Upscaling 4.1.1** on RDNA 3 RX 7000 discrete GPUs (formerly RDNA4-only), **FSR Frame Generation 4.0.1** for RDNA4, DX12/UE5 support, and DLL-based automatic version updates for eligible integrations. **Threat/opportunity:** materially widens AMD’s ML-upscaling footprint; OSS needs a vendor-neutral data/latency/quality benchmark rather than generic FSR 4 teacher-signal claims.
- **Apple — verified primary update:** Metal 4 adds quantized tensor formats/scale factors and Metal performance primitives for weight compression plus M5 Pro/Max Neural Accelerators. Redesigned MetalFX temporal upscaling can use Neural Engine + Neural Accelerators; its API adds dynamic-resolution subrectangles, motion-vector input, and distortion fields. **Opportunity:** explicit Metal/ANE student backend requirements now exist; prototype a motion-vector, quantized tiny-CNN path rather than treating Apple as an undifferentiated future port.
- **NVIDIA:** current primary developer-blog index shows July Rubin/Vera AI posts, but no fresh DLSS product post. RSS claims about “DLSS 5” SIGGRAPH demos remain unverified leads, not OSS competitive facts.
- **Intel/engine news:** XeSS 3.0.2 is verified above. RSS surfaced a UE5 secondary-upscaler report and Qualcomm AI Frame Fusion, but no primary Unreal/Unity/Qualcomm technical material was verified; no architecture decision should depend on them.

## Action Items / Follow-ups
- [ ] **Critical:** repair/restart the 3080 Ti trainer or metrics producer. Publication infrastructure is healthy, but step/loss/GPU telemetry are not advancing.
- [ ] Before committing #12, profile #13 and run a canvas-layout microbenchmark: SplatStream-style importance ordering + temporal deltas versus packed texture/low-rank layouts; record bytes/Gaussian, L2 hit rate, DRAM traffic, and quality-per-Gaussian.
- [ ] Deep-read TRaM-VSR only for motion/boundary importance targets; ablate the signal against v7 loss-adaptive density as teacher supervision, not runtime architecture.
- [ ] For #4/#9, benchmark NanoVSR/ANVIL-style externally supplied motion plus convolutional residual against the OSS student; isolate quality, latency, and quantization failure modes.
- [ ] Establish a reproducible AMD FSR 4.1.1 / FSR FG 4 vs OSS student benchmark protocol on RDNA3/RDNA4; position exact engine MVs and one-native-time canvas against AMD’s separate SR/FG stack.
- [ ] Add a Metal prototype decision memo: quantized tensors, M5 Neural Accelerators, motion-vector/subrect MetalFX-compatible API, and whether OSS can own a custom student pass.
- [ ] Add XeLL/Streamline native-D3D12 proxy, idle-Present marker, and `minimumIntervalUs` unit tests to DLL-shim acceptance work.

## Bibliography Update
- No append: **SplatStream** and **TRaM-VSR** were already added under `## Research brief 2026-07-29`; verified direct-page metadata refined their status/authors without duplicating entries.
