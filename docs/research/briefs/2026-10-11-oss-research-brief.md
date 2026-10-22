# OSS Research Brief — 2026-10-11

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (`git log --oneline -10` returns only this commit).
- **Open issues:** GitHub REST returned #14 VRAM footprint; #13 Nsight L2 hit-rate; #12 packed low-rank canvas; #10 G-buffer encoder ablation; #9 student reparameterization; #8 cross-vendor ports; #7 FSR 4 teacher distillation; #6 pico launch; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- **README:** States v7 N-D Gaussian/time-axis run `srcnn-v7.0-pico-005` is training, while dashboard content is stale/conflicting; shipping student remains untrained. README still provides historical v6 status and should not be mistaken for current runtime data.
- **Dashboard (rendered 2026-10-11):** Headline/card now identify **v7.0 Pico OSS-FX**, step **36,650/100,000 (36.7%)**, loss **0.19268**, viz at step 36,000, ETA ~1.1 days; current best remains v5-pixel-temporal, PSNR 25.70 / LPIPS 0.1666. But health tooltip says trainer **offline, stuck at step 36,650 for 534h** and GPU stats unavailable, while watcher (published 1s ago), CF Worker (200, 70ms), and R2 origin (200) are healthy. Static teacher/student panel is stale (v6.2-pico-002); README says v7-pico-005 training. **Critical inconsistency:** same step/loss persisted since the prior report, but stale offline counter advanced 12h versus 522h; visible live/status and trainer-offline status conflict. Treat step/ETA as published dashboard metadata, not independently confirmed active training; investigate heartbeat/metrics timestamps.

## New Papers (arXiv)
Search results used arXiv-indexed primary abstract pages/search snippets; abstracts only. Discovery queries were keyword searches, so no claim of exhaustive category coverage.

### Gaussian Splatting / Temporal
- **3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** (Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu; 2026-05-12; [arXiv:2605.11489](https://arxiv.org/abs/2605.11489)). Abstract describes differentiable post-rendering of low-resolution 3DGS: gradient-guided GRU supersampling plus compact U-Net-like temporal interpolation. **Why it matters:** direct combined SR+frame synthesis comparator and evidence that joint output quality can require an explicit temporal subnet—benchmark against OSS’s single time-conditioned canvas, while noting this is 3DGS-rendering-specific and not engine-agnostic game evidence. Newly surfaced as relevant; already present in BIBLIOGRAPHY.md.
- **Continuous Space-Time Video Super-Resolution with 3D Fourier Fields** (Alexander Becker, Julius Erbach, Dominik Narnhofer, Konrad Schindler; submitted 2025-09-30, revised 2026-03-04; [arXiv:2509.26325](https://arxiv.org/abs/2509.26325)). Learned continuous (x,y,t) Fourier field with analytic Gaussian PSF and arbitrary space/time sampling. **Why it matters:** alternative continuous-time representation/control for OSS’s native-time Gaussian canvas and antialiasing; abstract reports improved joint SR/temporal results, but this is video SR rather than engine-conditioned real-time inference.
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (Mingyu Shi et al.; 2026-04-20; [arXiv:2604.18047](https://arxiv.org/abs/2604.18047)). Flow-guided evolving 2D Gaussians, covariance resampling, adaptive offsets; abstract claims near-constant inference time for ×2–×8 temporal scales. **Why it matters:** direct temporal-Gaussian precedent for OSS's covariance/time-axis choices; video VSR differs from exact engine-MV conditioning.

### Super-Resolution / Upscaling
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (Pham Phuong Nam Nguyen, Nam Tien Le, Thi Kim Trang Vo, Nhu Tinh Anh Nguyen; 2026-04-22; [arXiv:2604.20291](https://arxiv.org/abs/2604.20291)). Low-resolution extract/refine, reparameterizable backbone, DCT and confidence-weighted teacher output supervision, followed by QAT on the fused deploy graph; abstract reports ~30.0 dB fixed-shape INT8 TFLite on MAI challenge. **Why it matters:** concrete candidate recipe for OSS student training/deployment—distill on the fused graph and benchmark DCT/teacher losses plus clipping/BN recalibration; single-image/mobile benchmark, not temporal game quality.


### Frame Extrapolation / Generation
- No newly verified direct real-time game-frame extrapolation paper in this pass. Broad searches returned reconstruction papers not applicable to engine motion-vector-conditioned game output.

### Inference Optimization
- **Streaming-Aware Diffusion for Real-Time Video Super-Resolution via Cross-Step Attention** (Harris Partaourides, Sotirios Chatzis; submitted 2026-10-08; [arXiv:2610.11746](https://arxiv.org/abs/2610.11746)). Reuses denoising features/states across frames and steps; reports >40 FPS at 512×512 after cold start. **Why it matters:** feature reuse and streaming state are useful student-distillation/training ideas, but iterative diffusion and reported resolution make it a weak direct fit for OSS's strict game latency target.

## Archive.org Findings
- Archive.org advanced-search returned only noisy inventories: three snapshot/listing records for `"gaussian splatting" super-resolution`, three mixed video/software records for `"frame generation" deep learning`, and zero for `"real-time upscaling" game`. These do not establish archival absence of relevant research.
- The requested named-paper Wayback CDX checks for GS-STVSR, GSASR, GRAPE, and STSS were not run successfully; no take-down/archival conclusion.

## GitHub / Open Source
- **Threat / competitive signal — [OptiScaler](https://github.com/optiscaler/OptiScaler):** search-indexed release notes point to XeFG integration and multi-frame generation capability; compatibility list indexed as updated Oct 10 claims 710 working games. This is middleware/algorithm swapping and access to vendor models, not a competing OSS model or unified Gaussian SR+extrapolation architecture. Still raises the bar for OSS's planned DLL-shim/runtime compatibility and broad game coverage.
- **Opportunity / active lead — [ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG):** newly updated open-source, platform-agnostic Vulkan frame-generation engine aimed at mobile, Apache-2.0; repository description calls it an experimental V1 proof of concept (pushed Oct 9). Verify code maturity, supported inputs, and measured latency before treating it as a usable integration or competitor.
- **Recent mod activity:** search-indexed OptiScaler fork release dated Oct 9 describes Vulkan frame-generation support and neural-rendering options; treat as an unverified fork/release lead until repository and benchmark inspection.
- No fresh verified open-source Gaussian-SR model repo found. GitHub topic/search results were noisy; broad Gaussian topics showed routine gsplat/editor activity rather than an OSS-equivalent.

## News
- **Apple / MetalFX (primary source):** Apple’s current Metal “What’s New” documents a redesigned temporal MetalFX upscaler using Neural Engine and Neural Accelerators on M5 Pro/Max, plus dynamic-resolution subrectangles, motion-vector input to reduce preprocessing, and distortion-field postprocessing. Important cross-vendor-port threat/opportunity: Metal is gaining specialized neural upscaling hardware-path integration, so OSS’s Metal plan must benchmark appropriate precision/operator support rather than assume CUDA-style quantization gains. [Apple Metal What's New](https://developer.apple.com/metal/whats-new/)
- **AMD FSR Redstone (primary source):** GPUOpen currently advertises FSR SDK 2.3 with ML FSR Upscaling 4.1.1 support for Radeon RX 7000, Frame Generation 4.0.1, and Ray Regeneration 1.2 updates; the frame-generation release notes identify motion-vector preprocessing fixes. This raises the bar for temporal input robustness and reinforces OSS’s need for motion-vector edge-case tests. [GPUOpen FSR SDK](https://gpuopen.com/)
- **DLSS:** search returned claims about DLSS 4.5/DLSS 5 and recent game launches, but NVIDIA newsroom extraction did not substantiate the DLSS-5 headline; no new NVIDIA feature claim verified here. Do not rely on low-quality roundup claims.
- **Unity / Unreal:** no directly relevant engine upscaling release verified in this sweep; results were unrelated Unity news or low-quality UE release roundups.

## Action Items / Follow-ups
- [ ] Reconcile dashboard's stale v6.1/v6.2 labels and **534h** offline indicator with the README's v7.0-pico-005 claim; verify trainer heartbeat/metrics timestamp because step/loss are unchanged despite live card/ETA.
- [ ] Inspect GSCV (arXiv:2610.07795) for applicable storage/rate-distortion techniques if packed-canvas profiling identifies a bandwidth bottleneck.
- [ ] Benchmark OptiScaler's recent XeFG/Vulkan ecosystem as an integration/positioning comparator; do not equate middleware support with a competing learned model.
- [ ] Inspect ZeroFG code maturity and run a reproducible Vulkan/mobile latency and quality comparison if it exposes compatible temporal inputs.
- [ ] Evaluate deployment-aware fused-graph QAT from arXiv:2604.20291 and 3DGS³ (arXiv:2605.11489) against the student/canvas plan; bibliography already tracks the latter.
- [ ] Run focused Wayback CDX checks for GS-STVSR, GSASR, GRAPE and STSS.

