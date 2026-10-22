# OSS Research Brief — 2026-06-04

## Project Status Summary
- Latest commits:
  - `a97a180 feat: add tailnet-gated training controls to dashboard`
  - `fcf8084 feat: tailnet-gated training control API + dashboard panel`
  - `0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc`
  - `56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)`
  - `d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume`
  - `e1ab7fe hat.py: use flash-attn 2 directly when available, fallback to SDPA`
  - `e9b4b65 trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)`
  - `26cd9da dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB`
  - `83b4f07 v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast`
  - `baf4b02 worker: apex .com → .org 301 redirect`
- Open issues:
  - #14: D3: measure v6.2-pico-002 VRAM footprint after v6.2 wiring and checkpoint exist
  - #13: H004: run Nsight L2 hit-rate profile on 3080 Ti CUDA host
  - #12: H004: implement packed low-rank canvas layout before claiming 32B/Gaussian
  - #10: v6.2.1 ablation: decoupled G-buffer encoder before concat-fusion
  - #9: v6.2.1 ablation: RepVGG/ECBSR/RepSR reparameterization for student backbone
  - #8: v6.2: cross-vendor kernel ports (HIP / Metal / Vulkan / SYCL)
  - #7: v6.2.1: distillation pipeline with FSR 4 binary as teacher signal
  - #6: v6.2-pico-002: launch on 3080 Ti once codex T1-T5 land
  - #5: v6.2: No-ML shader variant (ships alongside v6.2)
  - #4: v6.2: HAT-Tiny student distillation (≤1M params)
- README snapshot: OSS is pre-alpha; current public architecture is v7 (`srcnn-v7.0-pico-005`) with N-D Gaussian canvas carrying `(x,y,t)` and 3D Cholesky covariance, native time-axis OSS-FX rendering, HAT-Tiny teacher, parent-child loss-adaptive density, Mip-Splatting filters, and planned ≤1M-param distilled student. Cross-vendor kernels and DLL shim remain designed/not built.
- Training dashboard: healthy. Active run `srcnn-v7.0-pico-005` at step **11,550 / 100,000** (11.6%), loss **0.16769**, ETA ~1.5 days. Watcher/Worker/R2 healthy; GPU stats unavailable; held-out `score_log_v7.json` not published yet. Notable: dashboard now exposes tailnet-gated training controls per latest commits.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution** (arXiv:2501.06838, Chen et al., 2025-01-12; ICCV 2025) — Replaces INR arbitrary-scale SR rendering with efficient differentiable 2D CUDA scale-aware Gaussian rasterization. Why it matters to OSS: direct GSASR baseline for OSS’s Gaussian canvas; useful rasterization and scale-aware filtering reference, but reported 720×720 ×12 at ~91 ms is far from OSS realtime budgets.
- **Geometry Gaussians: Decoupling Appearance and Geometry in Gaussian Splatting** (arXiv:2606.05124, Zhou & Lähner, 2026-06-03) — Splits geometry from appearance in 3DGS to improve surface/geometry representation without degrading render quality. Why it matters to OSS: possible analogue for separating motion/covariance state from color features in v7 Gaussians.
- **ZipSplat: Fewer Gaussians, Better Splats** (arXiv:2606.05102, Veicht et al., 2026-06-03) — Feed-forward 3DGS method predicting a scene-complexity-dependent Gaussian budget rather than one Gaussian per input pixel. Why it matters to OSS: strong opportunity for canvas-count control and density budgeting; maps to open issue #12/#13 around packed layout and L2 behavior.
- **Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling** (OpenReview/ICLR 2026 hit) — 2D Gaussian ASSR with Deep Gaussian Prior and adaptive position learning. Why it matters to OSS: reinforces that continuous Gaussian primitives are becoming the dominant ASSR alternative to INR, but likely single-image rather than temporal/game-aware.

### Super-Resolution / Upscaling
- **The Eleventh NTIRE 2026 Efficient Super-Resolution Challenge Report** (arXiv:2604.03198, 2026-04-03) — Survey/report of efficient SISR submissions and tradeoffs. Why it matters to OSS: mine for current low-param SR blocks, reparameterization, and deployment tricks for the ≤1M student.
- **Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content** (arXiv:2602.11339, Bogatyrev et al., 2026-02-11; v2 2026-02-14) — Benchmarks/fine-tunes efficient SR for streaming and notes TensorRT gains. Why it matters to OSS: practical baseline for real-time TensorRT export, though content is streaming rather than game G-buffer conditioned.
- **Taming Real-World Space-Time Video Super-Resolution with One-Step Diffusion** (arXiv:2601.20308, 2026-01) — One-step diffusion for joint spatial/temporal video SR. Why it matters to OSS: teacher-quality/perceptual-loss reference, not likely shippable at <2 ms; could inspire distillation losses.
- **InstaVSR: Taming Diffusion for Efficient and Temporally Consistent Video Super-Resolution** (arXiv:2603.26134, 2026-03) — Efficient temporally consistent diffusion VSR. Why it matters to OSS: threat/opportunity on perceptual temporal quality, but still diffusion-heavy versus OSS’s real-time target.
- **MaCo-GAN: Manifold-Contrastive Adversarial Learning for Single Image Super-Resolution** (arXiv:2606.05068, Han et al., 2026-06-03) — Contrastive adversarial SISR to reduce hallucinated artifacts. Why it matters to OSS: possible loss component for teacher/student perceptual sharpness; needs caution for game HUD/temporal stability.

### Frame Extrapolation / Generation
- **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (arXiv:2406.18551, Wu et al., 2024-05-23) — Efficient neural frame extrapolation without requiring G-buffers. Why it matters to OSS: competitor baseline for low-latency FG; OSS can position as stronger when exact engine motion/G-buffers exist, but G-buffer-free fallback is worth studying.
- **ExWarp: Extrapolation and Warping-based Temporal Super-Resolution** (arXiv:2307.12607) — Warps previous frames with motion vectors and handles extrapolation artifacts. Why it matters to OSS: older but directly relevant to α<1 rendering, disocclusion, and motion-vector failure cases.
- **Consecutive Frame Extrapolation with Predictive Sparse Shading** (ACM TOG/2026 search hit) — Sparse shading critical areas while reusing generated pixels in low-variation areas for neural extrapolation. Why it matters to OSS: aligns with active-canvas / density-budget thinking; worth checking for heuristics to reduce materialized Gaussians.
- **Envisioning the Future, One Step at a Time** (arXiv:2604.09527, 2026-04) — Future trajectory/frame prediction from a single image plus short motion hint. Why it matters to OSS: adjacent extrapolation prior, but less directly useful than game-motion-conditioned methods.

### Inference Optimization
- **The Eleventh NTIRE 2026 Efficient Super-Resolution Challenge Report** (arXiv:2604.03198) — See above. Why it matters to OSS: most likely source for deployable tiny SR blocks and runtime/parameter tradeoffs.
- **Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content** (arXiv:2602.11339) — Reports practical TensorRT optimization benefits. Why it matters to OSS: useful sanity check for export path and benchmark methodology for the student.
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694, CVPR 2026) — Uses compressed-domain motion vectors for coarse warping plus learned residual deformable alignment. Why it matters to OSS: strong support for motion-vector-guided alignment; OSS has higher-quality engine motion vectors, so this is evidence that MV-first alignment is competitive.

### Game Engine / Motion Vectors
- **Super-Resolution with Structured Motion** (arXiv:2505.15961, Litterio et al., 2025-05-21; ICCP 2025) — Shows high-precision motion information plus sparse priors/convex optimization can increase resolution by large factors. Why it matters to OSS: theoretical/algorithmic validation for using exact motion as a first-class SR signal.
- **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (arXiv:2406.18551) — Relevant as an engine-integration fallback when G-buffers are absent or unreliable.

## Archive.org Findings
- `"gaussian splatting" super-resolution`: mostly archived GitHub snapshots of base 3DGS (`graphdeco-inria/gaussian-splatting`) and related repos (`DreamGaussian`, `InstantSplat`, UnityGaussianSplatting). No high-signal archived GSASR/GS-STVSR paper copy found in this run.
- `"frame generation" deep learning`: Internet Archive Scholar returned older/adjacent PDFs including **Future Frame Prediction for Anomaly Detection** and **Deep Inter Prediction with Error-Corrected Auto-Regressive Network**, not directly OSS-actionable.
- `"real-time upscaling" game`: mostly archived news/magazine pages; no actionable paper/tool found.
- `GS-STVSR / GSASR / GRAPE / STSS`: noisy results; `GRAPE` collides with agriculture/acronyms. No taken-down/updated target paper recovered.

## GitHub / Open Source
- **GaussianSR** (`tljxyys/GaussianSR`) — High-fidelity 2D Gaussian splatting for arbitrary-scale image SR. Opportunity: inspect rasterizer/position parameterization for OSS v7.
- **GSASR** (`ChrisDud0257/GSASR`) — ICCV25 official PyTorch code representing LR image as millions of continuous 2D Gaussians. Opportunity/threat: closest open-source ASSR Gaussian baseline; likely not realtime but useful for ablations.
- **SR-GS** (`XiangFeng66/SR-GS`) — HR novel-view synthesis over 3DGS via super-splatting. Adjacent; useful mostly for Gaussian SR literature links.
- **OptiScaler** (`optiscaler/OptiScaler`) — Open upscaler/FG bridge with experimental DX12 frame generation/HUDfix path. Threat/opportunity: integration layer could become OSS shim inspiration or competitor wrapper.
- **dlssg-to-fsr3** (`Nukem9/dlssg-to-fsr3`) — DLSS FG replacement with FSR3. Reinforces user demand for vendor-neutral FG shims.
- **LSFG-Android** (`FrankBarretta/LSFG-Android`) — Vulkan frame-generation pipeline to Android. Watch for mobile/handheld path relevance.
- **Prism OptiScaler fork** (`NodeNestor/prism-optiscaler`) — Claims GAN-trained neural upscaler with color/depth/motion inputs. Potential direct competitor if real; needs validation before reacting.
- **Arm neural-graphics-for-unreal** — Unreal plugin exposing neural upscaling via `ITemporalUpscaler`. Opportunity: inspect engine-plugin integration patterns, especially if OSS targets UE path.

## News
- **NVIDIA DLSS 4.5 / dynamic Multi Frame Generation 6X** — Search surfaced NVIDIA/Tom’s Hardware/CGMagazine reports of improved transformer SR + dynamic MFG, RTX 50-series/library expansion. Threat: vendor quality/marketing lead widens; OSS differentiation should emphasize vendor-neutral one-architecture SR+FG and open integration.
- **FSR 4** — No high-signal fresh result in this run. Continue watching AMD GPUOpen.
- **Intel XeSS** — Reports of **XeSS 3 Multi-Frame Generation** for Arc B580 and XeSS 2.1 expanding frame generation support to older AMD/NVIDIA GPUs. Threat/opportunity: Intel is pushing cross-vendor FG messaging; OSS should benchmark/position against XeSS openness and hardware reach.
- **Apple Metal / MetalFX** — Apple docs/search results mention MetalFX upscaling, denoising, and frame interpolation in Metal 4 context. Opportunity: Metal backend remains strategically important for Mac/Apple Silicon; track MetalFX API shape for OSS Metal path.
- **Unity / Unreal** — Unity render-pipeline strategy discussions include concern about missing frame generation support. Opportunity: vendor-neutral plugin/shim story could appeal if Unity’s official path lags.

## Action Items / Follow-ups
- [ ] Deep-read **ZipSplat** for Gaussian budget/density heuristics; map to canvas_count, materialized, and packed low-rank layout issues.
- [ ] Deep-read **GSASR / GaussianSR** code for scale-aware 2D CUDA rasterization and compare to OSS v7 Mip-Splatting filters.
- [ ] Skim **NTIRE 2026 Efficient SR** for ≤1M-param student blocks, reparameterization, and quantization-friendly designs.
- [ ] Validate **Prism OptiScaler** claims; if real, classify as direct open-source neural game upscaler competitor.
- [ ] Track DLSS 4.5 / XeSS 3 MFG rollout as competitive-positioning pressure on OSS-FX.
