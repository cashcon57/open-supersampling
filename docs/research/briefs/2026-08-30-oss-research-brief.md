# OSS Research Brief — 2026-08-30

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (only commit currently reachable from this checkout; no additional history returned by `git log -10`).
- **Open issues:** **12**. The immediate architecture/perf queue remains packed low-rank canvas layout and residency profiling (#12–#14), tile binning (#3), then ≤1M student distillation and cross-vendor kernels (#4/#7/#8/#9).
- **README snapshot:** pre-alpha v7 is an N-D `(x,y,t)` Gaussian-canvas **teacher**. The ≤1M shipping student and native CUDA/HIP/Metal/Level Zero/Vulkan backends remain unbuilt.
- **Training dashboard:** `srcnn-v7.0-pico-005` remains at **36,650 / 100,000**, loss **0.19268**. The dashboard says **Trainer — offline; stuck 157h** while Watcher, CF Worker, and R2 are healthy. Its active/live/ETA copy is contradictory and stale; no evidence of resumed training. GPU stats are unavailable.

## New Papers (arXiv)

No material paper newer than the 2026-08-29 sweep cleared the architecture/training/inference gate. The targeted sweep did reconfirm the following already-bibliographed controls; no duplicate bibliography entries were added.

### Gaussian Splatting / Temporal
- **Towards Alias-Free 4D Gaussian Representations with Motion-Aware Filtering** — Ankit Dhiman, Kunal A. Kathare, Pranav Vignesh, Lokesh R. Boregowda, Venkatesh Babu Radhakrishnan; 2026-08-22; [arXiv:2608.21828](https://arxiv.org/abs/2608.21828). Motion-conditioned smoothing selects filter strength from time/focal-depth statistics. **Why it matters:** directly supports v7’s native-time covariance and Mip-Splatting direction; use it as an anti-aliasing ablation, not a wholesale representation replacement.
- **4DGS-WAM: Bridging Past and Future with an Object-Centric World Action Model based on 4D Gaussian Splatting** — Yueen Ma, Zenglin Xu, Irwin King; 2026-08-26; [arXiv:2608.25956](https://arxiv.org/abs/2608.25956). Persistent 4D splats retain static background and predict only dynamic-object transformations. **Why it matters:** validates the static/dynamic reuse premise behind a persistent canvas, but is a world model with inferred action, not an engine-MV real-time SR system.
- **M³ISR: A Multi-Modal Multi-View Benchmark for 3D/4D Gaussian Splatting and Feedforward Compression** — Xinhui Liu, Lei Liu, Zhenghao Chen, Lebin Zhou, Wei Wang, Wei Jiang; 2026-08-23; [arXiv:2608.22465](https://arxiv.org/abs/2608.22465). Controlled benchmark with 4DGS streaming/compression tracks and rate-distortion formulations. **Why it matters:** useful external methodology for #12 packed canvas and #13 L2-residency experiments; not a direct game-data substitute.

### Super-Resolution / Upscaling
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** — Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11; [arXiv:2607.10495](https://arxiv.org/abs/2607.10495). Fully convolutional, structurally reparameterized VSR tailored to TensorRT; the 644k model is the relevant compact control. **Why it matters:** retain as a shipping-student baseline for #4/#9, using OSS engine motion/canvas output as the aligned input rather than abandoning explicit motion.
- Other returned Gaussian-SR work (CLEAR, RBS, QuADA-GS) is already tracked. It reinforces two existing choices—reliability-gated detail injection and loss-adaptive density—but is sparse-view/offline reconstruction rather than a <2ms game upscaler.

### Frame Extrapolation / Generation
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** — Sebastian Künzel et al.; 2026-08-11; [arXiv:2608.10423](https://arxiv.org/abs/2608.10423). Sparse layered scene state with adaptive shading for extrapolation. **Why it matters:** keep its visibility-age/refresh-cadence signals as controls for OSS-FX disocclusion and frame pacing; it is already tracked.
- **SNM-VFI: Symmetric Nonlinear Motion-Guided Generative Video Frame Interpolation** — Jisoo Jeong et al.; 2026-08-13; [arXiv:2608.13460](https://arxiv.org/abs/2608.13460). Motion-guided generative interpolation. **Why it matters:** a confidence-gated invalid-motion residual control only; it does not challenge the single-canvas, fractional-time render architecture.

### Inference Optimization
- No new material TensorRT/INT8/kernel-fusion paper changed priorities. Profile actual packed-canvas layout, sort/bin cost, and L2 hit rate (#12/#13) before locking precision, sorting, or fusion decisions.

## Archive.org Findings
- Advanced Search is still low signal: `"gaussian splatting" super-resolution` returned three GitHub snapshot artifacts; `"frame generation" deep learning` returned mixed FluidFrames/registration items; `"real-time upscaling" game` returned zero.
- Targeted terms were also non-confirmatory: GS-STVSR, GRAPE, and STSS returned zero; GSASR returned an unrelated CIA record. This does **not** establish absence/takedown—Archive.org discovery is simply not a credible primary archive for these papers.

## GitHub / Open Source
- **[IE-SRGS](https://github.com/ZheXues/IE-SRGS)**: official AAAI 2026 3DGS-SR implementation, 1 star, no asserted license, pushed Aug 29. Offline high-fidelity reconstruction; watch for selective detail-routing ideas, not a product threat.
- **[SelectiveSR](https://github.com/gomzee1024/SelectiveSR)**: 0 stars/unlicensed, pushed Aug 20. Claims scale-invariant GS-SR but has no credible deployment evidence.
- No credible new open-source DLSS/FSR analysis tool surfaced from GitHub REST search. Fresh “frame generation” and “neural upscaler” hits were diffusion/ComfyUI or hobby projects, not engine-MV competitors.

## News
- **NVIDIA:** DLSS 4.5’s second-generation SR transformer is available through NVIDIA App; Dynamic MFG/6× MFG are the announced trajectory. This is a quality/feature-bar threat but not an architectural refutation of OSS’s shared SR/OSS-FX canvas. [Primary source](https://blogs.nvidia.com/blog/dlss-path-tracing-g-sync-pulsar-ces-2026/).
- **AMD — material competitive change:** FSR SDK 2.3 packages FSR Upscaling **4.1.1**, ML FSR Frame Generation **4.0.1**, and Ray Regeneration 1.2. It says its ML upscaler now supports RX 7000 (RDNA 3) and provides driver-updatable, prebuilt/signed DX12 DLLs for previously integrated games. OSS’s deployment/runtime story needs equivalent versioning/compatibility discipline, even if it stays open. [Primary source](https://gpuopen.com/amd-fsr-sdk/).
- **Intel:** Intel’s Aug 25 Arc driver 32.0.101.8991 covers Arc A/B and Core Ultra platforms; Intel also announced launch-day XeSS 3 SR/MFG/low-latency enablement for two upcoming games. This raises the cross-vendor integration bar but yielded no new open implementation path. [Driver](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html) · [announcement](https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships).
- **Apple/Metal and UE/Unity:** no primary new MetalFX/Apple Silicon performance or engine-upscaling change was verified in the 14-day window.

## Action Items / Follow-ups
- [ ] **P0 ops:** recover, terminate, or explicitly document the v7 run; the measured step/loss have not moved and trainer-offline time is now 157h.
- [ ] **#12/#13:** add a rate-distortion/residency bench inspired by M³ISR: bytes/Gaussian, active Gaussians/tile, L2 hit rate, raster time, and quality deltas at fixed 1080p-LR budget.
- [ ] **Anti-aliasing ablation:** test motion-conditioned covariance/filter strength versus current Mip-Splatting under zoom and high-MV cases; report shimmer/temporal metrics, not PSNR alone.
- [ ] **Student gate:** benchmark a NanoVSR-style reparameterizable ≤1M residual CNN as a canvas-aware refinement/control with TensorRT FP8/INT8 and fused 1080p-LR latency.
- [ ] **Competitive watch:** track AMD’s driver-updatable FSR DLL behavior and RX 7000 support; OSS needs a compatibility/version-update plan before a DLL-shim claim is credible.

## Sources

- Project repository: [README](https://github.com/cashcon57/open-supersampling/blob/main/README.md) · [open issues](https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100)
- Dashboard: [opensupersampling.com](https://opensupersampling.com)
- Primary academic metadata: [arXiv Atom API](https://export.arxiv.org/api/query)
- Archive inventory: [Internet Archive Advanced Search](https://archive.org/advancedsearch.php)
- GitHub discovery: [GitHub repository search](https://api.github.com/search/repositories)
