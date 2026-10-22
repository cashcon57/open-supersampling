# OSS Research Brief — 2026-06-01

## Project Status Summary
- **Latest commits (top 10):**
  - `0660f55` wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc
  - `56a9a65` trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)
  - `d1255c8` trainer: WindowsPath shim on Linux for cross-OS checkpoint resume
  - `e1ab7fe` hat.py: use flash-attn 2 directly when available, fallback to SDPA
  - `e9b4b65` trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)
  - `26cd9da` dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB
  - `83b4f07` v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast
  - `baf4b02` worker: apex .com → .org 301 redirect
  - `eb0ec2e` trainer: bf16 autocast around forward + loss
  - `5c2f2e7` trainer: add throughput baselines (cudnn.benchmark, TF32, torch.compile)
- **Training dashboard:** `srcnn-v7.0-pico-005` at step 9,100/100K (9.1%), loss 0.12692, ETA ~1.6 days. **⚠ Trainer — degraded** (stale 23h, GPU host offline — GPU stats unavailable). Watcher/CF Worker/R2 all healthy.
- **Open issues:** 10 open — highest priority: #14 (measure v6.2-pico-002 VRAM), #13 (Nsight L2 hit-rate profile), #12 (packed low-rank canvas), #10 (decoupled G-buffer encoder ablation), #7 (FSR 4 binary teacher distillation)
- **Key observation:** Trainer host appears to have gone offline ~30h ago. v7-pico-005 stalled at step 9100. This needs immediate attention — the run may need restart on a different host.
- **UPDATE (late June 1):** Re-verified dashboard — trainer still degraded, same step 9100. No recovery in the past ~7h since first check.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** (arXiv:2605.11489, submitted 12 May 2026) — **HIGH RELEVANCE.** Unified post-rendering framework that jointly performs super sampling AND frame interpolation from low-resolution 3DGS renders. Gradient-Aware Super Sampling (GASS) leverages 3DGS differentiability for GRU-based refinement; Lightweight Temporal Frame Interpolation (LTFI) uses compact U-Net backbone fusing temporal + spatial cues. Code will be released upon acceptance. **Why it matters to OSS:** This is the closest conceptual analog to OSS's "one architecture for both SR and FG" thesis — they're doing exactly that (SS + FI from a shared GS representation) but as a post-processing layer rather than a native canvas. Their GASS module's use of analytical GS gradients for refinement is similar to GS-STVSR's covariance resampling and validates the approach.

- **STGV: Spatio-Temporal Hash Encoding for Gaussian-based Video Representation** (arXiv:2604.10910, revised 14 Apr 2026) — (Previously tracked in BIBLIOGRAPHY.) Decomposes video features into 2D spatial + 3D temporal hash encodings, +0.98 dB PSNR over other GS video methods. Key-frame canonical initialization prevents feature overlap. **Why it matters to OSS:** The spatial/temporal hash encoding split directly parallels OSS's static canvas + dynamic spawner design. The hash encoding approach could be adapted for OSS's canvas memory layout — reducing memory while maintaining quality.

- **RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting** (arXiv:2603.13783, CVPR 2026) — (Previously tracked.) Diagnoses 4DGS interpolation ghosting as "temporal aliasing"; optical-flow-guided initialization + triple-rendering supervision enables artifact-free continuous-time rendering. **Why it matters to OSS:** Validates OSS's temporal anti-aliasing via Mip-Splatting + covariance resampling. The explicit temporal behavior definition per Gaussian is analogous to v7's V_xt cross-correlation in the Cholesky.

- **VidSplat: Gaussian Splatting Reconstruction with Geometry-Guided Video Diffusion Priors** (arXiv:2605.11424, SIGGRAPH 2026, submitted 12 May 2026) — Training-free generative reconstruction using video diffusion priors for sparse-view (even single-image) 3D scene completion. Geometry-guided stage-wise denoising. **Why it matters to OSS:** The "training-free" + "generative prior" pattern could apply to OSS's data-augmentation problem — using diffusion to hallucinate unseen views for training data expansion, especially for the disocclusion-challenged frame pairs.

- **1000+ FPS 4D Gaussian Splatting for Dynamic Scene Rendering** (arXiv:2503.16422, NeurIPS 2025) — (Previously tracked.) Spatial-Temporal Variation Score pruning + active Gaussian masks achieve 41× storage reduction and 9× faster rasterization at comparable quality. **Why it matters to OSS:** Already adopted (S-T variation score + key-frame active mask). The 4DGS-1K paper is the direct source of OSS's per-frame splat reduction strategy. Recent acceptance at NeurIPS 2025 adds peer-reviewed validation.

- **StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video** (arXiv:2511.06046, AAAI 2026) — (Previously tracked.) Canonical GS as 2D image + temporal features as video stream; 170 KB/frame, adaptive bitrate, +1 dB PSNR. Code available at github.com/kkkzh/StreamSTGS. **Why it matters to OSS:** Compression design for canvas serialization in cloud/streaming deployment. The 2D-image encoding of canonical GS attributes is a concrete design pattern OSS can adopt.

- **3D Gaussian Splatting for Efficient Retrospective Dynamic Scene NVS with a Standardized Benchmark** (arXiv:2605.12437, CVPR 2026 Workshop, 12 May 2026) — Challenges the necessity of explicit temporal coupling for synchronized multi-view dynamic scenes. Propagates optimized Gaussians over time WITHOUT temporal deformation constraints. Includes Blender-based dynamic MV dataset framework. **Why it matters to OSS:** If synchronized MV alone provides sufficient spatial consistency without temporal coupling, OSS's canvas-warp-by-MV approach is validated (we use exact engine MVs, which are stronger than synchronized viewpoints). The dataset framework is useful for standardized OSS benchmarks.

### Super-Resolution / Upscaling
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047, Apr 20 2026) — **NEWLY DETAILED.** Models spatiotemporal evolution of Gaussian kernels via continuous motion modeling, bypassing dense grid queries entirely. Exploits temporal stability of covariance parameters for lightweight intermediate fitting. Optical flow-guided motion module + Covariance Resampling Alignment module prevents covariance drift. **Inference time nearly constant at ×2–×8 temporal scales, >3× speedup at ×32.** State-of-the-art on Vid4, GoPro, Adobe240. **Why it matters to OSS:** This is the most direct architectural precedent for OSS's canvas warping — GS-STVSR's covariance resampling alignment module is the offline analog of OSS's Mip-Splatting anti-aliasing filters. Their key insight (covariance parameters are temporally stable → lightweight interpolation) validates OSS's v7 V_xt cross-correlation in the Cholesky. The constant-time inference at all scales is the efficiency property OSS-FX needs.

- **CDA-VSR: Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694, CVPR 2026) — **NEW.** Uses compressed-domain info (motion vectors, residual maps, frame types) for online VSR. Motion-vector-guided deformable alignment: coarse warp from MVs + learned local residual offsets. Residual-map-gated fusion suppresses mismatched regions. Frame-type-aware reconstruction for adaptive compute. +0.13 dB PSNR over SOTA, >2× faster inference on REDS4. Code at github.com/sspBIT/CDA-VSR. **Why it matters to OSS:** The MV-guided alignment approach is directly applicable — OSS already has exact engine MVs; using them for coarse warping with learned fine offsets mirrors our canvas-warp design. The residual-map gating could inform OSS's disocclusion handling.

- **SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images** (arXiv:2511.12040, AAAI 2026 Oral) — Feed-forward 3D reconstruction from sparse LR views using external reference images (MLLM + diffusion generated) + internal texture cues. Reference-Guided Feature Enhancement (RGFE) + Texture-Aware Density Control (TADC). Strong cross-dataset generalization. **Why it matters to OSS:** The feed-forward Gaussian prediction + adaptive density control (TADC) is directly relevant to OSS's spawner logic — TADC is essentially "add more Gaussians where texture is complex," which parallels OSS's parent-child loss-adaptive density. RGFE's external-reference approach could help with OSS's disocclusion hallucination problem.

- **NTIRE 2026 Efficient Super-Resolution Challenge** (arXiv:2604.03198) — Eleventh iteration. 95 registered, 15 teams submitted. PSNR targets: 26.90 dB (DIV2K_LSDIR_valid), 26.99 dB (DIV2K_LSDIR_test). **Why it matters to OSS:** The challenge's efficiency metrics (parameters, FLOPs, runtime) are benchmark targets OSS can measure against. Winners often become the new SOTA architectures for lightweight SR.

- **Efficient INT8 Single-Image Super-Resolution via Deployment-Optimized Quantization** (arXiv:2604.20291, MAI 2026 Challenge) — **NEW.** 29.79 dB PSNR, 0.8634 SSIM on 4K SR with INT8 quantization. Deployment-optimized quantization scheme. **Why it matters to OSS:** Directly informs OSS's student model INT8/FP8 export strategy. Proves INT8 quality ceiling is viable for shipping-quality upscaling.

- **Mobile-GS: Real-time Gaussian Splatting for Mobile Devices** (arXiv:2603.11531, ICLR 2026) — **NEWLY NOTED.** First real-time GS on mobile: 116 FPS at 1600×1063 on Snapdragon 8 Gen 3. Depth-aware order-independent rendering + neural view-dependent color prediction. **Why it matters to OSS:** Proves GS rasterization is viable on mobile-class hardware. OSS's student model at ≤1M params with GS rasterization could run on Steam Deck and mobile dGPUs.

### Frame Extrapolation / Generation
- **IFRVP: Real-time Video Prediction With Fast Video Interpolation Model and Prediction Training** (arXiv:2503.23185, ICIP 2024) — Extends frame interpolation models for video prediction (extrapolation). ELAN-based residual blocks improve speed + accuracy. Best trade-off between prediction accuracy and speed among existing methods. **Why it matters to OSS:** Direct competitor for OSS-FX's extrapolation quality/speed claims. The "extend interpolation for prediction" approach is an alternative to OSS's α-conditioned canvas rendering — worth benchmarking against.

- **Consecutive Frame Extrapolation with Predictive Sparse Shading** (ACM, Dec 2025) — Neural architecture for consecutive frame extrapolation. **Why it matters to OSS:** Direct competitor for OSS-FX multi-frame extrapolation. The "predictive sparse shading" concept (compute only where motion is high) could complement OSS's active-Gaussian-mask approach.

### Inference Optimization
- No new TensorRT or INT8 quantization papers directly relevant to OSS were found this cycle. The INT8-SISR paper (Nguyen et al., CVPR 2026 Workshop) previously tracked in BIBLIOGRAPHY remains the primary reference for OSS's student model INT8/FP8 export strategy.

- **Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (arXiv:2603.26835) — Real-time 30→60 FPS frame interpolation on mobile NPUs using codec motion vectors. **Why it matters to OSS:** Demonstrates that motion vectors (even codec-level approximations) are sufficient for real-time frame interpolation on constrained hardware. Validates OSS's approach of using exact engine MVs as the primary motion signal.

## Archive.org Findings
- No substantive new archival discoveries. Most papers found are well-covered by arXiv + conference proceedings.

## GitHub / Open Source
- **Arm Neural Graphics SDK for Game Engines** (github.com/arm/neural-graphics-sdk-for-game-engines) — Active development with recent releases. Unified SDK for neural rendering use cases across engines. Includes NSS (Neural Super Sampling) Unreal Engine plugin. **Why it matters to OSS:** Arm is investing heavily in neural upscaling as an open, cross-platform SDK. This validates the "vendor-neutral upscaler" market thesis OSS is built on, but also represents a well-funded competitor targeting the same positioning.

- **Arm Neural Super Sampling (NSS)** — Shipping in late 2026 devices. 540p→1080p at ~4ms per frame on mobile GPUs. Pretrained open-source models via Hugging Face + GitHub. **Why it matters to OSS:** NSS proves mobile upscaling is viable. OSS's student model with cross-vendor kernels could target the same mobile market — the 4ms budget on mobile is comparable to OSS's <2ms target on desktop GPUs.

- **OptiScaler** (github.com/optiscaler/OptiScaler) — "Huge new release" (April 2026) with expanded upscaling and frame generation options. Now supports FSR 4 in Vulkan games, DLSS-to-FSR bridging, and frame gen injection. **Why it matters to OSS:** OptiScaler proves massive market demand for cross-vendor upscaler swapping. It's the integration-path precedent OSS can follow — if OptiScaler can inject FSR 4 into Vulkan games, OSS's DLL-shim can do the same.

- **OpenUSD v26.03** — New release adds native 3D Gaussian Splat schema + WebAssembly build support. **Why it matters to OSS:** Standardizing GS as a scene primitive in the dominant 3D interchange format. OSS's canvas serialization targeting glTF KHR_gaussian_splatting gets a second standardization path via USD.

- **WaveletSR: Blind Super-Resolution for Noisy 3D Gaussian Splatting** (CVPR 2026 Workshop, github.com/qzwamg-web/WaveletSR) — Wavelet-domain blind SR for noisy 3DGS renders. **Why it matters to OSS:** If OSS's canvas produces noisy Gaussians (especially at low SPP), wavelet-based denoising could be more efficient than traditional NR approaches.

## News

### 🔴 BREAKING: NVIDIA GTC Taipei at COMPUTEX 2026 (June 1, 2026 — TODAY)
- **Jensen Huang keynote delivered today** (May 31 PT / June 1 Taipei). Major announcements with OSS implications:
- **RTX Spark** — 1 petaflop AI PC built with MediaTek. Blackwell RTX GPU (6,144 CUDA cores) + 20-core Grace CPU fused by NVLink. Runs 100% of NVIDIA's software stack on Windows. Adobe rewriting Photoshop & Premiere for it (2× faster AI+graphics). **Why it matters to OSS:** RTX Spark creates a new hardware tier between desktop dGPU and iGPU — a target platform OSS's ≤1M-param student model could excel on. The "personal agent box" concept (always-on, 24/7) aligns with OSS's near-zero-marginal-cost frame extrapolation positioning.
- **DLSS 5 confirmed for Fall 2026** — "Real-time neural rendering model that infuses pixels with photoreal lighting and materials." Jensen's GTC keynote doubled down on this beyond upscaling into full neural scene re-rendering. **Why it matters to OSS:** DLSS 5's neural lighting+material re-render makes pure upscaling a subset. OSS's counter-positioning: the Gaussian canvas is engine-agnostic and accumulates whatever the renderer feeds it — including neural-lit content.
- **Cooperative Vectors now shipping across D3D12/Vulkan/Metal** — GDC 2026 finalized the cross-API standard for shader-level NN inference. AMD, Microsoft, NVIDIA, and Apple all aligned. Slang shader language supports neural shaders directly. **Why it matters to OSS:** This is THE deployment target for OSS's student model. Cooperative Vectors eliminates the GPU-CPU round-trip for NN inference — the student runs inside the rendering pipeline. OSS should design its ≤1M-param student export targeting Cooperative Vectors, not standalone TensorRT.
- **NVIDIA OpenShell agent runtime** — Secure sandboxed runtime for AI agents on Windows/Ubuntu/Red Hat. Individual sandboxes, centralized policy, governance gateway. **Why it matters to OSS:** OSS could eventually ship as an agent that auto-tunes per-game — OpenShell provides the secure runtime.
- **DGX Station for Windows** — Trillion-parameter AI supercomputer on the enterprise desk. **Why it matters to OSS:** Not directly relevant to game upscaling, but signals NVIDIA's commitment to putting massive AI compute on desks — the hardware ceiling keeps rising.

### NVIDIA / DLSS
- **DLSS 5 announced at GTC 2026** — "Real-time neural rendering model that infuses pixels with photoreal lighting and materials." Arriving Fall 2026. This is NVIDIA's biggest graphics leap since ray tracing — moving beyond upscaling into full neural scene re-rendering. **Why it matters to OSS:** DLSS 5 raises the competitive bar substantially. If NVIDIA ships neural lighting + materials as a driver-level feature, the "just upscaling" category becomes a subset of what DLSS offers. OSS's response: the Gaussian canvas is engine-agnostic and accumulates whatever the renderer feeds it — including neural-lit content. But DLSS 5's tight RTX integration gives NVIDIA a latency advantage OSS can't match without equivalent hardware access.

- **DLSS 4.5 + Multi Frame Generation 6X at CES 2026** — Updated models generate higher-quality upscaled frames and more of them dynamically. MFG 6X = up to 6 AI-generated frames per rendered frame. **Why it matters to OSS:** OSS-FX's α-conditioned rendering produces extrapolated frames at near-zero marginal cost — but DLSS 4.5's 6X MFG is a brute-force approach that may overwhelm OSS's single-canvas efficiency advantage if NVIDIA's hardware acceleration scales quadratically with frame count.

- **RTX Kit + Neural Shaders at GDC 2026** — NVIDIA embedding neural networks directly in shaders (Cooperative Vector support in DirectX/Vulkan via Microsoft). RTX Neural Shaders, RTX Neural Texture Compression, RTX Mega Geometry. **Why it matters to OSS:** Neural Shaders are the direct hardware path OSS would need to achieve <2ms on NVIDIA GPUs. The Cooperative Vectors API (DirectX) is the shader-level compute primitive that makes "run a tiny NN inside a pixel shader" practical. OSS's student model should target Cooperative Vectors as the deployment target rather than pure TensorRT — shader-level integration is lower latency and avoids the GPU-CPU round-trip.

### AMD / FSR
- **FSR Redstone** — AMD's "most comprehensive further development" of upscaling and frame generation. ML-powered. No-show at GDC 2026 is "troubling" per press. **Why it matters to OSS:** If FSR Redstone is delayed (no GDC presence), it creates a window for OSS to ship before AMD establishes its next-gen ML upscaler. Every month of FSR delay is a month OSS can claim "only vendor-neutral ML upscaler in active development."

- **FSR 4.1 coming to RDNA 3 (RX 7000) in July 2026, RDNA 2 (RX 6000) in 2027** — AMD extending ML-powered upscaling to older GPUs. INT8 quantization for older hardware. **Why it matters to OSS:** This is great news for OSS positioning — if AMD can run FSR 4.1 on RDNA 2 via INT8, OSS's student model (targeting FP8/INT8) can do the same. The "older GPU compatibility" story is now validated by a major vendor.

### Intel / XeSS
- **XeSS 3 multi-frame generation rolled out to all Arc GPUs** — Including Panther Lake iGPUs. Medium preset 1080p, 20 FPS on 4 Xe-unit iGPU (no upscaling). **Why it matters to OSS:** Intel is the only vendor shipping MFG on integrated graphics. OSS's student model at ≤1M params could potentially run on iGPU-class hardware — Intel's MFG deployment proves the hardware is capable of NN inference alongside rendering.

### Arm / Mobile
- **Arm Neural Frame Rate Upscaling** — Coming in 2026: AI-based frame rate doubling. Expands the NSS roadmap beyond spatial upscaling to temporal. **Why it matters to OSS:** Arm is converging on the same SR+FG unified architecture concept OSS is built on. They're doing it from the mobile side with hardware acceleration; OSS is doing it from the desktop side vendor-neutrally. The market intersection is significant.

### Game Engines
- **Unity deprecating Built-in Render Pipeline in 2026** — Moving to URP/HDRP only. Includes upscaling framework that's cross-pipeline. **Why it matters to OSS:** Unity's upscaling framework creates a natural integration point for OSS as a plugin. Less pipeline fragmentation = easier to target.

- **GDC 2026: Arm neural graphics, Vulkan ML, Unreal Engine neural rendering** — Mobile gaming trends converging on neural rendering in engines. **Why it matters to OSS:** Neural rendering is becoming table stakes across all platforms. OSS's vendor-neutral positioning is more valuable as the market fragments across NVIDIA DLSS, AMD FSR, Intel XeSS, Arm NSS, and Apple MetalFX.

## Action Items / Follow-ups
- [ ] **CRITICAL: Restart v7-pico-005 trainer** — Stuck at step 9100 for ~30h. GPU host offline. Follow recovery doc (step 5000 snapshot preserved) and resume on a working host.
- [ ] **Deep-read 3DGS³ (arXiv:2605.11489)** — Closest conceptual analog to OSS's unified SR+FG thesis. Their GASS + LTFI post-processing approach vs. OSS's native canvas approach needs comparative analysis.
- [ ] **Deep-read GS-STVSR (arXiv:2604.18047)** — Covariance resampling alignment module is the most direct precedent for OSS's v7 canvas warping. Their constant-time inference property validates OSS-FX's efficiency thesis.
- [ ] **Evaluate CDA-VSR's MV-guided alignment** (arXiv:2603.07694) — MV coarse warp + learned residual offsets directly parallels OSS's canvas warp. Their residual-map gating could improve disocclusion handling.
- [ ] **Adopt Cooperative Vectors as primary deployment target** — Cross-API standard (D3D12/Vulkan/Metal) finalized at GDC 2026. OSS's ≤1M-param student should export to Slang neural shaders, not standalone TensorRT.
- [ ] **Benchmark ContinuousSR (arXiv:2503.06617, ICLR 2026) against OSS** — 1ms/scale Gaussian field rendering. Direct competitor for SR path.
- [ ] **Evaluate SRSplat's TADC (Texture-Aware Density Control) for OSS spawner** — Adaptive Gaussian density based on texture complexity.
- [ ] **Track FSR Redstone launch timing** — If delayed beyond Q3 2026, OSS has a competitive window.
- [ ] **Evaluate RTX Spark as OSS target platform** — 1 petaflop AI PC with Blackwell GPU. Perfect hardware tier for ≤1M-param student model + GS rasterizer.
- [ ] **Monitor Arm NSS + Neural Frame Rate Upscaling** — Arm building mobile equivalent of OSS's unified SR+FG vision.
- [ ] **Study VidSplat's geometry-guided diffusion approach for OSS data augmentation** — Training-free generation of unseen views.
