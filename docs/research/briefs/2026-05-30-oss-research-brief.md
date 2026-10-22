# OSS Research Brief — 2026-05-30

## Project Status Summary

**Latest commits:** Same top-10 as yesterday — no new commits since the wipe-prep snapshot at `0660f55`.

**Training dashboard:** `srcnn-v7.0-pico-005` at step 8,900 / 100,000 (8.9% complete). **Trainer degraded** — stuck at step 8900 for 8min+ at time of check. Watcher, CF Worker, and R2 origin all healthy. Loss at step 8,900: 0.10726. ETA ~1.6 days (if it recovers).

**Open issues:** Same 12 open — no new issues, no closures.

**Key observation:** Trainer moved from step 5,000 (yesterday's snapshot) to 8,900 — so training DID resume after the wipe-prep commit, but is now exhibiting intermittent stalling. The "degraded" state may be a transient I/O or WSL2 issue given the recent `num_workers=0` revert.

---

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **[STGV: Spatio-Temporal Hash Encoding for Gaussian-based Video Representation]** (arXiv:2604.10910, Apr 2026) — Decomposes video features into 2D spatial + 3D temporal hash encodings, preventing static/dynamic entanglement. +0.98 dB PSNR over prior 2DGS video methods. **Why it matters to OSS:** The spatial/temporal hash decomposition mirrors OSS v7's separation of canvas (spatial) from motion (temporal V_xt in Cholesky). Could inform how v7 structures its feature encodings for better disentanglement. The key-frame canonical initialization strategy is also relevant to how v7 initializes its N-D Gaussians.

- **[RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting]** (arXiv:2603.13783, CVPR 2026) — Explicitly models temporal behavior of 3D Gaussians to avoid ghosting at arbitrary timestamps. Uses optical flow-guided initialization + triple-rendering supervision. **Why it matters to OSS:** Directly relevant to OSS-FX. RetimeGS identifies temporal aliasing as the root cause of ghost artifacts between discrete training frames — the same problem OSS-FX faces when rendering at α∈(0,1). The optical flow-guided initialization could strengthen v7's motion field quality, and the triple-rendering supervision strategy is worth investigating for OSS-FX training.

- **[3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering]** (arXiv:2605.11489, May 12 2026) — **HIGH RELEVANCE.** Proposes a unified post-rendering framework that jointly performs super sampling AND frame interpolation on low-resolution 3DGS outputs. Gradient-Aware Super Sampling (GASS) uses 3DGS differentiability for image gradients fed into a GRU refinement network. Lightweight Temporal Frame Interpolation (LTFI) with compact U-Net backbone. **Why it matters to OSS:** This is the closest academic analogue to OSS's "one architecture for SR+FG" thesis — but as a post-process on 3DGS renders rather than integrated into the canvas. The GASS module's use of differentiable rendering gradients is conceptually similar to OSS's covariance-aware rasterizer. Their LTFI module is a lightweight U-Net (not a separate large network), validating OSS's belief that frame interpolation doesn't need a heavy second model. Code pending acceptance — monitor for release.

- **[AA-Splat: Anti-Aliased Feed-forward Gaussian Splatting]** (arXiv:2603.29394, Mar 2026) — First FF-3DGS model with robust anti-aliased rendering at any resolution. Opacity-Balanced Band-Limiting (OBBL): 3D band-limiting post-filter + opacity balancing. 5.4–7.5 dB PSNR gains over DepthSplat baseline. **Why it matters to OSS:** The band-limiting + opacity balancing approach is directly applicable to v7's Mip-Splatting anti-aliasing filters. The 3D post-filter that integrates multi-view frequency bounds could replace or augment the current fp32 smoothing filters. The opacity balancing technique addresses the same expanded-Gaussian-overlap problem that v7's cosine-feathered overlapping rasterizer tiles were designed for.

- **[SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images]** (arXiv:2511.12040, AAAI 2026 Oral) — Feed-forward 3DGS SR from sparse LR views using MLLM-generated reference images + Texture-Aware Density Control. **Why it matters to OSS:** Low relevance for game SR (scene-specific reference gallery generation is offline/expensive), but the TADC adaptive density control based on internal texture cues could inform v7's parent-child loss-adaptive density mechanism.

### Super-Resolution / Upscaling

- **[ContinuousSR: Pixel to Gaussian — Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling]** (arXiv:2503.06617, ICLR 2026) — **HIGH RELEVANCE.** Builds a continuous 2D Gaussian field from LR images in a single feed-forward pass. Arbitrary-scale rendering in ~1ms per scale. From USTC + Huawei Noah's Ark Lab. Open source at github.com/peylnog/ContinuousSR. **Why it matters to OSS:** This is the closest prior work to OSS's 2D Gaussian canvas approach for image SR. The Pixel-to-Gaussian paradigm (feed-forward Gaussian prediction, bypassing per-scene optimization) is exactly what OSS does for game frames. Key difference: ContinuousSR is single-image (no temporal accumulation), but its 1ms rendering speed for arbitrary scales validates the viability of Gaussian-canvas rendering at OSS's latency targets. Strong candidate for integration as a baseline in the v7 eval pipeline. **Code is open source — action item to benchmark against v7.**

- **[GameSR: Real-Time Super-Resolution for In-Game Content]** (OpenReview, 2025/2026) — Lightweight neural SR operating directly on rendered frames without game source code access. **Why it matters to OSS:** Validates the market for engine-agnostic upscalers. The no-source-code-required approach is the same integration philosophy as OSS's DLL-shim. Worth benchmarking if code is available.

- **[NTIRE 2026 Efficient Super-Resolution Challenge]** (arXiv:2604.03198, CVPRW 2026) — Annual benchmark of efficient SR models. Samsung SRC-B won x4 SR track with cascaded framework. **Why it matters to OSS:** The challenge's efficiency metrics (runtime, params, FLOPs) provide a standardized baseline for OSS's ≤1M-param student model to target. The winning cascaded architecture may offer distillation insights.

### Frame Extrapolation / Generation

- **[Consecutive Frame Extrapolation with Predictive Sparse Shading]** (ACM, Dec 2025) — Neural network for consecutive frame extrapolation using sparse shading predictions. **Why it matters to OSS:** One of the few papers explicitly targeting frame *extrapolation* (not interpolation). The sparse shading approach reduces compute by predicting which regions need updates — conceptually similar to OSS's canvas reuse where only motion-affected regions change.

- **[Real-time Video Prediction With Fast Video Interpolation Model]** (arXiv:2503.23185) — Extends frame interpolation models for video prediction (extrapolation) using convolution-only architectures. **Why it matters to OSS:** Lightweight, convolution-only approach aligns with OSS's student model design constraints. The training method of extending interpolation models to prediction may offer strategies for OSS-FX training.

### Inference Optimization

- **[NVIDIA TensorRT FP8/INT8 for Super-Resolution]** — No new specific papers on TensorRT for SR in this scan window. General optimization landscape remains: INT8 quantization for 4× speedup, kernel fusion for reduced launch overhead. **Why it matters to OSS:** The v6.2 student model's TensorRT FP8 export path is the critical path to <2ms inference. The NTIRE 2026 challenge results provide practical INT8/FP8 quantization baselines from competing efficient SR models.

---

## Archive.org Findings

No substantive new archival discoveries this scan. The Gaussian splatting SR paper landscape is well-covered by arXiv and conference proceedings (CVPR, ICCV, NeurIPS 2025/2026, ICLR 2026). Key papers (GSASR, GaussianSR, ContinuousSR, SuperGS) all have live arXiv pages and GitHub repos — no takedowns detected.

---

## GitHub / Open Source

- **[ContinuousSR](https://github.com/peylnog/ContinuousSR)** — **NEW & RELEVANT.** Open-source implementation of the ICLR 2026 Pixel-to-Gaussian continuous SR framework. Feed-forward 2D Gaussian field prediction with arbitrary-scale rendering. **Action item: benchmark against v7 rasterizer.**

- **[GSASR](https://github.com/chrisdud0257/gsasr)** — ICCV 2025 code for generalized 2D Gaussian splatting arbitrary-scale SR. Already tracked in OSS's bibliography from the May 5 deep-dive. No major updates detected.

- **[OptiScaler](https://github.com/optiscaler/OptiScaler)** — Major April 2026 release with rewritten FG settings, expanded upscaling/frame-gen compatibility across GPUs. **Why it matters to OSS:** OptiScaler demonstrates real market demand for cross-GPU, vendor-neutral upscaler swapping — validates OSS's cross-vendor thesis. The April 2026 update adding more upscaling backends shows the ecosystem is actively fragmenting (DLSS 4.5, FSR 4.1, XeSS 3 all shipping). A DLL-shim approach is proven viable.

- **[NVIDIA RTX Kit 2026.2](https://github.com/NVIDIA-RTX/RTX-Kit)** — March 2026 release: ReSTIR PT for path tracing, DLSS-RR integration, Neural Shaders improvements. **Why it matters to OSS:** NVIDIA is doubling down on neural rendering as the future of game graphics. DLSS 4.5's second-gen transformer SR model is available in NvRTX 5.7 (Unreal Engine branch). The RTX Kit's Neural Shaders SDK allows embedding small neural networks directly in shaders — potentially competitive with OSS's student model if NVIDIA starts shipping in-shader SR models.

- **[Arm Neural Super Sampling](https://newsroom.arm.com/blog/takeaways-from-gdc-festival-of-gaming-2026)** — Arm's Neural Frame Rate Upscaling targets mobile GPUs with Vulkan ML extensions. 540p→1080p in ~5.4ms total (4ms NN + 1.4ms shader). Open models on GitHub/Hugging Face, Unreal Engine plugin. **Why it matters to OSS:** Arm entering the neural upscaling space validates the market; their ~5ms budget on mobile is roughly comparable to OSS's <2ms target on desktop (accounting for hardware gap). The Vulkan ML extension path is worth monitoring for OSS's cross-vendor kernel strategy.

---

## News

### DLSS 4.5 — Second-Gen Transformer SR (March 2026, live now)
NVIDIA's DLSS 4.5 (shipping since March) features a **second-generation transformer model** for Super Resolution — a direct upgrade to the DLSS 4 transformer. Available in 400+ titles via NVIDIA app override. Combined with Dynamic Multi-Frame Generation 6X. **Why it matters:** DLSS 4.5's quality bar is the moving target OSS needs to beat. The fact that NVIDIA upgraded from a first-gen to second-gen transformer in ~18 months suggests rapid iteration cycles. If the SR transformer keeps improving at this rate, OSS's PSNR gap to DLSS may be a moving target.

### FSR 4.1 — Backward Compatibility Expansion (May 14, 2026)
AMD announced FSR 4.1 ML-powered upscaling coming to RX 7000 series (July 2026) and RX 6000 series (early 2027). This is a reversal from the initial FSR 4 RDNA 4-only stance. **Why it matters:** AMD's backward compatibility push validates the market demand for hardware-agnostic upscaling. If OSS ships cross-vendor, it occupies the exact space FSR 4.1 is now expanding into — the window for OSS to claim "open, cross-vendor upscaling" is still open but narrowing as AMD fills it.

### FSR Diamond / RDNA 5 (GDC 2026, previously covered)
No new details beyond yesterday's coverage. FSR Diamond remains RDNA 5 exclusive, co-engineered with Microsoft for Project Helix.

### Intel XeSS 3 / Arc Cancellation
Intel XeSS 3 Multi-Frame Generation is now shipping on Arc Alchemist/Battlemage and Panther Lake integrated graphics. However, Intel cancelled Xe3P "Celestial" discrete gaming GPUs. **Why it matters:** Intel's pullback from discrete gaming GPUs narrows the dGPU upscaling landscape to NVIDIA + AMD. This is a small net positive for OSS — fewer competitors to target with cross-vendor kernels, and Intel's XeSS will likely remain limited to integrated graphics, reducing its relevance as a quality benchmark.

### Unity Render Pipeline Strategy 2026
Unity is deprecating the Built-in Render Pipeline in favor of URP/HDRP consolidation. **Why it matters:** For OSS's cross-engine strategy, fewer Unity render pipelines means fewer integration targets. Unity's "Neonage" upscaling framework is cross-pipeline — OSS should monitor whether it becomes a built-in competitor or an integration opportunity.

### NVIDIA RTX Kit + NvRTX 5.7 (March 2026)
NvRTX 5.7 (NVIDIA's Unreal Engine 5 branch) includes DLSS 4.5 Super Resolution with second-gen transformer, ReSTIR PT for path tracing, and Linear Swept Sphere for strand-based hair. **Why it matters:** This is the integration path NVIDIA is offering developers — OSS needs to understand how DLSS hooks into UE5 to design a compatible DLL-shim replacement.

---

## Action Items / Follow-ups

- [ ] **CRITICAL: Benchmark ContinuousSR against v7 rasterizer.** Open-source, ICLR 2026, 1ms arbitrary-scale rendering. The closest academic prior to OSS's 2D Gaussian canvas. Code at github.com/peylnog/ContinuousSR.
- [ ] **Deep-read 3DGS³ (arXiv:2605.11489).** Joint SR + frame interpolation in a single post-rendering framework. The GASS module's gradient-guided refinement could inform v7's rasterizer improvements. Monitor for code release.
- [ ] **Deep-read RetimeGS (arXiv:2603.13783, CVPR 2026).** Temporal anti-aliasing for 4DGS — directly applicable to OSS-FX ghosting prevention. Triple-rendering supervision strategy worth adapting.
- [ ] **Investigate AA-Splat's OBBL for v7 anti-aliasing.** Band-limiting post-filter + opacity balancing could improve upon current Mip-Splatting filters.
- [ ] **Monitor STGV (arXiv:2604.10910) spatial/temporal hash encoding.** The decomposition approach mirrors v7's canvas/motion separation — code not yet released.
- [ ] **Check ContinuousSR inference speed on 3080 Ti.** The claimed 1ms/scale is likely on datacenter GPUs; need to benchmark on consumer hardware to assess relevance to OSS's <2ms target.
- [ ] **Track FSR 4.1 RX 7000 launch (July 2026).** If AMD's ML upscaling becomes broadly available on older hardware, it blunts OSS's "open cross-vendor" advantage. Time-to-market matters.
- [ ] **Monitor Arm NSS open models.** If Arm releases trained models, they could serve as mobile baselines or distillation targets for OSS's student model on low-power targets.
- [ ] **Trainer health: investigate v7-pico-005 stall at step 8900.** If the degradation continues, may need manual intervention (WSL2 deadlock again?).
