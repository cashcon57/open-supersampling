# OSS Research Brief — 2026-07-26

_Collection: local git/README, GitHub REST, rendered dashboard, arXiv Atom plus direct arXiv page verification, Archive.org Advanced Search/Wayback CDX, GitHub repository search, direct NVIDIA/Intel/AMD sources, and Google News RSS. RSS-only claims are labelled as leads._

## Project Status Summary
- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated training controls and API; preceding work preserves the v7 recovery checkpoint, fixes WSL2 worker deadlock/cross-OS resume, enables FlashAttention-2/TF32 prefetch, and keeps smoothing filters FP32 under bf16.
- **Open issues:** **12**. Near-term architecture/perf gates remain #14 v6.2 VRAM measurement, #13 Nsight L2 profile, #12 packed low-rank canvas, #9 reparameterized student backbone, #8 cross-vendor ports, #7 FSR 4 teacher signal, and #4 ≤1M student distillation.
- **README snapshot:** v7 `srcnn-v7.0-pico-005` is a HAT-Tiny teacher using native `(x,y,t)` Gaussians and `V_xt` covariance motion; the ≤1M TensorRT-FP8 student and non-CUDA runtimes remain unbuilt. Published validation is still v5 only: 25.703 PSNR / 0.1666 LPIPS on held-out TartanAir `oldtown`.
- **Training dashboard (rendered 2026-07-26):** hero says **36,650 / 100,000**, loss **0.19268**, viz **36,000**, ETA ~1.1d. Service-health separately says **Trainer offline, stuck at step 36,650 for 5h**; watcher is healthy (published 1s ago), CF Worker HTTP 200 in 90ms, R2 HTTP 200, and GPU stats are unavailable. Versus the 2026-07-25 brief's 35,700 / 0.17966, the step advanced 950 while the loss changed non-monotonically; the run is now stalled again. Public `live` copy only reflects publication/CDN health, not trainer health.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **GrainGS: Gradient-Decoupled Gaussian Splatting for Efficient Dynamic Novel View Synthesis** — Jiahao He, Yihua Shao, Zhengkai Zhao, Pan Gao, Fei Ma, Jingcai Guo, Hao Tang, Nicu Sebe, Qi Tian; **2026-07-23**; [arXiv:2607.21448](https://arxiv.org/abs/2607.21448). Hierarchical anchors with local temporal offsets, canonical/residual appearance, and stop-gradient isolation stabilize dynamic Gaussians. **Why it matters:** direct v7 ablation reference for preserving canonical canvas state while updating `V_xt` motion/density; already tracked in `BIBLIOGRAPHY.md`.
- **Compression of 3D Gaussian Splatting Data Using GPU-friendly Graphics Texture Coding** — Amir Said, Randall Rauwendaal; **2026-07-16**; [arXiv:2607.14513](https://arxiv.org/abs/2607.14513). Uses BC1/BC7-style texture compression for random-access, hardware-decoded Gaussian color data with negligible/imperceptible loss. **Why it matters:** direct packed-canvas/bandwidth hypothesis for #12/#13 and cross-vendor decode paths; already tracked.

### Super-Resolution / Upscaling
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** — Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; **2026-07-11**; [arXiv:2607.10495](https://arxiv.org/abs/2607.10495). A fully convolutional, structurally reparameterized VSR model removes explicit flow/custom operations; its 644k variant reports 28.64 dB on REDS4 and 27.2 FPS on Jetson Orin NX. **Why it matters:** strongest concrete control for #9/#4’s deployment-friendly student path; its upstream MIT repo was pushed 2026-07-25 and has 27 stars. Already tracked.
- No fresher directly material game/VSR paper appeared after the prior brief; broad results were primarily diffusion/restoration or off-domain work and failed the OSS relevance gate.

### Frame Extrapolation / Generation
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** — Shibo Liu; **2026-03-27**; [arXiv:2603.26835](https://arxiv.org/abs/2603.26835). Reuses decoder motion vectors to prealign frames and keeps the learned residual convolution-dominated; reports 12.8ms 1080p INT8 on Snapdragon 8 Gen 3. **Why it matters:** a materially relevant but older baseline: exact external MVs can eliminate optical-flow/sampling-heavy inference, reinforcing OSS’s engine-MV advantage; already tracked in prior briefs.
- No new single-network frame-extrapolation result passed the architecture/relevance gate.

### Inference Optimization
- No new TensorRT/INT8/kernel-fusion SR paper surpassed the already tracked NanoVSR and GPU-texture-coded Gaussian compression leads. The wide arXiv scan did not yield a new material addition.

## Archive.org Findings
- Advanced Search remains noisy: `gaussian splatting super-resolution` returned three archived GitHub-list snapshots; `frame generation deep learning` returned 213 mostly unrelated items; `real-time upscaling game` returned 106 unrelated items.
- Targeted known-work queries found no recoverable item for GS-STVSR; GSASR yielded an unrelated 1961 record; GRAPE and STSS are acronym-noisy. Wayback CDX requests for GS-STVSR/GSASR/GRAPE returned HTTP 503 and STSS timed out. **No archival takedown/update was verified; this is an Archive.org availability failure, not negative evidence.**

## GitHub / Open Source
- Fresh REST searches (`pushed:>=2026-07-12`) found no credible new Gaussian-SR repository or DLSS/FSR/XeSS analysis tool. Frame-generation hits were unrelated projects and a zero-star OptiScaler fork.
- **NanoVSR** ([filippawlicki/nanovsr](https://github.com/filippawlicki/nanovsr)) is active (MIT, 27 stars, pushed 2026-07-25) and is now accepted to ECCV 2026; it merits source-level inspection before selecting #9’s control backbone.
- **RDNU** ([Adam-AtlasSoftware/RDNU](https://github.com/Adam-AtlasSoftware/RDNU)) remains the closest open G-buffer-guided game-upscaler lead (GPL-3.0, 15 stars, pushed 2026-07-24). Its GameIR/vKITTI2/Sintel data contract is useful for evaluation, but GPL prevents direct Apache-2.0 reuse.
- **Intel XeSS SDK 3.0.2** was released 2026-07-24: it updates XeLL to 1.3.2.10 and fixes non-Intel-GPU memory leakage, sleep-mode units, and Streamline-proxy compatibility. Opportunity: treat its XeLL/Streamline integration behavior as a DLL-shim compatibility test, not merely marketing.

## News
- **NVIDIA / DLSS (primary verified):** NVIDIA’s July-updated DLSS page states DLSS 4.5 adds Dynamic Multi Frame Generation and a second-generation Transformer SR model; MFG can generate up to five frames per rendered frame. **Threat:** OSS must prove quality/latency/artifact advantages over a dynamic multi-frame, transformer-SR incumbent—not only a simpler architecture.
- **AMD / FSR (primary verified):** AMD’s Q2 2026 GPUOpen result says FSR Upscaling 4.1 now extends prior RDNA 4 exclusivity to RDNA 3. This makes the cross-vendor baseline more available; collect actual quality/latency traces before using it as a teacher or positioning comparator. RSS reports of up-to-8× FSR MFG remain an unverified lead.
- **Intel / Arc (primary verified):** XeSS 3.0.2 is a real SDK release, but it is a XeLL/stability update rather than a new SR/FG model. No further engine-facing XeSS architecture change was verified this run.
- **Apple / Metal and engines:** no primary Apple MetalFX, Unreal, or Unity upscaling update was verified within the 14-day window.

## Action Items / Follow-ups
- [ ] **CRITICAL:** diagnose the 3080 Ti trainer/metrics producer. A 500-step advance since yesterday does not clear the new 5h offline state; watcher/CDN health only proves publication.
- [ ] Deep-read NanoVSR code; benchmark a ≤1M reparameterized-CNN control with exact engine-MV/canvas inputs, TensorRT export, and no custom operators against #9’s RepVGG/ECBSR/RepSR candidates.
- [ ] Combine #13 and #12: collect Nsight L2/DRAM/tile-density traces before selecting packed low-rank layout; evaluate BCn/texture-backed attributes only if random access, gradients, and temporal shimmer survive.
- [ ] Add XeLL/Streamline proxy behavior to eventual DLL-shim compatibility acceptance tests.
- [ ] Watch the AMD 8× MFG rumor, but do not elevate it to competitive fact without AMD release notes/source.
- [ ] Competitive watch: DLSS 4.5 Dynamic MFG and wider FSR 4.1 availability raise the proof burden for OSS’s unified SR/FX story.

## Bibliography Update
- No append: all material paper candidates above are already represented in `BIBLIOGRAPHY.md`; avoiding duplicate entries.
