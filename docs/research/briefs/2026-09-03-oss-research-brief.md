# OSS Research Brief — 2026-09-03

## Project Status Summary

- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke` (local `git log --oneline -10` checked).
- **Open issues:** 11. Highest-leverage queue remains low-rank rasterization (#1), counting-sort tile bin (#3), ≤1M HAT-Tiny student distillation (#4), no-ML fallback (#5), and cross-vendor ports (#8). The existing FSR-4-teacher distillation issue is #7.
- **README snapshot:** pre-alpha v7 research teacher; shipping student and native HIP/Metal/Level Zero/Vulkan kernels are not built. README still describes `srcnn-v7.0-pico-005` as training.
- **Training dashboard:** `v7.0 Pico OSS-FX` is still marked active at step **36,650** with total loss **0.19268** (SR Charbonnier **0.02360**, FG Charbonnier **0.07643**, canvas **2,304**).[1] The 2026-09-04T00:02Z health feed says the trainer is **offline/stuck at that step for 101h**; watcher, Worker, and R2 are healthy.[2] No step/loss movement versus the earlier same-day snapshot: treat the dashboard’s “training now” copy as stale and recover/diagnose the trainer before interpreting current v7 quality.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **CC-4DGS: Computational Deformation and Point-Cloud Compression for Storage-Efficient Dynamic Gaussian Splatting** — Kyungdae Park, Chae Eun Rhee; 2026-09-02; arXiv:2609.02184. Compact deformation features plus Gaussian-attribute compression target real-time dynamic splatting. **Why it matters:** direct packed temporal-canvas reference for #12: test its compression ideas against temporal primitive identity, disocclusion quality, and L2 residency rather than treating its deformation model as a replacement for exact engine MVs.[3]
- **Atlas: Algorithm-Hardware Co-Design for On-Device City-Scale 3D Gaussian Splatting in VR** — He Zhu, Zheng Liu, Xingyang Li, Anbang Wu, Zihan Liu, Ruyang Li, Hui Wei, Yaqian Zhao, Jingwen Leng, Minyi Guo, Yu Feng; 2026-09-02; arXiv:2609.02352. Pose/LoD-aware Gaussian streaming and stereo rasterization are co-designed for on-device VR. **Why it matters:** a strong runtime reference for canvas residency/visibility, tile selection, and eventual cross-vendor rasterization; its claimed acceleration is not directly transferable to OSS’s 2D temporal workload.[4]
- **GS-STVSR** remains the most directly aligned prior work rather than a new release: continuous space-time VSR using time-evolving 2D Gaussians. No canonical GitHub repo was found in the exact-name scan. **Why it matters:** validates the time-conditioned Gaussian direction of v7; OSS must differentiate via exact engine MVs, online persistent canvas state, and its student deployment target.[11]

### Super-Resolution / Upscaling

- **PixelIR: Fidelity-Perception Decoupling via Pixel-Space Image-Residual Flow Matching for Efficient One-Step Real-World Super-Resolution** — Bingtian Qiao, Yue Shi, Yong Guo, Wenjun Zhang, Jiezhang Cao; 2026-08-31; arXiv:2608.30782. Separates faithful reconstruction from perceptual residual synthesis, then distills into a one-step coarse-to-fine model. **Why it matters:** use the loss/teacher-target decomposition as a student-distillation ablation; its reported 32.9M deployment model is far above OSS’s ≤1M target.[5]
- No newly material post-cutoff paper was found for explicit practical game VSR or frame interpolation/extrapolation. Newly surfaced diffusion/restoration papers were excluded from the bibliography because they do not fit the real-time ≤1M student path.

### Frame Extrapolation / Generation

- No material new arXiv submission in the scan window. The existing v7 fractional-time renderer remains the differentiator: prioritize proving the α=0.5 gate once the stuck trainer is restored, rather than expanding the model family.

### Inference Optimization

- Atlas is the material new inference signal: investigate temporal/pose-aware LoD and streaming controls for active Gaussian selection before launching native backend work.[4]
- CC-4DGS is the material storage signal: run a #12-oriented compression experiment on the packed canvas before adopting compression schemes designed for offline dynamic 3D scenes.[3]

## Archive.org Findings

- Broad Archive.org queries for `gaussian splatting super-resolution`, `frame generation deep learning`, and `real-time upscaling game` were noisy and produced no material new lead.
- Targeted searches returned no new Archive.org item for GS-STVSR, GSASR, GRAPE, or STSS in this window. GSASR itself remains public official Apache-2.0 code (101 stars; last code push 2025-08-06); its 2026-08-28 GitHub update is metadata, not a code release.[12] An older Wayback capture is available from 2026-06-07.[15]
- GRAPE remains public but small (3 stars), with no declared license; STSS canonical repository remains unresolved. Do not infer absence of code from the repository-name search.

## GitHub / Open Source

- **IE-SRGS** is the only newly pushed direct Gaussian-SR code competitor retained: internal/external knowledge fusion for high-fidelity 3DGS SR; 1 star, last push 2026-08-29, no asserted license. It is a baseline lead, not a proven OSS competitor.[6]
- **OptiScaler** remains the meaningful compatibility reference: cross-vendor DLSS/XeSS/FSR bridge with FG enablement; 10,251 stars, pushed 2026-09-02, GPL-3.0. OSS should study its game-input compatibility constraints, not copy its licensing or generic-wrapper positioning.[7]
- **dlss5-bridge** (MIT, 220 stars, pushed 2026-09-03) claims a ReShade DLSS-5 bridge with Optical Flow fallback. Treat it as integration-market signal only until its feature claims are independently validated.[13]

## News

- **Threat / positioning:** NVIDIA announced DLSS 5 “3D-Guided Neural Rendering” for RTX 50-series/GeForce NOW launch usage, with developer/art-directable lighting/material detail and per-pixel uplift masks. OSS should preserve depth, motion, material/G-buffer inputs and add an optional per-pixel control-mask interface; do not depend on proprietary generative rendering.[8]
- NVIDIA also announced DLSS 4.5 Ray Reconstruction availability for all RTX GPUs and a Gamescom title wave. Keep SR, frame extrapolation, ray reconstruction/denoising, latency controls, and RT tiering separate in the runtime capability model.[9]
- Apple says macOS 27 is its final release supporting Rosetta, reinforcing arm64/macOS CI, universal packaging, and a native Metal backend. This is platform-enablement news, not a MetalFX update.[10]
- No date-qualified primary AMD FSR-4, Intel XeSS/Arc, Unreal, or Unity upscaling announcement was found in the past 14 days. AMD’s current FSR page remains a technical reference (FSR 4.1.1 / UE 5.3–5.8 support), not fresh release news.[14]

## Action Items / Follow-ups

- [ ] **P0 ops:** recover or diagnose `srcnn-v7.0-pico-005`; dashboard state is contradictory and the health feed reports 101h stuck at step 36,650.
- [ ] **#12 ablation:** benchmark CC-4DGS-inspired attribute quantization/compression with temporal-ID preservation, disocclusion quality, VRAM, and Nsight L2 hit rate.
- [ ] **Runtime experiment:** adapt Atlas-style active-set/LoD streaming concepts to 2D temporal Gaussian tiles; measure raster cost and quality before native backend investment.
- [ ] **#4 distillation:** test PixelIR-style fidelity/residual target separation in the ≤1M student recipe; retain conventional teacher-output and FSR-4-teacher controls.
- [ ] **Competitive watch:** track DLSS 5 control-mask semantics and DLSS-bridge integration constraints; OSS’s defensible lane is open, exact-engine-buffer temporal reconstruction—not an unverified generic post-process bridge.

## Sources

[1] https://opensupersampling.com/data.json
[2] https://opensupersampling.com/status.json
[3] https://arxiv.org/abs/2609.02184
[4] https://arxiv.org/abs/2609.02352
[5] https://arxiv.org/abs/2608.30782
[6] https://github.com/ZheXues/IE-SRGS
[7] https://github.com/optiscaler/OptiScaler
[8] https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering
[9] https://www.nvidia.com/en-us/geforce/news/gamescom-2026-dlss-4-5-ray-reconstruction-release-announcements-trailers
[10] https://developer.apple.com/news/?id=w5ngl9k2
[11] https://arxiv.org/abs/2604.18047
[12] https://github.com/ChrisDud0257/GSASR
[13] https://github.com/NIGos/dlss5-bridge
[14] https://gpuopen.com/amd-fsr-upscaling
[15] http://web.archive.org/web/20260607093324/https://github.com/ChrisDud0257/GSASR
