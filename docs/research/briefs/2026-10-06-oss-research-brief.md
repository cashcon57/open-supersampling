# OSS Research Brief — 2026-10-06

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (only commit returned by local `git log --oneline -10`).
- **Open issues (first 10):** #14 D3 VRAM footprint; #13 H004 Nsight L2 profile; #12 H004 packed low-rank canvas; #10 v6.2.1 G-buffer encoder ablation; #9 student reparameterization ablation; #8 cross-vendor ports; #7 distillation pipeline; #6 v6.2-pico-002 launch; #5 No-ML shader variant; #4 HAT-Tiny student distillation.
- **README snapshot:** v7 N-D Gaussian canvas with native time axis is the research/teacher architecture and dashboard run; no ≤1M-parameter student trained. README foreground says v7 Pico-005 training. v6.2 checkpoint retained as baseline; cross-vendor kernels and DLL integration are designed, not built. README's architecture context is older than the v7-named dashboard state—keep dashboard and README claims distinct.
- **Training dashboard (rendered/extracted 2026-10-06):** active row `srcnn-v7.0-pico-005`, step 36,650/100,000, loss 0.19268, headline “live,” ETA ~1.1d. Health panel says **Trainer — offline**, “stuck at step 36650 for 402h”; GPU stats unavailable / host offline or rsync failed. Watcher healthy (published 1s ago), CF Worker healthy (HTTP 200/107ms), R2 origin healthy (HTTP 200). The conflicting live/headline and trainer-health states are an operational discrepancy; the step has no proven recent progress from the page alone.

## New Papers (arXiv)
These are primary arXiv metadata/abstract-level summaries; not full-paper or independent benchmark reviews. Several are already in `BIBLIOGRAPHY.md`; no duplicate entries appended.

### Gaussian Splatting / Temporal
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** — Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha (2026-04-20; [arXiv:2604.18047](https://arxiv.org/abs/2604.18047)). Continuous-time 2D Gaussian video SR with flow-guided motion and covariance resampling; direct precedent for OSS's time/covariance representation, with game-rendered motion and extrapolation still distinct. Already bibliographed.
- **Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution (GSASR)** — Du Chen, Liyi Chen, Zhengqiang Zhang, Lei Zhang (2025-01-12; [arXiv:2501.06838](https://arxiv.org/abs/2501.06838)). Image-conditioned Gaussian prediction plus scale-aware differentiable CUDA rasterization supports unseen magnifications; useful scale-conditioning/rasterizer reference, though not a temporal game solution. Already represented in local literature.
- **Adaptive Densification for High-Fidelity and Efficient Sparse Gaussian Splatting in Arbitrary-Scale Super-Resolution (QuADA-GS)** — Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto (2026-06-28; [arXiv:2606.29400](https://arxiv.org/abs/2606.29400), v2 revised 2026-09-08). LR-conditioned allocation concentrates sparse Gaussians in complex regions; opportunity for v7 adaptive spawning and canvas-memory experiments. Already bibliographed.
- **Resonant Brane Splatting for Arbitrary-Scale Super-Resolution (RBS)** — Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto (2026-06-28; [arXiv:2606.29453](https://arxiv.org/abs/2606.29453), v2 revised 2026-07-01). Gaussian-Hermite “Branes” model spatially varying color/high-frequency detail with a differentiable rasterizer; possible quality/primitive-complexity idea, but assess overhead against OSS's tight latency budget. Already bibliographed.

### Super-Resolution / Upscaling
- **Stream-DiffVSR: Low-Latency Streamable Video Super-Resolution via Auto-Regressive Diffusion** — Hau-Shiang Shiu, Chin-Yang Lin, Zhixiang Wang, Chi-Wei Hsiao, Po-Fan Yu, Yu-Chih Chen, Yu-Lun Liu (2025-12-29; [arXiv:2512.23709](https://arxiv.org/abs/2512.23709)). Causal online VSR uses four-step distilled diffusion and motion-aligned temporal guidance; abstract reports 720p in 0.328s on RTX 4090, far outside OSS's <2ms target but a quality/streaming reference.
- **Neural Super-Resolution for Real-time Rendering with Radiance Demodulation** — Jia Li, Ziling Chen, Xiaolong Wu, Lu Wang, Beibei Wang, Lei Zhang (2023-08-13; [arXiv:2308.06699](https://arxiv.org/abs/2308.06699)). Separates lighting/material, adds occlusion-aware warping and recurrent temporal processing; relevant game-rendering baseline for disocclusion and temporal stability.
- **Efficient neural supersampling on a novel gaming dataset** — Antoine Mercier, Ruan Erasmus, Yashesh Savani, Manik Dhingra, Fatih Porikli, Guillaume Berger (2023-08-03; [arXiv:2308.01483](https://arxiv.org/abs/2308.01483)). QRISP supplies game imagery with motion/depth modalities; potential dataset/evaluation lead for engine-input training.

### Frame Extrapolation / Generation
- **Low-latency Space-time Supersampling for Real-time Rendering** — Ruian He, Shili Zhou, Yuqi Sun, Ri Cheng, Weimin Tan, Bo Yan (2023-12-18; [arXiv:2312.10890](https://arxiv.org/abs/2312.10890)). Unifies frame supersampling/extrapolation using engine G-buffer and motion vectors; abstract reports 4ms versus 17ms two-stage baseline. Direct conceptual comparator, though OSS uses a shared canvas/time query.

### Inference Optimization
- **New material lead:** Stream-DiffVSR's distilled few-step design is not a plausible <2ms shipping student by its reported latency, but can inform teacher-side temporal supervision. Existing bibliography already covers the more directly actionable compact/reparameterized student leads; no additional verified inference paper merits a new entry from this scan.

## Archive.org Findings
- Direct Wayback CDX checks for arXiv abstracts (including GS-STVSR, GSASR, Stream-DiffVSR and known related IDs) failed TLS certificate verification in this environment. No capture/no-capture conclusion is possible.
- Archive.org keyword searches for GS-STVSR, GSASR, and GRAPE were noisy/unrelated; no verified archived paper artifact found. This is **unresolved**, not evidence of takedown or absence.
- Prior reports in the repository also note acronym pollution for GRAPE/STSS; no new verified archived copy surfaced.

## GitHub / Open Source
- **Opportunity / competitive watch — Arm Neural Graphics SDK for Game Engines** ([SDK](https://github.com/arm/neural-graphics-sdk-for-game-engines), [UE plugin](https://github.com/arm/neural-graphics-for-unreal)). Arm's Sep 23 primary announcement describes MIT-licensed neural super-sampling plus Neural Frame Rate Upscaling (NFRU), Vulkan-backed, with UE 5.4/5.6 support. An open engine-level SR+FG stack is relevant to OSS portability/adoption; Arm performance figures are vendor claims, not independent benchmarks. Source: [Arm announcement](https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/introducing-neural-frame-rate-upscaling-in-arm-neural-graphics-sdk-for-game-engines-and-ue-plugins).
- NVIDIA's [Vulkan Gaussian Splatting sample](https://github.com/nvpro-samples/vk_gaussian_splatting) release 2026.2.9 (Sep 3) includes faster ray tracing and transparency improvements; useful rasterizer reference, but outside the 7–14-day window and not a game-upscaler competitor.
- No newly active, directly relevant Gaussian-SR or general open-source neural-upscaler repository was sufficiently verified in the checked results to claim as a fresh release.

## News
- **NVIDIA DLSS 4.5 (vendor-confirmed, Sep 29–Oct 1):** NVIDIA's [GeForce news](https://www.nvidia.com/en-us/geforce/news/) lists new game integrations including Dynamic Multi Frame Generation, Super Resolution and ray-tracing features. Competitive signal: multi-frame generation is becoming a current shipped baseline; no independent benchmark collected.
- **Arm NFRU (Sep 23; vendor primary source):** adds open neural intermediate-frame generation alongside NSS; relevant to cross-vendor/engine integration, but not direct proof of desktop-GPU parity.
- **Apple MetalFX:** current [Metal what's new](https://developer.apple.com/metal/whats-new/) documentation describes redesigned temporal upscaling using Neural Engine/Neural Accelerators on M5 Pro/Max, motion vectors, distortion fields, and dynamic-resolution subrectangles. No dated release in fetched page; platform-specific.
- **AMD FSR SDK:** [GPUOpen](https://gpuopen.com/amd-fsr-sdk/) lists SDK 2.3 with FSR Upscaling 4.1.1, FG 4.0.1, Ray Regeneration 1.2.0; June 2026, outside this window. FSR 4.1.1's RX 7000/RDNA 3 support is a material competitive compatibility expansion, subject to vendor rollout conditions.
- **Intel XeSS:** Intel's Aug 10 newsroom post names 2026 game launch partners for XeSS 3 SR/MFG/low-latency; outside the window. No corroborated Oct 2–6 XeSS/FSR/Apple announcement found in sources checked.

## Action Items / Follow-ups
- [ ] Verify the v7 trainer process/metrics source on the training host; reconcile dashboard's “live” label with offline/stuck-at-36,650-for-402h health state before treating ETA as meaningful.
- [ ] Run a targeted QuADA-GS adaptive-density ablation against v7 fixed/current spawning; record quality, memory, sparse communication and end-to-end latency.
- [ ] Track Arm SDK/plugin code, license/dependency details, tested devices and engine integration maturity; compare the input contract with OSS's motion/depth/canvas path.
- [ ] Deep-read GS-STVSR and the space-time supersampling baseline specifically for distinctions in continuous-time parameterization, motion reliability and game-input assumptions; maintain prior evaluations rather than treating literature similarity as a result.
- [ ] Re-run Archive.org CDX with a functioning TLS trust store and inspect candidate Intel/AMD/NVIDIA primary release pages for any post-window updates.
