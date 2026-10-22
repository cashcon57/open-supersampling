# OSS Research Brief — 2026-08-22

## Project Status Summary

- **Latest commits:** dashboard now exposes tailnet-gated training controls/API (`a97a180`, `fcf8084`); recovery work preserved the v7-pico-005 step-5K snapshot; recent trainer work covers WSL `DataLoader` deadlock avoidance, cross-OS checkpoint paths, FlashAttention 2, TF32/prefetch, and corrected live-PSNR math.[unverified]
- **Open issues:** 12 total. Highest architecture/perf-relevant items remain #12 packed low-rank canvas layout, #13 Nsight L2 profiling, #9 reparameterized student backbone, #7 FSR 4 teacher-signal distillation, #8 cross-vendor kernels, and #4 ≤1M student distillation.[12]
- **Training dashboard:** headline says v7 Pico OSS-FX is training at `srcnn-v7.0-pico-005`, step **36,650/100,000**, loss **0.19268**, viz at 36K.[11] But the health tooltip says **Trainer offline, stuck at step 36,650 for 303h**; watcher, CF Worker, and R2 origin are healthy. The unchanged step versus the 2026-08-21 brief makes this a real trainer outage, while the visible “live / updated just now” copy reflects publishing rather than training progress.[11]

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **S²GS: Structured Sparse Gaussian Streaming for Efficient Free-Viewpoint Video Reconstruction on Edge-IoT Devices** — Yiwei Li, Jiannong Cao, Weixun Gao, Rui Cao, Songye Zhu, Yinfeng Cao, Mingjin Zhang; 2026-08-20; [arXiv:2608.19639](https://arxiv.org/abs/2608.19639). Uses octree-organized Gaussian residuals plus temporal sparse-update gating. **Why it matters:** the closest new packed-canvas precedent for v7: benchmark its canonical-state + sparse-residual representation against #12 before committing a low-rank layout. Public MIT repo: [S2GS](https://github.com/liyw420/S2GS).[13][20]
- **4DAnyone: Create Anyone in 4D from a Casual Monocular Video** — Yudong Jin, Tao Xie, Qihang Zhang, Zehong Shen, Zhen Xu, Yujun Shen, Hujun Bao, Xiaowei Zhou, Yinghao Xu; 2026-08-20; [arXiv:2608.20335](https://arxiv.org/abs/2608.20335). Bounds accumulated reference context with Reference Context Packing and routes target context across groups. **Why it matters:** a useful O(1)-budget temporal-context analogue for packed canvas/history state; not a runtime candidate.[17]
- **Compact Feed-Forward 3D Gaussians via Saliency-Guided Primitive Merging** — Tim-Felix Fassch, Jochen Kall, Cyrill Stachniss; 2026-08-11; [arXiv:2608.10712](https://arxiv.org/abs/2608.10712). Post-processes redundant per-pixel Gaussians into saliency-guided variable-size clusters. **Why it matters:** candidate post-training compaction / merge stage for v7 parent-child density; paper claims comparable quality with about 1/20th of the primitive count.[21]
- **HiCo-GS: Hierarchical Context Aggregation and Geometric Consistency for Octree Gaussian Splatting** — Wei Zhang, Shengkai Yu, Shiqiang Gong, Qi Zhang, Qiang Li, Qi Wang; 2026-08-14; [arXiv:2608.14136](https://arxiv.org/abs/2608.14136). Hierarchical anchors exchange parent/self/child context and add alpha-weighted geometric consistency. **Why it matters:** evaluate hierarchy/regularization ideas for parent-child density; code exists but its GitHub license is `NOASSERTION`, so reuse is not cleared.[22][24]
- **QuARC-GS** (2026-08-18, already tracked) remains the strongest immediately actionable prior: canonical frame + quantized residuals plus change-gated densification map directly to #12 temporal deltas and bounded canvas growth.[14]

### Super-Resolution / Upscaling

- **MotionCraft: Latent World Modeling with Sparse Attention for Visual Upscaling** — Rong Fu, Chunlei Meng, Yangchen Zeng, Xiaowen Ma, Yongtai Liu, Wangyu Wu, Shuo Yin, Zijian Zhang, Sicheng Li, Yingrui Ji, Chenhao Wang, Simon Fong; 2026-08-09; [arXiv:2608.08553](https://arxiv.org/abs/2608.08553). Motion-aware latent state and sparse attention target controllable streaming VSR. **Why it matters:** teacher-side reference for temporal-state quality/performance modes, not a ≤1M student architecture.[23]
- No material new game-specific, real-time SR or distillation paper survived relevance filtering.[unverified] Continue treating NanoVSR / Pixel-to-Gaussian as the practical student baselines already in the bibliography.[unverified]

### Frame Extrapolation / Generation

- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** — Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11; [arXiv:2608.10423](https://arxiv.org/abs/2608.10423). Sparse layered image-space cache with predictive shading and adaptive tile lifetimes. **Why it matters:** the highest-priority OSS-FX non-neural competitor/baseline; explicitly models disocclusion without a second neural network. Already tracked; add it to the OSS-FX latency/disocclusion benchmark plan.[16]
- **SNM-VFI** (2026-08-13, already tracked) provides confidence-guided fusion between correspondence-supported and uncertain-region detail. Keep it as a teacher/evaluation reference only; diffusion is outside the shipping budget.

### Inference Optimization

- **3D Gaussian Accelerated Ray Tracing: Fast Training through Particle-Based Backward Propagation** — Laurent Vit, Oliver Batchelor, Richard Green; 2026-08-18; [arXiv:2608.17298](https://arxiv.org/abs/2608.17298). Reorganizes ray-Gaussian backward passes around tile–primitive mappings to avoid atomic-gradient contention. **Why it matters:** direct training-kernel reference for OSS-RR/secondary effects and possibly canvas-gradient accumulation; claims roughly 3–3.5× raw training speedup.[15]
- **RoofGS** (2026-08-16, already tracked) remains the primary #12/#13 measurement template: roofline-stage decomposition, compact attributes, low-width sort keys, approximation, culling, and fusion. Adopt its methodology, not its performance figures, until measured on the OSS 2D rasterizer.

## Archive.org Findings

- Archive.org advanced search found no authoritative paper archive for the exact `GS-STVSR`, `GSASR`, or “real-time upscaling game” queries; broad `GRAPE`/`STSS` queries are dominated by unrelated media and are not usable evidence.[19]
- Wayback CDX confirms six preserved HTML captures for the GS-STVSR arXiv abstract (including 2025-08-08), so the primary abstract has an archival fallback.[unverified] The checked unrelated candidate arXiv page also had captures; no OSS-specific project/repository snapshot appeared in the sampled CDX query.[unverified]

## GitHub / Open Source

- **S2GS** is the only high-signal new public implementation: MIT-licensed and directly relevant to streaming sparse Gaussian residuals.[20]
- **HiCo-GS** is public but license-unasserted; use it for reading/benchmark ideas only until license clarification.[24]
- Recent GitHub keyword hits outside those repos were mostly zero-star course/WIP projects; no credible new open-source DLSS/FSR reverse-engineering or analysis tool surfaced.[unverified]

## News

- **Primary vendor signal:** AMD GPUOpen’s current front page shows its 2026-06 FSR SDK 2.3 update (ML FSR Upscaling 4.1.1 on RX 7000, Frame Generation 4.0.1, Ray Regeneration 1.2), plus 2026-08 posts on DirectX 12 neural texture compression and RDNA 4 WMMA FP8/INT8 GEMM fusion. The latter is actionable for the eventual HIP student kernel path, but is not an FSR feature announcement.[18]
- **RSS-only leads, not confirmed product facts:** multiple 2026-08-19 outlets report Alan Wake 2 FSR 4/FSR Frame Generation support; a 2026-08-21 report alleges FSR 4 Frame Generation pacing problems. Treat both as integration/QA watch items, not architecture evidence.[26]
- No primary NVIDIA DLSS, Intel XeSS, Apple/Metal, Unreal, or Unity announcement materially changing OSS positioning was found in the 14-day scan.[unverified] Intel’s rendered search did expose an Arc graphics driver dated 2026-08-13, but no verified XeSS feature release.[27]

## Action Items / Follow-ups

- [ ] **Ops blocker:** inspect/restart the v7 trainer before interpreting the dashboard ETA; step 36,650 has been unchanged for 303h while the publisher remains healthy.[11]
- [ ] **#12 deep read / experiment:** compare S²GS canonical state + sparse residual gating, QuARC change-gated densification, and current loss-adaptive parent-child allocation with a fixed VRAM/frame-time budget.[13][14]
- [ ] **#13 measurement:** make RoofGS-style per-stage roofline/Nsight counters the acceptance gate for packed layout and raster fusion; separately profile gradients if OSS-RR enters scope.[15]
- [ ] **OSS-FX threat test:** include Amulet in a common disocclusion + multi-frame extrapolation benchmark; test whether a layered visibility cache is a deterministic fallback/companion to the Gaussian canvas.[16]
- [ ] **Competitor watch:** S²GS is a real MIT-licensed temporal Gaussian streaming implementation. Track its code and claims; it overlaps OSS’s compact persistent-state narrative, though it is free-viewpoint-video rather than exact-engine-MV game reconstruction.[13][20]

## Sources

[11] https://opensupersampling.com
[12] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open
[13] https://arxiv.org/abs/2608.19639
[14] https://arxiv.org/abs/2608.18285
[15] https://arxiv.org/abs/2608.17298
[16] https://arxiv.org/abs/2608.10423
[17] https://arxiv.org/abs/2608.20335
[18] https://gpuopen.com
[19] https://archive.org/advancedsearch.php
[20] https://github.com/liyw420/S2GS
[21] https://arxiv.org/abs/2608.10712
[22] https://arxiv.org/abs/2608.14136
[23] https://arxiv.org/abs/2608.08553
[24] https://github.com/WZ-CS/HiCo-GS
[26] https://news.google.com/rss/search?q=FSR+4+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen
[27] https://www.intel.com/content/www/us/en/search.html?ws=text%3DXeSS
