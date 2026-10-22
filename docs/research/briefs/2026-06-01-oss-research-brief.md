# OSS Research Brief — 2026-06-01

## Project Status Summary
- **Latest commits:**
  - `0660f55` — wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc
  - `56a9a65` — trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)
  - `d1255c8` — trainer: WindowsPath shim on Linux for cross-OS checkpoint resume
  - `e1ab7fe` — hat.py: use flash-attn 2 directly when available, fallback to SDPA
  - `e9b4b65` — trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)
  - `26cd9da` — dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB
  - `83b4f07` — v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast
  - `baf4b02` — worker: apex .com → .org 301 redirect
  - `eb0ec2e` — trainer: bf16 autocast around forward + loss
  - `5c2f2e7` — trainer: add throughput baselines (cudnn.benchmark, TF32, torch.compile)

- **Open issues (10 total):** #14 (v6.2-pico-002 VRAM footprint), #13 (Nsight L2 hit-rate profile), #12 (packed low-rank canvas layout), #10 (decoupled G-buffer encoder ablation), #9 (RepVGG/ECBSR/RepSR ablation), #8 (cross-vendor kernel ports), #7 (FSR 4 distillation pipeline), #6 (v6.2-pico-002 launch), #5 (No-ML shader variant), #4 (HAT-Tiny student distillation ≤1M params)

- **Training dashboard:** v7.0-pico-005 stalled at step 9,100 (stale 42h, trainer degraded). PSNR ~25.70 dB on held-out TartanAir oldtown. Cost so far: $0.23 / 3.79 GPU-hours (3080 Ti).

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **[GS-STVSR]** Shi et al. (Apr 2026). *GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting*. arXiv:2604.18047. Already in BIBLIOGRAPHY. Status: no new updates this week.

- **[RetimeGS]** Wang et al. (Mar 2026). *RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting*. arXiv:2603.13783 — **ACCEPTED CVPR 2026**. Introduces temporal anti-aliasing for 4DGS via optical flow-guided triple-rendering supervision. Why it matters to OSS: **Directly addresses ghosting in OSS-FX α<1 rendering path.** Triple-rendering supervision could strengthen v7 training by penalizing temporal aliasing between discrete frame reconstructions. The ghost-free claim under large motions is exactly the quality gate OSS-FX needs.

- **[SpeeDe3DGS]** CVPR 2026 Poster #433. *Speedy Deformable 3DGS with Temporal Pruning and Motion Grouping*. Temporal pruning and motion grouping analogous to OSS's key-frame active mask — pruning inactive Gaussians per frame to amortize canvas management costs.

- **[1000+ FPS 4DGS]** Yuan et al. (NeurIPS 2025). Already cited in BIBLIOGRAPHY as source of S-T variation score. CVPR 2026 camera-ready confirmed.

- **[Hybrid 3D-4DGS]** arXiv:2505.13215. Adaptive static/dynamic region assignment (3D for static, 4D for dynamic Gaussians). Why it matters: Spatial-temporal decomposition mirrors v7's canvas (spatial) vs. V_xt (temporal) separation philosophy. Could reduce canvas memory by not allocating 4D parameters to static Gaussians.

### Super-Resolution / Upscaling

- **[GameSR]** Rehman et al. (ICLR 2026). *GameSR: Real-Time Super-Resolution for Interactive Gaming*. Lightweight engine-independent SR on encoded game frames — reparameterized conv blocks + PixelUnshuffle + ConvLSTM. Achieves 240 FPS, 30-60% bandwidth reduction in cloud gaming. Reaches near-parity with DLSS/FSR without engine integration. Why it matters to OSS: **Direct competitor validation.** Demonstrates engine-agnostic upscaling market exists. ConvLSTM temporal component is 1/10th the complexity of a Gaussian canvas — raises the question of whether OSS's Gaussian approach is overengineered for the SR-only use case. However, GameSR has no frame extrapolation capability — OSS-FX's unified SR+FG remains unique.

- **[ContinuousSR (Pixel-to-Gaussian)]** Peng et al. (ICLR 2026 Poster). *Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling*. Builds 2D Gaussian field from LR input in single pass; arbitrary-scale rendering in **1ms per scale**, 19.5× speedup across 40 scales. Introduces Deep Gaussian Prior (DGP) and Adaptive Position Drifting. Why it matters: **1ms-per-scale is within OSS's <2ms budget.** DGP-driven covariance weighting could inform v7's Gaussian parameterization. However, single-image only — temporal extension is OSS's contribution to make. Closest work to OSS-Pico tier.

- **[GSASR]** Chen et al. (ICCV 2025). Already in BIBLIOGRAPHY. Feed-forward 2D Gaussian ASR with HAT-L encoder. No updates.

- **[SRSplat]** Hu et al. (Nov 2025). *Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images*. arXiv:2511.12040 — **AAAI 2026 Oral**. Texture-Aware Density Control for adaptive Gaussian allocation. Low direct relevance for game SR but TADC mechanism could inform v7's loss-adaptive density.

- **[OFTSR]** ICLR 2026. *One-Step Flow for Image Super-Resolution with Tunable Fidelity*. One-step flow-based SR with tunable fidelity-realism tradeoff. Distillation angle: flow matching as training objective for single-pass SR could accelerate OSS training.

### Frame Extrapolation / Generation

- **[GFFE]** Wu et al. (2024). *G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering*. arXiv:2406.18551. Already in BIBLIOGRAPHY. No G-buffer, no additional latency, real-time. Disocclusion handling via motion fragment analysis + light shading correction. Why it matters: **Closest non-Gaussian FG competitor.** Achieves extrapolation without G-buffer — if OSS can't beat this quality at lower integration cost, the Gaussian canvas overhead isn't justified for FG alone. Benchmark candidate.

- **[Surface-Aware Feed-Forward Quadratic Gaussian for Frame Interpolation]** NeurIPS 2025 Poster. Quadratic Gaussian parameterization for frame interpolation. Why it matters: Quadratic (vs. linear) Gaussian motion model could improve OSS-FX extrapolation accuracy for curved trajectories.

- **[Consecutive Frame Extrapolation with Predictive Sparse Shading]** ACM 2025. Sparse shading for extrapolated frames — only shades critical areas, reuses pixels in low-variation regions. Parallels OSS's motion-affected-only canvas update philosophy.

### Inference Optimization

- **[INT8-SR]** Nguyen et al. (Apr 2026). *Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training*. arXiv:2604.20291 — MAI Workshop @ CVPR 2026. Why it matters: **Already in BIBLIOGRAPHY.** Deployment-aware INT8 quantization with teacher guidance — recipe for OSS TensorRT FP8 student export. Should be implemented before v6.2 ships.

- **[CUTLASS v4.5.2]** NVIDIA (May 2026). Blackwell-optimized GEMM kernels with blockscaled quantization for FP8 inference. Why it matters: If OSS targets Blackwell RTX GPUs, provides direct path to <2ms FP8 student inference.

## Archive.org Findings
- GS-STVSR (arXiv:2604.18047) is publicly available and not paywalled. No takedown risk.
- GSASR code at `github.com/ChrisDud0257/GSASR` is actively maintained (ICCV 2025, repo live).
- ContinuousSR and GameSR both on OpenReview (ICLR 2026) — open access.
- No archived versions of removed/taken-down papers found this cycle.

## GitHub / Open Source

- **NVIDIA vk_gaussian_splatting** updated — Vulkan testbed now includes 3DGRT + 3DGUT + DLSS-RR integration. Closest reference for "DLSS-on-splats." Worth revisiting as OSS upper-bound benchmark target.

- **Upscayl** continues to dominate open-source image upscaler space (30k+ stars). No new game-specific features.

- **OptiScaler** — open-source modding tool to replace/modify upscalers in PC games. Growing ecosystem of DLSS/FSR/XeSS swap tools. Validates OSS's DLL-shim integration path is technically feasible and has user demand.

- **DLSS Swapper** now updates FSR, XeSS, and DLSS — community tool ecosystem maturing rapidly.

- **Arm Accuracy Super Resolution (ASR)** — Arm's open-source mobile upscaler. Targets mobile but validates open-source upscaler viability.

- **GPUOpen FidelityFX SDK** — FSR 4 now in 85+ games. SDK 2.0 with ML-based Frame Generation and Ray Regeneration (FSR Redstone). MIT-licensed parts still available despite force-push history.

- **Intel XeSS SDK** on GitHub — XeSS 2.0 SDK finally released. XeSS 3 with Multi-Frame Generation now in drivers.

## News

### DLSS 5 — "The GPT Moment for Graphics" (GTC 2026, March 16)
NVIDIA unveiled DLSS 5 at GTC 2026: a **real-time neural rendering model** that infuses pixels with photoreal lighting and materials via generative AI. Launches **Fall 2026** on RTX 50 series. Key details:
- Input: color + motion vectors → AI adds deterministic, temporally stable photoreal lighting
- Handles subsurface scattering, fabric sheen, hair light-material interactions
- End-to-end training on complex scene semantics
- Developer controls via Streamline framework (intensity, color grading, masking)
- Confirmed games: Starfield, Hogwarts Legacy, Assassin's Creed Shadows, Resident Evil Requiem, Elder Scrolls IV Oblivion, Delta Force, + many more
- **Threat level: HIGH but strategic opportunity.** DLSS 5 shifts the battlefield from performance (more frames) to **visual fidelity** (better frames). This is where OSS's Gaussian canvas has theoretical advantage — Gaussians are a superset of what DLSS 5's transformer does. If DLSS 5 is "GPT moment for graphics," OSS's Gaussian canvas is "neural radiance field for real-time" — different architectural bet, potentially higher ceiling.

### DLSS 4.5 (CES 2026, Jan 14)
- Second-generation transformer for Super Resolution (5× more compute, better lighting, finer edges)
- 6× Dynamic Multi Frame Generation (auto-adjusts multiplier)
- 250+ games with DLSS 4 Multi Frame Gen
- Ultra Performance mode now viable for 4K

### AMD FSR Redstone (CES 2026)
- FSR 4 now 85+ games (ML-based upscaling)
- FSR Redstone: ML Frame Generation + Ray Regeneration + Radiance Caching
- SDK 2.0 with UE 5.7 plugin
- **Note:** FSR Redstone was a no-show at GDC 2026 — execution concerns
- Frame pacing issues reported in independent testing

### Intel XeSS 3 (2026)
- Multi-Frame Generation now in drivers (up to 3 generated frames per real frame)
- Available on Arc A-series and B-series GPUs
- XeSS 2.0 SDK released for developers
- Intel Panther Lake (mobile) won Digital Foundry's CES 2026 — XeSS on handhelds

### Apple Metal 4 (WWDC 2025)
- MetalFX Temporal Upscaler: dynamic input resolution, exposure debugger, reactive mask support
- MetalFX Frame Interpolation: generates intermediate frames between rendered frames
- Upscaler + interpolator connection for combined optimization
- Metal 4 Ray Tracing: Intersection Function Buffers (analogous to DX Shader Binding Tables)
- **Threat level: MEDIUM.** Apple building credible upscaling + FG stack. MetalFX Frame Interpolation directly competes with OSS-FX on Apple Silicon. But Apple's walled garden means OSS cross-vendor approach still differentiated.

### Unity 6.5 Updates
- Render Pipelines strategy for 2026 under discussion
- SGSR 1 Mobile upscaling for Unity on Asset Store
- No major upscaling-specific announcements at GDC 2026

### Unreal Engine 5.7
- DLSS 5 integration confirmed
- FSR Redstone UE 5.7 plugin available
- Visual Studio 2026 compatibility issues reported
- No new built-in upscaler from Epic — still relying on third-party plugins

## Action Items / Follow-ups

- [ ] **CRITICAL: Trainer stall investigation.** v7-pico-005 stuck at step 9,100 for 42h. Trainer degraded. Need to investigate WSL2 deadlock, flash-attn compatibility, or checkpoint corruption as root cause.
- [ ] **Benchmark GS-STVSR on TartanAir held-out set.** Closest academic work to OSS — need apples-to-apples comparison to validate v7's Gaussian approach.
- [ ] **Evaluate ContinuousSR's 1ms-per-scale rendering.** If the single-pass Gaussian field construction + 1ms arbitrary-scale rendering can be adapted to game context, it's a candidate for OSS-Pico student tier (replacing or supplementing GRAPE).
- [ ] **Deep-read RetimeGS.** Triple-rendering supervision for temporal anti-aliasing could be adopted directly into v7 training loss.
- [ ] **Add GameSR as benchmark target.** Engine-agnostic upscaler reaching near-parity with DLSS/FSR — validates market but also represents a simpler competitor. Need to measure OSS vs. GameSR on same content.
- [ ] **DLSS 5 threat assessment.** DLSS 5's generative AI approach to visual fidelity shifts competitive landscape. OSS's Gaussian canvas needs to demonstrate unique value beyond what DLSS 5's transformer can do. Prepare differentiation memo.
- [ ] **FSR 4 distillation pipeline (#7) urgency check.** With FSR Redstone launching, the window for FSR 4 as a stable distillation teacher may be closing. Prioritize or re-scope.
- [ ] **Monitor NVIDIA GTC 2026 DLSS 5 developer sessions** for architectural details when recordings publish.
- [ ] **Add RetimeGS to BIBLIOGRAPHY** with temporal anti-aliasing note.
- [ ] **Add ContinuousSR to BIBLIOGRAPHY** with 1ms-per-scale rendering note.
- [ ] **Add GameSR to BIBLIOGRAPHY** as competitor benchmark.
