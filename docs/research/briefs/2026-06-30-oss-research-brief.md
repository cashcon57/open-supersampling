# OSS Research Brief — 2026-06-30

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
- Open issues (first 10 from GitHub API):
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
- README snapshot: still pre-alpha. v7 / `srcnn-v7.0-pico-005` is the active research-teacher architecture: N-D Gaussian canvas with native time axis, HAT-Tiny teacher, parent-child loss-adaptive density, Mip-Splatting filters, and alpha-time OSS-FX supervision. The shipping student, cross-vendor kernels, and DLL shim remain unshipped.
- Training dashboard: rendered `opensupersampling.com` redirects to `.org`. Visible state: **v7.0 Pico OSS-FX** at **30,300 / 100,000** steps, loss **0.18269**, viz @ step **26,000**, no held-out scores, GPU stats unavailable (`host offline or rsync failed`). Dashboard remains contradictory: header says `updated just now` / `live`, but service tooltip says **Trainer — offline, stuck at step 30300 for 69h**. Watcher healthy (published 0s ago), CF Worker healthy (HTTP 200 in 118ms), R2 origin healthy. Compared with 2026-06-29: step/loss/viz did not advance; stuck duration increased from 45h to 69h.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Resonant Brane Splatting for Arbitrary-Scale Super-Resolution** (arXiv:2606.29453v1; Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto; 2026-06-28) — Replaces flat Gaussians with Gaussian-Hermite “Brane” primitives that emit spatially varying colors, so a single primitive can carry edge/detail structure with less overlap. Why it matters to OSS: direct opportunity for v7 detail recovery and raster budget reduction; the primitive is richer than OSS’s current smooth Gaussian canvas and could reduce splat count for high-frequency edges.
- **Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution** (arXiv:2606.29400v1; Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto; 2026-06-28) — Feed-forward arbitrary-scale SR predicts where to allocate/densify 2D Gaussian primitives via a neural routing architecture and hierarchical pointer convolution. Why it matters to OSS: direct ablation target for parent-child/loss-adaptive density; strong fit for foveated/continuous-scale game upscaling and packed-canvas budgeting.
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047v1; Mingyu Shi, Xin Di, Long Peng, Boxiang Cao et al.; 2026-04-20) — Continuous spatio-temporal VSR with 2D Gaussian splatting, flow-guided motion, covariance-resampling alignment, and near-constant temporal-scale inference. Why it matters to OSS: already tracked, but remains the closest public math precedent for OSS’s covariance-resampled temporal canvas and should be re-read against v7’s native time-axis implementation.
- **Temporally Aware Densification for Dynamic 3D Gaussian Splatting** (arXiv:2606.23212v1; Vikram Sandu, Mayurdeep Pathak, Rajiv Soundararajan; 2026-06-22) — Visibility/lifespan-aware densification for dynamic 3DGS. Why it matters to OSS: already tracked; compare against v7 parent-child/loss-adaptive density when designing density thresholds.

### Super-Resolution / Upscaling
- **Decoder-Free Distillation for Quantized Image Restoration** (arXiv:2603.09624v1; S. M. A. Sharif, Abdur Rehman, Seongwan Kim, Jaeho Lee; 2026-03-10) — Quantization-aware distilled restoration uses decoder-free bottleneck distillation, learnable loss reweighting, and a lightweight degradation gate; reports INT8 restoration retaining ~96.5% of FP32 performance and 442 FPS on Jetson Orin. Why it matters to OSS: good fit for ≤1M-param student distillation because it addresses teacher/student capacity mismatch and quantization noise in low-level vision, not just classification.
- **Fidelity- and Perception-Aware Local Implicit Attention for Arbitrary-Scale Image Super-Resolution** (arXiv:2606.21910v1; Yu-Syuan Xu et al.; 2026-06-20) — Diffusion-backed ASR balances fidelity features with perceptual realism. Why it matters to OSS: already tracked; mostly teacher/pseudo-target inspiration, not a runtime path.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483v1; Antoine Mercier et al.; 2023-08-03) — Reappeared in motion-vector SR search. Why it matters to OSS: still the most directly game-specific baseline/dataset reference; no new version found today.

### Frame Extrapolation / Generation
- **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (arXiv:2406.18551v1; Songyin Wu, Deepak Vembar, Anton Sochenov, Selvakumar Panneer, Sungye Kim, Anton Kaplanyan, Ling-Qi Yan; 2024-05-23) — Real-time frame extrapolation without G-buffer inputs, combining heuristic disocclusion handling with a lightweight shading-correction network. Why it matters to OSS: already tracked but competitively important; it targets easier integration than OSS’s exact-motion-vector path, while OSS should win on engine-informed motion/canvas reuse if wired correctly.
- **UniRED: Unified RGB-D Video Frame Interpolation with Event Guidance** (arXiv:2606.24282v1; Yinuo Zhang et al.; 2026-06-23) — RGB-D interpolation augmented with event guidance for geometric consistency. Why it matters to OSS: conditioning analogy for depth/motion/material channels, but not game-runtime practical and already tracked.

### Inference Optimization
- **Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation** (arXiv:2606.23743v2; Yitong Li et al.; 2026-06-21) — Full-stack inference tuning across model, hardware, numerical format, cache, sparse attention, and kernels. Why it matters to OSS: reinforces that student+raster deployment needs co-designed scheduling/kernels, not only model pruning; already tracked.
- **SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference** (arXiv:2606.26587v1; Haoqian Meng et al.; 2026-06-25) — Training-free sparse+dense FP4 activation decomposition. Why it matters to OSS: LLM domain, but useful mental model for outlier-aware FP4/FP8 kernels if the student hits activation-scale pathologies; already tracked.
- **Neural Texture Compression using Hypernetworks** (arXiv:2606.26913v1; Laurent Belcour; 2026-06-25) — Hypernetwork emits per-material latent features and decoder weights for real-time neural texture decoding. Why it matters to OSS: possible material/detail-head reference for G-buffer-conditioned student recovery; already tracked.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"` — 3 hits, but top items are generic archived GitHub/list snapshots, not primary paper mirrors. No useful taken-down paper copy identified.
- `"frame generation" "deep learning"` — 4 hits including `FluidFrames` old builds and `DLSS Swapper` builds; useful for tool archaeology only, not OSS architecture.
- `"real-time upscaling" game` — 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS` — 16,596 noisy hits; acronym pollution dominated by unrelated “grape” media/items, so this is not evidence of GS-STVSR/GSASR/GRAPE/STSS archival coverage.
- Known-paper Wayback checks were limited this run; arXiv primary pages for GS-STVSR / QuADA-GS / RBS remain available, so no archive-only replacement needed.

## GitHub / Open Source
- `pranav-asthana/SplatSuRe` — CVPR 2026 selective super-resolution for multi-view consistent 3DGS; 82 stars, updated 2026-06-16, license reported `NOASSERTION`. Relevance: monitor for selective/high-frequency injection code patterns that could transfer to OSS canvas detail recovery.
- `Azusa309/MedGSSR` — ECCV 2026 medical-image SR 3D reconstruction via hierarchical feed-forward Gaussian splatting; 2 stars, MIT, updated 2026-06-22. Relevance: low priority due medical domain, but hierarchical feed-forward GS may be useful for density scheduling.
- `ShrekDino/FreeGen` — claims open-source Vulkan / FSR 3.1 upscaling and frame generation; 0 stars, Apache-2.0, updated 2026-06-16. Relevance: likely immature, but watch as a possible open-source FG wrapper/competitor signal.
- `AdvaitPalve11/Jojo-Player` and `gokusayayin641-ai/neural-upscaler-x4-page` — surfaced by broad searches but appear app/demo-level, not game upscaler research.
- No fresh DLSS/FSR analysis-tool repo appeared in GitHub REST search for the last 14 days.

## News
- **FSR 4 / 4.1 momentum** — Google News RSS showed multiple 2026-06-22 headlines claiming FSR 4.1 / INT8 model support expansion to RX 7000/RDNA3 and Steam Machine testing. Treat as headline-level leads, but competitive implication is clear: AMD is pushing quantized upscaling down-stack, overlapping OSS’s ≤1M-param INT8/FP8 student story.
- **DLSS 4 / 4.5 rollout** — RSS/vendor headlines show NVIDIA adding DLSS 4.5 to more games and a Zenless Zone Zero update with DLSS 4.5/ray tracing. Threat: NVIDIA keeps shipping visible game integrations while OSS is still pre-alpha; OSS positioning should emphasize vendor-neutral + one architecture for SR+FX, not raw installed base.
- **Linux/NVK DLSS experimentation** — RSS surfaced an article claiming experimental DLSS support in NVK via imported CUDA binaries. If verified, this weakens “DLSS is Windows/NVIDIA-stack only” talking points but strengthens the broader demand for upscaler interop layers.
- **Intel/XeSS / handhelds** — RSS was mostly Arc G3 handheld coverage and budget-gaming positioning, not major XeSS algorithm news. Keep Arc/mobile budgets in the Standard/Pico target matrix.
- **Apple / Metal** — RSS surfaced Game Porting Toolkit 4 and broad Metal gaming-performance commentary; no concrete MetalFX SR algorithm update found. Apple path remains a backend/deployment concern rather than an immediate architecture threat.
- **Game engine integration** — RSS surfaced AMD FSR SDK 2.3 / FSR 4.1 compatibility headlines and Halo PC requirements; no decisive Unreal/Unity upscaling API change found.

## Action Items / Follow-ups
- [ ] Deep-read **Resonant Brane Splatting** and decide whether higher-order color modes can be represented as extra per-Gaussian channels in v7 without blowing the packed-canvas budget.
- [ ] Deep-read **QuADA-GS** and compare Neural Routing / Hierarchical Pointer Convolution against OSS parent-child loss-adaptive density and issue #12 packed low-rank layout.
- [ ] Add an ablation note for **decoder-free distillation / QDR** in the student-distillation plan: bottleneck-only KD + learnable magnitude reweighting may be directly relevant to INT8/FP8 student stability.
- [ ] Ops: trainer is still stuck/offline at step 30,300 (69h) while watcher/origin are healthy; decide whether to restart/resume or preserve current checkpoint before more dashboard-only work.
- [ ] Monitor FSR 4.1 INT8/RDNA3 claims via primary AMD/GPUOpen release notes before using them in external positioning.
