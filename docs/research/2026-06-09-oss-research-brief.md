# OSS Research Brief — 2026-06-09

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
  - (+2 more; 12 total)
- README snapshot: public status still says pre-alpha; active architecture is **v7 / `srcnn-v7.0-pico-005`**, N-D Gaussian canvas with native time axis, HAT-Tiny teacher, parent-child loss-adaptive density, Mip-Splatting anti-aliasing, and alpha=0.5 OSS-FX supervision. Shipping student remains untrained (≤1M-param CNN/FP8 target, <2 ms budget).
- Training dashboard: **notable regression/ops issue** — dashboard reports `srcnn-v7.0-pico-005` at **step 20,950 / 100,000**, loss **0.1087**, but trainer is **offline/stuck for ~22h**. Watcher/Worker/R2 are healthy; no held-out `score_log_v7.json` rows yet, so OSS-FX +1 dB gate is still unmeasured.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **REFINE: Super-efficient 3D Gaussian Splatting Pruning via Rendering-Free Primitive Importance** (arXiv:2606.09074, Chen et al., 2026-06-08) — Proposes a rendering-free, Hessian-like primitive importance proxy for 3DGS pruning with claimed ~3000× pruning-compute reduction versus render-evaluate loops. Why it matters to OSS: strong candidate for v7 canvas pruning/density ablation; could replace or augment parent-child loss-adaptive density when pruning cost becomes nontrivial.
- **MaterialClusterGS: Palette-Based Material Decomposition and Physically-Based Relighting with 2D Gaussian Splatting** (arXiv:2606.09018, Zhang et al., 2026-06-08) — 2DGS material/BRDF palette decomposition for coherent editing and relighting. Why it matters to OSS: not SR/FG, but reinforces 2DGS as a compact editable scene primitive; low priority unless OSS-RR/denoising starts carrying material channels.
- **Latent Spatial Memory for Video World Models** (arXiv:2606.09828, Wang et al., 2026-06-08) — Stores persistent scene memory directly in latent space, avoiding pixel-space encode/render loops and reporting large speed/memory wins. Why it matters to OSS: conceptually aligned with v7 persistent latent/Gaussian canvas; worth a skim for memory-query mechanics even though it targets diffusion world models.

### Super-Resolution / Upscaling
- **LiteVSR: Lightweight Adaptation of Frozen Diffusion Transformers for Video Super-Resolution** (arXiv:2606.09250, Cao et al., 2026-06-08) — Uses a frozen video DiT plus lightweight state-aware adapters for VSR, with single-step sampling compatibility and low trainable-parameter cost. Why it matters to OSS: opportunity for teacher-side adaptation/distillation ideas, but likely too heavy for the ≤1M-param student path.
- **NGram-MoSE: Efficient Remote Sensing Super-Resolution via N-Gram Context and Mixture-of-Experts** (arXiv:2606.08535, Huang/Bui/Chuang, 2026-06-07) — Lightweight transformer SR with N-gram context injection to reduce window-boundary artifacts and sparse MoE FFNs for capacity without proportional inference cost. Why it matters to OSS: directly relevant to the historical HAT/window artifact problem and possible Pico/Standard student efficiency experiments.
- **TUDSR: Twice Upsampling-Diffusion for Higher Super-Resolution** (arXiv:2606.09608, Wu/Dong/Wei, 2026-06-08) — Two-stage/looped chunk-based diffusion SR for very high output resolutions. Why it matters to OSS: mainly offline/generative SR; useful as a quality teacher/data prior, not a runtime architecture.
- **Frequency Decoupled Framework for Screen Content Image Super-Resolution** (arXiv:2606.09029, Wang et al., 2026-06-08) — Amplitude/phase-factorized implicit SR for screen content. Why it matters to OSS: low direct relevance to photoreal games, but phase-consistency ideas may help HUD/UI paths if OSS handles full frame output instead of render-only buffers.

### Frame Extrapolation / Generation
- **Ultra Flash: Scaling Real-Time Streaming Video Generation to High Resolutions** (arXiv:2606.09150, Luxury et al., 2026-06-08) — Cascaded streaming high-res video generation with causal latent upsampler, single-step distillation, sparse causalization, and dynamic cache management, claiming ~30 FPS at 1K and ~18 FPS at 2K on one GPU. Why it matters to OSS: competitor-adjacent signal that cache-aware temporal SR/generation is accelerating; techniques may transfer to teacher distillation/cache scheduling, but it is not a game-engine motion-vector method.
- **Echo-Memory: A Controlled Study of Memory in Action World Models** (arXiv:2606.09803, King et al., 2026-06-08) — Controlled comparison of memory designs for action-conditioned video models. Why it matters to OSS: useful framing for evaluating persistent canvas memory; suggests replay metrics alone are insufficient for temporal consistency.
- **MemoryVLA++** / **AHA-WAM** (arXiv:2606.09827 / 2606.09811, 2026-06-08) — Robotics/world-action temporal-memory papers surfaced by the temporal-video query. Why it matters to OSS: mostly out-of-domain; only worth noting for memory/context-routing patterns.

### Inference Optimization
- **REFINE** (above) is the strongest optimization-relevant hit this run: render-free GS pruning could reduce training/eval overhead and inform runtime active-set selection.
- **Ultra Flash** (above) includes single-step distillation + dynamic cache management; relevant as a teacher-side distillation/caching pattern, not a direct TensorRT/kernel-fusion result.
- No new arXiv hit from the queried terms looked like a direct TensorRT/INT8/FP8 SR deployment paper this run.

## Archive.org Findings
- Archive.org API search returned mostly low-signal/general archive items for `"gaussian splatting" AND "super-resolution"` (including a broad archived GitHub-star list) rather than paper-specific captures.
- Direct Wayback CDX follow-up for **GS-STVSR**, **GSASR**, **GRAPE**, and **3DGS³** was attempted, but `web.archive.org/cdx` timed out / returned 503 from this environment. No verified archived paper/project captures this run.
- Follow-up: retry CDX with smaller per-URL calls or archive the live URLs directly when Wayback is healthy; full-text Archive.org search is too noisy for acronym-heavy paper names.

## GitHub / Open Source
- **Adam-AtlasSoftware/RDNU — Radeon Decoupled Neural Upscaler** (`https://github.com/Adam-AtlasSoftware/RDNU`, pushed 2026-06-09, 14 stars) — RDNA3/RX 7900 XTX-focused recurrent spatial-temporal neural upscaler using an 8-frame window plus engine G-buffers (depth, normals, albedo/BRDF, high-precision optical flow), fractional scaling + RCAS, FP16/FP8 tensors, and RDNA3 WMMA/Wave32/LDS optimization notes. README says it is built on RDG/BasicSR training configs and targets DirectML/AMD deployment. **Threat/opportunity:** very close to OSS positioning on open neural game upscaling + G-buffer guidance, but vendor-specific and pixel/recurrent rather than Gaussian-canvas/unified SR+FG; deep read license/runnability/weights immediately.
- **hvcl/3DCT_SR_DDNM_NAB-GS** (`https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS`, pushed 2026-05-28) — medical CT SR using diffusion projection priors and negative-alpha-blending GS. Low direct game relevance, but another GS+SR implementation to mine for raster/density details.
- **nova-studios-hw/novagpu-ts1t** (`https://github.com/nova-studios-hw/novagpu-ts1t`, pushed 2026-06-06) — open GPU RTL repo mentioning hardware ray tracing/frame generation. Mostly hardware/RTL, not a software upscaler competitor.
- Developer blog checks via direct pages/RSS found no obvious fresh NVIDIA/AMD/Intel upscaler posts in the scraped titles; Unreal feed was 403 from this environment, Unity feed fetched but yielded no relevant upscaling title in the inspected slice.

## News
- General web_search backend failed for this run (`Firecrawl search failed: Payment Required`), so news coverage is based on direct vendor/blog page probes plus GitHub/arXiv.
- Google News RSS direct queries surfaced several relevant-but-unverified items: NVIDIA **DLSS 4.5 Ray Reconstruction** coverage dated 2026-05-31/06-01; AMD **FSR 4.1** RDNA 3.5 iGPU support dispute dated 2026-06-04/05; Intel **XeSS 3 Multi Frame Generation** in Death Stranding 2 coverage dated 2026-05-31; and Apple WWDC26 developer-tool/Metal-adjacent coverage dated 2026-06-09. Treat as watchlist until primary vendor pages are checked.
- Confirmed operationally important competitor signal is GitHub-side: **RDNU** is the main new open-source item to watch.

## Action Items / Follow-ups
- [ ] **Deep read RDNU**: architecture, license, model size, datasets, runtime path, G-buffer contract, RDNA3-specific assumptions, and whether it has runnable weights.
- [ ] **Deep read REFINE** and test whether its rendering-free primitive-importance proxy can become a v7 active-set/pruning ablation.
- [ ] **Skim NGram-MoSE** for N-gram context injection as a fix for HAT/window/tile boundary artifacts and sparse-MoE student capacity.
- [ ] **Skim LiteVSR / Ultra Flash** for single-step distillation and temporal cache-management patterns that could improve teacher-to-student training.
- [ ] **Fix/inspect trainer stall**: dashboard stuck at step 20,950 for ~10h while publishing stack is healthy; no v7 score rows means architecture gating is blocked.
- [ ] Add Wayback CDX checks for known prior-art URLs (GS-STVSR, GSASR, GRAPE, STSS) instead of broad archive.org search.
