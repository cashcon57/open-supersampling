# Open Reconstruction Suite — Project Goals

**Last updated:** 2026-04-30

## Mission

Ship a vendor-agnostic, open-source, real-time Ray-Retracing stack that beats DLSS 4 Ray Reconstruction in quality and is the only option for ~60% of the GPU market (everything not modern NVIDIA).

## Non-negotiable goals

1. **Real-time on any GPU that supports hardware ray tracing.** Min spec: NV Turing+ / AMD RDNA2+ / Intel Arc Alchemist+ / Apple M3+ / PS5 / Xbox Series S/X. No benefit targeted for non-RT GPUs.

2. **Equal-or-better quality than DLSS 4 RR.** Bálint 2026 (`Mini Adaptive`, 2.6M params, JNDS-shape) hits **24.97 PSNR @ 0.25 spp** vs DLSS 4 RR's 22.88. Open published architecture. Quality leadership at low spp is non-aspirational — it's a published paper.

3. **Real-time perf is mandatory, not optional.** Cannot ship and be "worse than DLSS in most ways." Realistic target: parity with DLSS 4 RR on FP16 hardware (Turing/Ampere/RDNA3); 70-80% of DLSS 4 RR perf on Blackwell FP8 path; outright beat FSR Ray Regen / XeSS RT denoiser on AMD/Intel/Apple where DLSS doesn't run.

4. **Open everything.** Apache-2.0 SDK + shaders, MIT engine plugins, CC-BY-4.0 weights. Reproducible training pipeline. No proprietary blobs in the ship path. No reverse-engineered binary code. No leaked weights. Ever.

5. **Ship with broad install base on day 1.** Drop-in DLL replacement for `nvngx_dlssd.dll` (DLSS Ray Reconstruction) — every game that ships DLSS RR support is an ORS-capable game the day we ship.

## Strategic differentiators (what DLSS structurally cannot match)

1. **Cross-vendor.** DLSS doesn't ship on AMD/Intel/Apple. ORS does. ~60% of the GPU market is ours by default.

2. **Per-game LoRA adapters.** Community-trained low-rank adapters for specific games / mod loads / art styles. DLSS ships one frozen model per release; ORS evolves continuously per-title via the community.

3. **Variable-rate / tile-gated inference.** Skip tiles where reprojected residual < threshold. 30-50% perf savings, structurally incompatible with DLSS's monolithic dense network.

4. **Open weights + LoRA training pipeline.** Community can fine-tune. NVIDIA's training data + per-game profiles are closed; ours are open.

5. **No DRM / NGX / TRT init overhead.** Direct shader dispatch. ~300-800µs/frame saved.

6. **Apple Silicon support.** DLSS doesn't run on Mac. MetalFX underuses simdgroup_matrix. ORS would be the only Ray-Retracing-class denoiser on Apple Silicon.

7. **Modded games.** DLSS hallucinates on Skyrim modded content because its training data lacks those material distributions. Per-game LoRA solves this; closed vendors structurally cannot.

## Versioned milestones

### v0.1.0-mvp (SHIPPED 2026-04-29)

Pure-PyTorch reference implementation on macOS arm64. ORD denoiser (kernel-prediction U-Net + two-branch input), ORU upscaler (3-mode: rgb/rgb_aux/features), paired feature handoff (32-ch FP16 frozen contract), training pipeline (3 trainers with smoke mode), valuation harness. 23/23 tests pass. Tagged `v0.1.0-mvp`.

### v0.2 — UPSCALER drop-in DLL (target ~2-3 months, ~$1500 cloud GPU)

**Strategy: ship upscaler before denoiser.** Bigger install base (every DLSS/FSR/XeSS game ~1000+ titles), simpler API surface, validates drop-in DLL infrastructure before tackling the harder DLSS-RR API. The denoiser ships in v0.3 leveraging validated v0.2 infra.

**Denoising is NOT needed for the upscaler ship.** Pure upscalers (DLSS-SR, FSR, XeSS) consume clean input from the game's existing pipeline (rasterized or pre-denoised RT). The game's own denoiser handles noise BEFORE the upscaler runs. ORS-upscaler is a drop-in for the same contract.

**Hardware tiering (3 NN weight sets + 1 fallback path, same DLL, runtime detection):**

| Tier | Type | Target HW | Quality target | Perf target @ 1080p |
|---|---|---|---|---|
| **Spatial fallback** (inside ORU-Tiny DLL) | G-buffer-aware bicubic + adaptive sharpen, no NN | DX11/Vulkan 1.0/Metal 2.0 baseline (~2012+: Kepler/Maxwell, GCN 1-3, pre-Iris-Xe Intel) | beat FSR 1 / NIS via G-buffer edge awareness | ~0.5-1 ms @ 4K on Polaris-class |
| **ORU-Tiny** (~500K) | NN, FP16 compute | Pascal/Polaris+ (2016+), GTX 10/16, RX 400+, Iris Pro | match FSR 2 spatial+temporal | <2 ms on GTX 1660 |
| **ORU-Lite** (~1M) | NN, FP16 packed | RTX 20+, RDNA 2+, M-series base, Steam Deck | match DLSS-SR Quality | <1.5 ms RTX 4070 |
| **ORU-Standard** (~2.6M) | NN, coop_matrix | RTX 4080+, RX 9070 XT+, M3 Pro+ | beat DLSS-SR Quality | <2.5 ms RTX 4090 |

**Hardware coverage spans the entire gaming GPU market since ~2012.** The spatial fallback is a ~600-line shader (HLSL/GLSL/MSL combined) inside ORU-Tiny's dispatcher — not a separate tier with its own training/QA. Activates automatically when NN can't allocate or hardware can't run FP16 compute.

Marketing claim:
> ORS runs on any GPU with DX11 / Vulkan 1.0 / Metal 2.0 support (~2012+). Real-time NN-quality upscaling on Pascal/Polaris+ (2016+). G-buffer-aware spatial upscaling on older hardware — better than NIS or FSR 1 at similar perf cost. Bilinear fallback at the absolute floor.

**v0.2 deliverables:**
1. PyTorch architecture rebuild: ORU at 500K / 1M / 2.6M tiers (currently 121K, undersized).
2. Real training data from rasterized + RT game traces (~$300 cloud rendering).
3. ONNX export + ONNX Runtime DirectML inference path (Windows). Vulkan compute path for Linux.
4. **Drop-in DLL replacements**: `nvngx_dlss.dll` (DLSS-SR) + `amd_fidelityfx_dx12.dll` / `amd_fidelityfx_vk.dll` (FSR) + `libxess.dll` (XeSS). Three DLLs, one inference engine.
5. Per-game compatibility shim layer (community-maintainable game profiles).
6. First integration test: a popular game with FSR/DLSS-SR support (e.g., Helldivers 2, Starfield, Cyberpunk 2077 raster mode).
7. Wine/Proton compatibility validated for Linux gaming users.

**Out of v0.2 scope (deferred to v0.3):**
- ORD denoiser DLL ship
- DLSS-RR (`nvngx_dlssd.dll`) replacement (joint denoise+upscale, requires both products paired)
- FSR Ray Regen replacement (same reason)
- NRD shader replacement
- Adaptive sampling (Bálint research-only contribution)

### v0.3 — DENOISER ship + cross-vendor inference push
1. Real ORD training: JNDS-shape 2.6M (Bálint Mini Adaptive lineage). Published target: 24.97 PSNR @ 0.25 spp (+2.09 dB over DLSS 4 RR).
2. RAKD-style distillation: 15M Bálint teacher → 1M ORD-Lite student. Target: ~24.5 PSNR @ <3 ms 1080p RTX 4070.
3. **`nvngx_dlssd.dll` drop-in replacement** for DLSS Ray Reconstruction (paired ORD+ORU joint denoise+upscale). Cyberpunk 2077 Path Tracing as canonical test.
4. NRD-shader replacement option for engines using NRD directly (UE5 + custom RT pipelines).
5. Tile-gated variable-rate inference (30-50% perf reduction, the structural lead over DLSS dense).
6. LoRA adapter training pipeline + format spec + community hub bootstrapping.
7. Inline HLSL Cooperative Vectors / SPIR-V coop_matrix shader inference (no ML runtime in hot path).
8. Cross-vendor benchmark suite: RTX 4070 + RX 9070 XT + Arc B580 + Apple M3 — first published numbers in the space.

### v0.4 — Production hardening
1. Per-game compatibility shim layer (community-maintained game profiles).
2. UE5 path-tracer denoiser plugin (`IPathTracingDenoiser`).
3. Godot 4 / bevy / wgpu integrations.
4. Adapter-merging pipeline (DARE / TIES) for community-curated base-model bumps.
5. v1.0 release candidate.

### v1.0 — Public stable release
First open-source vendor-agnostic real-time Ray-Retracing stack with published cross-vendor benchmarks beating DLSS 4 RR on quality and matching/exceeding it on perf for everything that's not Blackwell.

## What we deliberately are NOT

- A frame-generation system (orthogonal — possible future sister project: ORX, Open Ray eXtrapolator, based on GFFE).
- A super-resolution-only system (we own denoising; upscaling is downstream-compatible with any TSR/DLSS-SR/FSR/XeSS).
- A console-targeted product (PS5/Series X DLL drop-in is impossible; consoles via partnership only).
- A research-only artifact (this is a product; the paper is a side effect).

## Honest risk register

1. **DLSS 4.5 (Jan 2026) hints** at folding Ray Reconstruction back into the upscaler. NV may move past a separate Ray Reconstruction pass entirely. Mitigation: ORS positioning is "the open + cross-vendor option" regardless of what NV ships.
2. **AMD FSR Ray Regen** may go cross-vendor someday. Mitigation: FSR-RR is joint-but-FSR-coupled; ORS pairs with itself or anything else.
3. **Sampler distillation is an open problem** (RAKD only validated denoisers). Mitigation: v0.2 ships fixed-budget Bálint-derived denoiser; adaptive sampling research deferred to v0.3+.
4. **Per-game LoRA poisoning** (adversarial weights). Mitigation: signed-contributor + automated quality-regression test suite + tiered curation (community / curated / official).
5. **Drop-in DLL legal exposure.** US DMCA §1201(f) covers interop RE; distribution of derived code is murkier. Mitigation: clean-room implementation only, no decompiled binary code, OptiScaler-style observation-based API matching.
6. **HLSL Cooperative Vectors final spec** (SM 6.10 "Linear Algebra Matrix") may diverge from current preview. Mitigation: SPIR-V coop_matrix (stable since 2023) is primary path.

## What "winning" looks like

- Day 1 (v0.2 ship): ORS DLL drops into Cyberpunk 2077 PT folder, replaces DLSS RR, produces visually-correct output at ~parity perf, +1.6 dB PSNR over DLSS 4 RR. Headline: "Open-source DLSS RR replacement now in Cyberpunk."
- Year 1: shipping in 20+ games via drop-in DLL. Active LoRA community on a HuggingFace-style hub. First published cross-vendor benchmark paper.
- Year 2: UE5 plugin shipped. AMD/Intel/Apple all measurably outperformed by ORS in their own ecosystem (where DLSS doesn't run). v1.0 stable.
- Long-term: the open standard for path-traced reconstruction the way OIDN became the open standard for offline denoising.
