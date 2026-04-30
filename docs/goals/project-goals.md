# Open Reconstruction Suite — Project Goals

**Last updated:** 2026-04-30

## Mission

Ship a vendor-agnostic, open-source, real-time ray-tracing reconstruction stack that beats DLSS 4 Ray Reconstruction in quality and is the only option for ~60% of the GPU market (everything not modern NVIDIA).

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

6. **Apple Silicon support.** DLSS doesn't run on Mac. MetalFX underuses simdgroup_matrix. ORS would be the only RR-class denoiser on Apple Silicon.

7. **Modded games.** DLSS hallucinates on Skyrim modded content because its training data lacks those material distributions. Per-game LoRA solves this; closed vendors structurally cannot.

## Versioned milestones

### v0.1.0-mvp (SHIPPED 2026-04-29)
Pure-PyTorch reference implementation on macOS arm64. ORD denoiser (kernel-prediction U-Net + two-branch input), ORU upscaler (3-mode: rgb/rgb_aux/features), paired feature handoff (32-ch FP16 frozen contract), training pipeline (3 trainers with smoke mode), valuation harness. 23/23 tests pass. Tagged `v0.1.0-mvp`.

### v0.2 — Drop-in DLL (target ~2-3 months, ~$1500 cloud GPU)
1. Architecture upgrade: 141K-param ORD → JNDS-shape 2.6M params (Bálint Mini Adaptive lineage). Quality target: ~24-25 PSNR @ 0.25 spp.
2. Real Bistro training data via Mitsuba 3 cloud rendering.
3. ONNX export + ONNX Runtime DirectML inference path.
4. **`nvngx_dlssd.dll` drop-in replacement** for DLSS Ray Reconstruction. Implements NGX RR API surface from open Streamline spec. OptiScaler is the engineering reference.
5. Cyberpunk 2077 Path Tracing as canonical first integration test.
6. Corkscrew integration (bundle ORS for Wine/CrossOver users).

### v0.3 — Quality + perf push
1. RAKD-style distillation: 15M Bálint teacher → ~1M student. Target: ~24.5 PSNR @ <3ms 1080p RTX 4070.
2. Tile-gated variable-rate inference (30-50% perf reduction).
3. LoRA adapter training pipeline + format spec.
4. Inline HLSL Cooperative Vectors / SPIR-V coop_matrix shader inference (no ML runtime in hot path).
5. Cross-vendor benchmark suite: RTX 4070 + RX 9070 XT + Arc B580 + Apple M3 — first published numbers in the space.

### v0.4 — Production hardening
1. Per-game compatibility shim layer (community-maintained game profiles).
2. UE5 path-tracer denoiser plugin (`IPathTracingDenoiser`).
3. Godot 4 / bevy / wgpu integrations.
4. Adapter-merging pipeline (DARE / TIES) for community-curated base-model bumps.
5. v1.0 release candidate.

### v1.0 — Public stable release
First open-source vendor-agnostic real-time ray-tracing reconstruction stack with published cross-vendor benchmarks beating DLSS 4 RR on quality and matching/exceeding it on perf for everything that's not Blackwell.

## What we deliberately are NOT

- A frame-generation system (orthogonal — possible future sister project: ORX, Open Ray eXtrapolator, based on GFFE).
- A super-resolution-only system (we own denoising; upscaling is downstream-compatible with any TSR/DLSS-SR/FSR/XeSS).
- A console-targeted product (PS5/Series X DLL drop-in is impossible; consoles via partnership only).
- A research-only artifact (this is a product; the paper is a side effect).

## Honest risk register

1. **DLSS 4.5 (Jan 2026) hints** at folding RR back into the upscaler. NV may move past separate RR pass entirely. Mitigation: ORS positioning is "the open + cross-vendor option" regardless of what NV ships.
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
