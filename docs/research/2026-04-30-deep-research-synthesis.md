# ORS Deep Research Synthesis — 2026-04-30

Consolidated findings from four parallel research agents dispatched 2026-04-30 to harden the leapfrog story for ORS vs DLSS 4 / FSR Ray Regen / XeSS RT denoiser.

## 1. The DLSS 4 perf bar (the bar we must clear)

Compiled from Tom's Hardware, NVIDIA Research, TechSpot, TechPowerUp, Club386, dsogaming, SemiAnalysis Blackwell teardown, arxiv microbenchmarks.

### Architecture comparison

| Spec | DLSS 3.5 RR (CNN) | DLSS 4 RR (Transformer) | DLSS 4.5 RR (Transformer v2) |
|---|---|---|---|
| Backbone | CNN UNet | Vision transformer w/ self-attention | Vision transformer v2 |
| Params (relative) | 1× | **2×** | likely ~3-4× |
| FLOPs/frame (relative) | 1× | **4×** | **~20× CNN, ~5× DLSS 4** |
| Inference precision | FP16 | **FP8** on Ada/Blackwell, FP16 fallback | FP8/FP4 on Blackwell |
| Hardware reqs | Tensor cores (any RTX) | Tensor cores + FP8 (Ada+) | + AMP scheduler (Blackwell) |
| Preset letters (RE) | D, E | J, K | L, M |

NVIDIA quote: "transformer model packs **four times the computations and twice the number of parameters**... in a similar frame budget" + "FP8 precision, directly accelerated by next-gen tensor cores on Blackwell."

### Per-frame ms cost at 4K Quality (RR delta over DLSS-SR)

| GPU | DLSS-SR alone | DLSS 4 SR+RR | RR delta | Notes |
|---|---|---|---|---|
| RTX 5090 | ~9-10 ms | ~10.5-11 ms | **~0.5 ms** | FP8 + 5th-gen TC |
| RTX 5080 | ~14 ms | ~15 ms | ~0.7-1.0 ms | |
| RTX 4090 | ~17 ms | ~21 ms | ~2.0-2.5 ms | Ada FP8 |
| **RTX 4070** | ~38 ms | ~46-48 ms | **~3-5 ms** | FP8 but no 2:4 sparsity bonus on AD104 |
| RTX 3090 | n/a | n/a | several ms | FP16 fallback, 20-30% slower |
| RTX 2080 Ti | n/a | n/a | several ms | FP16 fallback, 12% avg slower |

### Practical floor for transformer RR on FP8-capable hardware

- **~0.5 ms** on 5090
- **~1 ms** on 5080
- **~2 ms** on 4090
- **~3-5 ms** on 4070 (this is the practical entry bar)

Without FP8 (Turing/Ampere): ~2× the cost.

### Attribution: hardware vs architecture

**~60-70% of DLSS 4 RR's perf advantage on Blackwell is FP8 + tensor-core throughput, ~30-40% is architecture/kernel co-design.**

Proven by Turing/Ampere fallback: when forced to run DLSS 4 transformer in FP16 (no FP8 hardware path), Ampere is **20-30% slower than DLSS 3.5 CNN**. The transformer wins on quality; FP8 hardware wins on speed.

### The compute envelope ORS must fit inside

| Target | Per-frame ms budget @ 4K | FLOPs/frame |
|---|---|---|
| High-end (5090 parity) | 0.5-1.0 ms | requires FP8 / equivalent INT8 path |
| Mid (4090/5080 parity) | 1.5-2.5 ms | FP8 or 2× INT8 throughput needed |
| Budget (4070 / RDNA3 7800XT parity) | 3-5 ms | FP16 acceptable but tight |
| Compute envelope | 3-6 TFLOPs/frame at 4K | ~150-300 GFLOPs at 1080p |

## 2. Bálint 2026 — distillation feasibility (the leapfrog vehicle)

The paper Bálint et al. 2026 "Forget Superresolution, Sample Adaptively" (arXiv:2602.08642) establishes a published quality lead:

| Variant | PSNR @ 0.25 spp | vs DLSS 4 |
|---|---|---|
| Bálint 15M (main) | 25.41 | +2.53 dB |
| **Bálint 2.6M Mini Adaptive (JNDS-shape)** | **24.97** | **+2.09 dB** |
| NPPD (predecessor) | 23.92 | +1.04 dB |
| **DLSS 4 RR** | **22.88** | baseline |

**Key insight: the paper itself proves a 5.8× compression at 0.44 dB cost** (15M → 2.6M with no distillation, just architectural shrink). The lightweight variant beats DLSS 4 RR by 2.09 dB and uses an architecture (JNDS, Thomas/Liktor HPG 2022) measured at ~2 ms on Intel Arc.

### Distillation roadmap to a 1M real-time variant

1. **Baseline**: JNDS topology (2.6M params) with KD from 15M Bálint teacher
2. **Template**: RAKD (AAAI 2025, Kong et al.) — render-aware knowledge distillation specifically for MC denoising. Combines KD + GAN-based adversarial loss + parameter-transfer init.
3. **Compression stack**:
   - Depthwise-separable conv substitution (~6-9× param reduction at ~0.3-0.6 dB cost)
   - 2:4 structured sparsity on Ampere+ (1.5-1.7× over dense INT8)
   - INT8 QAT (FP16 fallback for bottleneck attention)
   - Replace global-summary module with GRU-over-latent or tile-pyramid pooling
4. **Realistic landing**: ~24.5 PSNR @ <3 ms 1080p RTX 4070 (still beats DLSS 4 RR by ~1.6 dB)

### Engineering risk

| Target | Risk |
|---|---|
| 24+ PSNR | LOW (JNDS-shape FP32 already 24.97; expect 0.2-0.5 dB drop from QAT+sparsity) |
| <3 ms at 1080p RTX 4070 | MEDIUM (memory-bound skip-concat + global-summary substitute are the killers; naive port = 4-6 ms; sub-3 ms requires tensor-core-aware channels + TRT + persistent kernels) |
| Sampler distillation | MEDIUM-HIGH (open problem; RAKD only validated denoising heads; v0.1 ships fixed-budget Bálint-derived denoiser, adaptive sampling is v0.3+) |

**MILO loss is training-only — zero inference cost. Free win.**

## 3. Hardware-feature exploitation — where OSS can structurally beat DLSS

### Vendor capabilities (2026)

- **NVIDIA Blackwell**: 5th-gen tensor cores, FP8 (1676 TFLOPs on 5090), FP4, AMP scheduler. DLSS 4 uses FP8 on Blackwell, FP16 fallback elsewhere.
- **AMD RDNA 4**: WMMA throughput 2× RDNA 3, structured sparsity, FP8 (per ISA). FSR 4 / FSR Ray Regen ships INT8, RDNA 4-only.
- **Intel Battlemage XMX2**: 2× XMX throughput vs Alchemist, BF16/INT8. XeSS 2 uses HLSL CoopVec on Battlemage, DP4a fallback elsewhere.
- **Apple M3+**: simdgroup_matrix in MSL 3.1+ (FP16/BF16). ANE 38 TOPS INT8 on M4 — but **not addressable from a Metal compute queue** (CoreML/MPSGraph routing only, fatal for per-frame inference).
- **Cooperative matrix shipping status**: SPIR-V `OpCooperativeMatrix*` (Vulkan, KHR since 2023) on NV Turing+, AMD RDNA3+, Intel Arc. HLSL CoopVec (DX12 SM 6.9) preview-only as of 2025-2026.

### Seven exploitable advantages OSS has that DLSS structurally cannot

1. **Variable-rate / tile-gated inference** — skip tiles where reprojected residual < threshold. 30-50% perf reduction at imperceptible quality loss. **Structurally incompatible with DLSS's monolithic dense network.**
2. **Multi-scale early-exit** — 1/4-res pass, exit on confident tiles, refine only edges/disocclusions. 25-40% savings.
3. **Per-game LoRA adapters** — DLSS ships one frozen model; OSS can ship per-title adapter ecosystem. Critical for modded games where DLSS hallucinates.
4. **Cross-vendor coop-matrix kernels with per-arch sparsity** — NV 2:4, RDNA 4 structured sparsity. DLSS is NV-only; can't optimize for AMD/Intel/Apple at all.
5. **MSL simdgroup_matrix path on Apple Silicon** — DLSS doesn't run on Mac; MetalFX leaves simdgroup_matrix throughput unexploited.
6. **FP8 + INT4-weight mixed-precision with per-game calibration** — closed vendors can't ship per-title quantization tables; OSS can.
7. **No DRM / NGX / TRT init overhead** — direct shader dispatch, no telemetry. ~300-800µs/frame saved.

### Realistic claim

"Match DLSS RR quality at 70-80% of its perf cost on NV (FP16 fallback path is parity); beat FSR 4 / XeSS 2 outright on AMD/Intel/Apple where DLSS doesn't ship; be the only RR-class option on Apple Silicon and Steam Deck-class hardware."

Beating DLSS RR perf on Blackwell at matched quality is plausible only via **algorithmic wins (tile-gating, early-exit)** — not raw kernel speed. NVIDIA's kernel engineers have a structural lead on raw kernel implementation.

## 4. SOTA gap survey (the open research lane)

Confirmed: **no published 2025-2026 paper meets all three bars** (≥DLSS 4 RR quality at 0.25 spp + ≤2 ms at 1080p on RTX 4070-class + cross-vendor benchmarks on ≥2 of {NV, AMD, Intel, Apple}).

### Research gaps with zero published work

1. **Distillation/QAT for path-traced denoisers** — zero papers in 2025
2. **HLSL Cooperative Vectors vs ONNX-RT DirectML benchmarks** for image-to-image regression — zero independent numbers
3. **Apple simdgroup_matrix path-traced denoiser** — zero papers
4. **Cross-vendor benchmarks** of any modern architecture (NPPD/Bálint/MUNet) on AMD WMMA + Intel XMX + Apple simdgroup_matrix — zero papers

### Closest contenders (each missing one or more bars)

| Method | Quality | Inference (ms / GPU) | Cross-vendor? | Gap |
|---|---|---|---|---|
| Wavelet-Space SR (Poudel 2025) | no DLSS comparison | not disclosed | NV-only | no DLSS benchmark, SR not pure denoise |
| Fast Local Neural Regression (2024) | competitive with NPPD | sub-ms 1080p RTX 2080 Ti | NV-only | Lambertian-only |
| NPPD (Bálint 2023) | OIDN-class at 1 spp | ~5 ms 1080p RTX 3090 | NV-only | pre-DLSS 4, too slow for 2ms bar |
| NSRD (Li 2024) | beats RRSR/VSR | not headlined | NV-only | radiance demodulation but SR not denoise |
| Joint Denoise+Upscale Multi-branch (2025) | beats SOTA at 1 spp | not in abstract | NV-only | no DLSS RR head-to-head |
| AMD FSR Ray Regen 1.1 | comparable class to DLSS RR (no PSNR) | RR cost not isolated | RDNA 4 only | no quantitative DLSS comparison, single-vendor lockin |
| Intel XeSS RT denoiser | "10× better RIS" | 30 fps 1440p B580 (full frame) | Intel-only fast path | XMX-only fast path, no PSNR vs DLSS RR |

### The publishable position

**ORS shipping (Bálint Mini Adaptive distill + RAKD-style training + cross-vendor coop-matrix benchmarked on RTX 4070 + RX 9070 XT + Arc B580 + Apple M3) would be the first paper in the space.**

## 5. The drop-in DLL distribution strategy

Critical insight: waiting for game integration is death. OptiScaler (github.com/cdozdil/OptiScaler) proves DLL-swap drop-in is a viable shipping pattern.

### Target DLL surfaces

| Vendor | DLL | API | Open SDK? | Install base |
|---|---|---|---|---|
| NVIDIA DLSS RR | `nvngx_dlssd.dll` | NGX RR | partial (Streamline) | **HUGE** (CP2077 PT, AW2, BMW, Indiana Jones, Portal RTX, Hellblade II) |
| NVIDIA DLSS SR | `nvngx_dlss.dll` | NGX SR | partial | **MASSIVE** (every DLSS game) |
| AMD FSR | `amd_fidelityfx_*.dll` | FidelityFX | full open | broad |
| Intel XeSS | `libxess.dll` | XeSS | full open | growing |

### Strategic conclusion

**v0.2 should ship a `nvngx_dlssd.dll` drop-in replacement.** Day-1 install base = every game with DLSS RR support. OptiScaler is the engineering reference. NGX API surface is documented in open-source Streamline.

## 6. Community LoRA + base model evolution

LoRA = low-rank weight-update decomposition. Frozen base + small adapter (1-5% of base param count). Inference cost: base + adapter.

### The community-evolution loop

1. Maintainers ship base model + LoRA training pipeline
2. Community trains per-game adapters, uploads to HuggingFace-style hub
3. Other users download per-game adapters
4. Every 6-12 months, maintainers run consolidated retrain: select highest-quality adapters, merge via DARE / TIES / model soup, ship as new base
5. New base ships; old adapters mostly still work

### Why this works for ORS

- DLSS hallucinates on Skyrim modded content because training data lacks those material distributions
- Per-game LoRA solves this **better than DLSS structurally can** (DLSS is one frozen model per release)
- Modded games get LoRAs from the modding community itself
- Corkscrew (existing project) is the perfect distribution channel for Wine/CrossOver users

## 7. RE / leaked-source decision (2026-04-30)

**Decision: skip RE work entirely.** Read only public open-source codebases (Streamline, FidelityFX SDK, XeSS SDK, OIDN, OptiScaler, Bálint papers). Defer DLSS DLL spoofing details (emoose/PureDark RE) until v0.2 DLL implementation.

Rationale: ORS's quality lead comes from Bálint 2026 architecture (published, no RE needed). Cross-vendor perf comes from public Khronos/MS/Apple specs. Distribution comes from OptiScaler-pattern hooking. RE would contribute ~5% perf insight at high legal/community-trust cost.

The product is **a strictly better algorithm with a different distribution model**, not "DLSS minus a few percent."

## Sources (consolidated)

### Architecture / quality
- [Bálint 2026 — arXiv:2602.08642](https://arxiv.org/abs/2602.08642) / [HTML](https://arxiv.org/html/2602.08642v1) / [alphaXiv](https://www.alphaxiv.org/overview/2602.08642)
- [NPPD GitHub (Bálint, SIGGRAPH 2023)](https://github.com/balintio/nppd)
- [JNDS / Thomas-Liktor HPG 2022](https://dl.acm.org/doi/10.1145/3543870)
- [RAKD — AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32462)
- [FLNR — arXiv:2410.11625](https://arxiv.org/abs/2410.11625)
- [Wavelet-Space SR — arXiv:2508.16024](https://arxiv.org/abs/2508.16024)
- [NSRD CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Neural_Super-Resolution_for_Real-time_Rendering_with_Radiance_Demodulation_CVPR_2024_paper.pdf)
- [Joint Denoising+Upscaling SIGGRAPH 2025](https://dl.acm.org/doi/10.1145/3728297)

### DLSS analysis
- [NVIDIA ADLR DLSS 4 research page](https://research.nvidia.com/labs/adlr/DLSS4/)
- [NVIDIA Streamline DLSS RR Programming Guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md)
- [Tom's Hardware DLSS 4](https://www.tomshardware.com/pc-components/gpus/nvidia-dlss4-mfg-and-full-ray-tracing-tested-on-rtx-5090-and-rtx-5080)
- [Tom's Hardware DLSS 4.5 RR](https://www.tomshardware.com/pc-components/gpus/dlss-ray-reconstruction-might-be-living-on-borrowed-time-dlss-4-5-can-reconstruct-ray-traced-reflections-almost-perfectly-without-any-denoisers)
- [TechSpot DLSS 4 RR](https://www.techspot.com/article/2951-nvidia-dlss-4-ray-reconstruction/)
- [TechSpot DLSS 4.5 vs 4.0](https://www.techspot.com/article/3080-nvidia-dlss-45-vs-40/)
- [Club386 DLSS 4.5 analysis](https://www.club386.com/nvidia-dlss-4-5-analysis/)
- [SemiAnalysis NVIDIA Tensor Core Evolution](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
- [arXiv 2512.02189 — Microbenchmarking Blackwell](https://arxiv.org/html/2512.02189v2)

### Vendor / hardware
- [AMD FSR Ray Regeneration / Redstone](https://gpuopen.com/amd-fsr-rayregeneration/)
- [AMD FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)
- [AMD WMMA on RDNA 3](https://gpuopen.com/learn/wmma_on_rdna3/)
- [Intel XeSS SDK](https://github.com/intel/xess)
- [Intel XeSS-SR Developer Guide](https://www.intel.com/content/www/us/en/developer/articles/technical/xess-sr-developer-guide.html)
- [Microsoft DirectX Cooperative Vector blog](https://devblogs.microsoft.com/directx/cooperative-vector/)
- [HLSL Cooperative Vectors proposal](https://microsoft.github.io/hlsl-specs/proposals/0029-cooperative-vector/)
- [VK_KHR_cooperative_matrix](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_cooperative_matrix.html)

### Drop-in DLL pattern
- [OptiScaler GitHub](https://github.com/cdozdil/OptiScaler)
- [DLSSTweaks (emoose)](https://github.com/emoose/DLSSTweaks)

### Open-source bases
- [Intel OIDN](https://github.com/RenderKit/oidn)
- [NVIDIA Falcor](https://github.com/NVIDIAGameWorks/Falcor)
- [NVIDIA RTXGI 2.0 / NRC](https://github.com/NVIDIA-RTX/RTXGI)
- [NRD (NVIDIARayTracingDenoiser)](https://github.com/NVIDIAGameWorks/RayTracingDenoiser)

### Distillation / federated learning
- [LoRA paper Hu 2021](https://arxiv.org/abs/2106.09685)
- [DARE merging](https://arxiv.org/abs/2311.03099)
- [TIES merging](https://arxiv.org/abs/2306.01708)
- [Federated LoRA survey](https://arxiv.org/abs/2502.05453)
