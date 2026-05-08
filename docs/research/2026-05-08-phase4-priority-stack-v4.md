# Phase 4 Priority Stack v4 — Hardware-Tier-Aware

**Date:** 2026-05-08
**Status:** Architecture v4 commitment (signed off 2026-05-08)
**Inputs:** 3 model council reports + 3 individual responses (GPT-5.5, Opus 4.7, Gemini 3.1) — see `docs/research/2026-05-08-phase4-msframe-council/`
**Supersedes:** prior in-flight v3 stack thinking (single-target 5-7ms on 4070-class)

---

## TL;DR

OSS ships as a SR+FG unified upscaler across **5 hardware tiers** from the same architecture. Reference design targets **3-5ms total SR+FG on 3080 Ti desktop and 4070 mobile**. NVIDIA-first with vendor-specific kernels for AMD/Intel/Apple following. No-ML shader variant ships **alongside** v6.2.

The Gaussian canvas is an **L2-friendly temporal residual cache** layered on top of reprojection — not the primary image generator.

---

## 1. Skepticism layer (claims that didn't survive vetting)

The council provided massive value but several claims did not hold up:

| Claim | Source | Status | Notes |
|-------|--------|--------|-------|
| "HAT-Tiny eats 2-3ms" at 4ms budget | Opus | **Contested** | At full BF16 TC throughput on 4070, 54.9G MACs ≈ 0.5ms. Opus mixed FP32 fallback throughput with TC-eligible ops. **Verify with measurement before committing to ≤0.4M student** (codex job queued). |
| "1-4ms total budget" framing | Council synthesis | **Wrong reference** | Compared OSS against SR-only competitors. OSS is SR+FG unified. Correct reference: DLSS4+FG ~3-5ms on Ada → our target band. |
| "Mode A 1ms competitive product" | Council synthesis + GPT-5.5 mode split | **Defer to v8** | Strips neural Gaussian work → loses our differentiation. Don't ship as primary. |
| "32 bytes/Gaussian fits L2 = guaranteed L2-resident" | Gemini | **Directional only** | Byte count is implementation choice (alignment). L2 is shared with engine compute → not guaranteed residency. Design-friendly target, not delivered guarantee. |
| "Frame-gen is essentially free, ~1ms" | Opus | **Mostly right, slightly optimistic** | Free-er than DLSS-FG (no optical flow accelerator) but still ~1.4-2ms for second warp+raster pass. |
| "TensorRT mandatory for runtime" | Multiple | **NVIDIA-path-only** | We're vendor-neutral product. TRT for NVIDIA fast-path; vendor-specific kernels for AMD/Intel/Apple per Path C decision. |
| "8-tier dynamic degradation governor required" | GPT-5.5 | **Defer to v8** | DLSS-class shipping behavior, but engineering scope blocks v6.2. v6.2 ships with 2-tier mode (Quality/Performance) + static config presets per hardware tier. |
| "R=4 mandatory baseline" | Council synthesis | **Tier-dependent** | R=4 baseline for mainstream/entry tiers; R=8 for mid-high reference (3080 Ti / 4070 mobile); R=12-16 for halo. Single-mode "R=4 only" misses headroom on capable cards. |
| "Cross-attention must be DELETED entirely" | Gemini standalone | **Mode-dependent** | True for Performance tier. False for Quality/Ultra tiers — local top-K=16 on disocclusion tiles (~5% windows) is affordable and improves disocclusion quality. |
| "STSS 0.4M / 4.4ms is achievable" | Opus | **Add 30% headroom** | Research benchmark in isolation. Real shipping context has post-process queue, allocator, engine compute pressure. Realistic shipping = 5.5-6ms on equivalent hardware. |

Despite these caveats, the council got most of the architectural reasoning right. Skepticism applies to specific numerical claims and product framing; the kernel-level patterns and design philosophy are sound.

---

## 2. Hardware tier matrix (this is the v6.2 product surface)

| Tier | Examples | TC/ML | Realistic SR+FG @ 1080p→4K | Architecture variant |
|------|----------|-------|----------------------------|----------------------|
| **Halo** | RTX 5090, 4090 | Massive (1500+ TFLOPS BF16) | 1.5-2.5ms | Full v4 + R=12-16 + multi-frame gen 4× |
| **Mid-high (REFERENCE)** | RTX 3080 Ti desktop, 4070 mobile, 7800 XT | Strong (110-300+ TFLOPS BF16) | **3-5ms ← v6.2 target** | Full v4 with R=8 |
| **Mainstream** | RTX 4060, 3060, RX 7600, Arc A580 | Modest (50-150 TFLOPS BF16) | 5-8ms | R=4, ≤0.5M student, FG 1× |
| **Entry ML** | RTX 2060, GTX 1660 Ti, RX 6600 | Limited or no TC | 8-12ms | R=4, no student (LUT only), no FG |
| **No-ML** | GTX 1060, RX 580, Steam Deck, integrated | None | 6-15ms shader path | **Shader variant: TAAU + Gaussian shader splat + classical resolve** |

Reference targets: **3080 Ti desktop (training rig) and 4070 mobile (dev rig)**. Both are mid-high tier. 4070 mobile is the binding bandwidth constraint (~256 GB/s vs 450 GB/s on 3080 Ti).

---

## 3. Architecture invariants (universal across tiers)

These are the same in every tier from Halo to No-ML:

- **Reprojection-first base pass** with motion-vector + depth + material validity mask
- **Persistent Gaussian canvas** as L2-friendly residual cache (FP16 packed, ~24 bytes/Gaussian)
- **Conic row recurrence** in raster inner loop (`w_{x+1} = w_x · r_x`, `Δ²q_x = 2a` constant)
- **Custom counting/radix tile bin** replacing `torch.sort`
- **Persistent tile lists** with incremental update (only Gaussians crossing boundaries)
- **Disocclusion-only hard-spawn** at pixel center, velocity from MV
- **DGP dictionary covariance** (M=8-16 prototype Σ + scalar scale + softmax)
- **Kalman 6-FLOP update** for existing Gaussians; spawner only for births (cap 256/frame)
- **Jacobian-free warp** branch on `|∇·V| < ε` (~90% of canvas)
- **Frame-gen as deterministic second warp+raster pass** (~1.4-2ms)
- **CUDA Graph / equivalent capture** of whole-frame DAG
- **TBDR backward** (training only): SMEM atomics → one block→global atomic

---

## 4. Scaling knobs (per-tier configuration, not new code)

| Knob | Halo | Mid-high (ref) | Mainstream | Entry ML | No-ML |
|------|------|----------------|------------|----------|-------|
| Latent rank R | 12-16 | **8** | 4 | 4 | 4 (RGB+conf direct) |
| Canvas capacity | 32k | **16k** | 8k | 6k | 4k |
| Student model | ~5M FP16/INT8 | **~1M INT8** | ~0.4M INT8 | None (LUT only) | None |
| Backbone temporal rate | 60Hz | **30Hz** | 15Hz | Never | Never |
| Active tile fraction | 50% | **25-30%** | 10-15% | 5% (disocc only) | 0% (canvas-only resolve) |
| Local attn K (disocclusion) | 32 | **16** | 8 | 0 | 0 |
| Frame-gen | MFG 4× | **FG 1×** | FG 1× | Off | Off |
| Runtime | TensorRT/CUDA | **TensorRT/CUDA** | DirectML/ONNX | Compute shader | Pure shader (Slang) |

---

## 5. Build order (sequenced dependencies)

### Tier 0 — free wins, parallel, ship immediately (1-2 days)
- **0a** `head_dim` 30→32 audit + pad
- **0b** Tight ellipse AABB culling: `r_x = √(τd/(ad-b²))`
- **0c** FMA-tightened conic eval
- **0d** Channel-count divisibility audit (mod-8/16/64 across `oss/sr/v6/` and `oss/cuda/`)

These unblock everything downstream and have no architecture commitment.

### Tier 1 — universal kernels (must be portable, target compute-shader semantics)

Core invariant kernels in compute-shader-friendly form:

- **1a** Reproject-first base pass + validity mask
- **1b** R=4 direct-RGB+confidence rasterizer (R=8 toggle for mid-high+, R=12 for halo)
- **1c** Conic row recurrence
- **1d** L2-friendly canvas state (FP16 packed)
- **1e** Custom counting-sort tile bin (no `torch.sort`)
- **1f** CUDA Graph capture / Vulkan command buffer recording
- **1g** TBDR backward (training only)

Implement once in CUDA for fast NVIDIA path; port to Slang/Vulkan compute for cross-vendor; vendor-specific kernels (HIP/ROCm AMD, oneAPI Intel, Metal Apple) follow.

### Tier 2 — architectural changes

- **2a** Concat-fusion + 1×1 conv replacing global cross-attn (Performance tier); local top-K=16 on disocclusion only (Quality/Ultra)
- **2b** Disocclusion-only spawner with hard pixel-center spawn + DGP dictionary
- **2c** Kalman 6-FLOP update for existing
- **2d** Jacobian-free warp branch
- **2e** Frame-gen as Tier 1 first-class

### Tier 3 — model + system

- **3a** HAT distillation to ~1M student (NAFNet block / 3-layer EfficientViT-lite). **Verify HAT-Tiny actual ms first** before committing to size.
- **3b** TensorRT INT8 export (NVIDIA primary), ONNX Runtime fallback for vendor-neutral
- **3c** 2-tier mode (Quality default / Performance preset) — NOT 8-tier governor
- **3d** Inference state split: persist conic Λ; drop scale/rot at runtime

### Tier 4 — defer to v8 / future

- 8-tier dynamic degradation governor (DLSS-class frame pacing)
- Mode A "1ms competitive" product preset
- TC-GS W·G matmul kernel rewrite (do AFTER R=4 baseline ships, measure if needed)
- LUT covariance codebook (do AFTER baseline measures show transcendentals are still bottleneck)
- Adaptive canvas capacity (`S = √(residual·motion entropy)`)
- Multi-rate per-stage temporal scheduling (raster 120Hz / backbone 60Hz / cross-attn 30Hz)
- Hierarchical two-level rasterizer (coarse H/4 + refinement)
- Half-res splat + analytic gradient upsample

### Parallel track — No-ML shader variant

Ships **alongside** v6.2:
- Slang/HLSL/GLSL compute shader rasterizer (port of 1b-1c-1e from Tier 1)
- Reproject base pass via standard fragment shader (TAAU style)
- Hard-coded covariance LUT (no learned spawner)
- Composite + tonemap fragment shader
- No student model, no attention, no frame-gen (initially)

Target: works on anything from Steam Deck to integrated graphics. Differentiator: the only canvas-residual SR via pure shaders.

---

## 6. Reference budget (Quality mode, 3080 Ti desktop, 1080p output)

| Stage | Budget |
|-------|--------|
| Validity mask + reproject base | 0.4ms |
| Canvas warp (Jacobian-free path) | 0.2ms |
| Counting-sort tile bin (incremental) | 0.2ms |
| Sparse R=8 raster (active tiles, conic recurrence) | 0.8-1.2ms |
| Sparse student backbone INT8 (active tiles, 30Hz amortized) | 0.4-0.6ms |
| Concat-fusion + 1×1 decode | 0.2ms |
| Local top-K attn on ~5% disocclusion tiles | 0.15ms |
| Composite/tonemap | 0.2ms |
| Frame-gen 2nd pass (warp+raster, hole correction) | 1.4ms |
| Graph + overhead | 0.15ms |
| **Total Quality+FG** | **~4.0-4.7ms** |

For 4070 mobile: same architecture, expect ~5-6ms total due to reduced bandwidth. Performance preset (no FG, no attn, R=4) targets ~2.5-3ms on 4070 mobile.

---

## 7. Compose vs conflict matrix

**Strong compose (ship together):**
- 1b (R=8 raster) ⊕ 1c (conic recurrence) — payload reduction × arithmetic reduction multiplicatively
- 2b (DGP+disocclusion spawner) ⊕ 2c (Kalman update) — orthogonal: spawner births, Kalman maintains
- 2d (Jacobian-free) ⊕ 1d (L2-friendly state) — both reduce bandwidth pressure
- 2a (concat-fusion) ⊕ 2b (disocclusion spawner) — fusion provides input to spawner-decision

**CONFLICT — gate carefully:**
- Tier 4 hierarchical raster ⊗ 1a validity mask — both reduce per-pixel raster work; pick one
- Tier 4 multi-rate per-stage ⊗ 3c 2-tier mode — combining without coordinator = mode confusion; ship 3c first

---

## 8. Ablation gates (must validate before commit)

| Change | Gate | Action if fail |
|--------|------|----------------|
| HAT-Tiny → ~1M student | PSNR within 0.3dB of HAT-teacher; LPIPS within 0.02 | Re-target to ≤2M student or keep HAT in async path |
| R=8 → R=4 (deferred decision) | PSNR delta < 0.05 dB at fixed canvas density | Keep R=8 baseline, R=4 as Performance toggle only |
| Conic row recurrence | Bit-exact equivalence vs naïve `expf` per pixel within atol=1e-5 | Halt, debug numerics |
| Counting-sort tile bin | <50μs at N=16k on 3080 Ti | Profile, fix; this is mandatory for budget |
| TBDR backward | 5-10× speedup on training step time | Profile shared-mem atomic contention |
| Disocclusion-only spawner | Stippling artifact eliminated visually + FFT residual λ=2px peak <50K | If stippling persists, add DGP dictionary + blue-noise jitter as compounded fix |
| Frame-gen second pass | Total ms with FG ≤ 1.5× ms without FG | Halt, profile share-able state |

---

## 9. References

- Council reports: `docs/research/2026-05-08-phase4-msframe-council/01-03`
- Individual responses: `docs/research/2026-05-08-phase4-msframe-council/04-06`
- Stippling artifact memo: `docs/superpowers/experiments/2026-05-08-v6.1-stippling-artifact-detection.md`
- NVIDIA tensor-core blog survey (covered in council references)
- STSS reference: arxiv 2312.10890 (architectural North Star)
- ContinuousSR DGP: arxiv 2503.06617 (covariance dictionary)
- TC-GS pattern: 2025 paper (alpha-blend → matmul mapping, ~2× rasterizer)
- GS-STVSR: arxiv 2604.18047 (covariance resampling alignment)

---

## 10. What changes for v6.2-pico-002

Killing v6.1-pico-001 (currently training with stippling). Restart pico-002 from scratch with v4 architecture:

- F=64 → R=8 latent feature splat
- Cross-attention → concat-fusion + 1×1 conv (Quality tier baseline; local top-K=16 on disocclusion in v6.2.1)
- Spawner: dense MLP → disocclusion-only + DGP + Kalman maintenance
- HAT-Tiny: keep as TEACHER for distillation; runtime path moves to ~1M student in v6.2.1
- Persistent inference state split (conic Λ persists; scale/rot training-only)

v6.2-pico-002 = first run with new architecture. Quality target: PSNR within 0.3dB of v6.1 teacher; LPIPS within 0.02; **stippling artifact gone**.
