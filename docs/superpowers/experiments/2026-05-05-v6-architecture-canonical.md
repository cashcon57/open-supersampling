# 2026-05-05 — v6 architecture: covariance-resampled online Gaussian-temporal SR

**Status:** Canonical v6 target design. Current implementation status: As of commit `fd8965f`, `V6Model.forward()` runs the full canonical Stage 2 critical path: HAT backbone → motion-vector + GS-STVSR covariance canvas warp → keyframe active mask → cross-attention pixel↔Gaussian fusion → V6Rasterizer renders the active canvas subset to HR → composite head produces 3-channel RGB → softplus / sigmoid → spawner writes fresh Gaussians from refined HAT features back into the persistent per-rank canvas → ST score state updates. 234 v6 tests pass. Trainer's trajectory loop with canvas continuity across frames + temporal-consistency loss is the next commit; OSS-FX (α<1 canvas rendering) is post-trainer-loop. The full diagram below is the target architecture. **Supersedes** the earlier "two-tier teacher/student distillation" memo (`2026-05-05-v6-design-two-tier-distillation.md`) once Stage 3 of staged validation begins.

**Predecessors:**
- `docs/research/2026-05-05-gaussian-temporal-research-deep-dive.md` — math + 2024-2026 research the architecture incorporates
- `docs/superpowers/specs/2026-05-04-v5-gaussian-temporal-design.md` — Sprint 5 v5-gaussian-temporal architecture (the validation step)
- `docs/superpowers/notes/2026-05-04-s7-game-integration-design.md` — DXGI hook + NGX shim (how v6 reaches users without dev cooperation)

**Implementation roadmap (read before starting v6 code):**
- `docs/research/2026-05-05-v6-external-baselines-integration-plan.md` — concrete sequenced action items derived from deep-reads of GSASR, AAA-Gaussians, AA-2DGS, Analytic-Splatting, vk_gaussian_splatting, GaussianVideo. Five-component plan (fusion module from GSASR, HAT-L warm-start, four-paper AA stack, NVIDIA upper-bound benchmark, GaussianVideo B-spline FX trajectory) with effort estimates and dependency chain.

---

## 0. The pitch in one sentence

> v6 targets **covariance-resampled online Gaussian-temporal super-resolution with score-based active pruning** — temporally stable by mathematical construction in a way pixel-space methods (DLSS, FSR, XeSS) cannot match on splat content, with frame extrapolation (OSS-FX) as a byproduct of fractional-α canvas rendering. It targets DLSS-tier quality at real-time latencies on every major GPU vendor via custom kernels. The planned integration path is a DLL shim for titles already exposing DLSS/FSR/XeSS inputs. No game integration has shipped yet; listed games are candidate validation targets.

---

## 1. Why Gaussian-temporal is the right foundation

The 2024-2026 research consolidates around three techniques none of which a pixel-grid SR can do:

| Technique | What it gives us | What pixel-space methods can do |
|---|---|---|
| **Covariance resampling (GS-STVSR)** | Anti-shimmering by construction — set Σ_recon to target output resolution, EWA filter handles aliasing automatically | Post-hoc filtering only — can't pre-emptively reshape reconstruction kernel |
| **Score-based active pruning (4DGS-1K)** | 14-34× rendering speedup with <0.3 dB quality cost | n/a — pixel grid has no per-primitive contribution to score |
| **Persistent canvas with analytical sub-pixel warp** | No resample-blur compounding across temporal accumulation; densification under disocclusion is exact | Bilinear/bicubic warp blurs every cycle; disocclusion fill is heuristic |

These three plus standard 3DGS rasterization are the architectural moat. Pixel-grid SR is bound by Nyquist of the grid; Gaussian-temporal SR is bound by the canvas's primitive count and the rasterizer's anti-aliasing. The latter ceiling is provably higher.

Target frame extrapolation reuses canvas rendering: render the same canvas at α ∈ (0, 1) along the motion field. One architecture, two products.

---

## 2. Architecture

Current implementation status: As of commit `fd8965f`, `V6Model.forward()` runs the full canonical Stage 2 critical path: HAT backbone → motion-vector + GS-STVSR covariance canvas warp → keyframe active mask → cross-attention pixel↔Gaussian fusion → V6Rasterizer renders the active canvas subset to HR → composite head produces 3-channel RGB → softplus / sigmoid → spawner writes fresh Gaussians from refined HAT features back into the persistent per-rank canvas → ST score state updates. 234 v6 tests pass. Trainer's trajectory loop with canvas continuity across frames + temporal-consistency loss is the next commit; OSS-FX (α<1 canvas rendering) is post-trainer-loop. The full diagram below is the target architecture.

```text
                                    ┌─────────────────────────┐
  current LR + G-buffers ──────────►│ OSS HAT-L-derived Heavy  │──► coarse SR features
  (RGB, depth, motion, normals)     └─────────────────────────┘                │
                                                                                ▼
                                                      ┌────────────────────────────┐
   persistent Gaussian canvas ──► analytical warp ───►│ cross-attention            │──► refined HR features
   (5K-15K Gaussians per scene,    by engine MVs +    │ (pixel queries × Gaussian  │
    accumulated across frames)     covariance         │  keys/values)              │
                                   resampling         └────────────────────────────┘
                                                                                │
   key-frame active mask (every K=10 frames) ─────────────► rasterizer ─────► HR output
                                                                                │
   Spatial-Temporal Variation Score pruning ◄─── update canvas ◄────────────────┘
   (every N steps, prune bottom 60-80% by S_i = SS_i · TS_i)

   For frame extrapolation (OSS-FX): rasterize canvas at α ∈ (0, 1) instead of α = 1.
   Cost: one in-place add to position tensor. Free.
```

### Component breakdown

| Component | Implementation | Source |
|---|---|---|
| **OSS HAT-L-derived Heavy spatial backbone** | Window attention + channel attention transformer, ~17M target params | Chen et al., 2023 (Hybrid Attention Transformer) |
| **Persistent Gaussian canvas** | `oss/gaussian/canvas/` — already implemented | OSS Sprint 5 |
| **Analytical warp w/ covariance resampling** | extends `oss/sr/gaussian_temporal/analytical_warp.py` with `Σ'_output = J_t Σ_t J_t^⊤ + Σ_recon` | GS-STVSR (Zhou et al., 2026) |
| **Cross-attention pixel↔Gaussian** | new layer; pixel grid features cross-attend to Gaussian token K/V | OSS-original |
| **Score-based pruning** | extends `oss/sr/gaussian_temporal/pruning.py` with Spatial-Temporal Variation Score | 4DGS-1K (Yuan et al., NeurIPS 2025) |
| **Key-frame active-Gaussian mask** | new cache in canvas; rasterizer skips inactive Gaussians per K=10-frame window | 4DGS-1K |
| **Tile-based rasterizer** | `oss/gaussian/renderer/` (gsplat-derived) — already implemented | Kerbl et al., SIGGRAPH 2023 |
| **Frame extrapolation** | `oss/gaussian/extrapolation/extrapolator.py` — already implemented; renders at α<1 | OSS Sprint 6 prep |

### What this is NOT

| Wrong path | Why we're not on it |
|---|---|
| Branch A: deformation field (4D-GS Wu) | Solves offline reconstruction with full temporal context — we have only past frames in a streaming game scenario |
| Branch B: native 4D primitives (4D-Rotor / Spacetime GS) | 1.5-3× more primitives for equal quality; topology change is already handled cleanly via online densification on disocclusion |
| Per-Gaussian time-MLPs (GTM) | Per-Gaussian compute too high for real-time budget — v7+ candidate when there's headroom |
| Gaussian Frosting (mesh + frost layer) | Asset-pipeline tech, not real-time SR — relevant for game-engine integration team, not v6 |

---

## 3. Three model tiers — same architecture, scaled

Latency targets below are ship goals, not current measurements. They are conditional on the per-vendor native-kernel sprint landing at vendor-stack optimization quality (CUDA + CUTLASS + tensor-core MMA on NVIDIA, HIP + rocWMMA on AMD desktop, Metal + ANE on Apple Silicon, Level Zero + XMX on Intel Arc, hand-tuned Vulkan compute for Deck-class hardware). Stock-runtime latency on the same models is several × the numbers below.

| Tier | Backbone | Canvas | Target hardware | Ship target (conditional) | Storage |
|---|---|---|---|---|---|
| **Pico** | HAT-Tiny (~1M params) | ~1-2K Gaussians | Steam Deck, integrated GPUs, mobile dGPU | <2 ms at 720p→1080p (Vulkan compute, no matrix accel) | ~12 MB |
| **Standard** | HAT-Small (~5M params) | ~5K Gaussians | Mainstream desktop (RTX 30+, RX 6700+, Arc, M2+) | <3 ms at 1080p→1440p | ~30 MB |
| **Heavy** | OSS HAT-L-derived Heavy (~17M target params) | ~15K Gaussians | Enthusiast (RTX 4080+, RX 7900+, M4 Max) | <4 ms at 1440p→4K | ~80 MB |

These targets bracket vendor latency bands. The target Pico latency band falls within handheld budgets; measurement pending. Standard sits in DLSS 2/3 SR territory (~1.5-2.5 ms at 1080p→4K typical). Heavy lands at DLSS 4 transformer territory (~3-4 ms at 1080p→4K on RTX 30+ FP16, ~1.5-2 ms with FP8 on RTX 40+/50+). FSR 4 ML on RDNA4 is in similar territory to DLSS 2/3 (~1.5-2 ms at 1080p→4K). XeSS XMX is ~2-3 ms on Arc; XeSS dp4a fallback is ~5-8 ms cross-vendor.

Distillation cascades **Heavy → Standard → Pico**. Same training data, same loss, same architecture, just sized.

**Handheld is back in scope** because custom Vulkan compute kernels — the same engineering intensity FSR uses on Steam Deck — are part of the project regardless of tier.

### Candidate Pico-tier architecture: GRAPE (Jang and Jin, WACV 2026)

The Pico tier above ("HAT-Tiny + ~1-2K Gaussians") is a sizing target, not a fixed architecture. **GRAPE** (Gaussian Rendering for Accelerated Pixel Enhancement, [WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Jang_GRAPE_Gaussian_Rendering_for_Accelerated_Pixel_Enhancement_Brings_Fast_and_WACV_2026_paper.html)) is a concrete published architecture in the right footprint:

| GRAPE attribute | Value | Pico-tier fit |
|---|---|---|
| Total params | 1.56M | matches Pico ~1M target |
| GPU memory | 1.10 GB | fits Steam Deck shared-memory budget |
| Throughput | 69.33 FPS at 4× on Urban100 (985×798) | demonstrates a compact Gaussian predictor in a relevant footprint |
| Speedup vs GSASR baseline | 315× | demonstrates the compact-Gaussian-predictor pattern is real-time-viable |
| Architecture | single point-wise layer predicts anisotropic Gaussian params (RGB + rotation + scale + offset), differentiable rasterizer renders HR in one pass | clean target for a Vulkan compute kernel port |

GRAPE is **single-image SR** (no temporal context). The OSS contribution would be extending it with the persistent canvas + analytical sub-pixel warp + covariance resampling that the rest of v6 already commits to. That extension is the first concrete prototype to build when Pico-tier work begins, after v5-Gaussian-temporal validates and the v6 Heavy / Standard tiers prove the recipe.

Also bookmarked from the same review pass:
- **DSA-SRGS** (Zhang et al., 2026, arXiv:2603.04770) — confidence-aware mixing of trusted-but-sparse HR signal with abundant-but-hallucinatory pseudo-labels. Directly applicable to v6.1's INSANE-mode-supersample-GT vs diffusion-teacher-synthesis mixing problem.
- **SR3R** (Feng et al., CVPR 2026, arXiv:2602.24020) — independent validation that feed-forward cross-scene Gaussian-field prediction is viable.

---

## 4. Per-vendor custom kernel strategy

ONE architecture across all vendors. Custom kernels for each vendor's primitives. This is the "per-vendor specialists" path the README has always called for.

| Vendor | Backend | Primary primitives | Implementation reference |
|---|---|---|---|
| **NVIDIA** (RTX 30+) | CUDA + CUTLASS + tensor-core MMA | Fused HAT attention + Gaussian rasterizer + cross-attention | `docs/superpowers/notes/cuda-mega-kernel-design.md` |
| **AMD desktop** (RDNA 3+) | HIP + rocWMMA / MFMA | Same set, AMD matrix-core path | `docs/superpowers/notes/vendor-optimization-audit.md` |
| **Apple Silicon** (M2+) | Metal + MPS / MLX, ANE-resident matmul | Metal-kernel HAT + ANE matmul | `oss/gaussian/ports/metal/` (scaffolded) |
| **Intel Arc** (XMX-equipped) | Level Zero + XMX | XMX attention + Vulkan rasterizer fallback | Vendor audit memo |
| **Steam Deck / fallback** (RDNA 2, no matrix accel) | Pure Vulkan compute | Hand-tuned compute shaders, FSR-style | Planned |

Wall-time estimate: 6-12 months of parallel engineering across the five backends. We don't shy away from this — it's exactly what every shipped ML upscaler (DLSS, XeSS-XMX, FSR 4) does internally.

### Per-vendor inference precision

Training is bf16 mixed precision (see §6). Shipping precision is per-vendor, picked to match each platform's matrix-engine native format. FP8 is the fast path where the tensor / matrix engine supports it natively; FP16 is the floor everywhere else; INT8 / dp4a is the fallback for Vulkan-compute-only paths with no matrix accelerator.

| Hardware | Inference precision | Path | Speedup vs FP16 reference |
|---|---|---|---|
| NVIDIA Ada (RTX 40-series) / Blackwell (RTX 50-series) | **FP8** | TensorRT FP8 PTQ on tensor cores | ~2× |
| NVIDIA Ampere (RTX 30) / Turing (RTX 20) | FP16 | TensorRT FP16 on tensor cores | baseline |
| AMD RDNA4 (RX 9000+) | **FP8** | HIP / ROCm + matrix cores (FP8 native on RDNA4) | ~2× |
| AMD RDNA3 (RX 7000) | FP16 | HIP / ROCm + matrix cores (RDNA3 has matrix but FP16-only) | baseline |
| AMD RDNA2 (RX 6000, Steam Deck) | INT8 / dp4a | Vulkan compute, no matrix accelerator | varies |
| Intel Arc B-series (Battlemage) | **FP8** | Level Zero + XMX FP8 native | ~2× |
| Intel Arc A-series (Alchemist) | FP16 | Level Zero + XMX (FP16-only on A-series) | baseline |
| Apple Silicon (M3+) | FP16 | Metal MPS + ANE (ANE does not expose FP8 as of 2026) | baseline |

The FP8 path is what closes the gap to DLSS 4's 1.5-2 ms numbers. PTQ FP8 calibration typically costs 0.1-0.3 dB PSNR vs FP16 with proper calibration; that delta is acceptable in the v6 ship target band. INT8 PTQ on Ampere we already measured (`docs/superpowers/experiments/2026-05-03-trt-int8-quantization.md`) — quality gate passed (+0.46 dB PSNR, −0.010 LPIPS vs FP32) but speed regressed at most resolutions on the v4 model because INT8 overhead exceeded precision savings at v4's parameter count. v6 at OSS HAT-L-derived Heavy size should benefit from FP8 PTQ where the v4 model didn't from INT8.

The shipping pipeline is therefore: **bf16 train → FP16 ONNX export → vendor-specific quantization** (TRT FP8 / HIP FP8 / Level Zero FP8) + **vendor-specific compiled engine**. Distillation is orthogonal: Heavy → Standard → Pico is parameter scaling, run before quantization, with each tier then quantized independently per target vendor.

---

## 5. Loss recipe (carries from prior v6 thinking)

| Loss | Weight | Purpose |
|---|---|---|
| Charbonnier (smooth L1) | 1.0 | Pixel fidelity, smoother gradients near zero (~+0.1-0.2 dB vs L1) |
| LPIPS (VGG) | 1.0 | Perceptual quality |
| Multi-scale VGG L1 | {0.1, 0.1, 1.0, 1.0, 1.0} on relu1_1 / relu2_1 / relu3_1 / relu4_1 / relu5_1 | Perceptual at multiple scales |
| Wavelet L1 | 0.5 | Explicit high-frequency supervision (targets thin-feature ghosting like power lines) |
| GAN (UNetD, hinge) | 0.05, **starts at step 20K** | Sharpness without instability |
| Edge (Sobel L1) | 0.2 | Sharpness regularizer |
| Temporal consistency (warp t→t+1, L1) | 0.5 | Temporal stability supervision |
| Gaussian regularization (anti-collapse) | 0.01 | Prevent runaway Gaussian growth/shrinkage |

GAN warmup: pixel-only first 20K steps, add discriminator after. Stabilizes training significantly.

---

## 6. Training recipe + dataset

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.99), wd=1e-4 |
| LR | 2e-4 cosine + 3 warm restarts, T_0=50K, T_mult=1 |
| Precision | bf16 (Ampere+ supports natively, more stable than fp16 for GAN) |
| Effective batch | 16 (batch=4, accum=4) for OSS HAT-L-derived Heavy teacher |
| Patch size | 256² (OSS HAT-L-derived Heavy teacher); 192² (Standard); 128² (Pico) |
| Patch sampling | 70% importance-sampled (variance-weighted) + 30% uniform |
| EMA | β=0.999 (teacher only, not students) |
| Steps (teacher / Heavy) | 300K |
| Steps (per student tier) | 80K with KD loss + GT loss |

### Data

| Dataset | Use | Held-out |
|---|---|---|
| TartanAir | 60% of training mix (real depth + flow + photoreal outdoor) | env `oldtown` |
| Hypersim | 30% of training mix (real depth + normals + photoreal indoor) | random 10% scenes |
| (eval) | 10% — held-out from both | combined eval set |

**No SRGD.** SRGD's zero G-buffers create distribution-mix mess. v6 commits to engine-provided G-buffers throughout the training mix.

### Channels

| Channel | Bytes | Purpose |
|---|---|---|
| RGB | 3 | Color |
| Depth | 1 | Geometry / disocclusion |
| Motion vectors | 2 | Engine-provided MVs for warp |
| Normals | 3 | Surface info / Gaussian fit prior |
| **Total** | **9** | — |

Drops the SRGD-era canvas hint (was wasted 3 channels on TartanAir/Hypersim — neither has it). Future-proof for INSANE-mode capture data: + albedo (3) + roughness (1) + metallic (1) = 14 channels when relighting becomes relevant (v7+).

---

## 7. Game integration — planned DLL shim, no dev cooperation

The planned integration path is a DLL shim for titles already exposing DLSS/FSR/XeSS inputs. No game integration has shipped yet; the listed games below are candidate validation targets. Games using DLSS / FSR / XeSS already provide depth + motion vectors + jitter through stable APIs; S7 builds the shim that can consume those inputs.

### Three integration tiers (all without dev cooperation)

| Game already supports | We shim | Quality |
|---|---|---|
| DLSS 2 / 3 / 4 (NVIDIA NGX) | `nvngx_dlss.dll` masquerade | best — full payload from game |
| FSR 2 / 3 (AMD FidelityFX) | `ffx_fsr2_*.dll` masquerade | best |
| XeSS (Intel) | `libxess_*.dll` masquerade | best |
| TAA only (older games) | DXGI resource intercept + heuristics | medium (tier 2 in `oss/model/oss_fx_warp.py`) |
| Custom AA / no temporal SR | DXGI intercept + on-the-fly RAFT-Small flow | lower (tier 3 fallback) |

### Candidate validation games (DLSS-supporting, no kernel anti-cheat)

Cyberpunk 2077, Alan Wake 2, Hogwarts Legacy, Starfield, Baldur's Gate 3, Returnal, Hellblade II, Forza Horizon 5, Ghost of Tsushima Director's Cut, and Black Myth: Wukong. These are candidate validation targets once the S7 shim exists.

### What's off-limits

Anything with kernel anti-cheat (Vanguard, BattlEye, EAC, Ricochet) — DLL injection trips them, ban risk. Off the supported-games list permanently.

### What dev cooperation WOULD buy (optional, post-launch)

Native engine plugin (UE / Unity), modder community reach, DLSS-API-update protection. Useful but not required for the S7 validation path.

---

## 8. Frame extrapolation (OSS-FX) as target byproduct

`oss/gaussian/extrapolation/extrapolator.py` already implements α-conditioned canvas rendering. v6 trained model produces:

| α | Output | Use case |
|---|---|---|
| α = 0 | render canvas at time t | baseline display |
| α = 1 | render canvas at time t + motion vec | SR (current frame) |
| α = 0.5 | render canvas at time t + 0.5·motion vec | FX intermediate frame (60→120 fps) |
| α ∈ (0, 1), variable schedule | scheduled by `alpha_scheduler.py` | arbitrary target FPS (90, 144, 240) |

Target cost above-and-beyond a normal canvas render: **one in-place add to the (N, 2) position tensor**. End-to-end measurement is pending.

Compare to DLSS Frame Generation, which is a separate ML network with its own training cost, latency, and hallucination artifacts. **Same v6 trained weights serve both products.**

---

## 9. glTF KHR_gaussian_splatting compatibility

Khronos extension for Gaussian splats in glTF, ratification Q2 2026, backed by Google + NVIDIA + Apple + Bentley.

**v6 outputs glTF-compatible Gaussian state from day 1.** Once ratified, any glTF-compatible engine or viewer loads OSS canvases natively — no plugin work for the consumer side.

This makes the asset story align with industry direction. The OSS canvas is just a glTF mesh primitive with the KHR_gaussian_splatting extension attached.

---

## 10. Wall-time + cost estimate

| Phase | Wall time on 3080 Ti | Wall time on rented A100 80GB | Cost |
|---|---|---|---|
| **Stage 0-2: validation** | ~7 hours | n/a (cheap, run locally) | $0 |
| **Stage 3: v5-gaussian-temporal full training** | ~36 hours | ~6 hours | $12 (A100) or $0 (3080 Ti) |
| **v6 architecture impl** (cross-attention, covariance resampling, S-T scoring) | ~2 weeks engineering | n/a | $0 |
| **v6-Heavy teacher training** | ~50-60 hours | ~10 hours | $20 (A100) or $0 (3080 Ti) |
| **v6-Standard distillation** | ~15 hours | ~3 hours | $6 |
| **v6-Pico distillation** | ~15 hours | ~3 hours | $6 |
| **Custom CUDA kernels (NVIDIA primary)** | ~3 months engineering, no GPU dependency | — | $0 |
| **HIP / Metal / Level Zero / Vulkan ports** | ~6-9 months parallel engineering | — | $0 |
| **DXGI hook + NGX shim runtime (S7)** | ~2-3 months engineering | — | $0 |
| **Total to first shippable demo** | **~5-6 months** | accelerated to ~3 months with rented A100 | <$100 GPU rental |

---

## 11. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Gaussian-temporal SR has no production shipping precedent | medium | Stage 0/1/2 validation gates de-risk the convergence question cheaply |
| Per-frame Gaussian fitter cost may blow real-time budget | medium-high | Score-based pruning + key-frame active mask reduce per-frame work 14-34× per 4DGS-1K |
| Cross-attention between pixel features and Gaussian tokens has no production precedent | medium | Custom CUDA kernel work; fall back to ORT-CUDA EP if blocked. Reference impl exists in research papers. |
| Differentiable Gaussian densification is research-stage | medium | Conservative densification thresholds; existing `oss/sr/gaussian_temporal/densification.py` already validated in tests |
| GAN training instability | medium | bf16 + hinge + UNetD + GAN warmup at 20K. Monitor D-loss. |
| 6-month timeline slips to 12 | medium-high | Plausible. Ship target is "first user-visible demo Q4 2026 / Q1 2027" |
| DLSS API changes break shim | low-medium | Version-detect + graceful fallback; dev partnerships harden against this in v6.1+ |

---

## 12. What this commits us to

1. **Stage 0/1/2 validation** of v5-gaussian-temporal architecture (~7 hours, decision-gate before paying for full training)
2. **Stage 3 full training** of v5-gaussian-temporal as the convergence baseline (~36h)
3. **v6 architecture implementation:** add OSS HAT-L-derived Heavy backbone, cross-attention layer, covariance resampling, S-T variation score pruning, key-frame active mask
4. **v6 training pipeline:** teacher (Heavy) → student distillation (Standard, Pico)
5. **Custom kernels per vendor:** CUDA (primary), then HIP, Metal, Level Zero, Vulkan
6. **DXGI hook + NGX shim runtime** (Sprint 7) — the planned integration path
7. **glTF KHR_gaussian_splatting output format** for canvas serialization
8. **Frame extrapolation** as a target byproduct via α-conditioned rendering
9. **Three model tiers** (Pico / Standard / Heavy) shipped as one architecture

---

## 13. Out of scope for v6 (parked, with rationale)

| Item | Where it goes |
|---|---|
| Per-Gaussian time-MLPs (GTM-style) | v7+ if specular/glossy quality wants a push |
| Native 4D primitives (4D-Rotor, Spacetime GS) | v7+ if streaming use case ever needs offline-style temporal context |
| Gaussian Frosting (mesh + frost) | OSS-FX / engine-integration team, not SR core |
| GRTX (ray-traced Gaussians) | OSS-RG track (separate from SR) |
| Relighting (GS³, GaRe) | v7+ when INSANE-mode capture data accumulates albedo + roughness + normal + metallic |
| 4DGC compression | shipping infrastructure, not training architecture |
| Generative 4D (DreamGaussian4D, Diffusion4D) | irrelevant — we're temporal SR, not generative content |

---

## 14. HDR migration plan (v6.0 partial → v6.1 full)

HDR support was made shippable-with-caveats in commit `694a0f3` (sigmoid → softplus on the fitter RGB head). Architecture now accepts and produces unbounded non-negative linear-light values; only the training corpus is the bottleneck.

| Phase | Change | Effort | Outcome |
|---|---|---|---|
| **v6.0 (now)** | softplus output + 8-bit sRGB training corpus | done | HDR-shippable at ~70-80% of SDR quality on same content |
| **v6.1 — HDR data** | INSANE-mode HDR capture from HDR-rendered games + Hypersim re-rendered in linear scRGB FP16; retrain teacher on HDR mix | ~2-4 weeks data + retrain | competitive with DLSS HDR on rendered content |
| **v6.2 — wide gamut** | BT.709 vs BT.2020 awareness; explicit linear-vs-sRGB transfer-function metadata in the canvas | ~1 week eng | correct color across HDR pipelines (PQ, HLG, scRGB) |
| **v7+ — perceptual loss for HDR** | replace VGG-LPIPS (trained on SDR ImageNet) with HDR-aware perceptual loss; possibly tonemapped-LPIPS during training | research-grade | closes the perceptual-quality gap on HDR specifically |

Risk: the cheap softplus-only fix may slightly regress SDR quality on the upper end of [0, 1] because training will pull bright values up under softplus's near-linear behavior at output > 0. Worth measuring on the held-out batch when v6 trains. If it regresses meaningfully, we add a runtime `color_activation` switch and ship two model variants (sigmoid-LDR / softplus-HDR) until v6.1 lands.

---

## 15. Followups for v7+

- Per-Gaussian time-MLP heads for view-dependent appearance
- Native 4D primitives if streaming context expands
- Relightable Gaussians once INSANE-mode dataset accumulates
- GRTX integration into OSS-RG track
- glTF KHR_gaussian_splatting extension features beyond static loading
- HDR-aware perceptual loss (per HDR migration plan §14)

---

## 16. References

See `docs/research/2026-05-05-gaussian-temporal-research-deep-dive.md` for the worked-out math underlying every component above.

Key papers:
- Kerbl et al., 3D Gaussian Splatting, SIGGRAPH 2023 (canvas + rasterizer foundation)
- Yuan et al., 1000+ FPS 4D Gaussian Splatting, NeurIPS 2025 (S-T variation score + key-frame mask)
- Zhou et al., GS-STVSR, 2026 (covariance resampling)
- Chen et al., HAT, 2023 (spatial backbone)
- Wang et al., Real-ESRGAN, CVPR 2021 (UNetD GAN training discipline)
