# OpenSuperSampling (OSS)

> Vendor-agnostic open-source real-time game super-resolution and frame extrapolation.

**Status:** Pre-alpha / active research. Single-frame upscaler trained and exported. Temporal track in design. Not yet suitable for production use.

---

## What is OSS?

OSS is a community-governed alternative to proprietary reconstruction stacks (DLSS, FSR 4, XeSS, DLSS RR). It targets the ~60% of the GPU market that vendor-locked solutions don't cover — Steam Deck, mid-range AMD, Intel Arc, Apple Silicon, older NVIDIA — alongside competitive support for current flagship hardware.

No closed weights. No SDK SLAs. No vendor lock-in. Pixel-based and Gaussian-based tracks running in parallel.

---

## Current state (2026-05-04)

### What works

- **OSS-SR** — 2× super-resolution CNN (`sr_cnn` / `simple` backbone, standard tier, ~605K params)
  - Trained on SRGD game content (17 scenes, 51K+ HR frames) with engine-aliased LR synthesis (Halton jitter + TAA blur + JPEG)
  - Production checkpoint: `srcnn-prod-v4-lpips/step-00385000.pt` — fork from v3 step-240K with `L1 + 0.1·SSIM + 0.1·LPIPS-VGG` perceptual loss for 12 hours
  - Held-out eval (rolling 8-sample batches across run):
    - **PSNR**: ~30 dB (vs bicubic ~27 dB, +2.5–3 dB margin)
    - **LPIPS**: ~0.30 (vs bicubic ~0.45, **lower is better; perceptually preferred 8/8 frames per eval**)
    - vs v3 (L1+SSIM only): **−23% LPIPS** for **−1.5 dB PSNR** — Real-ESRGAN-paper-consistent perceptual/fidelity trade
  - LR-synth fidelity: validated against engine-aliased capture — no bicubic-trap exposure
  - Auto-resume from latest checkpoint with optimizer state, rolling metrics dump on every checkpoint, live web dashboard

- **Inference pipeline**
  - **PyTorch FP16 + channels-last + CUDA Graphs** — runtime-optimized engine in `oss/sr/inference.py`
  - **ONNX export** with dynamic axes (bilinear-skip ONNX path; bicubic-skip-with-antialias is currently un-exportable on PyTorch 2.4.1 / opset 17 — measured quality delta is −0.003 dB, negligible)
  - **TensorRT FP16** with narrow optimization profiles per resolution
  - **TensorRT INT8 PTQ** — quality gate passes, but performance regresses vs FP16 on RTX 3080 Ti at most kernel shapes (one win at 900p input). Path documented but not the primary deployment target on Ampere.

- **Tooling**
  - Live training dashboard at `scripts/training_dashboard.py` — Chart.js panels for PSNR/LPIPS/loss/SSIM, polls `metrics.json` and `score_log.json`, downsamples large series
  - Lab-notebook-discipline experiment memos in `docs/superpowers/experiments/` — every result that drives a decision is documented before it does

### Measured inference latency (RTX 3080 Ti, TensorRT FP16, narrow profile)

| Input → Output | Median latency |
|---|---|
| Steam Deck 800p | 18.6 ms |
| 720p → 1440p | 15.6 ms (~64 fps headroom) |
| 1080p → 4K | 37.6 ms (~27 fps headroom) |

These numbers are honest and current. **The deliberate comparison is FSR 2/3 at ~0.7–1 ms** (hand-tuned compute shaders, no ML). We are roughly 20–40× slower than the dominant non-ML upscaler in the same quality tier. Closing that gap is the second-half roadmap below.

### Known limits

- **Single-frame only.** No temporal accumulation, no frame history. Motion vectors are fed as input but the model has no recurrent state to use them. v3 and v4 trained with zero G-buffers in SRGD (depth/motion/normals were placeholder zeros) — temporal gains are unrealized.
- **Inference cost dominates.** ML-based upscaling on a model this size is fundamentally heavier than FSR 2/3 shader passes. Bridging requires custom kernels per vendor (planned), distillation to lite tier, and likely sparse / tile-based execution.
- **Steam Deck not yet viable.** RDNA 2 has no matrix accelerators. Until lite tier + custom Vulkan compute kernels land, Deck will fall back to FSR 2 in the runtime.
- **Not deployment-ready.** No game integration shipped. No NGX/DXGI hook in production. Inference engine works in isolation; runtime plumbing is the next sprint.

---

## Roadmap

### Near-term: dual-track temporal v5

The current v4 is the strong single-frame baseline. Quality is now bottlenecked by missing temporal accumulation. We're racing two architectures so we have a fallback if the experimental path doesn't pan out.

#### v5-pixel-temporal (control track)

**FSR 2-class temporal warp+blend.** Proven recipe, bounded risk, predictable result.

- Architecture: warp prev-frame HR output by motion vec → blend with current SR via small disocclusion-gated head
- Disocclusion mask: depth disparity + motion-vec magnitude
- Loss: `L1 + 0.1·SSIM + 0.1·LPIPS-VGG + λ·temporal-consistency` (warp t→t+1, penalize delta)
- Init: warm-start from v4 step-385K weights, freeze first N steps to stabilize the warp head
- Dataset mix: TartanAir Easy (~600 GB extracted, real flow + depth) + Sintel (real flow + depth, validation/fine-tune)
- Training time: ~12–16 hours on RTX 3080 Ti
- Expected uplift (per FSR 2 / DLSS 2 literature): +2–4 dB PSNR, −30 to −50% LPIPS

#### v5-gaussian-temporal (research track)

**Persistent 2D Gaussian field as temporal scene memory.** Higher ceiling, real research risk.

- Architecture: per-frame Gaussian fitter (warm-started from prev frame's Gaussians); multi-frame transformer attending over Gaussian tokens (3–5 prev frames); densification under disocclusion
- Why Gaussians for temporal: analytical sub-pixel warping (no resample blur compounding), continuous representation with persistent positions, tractable token count vs pixel-attention (~5K Gaussians vs millions of pixels per frame)
- Why it might not work: per-frame fitter cost, refit drift / temporal flicker, differentiable insertion is delicate, no shipping precedent for production-grade Gaussian temporal SR
- Loss: rendered output `L1 + SSIM + LPIPS` + temporal consistency + Gaussian regularization (anti-collapse)
- Init: cold-start from V0.5 splat infrastructure
- Training time: ~24–48 hours, harder convergence
- Foundation literature: GaussianSR, GS-STVSR, 4D Gaussian Splatting (2024–2026 papers); no production deployments yet

**Pivotal point:** the V0.5 single-frame Gaussian SR experiment failed because Gaussians' unique advantages (analytical warping, sub-pixel persistence, densification under occlusion change) require multi-frame context to express. The single-frame negative result is not direct evidence against temporal Gaussians — they're testing different hypotheses.

Both tracks train on the same dataset mix and evaluate on the same fixed held-out batch. **Whichever wins ships as v5. The other becomes parallel research input for v6.**

### Mid-term: performance pass

Once v5 quality is locked:

1. **Distill to lite tier** (~150K params target, ~4× cost reduction)
2. **Custom CUDA mega-kernel for NVIDIA** — single fused dispatch, weights resident in shared memory, tensor-core-resident MMA. Target: ~2–5 ms at 1080p→4K on RTX 3080 Ti. This is the architecture DLSS, XeSS-XMX, and FSR 4 (ML path) all use.
3. **HIP equivalent for AMD desktop** (CDNA / RDNA 3+ matrix cores)
4. **Metal + CoreML for Apple Silicon** (ANE-resident compute where layers fit)
5. **Vulkan compute fallback** for Steam Deck and any remaining target — hand-written kernels, FSR-2-style portability
6. **Bump model capacity** once latency budget has headroom — quality push back to the ceiling temporal+control unlocks

### Long-term: integration

- **DXGI hook + NGX shim** (Windows DLSS-API-compatible swap-in for any DX12 game already shipping DLSS 2/3) — initial validation target Cyberpunk 2077 (no anti-cheat, well-documented hook patterns)
- **Vulkan layer** (Linux, Steam Deck Proton)
- **Metal frame interception** (CrossOver / native macOS games)
- **OSS-FX α-conditioned frame extrapolation** — once temporal infrastructure exists, frame extrapolation is a near-free byproduct of the same warp pass at fractional time

---

## Cross-vendor inference strategy

Three options ordered by engineering cost and performance ceiling:

| Path | NVIDIA | AMD | Apple | Intel | Steam Deck | Floor / ceiling |
|---|---|---|---|---|---|---|
| **(1) ONNX everywhere** | TensorRT | DirectML / MIGraphX | CoreML | OpenVINO | Vulkan EP / NCNN | Cheapest; 3–5× off peak |
| **(2) Portable shaders** | HLSL / SPIR-V | HLSL / SPIR-V | MSL | HLSL / SPIR-V | SPIR-V | Misses tensor cores; 2–4× off peak |
| **(3) Per-vendor specialists** | CUDA + CUTLASS + `wmma`/TMA | HIP + MFMA / WMMA | Metal + ANE/AMX | XMX via Level Zero | HIP DP4a / Vulkan compute | **Peak performance** |

**(1) is what we ship today.** PyTorch → ONNX → TensorRT FP16 is the production path on NVIDIA. DirectML / MIGraphX / CoreML / OpenVINO ports are scaffolded but not yet validated.

**(3) is the long-term destination.** A single fused kernel per vendor is what every shipped ML upscaler does (DLSS, XeSS-XMX, FSR 4 ML path). Engineering cost is high but we're committing the time.

**(2) is a trap on its own** — portable shader languages can't access tensor cores or matrix units, so peak performance is bounded above by ~25% of (3) on tensor-equipped hardware. Useful as a fallback path on hardware without matrix accelerators (Steam Deck), not as the primary deployment.

---

## Component status

| Component | Path | Status |
|---|---|---|
| **OSS-SR pixel upscaler** (CNN backbone) | `oss/sr/cnn.py`, `oss/sr/inference.py` | ✓ v4 trained, ONNX + TRT FP16 exported |
| **OSS-SR-temporal-pixel** (v5 control track) | `oss/sr/temporal/` | ⏳ design phase |
| **OSS-Gaussian-temporal** (v5 research track) | `oss/gaussian/canvas/`, `oss/gaussian/network/` | ⏳ design phase, V0.5 splat infrastructure available |
| **OSS-RG ray-reconstruction denoiser** | `oss/regen/` | 🔬 architecture validated, training blocked on NoiseBase data download |
| **Engine-aliased LR synthesis** | `oss/gaussian/data/lr_synthesis.py` | ✓ jitter + TAA blur + JPEG validated |
| **Training dashboard** | `scripts/training_dashboard.py` | ✓ live, polled remote runs |
| **DXGI hook / NGX shim** | (planned) | ❌ not started |
| **Vulkan layer (Linux / Deck)** | (planned) | ❌ not started |
| **Metal frame interception (macOS)** | `oss/gaussian/ports/metal/` | ✓ scaffolded; not wired to real games |
| **Custom CUDA mega-kernel** | (planned) | ❌ post-v5 |

---

## Hardware tiers (target, post-distillation)

| Tier | Params | Target hardware | Deploy backend |
|---|---|---|---|
| Pico | ~150K | Steam Deck, integrated GPUs, GTX 10/16 | Vulkan compute / NCNN |
| Lite | ~600K | RTX 20+, RDNA 2+ | TRT FP16 / DirectML / CoreML |
| Standard (current `v4`) | ~605K | RTX 30+, RDNA 3+, M3 Max | TRT FP16 / MIGraphX / CoreML |
| Heavy | ~2–5M | RTX 4080+, RX 9070 XT+ | TRT FP16 / custom CUDA |

Quality modes (planned):

| Mode | Internal res |
|---|---|
| Ultra Performance | 33% |
| Performance | 50% |
| Balanced | 59% |
| Quality | 67% |
| Ultra Quality | 77% |
| OSAA (anti-aliasing only) | 100% |

---

## Training data

- **SRGD** (Game Engine Data, ~51K HR frames across 17 scenes) — current v3 / v4 training, sequential frames, **no real G-buffers** (zeros — temporal gains pending)
- **TartanAir** (~600 GB extracted, 18 environments × Easy split) — real depth, real optical flow, sequential trajectories. Primary v5 temporal pretraining set.
- **Sintel** — real depth (`.dpt`), real flow (`.flo`), sequential frames. Validation + fine-tune. Already extracted.
- **NoiseBase** — primary planned data for OSS-RG. Download blocked on remote bandwidth allocation.
- **Vimeo-90K** — OSS-FX real-world motion diversity (planned).

We default to existing public datasets. Custom captures are last resort.

---

## Lab-notebook discipline

Every training run, ablation, and benchmark gets a memo in `docs/superpowers/experiments/YYYY-MM-DD-<slug>.md` **before** its result is allowed to drive a decision. Paper drafts live in `docs/papers/`. Track plans for SR-CNN, Gaussian-RR, and Gaussian-Temporal live in `docs/superpowers/`.

This is enforced — see [lab-notebook-discipline](docs/papers/lab-notebook-discipline.md). It exists because we have shipped ourselves into the wrong-conclusion swamp before, and the cost of one written paragraph before a result is far less than the cost of an unmoored decision.

---

## License

- SDK and shaders: **Apache-2.0**
- Plugins: **MIT**
- Model weights: **CC-BY-4.0**

---

## What we won't use

- NRD (RTX SDK SLA)
- DLSS / FSR / XeSS decompiled binaries or leaked weights
- `tiny-cuda-nn` (CUDA-only — defeats vendor-agnosticism)
- Quixel / Megascans / CC-BY-NC-* training assets

---

## Repository

Active development branch: `v0.2-dev`. Track docs:

- [oss-sr-cnn-track.md](docs/superpowers/oss-sr-cnn-track.md) — single-frame pixel upscaler, current production candidate
- [oss-gaussian-rr-track.md](docs/superpowers/oss-gaussian-rr-track.md) — Gaussian-based ray-reconstruction denoiser
- [oss-gaussian-temporal-track.md](docs/superpowers/oss-gaussian-temporal-track.md) — Gaussian temporal canvas, parent track for v5-gaussian-temporal

Recent decision memos in [docs/superpowers/experiments/](docs/superpowers/experiments/) — chronological log of what we tried, what worked, what didn't, and why.
