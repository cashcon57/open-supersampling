# OpenSuperSampling (OSS)

> Vendor-agnostic open-source real-time game super-resolution and frame extrapolation.

**Status:** Pre-alpha / active research. v4 single-frame upscaler trained and exported. v5-pixel-temporal warm-start in flight as Stage 3 validation; v5-Gaussian-temporal staged validation (Stage 0 → 1 → 2 → 3) queued via watchdog v4 to run as soon as v5-pixel finishes. **v6 canonical architecture locked**: covariance-resampled online Gaussian-temporal SR with HAT spatial backbone + cross-attention to Gaussian canvas + score-based active pruning + custom kernels per vendor + DLL-shim integration (no game-developer cooperation needed for any DLSS/FSR/XeSS-supporting title). Frame extrapolation (OSS-FX) ships free as the same canvas rendered at α<1. Three model tiers (Pico / Standard / Heavy) — same architecture, scaled. Sprint 7 community **OSS Capture Tool** in tandem build — four bandwidth tiers (trickle / lite / regular / INSANE) for opt-in training-data contribution. Not yet suitable for production use.

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

- **v5 dual-track temporal — implementation complete**
  - **Pixel-temporal** (`oss/sr/temporal/`): warp + disocclusion gate + temporal head + stateless ONNX wrapper, full model + dataset + inference engine landed
  - **Gaussian-temporal** (`oss/sr/gaussian_temporal/`): persistent Gaussian field + analytical warp + g-buffer encoder + multi-frame transformer + densify/prune + raster wrapper + regularization, full model + dataset + inference engine landed
  - Test suite: **113 passed, 1 skipped** as of 2026-05-04
  - Pixel-temporal training launched 2026-05-04 18:07 CDT on `<train-host>` (PID 21192), warm-started from v4 step-385K, ETA ~22:50 CDT tonight; held-out eval scheduled for the morning of 2026-05-05
  - Gaussian-temporal training queued sequentially behind the pixel run (shared 3080 Ti, sequential unless overlap-safety is verified)

- **Inference pipeline**
  - **PyTorch FP16 + channels-last + CUDA Graphs** — runtime-optimized engine in `oss/sr/inference.py` (single-frame), `oss/sr/temporal/` (pixel-temporal), `oss/sr/gaussian_temporal/` (Gaussian-temporal)
  - **ONNX export** with dynamic axes (bilinear-skip ONNX path; bicubic-skip-with-antialias is currently un-exportable on PyTorch 2.4.1 / opset 17 — measured quality delta is −0.003 dB, negligible). Pixel-temporal stateless ONNX wrapper landed; export design memo in [notes/2026-05-04-v5-pixel-temporal-onnx-export-design.md](docs/superpowers/notes/2026-05-04-v5-pixel-temporal-onnx-export-design.md).
  - **TensorRT FP16** with narrow optimization profiles per resolution
  - **TensorRT INT8 PTQ** — quality gate passes, but performance regresses vs FP16 on RTX 3080 Ti at most kernel shapes (one win at 900p input). Path documented but not the primary deployment target on Ampere.

- **Tooling**
  - Live training dashboard at `scripts/training_dashboard.py` — Chart.js panels for PSNR/LPIPS/loss/SSIM, polls `metrics.json` and `score_log.json`, downsamples large series
  - Lab-notebook-discipline experiment memos in `docs/superpowers/experiments/` — every result that drives a decision is documented before it does

### Architecture (v6 canonical, locked 2026-05-05)

**v6 = covariance-resampled online Gaussian-temporal SR.** v5-pixel and v5-Gaussian-temporal are validation steps; v6 is the actual ship target. Full memo at [experiments/2026-05-05-v6-architecture-canonical.md](docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md). Worked-out math at [research/2026-05-05-gaussian-temporal-research-deep-dive.md](docs/research/2026-05-05-gaussian-temporal-research-deep-dive.md).

```text
                                    ┌─────────────────────────┐
  current LR + G-buffers ──────────►│ HAT-Base spatial backbone│──► coarse SR features
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

   For frame extrapolation (OSS-FX): rasterize canvas at α ∈ (0, 1) instead of α = 1.
   Cost: one in-place add to position tensor. Free.
```

The architectural moat (why pixel-grid SR — DLSS, FSR, XeSS — provably cannot match this on splat content):

| Technique | What it gives us | What pixel-grid methods can do |
|---|---|---|
| **Covariance resampling** (GS-STVSR, 2026) | Anti-shimmering by mathematical construction — `Σ'_output = J_t·Σ_t·J_t^⊤ + Σ_recon` matched to target resolution | post-hoc filtering only — can't reshape reconstruction kernel pre-emptively |
| **Spatial-Temporal Variation Score pruning** (4DGS-1K, NeurIPS 2025) | 14-34× rendering speedup at <0.3 dB quality cost | n/a — no per-primitive contribution score |
| **Persistent canvas with analytical sub-pixel warp** | No resample-blur compounding; densification under disocclusion is exact | bilinear/bicubic warp blurs every cycle; disocclusion fill is heuristic |
| **Same canvas at α<1 = OSS-FX** | Frame extrapolation is one in-place add to position tensor | requires separate frame-generation network (DLSS-FG style, hallucination-prone) |

### Measured inference latency (RTX 3080 Ti, TensorRT FP16, narrow profile, single-frame v4)

| Input → Output | Median latency |
|---|---|
| Steam Deck 800p | 18.6 ms |
| 720p → 1440p | 15.6 ms (~64 fps headroom) |
| 1080p → 4K | 37.6 ms (~27 fps headroom) |

These numbers are honest and current. **The deliberate comparison is FSR 2/3 at ~0.7–1 ms** (hand-tuned compute shaders, no ML). We are roughly 20–40× slower than the dominant non-ML upscaler in the same quality tier. Closing that gap is the second-half roadmap below. v5 temporal latency will be measured once a checkpoint hits the held-out gate.

### Known limits

- **Temporal training is in flight, not yet shipped.** v3 and v4 trained with zero G-buffers in SRGD (depth/motion/normals were placeholder zeros) — temporal gains were unrealized at the v4 ship point. v5 implementation is complete and pixel-temporal is currently training on TartanAir with real flow + depth, but no v5 quality numbers exist yet, no v5 ship decision has been made, and the Gaussian track has not started training.
- **Inference cost dominates.** ML-based upscaling on a model this size is fundamentally heavier than FSR 2/3 shader passes. Bridging requires custom kernels per vendor (designed, see Sprint 6 prep), distillation to lite/pico tier, and likely sparse / tile-based execution.
- **Steam Deck not yet viable.** RDNA 2 has no matrix accelerators. Until pico tier + custom Vulkan compute kernels land, Deck will fall back to FSR 2 in the runtime.
- **Not deployment-ready.** No game integration shipped. No NGX/DXGI hook in production. Inference engines work in isolation; runtime plumbing is the Sprint 7 design (see [notes/2026-05-04-s7-game-integration-design.md](docs/superpowers/notes/2026-05-04-s7-game-integration-design.md)).
- **HDR support is partial: shippable at degraded quality, full HDR-trained quality scheduled for v6.1.** As of `694a0f3` the model uses a softplus output activation that accepts and produces unbounded non-negative linear-light values, so HDR input/output flows through architecturally without clipping. However, the training corpus (TartanAir, Hypersim, SRGD) is all 8-bit sRGB, so HDR-specific patterns (sun discs, neon, specular highlights, BT.2020 wide-gamut colors) are under-represented in what the model has learned. Expected HDR quality is roughly 70-80% of SDR quality on the same content class — better than bicubic, worse than DLSS HDR. Full HDR competence (v6.1) requires retraining with HDR-encoded data via INSANE-mode capture of HDR-rendered games and re-rendered Hypersim in linear scRGB.

---

## Roadmap

### Sprint progression

| Sprint | Theme | Status | Exit gate |
|---|---|---|---|
| **S1–S3** | Renderer scaffolding, hooks, tile classifier | ✓ done | components scaffolded + tests pass |
| **S4** | Single-frame SR-CNN trained, ONNX/TRT export | ✓ done (v3 + v4 shipped, A/B confirms v4 real) | v4 beats v3 on fixed-batch held-out |
| **S5** | v5 temporal validation tracks (pixel + Gaussian) | ⏳ pixel warm-start in flight; Gaussian staged validation queued via watchdog v4; both feed v6 not their own ship | each architecture's convergence verified |
| **S6 / v6** | **Covariance-resampled online Gaussian-temporal SR + 3-tier distillation + DLL-shim runtime** (current sprint) | 🚧 architecture locked, validation chain armed; full design at [experiments/2026-05-05-v6-architecture-canonical.md](docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md) | v6-Heavy beats DLSS 3.5 quality on held-out OR matches DLSS 4 on glassbench; Pico runs <3 ms on Steam Deck via Vulkan compute |
| **S7** | Game integration: DXGI hook + NGX/FSR/XeSS shim + Vulkan layer + OSS-FX | 📐 design memo landed; runtime build queued after v6 model ships | one DLL drop into a DLSS-supporting game produces upscaled output |
| **S7-data** | **OSS Capture Tool** — community training-data pipeline (one-click-install per-game DLL + 4 bandwidth modes + auto-upload + auto-delete) | 📐 design memo landed; tandem implementation in progress (Claude server-side, Codex client-side); burst-mode + 4-tier mode presets (trickle / lite / regular / INSANE) wired into schema | first contributor frame uploaded end-to-end through hosted ingest |
| **Custom kernels** (parallel to S6 + S7) | Per-vendor specialists: CUDA + CUTLASS, HIP + rocWMMA, Metal + MPS / MLX, Level Zero + XMX, Vulkan compute | 📐 design memos landed (CUDA mega-kernel, vendor audit); implementation queued after v6-Heavy model exists | each backend hits its inference budget at SR + FX |

### Sprint 5 — current sprint

The current v4 is the strong single-frame baseline. Quality is now bottlenecked by missing temporal accumulation. We're racing two architectures so we have a fallback if the experimental path doesn't pan out.

**Phase 0 (done 2026-05-04):**
- Fixed-batch A/B v3 vs v4 on CitySample held-out — v4 wins LPIPS 64/64 (-22%), PSNR tied. v4 is a real perceptual improvement.
- TartanAir extraction completed on the remote 3080 Ti host (72 zips, ~600 GB extracted, primary temporal training data)
- v5-pixel-temporal + v5-gaussian-temporal design specs written: [pixel](docs/superpowers/specs/2026-05-04-v5-pixel-temporal-design.md) | [Gaussian](docs/superpowers/specs/2026-05-04-v5-gaussian-temporal-design.md)

**Phase 1 (done 2026-05-04): implementation complete**
- Implementation plans landed under `docs/superpowers/plans/`
- Pixel modules landed under `oss/sr/temporal/` (warp, disocclusion gate, temporal head, model, dataset, inference engine, stateless ONNX wrapper, held-out manifest)
- Gaussian modules landed under `oss/sr/gaussian_temporal/` (Gaussian field, analytical warp, g-buffer encoder, multi-frame transformer, densification, pruning, rasterizer wrapper, regularization, full model, dataset, inference engine)
- Sequential frame pair / multi-frame window dataset loaders landed for TartanAir/Sintel-shaped data
- Local SR test suite is green after Codex review fixes (`113 passed, 1 skipped` as of 2026-05-04)

**Phase 2 (in progress): training**
- v5-pixel-temporal currently running on `<train-host>` (python PID 2732 as of last bounce), warm-started from v4 step-385K. ETA ~05:00 CDT tomorrow (2026-05-05).
- **Held-out env policy:** training excludes TartanAir env `oldtown` via `--held-out-envs oldtown`; the held-out manifest at `<train-host-data>/checkpoints/v5_held_out_manifest.json` draws all 64 pairs from `oldtown` only. Closes a data-leak gap from the original launch (which iterated the full TartanAir Easy split).
- **LR-synth distribution match:** training, eval, and viz all run through `EngineAliasedLRSynth` with shared config (`enable_jitter=True, enable_taa_blur=True, enable_jpeg=False, jpeg_quality=85, blur_sigma=0.5`). Earlier versions of the train script left LR-synth off, training the model on too-clean LR vs the engine-aliased LR seen at eval; that's now fixed.
- **Sintel:** `training/depth` package downloaded + extracted + junctioned today; v1 launch keeps `--sintel-root` off pending dual-manifest support, with a Sintel fine-tune follow-up runbook ready for after the main run.
- Gaussian training is queued until the pixel run completes, per the sequential-GPU directive ("sequential unless overlap is safe; test overlap first") for the shared RTX 3080 Ti.
- Live in-flight viz at `http://<tailnet-ip>:8080/` — 6-up `LR-bilinear · bicubic · v4-baseline · v5-temporal · GT · |err| heatmap` strip with timeline scrubber, plus PSNR/LPIPS charts annotated with bicubic/FSR 1-4/DLSS 2-4 published-benchmark reference lines.

**Phase 2.5 (in flight): definitive comparison harness**
- Codex C16 (`docs/superpowers/notes/2026-05-04-claude-codex-asks-r5.md`) implementing `scripts/sr_v5_race_compare.py` — multi-trajectory paired-Wilcoxon test across PSNR/LPIPS/temporal-stability + latency gate (Gaussian must explicitly beat pixel on ≥3/4 metrics, p<0.05, with latency ≤1.5× pixel). Auto-determines verdict against the spec race rule.

**Phase 3: comparison + ship decision**
- Same fixed held-out batch (TartanAir now; Sintel after the Depth subset is fetched or a tested no-depth fallback exists)
- Success criteria gates per spec (PSNR, LPIPS, temporal stability, latency)
- Whichever wins ships as v5; the other becomes v6+ research input
- **No v5 ship decision has been made.** Numbers must clear the held-out gate first.

### Sprint 5 dual-track details

#### v5-pixel-temporal (control track)

**FSR 2-class temporal warp+blend.** Proven recipe, bounded risk, predictable result.

- Architecture: warp prev-frame HR output by motion vec → blend with current SR via small disocclusion-gated head
- Disocclusion mask: depth disparity + motion-vec magnitude
- Loss: `L1 + 0.1·SSIM + 0.1·LPIPS-VGG + λ·temporal-consistency` (warp t→t+1, penalize delta)
- Init: warm-start from v4 step-385K weights, freeze first N steps to stabilize the warp head
- Dataset mix: TartanAir Easy (~600 GB extracted, real flow + depth) now; Sintel validation/fine-tune after Sintel Depth is fetched
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

Both tracks train on the same dataset mix and evaluate on the same fixed held-out batch once datasets are complete. **Whichever wins ships as v5. The other becomes parallel research input for v6.** Current status: pixel training in flight, Gaussian queued, results pending — no v5 ship decision has been made.

### Mid-term: performance pass (Sprint 6)

Once v5 quality is locked. S6 design work has already landed (see [Reference docs](#reference-docs)):

1. **Distill to pico tier** (~150K params target, ~4× cost reduction) — design memo: [notes/2026-05-04-pico-distillation-design.md](docs/superpowers/notes/2026-05-04-pico-distillation-design.md)
2. **Custom CUDA mega-kernel for NVIDIA** — single fused dispatch, weights resident in shared memory, tensor-core-resident MMA. Target: ~2–5 ms at 1080p→4K on RTX 3080 Ti. Design: [notes/cuda-mega-kernel-design.md](docs/superpowers/notes/cuda-mega-kernel-design.md). This is the architecture DLSS, XeSS-XMX, and FSR 4 (ML path) all use.
3. **HIP equivalent for AMD desktop** (CDNA / RDNA 3+ matrix cores) — covered in [notes/vendor-optimization-audit.md](docs/superpowers/notes/vendor-optimization-audit.md)
4. **Metal + CoreML for Apple Silicon** (ANE-resident compute where layers fit)
5. **Vulkan compute fallback** for Steam Deck and any remaining target — hand-written kernels, FSR-2-style portability
6. **Bump model capacity** once latency budget has headroom — quality push back to the ceiling temporal+control unlocks

### Long-term: integration (Sprint 7) — DLL shim, no dev cooperation needed

Design memo: [notes/2026-05-04-s7-game-integration-design.md](docs/superpowers/notes/2026-05-04-s7-game-integration-design.md).

The killer integration property: **OSS-SR works on hundreds of AAA games via DLL drop-in** because every game using DLSS / FSR / XeSS already provides depth + motion vectors + jitter via stable, documented APIs. We shim those DLLs.

| Game already supports | We shim | Quality |
|---|---|---|
| DLSS 2 / 3 / 4 (NVIDIA NGX) | `nvngx_dlss.dll` masquerade | best — full payload from game |
| FSR 2 / 3 (AMD FidelityFX) | `ffx_fsr2_*.dll` masquerade | best |
| XeSS (Intel) | `libxess_*.dll` masquerade | best |
| TAA only (older games) | DXGI resource intercept + heuristics | medium (tier 2 in `oss/model/oss_fx_warp.py`) |
| Custom AA / no temporal SR | DXGI intercept + on-the-fly RAFT-Small flow | lower (tier 3 fallback) |

Day-1 candidate games (DLSS-supporting + no kernel anti-cheat): Cyberpunk 2077, Alan Wake 2, Hogwarts Legacy, Starfield, Baldur's Gate 3, Returnal, Hellblade II, Forza Horizon 5, Ghost of Tsushima Director's Cut, Black Myth: Wukong — and 200+ more. **Drop the OSS DLL into the game directory, restart, you're upscaling with OSS instead of DLSS.**

Off-limits permanently: kernel anti-cheat (Vanguard, BattlEye, EAC, Ricochet) — DLL injection trips them, ban risk.

Other integration paths:
- **Vulkan layer** (Linux, Steam Deck Proton)
- **Metal frame interception** (CrossOver / native macOS games)
- **OSS-FX α-conditioned frame extrapolation** — same v6 trained canvas, rasterized at α<1. Free byproduct, no separate frame-generation network needed (cf. DLSS-FG which is a heavy separate ML pass).

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
| **OSS-SR pixel upscaler** (CNN backbone, single-frame) | `oss/sr/cnn.py`, `oss/sr/inference.py` | ✓ v4 trained, ONNX + TRT FP16 exported |
| **OSS-SR-temporal-pixel** (v5 control track) | `oss/sr/temporal/` (warp, disocclusion, temporal head, model, dataset, inference engine, stateless ONNX wrapper) | ✓ implementation complete; ⏳ training in flight on `<train-host>` |
| **OSS-Gaussian-temporal** (v5 research track) | `oss/sr/gaussian_temporal/` (field, analytical warp, encoder, transformer, densify, prune, rasterizer, regularization, model, dataset, inference engine) + `oss/gaussian/canvas/`, `oss/gaussian/network/` | ✓ implementation complete; ⏳ training queued behind pixel |
| **OSS-RG ray-reconstruction denoiser** | `oss/regen/` | 🔬 architecture validated, training blocked on NoiseBase data download |
| **Engine-aliased LR synthesis** | `oss/gaussian/data/lr_synthesis.py` | ✓ jitter + TAA blur + JPEG validated |
| **Training dashboard** | `scripts/training_dashboard.py` | ✓ live, polled remote runs |
| **DXGI hook / NGX shim** | (planned, S7) | 📐 design memo landed |
| **Vulkan layer (Linux / Deck)** | (planned, S7) | 📐 design memo landed |
| **Metal frame interception (macOS)** | `oss/gaussian/ports/metal/` | ✓ scaffolded; not wired to real games |
| **Custom CUDA mega-kernel** | (planned, S6) | 📐 design memo landed |
| **Pico-tier distillation** | (planned, S6) | 📐 design memo landed |
| **OSS Capture Tool — DLL** (game-side hook, capture mode) | `oss/gaussian/interception/` (planned) | 📐 design memo landed; Codex implementing C18 |
| **OSS Capture Tool — uploader** (client-side daemon) | `oss/capture/uploader.py` (planned) | 📐 design memo landed; Codex implementing C19 |
| **OSS Capture Tool — ingest server** (FastAPI + R2) | `server/oss_capture_ingest/` (planned) | 📐 design memo landed; Claude implementing in subagent |

---

## Hardware tiers (v6 — same architecture, scaled)

All three tiers share the v6 covariance-resampled Gaussian-temporal architecture (HAT spatial backbone + persistent Gaussian canvas + cross-attention + score-based pruning). Distillation cascades **Heavy → Standard → Pico**. Same training data, same loss, same architecture, just sized.

| Tier | Backbone | Canvas size | Target hardware | Inference budget | Backend |
|---|---|---|---|---|---|
| **Pico** | HAT-Tiny (~1M) | ~1-2K Gaussians | Steam Deck, integrated GPUs, mobile dGPU | ~3 ms at 720p→1080p | hand-tuned Vulkan compute (no matrix accel needed) |
| **Standard** | HAT-Small (~5M) | ~5K Gaussians | RTX 30+, RX 6700+, Arc, M2+ | ~5 ms at 1080p→1440p | custom CUDA / HIP / Metal / Level Zero kernel per vendor |
| **Heavy** | HAT-Base (~15M) | ~15K Gaussians | RTX 4080+, RX 7900+, M4 Max | ~10 ms at 1440p→4K | same per-vendor kernel path |

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

## OSS Capture Tool — contribute training data while you play

OSS gets better when it trains on real games. The OSS Capture Tool is a small DLL that drops into a supported game, captures rendered frames + engine G-buffers while you play, and ships them to the project's training-data bucket. Frames are deleted from your disk as soon as the server confirms receipt.

You set the bandwidth budget. You control which games are enabled. You can pause or uninstall at any time.

### Capture modes

Pick the mode that matches what you're willing to spend on bandwidth. Default is **lite** because the 99% case shouldn't have to think about it.

| Mode | Bandwidth | Who it's for | What you give up |
|---|---|---|---|
| **trickle** | ~100 MB/h | Anyone who doesn't want to notice it. Maximum-density data inside an invisible budget — full G-buffers, not stripped-down. | No long temporal sequences. Sparse capture (one short burst every 5–20 min, fired only when the camera is settled). |
| **lite** *(default)* | ~500 MB/h | Most contributors. The sweet spot for v5 temporal training — short pairs every 80 s + a 60-frame long sequence every 30 min. | No material BRDF channels (albedo / roughness / metallic). |
| **regular** | ~2 GB/h | Anyone with uncapped fiber who wants to maximize training value per hour. Adds material-aware channels (albedo + roughness) and denser bursts. | Higher network bill if you're metered. |
| **INSANE** | ~20–50 GB/h | Data warriors with high-end GPUs + uncapped uplink who want to contribute the data that lets OSS exceed DLSS. Full PBR + 4-second long bursts + automatic supersample ground truth. | The supersample-GT pass briefly stutters the game when the camera is settled. Documented in the install consent dialog. |

Mode is set at install time. Per-install live mode-switching (tray-icon menu) is on the v1+ list.

### What gets captured

- The game's rendered low-resolution frame and the upscaled high-resolution frame (when available), plus engine G-buffers: depth, motion vectors, surface normals.
- In **regular**+: surface albedo + roughness for material-aware temporal training.
- In **INSANE**: full PBR channels (metallic, emissive), FP32 depth/motion, DLAA captures, every-DLSS-mode pairing, scene-cut bursts, and an automatic 256-frame supersample ground truth on settled cameras.

Each captured frame carries metadata identifying the game, game version, capture mode, resolution, jitter offset, motion magnitude, and an opaque per-install token. No personally identifying information is in the metadata.

### What does NOT get captured

- No audio.
- No keyboard / mouse / controller input.
- No other windows. No desktop. Only the supported game's rendered output and its engine buffers.
- No webcam, no microphone, no chat, no save data, no network traffic.

### Network behavior

- Hard bandwidth cap per mode. The uploader respects backoff and retries; never hammers the server.
- Frames are deleted from disk immediately after the server confirms receipt. No long-term local storage.
- Per-game opt-in. You install for one game at a time.
- Tray-icon menu lets you pause uploads or uninstall any time.

### Anti-cheat — supported games only

We only support games where capture won't get you banned. The supported-games list is editorial — maintained by the project, not auto-detected. Don't try to install on a game outside the list; the installer won't let you, and circumventing that is not supported.

Initial validation target: Cyberpunk 2077 (no anti-cheat, well-documented hook patterns). Additional titles will be added as their hook patterns are validated and their EULAs reviewed.

### Where the data goes

- A FastAPI ingest server validates each frame's metadata schema, deduplicates by content hash, rate-limits per token + per game, then writes to Cloudflare R2 with the layout `<game_id>/<YYYY-MM>/<capture_mode>/<session_uuid>/<frame_uuid>.exr`.
- Contributor identity is a one-time opaque token. No account, no email, no PII.
- The dataset card publishes per-mode contribution counts so you can see exactly what your bytes did.
- All trained model weights derived from contributed data ship under **CC-BY-4.0** alongside the rest of OSS.

### Install

Installer is in build. First public contributor session lands with the S7-data exit gate (first end-to-end contributed frame). Track progress in [specs/2026-05-04-oss-capture-tool-design.md](docs/superpowers/specs/2026-05-04-oss-capture-tool-design.md).

---

## Training data

- **SRGD** (Game Engine Data, ~51K HR frames across 17 scenes) — current v3 / v4 training, sequential frames, **no real G-buffers** (zeros — temporal gains pending)
- **TartanAir** (~600 GB extracted, 18 environments × Easy split) — real depth, real optical flow, sequential trajectories. Primary v5 temporal pretraining set; pixel-temporal is currently training against it.
- **Sintel** — real depth (`.dpt`), real flow (`.flo`), sequential frames. Validation + fine-tune. Image+flow already extracted; the separate `training/depth` package is pending fetch (tracked in the pixel runbook).
- **NoiseBase** — primary planned data for OSS-RG. Download blocked on remote bandwidth allocation.
- **Vimeo-90K** — OSS-FX real-world motion diversity (planned).

We default to existing public datasets. Custom captures are last resort.

---

## Reference docs

Design memos and runbooks driving the current sprint and the next two:

**v5 (current sprint) — specs and runbooks:**
- [specs/2026-05-04-v5-pixel-temporal-design.md](docs/superpowers/specs/2026-05-04-v5-pixel-temporal-design.md) — pixel-temporal architecture spec
- [specs/2026-05-04-v5-gaussian-temporal-design.md](docs/superpowers/specs/2026-05-04-v5-gaussian-temporal-design.md) — Gaussian-temporal architecture spec
- [specs/2026-05-01-gaussian-temporal-canvas-design.md](docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md) — earlier Gaussian canvas design that fed into the v5 research track
- [notes/2026-05-04-v5-pixel-temporal-runbook.md](docs/superpowers/notes/2026-05-04-v5-pixel-temporal-runbook.md) — pixel training runbook
- [notes/2026-05-04-v5-pixel-launch-status-r2.md](docs/superpowers/notes/2026-05-04-v5-pixel-launch-status-r2.md) — current pixel training status
- [notes/2026-05-04-v5-pixel-sintel-finetune-runbook.md](docs/superpowers/notes/2026-05-04-v5-pixel-sintel-finetune-runbook.md) — Sintel fine-tune plan after main run
- [notes/2026-05-04-v5-gaussian-temporal-runbook.md](docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md) — Gaussian training runbook (queued)
- [notes/2026-05-04-v5-pixel-temporal-onnx-export-design.md](docs/superpowers/notes/2026-05-04-v5-pixel-temporal-onnx-export-design.md) — stateless ONNX wrapper design for the pixel track
- [notes/2026-05-04-v5-rolling-review.md](docs/superpowers/notes/2026-05-04-v5-rolling-review.md) — running notes against the v5 implementation

**S6 prep (performance pass):**
- [notes/vendor-optimization-audit.md](docs/superpowers/notes/vendor-optimization-audit.md) — survey of per-vendor primitive paths
- [notes/cuda-mega-kernel-design.md](docs/superpowers/notes/cuda-mega-kernel-design.md) — single fused-dispatch CUDA kernel design
- [notes/2026-05-04-pico-distillation-design.md](docs/superpowers/notes/2026-05-04-pico-distillation-design.md) — pico-tier distillation memo

**S7 prep (game integration):**
- [notes/2026-05-04-s7-game-integration-design.md](docs/superpowers/notes/2026-05-04-s7-game-integration-design.md) — DXGI / NGX / Vulkan / Metal / OSS-FX integration design

**v6 architecture (current sprint, locked 2026-05-05):**
- [experiments/2026-05-05-v6-architecture-canonical.md](docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md) — **canonical v6 design**: covariance-resampled online Gaussian-temporal SR with HAT spatial backbone + cross-attention + Spatial-Temporal Variation Score pruning + custom kernels per vendor + DLL-shim integration. Three tiers (Pico / Standard / Heavy) — same architecture, scaled. Frame extrapolation (OSS-FX) free as α<1 canvas render.
- [research/2026-05-05-gaussian-temporal-research-deep-dive.md](docs/research/2026-05-05-gaussian-temporal-research-deep-dive.md) — worked-out math of every technique v6 incorporates (3DGS rasterization foundation, deformation field vs native 4D fork, 4DGS-1K pruning algorithm, Gaussian Frosting, GRTX, GS-STVSR covariance resampling, glTF KHR_gaussian_splatting standardization, engine-integration plugins, summary table of 2024-2026 SOTA methods).
- [experiments/2026-05-05-v6-design-two-tier-distillation.md](docs/superpowers/experiments/2026-05-05-v6-design-two-tier-distillation.md) — **superseded** earlier same day; pixel-only design that retreated from the dual-track Gaussian commitment before the research synthesis was reckoned with. Retained for forensic value (loss recipe + training recipe + data plan all carry forward).
- [experiments/2026-05-05-v6-handheld-tier-deferred.md](docs/superpowers/experiments/2026-05-05-v6-handheld-tier-deferred.md) — **reversed** later same day; handheld tier (Steam Deck, integrated GPUs) is back in scope as the v6 Pico tier once custom Vulkan compute kernels became part of the project plan. Memo retained showing the brief defer-and-reinstate decision context.
- [specs/2026-05-04-v6-research-tracks-design.md](docs/superpowers/specs/2026-05-04-v6-research-tracks-design.md) — **superseded**; pre-research-synthesis race-resolution framing.

**OSS Capture Tool (community training data, S7-adjacent):**
- [specs/2026-05-04-oss-capture-tool-design.md](docs/superpowers/specs/2026-05-04-oss-capture-tool-design.md) — one-click-install per-game DLL, four capture modes (trickle ~100 MB/h · lite ~500 MB/h · regular ~2 GB/h · INSANE ~20–50 GB/h), burst-mode sampler (short pairs + long sequences), auto-upload + delete-immediately, FastAPI ingest + R2 layout, tandem implementation split
- [d3d12-hook-design.md](docs/superpowers/d3d12-hook-design.md) — parent DLL hook architecture (Detours/MinHook + NGX spoofing) shared with S7 inference shim

Sprint reference (high-level, predates the v5 implementation work):
- [specs/oss-gaussian-sprint-1.md](docs/superpowers/specs/oss-gaussian-sprint-1.md) through [specs/oss-gaussian-sprint-7.md](docs/superpowers/specs/oss-gaussian-sprint-7.md)

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
