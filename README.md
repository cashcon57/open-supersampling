# OpenSuperSampling (OSS)

> Vendor-agnostic open-source real-time game super-resolution and frame extrapolation.

**Status:** Pre-alpha / active research. Single-frame upscaler trained and exported. Sprint 5 dual-track temporal **implementation is complete** (113 v5 tests passing); pixel-temporal training is **in flight** on the 3080 Ti host, Gaussian-temporal queued behind it. Held-out results pending. Not yet suitable for production use.

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

### Architecture (v5)

The two v5 tracks share data plumbing and held-out eval but train independent networks.

**Pixel-temporal (control track):**

```text
                ┌──────────────────┐
  prev HR ───►──│ motion-vec warp  │──► warped prev ─┐
                └──────────────────┘                 │
                                                     ▼
  current LR ──► v4 backbone (frozen N steps) ──► single-frame SR ──► ┌──────────────────┐
                                                                      │  temporal head   │──► final HR
  depth + mvec ──────► disocclusion gate (mask) ──────────────────────►│ (gated blend)    │
                                                                      └──────────────────┘
```

**Gaussian-temporal (research track):**

```text
  current LR + G-buffers ──► g-buffer encoder ──► fitted Gaussians (token set)
                                                          │
                                  prev N Gaussian sets ───┤
                                                          ▼
                                          analytical sub-pixel warp
                                                          │
                                                          ▼
                                          multi-frame transformer
                                                          │
                                                          ▼
                                          densify (under disocclusion) + prune
                                                          │
                                                          ▼
                                              differentiable rasterizer ──► HR output
```

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

---

## Roadmap

### Sprint progression

| Sprint | Theme | Status | Exit gate |
|---|---|---|---|
| **S1–S3** | Renderer scaffolding, hooks, tile classifier | ✓ done | components scaffolded + tests pass |
| **S4** | Single-frame SR-CNN trained, ONNX/TRT export | ✓ done (v3 + v4 shipped, A/B confirms v4 real) | v4 beats v3 on fixed-batch held-out |
| **S5** | **v5 dual-track temporal** (current sprint) | ⏳ implementation complete; pixel training in flight on 3080 Ti; results pending closeout memo | one track meets success criteria, ships as v5 |
| **S6** | Performance pass: distill, custom CUDA mega-kernel, vendor ports | 📐 design memos landed (vendor audit, CUDA mega-kernel, pico distill, ONNX export); not yet started | TRT FP16 latency cut ≥3× |
| **S7** | Game integration: DXGI hook + NGX shim + Vulkan layer + OSS-FX | 📐 design memo landed; not yet started | runtime swap working in one DX12 title |

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
- v5-pixel-temporal launched 2026-05-04 18:07 CDT on `<train-host>` (python PID 21192), warm-started from v4 step-385K, ETA ~22:50 CDT tonight
- Pixel run is currently TartanAir-only because the remote Sintel tree is missing the separate `training/depth` package required by `SintelGaussianDataset`; runbook tracks the remediation
- Gaussian training is queued until the pixel run completes, per the sequential-GPU directive ("sequential unless overlap is safe; test overlap first") for the shared RTX 3080 Ti
- Checkpoints and metrics rolling under `<train-host-data>/checkpoints/srcnn-v5-pixel-temporal/`; held-out eval scheduled for the morning of 2026-05-05

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

### Long-term: integration (Sprint 7)

Design memo: [notes/2026-05-04-s7-game-integration-design.md](docs/superpowers/notes/2026-05-04-s7-game-integration-design.md).

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

**v6 research direction (post-v5-race candidate):**
- [specs/2026-05-04-v6-research-tracks-design.md](docs/superpowers/specs/2026-05-04-v6-research-tracks-design.md) — race-resolution gates, scenarios A/B, common productization, 6-month sequencing

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
