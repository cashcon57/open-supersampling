# OpenSuperSampling (OSS)

Open-source real-time super-resolution and frame extrapolation for games. Cross-vendor (NVIDIA, AMD, Apple, Intel, Steam Deck), no SDK contract, no vendor lock-in. The planned integration path is a DLL shim for titles already exposing DLSS, FSR, or XeSS inputs.

Pre-alpha. Active research. Apache 2.0 licensed — use it freely in commercial games.

Maintained by Cash Conway (<cashcon57@gmail.com>). Available for studios + GPU vendors needing OSS integration consulting, custom per-vendor kernel work, training-recipe tuning, or full-time engineering. Reach out if you want to ship vendor-neutral SR in your title.

---

## Latest results (v5-pixel-temporal, 2026-05-06)

**PSNR 25.703 dB · LPIPS 0.1666 · temporal-stability ratio 0.337×** on the TartanAir `oldtown` held-out batch (64 frames, 2× super-resolution from engine LR + G-buffers).

| | PSNR ↑ | LPIPS-VGG ↓ | Temporal ratio ↓ |
|---|---|---|---|
| bicubic baseline | 23.909 | 0.2945 | n/a |
| **OSS v5-pixel-temporal** | **25.703** | **0.1666** | **0.337** |
| v4 single-frame (distribution-shifted on TartanAir) | 11.718 | 0.6367 | reference (1.000) |

v5-pixel-temporal **beat bicubic on 64/64 held-out frames** (100% — spec target ≥95%) and improved temporal stability over v4 by ~3× (spec target ≥2×). Every quality gate of the validation memo passed. Full eval, methodology, and reproduction commands: [`docs/superpowers/experiments/2026-05-06-v5-pixel-temporal-final-held-out-eval.md`](docs/superpowers/experiments/2026-05-06-v5-pixel-temporal-final-held-out-eval.md).

These are the first measured OSS results that establish the temporal-SR architecture works end-to-end on photoreal game-engine input. v6 is the production-target architecture currently in training (see below).

(v4's 11.718 dB on TartanAir is the SRGD-trained-model-on-TartanAir distribution-shift failure mode. v4 measured ~30.1 dB / 0.30 LPIPS on its native SRGD held-out batch.)

---

## Where things stand

![v5-pixel-temporal in flight at training step 42K](docs/results/v5-pixel-temporal/in-flight/step-00042000.png)

The image above is the in-flight viz strip from a partial training run. Six panels in reading order: LR-bilinear, bicubic, v5-pixel-temporal, GT, and |error|. It is a mid-training snapshot, not a final measurement and not a vendor comparison. See [docs/results/](docs/results/README.md) for layout and how to regenerate it from a checkpoint.

The v4 single-frame upscaler is trained and exported. On the SRGD held-out batch it sits around 30 dB PSNR / 0.30 LPIPS, against bicubic at 27 dB / 0.45. That is the working baseline.

v5-pixel-temporal completed the validation pass on 2026-05-06. On the TartanAir `oldtown` held-out batch it measures PSNR 25.703 / LPIPS 0.1666 with temporal ratio 0.337 versus v4. See [2026-05-06-v5-pixel-temporal-final-held-out-eval](docs/superpowers/experiments/2026-05-06-v5-pixel-temporal-final-held-out-eval.md). v5-Gaussian-temporal is no longer the main baseline path; Option A was taken on 2026-05-06.

v6 is the architecture intended to ship. As of commit `732166a`, the model + trainer together run the canonical Stage 2 path end-to-end:

- `V6Model.forward()` (`fd8965f`): HAT backbone → motion-vector + GS-STVSR covariance canvas warp → keyframe active mask → cross-attention pixel↔Gaussian fusion → V6Rasterizer renders the active canvas subset to HR → composite head emits 3-channel RGB → softplus / sigmoid → spawner writes fresh Gaussians from refined features back into the persistent per-rank canvas → ST score state updates.
- `scripts/sr_train_v6.py` (`732166a`): per-step samples a trajectory of T consecutive frames (default 4); resets canvas at the trajectory start, threads engine motion vectors between frames, accumulates per-frame loss, runs backward once. The canonical-memo §5 motion-aware temporal-consistency term `||warp(pred_t, motion_t→t+1) − pred_{t+1}||₁` (weight 0.5) keeps adjacent predictions in the graph. `--first-ckpt-step` writes the first non-smoke checkpoint at step 100 by default, so issues surface within minutes of training start.

239 v6 tests pass (`./venv-py312/bin/python -m pytest tests/sr/v6/ -q`). OSS-FX (α<1 canvas rendering) is the next thing on top of the wired forward. The full diagram below is the target architecture; the wired forward path matches it modulo the OSS-FX α path.

The target design uses a persistent 2D Gaussian canvas, warped by exact engine motion vectors with covariance resampling at the rasterizer, fused with a HAT spatial backbone via cross-attention, with score-based active pruning to keep per-frame cost bounded. Three tiers share one architecture: Pico for handhelds, Standard for mainstream desktop, Heavy for enthusiast. Custom kernels per GPU vendor (CUDA, HIP, Metal, Level Zero, Vulkan compute) target real-time latency. Frame extrapolation (OSS-FX) is the same canvas rendered at fractional time positions instead of α=1; it does not require a separate ML network the way DLSS Frame Generation does.

The canonical v6 design lives in [experiments/2026-05-05-v6-architecture-canonical.md](docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md). The implementation roadmap, derived from deep-reads of GSASR, AAA-Gaussians, AA-2DGS, vk_gaussian_splatting, and GaussianVideo, is in [research/2026-05-05-v6-external-baselines-integration-plan.md](docs/research/2026-05-05-v6-external-baselines-integration-plan.md).

Target architecture:

```text
                                    ┌─────────────────────────┐
  current LR + G-buffers ──────────►│ HAT spatial backbone    │──► coarse SR features
  (RGB, depth, motion, normals)     └─────────────────────────┘                │
                                                                               ▼
                                                      ┌────────────────────────────┐
   persistent Gaussian canvas ──► analytical warp ───►│ cross-attention            │──► refined HR features
   (5K-15K Gaussians, accumulated   by engine MVs +   │ (pixel queries × Gaussian  │
    across frames)                  covariance        │  keys/values)              │
                                    resampling        └────────────────────────────┘
                                                                               │
   key-frame active mask (every K=10 frames) ─────────────► rasterizer ─────► HR output
                                                                               │
   Spatial-Temporal Variation Score pruning ◄─── update canvas ◄───────────────┘

   For frame extrapolation: rasterize canvas at α ∈ (0, 1) instead of α = 1.
   Cost: one in-place add to the position tensor.
```

---

## What is measured, what is not

v4 inference latency on RTX 3080 Ti, TensorRT FP16, narrow optimization profiles:

| Input → Output | Latency |
|---|---|
| 720p → 1440p | 15.6 ms |
| 1080p → 4K | 37.6 ms |

Source: [trt-int8-quantization](docs/superpowers/experiments/2026-05-03-trt-int8-quantization.md).

Steam Deck latency is unmeasured. Real Deck workloads upscale from 540p, 360p, or 240p up to Deck's native 800p (1280×800). Nothing in this table corresponds to a Deck workload, and v4 has not been benchmarked on Deck hardware. The Pico tier and hand-tuned Vulkan compute kernels (both v6) are prerequisites for that measurement.

v4's current latency reflects stock TensorRT FP16 with no per-vendor kernel optimization. Closing the gap to vendor stacks is the entire point of the custom-kernel sprint planned for v6. The table below gives the budgets v6 is targeting parity or competitiveness with.

| System | 1080p → 4K (typical) | 720p → 1440p (typical) | Hardware | Notes |
|---|---|---|---|---|
| FSR 2 / FSR 3 SR (compute shader) | ~0.7–1.0 ms | ~0.4–0.7 ms | RDNA2+, generic GPUs | hand-tuned shaders, no ML |
| DLSS 2 / DLSS 3 SR (CNN) | ~1.5–2.5 ms | ~1.0–1.5 ms | RTX 20+ via NGX | tensor-core-resident MMA |
| DLSS 4 SR (transformer, 2025) | ~3–4 ms | ~2–3 ms | RTX 30+ FP16 | ~4× compute of CNN model |
| DLSS 4 SR (transformer, FP8 path) | ~1.5–2 ms | ~1.0–1.5 ms | RTX 40+ / RTX 50+ FP8 | tensor-core FP8 |
| FSR 4 (ML, 2025) | ~1.5–2 ms | ~1.0–1.5 ms | RDNA4 only | hand-tuned matrix-core kernels |
| XeSS XMX | ~2–3 ms | ~1.5–2 ms | Intel Arc | XMX matrix engines |
| XeSS dp4a fallback | ~5–8 ms | ~3–5 ms | non-Arc, cross-vendor | graceful degradation |

Numbers are approximate published or independently-measured ranges and vary by GPU SKU, scene, and exact resolution. Primary sources: NVIDIA DLSS technical blog, AMD GPUOpen FSR documentation, Intel XeSS whitepaper, Digital Foundry latency-analysis measurements.

v5-pixel-temporal final held-out result: PSNR 25.703 / LPIPS 0.1666 / temporal ratio 0.337 on the TartanAir `oldtown` held-out batch. Source: [2026-05-06-v5-pixel-temporal-final-held-out-eval](docs/superpowers/experiments/2026-05-06-v5-pixel-temporal-final-held-out-eval.md). v5-Gaussian-temporal is implemented but no longer the main baseline after Option A on 2026-05-06.

---

## Known limits

HDR support is partial. The output activation has been changed from sigmoid to softplus, so HDR linear-light values above 1.0 flow through architecturally without clipping. The training corpus (TartanAir, Hypersim, SRGD) is 8-bit sRGB, so HDR-specific patterns like sun discs, neon, specular highlights, and BT.2020 wide gamut are under-represented. Expected quality on HDR content is roughly 70 to 80% of SDR quality on the same content class. Full HDR competence is v6.1 work, scheduled to retrain on HDR-encoded data captured via INSANE-mode contributors and Hypersim re-rendered in linear scRGB.

No game integration has shipped. The DXGI hook and NGX shim that let OSS substitute for DLSS, FSR, or XeSS at runtime are designed but not built. That work is Sprint 7. Inference engines run in isolation; the runtime plumbing is missing.

ML inference at this quality is fundamentally heavier than FSR's hand-tuned shader passes. The plan to bridge the gap is per-vendor custom kernels (v6) plus distillation to a small Pico variant for handhelds and integrated GPUs.

The supported-games list is editorial. Anti-cheat titles using kernel-level systems (Vanguard, BattlEye, EAC, Ricochet) are off the list permanently because DLL injection trips them and risks player bans.

---

## Roadmap

S5 is closed for the baseline decision: v5-pixel-temporal produced the carried-forward result on 2026-05-06, and v5-Gaussian-temporal is parked unless staged smoke tests justify reopening it.

S6 is v6: the covariance-resampled Gaussian-temporal architecture summarized above. Modules, orchestrator, Stage 2 wire-up of `V6Model.forward()` putting canvas in the HR critical path (commit `fd8965f`), and the trajectory training loop with canvas continuity + motion-aware temporal-consistency loss + early checkpoint at step 100 (commit `732166a`) have landed. 239 v6 tests passing. OSS-FX integration is next on the roadmap. Three tiers (Pico, Standard, Heavy), custom per-vendor kernels, and DLL-shim integration remain target work.

S7 is the planned DLL-shim runtime that lets OSS replace DLSS, FSR, or XeSS in already-shipping games without developer cooperation. No game integration has shipped yet. Game requirements: must already use one of the three (which is how OSS receives depth, motion vectors, and jitter), and must not use kernel-level anti-cheat. Candidate validation targets include Cyberpunk 2077, Alan Wake 2, Hogwarts Legacy, Starfield, Baldur's Gate 3, Returnal, Hellblade II, Forza Horizon 5, and Black Myth: Wukong.

S7-data is the OSS Capture Tool. A small DLL drops into supported games, captures rendered frames and engine G-buffers while you play, then deletes the local copies once the server confirms upload. Four bandwidth tiers: trickle (~100 MB/h), lite (~500 MB/h, the default), regular (~2 GB/h), and INSANE (~20–50 GB/h). The server is FastAPI on Cloudflare R2. Codex is implementing the client side; the server side ships under this repo.

The custom-kernel work runs in parallel to S6 and S7. NVIDIA gets CUDA + CUTLASS, AMD desktop gets HIP + rocWMMA, Apple Silicon gets Metal + ANE, Intel Arc gets Level Zero + XMX, and Steam Deck plus everything without matrix accelerators gets hand-written Vulkan compute. These are 6 to 12 month engineering bets, not week-scale tasks. Per-vendor design notes: [vendor-optimization-audit](docs/superpowers/notes/vendor-optimization-audit.md), [cuda-mega-kernel-design](docs/superpowers/notes/cuda-mega-kernel-design.md).

---

## OSS Capture Tool

OSS improves when it trains on real games. The Capture Tool is the data-collection path.

You drop a per-game DLL in. You play. Frames upload, then delete from disk. You set the bandwidth budget at install time. You can pause uploads or uninstall whenever.

What does not get captured: audio, keyboard, mouse, controller input, any other window, your desktop, webcam, microphone, chat, save data, or network traffic. Only the supported game's rendered output and its engine buffers.

| Mode | Bandwidth | Trade-off |
|---|---|---|
| trickle | ~100 MB/h | sparse capture, no long temporal sequences |
| lite (default) | ~500 MB/h | the v5/v6 sweet spot, no albedo or roughness |
| regular | ~2 GB/h | adds material channels, higher network bill |
| INSANE | ~20–50 GB/h | full PBR plus 256-frame supersample ground truth on settled cameras; brief stutters disclosed at install |

Identity is a one-time opaque token. No account, no email, no PII. Frames are written to R2 under `<game_id>/<YYYY-MM>/<capture_mode>/<session_uuid>/<frame_uuid>.exr`. Per-mode contribution counts are public so you can see what your bytes did. Trained model weights derived from contributed data ship under CC-BY-4.0.

Cyberpunk 2077 is the initial validation target. Full design: [oss-capture-tool-design](docs/superpowers/specs/2026-05-04-oss-capture-tool-design.md).

---

## Hardware tiers (v6)

All three tiers share the v6 architecture, scaled. Distillation chain: Heavy → Standard → Pico. Latency targets below are ship goals, not current measurements. They are conditional on the per-vendor native-kernel work landing at vendor-stack optimization quality (CUDA + CUTLASS + tensor-core MMA on NVIDIA, HIP + rocWMMA on AMD desktop, Metal + ANE on Apple Silicon, Level Zero + XMX on Intel Arc, hand-tuned Vulkan compute for Steam Deck and matrix-accelerator-less hardware). Stock-runtime latency is several × these numbers; the gap closes through the custom-kernel sprint, not for free.

| Tier | Backbone | Canvas | Target hardware | Ship target (conditional) | Backend |
|---|---|---|---|---|---|
| Pico | HAT-Tiny (~1M params) | ~1–2K Gaussians | Steam Deck, integrated GPUs, mobile dGPU | <2 ms at 720p → 1080p | hand-tuned Vulkan compute |
| Standard | HAT-Small (~5M params) | ~5K Gaussians | RTX 30+, RX 6700+, Arc, M2+ | <3 ms at 1080p → 1440p | CUDA / HIP / Metal / Level Zero |
| Heavy | OSS HAT-L-derived Heavy (~17M target params) | ~15K Gaussians | RTX 4080+, RX 7900+, M4 Max | <4 ms at 1440p → 4K | same per-vendor path |

The targets bracket vendor latency bands: the target Pico latency band falls within handheld budgets; measurement pending. Standard sits in DLSS 2/3 SR territory, and Heavy lands at DLSS 4 transformer territory. See the budget comparison table earlier in this README for context.

Quality modes (planned, by upscale ratio): Ultra Performance 33%, Performance 50%, Balanced 59%, Quality 67%, Ultra Quality 77%. OSAA is anti-aliasing at native resolution (100%).

---

## Training data

SRGD: 51K high-resolution frames across 17 scenes. Sequential, but the depth, motion, and normal channels were placeholder zeros. Used for v3 and v4 training; no temporal gains were realized at v4 ship because of the missing G-buffers.

TartanAir: roughly 600 GB extracted across 18 environments, with real depth and real optical flow from a photoreal sim engine. Primary v5 temporal training set.

Hypersim: photoreal indoor scenes from Blender, with real depth and normals. Not yet integrated.

Sintel: cinema-quality with real depth and flow. Used for validation and fine-tuning.

Vimeo-90K: planned for OSS-FX, real-world motion diversity.

NoiseBase: planned for OSS-RG (denoiser track). Download is currently blocked on bandwidth allocation on the remote host.

Public datasets are the default. Custom captures are a last resort.

---

## Why this exists

DLSS-class quality is locked to NVIDIA RTX hardware through the proprietary NGX runtime. FSR is cross-vendor and open but bounded above by what hand-tuned shaders without learned components can express. XeSS XMX is Intel-only at peak quality. There is no open, cross-vendor, ML-based real-time SR with comparable quality to DLSS.

OSS is the attempt to fill that gap. It also unifies frame extrapolation and super-resolution under a single architecture: render the same canvas at α=1 for the current frame, render at α<1 for an intermediate frame. DLSS solves frame extrapolation with a separate ML network that has its own latency and known artifacts. The Gaussian-canvas approach gets it as one in-place add to the position tensor.

Cross-vendor, open, and unified with frame extrapolation is the bet.

---

## Lab-notebook discipline

Every training run, ablation, and benchmark gets a memo in `docs/superpowers/experiments/YYYY-MM-DD-<slug>.md` before the result is allowed to drive a decision. The cost of writing a paragraph before measuring something is much less than the cost of an unmoored decision later. The discipline is documented in [lab-notebook-discipline](docs/papers/lab-notebook-discipline.md).

---

## Reference docs

v5 (current sprint), specs and runbooks:

- [v5-pixel-temporal design](docs/superpowers/specs/2026-05-04-v5-pixel-temporal-design.md)
- [v5-gaussian-temporal design](docs/superpowers/specs/2026-05-04-v5-gaussian-temporal-design.md)
- [Gaussian temporal canvas design](docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md)
- [v5 pixel-temporal runbook](docs/superpowers/notes/2026-05-04-v5-pixel-temporal-runbook.md)
- [v5 pixel launch status](docs/superpowers/notes/2026-05-04-v5-pixel-launch-status-r2.md)
- [v5 Sintel fine-tune runbook](docs/superpowers/notes/2026-05-04-v5-pixel-sintel-finetune-runbook.md)
- [v5 Gaussian temporal runbook](docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md)
- [v5 pixel temporal ONNX export design](docs/superpowers/notes/2026-05-04-v5-pixel-temporal-onnx-export-design.md)
- [v5 rolling review](docs/superpowers/notes/2026-05-04-v5-rolling-review.md)

v6 architecture (locked 2026-05-05):

- [v6 canonical design](docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md)
- [v6 external-baselines integration plan](docs/research/2026-05-05-v6-external-baselines-integration-plan.md)
- [GS research deep-dive (math)](docs/research/2026-05-05-gaussian-temporal-research-deep-dive.md)
- [Existing Gaussian-Splatting repos survey](docs/research/2026-05-05-existing-gaussian-splatting-repos-survey.md)
- [GSASR + upscale3dgs deep-read](docs/research/2026-05-05-gsasr-and-upscale3dgs-deep-read.md)
- [Anti-aliasing stack deep-read](docs/research/2026-05-05-anti-aliasing-stack-deep-read.md)
- [NVIDIA vk + GaussianVideo deep-read](docs/research/2026-05-05-nvidia-vk-and-gaussianvideo-deep-read.md)

S6 prep (performance pass):

- [Vendor optimization audit](docs/superpowers/notes/vendor-optimization-audit.md)
- [CUDA mega-kernel design](docs/superpowers/notes/cuda-mega-kernel-design.md)
- [Pico distillation design](docs/superpowers/notes/2026-05-04-pico-distillation-design.md)

S7 prep (game integration):

- [S7 game integration design](docs/superpowers/notes/2026-05-04-s7-game-integration-design.md)

OSS Capture Tool:

- [Capture tool design](docs/superpowers/specs/2026-05-04-oss-capture-tool-design.md)
- [D3D12 hook design](docs/superpowers/d3d12-hook-design.md)

Top-level project paper drafts and citation:

- [RESEARCH.md](RESEARCH.md), [BIBLIOGRAPHY.md](BIBLIOGRAPHY.md), [CITATION.cff](CITATION.cff)

---

## License

SDK and shaders: Apache-2.0. Plugins: MIT. Model weights: CC-BY-4.0.

## What I won't use

- NRD (RTX SDK contract)
- DLSS / FSR / XeSS decompiled binaries or leaked weights
- `tiny-cuda-nn` (CUDA-only by design; defeats the cross-vendor commitment)
- Quixel, Megascans, or CC-BY-NC training assets

## Repository

Active development branch: `main`. Track docs:

- [oss-sr-cnn-track](docs/superpowers/oss-sr-cnn-track.md): single-frame pixel upscaler, current production candidate
- [oss-gaussian-rr-track](docs/superpowers/oss-gaussian-rr-track.md): Gaussian-based ray-reconstruction denoiser
- [oss-gaussian-temporal-track](docs/superpowers/oss-gaussian-temporal-track.md): Gaussian temporal canvas, parent track for v5-gaussian-temporal

Recent decision memos: [docs/superpowers/experiments/](docs/superpowers/experiments/).
