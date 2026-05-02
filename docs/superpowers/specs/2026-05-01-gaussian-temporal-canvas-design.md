# OSS Gaussian Temporal Canvas — Design Spec

**Date:** 2026-05-01
**Status:** Approved for implementation
**Track:** Sits alongside existing pixel-based OSS/OSSPico/OSSFx until verified better.
**Predecessor:** `2026-04-30-v0.2-deck-first-design.md` (pixel-based track, untouched)

---

## 1. Goal

Build a real-time game super-sampler + frame extrapolator using a **persistent 2D Gaussian temporal canvas** as the intermediate representation, instead of pixel accumulation buffers. The canvas warps with motion vectors, freezes covariance frame-to-frame, and predicts only position deltas + color via a small neural network — yielding structural ghosting elimination, unified SR + frame extrapolation, and a single Gaussian-count knob that scales from Steam Deck to RTX 4090.

First validation target: **Cyberpunk 2077 / Windows / RTX 3080 Ti** for direct DLSS 4 comparison.

## 2. Why Gaussian Canvas

- **Ghosting is structural, not mitigated.** Warping geometric primitives (Gaussians) with motion vectors is fundamentally more correct than warping pixel colors. A misplaced Gaussian shows as one local error and gets pruned/replaced — not smeared history across the buffer.
- **Frame extrapolation is free.** Warp the same canvas to t+α and render — no separate frame-gen network. DLSS Frame Generation is an additive heavy pass; OSS-Gaussian's frame gen is a parameter on the warp step.
- **Single model, all hardware tiers.** Gaussian count is the quality budget knob. 1K Gaussians on Steam Deck → 15K Gaussians on RTX 4090. No separate Pico model.
- **Resolution-independent.** Renderer outputs at any target resolution from the same Gaussian set. Train at 2× upscale, deploy at 4×.
- **Covariance is temporally stable.** From GS-STVSR (2025): correlation 0.99 between adjacent frames. Network only predicts position deltas + color corrections each frame — covariance reuses prior values via Covariance Prior Bank weights.
- **0.3K MACs per pixel decode** (Image-GS, SIGGRAPH 2025) — hardware-friendly memory-bandwidth-bound pipeline.

## 3. Architecture

### 3.1 Component map

```
LR frame + G-buffers (depth, motion, normals)
    │
    ├─→ Tile Classifier (gradient magnitude → complex/simple mask)
    │
    ├─→ Persistent Gaussian Canvas (GPU buffer, N Gaussians)
    │       │
    │       ├─→ Motion Warp (positions only; covariance frozen)
    │       │
    │       └─→ Error Detection (per-Gaussian reconstruction error)
    │                   ↓
    │           Prune + Replace from LR input on disocclusion
    │
    ├─→ Gaussian Param Network (complex tiles only)
    │       Input: LR + G-buffers + canvas state
    │       Output: (Δposition, Cov Prior Bank weights, color) per tile
    │
    └─→ Tile-based Top-K Renderer (Image-GS CUDA renderer reused)
            ↓
        Native-res output (or t+α extrapolated frame)
```

### 3.2 Components

| # | Component | Description |
|---|---|---|
| 1 | CUDA Gaussian Renderer | Reuse Image-GS tile-based top-K renderer. No port — native CUDA. |
| 2 | D3D12 Frame Interception | DXGI `Present()` hook in Cyberpunk 2077. Extract color, depth, motion vectors. DLL-swap pattern. |
| 3 | Tile Classifier | CUDA compute pass. Gradient magnitude on LR frame → 16×16 tile mask. Simple tiles bypass network. |
| 4 | Gaussian Param Network | Lightweight CNN. Input LR+G-buffers+canvas. Output per-tile Gaussian params. Train PyTorch, infer TensorRT INT8. |
| 5 | Persistent Canvas + Warp | GPU buffer of N Gaussian structs across frames. Motion-vector warp. Error detection + Gaussian replacement. |
| 6 | Frame Extrapolation | Warp canvas by motion × α; render at t+α. No separate model. |
| 7 | Cross-platform Ports | Sprint 7: M3 Max (Metal MSL renderer + CoreML net), Steam Deck (ncnn/Vulkan + 1K Gaussians). |
| 8 | Code Review Pipeline | 2 reviewer agents + 1 judge agent at each sprint checkpoint. Judge verdict gates next sprint. |

### 3.3 Covariance Prior Bank

To prevent degenerate Gaussians and reduce network output dimensionality, the network predicts weights over a fixed bank of ~16 pre-defined covariance shapes (circular, elongated horizontal/vertical/diagonal at varying scales) rather than raw 2×2 matrices. Final covariance = softmax-weighted sum of bank entries.

Bank size and shape vocabulary: tune during Sprint 4. Initial bank: 16 entries.

### 3.4 Hardware tiers (Gaussian count budget)

| Tier | GPU | Gaussian count | Resolution target |
|---|---|---|---|
| Ultra | RTX 4090 | 15K | 4K |
| Standard | RTX 3080 Ti | 8K | 1440p / 4K |
| Lite | M3 Max / RTX 4070 | 5K | 1440p |
| Pico | Steam Deck | 1K | 1280×800 |

Single trained model. Gaussian count is a runtime parameter.

## 4. Targets

### 4.0 Scope: any DLSS-using DX12 game

OSS-Gaussian targets the universal DLSS API surface (10 `NVSDK_NGX_D3D12_*` exports + DXGI proxy), not any single game. The interception DLL is renamed `dxgi.dll` and dropped in any DX12 game's `bin\x64\` folder; game-local DLL search resolves it before system32. Compatible games are: any DX12 title that ships native DLSS 2/3, has no kernel-level anti-cheat, and runs without Path Tracing (PT forces DLSS-RR which is a different API surface — see Sprint 2 T2.11).

Per-game compat profiles (community-editable JSON, runtime-loaded) handle game-specific quirks post-v1. No game-specific code in the DLL itself.

### 4.1 Primary validation target (Sprints 1–6)
- **Hardware:** RTX 3080 Ti (Windows, apartment LAN via Tailscale)
- **Game:** Cyberpunk 2077 (DX12, native DLSS, no anti-cheat, well-documented hook patterns from OptiScaler)
- **Integration:** universal DLL swap (DLSS DLL replacement), DXGI hook for G-buffer extraction
- **Output target:** 1440p with frame extrapolation 60→120 FPS

### 4.1.b Post-v1 expansion target list

Post-Sprint 6, validate on at least 2 more DLSS-shipping titles to prove the universal DLL approach. Candidate matrix:
- Hogwarts Legacy (DX12, DLSS 3 + FG, no AC)
- Portal RTX (DX12, DLSS-RR — tests PT-detection guardrail)
- Alan Wake 2 (DX12, DLSS 3.5)
- Cyberpunk 2077 modded (DLSS 4 / Frame Warp)

Compatibility matrix updated in `docs/compatibility.md` (TBD post-Sprint 6).

### 4.2 Cross-platform ports (Sprint 7)
- **M3 Max MacBook Pro:** Metal compute renderer + CoreML network. Test under CrossOver with same Cyberpunk 2077 install or other CrossOver-compatible game.
- **Steam Deck:** ncnn/Vulkan inference, 1K Gaussian budget, 1280×800 output. Sintel baseline (no game integration in Sprint 7 — that's a future sprint).

## 5. Graduation criterion (Gaussian → primary)

Gaussian track replaces existing pixel-based modules when ALL of:

1. **Automated metric:** Gaussian PSNR + SSIM beats OSSPico baseline on a fixed Cyberpunk 2077 test frame set (≥500 frames sampled across multiple in-game environments).
2. **Temporal stability:** Quantified ghosting metric (frame-to-frame pixel delta in flat regions) ≤ OSSPico baseline.
3. **Subjective:** User confirms visual quality is better in side-by-side video.
4. **Latency parity:** Total frame time < 110% of OSSPico at equivalent quality mode (i.e., ≤10% latency regression maximum, "not meaningfully worse").

Until graduation, both tracks coexist in the repo. After graduation, pixel-based modules archived under `oss/legacy/`.

## 6. Code Review Pipeline

At end of each sprint:

1. **Reviewer Agent A** — examines code for correctness, security, performance, idiomatic style.
2. **Reviewer Agent B** — examines code for spec adherence, test coverage, edge cases, integration risks.
3. **Judge Agent** — reads both reviews + the diff. Issues verdict: **APPROVE** / **REQUEST CHANGES** / **BLOCK**.
4. **APPROVE** → sprint closes, next sprint starts.
5. **REQUEST CHANGES** → addressed and re-reviewed.
6. **BLOCK** → escalates to user for decision.

Pipeline implemented as Python script invoking subagents with structured prompts. Reviewer outputs are JSON, judge consumes both.

## 7. Build Order (with dependencies)

```
Sprint 1: CUDA Gaussian Renderer       (no deps, ~3 days)
Sprint 2: D3D12 Hook + G-buffers       (deps: 1, ~1.5 weeks)
Sprint 3: Tile Classifier              (deps: 1, ~1 week, can parallel with 2)
Sprint 4: Gaussian Param Network       (deps: 1, 3, ~4 weeks training)
Sprint 5: Canvas + Warp + Error Detect (deps: 1-4, ~3 weeks)
Sprint 6: Frame Extrapolation          (deps: 5, ~1 week)
Sprint 7: Cross-platform Ports         (deps: 1-6, ~2 weeks)
```

**Total estimate:** ~13–14 weeks to fully operational Gaussian track on Cyberpunk 2077 with DLSS comparison data + cross-platform ports.

## 8. Repo Layout

```
open-reconstruction-suite/
├── oss/                       # existing pixel-based code, untouched
├── oss/gaussian/              # new — Gaussian temporal canvas track
│   ├── renderer/              # Sprint 1: CUDA renderer (Image-GS integration)
│   ├── interception/          # Sprint 2: D3D12 hook, DLL swap
│   ├── classifier/            # Sprint 3: tile classifier
│   ├── network/               # Sprint 4: Gaussian param net
│   ├── canvas/                # Sprint 5: persistent canvas + warp
│   ├── extrapolation/         # Sprint 6: frame extrap
│   ├── ports/                 # Sprint 7: metal/, vulkan_ncnn/
│   └── review/                # code review pipeline
└── docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md
```

## 9. Datasets

Reuse the in-progress download set:
- **Sintel** — synthetic motion + depth ground truth (validation)
- **TartanAir** — synthetic depth + flow at scale (training)
- **HyperSim** — photorealistic depth (validation)
- **SRGD** — synthetic SR ground truth pairs
- **Cyberpunk 2077 RenderDoc captures** — domain-specific training data, captured locally on 3080 Ti

License: only CC-BY / CC0 / Apache assets used directly. Cyberpunk captures used for input distribution only, not as training labels.

## 10. Out of scope

- Denoising — OSSRG remains the denoiser path. Gaussian canvas is upscaling + frame gen only in v1.
- Steam Deck DLL swap — Sprint 7 ports the engine but full game integration on Deck is post-v1.
- Console partnerships, UE5 plugin, Vulkan layer for non-DLL games — all post-v1.

## 11. Risks + mitigations

1. **Image-GS CUDA renderer not real-time enough.** Mitigation: render path is memory-bandwidth-bound (3.7ms for 10K Gaussians on A6000). 3080 Ti has comparable bandwidth. Tile classification skips 70% of tiles — should land at ~1.7ms. If too slow: reduce Gaussian count, increase tile size, or fuse rendering with classification.
2. **D3D12 hook breaks Cyberpunk anti-cheat or Red Engine version drift.** Mitigation: OptiScaler reference implementation works today on Cyberpunk. Pin to known-good Cyberpunk version during MVP.
3. **Network can't predict good Gaussian params from G-buffers in real time.** Mitigation: train against differentiable Image-GS renderer. If quality insufficient, fall back to higher Gaussian count with simpler network output (no covariance prediction, just position + color).
4. **Covariance Prior Bank vocabulary too narrow.** Mitigation: ablate bank size during Sprint 4. Increase from 16 → 32 if needed.
5. **DLSS comparison shows OSS-Gaussian worse.** Mitigation: that's the point of the graduation criterion — pixel-based track remains primary if Gaussian doesn't beat it. Lessons folded back into existing OSSPico training.

## 12. Success criteria for v1

OSS-Gaussian v1 ships when:

- Cyberpunk 2077 runs with OSS-Gaussian replacing DLSS via DLL swap on RTX 3080 Ti
- Direct DLSS 4 vs OSS-Gaussian comparison video produced
- Frame extrapolation 60→120 FPS works in-game
- M3 Max + Steam Deck ports compile and run a Sintel benchmark
- Graduation criterion met OR explicitly determined not met (with data showing why)

## 13. Open questions (non-blocking)

1. Cyberpunk 2077 specific DLSS DLL filename + version pinning — resolved during Sprint 2.
2. Optimal Covariance Prior Bank vocabulary — resolved during Sprint 4 ablations.
3. Tile size: 16×16 (Image-GS default) vs 32×32 (potentially better cache behavior on RTX 30-series) — resolved during Sprint 3.
4. Whether to train on RenderDoc Cyberpunk captures directly or only as input distribution — resolved during Sprint 4 license review.
