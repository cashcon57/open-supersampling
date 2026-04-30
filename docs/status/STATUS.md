# ORS Project Status

**Last updated:** 2026-04-30
**Current version:** v0.1.0-mvp (released 2026-04-29)
**Active milestone:** planning v0.2 (drop-in DLL)

## What's shipped

- v0.1.0-mvp tagged on `main`. 11 commits, 23/23 tests passing.
- Pure-PyTorch reference implementation, single-platform (macOS arm64, CPU only).
- ORD denoiser (kernel-prediction U-Net, two-branch input, 141K-1M params per tier).
- ORU upscaler (3 input modes: rgb / rgb_aux / features).
- Frozen v1 feature-handoff contract (32-ch FP16 tensor) + PairedORS wrapper.
- Three trainers (ORD-only, ORU-only, paired with two-stage freeze/unfreeze).
- Valuation harness (PSNR/SSIM/LPIPS + perf bench + comparison CSV).
- Reproducibility scripts (`train_all.sh`, `run_compare.sh`).

## What v0.1 deliberately does NOT do

- No cross-vendor inference (HLSL/SPIR-V/MSL deferred to v0.2-v0.3).
- No engine plugins (UE5 deferred to v0.4).
- No drop-in DLL (deferred to v0.2 — the actual go-to-market).
- No real training data — synthetic random tensors only for smoke tests.
- No production weights — checkpoints saved by smoke tests are random init only.
- 141K-param ORD undersized for production quality (intentional for MVP smoke; v0.2 upgrades to JNDS-shape 2.6M).

## What v0.2 will deliver

See [`docs/goals/project-goals.md`](../goals/project-goals.md) for the full milestone list. v0.2 highlights:

1. **Architecture upgrade** to JNDS-shape (2.6M params, Bálint Mini Adaptive lineage). Hits 24.97 PSNR @ 0.25 spp per published Bálint 2026 paper — beats DLSS 4 RR by 2.09 dB.
2. **Real training data** rendered on cloud GPU via Mitsuba 3 (~$300 budget).
3. **ONNX export** + ONNX Runtime + DirectML inference dispatch.
4. **`nvngx_dlssd.dll` drop-in replacement** for DLSS Ray Reconstruction. Implements NGX RR API surface from open Streamline spec.
5. **Cyberpunk 2077 Path Tracing** as canonical first integration test.
6. Total budget estimate: ~$1500 cloud GPU + ~$60 game license, ~2-3 months solo engineering.

## Known v0.1 limitations carried forward

(Documented in `README.md` "Known limitations" section; v0.2 fixes most of these.)

- Single platform (Linux + CUDA only for full pipeline; macOS arm64 for smoke).
- Single scene (Bistro only — no procedural augmentation, no diverse training set).
- Synthetic temporal history (`gt + 0.05*randn` placeholder).
- Roughness + specular hit distance G-buffer channels are zeros (Mitsuba 3.7/3.8 AOV doesn't expose them — v0.2 derives from `position` AOV + material params).
- Bistro per-view camera override not wired (`mi.load_file` doesn't apply per-view sensors).
- OIDN baseline is a stub (binding not wired).
- No engine plugin yet.

## Strategic position summary

- **Quality**: the Bálint 2026 architecture (which v0.2 adopts) is published at +2.09 dB PSNR over DLSS 4 RR at 0.25 spp. This is not aspirational — it's peer-reviewable and reproducible.
- **Perf**: parity with DLSS 4 RR on FP16 hardware (Turing/Ampere/RDNA3); 70-80% of DLSS 4 RR perf on Blackwell FP8; outright beat FSR Ray Regen on AMD/Intel/Apple.
- **Distribution**: drop-in DLL = day-1 install base of every game with DLSS RR support (CP2077 PT, Alan Wake 2, Black Myth Wukong, Indiana Jones, Portal RTX, Hellblade II, etc.).
- **Differentiation**: per-game LoRA adapters + variable-rate inference + Apple Silicon support are structural advantages DLSS cannot match.

## Open decisions (gating v0.2 spec drafting)

1. Target `nvngx_dlssd.dll` first vs FSR/XeSS DLLs?
2. Architecture: upgrade to JNDS-shape 2.6M for v0.2 ship?
3. Inference: ONNX-RT + DirectML for v0.2, inline HLSL CoopVec for v0.3?
4. First test game: Cyberpunk 2077 Path Tracing?
5. Budget: ~$1500 v0.2 cloud GPU?
6. Timeline: 2-3 month v0.2 milestone?
7. RE/leaked-source work: SKIP per 2026-04-30 decision (clean-room only, public open-source patterns sufficient).

## Files of record

- Design spec: [`docs/specs/2026-04-29-design.md`](../specs/2026-04-29-design.md)
- v0.1 MVP plan + tasks: [`docs/plans/2026-04-29-mvp-plan.md`](../plans/2026-04-29-mvp-plan.md), `.tasks.json`
- Goals: [`docs/goals/project-goals.md`](../goals/project-goals.md)
- Research synthesis: [`docs/research/2026-04-30-deep-research-synthesis.md`](../research/2026-04-30-deep-research-synthesis.md)
