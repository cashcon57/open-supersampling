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

## What v0.2 will deliver — UPSCALER-FIRST (revised 2026-04-30)

See [`docs/goals/project-goals.md`](../goals/project-goals.md) for the full milestone list. **Strategic pivot 2026-04-30**: ship upscaler DLL before denoiser DLL. Bigger install base (every DLSS/FSR/XeSS game ~1000+ titles vs ~10-20 RT/PT games), simpler API surface, validates drop-in DLL infrastructure with broader user feedback before tackling the harder DLSS-RR API. Denoiser ships in v0.3 leveraging validated v0.2 infra.

Highlights:

1. **Three-tier ORU architecture** — ORU-Tiny (~500K, GTX 10/16/RX 5000/integrated), ORU-Lite (~1M, RTX 20+/RDNA2+/M-series base/Steam Deck), ORU-Standard (~2.6M, RTX 4080+/RX 9070 XT+/M3 Pro+ with coop_matrix). Same DLL, runtime detection picks tier. Hardware coverage spans entire gaming GPU market since ~2016.
2. **Real training data** from rasterized + RT game traces on cloud GPU (~$300 budget).
3. **ONNX export** + ONNX Runtime + DirectML inference dispatch (Windows). Vulkan compute path (Linux).
4. **Three drop-in DLL replacements**, one inference engine:
   - `nvngx_dlss.dll` (DLSS Super Resolution)
   - `amd_fidelityfx_dx12.dll` / `amd_fidelityfx_vk.dll` (FSR 2/3/4 upscaler)
   - `libxess.dll` (XeSS)
5. **Per-game compatibility shim layer** (community-maintainable game profiles).
6. **First integration test** — popular game with FSR/DLSS-SR support (Helldivers 2, Starfield, or CP2077 raster mode).
7. **Wine/Proton compatibility** validated for Linux gaming users.
8. Total budget estimate: ~$1500 cloud GPU + ~$60 first-test game license, ~2-3 months solo engineering.

**Note: denoising is NOT bundled in v0.2.** Pure upscalers consume clean input from the game's existing pipeline. The game's own denoiser (NRD, hand-tuned, or DLSS-RR if installed) handles noise BEFORE the upscaler runs. ORS-upscaler is a drop-in for the same contract FSR/XeSS/DLSS-SR all already meet.

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

## Decisions made 2026-04-30

1. **v0.2 = upscaler DLL** (not denoiser). Bigger install base, simpler API. Denoiser → v0.3.
2. **Three hardware tiers** (Tiny / Lite / Standard) so we cover GTX 10-series + Steam Deck + flagship in one DLL.
3. **Inference: ONNX-RT + DirectML** for v0.2 ship. Inline HLSL CoopVec / SPIR-V coop_matrix → v0.3 perf push.
4. **DLL targets**: `nvngx_dlss.dll` + `amd_fidelityfx_*.dll` + `libxess.dll` (three pure-upscaler surfaces).
5. **Budget**: ~$1500 cloud GPU, ~2-3 month timeline.
6. **RE/leaked-source work: SKIP** (clean-room only, public open-source patterns sufficient — quality lead from Bálint 2026 published architecture, not RE).
7. **First test game**: TBD — pick a game with broad FSR/DLSS-SR support that's accessible without expensive licensing.

## Open questions / next decisions

1. Which specific first-test game? (CP2077 raster, Helldivers 2, Starfield, others?)
2. Engine integration plugin track (UE5 path-tracer denoiser plugin) — keep on v0.4 or accelerate?
3. LoRA framework — v0.3 or v0.4?
4. Adaptive sampling research track (Bálint 2026 paradigm shift) — v1.0 sister project or v0.5+ ORS extension?

## Files of record

- Design spec: [`docs/specs/2026-04-29-design.md`](../specs/2026-04-29-design.md)
- v0.1 MVP plan + tasks: [`docs/plans/2026-04-29-mvp-plan.md`](../plans/2026-04-29-mvp-plan.md), `.tasks.json`
- Goals: [`docs/goals/project-goals.md`](../goals/project-goals.md)
- Research synthesis: [`docs/research/2026-04-30-deep-research-synthesis.md`](../research/2026-04-30-deep-research-synthesis.md)
