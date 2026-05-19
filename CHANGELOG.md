# Changelog

All notable changes to OpenSuperSampling. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: SemVer pre-1.0 (PATCH = bug fixes / docs; MINOR = new features; MAJOR reserved for v1.0).

## [Unreleased] — `v0.2-dev` branch

### Host migration prep — 3080 Ti wipe Windows → CachyOS Linux (2026-05-17)

The 3080 Ti training host is being reinstalled from Windows 11 + WSL2 to native CachyOS Linux to escape a series of WSL2-specific failure modes that surfaced during the v7 training run:

- WSL2's 9P cross-filesystem protocol stalls walks over `/mnt/e/datasets/tartanair_extracted` for tens of minutes on first launch.
- Multiple silent process-death modes (the `expandable_segments` env var, MSVC `cl.exe` missing for torch.compile inductor, PowerShell `Start-Process` pipe lifetime quirks, AVX-512 codegen bugs in torch 2.9+ Windows wheels on Zen 3 CPUs, etc.).
- A Hyper-V Compute Service API storm during the wipe-prep day correlated with dockerd restarting all 18+ home-lab containers simultaneously, which knocked the Cloudflare Tunnel into a bad state until origin routing was changed.

Linux native sidesteps all of the above (no WSL, GCC-built torch wheels avoid the AVX-512 bug, real systemd for process management, normal filesystem semantics for the dataset).

- New: `Starting up after wipe.md` — comprehensive recovery runbook for the new host.
- New: `docker/trainer/` — Dockerfile + entrypoint + docker-compose.yml that replace the Windows PowerShell launchers. Healthcheck on `history.jsonl` mtime, hard `mem_limit=12g` so a runaway trainer cannot kill dockerd.
- New: `archive/v7-pico-005-snapshot-2026-05-16/` — full preserved set of checkpoints (`step-00000100.pt` through `step-00005000.pt`), `history.jsonl`, `score_log_v7.json`, and `gpu_status.json`. Trainer auto-resumes from step 5000 on the new host.
- New: `archive/legacy-runs/` — metric-only snapshots of the prior runs (prod-v4-lpips, v5-pixel-temporal-validated, v6-pico-001, v6.1-pico-001, v6.2-pico-002). No checkpoints — those weights are not coming back, but the loss / held-out / event JSON is preserved for historical analysis.
- New: `archive/legacy-windows-launchers/` — historical reference copies of `launch_v7_debug.ps1`, `restart_v7.ps1`, `check_v7.ps1`, `find_v7.ps1` from the wiped host. On Linux these are replaced by the docker-compose entry; preserved only so the wipe is fully forensic-recoverable for a few months.
- New: `archive/README.md` — index of what is preserved, what is on R2, and what is intentionally not preserved (TartanAir dataset, viz strips, v6.2 intermediate ckpts).
- `.gitignore`: added explicit overrides for `archive/**/*.pt` and related metric formats so the snapshot survives the existing `*.pt` ignore rule.
- Secrets: nightly age-encrypted backup of `.secrets/` to R2 bucket `oss-secrets-backup` ran on 2026-05-19 02:42 UTC (`secrets-cashs-macbook-pro-20260519T024248Z.tar.gz.age`). Recovery procedure: `.secrets/RECOVERY-README.md`.

The active training run state at wipe time: **step 5000**, `total_loss=0.11950`, `canvas_count=2304`. Compare any post-wipe step 5050+ metrics against `archive/v7-pico-005-snapshot-2026-05-16/history.jsonl` to detect regressions from the host transition.

### v7 Phase 3 kickoff + v6.2-pico-002 stopped early (2026-05-14)

`srcnn-v6.2-pico-002` terminated 2026-05-12 at step 74,000 of 100,000 when the GPU was reclaimed by another process. Rather than resume the remaining 26K steps the project moved to v7-pico-005 because the architecture has changed substantially (N-D Gaussian primitive with V_xt cross-correlation in the Cholesky covariance, parent-child loss-adaptive density, OSS-FX time-slice rendering, Mip-Splatting anti-aliasing filters) and the marginal v6.2 training would not validate any of that. The step-00074000.pt checkpoint is preserved as the **α=1 SR PSNR baseline-to-beat** per the Phase 3 pass criterion.

- `srcnn-v7.0-pico-005` launched on 3080 Ti 2026-05-14 with `--curriculum --enable-parent-child --max-hr-crop 256 --canvas-capacity 16384` against TartanAir. 100K steps planned, ~7.4 days wall-clock at ~6.4 s/step.
- Graduated checkpoint schedule: ckpts at step 100, 500, 1000 (early-warning), then every 10K. Added `--ckpt-warmup-steps` CLI flag to `scripts/sr_train_v7.py` to support this.
- Dashboard updated: v6.2 status reflects "stopped at 74K -- baseline for v7", v7 RUN_CONFIG entry now the default-open run.

### v7 Phase 2 closed — N-D Gaussian model wiring + training scaffold (2026-05-12)

The OSS-FX pivot from inference-time canvas-scaling (H010, falsified) to a native N-D Gaussian canvas with time-slice rasterization is implemented end-to-end. Phase 0 ref rasterizer → Phase 2A canvas + spawner + model + loss + dataset → Phase 2B BackboneSpawner wiring → Phase 2C HAT-Tiny backbone swap-in + canvas pruning policy → Phase 2C closeout end-to-end training-step integration test. 61/61 v7 tests pass.

- `oss/sr/v7/`: 8 modules (nd_rasterizer, nd_canvas_state, parent_child_spawner, backbone_spawner, model, losses, intermediate_dataset; model variants for placeholder + hat_tiny + hat_small + hat_l).
- `scripts/sr_train_v7.py`: training scaffold with `--backbone-kind` CLI arg, per-rank canvas (B=1 inner loop), two-frame spawn flow (spawn at t=0 + t=2, render OSS-FX at t=1).
- `tests/sr/v7/`: 61 tests covering math primitives, state mgmt, spawner mechanics, model composition, loss components, dataset adapter, full training step.
- Closeout memo: `docs/architecture/2026-05-12-v7-phase-2-closeout.md`.
- Next: Phase 3 = v7-pico-005 training run on 3080 Ti (100K steps, ~6 d). TartanAir smoke test on remote precedes the full run.

### Docs — Teacher / student split clarified across dashboard + README + RESEARCH.md (2026-05-11)

Made it unambiguous, in every public-facing surface, that every model on the live dashboard (v5, v6.1, v6.2-pico-002, …) is a **research / teacher** model and NOT the end-user inference model. The HAT-Tiny backbone used in those runs is too expensive for real-time game upscaling: measured FP16 eager forward on RTX 3080 Ti (idle) is **54.5 ms at 270×480 LR** and **1,890 ms at 1920×1080 LR**, versus a <2 ms DLSS-/FSR-class budget. The end-user shipping model is a **≤1M-param student** distilled from these teachers (per the unanimous Phase 4 council 2026-05-08 decision and H006), exported to TensorRT FP8 with custom cross-vendor kernels. That student is not yet trained.

- `dashboard-public/index.html`: amber banner under the headline; HAT-Tiny glossary tooltip updated with measured ms and teacher-only note.
- `README.md`: TL;DR teacher/student note; v6.1 / v6.2-pico-002 status rows reframed as research/teacher; new "Distilled student (end-user inference model)" status row; Hardware-tiers table split into teacher / student columns.
- `RESEARCH.md`: teacher/student disclaimer added to the status block under the title.
- New memo: `docs/research/hypotheses/H007-hat-tiny-1080p-lr-actual-ms.md` — clean idle bench at the actual real-world LR shape.

### Rename — Ray-Retracing component branding (2026-05-07)

> **Ray-Retracing** — OSS's temporal denoising + spatial reconstruction component. We don't cast new rays; we reuse existing samples by reprojecting them via motion vectors — tracing the original camera ray's screen-space path backward through time. Same surface area as DLSS Ray Reconstruction; different algorithm (we use the persistent Gaussian canvas as the temporal accumulator rather than a learned denoiser network).

- Renamed OSS's own Gaussian denoising / ray-reconstruction-alike track to **Ray-Retracing** / **OSS Ray-Retracing**.
- Preserved NVIDIA product names such as DLSS Ray Reconstruction, DLSS-RR, DLSS RR, and `nvngx_dlssd.dll` where the repo is discussing NVIDIA's commercial product or API surface.

### Sprint 1 — CLOSED ✓ (2026-05-01)

CUDA Gaussian renderer integration. T1.1 through T1.8 complete. Heuristic dry-run review verdict: APPROVE. Bench numbers on RTX 3080 Ti: 1080p 3.3ms / 1440p 5.0ms / 4K 10.3ms across 1K–15K Gaussian counts (raster-bound, not Gaussian-count-bound). 129 pass / 2 CUDA backward fail / 3 skip on 3080 Ti; 121 pass / 4 CUDA skip on M3 Max.


### Gaussian track (new — sits alongside pixel-based modules)

A vector-based real-time game upscaler. Where DLSS and FSR work in pixels, OSS-Gaussian works in continuous 2D Gaussian primitives that warp coherently with engine motion vectors — eliminating ghosting structurally and producing frame extrapolation as a free byproduct of the same canvas. See [README.md](README.md#gaussian-track) and [design spec](docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md) for architecture.

#### Added

- **Sprint 1** — CUDA Gaussian renderer integration. Image-GS vendored as git submodule at commit `03088368`. Rasterizer wrapper with CUDA (gsplat) + PyTorch reference backend, auto-select. Forward + differentiable backward. 19 tests pass on M3 Max. CUDA backend: 7 reference tests pass on RTX 3080 Ti; 3 gsplat-specific tests fail due to gsplat 1.4.0 API drift from the Image-GS pin (debug TBD).
- **Sprint 2 scaffold** — D3D12 frame interception DLL for Cyberpunk 2077. Detours submodule vendored. NGX 10-export shim, G-buffer extractor skeleton, file-logger, CMake build for VS 2026.
- **Sprint 3** — Tile classifier. 16×16 tile complex/simple mask via gradient magnitude + depth discontinuity + motion magnitude. Pure PyTorch, CPU + CUDA. Visualization helper. 15 tests pass.
- **Sprint 4** — Gaussian param network. CovariancePriorBank (16-entry default vocabulary), GaussianParamNetwork (encoder-decoder UNet, 4 tiers: pico/lite/standard/ultra), OutputHead. 25 tests pass. Trainer wired end-to-end with composite loss (L1 + SSIM proxy), checkpoint, JSON metrics. Smoke-tested on M3 Max CPU (75K param pico tier, 3 steps in 700ms).
- **Sprint 4 datasets** — `oss/gaussian/data/`: Sintel, TartanAir, HyperSim, SRGD adapters + Mixed weighted sampler. 14 tests pass.
- **Sprint 5** — Persistent canvas. SoA tensor layout, motion warp on positions only (covariance frozen per GS-STVSR 0.99 correlation finding), per-tile error detection, prune+spawn lifecycle. 23 tests pass.
- **Sprint 6** — Frame extrapolation. α-conditioned canvas warp; rasterise at any t+α; alpha scheduler for 60→{90,120,144} fps presets. 14 tests pass.
- **Sprint 7 scaffolds** — Metal MSL kernel skeleton + Swift host + CoreML export for M3 Max; Vulkan compute kernel skeleton + ncnn export for Steam Deck. 12 tests pass + 1 conditional skip.
- **Cross-sprint integration test** — validates classifier → network → renderer → canvas → extrapolation pipeline end-to-end. 5 tests pass.
- **Code review pipeline** — `oss/gaussian/review/`: 2 reviewer agents (correctness, spec-adherence) + 1 judge agent. Anthropic SDK dispatch via `--use-api` flag (claude-sonnet-4-6, prompt caching, 429/529 backoff). Dry-run mode (heuristic verdict) by default. CLI: `python -m oss.gaussian.review.run --sprint N --commit-range A..B`.
- **Master plan + per-sprint plans + design spec + research synthesis** — `docs/superpowers/`. 7 sprint plans, design spec with graduation criterion, research synthesis from two external review batches.
- **Baseline upscalers** — `oss/gaussian/bench/baselines.py`: bicubic + lanczos implemented; FSR2 / DLSS Quality / DLSS Frame Gen as shimmed classes for Sprint 4 close-out gate. 9 tests pass.
- **Sprint 4 close-out gate** — `docs/superpowers/plans/2026-05-01-gaussian-sprint-4-closeout-gate.md`. Iso-latency comparison vs FSR 2 Quality on RTX 3080 Ti is the gate that releases Sprint 5 work.
- **CI** — split into `gaussian-track` (strict) + `pixel-track` (continue-on-error for 7 pre-existing fails). Submodules pulled at checkout.
- **3080 Ti automation** — Miniconda installed via `winget` SYSTEM scheduled task; CUDA toolkit installed via conda nvidia channel (system installer was broken with VS 2026); gsplat built with `NVCC_PREPEND_FLAGS=-allow-unsupported-compiler`. Watcher + post-cuda-build scripts in `scripts/<train-host>-*.ps1`.
- **Lambda training launcher** — `scripts/lambda_train_gaussian.py`. Dry-run only; real provision deferred to Sprint 4 production run.

#### Documentation

- README rewritten to lead with the Gaussian track + two framings ("vector-based upscaler", "3D-aware temporal accumulation").
- `docs/superpowers/welcome-back.md` — context-recovery snapshot.
- `docs/superpowers/research-synthesis-2026-05-01.md` — two external research batches consolidated into 4 plan updates.
- `docs/superpowers/integration-points.md` — pixel-based ↔ Gaussian-track integration map.
- `docs/superpowers/d3d12-hook-design.md` — Sprint 2 DLL architecture, NGX exports, OptiScaler reference.
- `docs/superpowers/code-review-pipeline.md` — review pipeline architecture + reviewer/judge prompts.
- `docs/superpowers/gaussian-canvas-design.md` — Sprint 5 SoA + prune/spawn rationale.
- `docs/superpowers/gaussian-network-architecture.md` — Sprint 4 network + bank vocabulary + tier scaling.
- `docs/superpowers/gaussian-frame-extrapolation.md` — Sprint 6 alpha-warp design + DLSS-FG comparison.
- `docs/superpowers/gaussian-port-metal.md` + `gaussian-port-vulkan-ncnn.md` — Sprint 7 cross-platform port scoping.

#### Fixed (pixel-based track repo hygiene)

- `ORD → OSSRG` / `ORU → OSS` / `PairedORS → PairedOSS` rename gaps in `oss/valuation/compare.py`, `oss/model/adapter.py`, and 8 test files. The rename was incomplete on `v0.2-dev` (commit `ffc6770`) and was breaking test collection. 88 of 95 pixel-track tests now pass; the remaining 7 failures are unrelated to this fix (onnx export, fx losses, runpod optional dep, standard-tier param budget, smoke pico, mitsuba zarr).

#### Known issues

- ~~3 CUDA-specific renderer tests fail~~. **RESOLVED**: gsplat 1.4.0 expects `xy` and `scale` in normalized [0, 1] coordinates, not pixel-space. Wrapper now normalizes internally; public API stays in pixel-space (commit `202c187`). Forward CUDA test passes on RTX 3080 Ti. Test status: **129 pass, 2 fail, 3 skip**.
- 2 CUDA backward tests still fail (`test_cuda_backend_gradients_flow`, `test_cuda_backend_optimization_converges`) — gradient flow through the normalization step likely has a fixture-value edge case (small Gaussians may not hit any tile after normalization, producing zero gradient). Targeted task: update test fixtures to use Gaussian sizes that survive normalization; wrapper logic is correct.
- The 7 pre-existing pixel-track test failures are not fixed (out of scope for the Gaussian work).
- The pixel-based track has a v0.1.0-mvp tag; the Gaussian track has not been tagged yet.

### Sprint 2 — In Progress

- **T2.1 + T2.2 ✓** — `dxgi.dll` (1.4 MB) built clean on RTX 3080 Ti via VS 2026 / MSVC 14.50 / CMake. All 10 `NVSDK_NGX_D3D12_*` exports present (verified via `dumpbin /exports`). Detours static lib linked. CMakeLists.txt was missing from origin — `.gitignore` had `*.txt` blanket rule swallowing it; carve-out added.

- **T2.3 ✓** — DXGI export forwarder (19 exports) + game-agnostic positioning. First build hit MSVC C2375 redefinition errors (system `<dxgi.h>` already declares CreateDXGIFactory etc. as `dllimport`); resolved with `.def` file rename pattern (internal C++ uses `OssgCreateDXGIFactory` etc.; `.def` aliases to public DXGI names). Build clean, dxgi.dll exports all 19 DXGI + 10 NGX + 3 PIX symbols verified via `dumpbin /exports`.

### Sprint 4 — TRACK PIVOTED (2026-05-02)

> **Result of Sprint 4:** the 2D Gaussian splat representation cannot do single-image super-resolution competitively against bicubic at our resource budget. This was triple-checked across five independent paths (see `docs/superpowers/experiments/2026-05-02-splats-cannot-SR-definitive.md`). The implementation is correct; the representation is the limit.
>
> **Pivot:** OSS-SR forks off the Gaussian track as a CNN-based super-resolver (V0.5 architecture, drop the splat dead-code). The Gaussian track redirects to OSS Ray-Retracing (denoising / DLSS-RR replacement) where Image-GS at n=1000 already beat OIDN on PSNR per the D1 memo (`docs/superpowers/experiments/2026-05-01-gaussian-denoising-naive-test.md`). Sprint 5 (persistent canvas) is suspended until the Ray-Retracing track produces a usable per-frame splat signal.
>
> **What survives:** every piece of Sprint 4 infrastructure that isn't splat-specific — `oss/gaussian/data/` (lr_synthesis, dataset adapters), `oss/gaussian/train/train.py` (DataLoader path, bicubic comparison, checkpointing, diagnostic instrumentation), `scripts/held_out_scene_probe.py`, and the lab notebook discipline. The trainer just gets a different model wired into it.

Triggered by the 5-test pre-training validation suite (`docs/superpowers/experiments/2026-05-01-validation-decision-memo.md`). Three architectural prerequisites landed before any cloud-GPU spend; live training on the 3080 Ti uncovered hyperparameter issues that, when fixed, exposed the underlying representational limit.

- **Engine-aliased LR synthesis pipeline ✓** — `oss/gaussian/data/lr_synthesis.py`. Halton(2,3) subpixel jitter (idx+1 per Unreal/DLSS convention), area-filter downsample, configurable TAA Gaussian blur (σ=0.5 mild → σ=1.5 aggressive, kernel size auto-fits 3σ), optional JPEG q≥85. `EngineAliasedLRSynth` dataclass orchestrator threaded through all four dataset adapters (sintel/tartanair/hypersim/srgd) via opt-in `lr_synth=` parameter. Default behaviour preserved (backward compat for fixtures). 28 new tests in `test_lr_synthesis.py` including directional assertions that catch sign-flips and row/col swaps; `test_apply_jitter_direction_x_axis` was self-corrected after an inverted assertion.

- **Anisotropic G-buffer-conditioned covariance bias ✓** — `oss/gaussian/network/output_head.py`. New `GBufferCovarianceBias` module: per-tile (mean normal, mean depth gradient) → 5-channel feature → zero-init linear → additive bias on bank logits before softmax. Bias is per-tile (shared across the K Gaussians in that tile). When `enable_gbuffer_bias=False` (default) or `depth=normals=None` is passed, behaviour matches the pre-existing `OutputHead` bit-for-bit. 7 new tests covering backward-compat, zero-input invariance, per-tile sharing, gradient flow, and shape validation.

- **Trainer wired to real data ✓** — `oss/gaussian/train/train.py`. Real `DataLoader` over `SintelGaussianDataset` or `SRGDGaussianDataset` (selectable via `--dataset`), `--smoke-test` mode (pico tier, batch=2, 3-hr wall clock, aggressive σ=1.5 + JPEG defaults), `--force-lr-synth` to bypass pre-baked LR (avoids the bicubic-LR-trap), `--renderer-backend {auto,cuda,reference}`, tile-aligned center-crop helper for non-multiple-of-16 datasets like SRGD (540×960 → 256×480 LR), bicubic-vs-model PSNR evaluation at `--eval-every` steps, wall-clock kill switch with orderly final-eval+checkpoint, `SMOKE TEST RESULT: PASS/FAIL` verdict line. Backward-compat `--use-synthetic-batch` preserves the random-tensor CI sanity path.

- **3080 Ti smoke-test results (architecture-validation but uncovered hyperparameter issues)** — Pico tier (75K params) on SRGD ActionRPG: model PSNR flat at 11–13 dB across 5K steps while bicubic baseline sat at 33–37 dB. Gradient probe (`scripts/probe_cuda_grad_flow.py`) confirms gradients flow on both reference and CUDA renderer backends — CUDA grads are 5–100× weaker than reference but non-zero on every leaf, expected from different forward semantics, not a backward bug. Lite tier (178K) at lr=5e-4 with aggressive multi-scene LR synth showed unstable / diverging loss (model PSNR went 13.8 → 13.2 → 7.97 over steps 1k–3k). Diagnostic conclusion: pico is undersized for 540×960 SR; lr=5e-4 is too high for lite tier; aggressive LR synth (σ=1.5 + JPEG q=85) successfully drops the bicubic ceiling from ~35 dB to ~26 dB so the model has something to optimise against. Lite-tier rerun at lr=1e-4 in progress.

- **🎯 BICUBIC GATE CLEARED (2026-05-02)** — V0.5 (lite tier + 12K-param pixel-residual head) trained on SRGD ActionRPG hits +1.47 dB above bicubic on the training scene (8/8 samples), and **+0.84 to +2.08 dB above bicubic on three held-out scenes never seen during training**: CitySample (+1.26, 16/16), StylizedRendering (+0.84, 16/16), ArchVizInterior (+2.08, 16/16). 56/56 held-out samples beat bicubic. This unblocks Sprint 5 work per the validation memo and authorises a multi-day production training run on standard tier. **Caveat:** ablation shows the splat path is decorative — the residual CNN does ~all the SR work; the 178K-param param-net produces ~constant gray (12 dB). Whether the splats start contributing at standard tier + multi-day training is the next open question. See `docs/superpowers/experiments/2026-05-02-v05-pixel-residual-success.md`.

- **V0.5 pixel-residual head** — `oss/gaussian/network/pixel_residual.py`. 3-conv CNN (~12K params) that takes (rendered HR, bicubic-upsampled LR HR) and produces a per-pixel RGB residual. Final HR = `(rendered + residual).clamp(0,1)`. Zero-init last conv so V0.5 bit-exactly matches V0 at init. Wired through trainer with `--enable-pixel-residual`; checkpoints round-trip the residual state; `evaluate_against_bicubic` honours it; `scripts/held_out_scene_probe.py` loads + evaluates on a different SRGD scene.

- **Output head init bugs found and fixed** — Two distinct dead-init bugs prevented training. (1) Zero-init weights+bias on `GaussianParamNetwork.head` produced identical output for all K Gaussians per tile → symmetric gradient → never broke. Fixed in `6900300` with N(0, 1e-3) weight init + N(0, 0.05) bias init. (2) gsplat 1.4.0's CUDA backward returns silent zero when Gaussians are too small to hit any tile — bank entry 0 (σ=1px) at scale_factor=exp(0)=1 normalises to ~1/256 on a 256-wide LR. Fixed in `6c02cc8` by adding `log(8)≈2.08` to the bias for each Gaussian's log_scale channel so initial scale_factor≈8 and Gaussians cover several tiles. Both fixes pre-requisite to V0.5 training.

- **Diagnostic instrumentation** — `_compute_diagnostics` (bank entropy, position deviation, color std) and `_param_health` (head bias/weight abs-mean and grad-norm) logged at every `--log-every` step. Surfaced both init bugs above; would have caught them within minutes had it existed sooner. Lab notebook discipline at `docs/papers/lab-notebook-discipline.md` formalises the "every run gets a memo before the result drives a decision" rule going forward.

- **Critical training-data correction implemented** — `--force-lr-synth` ensures the SRGD adapter ignores any pre-baked DownscaleData and always synthesises LR via `EngineAliasedLRSynth`. SRGD's `DownscaleData/` appears to be bicubic-downsampled, which would make bicubic upsampling its near-inverse — exactly the bicubic-LR-trap that 2U flagged in the validation memo. Smoke-test mode now hard-overrides aggressive defaults (σ=1.5, JPEG=on) to drop the baseline ceiling.

- **CUDA backward gradient flow verified** — Adds `scripts/probe_cuda_grad_flow.py`. Single forward+backward step on both reference and CUDA renderers, prints per-leaf gradient L2 norms. On the 3080 Ti both backends produce non-zero gradients on `net.stem.conv.weight`, `net.head.weight`, `head.gbuffer_bias.proj.weight`, and `bank.log_sx`. Settles the open question from Sprint 1's known issue list — the smoke-test learning failure is *not* caused by silent-zero CUDA backward; it's tier capacity + learning-rate.

#### New tests

- `tests/gaussian/test_lr_synthesis.py` — 28 tests for halton_jitter / apply_jitter / area_downsample regression / taa_blur_approx HF reduction / EngineAliasedLRSynth orchestration / JPEG round-trip / directional sign + row-col-swap detection.
- `tests/gaussian/test_train_smoke.py` — 6 tests for evaluate_against_bicubic, smoke-test arg overrides, build_dataloader sequence filtering.
- `tests/gaussian/test_network.py` — 7 new gbuffer-bias tests (default disabled, zero-init invariance, per-tile sharing across K, gradient flow, shape rejection).
- `tests/gaussian/test_datasets.py` — 1 integration test for `SintelGaussianDataset(lr_synth=...)`.

#### Open issues / next moves

- Lite tier at lr=1e-4 currently running; if it stabilises and PSNR climbs monotonically, scale to standard tier multi-day. If still diverges, investigate (1) loss function (the misnamed `ssim_proxy` is mathematically equivalent to a pooled L1, not SSIM — may be redundant or actively unhelpful), (2) gradient clipping threshold, (3) per-scene learning-rate warmup.
- Multi-day production run on 3080 Ti targets standard tier (500K params) on full SRGD GameEngineData — NOT scheduled until lite-tier stability is proven on this dataset.
- Lambda H100 cloud spend remains gated and is currently out of budget; v0 MVP must come from 3080 Ti only.
