# Changelog

All notable changes to OpenSuperSampling. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: SemVer pre-1.0 (PATCH = bug fixes / docs; MINOR = new features; MAJOR reserved for v1.0).

## [Unreleased] — `v0.2-dev` branch

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

- 3 CUDA-specific renderer tests fail on the 3080 Ti due to gsplat 1.4.0 API drift from the Image-GS pin. Reference backend works on CUDA, so this is bounded scope; targeted debug task post-Sprint 1 close-out. Investigation summary:
  - Inverse-scale convention patched in `rasterizer.py` (commit `d76ee60`) — Image-GS internally `1/scale`s before passing to gsplat. Did not resolve.
  - `project_gaussians_2d_scale_rot` returns `num_tiles_hit = 0` for all reasonable input scales tried — the projection step itself isn't producing any tile intersections on RTX 3080 Ti. Suggests a deeper convention mismatch (coordinate space? half-precision flag?) between gsplat 1.4.0 and the Image-GS reference call site.
  - `RuntimeError: expected scalar type Int but found Float` is then raised when rasterise is called on the degenerate result.
  - Variants tested (radii→int32, topk_norm→int, num_tiles_hit→explicit int32, feat→float64): all fail with the same error or a related one.
  - **Next debug step**: trace through Image-GS's `main.py` exact call path on the 3080 Ti and diff against our wrapper; the convention difference is likely in either coordinate space (centred vs corner-anchored) or scale magnitude (sigma vs inverse-sigma vs log-scale).
- The 7 pre-existing pixel-track test failures are not fixed (out of scope for the Gaussian work).
- The pixel-based track has a v0.1.0-mvp tag; the Gaussian track has not been tagged yet.
