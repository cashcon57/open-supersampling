# Codex orientation — OpenSuperSampling

Read this once at the start of every Codex session. It compresses the project state into the minimum context a fresh reviewer needs.

## What OSS is

Open-source real-time super-resolution and frame extrapolation for games. Cross-vendor (NVIDIA / AMD / Apple / Intel / Steam Deck), no SDK contract, no vendor lock-in. Designed to drop in as a DLL replacement for any game already using DLSS, FSR, or XeSS. Pre-alpha. Active research. Repo at `<repo-root>`.

The full README is at `README.md`. The paper-style overview for academic readers is `RESEARCH.md`. The bibliography is `BIBLIOGRAPHY.md`. Citation file is `CITATION.cff`.

## Branch + version

Active development branch: `v0.2-dev`. Always commit + push there.

| Version | State |
|---|---|
| v3 | Earlier SRCNN baseline, shipped, deprecated. |
| v4 | Single-frame SR-CNN, trained, exported. ~30 dB / 0.30 LPIPS on SRGD held-out. |
| v5-pixel-temporal | Validated baseline as of 2026-05-06. PSNR 25.703 / LPIPS 0.1666 on TartanAir held-out (oldtown), 64/64 beats bicubic, temporal stability 0.337× v4. |
| v5-Gaussian-temporal | Implemented but skipped (Option A taken on 2026-05-06). |
| v6 | Architecture locked, modules landed, V6Model orchestrator landed (commit `7ad1033`). Training loop is the next commit. |

## v6 architecture — what's implemented

The canonical design memo: `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md`.
The implementation roadmap: `docs/research/2026-05-05-v6-external-baselines-integration-plan.md`.
The source-extraction notes (architectural details from external repos): `docs/research/2026-05-06-v6-source-extraction-notes.md`.

Code lives at `oss/sr/v6/`. Module index is `oss/sr/v6/__init__.py`. Module-by-module:

| Module | Status | Tests |
|---|---|---|
| `hat.py` (HAT-Tiny / HAT-Small / HAT-L backbones) | shipped | 6 |
| `cross_attention.py` (PixelGaussianFusion with ROPE) | shipped | 7 |
| `covariance_resampling.py` (GS-STVSR Σ resample) | shipped | 12 |
| `st_variation_score.py` (4DGS-1K pruning score) | shipped | 13 |
| `keyframe_active_mask.py` (4DGS-1K keyframe mask cache) | shipped | 15 |
| `losses.py` (Charbonnier + LPIPS + multi-scale VGG + wavelet L1 + Sobel + GAN hinge) | shipped | 11 |
| `discriminator.py` (UNetD per Real-ESRGAN) | shipped | 5 |
| `ema.py` | shipped | 6 |
| `schedules.py` (cosine + warm restarts) | shipped | 8 |
| `dataset.py` (TartanAir + Hypersim 60/30 mix) | shipped | 8 |
| `patch_sampling.py` (70/30 importance/uniform) | shipped | 10 |
| `aa_perpendicular_dilation.py` (AAA-Gaussians Eq. 10) | shipped | 15 |
| `aa_view_space_angular.py` (AAA-Gaussians Eqs. 14-17) | shipped | 12 |
| `aa_2dgs_object_space_mip.py` (AA-2DGS Mip filter) | shipped | 14 |
| `aa_analytic_splat.py` (Analytic-Splatting CDF integral) | shipped | 13 |
| `model.py` (V6Model orchestrator) | shipped | 15 |
| `test_integration_smoke.py` (cross-module bf16 + edge cases) | shipped | 9 |

Total v6 tests: 184, all passing. Run with `./venv-py312/bin/python -m pytest tests/sr/v6/ -q`.

## Conventions in this codebase

- **Lab-notebook discipline.** Every result that drives a decision gets a memo at `docs/superpowers/experiments/YYYY-MM-DD-<slug>.md` BEFORE the result drives the decision. Index: `docs/papers/experiments-index.md`.
- **Honest framing.** No "honest" / "genuinely" / "killer" / "real" / "actual" as performative qualifiers. State numbers cold or flag them as estimates. No emojis in committed files.
- **bf16 mixed precision.** All v6 training is bf16 mixed precision (Ampere+ supports natively, more numerically stable for GAN than fp16). Pretrained backbones (VGG-19 in multi-scale loss, LPIPS-VGG) are forced to fp32 internally then cast back; the composite loss is bf16-safe end-to-end.
- **DDP-safe.** Model parameters are the only DDP-synced state. Module-level mutable state (canvas, ST score, keyframe mask) is per-rank-local. `find_unused_parameters=True` is set in the v5 trainer; v6 trainer follows the same pattern.
- **Conditional G-buffer zeroing.** v4 was trained on SRGD with depth/motion/normals = 0. The v5 `TemporalSRModel` has a `zero_gbuffer_into_backbone` flag that defaults to True for warm-started runs (SRGD-distribution match) and False for from-scratch runs (real G-buffers). v6 trains on real G-buffers from the start; default is False.
- **Per-vendor inference precision.** v6 ships at FP8 on Ada+ / RDNA4+ / Arc-B-series; FP16 elsewhere; INT8 / dp4a Vulkan compute on Steam Deck (RDNA2). Documented at `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md` §4.5.
- **Output activation.** Default softplus (HDR-capable, unbounded non-negative). Sigmoid is opt-in for SDR-only callers.
- **No new pip deps without flagging.** Reuse existing pyproject deps.
- **Match commit message style.** See `git log --oneline | head -50` for examples. Commit message body explains WHY; subject is concise.

## Key tooling

| Command | Purpose |
|---|---|
| `./venv-py312/bin/python -m pytest tests/sr/v6/` | Run all v6 tests |
| `./venv-py312/bin/python -m pytest tests/ -m "not gpu and not mitsuba" --ignore=tests/gaussian` | Run pixel-track tests |
| `./venv-py312/bin/python -m pytest tests/gaussian/ -m "not gpu"` | Run Gaussian-track tests |
| `git log --oneline -20` | Recent commits |
| `gh run list --limit 5` | CI status |

## Coordination protocol

Codex sessions write findings to `docs/coordination/codex-review-YYYY-MM-DD-<slug>.md`. One file per review session. Each finding lists:

- Severity (HIGH / MEDIUM / LOW)
- File and line range
- Description of the issue
- Suggested fix

Claude reads these and either acts on them or notes why an item is wontfix (and pushes the wontfix-rationale to the same file as a reply). All review work is committed to `v0.2-dev`.

## Honest limits

- v6 training loop is not yet implemented. Constructing V6Model and running --smoke works; actually training does not.
- Gaussian-canvas write path (mutating canvas across forward calls) is not yet wired in V6Model; the canvas stays empty for all forwards in the current code, meaning the cross-attention layer is effectively identity-passthrough today.
- AA-stack rasterizer modifications are PyTorch reference implementations; production CUDA kernels are a separate ~6-12-month engineering project.
- Real DLSS / FSR / XeSS comparison requires the S7 DLL-shim runtime, which is unbuilt.

## Where to push back

If a Codex review finds something Claude implemented that violates the conventions above, file it as a finding. If Claude has documented a deviation already (e.g. softplus instead of sigmoid), note the deviation and confirm or flag it.

If a Codex review reports a bug in code Claude wrote, prefer suggesting the fix — but verify that it doesn't conflict with the v6 architectural memo before recommending.

If you find an issue in tests that aren't catching a real bug, propose the missing test.
