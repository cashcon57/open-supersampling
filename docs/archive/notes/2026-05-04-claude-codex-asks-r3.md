# 2026-05-04 — Claude→Codex asks, round 3

R1 (C1–C4) and R2 (C5–C8) discharged. R3 below — Cash said "all in parallel" and Codex picks whichever is least-blocked. Mark each item `claimed by Codex` / `done by Codex at HH:MM CDT` when handled. File real bugs under `## Open Findings` in `2026-05-04-v5-rolling-review.md`.

## C9 — Refactor `sr_temporal_held_out.py` to consume the manifest

Status: done by Codex at 18:34 CDT. Commit: `a472851` (`v5-pixel(sr): held-out eval consumes deterministic manifest`).

Severity: medium (eval reproducibility)

Background: `scripts/sr_freeze_held_out_manifest.py` (`ab08f73`) writes a deterministic manifest of TartanAir frame-pair indices to `docs/superpowers/experiments/v5_held_out_manifest.json`. The companion loader is `oss.sr.temporal.held_out_manifest.load_manifest`. The eval script `scripts/sr_temporal_held_out.py` currently re-picks frames via `shuffle=False` on every run — works in principle but means a re-run on the same dataset on a different machine could pick differently if the trajectory enumeration order changes (Windows vs Linux directory ordering, for instance).

Deliverable:

- Add `--manifest <path>` CLI arg to `scripts/sr_temporal_held_out.py`. When set, the script bypasses the LR-synth + DataLoader path and instead loads frames per the manifest's `(trajectory, idx_t, idx_t_plus_1)` triples. Default behavior (no `--manifest`) is unchanged.
- The first time the script runs WITH `--manifest`, it asserts that `manifest["lr_scale"]` matches `args.scale` and `manifest["lr_synth_args"]` matches the script's LR-synth config — bail with a clear error on mismatch.
- Add a unit test at `tests/sr/temporal/test_held_out_uses_manifest.py` that exercises the manifest-load path against a tiny synthetic manifest + stub dataset.

Constraints: do NOT change the no-manifest default. The running pixel training will produce a checkpoint at ~03:00 CDT, and the morning eval will use the manifest path; both code paths must coexist. Final commit message suggestion: `v5-pixel(sr): held-out eval consumes deterministic manifest`.

## C10 — `scripts/sr_export_temporal_onnx.py` scaffold

Status: done by Codex at 18:34 CDT. Commit: `31aad5b` (`v5-pixel(sr): ONNX export script for stateless temporal wrapper + smoke test`).

Severity: low (S6 prep — won't run until v5 ckpt lands)

Background: The stateless wrapper `oss.sr.temporal.stateless_export.TemporalSRModelStateless` (`726b629` + `b4f4023`) is in place. The matching ONNX export script is not. Once the pixel training finishes overnight, we'll want to immediately attempt the ONNX export to catch any opset-17 issues early.

Deliverable:

- `scripts/sr_export_temporal_onnx.py` taking `--ckpt <path>`, `--output <path>`, `--lr-h`, `--lr-w` (LR resolution for the export), `--opset 17`. Loads the temporal ckpt via `TemporalSRModelStateless.from_temporal_checkpoint`, sets `model.train(False)`, runs `torch.onnx.export` with the 5 named inputs (`lr_inputs, prev_hr_input, depth_hr_curr, depth_hr_prev, motion_lr`) + 2 named outputs (`out_hr, disocclusion_mask`), dynamic axes on H_lr/W_lr.
- A subprocess test at `tests/sr/temporal/test_onnx_export_smoke.py` that constructs a tiny TemporalSRModel, saves a synthetic ckpt, runs the export script with `--lr-h 64 --lr-w 64`, asserts the ONNX file is produced + `onnx.checker.check_model` passes.
- `--help` works on a vanilla python (lazy-import torch + onnx).

Constraints: do NOT actually try to load `srcnn-prod-v4-lpips` or any real ckpt — synthetic tiny ckpts only for the test. The real export against the v5 ckpt is a manual follow-up by Cash. Final commit message suggestion: `v5-pixel(sr): ONNX export script for stateless temporal wrapper + smoke test`.

## C11 — Pico-tier distillation design memo (post-v5 perf prep)

Status: done by Codex at 18:34 CDT. Commit: `1355110` (`docs(notes): pico-tier distillation design memo for S6 prep`).

Severity: low (S6 prep)

Background: The v5-pixel-temporal model is 626K params (standard tier). Steam Deck and integrated GPUs need a Pico tier (~150K params target). Distillation from v5 is the standard recipe but we haven't designed the pipeline yet.

Deliverable: `docs/superpowers/notes/2026-05-04-pico-distillation-design.md` covering:

- Architecture sketch for `oss/sr/temporal_pico.py` — same `TemporalSRModel` interface but with `srcnn_for_tier("pico")` (16ch, 2 blocks per `oss/sr/cnn.py`'s `SR_TIER_CONFIGS`) and a smaller `TemporalHead` (e.g., 16-ch hidden instead of 32). Disocclusion gate stays the same (3 scalars).
- Distillation loss: `L_l1(pico_out, teacher_out) + 0.1·L_perceptual(pico_out, teacher_out)`. No GT supervision in the distillation phase; the teacher IS the GT.
- Training data: same TartanAir mix; teacher renders precomputed once per frame (offline), pico trains against those.
- Schedule: ~50K steps, ~6h on RTX 3080 Ti.
- Open questions: (a) does pico need a smaller disocclusion gate too, or are 3 scalars universal? (b) does the 1-frame history budget still apply, or could pico use a more aggressive temporal accumulation since its per-frame cost is lower?

Constraints: docs only. Final commit message suggestion: `docs(notes): pico-tier distillation design memo for S6 prep`.

## C12 — Vendor port stub smoke imports

Status: done by Codex at 18:34 CDT. Commit: `c437cb2` (`tests(ports): smoke-import vendor port scaffolds with platform skips`).

Severity: low (sanity check — won't catch real bugs but will catch broken Python)

Background: `oss/gaussian/ports/metal/` exists per the README ("scaffolded; not wired to real games"). Other vendor backends (DirectML, MIGraphX, OpenVINO) may have stubs scattered around. Cash has noted these are scaffolds, not validated.

Deliverable:

- Walk `oss/gaussian/ports/` and any other directory containing vendor scaffolds (`oss/sr/ports/` if it exists). For each stub module, write a tiny smoke import test at `tests/gaussian/test_port_stubs_import.py`. Each test does `importlib.import_module(name)` and asserts the module loads without error. Skip with a clear message if the platform doesn't support the import (e.g., `coremltools` only on macOS).
- Report (in your final memo) which stubs exist, which import cleanly on the test host, and which need explicit platform guards.

Constraints: do NOT alter the stub modules themselves — only add the test file. Final commit message suggestion: `tests(ports): smoke-import vendor port scaffolds with platform skips`.
