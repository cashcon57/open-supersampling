# 2026-05-04 — Claude→Codex asks, round 4

R1 (C1–C4), R2 (C5–C8), R3 (C9–C12) — most discharged or in flight. R4 below — narrow scope, all unblocked by today's Sintel Depth fetch + pixel training run still in flight.

## C13 — Sintel held-out manifest + dual-manifest support

Status: done by Codex at 18:43 CDT. Commits: `3cfa9f9` (dual-manifest support) and `f822c89` (Sintel manifest + resolver fix).

Severity: medium (eval coverage)

Background: Sintel Depth was downloaded today and junctioned into `<train-host-data>/datasets/sintel/training/depth/` on the remote. `SintelGaussianDataset(root="<train-host-data>/datasets/sintel", scale=2.0, pass_name="clean")` now loads 1041 frame pairs cleanly. The current held-out manifest (`<train-host-data>/checkpoints/v5_held_out_manifest.json`, schema in `oss/sr/temporal/held_out_manifest.py`) is TartanAir-only.

Two parts:

**Part A — generate a Sintel-only manifest.** Either extend `scripts/sr_freeze_held_out_manifest.py` with a `--dataset-kind {tartanair,sintel}` flag, or fork it as `scripts/sr_freeze_sintel_held_out_manifest.py`. Output 64 deterministic Sintel frame pairs with `trajectory` set to the Sintel sequence dir (e.g. `<train-host-data>\datasets\sintel\training\clean\alley_1`). Default output path: `docs/superpowers/experiments/v5_held_out_manifest_sintel.json`.

**Part B — held-out script consumes both manifests.** Extend the C9 work in `scripts/sr_temporal_held_out.py` so `--manifest` accepts a comma-separated list of paths AND the script reports per-manifest result blocks (`=== Sintel held-out ===`, `=== TartanAir held-out ===`) plus an aggregate. The morning closeout uses both manifests in a single eval run.

Tests: extend `tests/sr/temporal/test_held_out_uses_manifest.py` (assuming Codex named it this in C9) with a synthetic dual-manifest case.

Constraints: do NOT regenerate the existing TartanAir manifest — it's already canonical. Final commit message suggestion: `v5-pixel(sr): Sintel held-out manifest + dual-manifest eval support`.

## C14 — Sintel fine-tune follow-up runbook

Status: done by Codex at 18:41 CDT. Commit: `a432b92`.

Severity: low (post-morning operational guide)

Background: The current pixel training run (PID 21192) is TartanAir-only. Phase 3 in the spec is "Sintel-only fine-tune at LR×0.01" — currently falling back to TartanAir because the running launch dropped `--sintel-root`. Sintel Depth is now ready on the remote.

Cash may want to run a follow-up fine-tune from the v5 ckpt (`step-00080000.pt`) on Sintel after morning eval. Deliverable: a runbook at `docs/superpowers/notes/2026-05-04-v5-pixel-sintel-finetune-runbook.md` with:

- Pre-flight: same SSH + env + dataset checks as the main runbook, plus an explicit `Test-Path <train-host-data>\datasets\sintel\training\depth` confirmation
- Launch command (WMI orphan-spawn):
  ```
  scripts\sr_train_temporal.py
      --output-dir <train-host-data>\checkpoints\srcnn-v5-pixel-sintel-finetune
      --warm-start <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\step-00080000.pt
      --sintel-root <train-host-data>\datasets\sintel
      --max-steps 100000        (bump max-steps so phase-3 has room)
      --warmup-steps 80000      (no Phase-1 warmup needed; backbone already trained)
      --joint-end 80000         (skip straight into Phase 3 Sintel fine-tune)
      --lr 1e-6                 (very small for fine-tune)
      --device cuda
      --num-workers 4
  ```
  (verify CLI args against the script — defaults may need adjustment).
- Estimated runtime: ~2-3 h on RTX 3080 Ti (20K Sintel-only steps at LR×0.01)
- Post-completion: re-run held-out eval, compare to the pre-finetune v5-pixel-temporal numbers

Constraints: docs only. Final commit message suggestion: `docs(notes): Sintel fine-tune follow-up runbook for v5-pixel-temporal polish`.

## C15 — Inspect Phase-1→Phase-2 transition logs at step 10000

Status: done by Codex at 18:49 CDT. Phase transition observed once at 18:46:11 CDT; details appended to `2026-05-04-v5-pixel-launch-status-r2.md`. One low-severity logging finding was filed and then fixed in `d5a8c55`: future Phase-2/3 rows print LPIPS components when present.

Severity: low (verification, real-time)

Background: Phase 1 → Phase 2 transition is at step 10000 in the running pixel training. At that point the script logs `phase transition: 1 -> 2 (lr=1.00e-05, backbone_frozen=False)` and the backbone unfreezes. The Phase-2 dynamics differ markedly from Phase-1 (full appearance loss + temporal-consistency loss + LPIPS, all four backbone groups receiving gradients).

Around the time the run crosses step 10000 (~19:00 CDT if rate sustains), spot-check the train log for:

- Phase transition log line appears once and only once
- Loss does NOT spike upward by > 2× immediately after (would indicate the backbone unfreeze destabilized the head)
- LPIPS values appear in the log (Phase 2 enables `--lpips-weight 0.1` per the script default)
- Step throughput does NOT drop > 50% (would indicate worker contention with the suddenly-trainable backbone)

Append a short "Phase-2 transition observed" subsection to `docs/superpowers/notes/2026-05-04-v5-pixel-launch-status-r2.md` with the actual loss numbers ±1000 steps around the transition.

Constraints: read-only spot check; do not bounce the training. If something looks wrong, file under `## Open Findings`.
