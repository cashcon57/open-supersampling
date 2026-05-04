# 2026-05-04 — v5 Rolling Review

**Status:** Active living document  
**Purpose:** Shared rolling review surface for Sprint 5 dual-track implementation planning and code review. Claude/Codex agents should read and update this file before dispatching implementation or reviewer subagents.  
**Last updated:** 2026-05-04 16:27 CDT

**Watcher:** Codex (review) / Claude (implementer-controller)

## Current State

Working tree baseline:

- Branch: `v0.2-dev`
- Repo path: `<repo-root>`
- Remote: `https://github.com/cashcon57/open-supersampling.git`
- Handoff path `<home>/open-reconstruction-suite` does not exist on this machine.

Pre-existing untracked local artifacts, not attributed to the current Sprint 5 agent unless modified:

- `OpenSuperSampling.code-workspace`
- `docs/superpowers/experiments/2026-05-01-gaussian-denoising-naive-test-images/`
- `docs/superpowers/experiments/assets/`

Sprint 5 planning artifacts created during this watch:

- `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md`
- `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md.tasks.json`
- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md`
- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md.tasks.json`

Latest observed hashes for active Sprint 5 files:

```text
a96a41171864e7a61fca9945884d06a9ba23e21929ab6b93c22551e0ec7bd961  docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md
d8ff616d796eb58860bfa03dbb3a3ec3372e049e7205a57b2b9ea00fe8c8dc89  docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md.tasks.json
c0fb77be7a61f0bac0c34319e6782f613f1c4ae764315bceab44af286eb47533  docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md
23005f82f48ea81eb9579dfaeff122514761c323cf784fbda91dda111945d286  docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md.tasks.json
d3bdf697fd9adaa479cd87af6b7c18fe0ab0408deb219f300a4d59095a450303  oss/sr/temporal/warp.py
029945f1d1fb64a1e4d383d16811c52da06be8c2457f7d9d18a861b835763efe  oss/sr/temporal/disocclusion.py
b30b3cce0bada0f5d9a7053e6aef05456d9db15d82a1af634453b7513afce6a9  oss/sr/temporal/temporal_head.py
6cd990a49cfbfa009c2e87df677ad54c25675dd2b3fd8c7eb38ce7f6ebf7f244  oss/sr/temporal/model.py
554f70dcf2b758643f7eb23d0d7455f2c2098e93591a6e5df330588b49e53c6f  oss/sr/temporal/dataset.py
1d69b6a82326dda8879194bd5076ea084b8ad18947a4f6fecc40c2ef87f2f9b6  oss/sr/temporal/__init__.py
eaf65423b82f310ffccf29fa0f56f4f4ed4ac0b171b58cc06341b8e4aaa3fc70  oss/sr/inference.py
31be748fdbf384205b1c127bd6ae8c555e6b9ba8db689cc4b7c3c385b601b14b  scripts/sr_train_temporal.py
38ef0d2bdabef9cd178fc3154a7a499165489f7fbdec9c63bd49c6d69b85b37b  scripts/sr_temporal_held_out.py
060a9361ca2be77680baac4e699e0c29514c7eba6d1846672489341f3cb46a9d  oss/sr/gaussian_temporal/analytical_warp.py
997231cb12f6abaf3f10cf161a182fa551bbca41cd4879458049dc8251585a01  oss/sr/gaussian_temporal/densification.py
b6887cd0622263778a035ea3f83335fc6babe244fd08ab77b1155a8e01310955  oss/sr/gaussian_temporal/g_buffer_encoder.py
99a38a13105e09a78c2cb4af1d066705f044828382e4226c77e09b0ad7307fe1  oss/sr/gaussian_temporal/gaussian_field.py
d4c35220845fc76d024afd7acc8cc132a5b6959de384897625d93c9dfe816057  oss/sr/gaussian_temporal/pruning.py
d618dddeda39c2d2f625cc02ddebc1fc35e4b3148759e09880923d33cd574a51  oss/sr/gaussian_temporal/rasterizer.py
42bb0fa6f1119a3155018008340a1d50eeeb3561b1d566ef775d91a50ef4ce9b  oss/sr/gaussian_temporal/regularization.py
478c4be7f4ba9b9d4a3f3f01f262f7e5767ef7a45648a9894cf9efc27ba36ada  oss/sr/gaussian_temporal/transformer.py
2b6874b088bd3167c94f644e92a4f4ce2b5b728f1a725dcfa8a90a03254325d3  oss/sr/gaussian_temporal/model.py
f4d20aac0cf78e734025e284d3546a6b66047ac95abb46d7d78a7f80ff2b0b4b  oss/sr/gaussian_temporal/__init__.py
2534dc26dc5bc507cb815160095aaf67f18c038f4d85348a8f8f676f618d6a50  tests/sr/temporal/test_inference_state.py
a9cc062811f3b5b6a6e73989c006404ec5b2c20cf5f829249dc0aa56f0b170f5  tests/sr/temporal/test_train_smoke.py
187f5d09bae23a0ca8625a7240b10c31c1b6125151f548f4f08f7aa639b3a298  tests/sr/temporal/test_held_out_argparse.py
a6522b387e5cc31bc2c29f639e0efb9f1d9f6e80bc3696487017a7796163030e  docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out-template.md
b9a46341fda4b46d15d62fe406c30567012f359998ccbbe264df62325426891d  tests/sr/temporal/test_loss_pipeline.py
f0183c261e44355c28498c69ded22cd2e5d0ab0ed659a761afc2e8f79900becb  tests/sr/temporal/test_dataset.py
c51c9195ac4494bbc6c5e86836ed924bed160f88c9e51b2c1bfb00d8ad84f54c  tests/sr/gaussian_temporal/test_pruning.py
150708aaae5f03ba996b7c34f86f1e6f16e5ad599ae8afbf98fd73bae9ee9ebf  tests/sr/gaussian_temporal/test_rasterizer_wrapper.py
9e0ec8fe2910cd7626d7b8f0f46b4fc1d18f0e077f25213fc163876dcf452694  tests/sr/gaussian_temporal/test_regularization.py
24d2250c40175832b56e4d8bf963570384c23ad1f4150a0b3ac9641c6f133fc7  tests/sr/gaussian_temporal/test_model_full_step.py
e619e9f695e3e21005ea392c5d90f93569fdb7b907d15868ef6e72e95d169efb  tests/sr/gaussian_temporal/test_analytical_warp.py
3614b8fac4f1a6e597f78197b2fcd06b050eeae01f3124fe9135703f9c782017  tests/sr/gaussian_temporal/test_transformer.py
```

Implementation files now present:

- `oss/sr/temporal/__init__.py`, `oss/sr/temporal/warp.py`
- `oss/sr/gaussian_temporal/__init__.py`, `oss/sr/gaussian_temporal/gaussian_field.py`
- `tests/sr/temporal/__init__.py`, `tests/sr/temporal/test_warp.py`
- `tests/sr/gaussian_temporal/__init__.py`, `tests/sr/gaussian_temporal/test_gaussian_field.py`
- Pixel Task 1 committed: `oss/sr/temporal/disocclusion.py`, `tests/sr/temporal/test_disocclusion.py`
- Gaussian Task 1 committed: `oss/sr/gaussian_temporal/analytical_warp.py`, `tests/sr/gaussian_temporal/test_analytical_warp.py`
- Pixel Task 2 committed: `oss/sr/temporal/temporal_head.py`, `tests/sr/temporal/test_temporal_head.py`
- Gaussian Task 2 committed: `oss/sr/gaussian_temporal/g_buffer_encoder.py`, `tests/sr/gaussian_temporal/test_g_buffer_encoder.py`
- Pixel Task 3 committed: `oss/sr/temporal/model.py`, `tests/sr/temporal/test_model.py`
- Gaussian Task 3 committed: `oss/sr/gaussian_temporal/transformer.py`, `tests/sr/gaussian_temporal/test_transformer.py`
- Pixel Task 4 committed: `oss/sr/temporal/dataset.py`, `tests/sr/temporal/test_dataset.py`
- Gaussian Task 4 committed: `oss/sr/gaussian_temporal/densification.py`, `tests/sr/gaussian_temporal/test_densification.py`
- Pixel Task 5 committed: `tests/sr/temporal/test_loss_pipeline.py`
- Gaussian Task 5 committed: `oss/sr/gaussian_temporal/pruning.py`, `tests/sr/gaussian_temporal/test_pruning.py`
- Gaussian Task 1 and Task 3 fixes committed: `0618e46`
- Pixel Task 6 committed: `oss/sr/inference.py`, `tests/sr/temporal/test_inference_state.py`
- Gaussian Task 6 committed: `oss/sr/gaussian_temporal/rasterizer.py`, `tests/sr/gaussian_temporal/test_rasterizer_wrapper.py`
- Gaussian Task 7 committed: `oss/sr/gaussian_temporal/regularization.py`, `tests/sr/gaussian_temporal/test_regularization.py`
- Pixel Task 7 committed: `scripts/sr_train_temporal.py`, `tests/sr/temporal/test_train_smoke.py`
- Pixel Task 8 committed: `scripts/sr_temporal_held_out.py`, `tests/sr/temporal/test_held_out_argparse.py`, `docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out-template.md`
- Gaussian Task 8 committed: `oss/sr/gaussian_temporal/model.py`, `tests/sr/gaussian_temporal/test_model_full_step.py`
- Working-tree fixes present but not committed: Gaussian Task 8 history-buffer fix; Pixel Task 4 `pair_stride` fix.

Targeted completed-task tests pass under the working local env:

```bash
venv-py312/bin/python -m pytest tests/sr/temporal/test_warp.py -v
venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_gaussian_field.py -v
```

Results:

- `tests/sr/temporal/test_warp.py` → 3 passed in 0.63s
- `tests/sr/gaussian_temporal/test_gaussian_field.py` → 5 passed in 0.51s
- `tests/sr/temporal/test_disocclusion.py` → 5 passed in 0.51s
- `tests/sr/gaussian_temporal/test_analytical_warp.py` → 5 passed in 0.55s after working-tree fix
- `tests/sr/temporal/test_temporal_head.py` → 4 passed in 0.50s
- `tests/sr/gaussian_temporal/test_g_buffer_encoder.py` → 3 passed in 0.50s
- `tests/sr/temporal/test_model.py` → 5 passed in 0.54s
- `tests/sr/gaussian_temporal/test_transformer.py` → 4 passed in 0.56s after working-tree fix
- `tests/sr/temporal/test_dataset.py` → 4 passed in 0.47s
- `tests/sr/gaussian_temporal/test_densification.py` → 3 passed in 0.48s
- `tests/sr/temporal/test_loss_pipeline.py` → 1 passed in 1.46s with 2 torchvision deprecation warnings
- `tests/sr/gaussian_temporal/test_pruning.py` → 3 passed in 0.54s
- `tests/sr/temporal/test_inference_state.py` → 3 passed in 0.55s
- `tests/sr/gaussian_temporal/test_rasterizer_wrapper.py` → 4 passed in 0.47s
- `tests/sr/gaussian_temporal/test_regularization.py` → 11 passed in 0.45s
- `tests/sr/temporal/test_train_smoke.py` → 1 passed in 1.32s
- Direct smoke command `venv-py312/bin/python scripts/sr_train_temporal.py --smoke --device cpu --max-steps 5 --output-dir /tmp/oss_smoke_temporal_review` exited 0, printed finite `final_loss=2.357021`, and wrote `metrics.json`, `score_log.json`, and `step-00000005.pt`.
- `tests/sr/temporal/test_held_out_argparse.py` → 1 passed in 1.12s
- `tests/sr/gaussian_temporal/test_model_full_step.py` → 5 passed in 0.55s
- Combined working-tree fix verification: `tests/sr/gaussian_temporal/test_model_full_step.py tests/sr/temporal/test_dataset.py tests/sr/temporal/test_held_out_argparse.py` → 13 passed in 1.32s
- Extra spec probe for Gaussian analytical warp identity preservation now passes and is committed in `0618e46`.
- Extra spec probe for Gaussian transformer gradient flow now passes and is committed in `0618e46`.
- Extra spec probe for pixel `SequentialPairDataset(base, pair_stride=2)` is now covered by tests and passes in the working tree.

Verification caveat: default `python3` and `.venv/bin/python` do not have `torch` or `pytest`; `venv/bin/python` has `pytest` but not `torch`. Use `venv-py312/bin/python` for local CPU tests. For CUDA/PyTorch-heavy verification, Cash notes that PyTorch is also available on at least one Tailnet machine plus the RunPod and Lambda instances used by the project.

Commits on `v0.2-dev` (not pushed):
- `0755e9b` v5-pixel(sr): add held-out eval + memo template
- `9d66af1` v5-gaussian(sr): wire full GaussianTemporalSRModel pipeline
- `1557522` v5-pixel(sr): add training entry with 3-phase schedule + smoke test
- `8db02a8` v5-gaussian(sr): add Gaussian regularization (drift + area + count)
- `6323a9b` v5-gaussian(sr): add rasterizer wrapper around existing renderer
- `e15324f` v5-pixel(sr): add stateful TemporalSRInferenceEngine + scene-cut reset
- `0618e46` v5-gaussian(sr): fix two HIGH Codex findings — head init + identity-J preservation
- `1a09ccc` v5-gaussian(sr): add opacity + count pruning
- `b3fbc1c` v5-pixel(sr): add end-to-end loss pipeline integration test
- `53fda3c` v5-pixel(sr): add SequentialPairDataset + tartanair/sintel shims
- `62234d2` v5-gaussian(sr): add residual-driven densification (heuristic v5)
- `1437656` v5-gaussian(sr): add multi-frame transformer with RoPE on Gaussian mu
- `9d1a256` v5-pixel(sr): add TemporalSRModel with v4 warm-start + freeze toggle
- `7225b20` v5-gaussian(sr): add tile-level G-buffer encoder
- `e912bf5` v5-pixel(sr): add temporal-head conv stack with near-identity init
- `8bb7693` v5-gaussian(sr): add analytical warp — mu shift + covariance Jacobian
- `a993cda` v5-pixel(sr): add disocclusion gate with learnable alpha/beta/gamma
- `f9f4fd5` sprint5(sr): fix stale arg doc on warp_prev_hr + rolling-review update
- `f00f7a4` sprint5(plans): patch reviewer findings on warp doc + Gaussian renderer + first-frame render
- `2d315e1` v5-pixel(sr): add motion-vec upsample + backward HR warp helpers
- `0820439` v5-gaussian(sr): add GaussianField SoA + history container

Tracked pixel Task 8 and Gaussian Task 8 code are committed. Two earlier review findings are fixed in the working tree but not committed yet: Gaussian Task 8 history population and Pixel Task 4 `pair_stride`. The open implementation findings are the pixel temporal motion-vector off-by-one and pixel Task 7 dashboard score schema mismatch.

## Tasks for Codex

Cash authorized Claude (controller) to assign verification work here. Codex picks these up on its monitoring cadence; mark each item as "claimed by Codex" or "done" with a brief note when handled.

Active asks:

- **C1 — Verify history-buffer ordering after frame N≥6.** Claude added `_history.appendleft` ordering so newest is first; ran `test_history_populates_across_frames` covering frames 0..7 (cap at 5). Independent probe wanted: take the field returned at frame 7, render its history into the transformer, and confirm the **newest-first** invariant holds (i.e., `history[0]` is the snapshot from frame 6, `history[1]` is frame 5, etc.). Ref: `oss/sr/gaussian_temporal/model.py:131-145`.
- **C2 — Audit pixel-temporal flow direction at the held-out level.** Flow fix `38cf507` updated `sr_train_temporal.py` + `sr_temporal_held_out.py` to use `t_motion` for the t→t+1 warp. Independent probe wanted: synthesize a 2-frame pair where `t_motion` and `tp1_motion` differ in sign or magnitude, run `sr_temporal_held_out.py` on a synthetic `TemporalSRModel`, and verify that the temporal-stability metric is consistent with `t_motion`-aligned warping (would diverge if `tp1_motion` were used).
- **C3 — Spec-compliance review of Pixel Tasks 0–9.** Each task in `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md` has explicit acceptance criteria. Walk through Tasks 0–9 and report: (a) any acceptance criterion not covered by the committed test file, (b) any deviation that's documented but not tested. Pixel Task 10 (closeout) is post-training; skip.
- **C4 — Spec-compliance review of Gaussian Tasks 0–9.** Same protocol as C3 against `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md`. Tasks 10–14 are upcoming; skip.

If a probe finds a real bug, file it under `## Open Findings` with severity + file:line citations as you've been doing. Claude will patch.

## Review Gate

Do not dispatch implementation workers from stale `.tasks.json` files. Both JSON files were generated before later plan edits and did not update when the source plan docs changed.

Preferred next step:

1. Clean the remaining plan nits below.
2. Regenerate both `.tasks.json` files from the corrected plan docs, or explicitly instruct implementers to ignore the JSON and read the plan docs directly.
3. Begin pixel implementation first unless Cash explicitly chooses parallel track work.

## Open Findings

### Pixel temporal uses next-frame flow for t→t+1 alignment

Severity: high

The pair dataset returns `t_motion` from frame `t` and `tp1_motion` from frame `t+1`. TartanAir stores `flow/000000_000001_flow.npy` as forward flow to the next frame, and Sintel `.flo` is likewise the flow for the current frame to the next frame. Therefore, for a `(t, t+1)` pair, the flow that aligns `out_t` to `out_{t+1}` is `t_motion`, not `tp1_motion`.

Current affected code:

- `scripts/sr_train_temporal.py:372` builds `x_tp1` with `p_motion`.
- `scripts/sr_train_temporal.py:376` passes `motion_lr=p_motion` when rendering `out_tp1` from `prev_hr=out_t`.
- `scripts/sr_train_temporal.py:396` computes `temporal_consistency_loss(out_tp1, out_t, p_motion, ...)`.
- `scripts/sr_temporal_held_out.py:240` renders temporal `t+1` with `prev_hr=base_out_t.detach()`.
- `scripts/sr_temporal_held_out.py:252` passes `motion_lr=p_motion` for that render.
- `scripts/sr_temporal_held_out.py:262`/`:263` warp `out_t` to `out_t+1` using `p_motion`.

Why this matters:

- The v5 pixel spec says the temporal model consumes motion vector `t-1 -> t`.
- For the current frame `t+1`, that is the pair's `t_motion`.
- Using `tp1_motion` instead uses `t+1 -> t+2`, corrupting temporal-head alignment, disocclusion input, the temporal-consistency loss, and the held-out temporal-stability metric.

Fix direction:

- For the second frame in a pair, pass `motion_lr=t_motion` to `TemporalSRModel` and `temporal_consistency_loss`.
- In held-out eval, use `t_motion` for temporal `t+1` render and for `warp_prev_hr(out_t, ...)`.
- Add a regression test with synthetic pair motion where `t_motion != tp1_motion`, asserting the evaluator/training step uses `t_motion` for `t -> t+1`.

### Pixel Task 7 score_log dashboard schema mismatch

Severity: medium

The committed training script passes the smoke test and writes `metrics.json` plus `score_log.json`, but the `score_log.json` row schema does not match `scripts/training_dashboard.py`.

- `scripts/sr_train_temporal.py:672` writes rows with `step`, `loss`, `phase`, `psnr`, and `lpips`.
- `scripts/training_dashboard.py:365` reads `model_psnr_mean`, `bicubic_psnr_mean`, and `model_beats_bicubic_count` for the latest eval card.
- `scripts/training_dashboard.py:431` charts `model_psnr_mean` / `bicubic_psnr_mean`; LPIPS chart similarly expects `model_lpips_mean` / `bicubic_lpips_mean`.

Observed from direct smoke:

- `/tmp/oss_smoke_temporal_review/score_log.json` is non-empty, so the dashboard will enter the "latest eval" path.
- The expected fields are missing, which will produce undefined/NaN eval-card values and empty PSNR/LPIPS charts.

Fix direction:

- Either leave `score_log.json` empty until Task 8 writes real held-out eval rows, or emit dashboard-compatible placeholder/eval rows with the keys the dashboard actually reads.
- If Task 7 keeps approximate smoke PSNR, store it in `metrics.json` train rows or add dashboard support for temporal training rows without pretending they are bicubic-vs-model eval rows.

### Stale .tasks.json sidecars

Severity: low (does not block dispatch — implementer subagents are instructed to read the plan `.md` directly).

Fix: regenerate the task JSON files before any cross-session resume via `/superpowers-extended-cc:executing-plans`.

## Resolved Findings

### Working-Tree Fixes Pending Commit

Resolved in working tree, pending commit: high severity Gaussian Task 8 history buffer.

The earlier probe showed recurrent rollouts kept `len(history) == 0`, so the Gaussian transformer never received multi-frame Gaussian context. Current working tree now pushes prior-field snapshots before returning `new_field`:

- `oss/sr/gaussian_temporal/model.py:131` populates history.
- `tests/sr/gaussian_temporal/test_model_full_step.py` adds `test_history_populates_across_frames`.
- Verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_model_full_step.py -v` is included in the 13-test combined pass.

Resolved in working tree, pending commit: medium severity Pixel Task 4 `pair_stride` API gap.

The earlier probe showed `SequentialPairDataset(_FakeBase(), pair_stride=2)` raised `TypeError`. Current working tree now accepts `pair_stride`, validates it, and excludes pairs crossing trajectory boundaries:

- `oss/sr/temporal/dataset.py:34` adds `pair_stride: int = 1`.
- `oss/sr/temporal/dataset.py:45` builds pairs using `i + pair_stride`.
- `tests/sr/temporal/test_dataset.py` adds stride-2 and invalid-stride coverage.
- Verification: `venv-py312/bin/python -m pytest tests/sr/temporal/test_dataset.py -v` is included in the 13-test combined pass.

### Gaussian Implementation

Resolved: high severity Task 1 analytical-warp identity preservation.

The earlier extra probe showed identity flow changed `log_scale` and `rotation` for rotated unequal-scale fields. Commit `0618e46` now preserves original `log_scale` and `rotation` when the sampled Jacobian is identity/pure translation, and adds `test_identity_flow_preserves_rotated_unequal_scale`. Verification:

- `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_analytical_warp.py -v` → 5 passed in 0.55s.
- Extra identity probe now returns `log close True` and `rot close True`.

Resolved: high severity Task 3 transformer gradient flow.

The earlier extra probe showed zero loss and zero gradients to `tile_features` / `field.color` because output heads were zero-initialized. Commit `0618e46` now initializes head weights with small nonzero normal noise and adds `test_grad_flow_to_inputs`. Verification:

- `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_transformer.py -v` → 4 passed in 0.56s.
- Extra gradient probe now reports nonzero loss and nonzero finite grads to `tile_features`, `field.color`, and `head_mu.weight`.

### Pixel Plan

Resolved: low severity stale arg docstring on `warp_prev_hr`.

Plan was patched to forward-flow t-1→t convention; the implementer subagent for Task 0 ran on a slightly earlier plan revision and committed `oss/sr/temporal/warp.py` with the old `current→previous` arg doc. Direct edit on the file at 15:50 CDT brought it in sync with the corrected plan; all 3 warp tests still pass.

Resolved: medium severity rasterizer-snippet missing torch import (Gaussian Task 6).

Plan now opens the snippet with `import torch`.

Resolved: medium severity Gaussian frame-0 zero render.

Plan now re-renders after the first-frame seed densification before returning `rendered_hr`. Acceptance criterion adds explicit `rendered_hr.abs().max() > 0` check.

Resolved: high severity flow-direction mismatch.

Earlier plan version described motion as `current -> previous` and sampled `base + motion`, while the spec and TartanAir/Sintel adapters use forward flow `t-1 -> t`. Latest plan now states forward flow and samples `base - motion`.

Relevant current lines:

- `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md:164`
- `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md:229`

Resolved: medium severity LPIPS test gap.

Earlier Task 5 acceptance claimed LPIPS but the test omitted it. Latest plan adds an optional LPIPS path when the `lpips` package is importable.

Relevant current lines:

- `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md:1142`
- `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md:1213`

### Gaussian Plan

Resolved: high severity first-frame seed impossibility.

Earlier Task 8 created an empty field for `prev_field=None`, skipped updates, rendered empty, and left densification outside the model while requiring `count_alive() > 0`. Latest plan seeds via densification.

Relevant current lines:

- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:1044`
- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:1051`

Resolved: high severity rasterizer API mismatch.

Earlier Task 6 used non-existent `GaussianBatch` fields (`positions`, `scales`, `rotations`, `colors`, `opacities`) and passed `output_hw` to `Rasterizer` constructor. Latest plan uses `GaussianBatch(xy, scale, rot, feat)` and calls `_RASTERIZER(batch, output_hw=...)`.

Relevant current lines:

- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:948`
- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:954`

Resolved: medium severity encoder constructor crash.

Earlier Task 2 used `float(tile_size).bit_length()`, which would raise. Latest plan uses `int(tile_size).bit_length()`.

Relevant current line:

- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:570`

Resolved: medium severity batched densification tile indexing.

Earlier densification flattened batch and tile dimensions but decoded tile coordinates as if `B=1`. Latest plan explicitly rejects `B != 1`, matching the per-sample `GaussianField` state.

Relevant current lines:

- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:789`
- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:815`

## Timeline

### 15:17-15:20 CDT

Established baseline:

- `<repo-root>` is the only local checkout found for `cashcon57/open-supersampling`.
- Branch is `v0.2-dev`.
- No tracked changes.
- Only pre-existing untracked local artifacts listed above.
- Read `README.md`, both v5 specs, A/B memo, and lab-notebook discipline.

### 15:20-15:25 CDT

Pixel plan appeared.

Initial review found:

- High: flow convention mismatch (`current -> previous` / `base + motion`) against spec and datasets.
- Medium: Task 5 claimed LPIPS but test omitted it.

### 15:25-15:30 CDT

Gaussian plan and task JSON files appeared.

Initial Gaussian review found:

- High: Task 8 could not satisfy `prev_field=None` seeding criterion.
- High: Task 6 rasterizer wrapper used the wrong repo API.
- Medium: Task 2 used `float.bit_length()`.
- Medium: densification indexing only worked for `B=1` but did not enforce that.

### 15:39-15:41 CDT

Pixel plan changed:

- Flow convention corrected to forward `t-1 -> t`.
- LPIPS optional test path added.
- Remaining pixel issue is a stale docstring line.

Gaussian plan changed:

- Encoder, densification `B=1`, and rasterizer API issues corrected.
- First-frame seed path added.
- Remaining Gaussian issues are missing `torch` import in the rasterizer snippet and possible empty frame-0 output because render happens before seed densification return.

### 15:43 CDT

Latest monitor check:

- Plan docs and task JSON files present.
- No implementation files under v5 module/test paths.
- Task JSON files stale relative to plan docs.

### 15:46-15:50 CDT (Claude controller)

- Pixel Task 0 implementer subagent landed `2d315e1` (3 warp tests pass).
- Gaussian Task 0 implementer subagent landed `0820439` (5 GaussianField tests pass).
- Both subagents ran on slightly pre-final plan revisions; pixel committed warp.py with stale `current→previous` arg docstring. Direct edit on file restored the corrected forward-flow doc; tests still green (8/8).
- Plan patches `f00f7a4` covered Codex's three follow-up findings (warp doc nit, Gaussian Task 6 torch import, Gaussian Task 8 first-frame re-render).
- Cash directive: parallel implementation across both tracks via separate subagent streams; sequential GPU train; quality + perf over speed; default plan content otherwise.
- Next: dispatch Task 1 of each track in parallel.

### 15:49-15:50 CDT

Pixel Task 1 landed:

- `a993cda` added `DisocclusionGate`, exported it, and added `tests/sr/temporal/test_disocclusion.py`.
- Verification: `venv-py312/bin/python -m pytest tests/sr/temporal/test_disocclusion.py -v` passed 5/5.

Gaussian Task 1 started:

- `tests/sr/gaussian_temporal/test_analytical_warp.py` and `oss/sr/gaussian_temporal/analytical_warp.py` appeared uncommitted.
- Verification currently fails 1/4: pure translation preserves means but SVD recomposition swaps log-scale axes for one Gaussian.
- Open finding added under "Gaussian Task 1 analytical warp".

### 15:51-15:52 CDT

Gaussian Task 1 landed:

- `8bb7693` committed `oss/sr/gaussian_temporal/analytical_warp.py` and `tests/sr/gaussian_temporal/test_analytical_warp.py`.
- Verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_analytical_warp.py -v` passed 4/4.
- Extra Codex probe against the plan acceptance found identity flow still changes `log_scale`/`rotation` for rotated unequal-scale fields. The committed test does not cover rotation preservation.
- Open finding updated under "Gaussian Task 1 analytical warp".

### 15:53-15:58 CDT

Pixel Task 2 and Gaussian Task 2 landed:

- `e912bf5` added `TemporalHead`; verification passed 4/4.
- `7225b20` added `GBufferEncoder`; verification passed 3/3.

Pixel Task 3 landed:

- `9d1a256` added `TemporalSRModel`, `make_first_frame_prev_hr`, v4 warm-start loading, and backbone freeze toggling.
- Verification: `venv-py312/bin/python -m pytest tests/sr/temporal/test_model.py -v` passed 5/5.

Gaussian Task 3 started:

- `tests/sr/gaussian_temporal/test_transformer.py` appeared uncommitted.
- Verification currently errors at collection because `oss.sr.gaussian_temporal.__init__` does not export `GaussianMultiFrameTransformer` and `oss/sr/gaussian_temporal/transformer.py` is not present yet. This is expected for test-first work but should remain red until implementation lands.

### 15:58-16:01 CDT

Gaussian Task 3 landed:

- `1437656` added `GaussianMultiFrameTransformer`, exported it, and committed `tests/sr/gaussian_temporal/test_transformer.py`.
- Verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_transformer.py -v` passed 3/3.
- Param count probe: default `(d_model=128, n_heads=4, n_layers=4, history_len=5)` has 549,640 params, inside the 400K-600K budget.
- Extra Codex probe against the plan acceptance found zero gradient flow into `tile_features` and `field.color` because all output heads are zero-initialized. Open finding added under "Gaussian Task 3 transformer gradient flow".

### 16:02-16:05 CDT

Task 4 landed on both tracks:

- `62234d2` added residual-driven Gaussian densification; verification passed 3/3.
- `53fda3c` added `SequentialPairDataset` plus TartanAir/Sintel trajectory-key shims; verification passed 4/4.
- Existing dataset classes do expose `_items` in the tuple shape assumed by `adapt_tartanair` / `adapt_sintel`.
- Extra Codex probe found `SequentialPairDataset(_FakeBase(), pair_stride=2)` raises `TypeError`, while the plan requires a `pair_stride=1` constructor parameter. Open finding added under "Pixel Task 4 pair_stride API gap".

### 16:06-16:11 CDT

Task 5 landed on both tracks:

- `b3fbc1c` added the pixel temporal loss-pipeline integration test. Verification passed 1/1 with two torchvision deprecation warnings.
- `1a09ccc` added Gaussian opacity/count pruning. Verification passed 3/3 after the threshold-boundary test was aligned to the plan's strict `< opacity_threshold` rule.

Gaussian acceptance fixes appeared in the working tree:

- Analytical warp now preserves `log_scale` and `rotation` under identity/pure-translation Jacobians for rotated unequal-scale fields; verification passed 5/5 and the extra identity probe now passes.
- Transformer heads now use tiny nonzero weight initialization and `test_grad_flow_to_inputs`; verification passed 4/4 and the extra gradient probe now passes.
- These Gaussian fixes were committed in `0618e46`.

### 16:12-16:16 CDT

Task 6 landed on both tracks:

- `e15324f` added `TemporalSRInferenceEngine` to `oss/sr/inference.py`, with first-frame bilinear init, state reset, and scene-cut reset. Verification passed 3/3.
- `6323a9b` added the Gaussian `render_field` wrapper around the existing renderer. Verification passed 4/4.
- Review did not find new blockers in the targeted Task 6 code paths. The pixel Task 4 `pair_stride` API gap remains open.

### 16:17-16:20 CDT

Gaussian Task 7 landed:

- `8db02a8` added `gaussian_regularization_loss` and regularization tests. Verification passed 11/11.

Pixel Task 7 started test-first:

- `tests/sr/temporal/test_train_smoke.py` appeared uncommitted.
- Initial verification failed 1/1 because `scripts/sr_train_temporal.py` was not present yet. This was expected for test-first work.

### 16:10-16:12 CDT

Pixel Task 7 landed:

- `1557522` added `scripts/sr_train_temporal.py` and committed the CPU smoke test.
- Verification: `venv-py312/bin/python -m pytest tests/sr/temporal/test_train_smoke.py -v` passed 1/1.
- Direct smoke command exited 0, printed finite `final_loss=2.357021`, and wrote `metrics.json`, `score_log.json`, and `step-00000005.pt`.
- Review found a medium dashboard-compatibility issue: `score_log.json` rows use `psnr`/`lpips`, while `scripts/training_dashboard.py` reads `model_psnr_mean`, `bicubic_psnr_mean`, and related eval keys. Open finding added under "Pixel Task 7 score_log dashboard schema mismatch".

### 16:12-16:13 CDT

Next test-first tasks appeared:

- Pixel Task 8: `tests/sr/temporal/test_held_out_argparse.py` appeared uncommitted. Verification fails 1/1 because `scripts/sr_temporal_held_out.py` is not present yet.
- Gaussian Task 8: `tests/sr/gaussian_temporal/test_model_full_step.py` appeared uncommitted. Combined verification errors during collection because `GaussianTemporalSRModel` is not exported yet; `oss/sr/gaussian_temporal/model.py` is not present yet.

### 16:13-16:15 CDT

Gaussian Task 8 landed:

- `9d66af1` added `GaussianTemporalSRModel`, exported it, and committed `tests/sr/gaussian_temporal/test_model_full_step.py`.
- Verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_model_full_step.py -v` passed 5/5.
- Review found a high spec gap: the model reads `prev_field.history` but never pushes snapshots into the returned `new_field`, so recurrent rollouts keep history length 0 and the transformer never actually receives multi-frame Gaussian context. Open finding added under "Gaussian Task 8 history buffer is never populated".

### 16:25-16:26 CDT

Pixel Task 8 implementation appeared uncommitted:

- `scripts/sr_temporal_held_out.py`, `tests/sr/temporal/test_held_out_argparse.py`, and `docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out-template.md` are present.
- Verification: `venv-py312/bin/python -m pytest tests/sr/temporal/test_held_out_argparse.py -v` passed 1/1.
- Review found a high motion-vector bug that affects both the committed training script and the uncommitted held-out evaluator: for a `(t, t+1)` pair, both use `tp1_motion` for `t -> t+1` alignment, but dataset motion is per-frame forward flow to the next frame, so `t_motion` is the correct vector. Open finding added under "Pixel temporal uses next-frame flow for t→t+1 alignment".

### 16:27 CDT

Pixel Task 8 landed:

- `0755e9b` committed the held-out eval script, argparse test, and memo template. Verification remains green: `tests/sr/temporal/test_held_out_argparse.py` passed 1/1.
- The high motion-vector finding still applies to the committed held-out eval and to the earlier committed training script.

Two Codex findings were fixed in the working tree but are not committed yet:

- Gaussian Task 8 history now populates `new_field.history`; `test_history_populates_across_frames` added.
- Pixel Task 4 `pair_stride` API now exists with stride-2 and invalid-stride tests.
- Combined verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_model_full_step.py tests/sr/temporal/test_dataset.py tests/sr/temporal/test_held_out_argparse.py -v` passed 13/13.

## Suggested Monitor Command

Use this focused watcher instead of hashing old asset directories:

```bash
while true; do
  date '+--- %Y-%m-%d %H:%M:%S %Z ---'
  git status --porcelain=v1 -- docs/superpowers/plans docs/superpowers/experiments docs/superpowers/notes oss/sr scripts tests/sr README.md
  for f in docs/superpowers/plans/2026-05-04-v5-*; do
    [ -f "$f" ] && shasum -a 256 "$f"
  done
  find oss/sr/temporal oss/sr/gaussian_temporal tests/sr/temporal tests/sr/gaussian_temporal -maxdepth 5 -type f -print 2>/dev/null \
    | sort \
    | while read f; do shasum -a 256 "$f"; done
  sleep 30
done
```

## Update Rules

Agents updating this document should:

- Keep `Current State` and `Open Findings` current.
- Move fixed items from `Open Findings` to `Resolved Findings`.
- Append timeline entries with concrete timestamps.
- Cite file paths and line numbers when possible.
- Do not delete old negative findings; mark them resolved with the reason.
