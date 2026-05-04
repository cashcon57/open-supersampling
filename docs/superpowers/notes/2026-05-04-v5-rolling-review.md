# 2026-05-04 — v5 Rolling Review

**Status:** Active living document  
**Purpose:** Shared rolling review surface for Sprint 5 dual-track implementation planning and code review. Claude/Codex agents should read and update this file before dispatching implementation or reviewer subagents.  
**Last updated:** 2026-05-04 17:28 CDT

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

Claude/Codex launch-status note:

- `docs/superpowers/notes/2026-05-04-v5-pixel-launch-status.md` is tracked. It records the failed early launch attempts and the active TartanAir-only relaunch on `<train-host>`: python PID `2360`, parent `cmd.exe` PID `15652`, dashboard PID `14952`. Latest Codex check at 17:27 CDT: PID `2360` alive and log reached step `880` with finite Phase-1 losses.

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
25296e31403215daf133d03c1a8b4585e478b7cba3f93c9cdd0ce9fa4344a8cc  oss/sr/temporal/dataset.py
1d69b6a82326dda8879194bd5076ea084b8ad18947a4f6fecc40c2ef87f2f9b6  oss/sr/temporal/__init__.py
8822082b83a5a6631f179e22acc83faf8706047569efa3c3eb441f1a40ed373a  oss/sr/inference.py
848bb3c2accb797fe46939a23d676fa6a8f57493c60c84ae52631426cbd48fa9  scripts/sr_train_temporal.py
9c7d33c094387b7e38c0e9ad5826f46f063cf7d2f9f8ced9fcaa19bc1afcf812  scripts/sr_temporal_held_out.py
2afe6c5aff9ada4b8268ce198282cb3614ac09f9123ec90f0711a5c1fc96143d  scripts/sr_train_gaussian_temporal.py
fb36d8d2ee1b76937bc2482eeb16ad9b1752c2478b845f44c1128a12e8787c85  scripts/sr_gaussian_temporal_held_out.py
060a9361ca2be77680baac4e699e0c29514c7eba6d1846672489341f3cb46a9d  oss/sr/gaussian_temporal/analytical_warp.py
997231cb12f6abaf3f10cf161a182fa551bbca41cd4879458049dc8251585a01  oss/sr/gaussian_temporal/densification.py
b6887cd0622263778a035ea3f83335fc6babe244fd08ab77b1155a8e01310955  oss/sr/gaussian_temporal/g_buffer_encoder.py
99a38a13105e09a78c2cb4af1d066705f044828382e4226c77e09b0ad7307fe1  oss/sr/gaussian_temporal/gaussian_field.py
d4c35220845fc76d024afd7acc8cc132a5b6959de384897625d93c9dfe816057  oss/sr/gaussian_temporal/pruning.py
d618dddeda39c2d2f625cc02ddebc1fc35e4b3148759e09880923d33cd574a51  oss/sr/gaussian_temporal/rasterizer.py
42bb0fa6f1119a3155018008340a1d50eeeb3561b1d566ef775d91a50ef4ce9b  oss/sr/gaussian_temporal/regularization.py
1968722da0aedbcf88aeabe053245d36d9f26548ccea3462c3e3ca0d0cc8aab4  oss/sr/gaussian_temporal/transformer.py
39421b60b799a52cee9337fa9ebcaf6eb69bfe7ca85f538bbcebf6445e8bfa7a  oss/sr/gaussian_temporal/model.py
2c99d4b7fd8da2cc65b85625f98f94a469c8937c6654f8001e33d14729593e1b  oss/sr/gaussian_temporal/dataset.py
cdbc97c1f4c51c1bd380729466784a9354a6b3ff50db2740473576ffb2719ddf  oss/sr/gaussian_temporal/__init__.py
17f16008c472ecd73dd7bdf8eb45222aeeee87ae00b42273dbabca9d166b845a  tests/sr/temporal/test_inference_state.py
e43ce17eac512ad20150ba8d35a9c18b13c36ded8d45f485c1ed0b2a734ebf84  tests/sr/temporal/test_train_smoke.py
187f5d09bae23a0ca8625a7240b10c31c1b6125151f548f4f08f7aa639b3a298  tests/sr/temporal/test_held_out_argparse.py
a6522b387e5cc31bc2c29f639e0efb9f1d9f6e80bc3696487017a7796163030e  docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out-template.md
b9a46341fda4b46d15d62fe406c30567012f359998ccbbe264df62325426891d  tests/sr/temporal/test_loss_pipeline.py
62416b833fcea70694105bb9a6db88d48db8f44357c6a15e742099bab7dfec12  tests/sr/temporal/test_dataset.py
c51c9195ac4494bbc6c5e86836ed924bed160f88c9e51b2c1bfb00d8ad84f54c  tests/sr/gaussian_temporal/test_pruning.py
150708aaae5f03ba996b7c34f86f1e6f16e5ad599ae8afbf98fd73bae9ee9ebf  tests/sr/gaussian_temporal/test_rasterizer_wrapper.py
9e0ec8fe2910cd7626d7b8f0f46b4fc1d18f0e077f25213fc163876dcf452694  tests/sr/gaussian_temporal/test_regularization.py
5267d25231a5787feb853e096036005bd121a018c7ca4a44db3377e7797dba25  tests/sr/gaussian_temporal/test_model_full_step.py
8c06871e7a010744319fd982575bec00e407d859a4fbb10ceade096fddf5e57e  tests/sr/gaussian_temporal/test_dataset.py
ae84b9e3cb045ca4a9f9ca6020c5ee7803c43fe6469ac6e1ab3dddd94a3f71af  tests/sr/gaussian_temporal/test_inference_state.py
d08dbc2191ef4eb5f0943b0117e61242aafbb03c34dff3d9866f36ebbf0a2c5b  tests/sr/gaussian_temporal/test_train_smoke.py
b61a77b9cca542c738f87b1bd6f7f28a2647bfd7f27ca51c8fd1f4f84842fba9  tests/sr/gaussian_temporal/test_held_out_argparse.py
e619e9f695e3e21005ea392c5d90f93569fdb7b907d15868ef6e72e95d169efb  tests/sr/gaussian_temporal/test_analytical_warp.py
3614b8fac4f1a6e597f78197b2fcd06b050eeae01f3124fe9135703f9c782017  tests/sr/gaussian_temporal/test_transformer.py
e07419e61a91d399a47de2bc504c0ebac3de8562a1424f0de1e22f3c1b3d2b2b  docs/superpowers/experiments/2026-05-04-v5-pixel-temporal-train-start.md
6d12db027efda9def95d45fb5b3445fcfd2505c31ff6b8d82d35f5c2e4e74e9b  docs/superpowers/notes/2026-05-04-v5-pixel-temporal-runbook.md
2af29b55221b84623e810bc2b37ab0a3824d2123ab7dff9c77b56f7ad877ba8a  docs/superpowers/experiments/2026-05-04-v5-gaussian-temporal-train-start.md
9e4c98255cac34d2f3f858ab04287e2ffea2322414dbdcbd5d4aee2437cd90df  docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md
b92dcc1f373c9e492b5d4e09b567b0982675da1ba7910bef927c0e46346dd3ab  docs/superpowers/experiments/2026-XX-XX-v5-gaussian-temporal-held-out-template.md
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
- Pixel train-start lab notebook memo and remote runbook committed: `docs/superpowers/experiments/2026-05-04-v5-pixel-temporal-train-start.md`, `docs/superpowers/notes/2026-05-04-v5-pixel-temporal-runbook.md`
- Gaussian Task 9 committed: `oss/sr/gaussian_temporal/dataset.py`, `tests/sr/gaussian_temporal/test_dataset.py`
- Pixel flow-direction fix committed: `scripts/sr_train_temporal.py` and `scripts/sr_temporal_held_out.py` now use `t_motion` for `t -> t+1` alignment.
- Gaussian Task 10 committed: `oss/sr/inference.py`, `tests/sr/gaussian_temporal/test_inference_state.py`
- Gaussian Task 11 committed: `scripts/sr_train_gaussian_temporal.py`, `tests/sr/gaussian_temporal/test_train_smoke.py`
- Gaussian train-start memo/runbook committed: `docs/superpowers/experiments/2026-05-04-v5-gaussian-temporal-train-start.md`, `docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md`
- Gaussian Task 12 committed: `scripts/sr_gaussian_temporal_held_out.py`, `tests/sr/gaussian_temporal/test_held_out_argparse.py`, `docs/superpowers/experiments/2026-XX-XX-v5-gaussian-temporal-held-out-template.md`
- Sprint 5 review fixes through `c1bad69` are committed: score logs stay empty during training; runbook dashboard commands use `scripts/training_dashboard.py`; Gaussian phase isolation uses a trainable per-frame fitter in Phase 1 and `effective_layers=2` in Phase 2.
- `bd1f77a` committed trainer docstring cleanup so the pixel and Gaussian training headers now match the held-out-owned `score_log.json` behavior.

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
- Combined post-commit verification: `tests/sr/gaussian_temporal/test_model_full_step.py tests/sr/temporal/test_dataset.py tests/sr/temporal/test_train_smoke.py tests/sr/temporal/test_held_out_argparse.py` → 14 passed in 2.51s
- Changed-test verification after `38cf507`: `tests/sr/gaussian_temporal/test_dataset.py tests/sr/temporal/test_train_smoke.py tests/sr/temporal/test_held_out_argparse.py` → 7 passed in 2.41s
- Codex C1 history-order probe: after frame 7, stamped history color means were `[0.6, 0.5, 0.4, 0.3, 0.2]`, confirming `history[0]` is frame 6 and newest-first ordering holds through the cap.
- `tests/sr/gaussian_temporal/test_inference_state.py` → 3 passed in 0.52s
- `tests/sr/gaussian_temporal/test_train_smoke.py tests/sr/gaussian_temporal/test_held_out_argparse.py` → 2 passed in 2.19s
- Codex C2 pixel held-out flow-direction probe passed: synthetic `t_motion=+1`, `tp1_motion=-2` produced temporal model motion calls `[1.0, 1.0]` and `tstab_temporal=[0.0]`, confirming held-out render and stability warp are aligned to `t_motion`.
- Working-tree fix verification: `tests/sr/temporal/test_train_smoke.py tests/sr/gaussian_temporal/test_train_smoke.py tests/sr/gaussian_temporal/test_model_full_step.py tests/sr/gaussian_temporal/test_transformer.py` → 16 passed in 3.59s.
- Full local SR verification after `c1bad69`/`bd1f77a`: `venv-py312/bin/python -m pytest tests/sr/temporal tests/sr/gaussian_temporal -v` → 87 passed, 2 torchvision deprecation warnings in 5.83s.
- Direct score-log behavior check after the fake-eval-row fix: pixel smoke with `--max-steps 2` wrote `/tmp/oss_pixel_scorelog_check/score_log.json` as `[]`; Gaussian smoke with `--max-steps 2` wrote `/tmp/oss_gauss_scorelog_check/score_log.json` as `[]`.
- C3-triggered Pixel Task 6 fix: `tests/sr/temporal/test_inference_state.py` now covers `TemporalSRInferenceEngine.from_checkpoint` honoring `args.backbone_kind`; implementation fixed in `oss/sr/inference.py`.
- C3-triggered Pixel Task 7 coverage: `tests/sr/temporal/test_train_smoke.py` now asserts `score_log.json == []`, phase/LR schedule behavior, and smoke auto-resume from `step-*.pt`.
- Full local SR verification after C3 fixes: `venv-py312/bin/python -m pytest tests/sr -v` → 110 passed, 1 skipped, 14 warnings in 10.83s.
- Remote launch-blocker regression: `default_collate_pair` now accepts real `GaussianTrainingExample` objects and fills missing normals with zeros; `tests/sr/temporal/test_dataset.py` covers this. Commit: `4238915`.
- Full local SR verification after the collator fix: `venv-py312/bin/python -m pytest tests/sr -v` → 111 passed, 1 skipped, 14 warnings in 10.86s.
- `git diff --check` passed at 16:32 CDT.
- Extra spec probe for Gaussian analytical warp identity preservation now passes and is committed in `0618e46`.
- Extra spec probe for Gaussian transformer gradient flow now passes and is committed in `0618e46`.
- Extra spec probe for pixel `SequentialPairDataset(base, pair_stride=2)` is now covered by tests and passes in the working tree.

Verification caveat: default `python3` and `.venv/bin/python` do not have `torch` or `pytest`; `venv/bin/python` has `pytest` but not `torch`. Use `venv-py312/bin/python` for local CPU tests. For CUDA/PyTorch-heavy verification, Cash notes that PyTorch is also available on at least one Tailnet machine plus the RunPod and Lambda instances used by the project.

Observed commits on `v0.2-dev`:
- `d6bc655` v5-gaussian(sr): warp field in HR coordinates
- `2e4f43a` sprint5(notes): update pixel launch recovery
- `10e75df` v5-pixel(sr): skip unreadable frame pairs lazily
- `b8b08c5` data(tartanair): skip corrupt npy triples
- `7691e5f` sprint5(notes): pixel training confirmed running PID 27732 (ETA ~07:54 tomorrow)
- `913cc9f` sprint5(notes): launch-status note for pixel run (PID 8348 attempted, see body)
- `4238915` v5-pixel(sr): collate GaussianTrainingExample pairs
- `96dad76` v5-pixel(sr): make trajectory_key shims worker-pickleable on Windows
- `b185df6` v5-pixel(sr): cover train schedule and resume
- `4ed319a` v5-pixel(sr): honor backbone_kind in inference ckpt
- `bd1f77a` sprint5(sr): align train score-log docs
- `c1bad69` sprint5(sr): patch 3 Codex findings — phase isolation, fake eval rows, runbook cmds
- `15513b4` v5-gaussian(sr): add held-out eval + memo template
- `5e14312` v5-gaussian(sr): add training entry with 4-phase schedule + smoke test
- `a1144a0` v5-gaussian(sr): lab-notebook train-start memo + remote runbook
- `291adb8` v5-gaussian(sr): add stateful GaussianTemporalSRInferenceEngine
- `415d664` sprint5(notes): add 'Tasks for Codex' section with 4 verification probes
- `38cf507` v5-pixel(sr): fix HIGH Codex finding — wrong frame's motion vec for t->t+1 warp
- `ab6c5f9` v5-gaussian(sr): add multi-frame trajectory window dataset
- `7cb3c44` v5-pixel(sr): lab-notebook train-start memo + remote runbook
- `8a757f0` sprint5(sr): patch 3 Codex findings — history, pair_stride, score schema
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

Tracked pixel Task 8 and Gaussian Tasks 8-12 are committed. The prior high pixel flow-direction finding is fixed in `38cf507`. The three later implementation findings are fixed in `c1bad69`, with score-log documentation aligned in `bd1f77a`. C3 found and fixed one pixel inference checkpoint-loader bug in `4ed319a`, then expanded Task 7 train schedule/resume coverage in `b185df6`. C4 found and fixed one Gaussian model coordinate-space bug in `d6bc655`. Remaining open item is the stale plan task sidecars.

## Tasks for Codex

Cash authorized Claude (controller) to assign verification work here. Codex picks these up on its monitoring cadence; mark each item as "claimed by Codex" or "done" with a brief note when handled.

Active asks:

- **C1 — done by Codex at 16:32 CDT.** History-buffer ordering after frame N≥6 is newest-first. Probe stamped returned fields with frame ids, rolled through frame 7, and observed history color means `[0.6, 0.5, 0.4, 0.3, 0.2]`, matching frames 6, 5, 4, 3, 2. Ref: `oss/sr/gaussian_temporal/model.py:131-145`.
- **C2 — done by Codex at 16:40 CDT.** Synthetic held-out probe used `t_motion=+1` and `tp1_motion=-2`; fake temporal model saw motion calls `[1.0, 1.0]`, and `tstab_temporal` was exactly `0.0`, which would not hold if `tp1_motion` were used for the second render or stability warp.
- **C3 — done by Codex at 17:02 CDT.** Pixel Tasks 0-9 reviewed against `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md`. One real bug fixed (`TemporalSRInferenceEngine.from_checkpoint` now honors saved `backbone_kind`), and Pixel Task 7 gained schedule, score-log, and auto-resume tests. Remaining gaps are documented below under "C3 Pixel Spec Compliance Review"; none currently block the launched pixel run.
- **C4 — done by Codex at 17:28 CDT.** Gaussian Tasks 0-9 reviewed against `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md`. One real bug fixed: `GaussianTemporalSRModel` now lifts LR motion into HR field coordinates before warping persistent Gaussians (`d6bc655`). Full Gaussian-temporal suite passed 59/59 after the fix.

If a probe finds a real bug, file it under `## Open Findings` with severity + file:line citations as you've been doing. Claude will patch.

## Review Gate

Do not dispatch implementation workers from stale `.tasks.json` files. Both JSON files were generated before later plan edits and did not update when the source plan docs changed.

Preferred next step:

1. Clean the remaining plan nits below.
2. Regenerate both `.tasks.json` files from the corrected plan docs, or explicitly instruct implementers to ignore the JSON and read the plan docs directly.
3. Begin pixel implementation first unless Cash explicitly chooses parallel track work.

## C3 Pixel Spec Compliance Review

Scope: Pixel Tasks 0-9 in `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md`, checked against committed tests and implementation.

Result: no launch-blocking pixel finding remains after `4ed319a` and `b185df6`.

Fixed during C3:

- Task 6 checkpoint-loader bug: training checkpoints save `args.backbone_kind`, but `TemporalSRInferenceEngine.from_checkpoint` only inspected legacy `args.sr_backbone`, so non-simple temporal checkpoints would instantiate the wrong backbone before `load_state_dict`. Fixed in `4ed319a`; regression test added.
- Task 7 test gap: the original smoke test only checked exit/files. `b185df6` now asserts `score_log.json` stays empty during training, covers phase boundaries and LR multipliers, verifies backbone freeze/unfreeze at phase transitions, and exercises auto-resume from the latest checkpoint.
- Task 4 real-data launch blocker: `default_collate_pair` assumed mapping samples, while real TartanAir/Sintel loaders return `GaussianTrainingExample` objects. Fixed in `4238915`; regression test added for dataclass examples and `normals=None`.

Remaining documented coverage gaps:

- Task 0: `test_translation_warp` covers warp direction with a stripe, but does not assert the exact overlapping-region equality phrased in the plan. Current test is adequate for direction/regression but less strict than the acceptance text.
- Task 2: tests cover parameter budget, output shape, near-identity initial output, and gradients, but do not directly assert `conv_out.bias == 0` or weight std. This is behaviorally covered by the near-identity test, not mechanically covered.
- Task 4: pair counting/boundary behavior is covered with a fake 5+3-frame base dataset and `pair_stride=2`, but not with an actual synthetic 4-frame `TartanAirGaussianDataset`; `adapt_tartanair` / `adapt_sintel` trajectory-key shims are not directly unit-tested against real `_items` tuples.
- Task 5: the integration test includes L1, SSIM-like proxy, optional LPIPS, temporal consistency, and gradients. The LPIPS path is best-effort (`try/except`) instead of the plan's exact `pytest.importorskip("lpips")` wording, and the SSIM term is a lightweight proxy in the test.
- Task 8: argparse/import smoke is covered, and C2 independently verified held-out flow direction. Full PSNR/LPIPS/temporal-stability result correctness still depends on a real checkpoint + datasets after training.
- Task 9: memo/runbook exist and were used for launch; warm-start hash and remote launch state are operationally verified, not unit-tested.

## C4 Gaussian Spec Compliance Review

Scope: Gaussian Tasks 0-9 in `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md`, checked against committed tests and implementation.

Result: no Gaussian Task 0-9 blocking finding remains after `d6bc655`.

Fixed during C4:

- Task 8 coordinate-space bug: first-frame densification seeds Gaussian means in HR pixel coordinates, but temporal warp was sampling the LR motion field and using LR frame bounds. That killed valid HR-space Gaussians outside the LR bounds on the next frame. Fixed in `d6bc655`; `oss/sr/gaussian_temporal/model.py:119` upsamples LR motion to HR resolution, scales displacement by `model.scale`, and calls `warp_field(..., hw=(h_hr, w_hr))`. Regression: `tests/sr/gaussian_temporal/test_model_full_step.py:98`.

Remaining documented coverage gaps:

- Task 1: analytical covariance warp is covered for identity, pure translation, and a diagonal smooth-flow Jacobian. There is no direct finite-difference probe for off-diagonal shear/rotation coupling.
- Task 3: transformer tests cover param budget, shape, permutation equivariance, RoPE-keyed gradient path, and input gradients. There is no direct inspection test proving absence of learned positional embeddings beyond behavior.
- Task 4: densification is explicitly heuristic and per-sample (`B=1`). The test covers residual selection, free-slot insertion, tile color, and color gradients, but not a soft top-K variant, which is post-v5 by spec.
- Task 8: full-step synthetic training and phase isolation are covered locally. Real-data Gaussian training remains intentionally queued until the pixel control run completes or Cash approves GPU overlap.
- Task 9: window-boundary behavior is covered with a fake trajectory-keyed base dataset and collate coverage. Real TartanAir/Sintel adapter behavior shares the pixel-track adapter shims but is not directly unit-tested against real remote dataset trees.

## Open Findings

### Remote Sintel Dataset Missing Depth

Severity: medium for active training, high before final v5 success-criteria eval.

The remote `<train-host-data>/datasets/sintel` tree has `training/{clean,final,flow,...}` but no `training/depth`, so `SintelGaussianDataset(root=<train-host-data>/datasets/sintel, pass_name="clean")` discovers no `(frame, depth, flow)` triples and raises `FileNotFoundError`.

Impact:

- Active pixel training was relaunched TartanAir-only at 17:08 CDT to keep the control track moving.
- Phase 3 is no longer true Sintel fine-tune until Sintel Depth is fetched/restored or the loader gains a depth fallback.
- Held-out eval against Sintel cannot run with the current loader/data layout.

Fix direction:

- Fetch/extract the Sintel Depth package into `<train-host-data>/datasets/sintel/training/depth/<seq>/frame_NNNN.dpt`, or explicitly add and test a no-depth fallback before using Sintel for v5 gates.

### Stale .tasks.json sidecars

Severity: low (does not block dispatch — implementer subagents are instructed to read the plan `.md` directly).

Fix: regenerate the task JSON files before any cross-session resume via `/superpowers-extended-cc:executing-plans`.

## Resolved Findings

### Sprint 5 Fixes Committed In `c1bad69` / `bd1f77a`

Resolved in `c1bad69` with docs aligned in `bd1f77a`: medium severity training `score_log` eval-row semantics.

The pixel and Gaussian training scripts no longer append train-loss-derived pseudo-eval rows to `score_log.json`; training progress stays in `metrics.json`, and held-out scripts remain responsible for real eval rows.

- `scripts/sr_train_temporal.py:683-698` now saves checkpoints and dumps metrics without appending to `score_log`.
- `scripts/sr_train_gaussian_temporal.py:725-740` follows the same rule.
- Verification: both temporal and Gaussian train smoke tests pass in the 16-test working-tree verification; direct two-step smoke runs wrote `score_log.json` as `[]` for both trainers.

Resolved in `c1bad69`: high severity Gaussian Task 11 phase schedule mismatch.

The model now exposes explicit phase control, Phase 1 bypasses the transformer while retaining a trainable per-frame fitter path, and Phase 2 passes `effective_layers=2` into the transformer.

- `oss/sr/gaussian_temporal/model.py:44-48` adds a small trainable per-frame RGB fitter head off encoder features.
- `oss/sr/gaussian_temporal/model.py:58-124` accepts `phase`, bypasses transformer in Phase 1, and passes `effective_layers=2` in Phase 2.
- `oss/sr/gaussian_temporal/transformer.py:285-363` accepts and enforces `effective_layers`.
- `scripts/sr_train_gaussian_temporal.py:408-416` passes the current phase into the model.
- `tests/sr/gaussian_temporal/test_model_full_step.py` adds coverage for Phase 1 transformer bypass, Phase 1 encoder/fitter gradients, Phase 2 two-layer warmup, and Phase 3 full-layer behavior.
- Verification: `tests/sr/gaussian_temporal/test_model_full_step.py tests/sr/gaussian_temporal/test_transformer.py tests/sr/gaussian_temporal/test_train_smoke.py` passed in the 16-test working-tree verification.

Resolved in `c1bad69`: medium severity stale runbook dashboard commands.

Both remote runbooks now call the actual dashboard script with its real CLI:

- `docs/superpowers/notes/2026-05-04-v5-pixel-temporal-runbook.md:117` uses `scripts\training_dashboard.py --output-dir ... --log-file ... --port 8080 --host 0.0.0.0`.
- `docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md:153` uses the same fixed pattern.

### Committed Sprint 5 Fixes

Resolved in `38cf507`: high severity pixel temporal flow-vector off-by-one.

The pair dataset returns `t_motion` from frame `t` and `tp1_motion` from frame `t+1`; TartanAir/Sintel motion is forward flow from the current frame to the next. The training script and held-out eval previously used `tp1_motion` for the `t -> t+1` render and temporal-stability warp. Current code uses `t_motion` in all critical sites:

- `scripts/sr_train_temporal.py:381-385` passes `motion_lr=t_motion` for the `t+1` recurrent render.
- `scripts/sr_train_temporal.py:405-408` passes `t_motion` into `temporal_consistency_loss`.
- `scripts/sr_temporal_held_out.py:255-259` passes `motion_lr=t_motion` for temporal `t+1`.
- `scripts/sr_temporal_held_out.py:269-271` uses `t_motion` for temporal-stability warps.
- Verification: changed-test slice passed 7/7; Claude-reported full v5 suite passed 78/78 with two warnings.

Resolved in `8a757f0`: high severity Gaussian Task 8 history buffer.

The earlier probe showed recurrent rollouts kept `len(history) == 0`, so the Gaussian transformer never received multi-frame Gaussian context. Current working tree now pushes prior-field snapshots before returning `new_field`:

- `oss/sr/gaussian_temporal/model.py:131` populates history.
- `tests/sr/gaussian_temporal/test_model_full_step.py` adds `test_history_populates_across_frames`.
- Verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_model_full_step.py -v` is included in the 13-test and 14-test combined passes.
- Codex C1 probe confirmed newest-first history ordering after frame 7.

Resolved in `8a757f0`: medium severity Pixel Task 4 `pair_stride` API gap.

The earlier probe showed `SequentialPairDataset(_FakeBase(), pair_stride=2)` raised `TypeError`. Current working tree now accepts `pair_stride`, validates it, and excludes pairs crossing trajectory boundaries:

- `oss/sr/temporal/dataset.py:34` adds `pair_stride: int = 1`.
- `oss/sr/temporal/dataset.py:45` builds pairs using `i + pair_stride`.
- `tests/sr/temporal/test_dataset.py` adds stride-2 and invalid-stride coverage.
- Verification: `venv-py312/bin/python -m pytest tests/sr/temporal/test_dataset.py -v` is included in the 13-test and 14-test combined passes.

### Gaussian Implementation

Resolved in `d6bc655`: high severity Task 8 HR/LR coordinate mismatch in temporal Gaussian warp.

The Gaussian field is rendered in HR pixel coordinates, but the model previously passed the LR motion tensor and LR `(h, w)` bounds into `warp_field` for recurrent frames. A zero-motion second frame could mark valid HR-space Gaussians dead simply because `mu.x >= w_lr` or `mu.y >= h_lr`.

- `oss/sr/gaussian_temporal/model.py:119` now upsamples `motion_lr` to `(h_hr, w_hr)`, scales displacement by `model.scale`, and warps against HR bounds.
- `tests/sr/gaussian_temporal/test_model_full_step.py:98` adds `test_temporal_warp_uses_hr_field_coordinates`.
- Verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_model_full_step.py -v` → 11 passed in 0.70s.
- Verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal -v` → 59 passed in 2.77s.

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

### 16:29-16:32 CDT

Follow-up commits landed:

- `8a757f0` committed three Codex findings: Gaussian history population, pixel `pair_stride`, and dashboard-shaped score rows.
- `7cb3c44` added the pixel temporal train-start memo and remote launch runbook before GPU time, satisfying lab-notebook discipline.
- `ab6c5f9` added Gaussian Task 9 `TrajectoryWindowDataset` and tests.
- `38cf507` fixed the high pixel flow-direction bug by using `t_motion` for `t -> t+1` render/consistency/stability.

Verification:

- Combined post-fix suite: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_model_full_step.py tests/sr/temporal/test_dataset.py tests/sr/temporal/test_train_smoke.py tests/sr/temporal/test_held_out_argparse.py -v` passed 14/14.
- Changed-test slice after the latest commits passed 7/7.
- `git diff --check` passed.
- Codex C1 probe confirmed Gaussian history newest-first ordering after frame 7.

Remaining review findings:

- Pixel training `score_log` rows now have dashboard keys but still represent train approximations as eval rows; latest eval margin can be misleading when bicubic fields are `None`.
- Pixel runbook dashboard restart command is stale (`scripts\sr_dashboard.py --run-dir` instead of `scripts\training_dashboard.py --output-dir ... --log-file ...`).

### 16:33-16:35 CDT

More Gaussian track progress:

- `415d664` added the `Tasks for Codex` section to this rolling report, assigning C1-C4 verification probes.
- `291adb8` added `GaussianTemporalSRInferenceEngine` and its state/reset/scene-cut tests.
- Verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_inference_state.py -v` passed 3/3.
- Gaussian Task 11 started test-first: `tests/sr/gaussian_temporal/test_train_smoke.py` appeared uncommitted and currently fails because `scripts/sr_train_gaussian_temporal.py` does not exist yet. This is expected-red until the implementation file lands.

### 16:36 CDT

Additional Gaussian test/docs appeared:

- Uncommitted train-start memo/runbook: `docs/superpowers/experiments/2026-05-04-v5-gaussian-temporal-train-start.md`, `docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md`.
- Uncommitted Task 12 argparse smoke test: `tests/sr/gaussian_temporal/test_held_out_argparse.py`.
- Verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_held_out_argparse.py -v` fails 1/1 because `scripts/sr_gaussian_temporal_held_out.py` does not exist yet. This is expected-red until Task 12 implementation lands.
- The Gaussian runbook repeats the same stale dashboard command pattern as the pixel runbook; open finding updated to cover both.

### 16:37-16:38 CDT

Gaussian Tasks 11-12 landed:

- `a1144a0` committed the Gaussian train-start memo + remote runbook.
- `5e14312` committed `scripts/sr_train_gaussian_temporal.py` and the Gaussian train smoke test.
- `15513b4` committed `scripts/sr_gaussian_temporal_held_out.py`, its argparse smoke test, and the held-out memo template.
- Verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_train_smoke.py tests/sr/gaussian_temporal/test_held_out_argparse.py -v` passed 2/2.

Review added one high finding:

- Gaussian Task 11's phase schedule is not actually phase-isolated: Phase 1 still calls the model path that invokes the transformer after densification, and Phase 2 does not use a real 2-layer transformer warmup.

### 16:40 CDT

Codex completed C2:

- Synthetic pixel held-out probe used a pair with `t_motion=+1` and `tp1_motion=-2`.
- Fake temporal model recorded both calls as `+1`, and the temporal-stability metric was exactly `0.0`.
- This independently verifies `scripts/sr_temporal_held_out.py` is using `t_motion` for both the `t+1` render and the stability warp after `38cf507`.

### 16:45-16:53 CDT

Open findings patched in the working tree:

- Claude started a phase-isolation patch in `oss/sr/gaussian_temporal/model.py` and `transformer.py`; initial targeted verification failed because bypassing the transformer in Phase 1 removed the trainable gradient path.
- Codex added a small per-frame RGB fitter head off encoder features so Phase 1 remains trainable without invoking temporal attention.
- Phase controls now flow from `scripts/sr_train_gaussian_temporal.py` into `GaussianTemporalSRModel`; Phase 2 uses `effective_layers=2`; Phase 3+ uses all layers.
- Pixel and Gaussian trainers now leave `score_log.json` empty during training; held-out eval scripts remain responsible for real score rows.
- Pixel and Gaussian runbooks now invoke `scripts\training_dashboard.py --output-dir ... --log-file ...` instead of stale `scripts\sr_dashboard.py --run-dir`.
- Verification: `venv-py312/bin/python -m pytest tests/sr/temporal/test_train_smoke.py tests/sr/gaussian_temporal/test_train_smoke.py tests/sr/gaussian_temporal/test_model_full_step.py tests/sr/gaussian_temporal/test_transformer.py -v` passed 16/16.

### 16:56-16:58 CDT

Claude/Codex follow-up state:

- `c1bad69` committed the three reviewed fixes: phase isolation, fake eval-row removal, and dashboard runbook command correction.
- Codex committed `bd1f77a`, aligning the pixel and Gaussian trainer docstrings with the new behavior that training keeps `score_log.json` empty and held-out eval owns score rows.
- Full local SR verification passed: `venv-py312/bin/python -m pytest tests/sr/temporal tests/sr/gaussian_temporal -v` → 87 passed, 2 existing torchvision deprecation warnings.
- Direct behavior check passed: both two-step CPU smoke runs produced `score_log.json` as `[]` under `/tmp/oss_pixel_scorelog_check` and `/tmp/oss_gauss_scorelog_check`.
- `git diff --check` passed.

### 17:00-17:02 CDT

C3 pixel spec-compliance pass:

- Codex found and fixed a checkpoint-loader bug: `TemporalSRInferenceEngine.from_checkpoint` now honors trainer-saved `args.backbone_kind` before falling back to legacy `args.sr_backbone`. Commit: `4ed319a`.
- Pixel Task 7 tests now cover score-log emptiness, phase/LR schedule, backbone freeze/unfreeze, and auto-resume. Commit: `b185df6`.
- Verification: `venv-py312/bin/python -m pytest tests/sr/temporal/test_inference_state.py -v` passed 4/4.
- Verification: `venv-py312/bin/python -m pytest tests/sr/temporal/test_train_smoke.py -v` passed 3/3.
- Full local SR suite passed after both C3 fixes: `venv-py312/bin/python -m pytest tests/sr -v` → 110 passed, 1 skipped, 14 warnings.
- Claude-created `docs/superpowers/notes/2026-05-04-v5-pixel-launch-status.md` reports pixel temporal training launched on the remote 3080 Ti at 16:57 CDT; Codex has not committed that untracked note.

### 17:03-17:23 CDT

Remote launch recovery:

- Remote pixel process PID 8348 was dead. Log showed `TypeError: 'GaussianTrainingExample' object is not subscriptable` in `default_collate_pair`.
- `4238915` fixed the collator to support both mapping samples and `GaussianTrainingExample` dataclass objects, including `normals=None`; `tests/sr/temporal/test_dataset.py` now covers the real sample shape.
- Full local SR suite after the fix: `venv-py312/bin/python -m pytest tests/sr -v` → 111 passed, 1 skipped, 14 warnings.
- Pushed `v0.2-dev` and fast-forwarded remote `<train-host-data>/oss-gaussian` to `4238915`.
- One-step real-data CUDA preflight with both roots failed because remote Sintel lacks `training/depth`; one-step TartanAir-only CUDA preflight passed and wrote `<train-host-data>/checkpoints/srcnn-v5-pixel-temporal-preflight-tartan/step-00000001.pt`.
- Relaunch at 17:08 CDT reached step 260, then crashed on a corrupt TartanAir flow `.npy` (`cannot reshape array of size 90040 into shape (480,640,2)`).
- `b8b08c5` added eager corrupt-npy filtering, but startup was too slow at full TartanAir scale. Codex stopped that attempt before training began.
- `10e75df` replaced eager scanning with lazy unreadable-pair skipping in `SequentialPairDataset`; TartanAir loader now reports corrupt npy paths clearly, and the pair dataset advances to the next readable pair.
- Verification after `10e75df`: `venv-py312/bin/python -m pytest tests/gaussian/test_datasets.py tests/sr -v` → 128 passed, 1 skipped, 15 warnings.
- Relaunched long pixel training as TartanAir-only at 17:20 CDT. Active process: python PID `2360`, parent cmd PID `15652`; latest observed log reached step 340 with finite loss, past the previous crash point.
- `docs/superpowers/notes/2026-05-04-v5-pixel-launch-status.md` is tracked in `913cc9f` and was updated locally with the active PID and TartanAir-only caveat.

### 17:24-17:28 CDT

C4 Gaussian spec-compliance pass and live monitor:

- Remote pixel training PID `2360` remained alive. Latest observed tail reached step `880` at 17:27 CDT with finite Phase-1 losses.
- C4 review found a high Gaussian Task 8 bug: persistent Gaussian means live in HR coordinates, but recurrent warp used LR motion bounds. This would kill most HR-space Gaussians on the next temporal frame.
- `d6bc655` fixed the model by lifting LR motion into HR field space before `warp_field`; regression `test_temporal_warp_uses_hr_field_coordinates` added.
- Verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_model_full_step.py -v` → 11 passed.
- Verification: `venv-py312/bin/python -m pytest tests/sr/gaussian_temporal -v` → 59 passed.
- `d6bc655` pushed to `origin/v0.2-dev` for Claude/remote sync.

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
