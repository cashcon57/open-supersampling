# Codex handoff — v6 Stage 2 trajectory trainer

Date: 2026-05-06

Commit/push blocked by sandbox:

```text
fatal: Unable to create '<repo-root>/.git/index.lock': Operation not permitted
```

Use this commit title:

```text
v6(train): trajectory training loop with canvas continuity + temporal-consistency loss + early checkpoint
```

## Files changed

- `<repo-root>/scripts/sr_train_v6.py`
- `<repo-root>/oss/sr/v6/dataset.py`
- `<repo-root>/oss/sr/v6/losses.py`
- `<repo-root>/tests/sr/v6/test_train_loop.py`
- `<repo-root>/tests/sr/v6/test_dataset.py`
- `<repo-root>/tests/sr/v6/test_losses.py`

## Diff stat

```text
 oss/sr/v6/dataset.py           | 110 ++++++++++++++++
 oss/sr/v6/losses.py            |  14 ++-
 scripts/sr_train_v6.py         | 276 ++++++++++++++++++++++++++++++++++++-----
 tests/sr/v6/test_dataset.py    |  28 +++++
 tests/sr/v6/test_losses.py     |  10 ++
 tests/sr/v6/test_train_loop.py |  98 ++++++++++++++-
 6 files changed, 499 insertions(+), 37 deletions(-)
```

## Summary

- Added `--trajectory-length`; normalized default is `4` normally and `2` in `--smoke`.
- Added `--first-ckpt-step` default `100`; trainer writes that checkpoint in addition to regular cadence/final checkpoint.
- Added trajectory-yielding dataset wrappers so sampler indexes windows, not individual frames, while keeping source mix and held-out filtering.
- Reworked v6 trainer step to reset canvas once per trajectory, forward each frame with `motion_lr=None` on frame 0 and adjacent motion afterward, sum trajectory losses, backward once, then run EMA/scheduler/prune cadence as before.
- Wired motion-aware temporal consistency through `V6CompositeLoss(pred_prev=..., motion_lr=..., scale_factor=...)` so adjacent predictions stay alive in graph.
- Hardened auto-resume for older v6 checkpoint schema drift by loading model state non-strictly and skipping incompatible optimizer/EMA state with warnings.
- Added tests for trajectory smoke/resume, canvas continuity, trajectory boundary reset, non-zero `loss_tc`, and early checkpoint.

## Validation

```text
./venv-py312/bin/python -m pytest tests/sr/v6/test_train_loop.py -q
10 passed, 1 warning in 3.37s
```

```text
./venv-py312/bin/python -m pytest tests/sr/v6/ -q
239 passed, 10 warnings in 16.30s
```

Direct smoke:

```text
./venv-py312/bin/python scripts/sr_train_v6.py \
  --output-dir /private/tmp/oss-v6-smoke-stage2-10378 \
  --smoke --device cpu --backbone hat-tiny --patch-size 32 \
  --batch-size 1 --grad-accum 1 --first-ckpt-step 2 \
  --ckpt-every 100 --no-bf16
```

Result:

```text
final_step=5
checkpoint -> /private/tmp/oss-v6-smoke-stage2-10378/step-00000005.pt
```

Also wrote `/private/tmp/oss-v6-smoke-stage2-10378/step-00000002.pt`, and `metrics.json` had non-zero `loss_tc` on every smoke row.
