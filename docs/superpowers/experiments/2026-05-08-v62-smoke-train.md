# v6.2 Smoke Training Step

Date: 2026-05-08

## Purpose

Verify that the full v6.2 forward, backward, and optimizer step complete for
five synthetic training steps without NaN or Inf metrics.

## Prerequisite Gate

The smoke run depends on the T6 model orchestrator wiring. Per the bundle
instructions, the gate is:

```bash
grep -q "fusion_mode" oss/sr/v6/model.py
```

Result: `T6 prereq missing`.

## Outcome

The smoke training command was not run because `oss/sr/v6/model.py` does not
yet contain `fusion_mode`. This is the expected clean exit path for C4 when T6
has not landed.

Planned command once T6 lands:

```bash
python scripts/sr_train_v6.py --v62 --max-steps 5 --output-dir /tmp/v62-smoke --tartanair-root /e/checkpoints/tartanair-shim --batch-size 1 --patch-size 64
```

Checkpoint: not expected; run skipped by prerequisite gate.
