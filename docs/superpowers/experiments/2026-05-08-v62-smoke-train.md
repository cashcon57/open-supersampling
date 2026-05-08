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

**C4 SMOKE GATE: PASSED.** 2026-05-08 17:40 on 3080 Ti.

T6 wiring landed in commit `003fe2c` (V6Config + branched construction +
forward + 3-frame integration test) and the trainer CLI surface in
`8eb7d2f`. The C4 smoke command then ran end-to-end:

```powershell
python scripts/sr_train_v6.py `
    --backbone hat-tiny `
    --fusion-mode concat `
    --spawner-mode disocclusion `
    --latent-rank 16 `
    --max-steps 5 `
    --output-dir C:\temp\v62-smoke `
    --tartanair-root E:\datasets\tartanair_extracted `
    --batch-size 1 --patch-size 64 --smoke --no-bf16 `
    --num-workers 0 --first-ckpt-step 5 --device cuda
```

Result:

```text
G params=1,013,544  D params=4,304,513
step=1 loss=9.5207 char=0.2004 lpips=0.4111
step=2 loss=9.3541 char=0.2003 lpips=0.3707
step=3 loss=9.3041 char=0.2010 lpips=0.3649
step=4 loss=9.1725 char=0.1982 lpips=0.3591
step=5 loss=9.3441 char=0.2014 lpips=0.3769
ckpt -> C:\temp\v62-smoke\step-00000005.pt
elapsed=3.2s  final_loss=9.344100  exit=0
```

Loss trends down 9.52 -> 9.17 across the first 4 steps, with a small
step-5 bump. All values finite. No NaN / Inf. Checkpoint written.

**Pico-002 launch is unblocked.**
