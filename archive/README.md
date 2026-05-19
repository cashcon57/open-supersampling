# `archive/` — snapshots preserved across host wipes

This tree holds artifacts that would otherwise be lost during a host reinstall.
The primary use is the 2026-05-17 wipe of the 3080 Ti training host as it
moves from Windows + WSL2 to CachyOS Linux native.

See `../Starting up after wipe.md` in the repo root for the runbook that
consumes these files.

## Layout

```
archive/
├── v7-pico-005-snapshot-2026-05-16/   # active run, full checkpoints
│   ├── step-00000100.pt               # 3.8 MB — earliest warmup ckpt
│   ├── step-00000500.pt               # 3.8 MB
│   ├── step-00000500-final.pt         # 3.8 MB — clean-exit save
│   ├── step-00000506-final.pt         # 11 MB — first save under the
│   │                                  #          parent-child enabled
│   │                                  #          (canvas grew at 506)
│   ├── step-00001000.pt               # 11 MB
│   ├── step-00002000.pt               # 11 MB
│   ├── step-00005000.pt               # 11 MB — RESUME FROM THIS ONE
│   ├── history.jsonl                  # 47 rows of training metrics
│   ├── score_log_v7.json              # held-out eval scores
│   ├── gpu_status.json                # last-seen GPU status snapshot
│   └── stdout.log                     # one of the early launcher logs
│
├── legacy-runs/                        # metrics only — no checkpoints
│   ├── srcnn-prod-v4-lpips/           # v4 baseline, ~step 419k
│   ├── srcnn-v5-pixel-temporal-validated/  # v5 baseline, step 80k
│   ├── srcnn-v6-pico-001/             # v6 early
│   ├── srcnn-v6.1-pico-001/           # v6.1, step 14257 final
│   └── srcnn-v6.2-pico-002/           # v6.2 pico, step 74085 final
│                                       # (~32 GB of intermediate ckpts
│                                       # NOT preserved — viz frames live
│                                       # on R2 at opensupersampling.org)
│
├── legacy-windows-launchers/           # historical reference
│   ├── launch_v7_debug.ps1            # primary trainer launcher
│   ├── restart_v7.ps1                 # restart helper
│   ├── check_v7.ps1                   # status probe
│   └── find_v7.ps1                    # locate-running-trainer helper
│                                       # (these are PowerShell — they were
│                                       # specific to the WSL2 + Windows
│                                       # setup. On Linux native, the
│                                       # docker-compose.yml + entrypoint
│                                       # in docker/trainer/ replaces them
│                                       # entirely.)
│
└── README.md                          # this file
```

## What to do with this after the new host is stable

Once the 3080 Ti is running on Linux and the training run has continued for
at least a week without regression:

1. **Delete `archive/v7-pico-005-snapshot-2026-05-16/step-*.pt`** — the
   binary blobs are large (~63 MB) and the next regular checkpoints will be
   on the new host. Keep the JSON / JSONL metric files: they are tiny and
   form a permanent historical record.
2. **Leave `archive/legacy-runs/`** in place — those metric files
   summarize prior training runs whose canvases / weights are not coming
   back. They are reference-only.
3. **Leave `archive/legacy-windows-launchers/`** for a few months in case
   anyone (human or agent) needs to reconstruct what the wiped Windows host
   was doing. Beyond ~6 months from the wipe, delete.

## Why these specific files and not others

The TartanAir dataset (553 GB / 550k files) is NOT preserved — it is large
enough to make git unhappy and the source-of-truth is the upstream dataset
at <https://theairlab.org/tartanair-dataset/>. Re-download time is on the
order of 1-2 hours on a fast connection.

The viz strip PNGs (~150 MB for the v7 run) are NOT preserved here — they
are already served from R2 at
`https://opensupersampling.org/runs/srcnn-v7.0-pico-005/viz/`. Pulling them
back into the local checkpoint directory after wipe is optional and only
matters if you want offline viewing of the lineage strips.

The intermediate v6.2-pico-002 checkpoints (every-N-step .pt files totaling
~32 GB) are NOT preserved — that run's terminal state and all of its
post-training analysis live in the metric files and on the dashboard. We do
not need the weights themselves for any active work.
