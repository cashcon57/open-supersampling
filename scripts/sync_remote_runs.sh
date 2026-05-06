#!/usr/bin/env bash
#
# sync_remote_runs.sh — periodically pull metrics.json + viz/*.png + log
# from a remote training host into a local mirror that the dashboard
# reads.
#
# Default: pulls from <train-host> (the v6 + v5 + v4 training host)
# into /tmp/oss-runs/. The dashboard's --output-dir points at one of
# these mirrored dirs; the run-picker discovers the rest by globbing
# the parent.
#
# Background-safe: uses scp with -B, polls every INTERVAL seconds
# (default 30). Logs progress to /tmp/sync-remote-runs.log.
#
# Usage:
#   ./scripts/sync_remote_runs.sh                       # uses defaults
#   ./scripts/sync_remote_runs.sh 60                    # poll every 60s

set -uo pipefail

INTERVAL="${1:-30}"
LOCAL_ROOT="${LOCAL_ROOT:-/tmp/oss-runs}"
REMOTE_HOST="${REMOTE_HOST:-<train-host>}"
REMOTE_ROOT="${REMOTE_ROOT:-<train-host-data>/checkpoints}"
LOG="${SYNC_LOG:-/tmp/sync-remote-runs.log}"

# Run dirs to mirror. Pattern matches what RUN_DIR_PATTERNS in the
# dashboard accepts.
RUNS=(
  "srcnn-v6-heavy-001"
  "srcnn-v5-pixel-temporal-validated"
  "srcnn-prod-v4-lpips"
)

mkdir -p "$LOCAL_ROOT"

echo "[$(date '+%H:%M:%S')] sync_remote_runs starting; interval=${INTERVAL}s" >> "$LOG"

while :; do
  for run in "${RUNS[@]}"; do
    local_dir="$LOCAL_ROOT/$run"
    mkdir -p "$local_dir/viz"

    # metrics.json + score_log.json — always pull (small files).
    scp -B -p -q "$REMOTE_HOST:$REMOTE_ROOT/$run/metrics.json" "$local_dir/" 2>/dev/null || true
    scp -B -p -q "$REMOTE_HOST:$REMOTE_ROOT/$run/score_log.json" "$local_dir/" 2>/dev/null || true

    # train.log — the dashboard uses its mtime to compute the
    # "training active" liveness signal and parses the header for
    # max_steps. Lives at <train-host-data>\logs\<run>.log on the windows host
    # convention. Mirror it to <local_dir>/train.log so the
    # dashboard's --log-file argument resolves.
    scp -B -p -q "$REMOTE_HOST:<train-host-data>/logs/$run.log" "$local_dir/train.log" 2>/dev/null || true

    # viz/*.png — only pull files that don't exist locally already.
    # Use a remote ls + per-file check so we don't repeat large transfers.
    remote_viz=$(ssh -o BatchMode=yes "$REMOTE_HOST" \
      "if (Test-Path '$REMOTE_ROOT/$run/viz') { Get-ChildItem '$REMOTE_ROOT/$run/viz' -Filter '*.png' | Select-Object -ExpandProperty Name }" \
      2>/dev/null | tr -d '\r')
    for png in $remote_viz; do
      [[ -z "$png" ]] && continue
      if [[ ! -f "$local_dir/viz/$png" ]]; then
        scp -B -p -q "$REMOTE_HOST:$REMOTE_ROOT/$run/viz/$png" "$local_dir/viz/" 2>/dev/null \
          && echo "[$(date '+%H:%M:%S')] $run/viz/$png synced" >> "$LOG"
      fi
    done
  done

  sleep "$INTERVAL"
done
