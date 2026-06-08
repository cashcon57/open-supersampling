#!/usr/bin/env bash
# Backfill v7 dashboard viz strips on the Linux training host.
#
# The trainer checkpoints every 500 steps, but the public dashboard only needs
# clean 1000-step milestones. This script is safe to run repeatedly: it renders
# missing milestone PNGs and exits silently when there is nothing to do.

set -euo pipefail

RUN_DIR="${RUN_DIR:-/home/cashc/checkpoints/srcnn-v7.0-pico-005}"
CONTAINER="${CONTAINER:-oss-trainer}"
REPO_IN_CONTAINER="${REPO_IN_CONTAINER:-/workspace/oss-gaussian}"
CHECKPOINT_IN_CONTAINER="${CHECKPOINT_IN_CONTAINER:-/checkpoints/srcnn-v7.0-pico-005}"
DATASET_IN_CONTAINER="${DATASET_IN_CONTAINER:-/datasets/tartanair_extracted}"
LOG="${LOG:-/home/cashc/v7-viz-backfill.log}"
MAX_PER_RUN="${MAX_PER_RUN:-2}"
MIN_STEP="${MIN_STEP:-6000}"
STEP_STRIDE="${STEP_STRIDE:-1000}"
N_PAIRS="${N_PAIRS:-6}"

[ -d "$RUN_DIR" ] || exit 0
command -v docker >/dev/null 2>&1 || exit 0
docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || exit 0

mkdir -p "$RUN_DIR/viz" "$(dirname "$LOG")"
count=0

while IFS= read -r ckpt; do
  base="$(basename "$ckpt")"
  step="${base#step-}"
  step="${step%.pt}"

  # Ignore aliases like step-00000500-final.pt; render only numeric checkpoints.
  if [[ ! "$step" =~ ^[0-9]{8}$ ]]; then
    continue
  fi

  # Dashboard strips use clean milestones: 1000, 2000, ...
  if (( 10#$step < MIN_STEP || 10#$step % STEP_STRIDE != 0 )); then
    continue
  fi

  out="$RUN_DIR/viz/step-${step}.png"
  if [ -f "$out" ]; then
    continue
  fi

  {
    echo "[$(date -Is)] render step-${step}"
    docker exec -w "$REPO_IN_CONTAINER" "$CONTAINER" bash -lc "python -u scripts/sr_temporal_inflight_viz.py \
      --output-dir '$CHECKPOINT_IN_CONTAINER' \
      --ckpt '$CHECKPOINT_IN_CONTAINER/step-${step}.pt' \
      --primary-version v7 \
      --tartanair-root '$DATASET_IN_CONTAINER' \
      --device cpu \
      --n-pairs '$N_PAIRS' \
      --once"
  } >>"$LOG" 2>&1

  test -f "$out"
  count=$((count + 1))
  if (( count >= MAX_PER_RUN )); then
    break
  fi
done < <(find "$RUN_DIR" -maxdepth 1 -name 'step-*.pt' | sort)

# Intentionally silent on success. systemd/cron should alert only on nonzero exit.
exit 0
