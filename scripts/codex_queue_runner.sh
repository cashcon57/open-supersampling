#!/usr/bin/env bash
#
# codex_queue_runner.sh — sequentially dispatch codex prompts from a queue.
#
# Reads prompts from /tmp/codex-queue/*.txt in lexical order. Waits for any
# in-flight `codex exec` to finish before dispatching the next. Survives
# macOS sleep cycles (sleep pauses the script; wake resumes it). Pure bash;
# no Python/Node deps.
#
# Each prompt-file's basename (minus `.txt`) becomes the codex slug.
# Lex order: prefix prompts with `010_`, `020_`, etc. for desired ordering.
#
# Run: nohup bash scripts/codex_queue_runner.sh > /tmp/codex-queue.log 2>&1 &
# Stop: touch /tmp/codex-queue/.stop  (graceful — waits for in-flight to finish)
# Kill: pkill -f codex_queue_runner.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_DIR="${OSS_QUEUE_DIR:-/tmp/codex-queue}"
DONE_DIR="${OSS_QUEUE_DONE_DIR:-/tmp/codex-queue-done}"
STOP_MARKER="${QUEUE_DIR}/.stop"
POLL_SEC="${POLL_SEC:-30}"
# MAX_PARALLEL: how many codex exec processes can run concurrently.
# Default 1 (serial, safe — no git-commit races). Bump to 3+ when prompts
# work on disjoint files / don't commit / use git worktrees.
MAX_PARALLEL="${MAX_PARALLEL:-3}"

mkdir -p "$QUEUE_DIR" "$DONE_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

active_codex_count() {
  pgrep -af "codex exec" 2>/dev/null | grep -vc "codex_queue_runner\|grep" || true
}

log "queue runner starting; queue=${QUEUE_DIR} done=${DONE_DIR} poll=${POLL_SEC}s max_parallel=${MAX_PARALLEL}"

while :; do
  # Stop marker check
  if [[ -f "$STOP_MARKER" ]]; then
    log "stop marker found; waiting for in-flight then exiting"
    while (( $(active_codex_count) > 0 )); do
      sleep "$POLL_SEC"
    done
    log "graceful exit"
    rm -f "$STOP_MARKER"
    exit 0
  fi

  # Throttle: wait while at MAX_PARALLEL active codex jobs.
  active="$(active_codex_count)"
  if (( active >= MAX_PARALLEL )); then
    sleep "$POLL_SEC"
    continue
  fi

  # Pop next prompt (lexical order)
  next="$(ls "$QUEUE_DIR"/*.txt 2>/dev/null | sort | head -1 || true)"
  if [[ -z "$next" || ! -f "$next" ]]; then
    sleep 60
    continue
  fi

  slug="$(basename "$next" .txt)"
  log "dispatching slug=$slug prompt=$next active=${active}/${MAX_PARALLEL}"

  cd "$REPO_ROOT"
  nohup bash scripts/dispatch_codex.sh "$slug" "$next" \
    > "/tmp/codex-queue-${slug}.dispatch.log" 2>&1 &
  disown

  # Brief grace period for dispatch_codex.sh to slurp the prompt file
  # synchronously into FULL_PROMPT (line 73 of that script). If we move
  # the file too soon, dispatch fails with "prompt file not found" and
  # no codex actually spawns. 5s is plenty for `cat <prompt>`.
  sleep 5

  # Now safe to archive the prompt — dispatch_codex has already read it.
  mv "$next" "$DONE_DIR/"

  # Brief wait for the new codex to register before popping the next slot.
  # We do NOT block until completion — that's what made the runner serial.
  # Instead we rely on the active-count throttle at top of the loop to
  # gate new dispatches until in-flight count drops below MAX_PARALLEL.
  attempts=0
  while (( $(active_codex_count) <= active )) && (( attempts < 12 )); do
    sleep 5
    attempts=$((attempts+1))
  done

  if (( $(active_codex_count) <= active )); then
    log "WARN slug=$slug failed to spawn within 60s; check /tmp/codex-queue-${slug}.dispatch.log"
  fi
done
