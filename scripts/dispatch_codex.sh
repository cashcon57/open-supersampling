#!/usr/bin/env bash
#
# dispatch_codex.sh — fire a codex-exec task with full git access and
# stream the log to /tmp/codex-<slug>.log.
#
# Usage:
#   ./scripts/dispatch_codex.sh <slug> < /path/to/prompt.txt
#   ./scripts/dispatch_codex.sh <slug> /path/to/prompt.txt
#   echo "task..." | ./scripts/dispatch_codex.sh <slug>
#
# Bakes in --sandbox danger-full-access so codex can git commit + push
# from inside the task. The OSS project is marked
# `trust_level = "trusted"` in ~/.codex/config.toml; this is the right
# permission level for OSS-internal codex sessions where the prompt
# is being driven by Claude (not arbitrary input from the network).
#
# DO NOT use this helper for prompts you didn't author. The
# danger-full-access mode lets the codex agent run any shell command,
# including outside the workspace. For any externally-driven codex
# task, fall back to the default workspace-write sandbox and accept
# the manual-commit handoff workflow.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <slug> [prompt-file]" >&2
  echo "       echo task | $0 <slug>" >&2
  exit 2
fi

SLUG="$1"
PROMPT_FILE="${2:-}"

if [[ -n "$PROMPT_FILE" ]]; then
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "prompt file not found: $PROMPT_FILE" >&2
    exit 2
  fi
  PROMPT_SRC="$PROMPT_FILE"
else
  PROMPT_SRC="-"
fi

LOG="/tmp/codex-${SLUG}.log"
echo "[dispatch_codex] slug=${SLUG} log=${LOG} prompt=${PROMPT_SRC}" >&2

if [[ "$PROMPT_SRC" == "-" ]]; then
  exec codex exec \
    --skip-git-repo-check \
    --sandbox danger-full-access \
    - > "$LOG" 2>&1
else
  exec codex exec \
    --skip-git-repo-check \
    --sandbox danger-full-access \
    "$(cat "$PROMPT_SRC")" > "$LOG" 2>&1
fi
