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
# Every dispatched prompt gets a SUBAGENT-ORCHESTRATION PREAMBLE prepended
# (see PREAMBLE below). This nudges codex to fan out independent work to
# parallel subagents instead of executing sequentially — high throughput
# matters more to us than minimum-token completion.
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
LOG="/tmp/codex-${SLUG}.log"

PREAMBLE='# Multi-agent orchestration directive (auto-prepended by dispatch_codex.sh)

If this prompt contains 2+ independent tasks (multiple commits, multiple files
that do not depend on each other, multiple research questions), DISPATCH THEM
TO PARALLEL SUBAGENTS rather than executing sequentially. Codex supports
spawning subagents via Agent or task-spawn tools; use them.

Heuristics for what qualifies as "independent":
- Two file edits that touch different files and dont share state -> parallel
- Five Playwright screenshots at five viewports -> parallel
- Reading three reference files to inform one edit -> parallel reads, then
  one edit
- Editing the same file in two ways -> sequential (avoid race)
- A commit that depends on a prior tests pass -> sequential

Prefer parallel by default. Only serialize when there is a real dependency
between tasks. Wallclock throughput matters more than per-task efficiency.

After parallel work converges, do a final integration pass on the main thread
to commit + push.

End of preamble. Original prompt follows below.
---
'

echo "[dispatch_codex] slug=${SLUG} log=${LOG} prompt=${PROMPT_FILE:-stdin}" >&2

# Build full prompt = PREAMBLE + (file or stdin)
if [[ -n "$PROMPT_FILE" ]]; then
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "prompt file not found: $PROMPT_FILE" >&2
    exit 2
  fi
  FULL_PROMPT="${PREAMBLE}$(cat "$PROMPT_FILE")"
else
  FULL_PROMPT="${PREAMBLE}$(cat -)"
fi

# Pipe via stdin to avoid ARG_MAX + special-char parsing issues.
exec codex exec \
  --skip-git-repo-check \
  --sandbox danger-full-access \
  - <<<"$FULL_PROMPT" > "$LOG" 2>&1
