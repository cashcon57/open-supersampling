#!/usr/bin/env bash
#
# ci_auto_heal.sh — poll the most recent CI run on origin/main and, on
# failure, dispatch a codex session to diagnose and fix.
#
# Modes:
#   --once        Check once, dispatch on failure if needed, exit.
#   --watch       Loop forever (default 60s interval); dispatch on every
#                 NEW failure (tracked by the headSha so we don't double-fire).
#   --interval N  Poll every N seconds in --watch mode (default 60).
#   --timeout N   In --once mode, wait up to N seconds for an in-progress run
#                 to complete (default 0 = no wait).
#
# Dependencies: gh (authenticated), bash, scripts/dispatch_codex.sh.
#
# Exit codes:
#   0  CI is green or a fix has been dispatched / completed.
#   1  Hard error (gh auth, repo not found, etc).
#   2  Failure detected, codex dispatch failed.
#
# Operator integration:
#   - Run after every push: bash scripts/ci_auto_heal.sh --once --timeout 600
#   - Run forever in a tab: bash scripts/ci_auto_heal.sh --watch
#   - Or launchd-ize with a com.openssampling.ci-heal.plist watching every 5 min.

set -euo pipefail

REPO="cashcon57/open-supersampling"
BRANCH="main"
MODE="once"
INTERVAL=60
TIMEOUT=0
STATE_FILE="/tmp/.ci_auto_heal_last_dispatched_sha"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once) MODE="once"; shift ;;
    --watch) MODE="watch"; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

log() { echo "[ci_heal $(date +%H:%M:%S)] $*"; }

dispatch_fix() {
  local run_id="$1"
  local head_sha="$2"
  local short_sha="${head_sha:0:7}"

  if [[ -f "$STATE_FILE" ]] && [[ "$(cat "$STATE_FILE")" == "$head_sha" ]]; then
    log "already dispatched fix for $short_sha; skipping"
    return 0
  fi

  local slug="ci-heal-${short_sha}"
  local prompt_file="/tmp/prompt-${slug}.txt"
  local log_file="/tmp/codex-${slug}.log"

  log "fetching failed-job logs for run $run_id ($short_sha)"
  local fail_logs
  fail_logs="$(gh -R "$REPO" run view "$run_id" --log-failed 2>/dev/null | tail -300 || echo '(unable to fetch logs)')"

  local commit_title
  commit_title="$(gh -R "$REPO" run view "$run_id" --json displayTitle -q .displayTitle 2>/dev/null || echo '?')"

  cat > "$prompt_file" <<PROMPT
You are an autonomous CI healer for OpenSuperSampling at /Users/cashconway/OpenSuperSampling.

CI failed on commit ${short_sha} ("${commit_title}") on branch ${BRANCH}. Diagnose and fix.

Read first:
- /Users/cashconway/OpenSuperSampling/docs/coordination/codex-project-context.md (repo conventions, dispatch helper)

Failed-job log tail (last 300 lines):
---
${fail_logs}
---

Tasks:
1. Identify the root cause from the log tail. Common categories:
   - Test failures (assertion errors, fixture issues, regression-lock breakage)
   - Lint / type-check failures (ruff, mypy)
   - Import errors (missing dep, circular import after a refactor)
   - Flaky tests (only retry if you can prove it's flaky; otherwise treat as a real failure)
   - Format/whitespace failures
2. Fix the root cause in the source. Do NOT skip / xfail / disable a test unless it is provably testing wrong behavior — and if you do skip, add a TODO comment with a memo path.
3. Run pytest locally on the affected tests OR a focused subset to prove the fix works before pushing.
4. Commit with title: \`ci: fix <one-line description> (re ${short_sha})\`
5. Push to origin/main directly. The dispatch helper passes --sandbox danger-full-access.
6. After push, monitor the new CI run via \`gh -R ${REPO} run list --limit 3\`. Wait up to 10 minutes for it to complete. If GREEN: exit 0. If RED again: write \`docs/superpowers/experiments/$(date +%Y-%m-%d)-ci-heal-stuck-${short_sha}.md\` describing what you tried and why it failed, commit + push that memo, exit 2.

Autonomy rules: do not block on operator input. Use defensible judgment when ambiguous; document the choice in the commit message.

Time budget: 30 minutes wallclock max.

GO.
PROMPT

  log "dispatching codex for fix: slug=$slug log=$log_file"
  cd /Users/cashconway/OpenSuperSampling
  nohup bash scripts/dispatch_codex.sh "$slug" "$prompt_file" \
    > "/tmp/${slug}.dispatch.log" 2>&1 &
  disown
  echo "$head_sha" > "$STATE_FILE"
  log "codex dispatched (PID watchable via /tmp/${slug}.dispatch.log)"
  return 0
}

check_once() {
  local waited=0
  while :; do
    local row
    row="$(gh -R "$REPO" run list --branch "$BRANCH" --workflow CI \
            --json databaseId,status,conclusion,headSha,displayTitle \
            --limit 1 -q '.[0]')"
    if [[ -z "$row" ]] || [[ "$row" == "null" ]]; then
      log "no recent CI runs; nothing to do"
      return 0
    fi

    local status conclusion head_sha db_id
    status="$(echo "$row" | jq -r .status)"
    conclusion="$(echo "$row" | jq -r .conclusion)"
    head_sha="$(echo "$row" | jq -r .headSha)"
    db_id="$(echo "$row" | jq -r .databaseId)"

    if [[ "$status" != "completed" ]]; then
      if (( TIMEOUT > 0 )) && (( waited < TIMEOUT )); then
        log "in_progress (${head_sha:0:7}); waiting ${INTERVAL}s (waited=${waited}s of ${TIMEOUT}s)"
        sleep "$INTERVAL"
        waited=$((waited + INTERVAL))
        continue
      fi
      log "in_progress (${head_sha:0:7}); not waiting"
      return 0
    fi

    if [[ "$conclusion" == "success" ]]; then
      log "GREEN (${head_sha:0:7})"
      return 0
    fi

    log "RED — conclusion=$conclusion sha=${head_sha:0:7} run=$db_id"
    dispatch_fix "$db_id" "$head_sha"
    return 0
  done
}

case "$MODE" in
  once)
    check_once
    ;;
  watch)
    log "watching $REPO@$BRANCH every ${INTERVAL}s; dispatching on new failures"
    while :; do
      check_once || true
      sleep "$INTERVAL"
    done
    ;;
esac
