#!/usr/bin/env bash
#
# watch_and_publish.sh — long-running file watcher + R2 publisher.
#
# Designed to run on the 3080ti training host. Watches the local
# /tmp/oss-runs/ mirror (synced from E:/checkpoints/ via the existing
# sync helper), regenerates dashboard-public/data.json + viz strips,
# and pushes deltas to R2 through the upload.opensupersampling.com
# Worker proxy every $INTERVAL seconds.
#
# Idempotent + crash-tolerant. Runs forever; orphan-spawn it via
# nohup or the WMI Win32_Process Create pattern used for training.
#
# Env (read from .secrets/r2-credentials.env):
#   WORKER_UPLOAD_URL      e.g. https://upload.opensupersampling.com
#   WORKER_SHARED_SECRET   bearer token
#
# Env (optional):
#   INTERVAL=30            poll cadence in seconds (default 30)
#   SOURCE_DIR=...         local source tree (default /tmp/oss-runs)
#   STAGING_DIR=...        where build_public_dashboard.py writes
#                          (default /tmp/dashboard-public-staging)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CREDS="${REPO_ROOT}/.secrets/r2-credentials.env"

if [[ ! -f "$CREDS" ]]; then
  echo "missing $CREDS" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$CREDS"

INTERVAL="${INTERVAL:-30}"
# SOURCE_DIR auto-detect:
#   - WSL2 (3080ti)            → /mnt/e/checkpoints
#   - Git Bash MSYS (Windows)  → /e/checkpoints
#   - Mac mirror               → /tmp/oss-runs
if [[ -z "${SOURCE_DIR:-}" ]]; then
  if [[ -d /mnt/e/checkpoints ]]; then
    SOURCE_DIR="/mnt/e/checkpoints"
  elif [[ -d /e/checkpoints ]]; then
    SOURCE_DIR="/e/checkpoints"
  elif [[ -d "E:/checkpoints" ]]; then
    SOURCE_DIR="E:/checkpoints"
  else
    SOURCE_DIR="/tmp/oss-runs"
  fi
fi
STAGING_DIR="${STAGING_DIR:-/tmp/dashboard-public-staging}"
HASH_FILE="${STAGING_DIR}/.last-hashes"

mkdir -p "$STAGING_DIR/runs" "$STAGING_DIR/viz" || true

content_type_for() {
  case "$1" in
    *.html|*.htm) echo "text/html; charset=utf-8" ;;
    *.json) echo "application/json" ;;
    *.png)  echo "image/png" ;;
    *.txt|*.log) echo "text/plain; charset=utf-8" ;;
    *) echo "application/octet-stream" ;;
  esac
}

upload_one() {
  local src="$1" key="$2"
  local ctype
  ctype="$(content_type_for "$src")"
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 60 \
    -X PUT "${WORKER_UPLOAD_URL}/upload/${key}" \
    -H "Authorization: Bearer ${WORKER_SHARED_SECRET}" \
    -H "Content-Type: ${ctype}" \
    --data-binary "@${src}")" || code="curl-fail"
  if [[ "$code" != "200" ]]; then
    echo "  PUT FAIL ${key} (HTTP ${code})" >&2
    return 1
  fi
}

stage_run_files() {
  # Mirror the curated allow-list of run dirs into the staging tree
  # the build script expects. Skip if nothing changed.
  local allow_list=(
    srcnn-v6.1-pico-001
    srcnn-v6-pico-001
    srcnn-v6-heavy-001
    srcnn-v5-pixel-temporal-validated
    srcnn-v5-pixel-temporal-clean-restart-override
    srcnn-prod-v4-lpips
  )
  for run in "${allow_list[@]}"; do
    local src_run="${SOURCE_DIR}/${run}"
    local dst_run="${STAGING_DIR}/runs/${run}"
    [[ -d "$src_run" ]] || continue
    mkdir -p "$dst_run/viz"
    # rsync (portable Mac + WSL + Linux); -a archive, -u update-only-newer
    rsync -au "$src_run/metrics.json" "$dst_run/" 2>/dev/null || true
    rsync -au "$src_run/score_log.json" "$dst_run/" 2>/dev/null || true
    rsync -au "$src_run/events.json" "$dst_run/" 2>/dev/null || true
    rsync -au "$src_run/gpu_status.json" "$dst_run/" 2>/dev/null || true
    if [[ -d "$src_run/viz" ]]; then
      rsync -au --include='*.png' --exclude='*' "$src_run/viz/" "$dst_run/viz/" 2>/dev/null || true
    fi
  done
}

# Capture a fresh GPU snapshot for the active run. Tries local nvidia-smi
# first (when running on the training host), then falls back to SSHing the
# tailnet-aliased training host (when running on the maintainer's macbook).
GPU_REMOTE_HOST="${GPU_REMOTE_HOST:-3080ti-windows}"
GPU_SSH_OPTS="${GPU_SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=5}"

capture_gpu_status() {
  local active="srcnn-v6.1-pico-001"
  local dst="${STAGING_DIR}/runs/${active}"
  [[ -d "$dst" ]] || return 0
  local csv=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    csv="$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" || csv=""
  fi
  if [[ -z "$csv" && -n "$GPU_REMOTE_HOST" ]]; then
    # SSH fallback — works from macbook against 3080ti-windows tailnet alias.
    csv="$(ssh ${GPU_SSH_OPTS} "${GPU_REMOTE_HOST}" 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits' 2>/dev/null)" || csv=""
  fi
  [[ -n "$csv" ]] || return 0
  local captured_at tmp_json
  captured_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  tmp_json="$dst/gpu_status.json.tmp"
  python3 -c "
import json, sys
csv = '''$csv'''.strip()
parts = [p.strip() for p in csv.splitlines()[0].split(',')]
if len(parts) < 4: sys.exit(1)
name, used, total, util = parts[0], int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))
payload = dict(captured_at='$captured_at', gpu_name=name, memory_used_mib=used, memory_total_mib=total,
               memory_used_pct=round(used*100/total, 1) if total else 0.0, utilization_pct=util)
print(json.dumps(payload, indent=2, sort_keys=True))
" > "$tmp_json" 2>/dev/null && mv "$tmp_json" "$dst/gpu_status.json"
}

publish_changed() {
  # Walk the staging tree, hash each file, upload only changed.
  local now_hashes
  now_hashes="$(cd "$STAGING_DIR" && find . -type f -not -path '*/.last-hashes' -not -path '*/\.*' -print0 | \
    xargs -0 -P4 -I{} sh -c 'printf "%s  %s\n" "$(shasum -a 256 "$1" | cut -d" " -f1)" "${1#./}"' _ {} | sort)"
  local prev_hashes=""
  [[ -f "$HASH_FILE" ]] && prev_hashes="$(cat "$HASH_FILE")"

  local changed
  changed="$(diff <(printf '%s' "$prev_hashes") <(printf '%s' "$now_hashes") | grep '^>' | awk '{print $3}')"
  local n=0
  while IFS= read -r key; do
    [[ -z "$key" ]] && continue
    upload_one "${STAGING_DIR}/${key}" "$key" && n=$((n+1)) || true
  done <<< "$changed"
  printf '%s' "$now_hashes" > "$HASH_FILE"
  printf '[%s] cycle pushed=%d total=%d\n' "$(date '+%H:%M:%S')" "$n" "$(printf '%s' "$now_hashes" | wc -l)"
}

echo "[watch_and_publish] interval=${INTERVAL}s source=${SOURCE_DIR} staging=${STAGING_DIR} target=${WORKER_UPLOAD_URL}"

while :; do
  stage_run_files
  capture_gpu_status
  # Build the static index.html + data.json from staging tree.
  if [[ -f "${REPO_ROOT}/scripts/build_public_dashboard.py" ]]; then
    python3 "${REPO_ROOT}/scripts/build_public_dashboard.py" \
      --runs-dir "${STAGING_DIR}/runs" \
      --out "${STAGING_DIR}" >/dev/null 2>&1 || \
      echo "[watch_and_publish] build_public_dashboard.py failed, continuing"
  fi
  if [[ -f "${REPO_ROOT}/tools/check_data_schema.py" ]]; then
    if ! python3 "${REPO_ROOT}/tools/check_data_schema.py" "${STAGING_DIR}/data.json" >/dev/null 2>&1; then
      echo "[watch_and_publish] schema check FAILED; skipping publish this cycle"
      sleep "$INTERVAL"
      continue
    fi
  fi
  # Canonical index.html lives in dashboard-public/ (edited by codex/hand).
  # build_public_dashboard.py only regenerates it when __PITCH_HTML__ marker
  # is still present, which means once the staging copy has been substituted
  # it freezes. Force-rsync the canonical file every cycle so codex edits to
  # dashboard-public/index.html actually land on R2.
  if [[ -f "${REPO_ROOT}/dashboard-public/index.html" ]]; then
    rsync -au "${REPO_ROOT}/dashboard-public/index.html" "${STAGING_DIR}/index.html" 2>/dev/null || true
  fi
  publish_changed
  sleep "$INTERVAL"
done
