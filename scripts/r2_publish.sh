#!/usr/bin/env bash
#
# r2_publish.sh — push the dashboard-public/ tree to R2 via the
# upload.opensupersampling.com Worker proxy.
#
# Reads creds from .secrets/r2-credentials.env (mode 600, gitignored).
# Walks the local source dir, infers content-type from extension, PUTs
# each file with bearer auth. Idempotent: re-running re-uploads.
#
# Usage:
#   bash scripts/r2_publish.sh                  # push dashboard-public/
#   bash scripts/r2_publish.sh /path/to/dir     # push an arbitrary dir
#   PARALLEL=12 bash scripts/r2_publish.sh      # tune parallelism
#
# Reports per-file pass/fail with totals at the end.

set -uo pipefail

SRC="${1:-dashboard-public}"
PARALLEL="${PARALLEL:-8}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CREDS="${REPO_ROOT}/.secrets/r2-credentials.env"

if [[ ! -f "$CREDS" ]]; then
  echo "missing $CREDS" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CREDS"

if [[ -z "${WORKER_UPLOAD_URL:-}" || -z "${WORKER_SHARED_SECRET:-}" ]]; then
  echo "credentials missing WORKER_UPLOAD_URL or WORKER_SHARED_SECRET" >&2
  exit 2
fi

if [[ ! -d "$SRC" ]]; then
  echo "source dir not found: $SRC" >&2
  exit 2
fi

content_type_for() {
  case "$1" in
    *.html|*.htm) echo "text/html; charset=utf-8" ;;
    *.json) echo "application/json" ;;
    *.css)  echo "text/css; charset=utf-8" ;;
    *.js)   echo "application/javascript; charset=utf-8" ;;
    *.svg)  echo "image/svg+xml" ;;
    *.png)  echo "image/png" ;;
    *.jpg|*.jpeg) echo "image/jpeg" ;;
    *.webp) echo "image/webp" ;;
    *.mp4)  echo "video/mp4" ;;
    *.webm) echo "video/webm" ;;
    *.txt|*.log) echo "text/plain; charset=utf-8" ;;
    *.md)   echo "text/markdown; charset=utf-8" ;;
    *)      echo "application/octet-stream" ;;
  esac
}

upload_one() {
  local src="$1" key="$2"
  local ctype
  ctype="$(content_type_for "$src")"
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' \
    -X PUT "${WORKER_UPLOAD_URL}/upload/${key}" \
    -H "Authorization: Bearer ${WORKER_SHARED_SECRET}" \
    -H "Content-Type: ${ctype}" \
    --data-binary "@${src}")"
  if [[ "$code" == "200" ]]; then
    printf '  OK  %s  (%s)\n' "$key" "$ctype"
  else
    printf '  ERR %s  (HTTP %s)\n' "$key" "$code" >&2
    return 1
  fi
}
export -f upload_one content_type_for
export WORKER_UPLOAD_URL WORKER_SHARED_SECRET

echo "[r2_publish] source=${SRC} target=${WORKER_UPLOAD_URL} parallel=${PARALLEL}"

cd "$SRC"
find . -type f -not -path '*/.*' -print0 | \
  xargs -0 -I{} -P "$PARALLEL" bash -c '
    src="{}"
    key="${src#./}"
    upload_one "$src" "$key"
  '

echo "[r2_publish] done"
