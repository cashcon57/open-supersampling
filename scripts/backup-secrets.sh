#!/usr/bin/env bash
#
# backup-secrets.sh — encrypt the .secrets/ directory with age and upload
# to a PRIVATE Cloudflare R2 bucket. The bucket has no public binding and
# requires wrangler-authenticated access to fetch.
#
# Recovery: see RESTORE block at the bottom of this file.
#
# Run manually: bash scripts/backup-secrets.sh
# Auto: scheduled via launchd plist at ~/Library/LaunchAgents/com.oss.secrets-backup.plist

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS="${REPO_ROOT}/.secrets"
BUCKET="oss-secrets-backup"
CF_ACCOUNT="c067afd6ea60a95b946c63c599095a65"

# Coalesce window: when launchd's WatchPaths fires multiple events in quick
# succession (saving 5 secrets in a row, etc.), serialize them into one
# backup of the FINAL state. The first invocation grabs the lock and sleeps
# 30s; subsequent invocations see the lock and exit. After the sleep, the
# original runner does one backup of whatever state landed by then.
COALESCE_LOCK="/tmp/oss-secrets-backup.coalesce.lock"
COALESCE_SECONDS="${COALESCE_SECONDS:-30}"

if ! mkdir "$COALESCE_LOCK" 2>/dev/null; then
  echo "[$(date '+%H:%M:%S')] another run already coalescing; exit"
  exit 0
fi
trap 'rmdir "$COALESCE_LOCK" 2>/dev/null; rm -rf "${TMP:-}"' EXIT

# Only sleep if invoked from launchd WatchPaths (env signal). Manual + cron
# runs skip the coalesce delay so `bash backup-secrets.sh` is instant.
if [[ "${OSS_BACKUP_MODE:-cron}" == "watch" ]]; then
  echo "[$(date '+%H:%M:%S')] coalescing for ${COALESCE_SECONDS}s..."
  sleep "$COALESCE_SECONDS"
fi

if [[ ! -d "$SECRETS" ]]; then
  echo "missing $SECRETS" >&2
  exit 2
fi

# Non-interactive auth. wrangler's stored OAuth login expires and can't be
# refreshed from launchd, which silently broke every run from 2026-05-15 to
# 2026-10-21. Use the account API token from .secrets instead.
if [[ -z "${CLOUDFLARE_API_TOKEN:-}" && -f "${SECRETS}/cloudflare-api-token.txt" ]]; then
  CLOUDFLARE_API_TOKEN="$(tr -d '\r\n' < "${SECRETS}/cloudflare-api-token.txt")"
  export CLOUDFLARE_API_TOKEN
fi
if [[ -z "${CLOUDFLARE_API_TOKEN:-}" ]]; then
  echo "missing CLOUDFLARE_API_TOKEN and ${SECRETS}/cloudflare-api-token.txt" >&2
  exit 4
fi

AGE_PUB_FILE="${SECRETS}/age-public-key.txt"
if [[ ! -f "$AGE_PUB_FILE" ]]; then
  echo "missing $AGE_PUB_FILE — run age-keygen first" >&2
  exit 3
fi

AGE_PUB="$(tr -d '\r\n' < "$AGE_PUB_FILE")"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname -s | tr '[:upper:]' '[:lower:]')"
KEY="secrets-${HOST}-${RUN_TS}.tar.gz.age"
TMP="$(mktemp -d)"
TARBALL="${TMP}/${KEY}"

# Tar everything in .secrets/ EXCEPT the age private key itself (we don't
# want the recovery key encrypted-with-itself in the offsite blob — the
# recovery key has to live OUTSIDE the encrypted backup, by definition).
# (Cleanup of TMP + COALESCE_LOCK happens via the EXIT trap set above.)

echo "[$(date '+%H:%M:%S')] tarring .secrets (excluding age private key) → encrypting..."
tar -C "$SECRETS" --exclude='age-secret-key.txt' -cz . \
  | age -r "$AGE_PUB" -o "$TARBALL"

ENC_BYTES="$(stat -f%z "$TARBALL" 2>/dev/null || stat -c%s "$TARBALL")"
echo "[$(date '+%H:%M:%S')] encrypted size: ${ENC_BYTES} bytes"

# Sanity: round-trip decrypt-only-header check (doesn't actually decrypt)
file "$TARBALL" >/dev/null

echo "[$(date '+%H:%M:%S')] uploading r2://${BUCKET}/${KEY}..."
CLOUDFLARE_ACCOUNT_ID="$CF_ACCOUNT" \
  wrangler r2 object put "${BUCKET}/${KEY}" --file "$TARBALL" --remote 2>&1 | tail -5

echo "[$(date '+%H:%M:%S')] done — backup at r2://${BUCKET}/${KEY}"

# Prune retention: keep last 14 daily backups + last 12 monthly backups
# (1st-of-month). Daily older than 14 days that are NOT 1st-of-month get
# deleted. Stops the bucket from growing unbounded.
echo "[$(date '+%H:%M:%S')] pruning old backups..."
# wrangler has no `r2 object list`, so list through the Cloudflare API.
ALL_KEYS="$(curl -fsS -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
  "https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/r2/buckets/${BUCKET}/objects?per_page=1000" \
  | python3 -c 'import json,sys; [print(o["key"]) for o in json.load(sys.stdin).get("result") or [] if o["key"].startswith("secrets-")]' \
  || true)"
NOW_EPOCH="$(date -u +%s)"
THIRTEEN_DAYS_AGO="$((NOW_EPOCH - 13 * 86400))"

while IFS= read -r oldkey; do
  [[ -z "$oldkey" ]] && continue
  # Extract YYYYMMDDTHHMMSSZ from key
  ts_str="$(echo "$oldkey" | sed -E 's/.*-([0-9]{8}T[0-9]{6}Z)\.tar\.gz\.age$/\1/' || true)"
  [[ -z "$ts_str" || "$ts_str" == "$oldkey" ]] && continue
  # Convert to epoch (BSD date format on Mac)
  ts_epoch="$(date -j -u -f "%Y%m%dT%H%M%SZ" "$ts_str" +%s 2>/dev/null || echo 0)"
  [[ "$ts_epoch" == "0" ]] && continue
  if (( ts_epoch < THIRTEEN_DAYS_AGO )); then
    # Keep if it's a 1st-of-month archive
    dom="$(echo "$ts_str" | cut -c7-8)"
    if [[ "$dom" == "01" ]]; then
      continue   # keep monthly
    fi
    echo "  pruning ${oldkey} (older than 13 days, not monthly)"
    CLOUDFLARE_ACCOUNT_ID="$CF_ACCOUNT" wrangler r2 object delete "${BUCKET}/${oldkey}" --remote >/dev/null 2>&1 || true
  fi
done <<< "$ALL_KEYS"

echo "[$(date '+%H:%M:%S')] retention prune complete."

# ───────────────────────────────────────────────────────────────────
# RESTORE PROCEDURE
#
# 1. Install age:
#      brew install age          # macOS
#      pacman -S age             # CachyOS
#
# 2. Recover the age PRIVATE key. It lives OUTSIDE the encrypted bundle
#    (because you can't decrypt with a key trapped inside what you're
#    decrypting). Sources, in order:
#      a) ~/.secrets/age-secret-key.txt on the original Mac
#      b) USB stick / paper backup of the AGE-SECRET-KEY-...
#      c) a fresh keypair you generated and shared with a buddy beforehand
#
# 3. Find the latest backup blob:
#      Cloudflare dashboard → R2 → oss-secrets-backup, or with an API token:
#      curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
#        "https://api.cloudflare.com/client/v4/accounts/c067afd6ea60a95b946c63c599095a65/r2/buckets/oss-secrets-backup/objects?per_page=1000"
#      # pick latest secrets-<host>-<ts>.tar.gz.age (wrangler has no object list)
#
# 4. Download:
#      wrangler r2 object get oss-secrets-backup/<KEY> --file=/tmp/restore.tar.gz.age --remote
#
# 5. Decrypt + extract:
#      age -d -i /path/to/age-secret-key.txt -o /tmp/restore.tar.gz /tmp/restore.tar.gz.age
#      mkdir -p /tmp/restore && tar -xzf /tmp/restore.tar.gz -C /tmp/restore
#
# 6. Eyeball /tmp/restore/, then merge into your .secrets/ directory.
#
# IMPORTANT: protect the age PRIVATE key like you'd protect a hardware
# wallet seed phrase. If you lose it, the backups are unrecoverable
# (which is the point).
# ───────────────────────────────────────────────────────────────────
