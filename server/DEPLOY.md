# OSS Capture Ingest — deployment runbook

First-time deploy of the FastAPI ingest server to Fly.io, fronting Cloudflare R2.

## Why Fly.io

- Single-region single-machine deploy in <5 minutes.
- Free tier covers initial dogfood traffic (single shared-cpu-1x machine, scale-to-zero).
- Native HTTPS + auto-renewing certs.
- Persistent volume for the JSON token store survives machine restarts.

The server itself is platform-agnostic (a FastAPI app + boto3 client) — Fly.io is the path of least resistance, not a hard dependency. Anything that can run a Dockerfile + bind a `/data` volume + set env vars will work (Railway, Render, fly.io, AWS App Runner, GCP Cloud Run, your own VPS).

## Prerequisites

1. R2 bucket exists (`ors-captures`) with an API token that has read+write+list scope on it.
2. R2 endpoint URL is known (`https://<account-id>.r2.cloudflarestorage.com`).
3. `flyctl` installed locally (`brew install flyctl`).
4. Logged into Fly: `flyctl auth login`.

## First deploy

From the repo root:

```bash
cd server
flyctl launch --no-deploy --copy-config --name oss-capture-ingest --region iad
```

This reads the existing `fly.toml`, registers the app, but doesn't start a machine yet.

Set R2 credentials as Fly secrets (NEVER bake into the image or commit):

```bash
flyctl secrets set \
  R2_ACCESS_KEY_ID="$(cat ../.secrets/r2-credentials.env | grep R2_ACCESS_KEY_ID | cut -d= -f2)" \
  R2_SECRET_ACCESS_KEY="$(cat ../.secrets/r2-credentials.env | grep R2_SECRET_ACCESS_KEY | cut -d= -f2)" \
  R2_ENDPOINT="$(cat ../.secrets/r2-credentials.env | grep R2_ENDPOINT | cut -d= -f2)" \
  R2_BUCKET=ors-captures
```

Deploy:

```bash
flyctl deploy
```

Verify:

```bash
curl https://oss-capture-ingest.fly.dev/healthz
# {"status":"ok","version":"1.0.0"}
```

## Custom domain (optional, for `capture.oss-supersampling.dev`)

Once the bare Fly URL is live and healthy:

```bash
flyctl certs create capture.oss-supersampling.dev
flyctl certs show capture.oss-supersampling.dev
```

Add the CNAME / A / AAAA records that `flyctl certs show` prints to the DNS provider for `oss-supersampling.dev` (Cloudflare). Wait ~30 s for cert issuance, then re-verify health on the new hostname.

## Mint the first contributor token

The token registry is process-local-with-disk-persistence. Mint a token via SSH to the live machine:

```bash
flyctl ssh console -C "python -m server.oss_capture_ingest.main mint-token --label dogfood-001"
```

This prints a UUID4 hex token and writes it to `/data/oss-capture-tokens.json`. Bake it into a per-game installer:

```bash
# back on local
python scripts/build_capture_installer.py \
  --game cyberpunk-2077 \
  --game-exe-name Cyberpunk2077.exe \
  --proxy-dll-name dxgi.dll \
  --capture-api-base https://capture.oss-supersampling.dev \
  --token <UUID-from-mint-token> \
  --mode lite \
  --output ./out/installer-cp77/
```

(`--mode` flag pending on Codex C24 in R9 ask.)

## Smoke test from local

```bash
# Build a tiny dummy frame + meta
python -c "
import json, requests, uuid
meta = {
    'schema_version': 1, 'game_id': 'cyberpunk-2077', 'game_version': '2.13',
    'session_uuid': str(uuid.uuid4()), 'frame_uuid': str(uuid.uuid4()),
    'captured_at_unix': 1777940000.0, 'lr_resolution': [1920, 1080],
    'hr_resolution': [3840, 2160], 'hr_source': 'dlss-quality',
    'jitter_offset_uv': [0.234, 0.781], 'motion_mean_magnitude_px': 12.4,
    'perceptual_hash_64': '0x0123456789abcdef',
    'user_consent_token': 'dogfood', 'uploader_version': '1.0.0',
    'capture_mode': 'lite',
}
r = requests.post(
    'https://capture.oss-supersampling.dev/ingest',
    headers={'Authorization': 'Bearer <YOUR-TOKEN>'},
    files={'frame': ('f.exr', b'EXR-MOCK-BODY' * 100, 'image/x-exr')},
    data={'meta': json.dumps(meta)},
    timeout=30,
)
print(r.status_code, r.json())
"
```

Expected: `200 {'status': 'ok', 'frame_uuid': '...', 'exr_key': 'cyberpunk-2077/2026-05/lite/<session>/<frame>.exr', ...}`. Verify the EXR landed in R2:

```bash
aws --endpoint-url $R2_ENDPOINT s3 ls s3://ors-captures/cyberpunk-2077/ --recursive
```

## Operational checks

- Live logs: `flyctl logs`
- Live machine status: `flyctl status`
- Restart: `flyctl machine restart` (token store survives via `/data` volume)
- Per-mode contribution stats: `curl https://capture.oss-supersampling.dev/stats`

## Cost ceiling

Free tier: 3 shared-cpu-1x machines × 256 MB + 3 GB persistent storage + 160 GB egress/month, indefinite.

At lite (~500 MB/h × 24h × 30 days × 100 contributors) = ~36 TB ingress/month. **Egress** from Fly.io to R2 is what matters and it's free both ways (R2 zero-egress is the whole point). Server-to-Internet egress is the `/stats` endpoint + healthchecks + multipart 2xx response bodies (kilobytes). Free tier holds for as long as we have <a few thousand contributors.

When we outgrow free tier: bump to `shared-cpu-2x / 512 MB` (~$5/month) and add a second region (`fra` — EU coverage) for ~$10/month total.

## Rollback

```bash
flyctl releases list
flyctl releases rollback <release-id>
```

The volume + token store are preserved across rollbacks (immutable on the volume, not the image).
