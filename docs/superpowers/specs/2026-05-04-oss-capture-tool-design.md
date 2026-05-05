# OSS Capture Tool — Community Training Data Pipeline

**Status:** Design (Sprint S7-adjacent; tandem with Codex)
**Date:** 2026-05-04
**Author:** Cash Conway + Claude (Opus 4.7)
**Parent designs:** [`d3d12-hook-design.md`](../d3d12-hook-design.md), [`notes/2026-05-04-s7-game-integration-design.md`](../notes/2026-05-04-s7-game-integration-design.md)
**Companion R&D:** [`research/2026-05-01-ue5-training-data.md`](../../research/2026-05-01-ue5-training-data.md)

## Goal

Ship a one-click-install community tool that lets users contribute real-game training data to OSS while they play. Data is captured locally, useful samples are auto-uploaded to a central R2 bucket, and the local copy is deleted immediately after a successful upload. Network and disk respectful; opt-in upload; per-game install.

Cash directives:

- **One-click install per-game.** Each supported game gets its own installer. The user picks the game, points the installer at the game's install dir, clicks Install, done.
- **Auto-upload by default.** No manual review step. User has consented at install; upload runs unattended.
- **Network-respectful.** Only capture the data we actually need. Sample sparsely. Skip frames that won't add signal (menus, loading, near-duplicate scenes).
- **Immediately delete local data we don't need.** As soon as a frame is uploaded (or rejected by the sampling policy), it leaves the user's machine.
- **Anti-cheat is Cash's problem, not the tool's.** The supported-games list lives in user-facing docs. The tool itself doesn't try to detect AC and will inject into anything it's pointed at.

## Non-goals

- Per-frame anonymization or PII redaction — captured frames are gameplay G-buffers, not personal data, and the supported-games list will exclude games with chat overlays or competitive UIs.
- Selecting WHICH games to support — that's a Cash editorial decision, not a tool feature.
- Replacing the upscaler at runtime — that's the S7 inference DLL, sibling to but not the same as this capture DLL.

## Architecture

```
┌──────────────────── User's machine ────────────────────┐
│                                                          │
│   Game.exe                                               │
│     │ loads (from game-local DLL search)                 │
│     ▼                                                    │
│   oss_capture.dll  (renamed dxgi.dll proxy)              │
│     │                                                    │
│     ├─ Detours/MinHook on:                               │
│     │    • IDXGISwapChain::Present                       │
│     │    • ID3D12CommandQueue::ExecuteCommandLists       │
│     │    • NVSDK_NGX_D3D12_EvaluateFeature (DLSS path)   │
│     │                                                    │
│     ├─ Sampling policy decides: capture this frame? Y/N  │
│     │    • temporal stride (1 frame / N seconds)         │
│     │    • motion-magnitude bucketing                    │
│     │    • dedup (perceptual hash vs recent captures)    │
│     │                                                    │
│     ├─ If Y: copy LR + HR + G-buffers from RTs to        │
│     │       staging buffer, EXR-encode in a worker       │
│     │       thread, write to %LOCALAPPDATA%\oss-capture\ │
│     │       pending\<game_id>\<session>\<frame>.exr      │
│     │                                                    │
│     └─ Always: pass through to original NGX/DXGI calls,  │
│        zero perceptible game-side latency.               │
│                                                          │
│   oss_capture_uploader.exe (Windows scheduled task)      │
│     │  runs every N minutes                              │
│     ├─ scans pending dir                                 │
│     ├─ uploads to https://capture.oss.../ingest          │
│     ├─ on 200: deletes local file                        │
│     ├─ on 4xx: deletes local file (server rejected;      │
│     │   no point retrying)                               │
│     └─ on 5xx/network: exp-backoff retry up to N times,  │
│        then drop                                         │
│                                                          │
└──────────────────────────────────────────────────────────┘

                       │ HTTPS POST + auth token
                       ▼

┌──────────────── OSS capture-ingest server ───────────────┐
│                                                            │
│   FastAPI service (capture.oss-supersampling.dev)          │
│     │                                                      │
│     ├─ /ingest                                             │
│     │   • bearer-token auth (per-installer rotating key)   │
│     │   • multipart upload: EXR blob + JSON metadata       │
│     │   • dedup: SHA256 of frame content vs index          │
│     │   • spam filter: rate-limit per-token + per-game     │
│     │   • on accept: write to R2 bucket ors-captures/      │
│     │                                                      │
│     └─ /stats                                              │
│         • per-contributor frame count                      │
│         • global dataset size                              │
│                                                            │
│   R2 bucket: ors-captures                                  │
│     organized as <game_id>/<session_uuid>/<frame_uuid>.exr │
│     companion .json with metadata                          │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

## Capture sampling policy

Network-respect bound: target <500 MB/hour of gameplay per user. With ~3 MB compressed per frame (LR 1080p + HR 4K + 4 G-buffers, EXR + zlib), that's <170 frames/hour.

**Burst-mode capture (revised 2026-05-04 evening):** earlier draft sampled isolated single frames every 20s, but those are useless for temporal training because there's no (t, t+1) pair connected by motion vectors. Capture instead in BURSTS of N consecutive frames per event, stride M seconds between events.

Defaults (configurable per-game via `/session/start`):

| Use case | Burst N | Stride M | Frames/hour | MB/hour @ 3 MB/frame |
|---|---|---|---|---|
| Pixel-temporal (pairs) | 2 | 40s | 180 | 540 |
| **Gaussian-temporal (windows of 5)** | **5** | **80s** | **225** | **675** |
| OSS-FX α-extrapolation (4-frame) | 4 | 60s | 240 | 720 |
| Hybrid default (4-frame burst) | 4 | 80s | 180 | 540 |

A "burst" is N CONSECUTIVE swap-chain frames (no skipping). At 60fps that's N/60 seconds of gameplay = ~33–83 ms motion window. Enough variation between frame-0 and frame-N-1 for the temporal head's prev_hr→out_t+1 warp, the Gaussian transformer's history-attention, and the eventual OSS-FX α-conditioned intermediate-frame supervision.

Sampling rules, in order:

1. **Stride gate.** Default: 1 burst event per 80 seconds of gameplay. Stride starts when the previous burst's last frame committed to disk.
2. **Motion bucket.** Compute mean motion-vector magnitude on the candidate. Reject if it's the dominant bucket already over-represented this session (we want diverse motion, not 100 frames of the player standing still).
3. **Perceptual dedup.** Compute a 64-bit perceptual hash (resized 8×8 grayscale, sign of DCT). Reject if Hamming distance < 5 from any frame captured in the last 5 minutes.
4. **G-buffer sanity.** Reject if depth is degenerate (all zeros / all max), motion vectors are NaN, or RT format unsupported.
5. **First-frame-after-loading-screen guard.** Reject if previous candidate was >30 seconds old (likely a loading transition, frame is uninteresting).

Frames that pass the sampling rules are captured. Frames that fail are NEVER WRITTEN — the deletion guarantee starts at the buffer level: rejected frames are immediately freed without touching disk.

## On-disk capture format

`%LOCALAPPDATA%\oss-capture\pending\<game_id>\<session_uuid>\<frame_uuid>.exr` — multi-channel EXR with named channels:

| Channel | Layer | Notes |
|---|---|---|
| `LR.{R,G,B}` | LR linear color | pre-upscale render target |
| `HR.{R,G,B}` | HR linear color | DLSS / native HR output (the "GT" for our model — labeled clearly that this is DLSS-as-pseudo-GT, not raytraced GT) |
| `Depth.Z` | linear depth | 32-bit float, in world units if available |
| `Motion.{X,Y}` | screen-space MV | NDC offsets per pixel |
| `Normals.{X,Y,Z}` | world-space normal | unit vectors |

Companion `<frame_uuid>.json`:

```json
{
  "schema_version": 1,
  "game_id": "cyberpunk-2077",
  "game_version": "2.13",
  "session_uuid": "...",
  "frame_uuid": "...",
  "captured_at_unix": 1777940000,
  "lr_resolution": [1920, 1080],
  "hr_resolution": [3840, 2160],
  "hr_source": "dlss-quality" | "dlss-balanced" | "native" | "fsr-...",
  "jitter_offset_uv": [0.234, 0.781],
  "motion_mean_magnitude_px": 12.4,
  "perceptual_hash_64": "0x...",
  "user_consent_token": "<opaque, mapped to install instance>",
  "uploader_version": "1.0.0"
}
```

EXR compression: zlib level 5 (fast + good compression for HDR float data). Per-frame target: 1.5–4 MB.

## Local cache lifecycle (delete-immediately guarantee)

1. **Frame captured** → written to `pending/<...>.exr`
2. **Uploader sees it** → POSTs to `/ingest`
3. **Server returns 200** → uploader does `Path.unlink()` immediately. No retention.
4. **Server returns 4xx (rejected)** → uploader logs reason, deletes local file (server thinks it's a duplicate or corrupt; no point keeping).
5. **Network failure / 5xx** → uploader exp-backoff retries up to 5 attempts over ~30 minutes, then deletes the local file (we'd rather lose a frame than fill the user's disk).
6. **Hard cap on `pending/` dir size:** 2 GB. If exceeded (e.g., uploader stuck), oldest files deleted first.

User-visible: `pending/` directory should be at most a few hundred MB at any time. If it persistently grows, something's broken.

## Server-side ingestion

FastAPI service. Endpoints:

```
POST /ingest
  headers: Authorization: Bearer <install-token>
  body:    multipart/form-data
             - frame: <frame_uuid>.exr (binary)
             - meta:  <frame_uuid>.json (JSON string)

  responses:
    200 OK     - frame accepted, written to R2
    400        - malformed metadata
    401        - bad/missing token
    409        - duplicate (perceptual_hash_64 + game_id already in index)
    413        - frame larger than 16 MB (config)
    429        - rate limit
    500        - server error (uploader will retry)

POST /session/start
  body: { game_id, game_version, install_token }
  response: { session_uuid, server_time_unix, suggested_capture_rate }

GET /stats?token=<...>
  response: { frames_uploaded, total_bytes, contributor_rank }
```

Auth model: each installer ships with a unique `install_token` baked into the DLL+uploader pair (not the user, the install — anonymous-but-rate-limitable). Token is rotated yearly; old tokens accepted with a deprecation warning.

R2 bucket layout:

```
ors-captures/
  <game_id>/                 e.g., cyberpunk-2077/
    <YYYY-MM>/
      <session_uuid>/
        <frame_uuid>.exr
        <frame_uuid>.json
    _index.parquet           one row per frame, regenerated daily
```

Daily cron rolls up the day's captures into a parquet index for the training data loader. The training pipeline reads `_index.parquet` and pulls only frames it needs (e.g., motion-bucket-balanced sampling).

## One-click installer

Per-game installer. Built once via a build script that takes:

```
build_installer.py --game cyberpunk-2077 \
                   --game-exe-name Cyberpunk2077.exe \
                   --proxy-dll-name dxgi.dll \
                   --output dist/oss_capture_cyberpunk_v1.0.0.msi
```

The installer:

1. Asks user to point at the game's install dir (with default for Steam paths)
2. Verifies it's the right game (checks for `Cyberpunk2077.exe` in `bin\x64\`)
3. Copies `oss_capture.dll` → `<game_dir>\bin\x64\dxgi.dll` (after backing up any existing `dxgi.dll`)
4. Copies `oss_capture_uploader.exe` → `%LOCALAPPDATA%\oss-capture\`
5. Generates a per-install `install_token` and writes to `%LOCALAPPDATA%\oss-capture\config.json`
6. Registers a Windows scheduled task to run the uploader every 10 minutes
7. Shows a brief consent dialog: "By installing, you agree to upload anonymized gameplay frames from <game> to OSS for AI training. Captures use ~500 MB/hour and are deleted from your machine after upload. Click Install to confirm."
8. On completion, opens https://oss-supersampling.dev/contributor-thanks (page acknowledges the contributor + lists supported games)

Uninstaller restores any backed-up `dxgi.dll`, removes the scheduled task, deletes `%LOCALAPPDATA%\oss-capture\` (including any pending captures).

## Distribution

- Per-game installers signed with our developer cert (need to obtain — Microsoft EV cert ~$300/year, optional for v1 but reduces SmartScreen friction)
- Hosted on https://github.com/cashcon57/open-supersampling/releases under each version tag
- README on the github page lists supported games + per-game install instructions
- An optional contributor leaderboard (game frames contributed) at https://oss-supersampling.dev/contributors — opt-in display name, anonymous by default

## Tandem implementation split

Cash directive: Claude and Codex implement in tandem, cross-review.

**Claude (me) implements:**

- Server-side ingestion API (`server/oss_capture_ingest/`) — FastAPI app, R2 upload, dedup index, rate limiting
- R2 bucket layout + daily index cron (`server/scripts/build_capture_index.py`)
- The build script (`scripts/build_installer.py`) that produces per-game MSIs

**Codex implements:**

- Capture-mode addition to the C++ DLL (`oss/gaussian/interception/oss_capture.cpp`) — sampling policy, EXR encoding, write-to-pending-dir
- The Windows uploader daemon (`oss/gaussian/interception/oss_capture_uploader.cpp` or Python — Codex picks)
- Local cache lifecycle enforcement (delete-after-upload, 2GB cap)

**Cross-review checkpoint:** when both halves are working independently against synthetic test fixtures, exchange PR review (Claude reviews Codex's DLL+uploader; Codex reviews Claude's ingest server) before any user-facing release.

**Test fixtures (shared, written first):**

- A synthetic EXR generator (`tests/capture/test_fixtures.py`) — produces realistic LR/HR/G-buffer test frames with known metadata
- An end-to-end test (`tests/capture/test_e2e.py`) — uploader + server + R2-stub round-trip on synthetic frames

## Open questions

1. **HR source labeling:** when DLSS is the HR source, are we capturing ground truth or DLSS output? It's DLSS output, which is a lossy approximation of the path-traced reference. Training on this means we're ultimately bounded by DLSS quality. Mitigation: capture the LR + game's MV/depth/normals separately so we can re-render reference HR offline if needed (using the path tracer when available). Document the limitation in the dataset card.
2. **Frame timing:** capture happens BEFORE DLSS (we want the LR) AND AFTER DLSS (we want the HR). Hook needs to tap two different pipeline stages. Need to verify the D3D12 fence semantics so the HR readback doesn't stall the game.
3. **R2 bucket access cost:** uploads are free (egress free for R2). Daily index parquet generation costs ~$0.01/GB scanned/day. At 1TB total dataset → $10/day worst-case. Acceptable.
4. **Supported-games list governance:** Cash maintains the list. Adding a game means writing a per-game build config + smoke-testing the hook on it. Should we have a community process for users to nominate games + verify they're not anti-cheat'd? Probably yes, eventually; not in v1.
5. **GDPR / data residency:** users in EU contributing data → R2 has EU regions; should we route EU contributors to an EU bucket? Probably yes for v2; v1 ships single-bucket US.

## Phasing

1. **Phase 1 (now, this hour):** finalize this design memo (you're reading it).
2. **Phase 2 (next 1–2 weeks, post-v5-closeout):**
   - Claude: server-side ingest scaffold + R2 wiring
   - Codex: DLL capture-mode prototype + uploader daemon, Cyberpunk-only
   - Cross-review
3. **Phase 3:** internal dogfood — Cash captures from Cyberpunk, validates end-to-end. Bug-fix.
4. **Phase 4:** signed installer, README, single-game beta release. ~10 trusted contributors.
5. **Phase 5:** add second game (BG3? Spider-Man Remastered?), public release, leaderboard.

## Failure modes to design against

- **Server outage:** uploader retries with backoff, falls back to deleting after N attempts. User's disk doesn't fill.
- **Bad frame:** server returns 400, uploader deletes. No retry storm.
- **DLL crashes the game:** must NEVER crash the game. All hook callbacks wrapped in try/catch (or `__try`/`__except` in C++); on any exception, pass through to original API and log the error. Game-stability rule: capture is a side-effect; the game's render path is sacred.
- **Upload service compromised:** install-token is opaque, doesn't grant access to anything besides the upload endpoint. Worst case: a malicious actor uploads garbage frames; server-side spam filter + size cap + perceptual-hash dedup catch it.
- **User rage-uninstalls mid-upload:** uploader is a separate process; uninstaller blocks until it exits cleanly, then removes the cache.
