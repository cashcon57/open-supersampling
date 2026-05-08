# OSS Capture Tool — Next Phases

**Date:** 2026-05-08
**Parent spec:** `docs/superpowers/specs/2026-05-04-oss-capture-tool-design.md`
**Prior handoff:** `docs/archive/coordination/codex-handoff-2026-05-06-capture.md`

## A. State audit

### IN-REPO and working

- Server-side ingest (FastAPI, mature): `server/oss_capture_ingest/{main, auth, dedup, r2, schema, routes/}` with token registry, R2 layout, durable dedup, multipart `/ingest`, `/session/start`, `/stats`.
- Uploader daemon (Python, mature): `oss/capture/uploader.py` + pyinstaller spec + Windows scheduled task XML.
- Tray + per-session config (Python, mature): `oss/capture/tray/` with Steam-library scan, allowlist, process watcher, DLL-inject path.
- Installer scaffolding: `scripts/build_capture_installer.py` produces `config.json` + `installer_manifest.json`. Does NOT compile MSI (separate WiX step).
- Capture-policy logic (C++, mature, NOT wired to D3D12): `oss/gaussian/interception/oss_capture.cpp` has full `CaptureSampler` (mode/tier/stride gates, motion-bucket, perceptual-hash dedup, G-buffer sanity).
- EXR writer + pHash: `oss/gaussian/interception/exr_writer.cpp`, `perceptual_hash.cpp`.
- DXGI proxy stub: `oss/gaussian/interception/src/dxgi_proxy.cpp`.
- Tests (76 passing, 1 skipped): `tests/capture/test_{e2e, fixtures, uploader, ingest_server, r2_layout, build_capture_index, tray_config, tray_launch_flow}.py` + `test_capture_unit.cpp`.

### SPEC (gap)

- **Actual D3D12 capture path** — `oss/gaussian/interception/src/dllmain.cpp` has `TODO(T2.8)` for IDXGIFactory hook + `TODO(T2.x)` for Detours-detach. MinHook/Detours not yet vendored.
- `on_present_impl`, `on_execute_command_lists_impl`, `on_ngx_evaluate_feature_impl` are logging-only stubs — never read backbuffer texture data.
- No GPU→CPU staging copy. No NVSDK_NGX `EvaluateFeature` parameter unpack. No worker-thread EXR-write pool.
- No DLL-side JSON config reader.
- No MSI compile pipeline (WiX/candle unbuilt).
- No code-signing pipeline (~$300/yr EV cert, optional v1).
- No supported-games README beyond the spec.
- No contributor leaderboard panel on the public dashboard.
- No deploy manifest for the FastAPI service.

## B. Critical-path next phase — the C++ DLL hook

Single biggest gap. Without it, every other piece has no real frames flowing through.

Scope:
1. Hook `IDXGIFactory::CreateSwapChain*` → vtable-patch `IDXGISwapChain::Present`.
2. Hook `ID3D12CommandQueue::ExecuteCommandLists` for queue retention.
3. Hook `NVSDK_NGX_D3D12_EvaluateFeature` → unpack DLSS params (LR/HR/depth/motion/normals).
4. Per-frame `CaptureSampler::Consider()` (already implemented).
5. GPU→CPU staging copy via `D3D12_HEAP_TYPE_READBACK` + fence + worker-thread EXR write.
6. DLL-side JSON config reader (nlohmann/json).
7. `__try`/`__except` wrap every hook callback.

Estimate: 2-3 weeks focused C++/Win32 work. Out of autonomous scope — needs Windows host + D3D12 game (Cyberpunk 2077 per spec).

## C. Smaller batches achievable autonomously

### C.1 Operator test harness (~0.5 day)

`scripts/capture_simulator.py` — wraps `tests/capture/test_fixtures.py::make_synthetic_capture`, drives `oss/capture/uploader.py::drain_once` against running ingest server. Validates server-side stack without DLL.

### C.2 Fly.io ingest deployment artifact (~1 day)

Decision matrix recommended **Fly.io** for v1 (preserves FastAPI+boto3+R2 stack, 3-VM free tier covers ~50 req/s peak). New: `deploy/fly/{fly.toml, Dockerfile}` + `deploy/README.md` runbook.

### C.3 Contributor leaderboard panel (~1 day)

New `GET /leaderboard` route (`server/oss_capture_ingest/routes/leaderboard.py`) returning SHA256-prefix-anonymized top-N counts. New panel in `dashboard-public/index.html`. Privacy default: hash prefix only, no display names in v1.

### C.4 Per-game install README (~0.5 day)

`docs/capture/INSTALL.md` (user runbook) + `docs/capture/SUPPORTED_GAMES.md` (Cash's editorial supported-games list with anti-cheat disclaimers).

### C.5 Test fixture extension (~0.5 day)

Extend `tests/capture/test_e2e.py` for 401, 409, 413, 429, network timeout edge cases. Verify uploader's 429 (rate-limit) handling: backoff, do NOT delete, retry on next pass.

### C.6 Daily R2 index cron (~0.5 day)

Audit/create `server/scripts/build_capture_index.py` for the daily `_index.parquet` roll-up (test exists at `tests/capture/test_build_capture_index.py`).

## D. Phased rollout

- **C0 (now, autonomous):** finish C.1-C.6. ~5-6 days serial, faster in parallel.
- **C1 (~2 weeks, needs Windows + game):** C++ DLL hook prototype on Cyberpunk 2077.
- **C2 (~2 weeks):** DLL ↔ ingest round-trip + internal dogfood.
- **C3 (~1 week):** signed installer, public beta, 10 trusted contributors.
- **C4 (ongoing):** add games, leaderboard, scaling.

## E. Open questions for operator

1. Dev ingest hosting: Fly.io (recommended v1) vs Cloudflare Workers (better long-term, requires Python rewrite)?
2. Anti-cheat: docs disclaimer only (recommended v1) vs runtime AC-detection bypass attempt?
3. Leaderboard privacy: hash prefix only (recommended v1) vs optional self-chosen handle?
4. Default sampling: `lite` mode + 80s burst stride (per spec) — confirm?
5. MSI compile pipeline: GitHub Actions (long-term) vs Cash's Windows box (interim)?

## Critical files

- `oss/gaussian/interception/oss_capture.cpp` — C++ entry for D3D12 wiring
- `oss/gaussian/interception/src/dllmain.cpp` — TODO T2.8/T2.x for hook installation
- `server/oss_capture_ingest/main.py` — FastAPI factory
- `oss/capture/uploader.py` — needs C.5 fix for 429 handling
- `scripts/build_capture_installer.py` — canonical config schema
