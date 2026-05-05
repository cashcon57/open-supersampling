# 2026-05-04 — Claude→Codex asks, round 6: OSS Capture Tool tandem

Per Cash directive, this round we work on the OSS Capture Tool **in tandem**: Claude implements the server side, Codex implements the client side, then we cross-review.

**Read first:** `docs/superpowers/specs/2026-05-04-oss-capture-tool-design.md` (commit `af66a78`). Full architecture, sampling policy, on-disk format, R2 layout, install/uninstall semantics, failure modes.

Cash directives:
- One-click install per game; auto-upload (no per-frame review)
- Network-respectful: <500 MB/hour cap via temporal-stride + motion-bucket + perceptual-dedup
- Delete-immediately-after-upload with 2 GB hard cap on pending dir
- Anti-cheat is Cash's editorial decision; tool itself injects wherever pointed

## C18 — DLL capture mode (`oss/gaussian/interception/`)

Severity: high (core deliverable)

The existing `d3d12-hook-design.md` covers the inference-mode hook (replace DLSS). Capture-mode is the same DLL with a different code path. Build on the same Detours/MinHook + NGX-spoofing pattern.

**Files to create / extend:**
- `oss/gaussian/interception/oss_capture.h` — public configuration (capture rate, output dir, install_token)
- `oss/gaussian/interception/oss_capture.cpp` — DXGI proxy DLL with capture mode
  - Detours-hook on `IDXGISwapChain::Present` for HR backbuffer
  - Detours-hook on `NVSDK_NGX_D3D12_EvaluateFeature` for LR + game-supplied G-buffers (depth, motion, normals)
  - Sampling decision per spec §"Capture sampling policy" (5 rules in order: temporal stride, motion bucket, perceptual dedup, G-buffer sanity, post-loading-screen guard)
  - On accept: copy RT textures → CPU staging buffer → enqueue an async EXR write on a worker thread
  - On reject: free buffers immediately, NEVER touch disk
  - All hook callbacks wrapped in `__try`/`__except` so the game never crashes from a capture bug
- `oss/gaussian/interception/exr_writer.cpp` — multi-channel EXR write (LR.RGB, HR.RGB, Depth.Z, Motion.XY, Normals.XYZ) using OpenEXR's IlmImf C++ API or tinyexr (header-only, simpler). Compression zlib level 5.
- `oss/gaussian/interception/perceptual_hash.cpp` — 64-bit pHash for dedup (8x8 grayscale + DCT-sign). Hamming-distance window of 5 minutes recent captures.

**Build target:** Windows DLL renamed to `dxgi.dll` for game-local DLL search override. Single MSVC `.sln` (or CMake) at `oss/gaussian/interception/CMakeLists.txt`. Static link the C++ runtime so the DLL is self-contained.

**Tests** at `tests/capture/test_capture_unit.cpp` (Google Test or Catch2): sampling-policy unit tests against synthetic frame metadata; EXR roundtrip test; pHash determinism + sensitivity tests.

Final commit message: `capture(dll): D3D12 capture-mode hook + EXR writer + perceptual hash`.

## C19 — Uploader daemon

Severity: high

A standalone process that drains `%LOCALAPPDATA%\oss-capture\pending\` to the ingest server. Per Cash directive: language is your call (C++, Rust, or Python with pyinstaller). Recommendation: **Python** because it's already in our toolchain, the binary is small via pyinstaller, and HTTP+retry+exponential-backoff is one-import-away.

**Files to create:**
- `oss/capture/uploader.py` — main loop:
  - Scan `%LOCALAPPDATA%\oss-capture\pending\<game_id>\<session>\*.exr` every 60s
  - For each frame: POST multipart to `https://capture.oss.../ingest` with bearer token from `%LOCALAPPDATA%\oss-capture\config.json`
  - On 200/4xx (terminal): delete file
  - On 5xx/network: exponential backoff (1s, 5s, 30s, 2m, 10m) then drop
  - Enforce 2 GB cap on `pending/`: delete oldest first if exceeded
- `oss/capture/installer/oss_capture_uploader_pyinstaller.spec` — pyinstaller config to produce a single .exe
- `oss/capture/installer/scheduled_task.xml` — Windows Task Scheduler XML (10-min recurring trigger) registered by the MSI
- `tests/capture/test_uploader.py` — unit tests with `requests-mock`: 200 → delete, 400 → delete, 500 → backoff retry, 2GB cap → delete oldest

Final commit message: `capture(uploader): Python uploader daemon with retry + 2GB cap + delete-after-upload`.

## C20 — Test fixtures (shared, written FIRST)

Severity: medium (blocks both C18 + C19 verification)

Both halves need a shared synthetic frame generator + e2e test harness BEFORE the cross-review.

**Files to create:**
- `tests/capture/test_fixtures.py` — generate synthetic LR/HR/depth/motion/normals tensors with known content, write to EXR matching the spec's channel layout. Helper to construct a synthetic `<frame_uuid>.json` matching the schema.
- `tests/capture/test_e2e.py` — pytest integration:
  - 1) generate 5 synthetic EXR frames into a tmp pending dir
  - 2) start a fake `/ingest` server (FastAPI test client or aiohttp test fixture) that returns 200 / 400 / 500 in scripted patterns
  - 3) run the uploader against the tmp dir
  - 4) assert: 200-frames are deleted, 400-frames are deleted, 500-frames are retried then deleted after exhaust, 2GB cap evicts oldest

Final commit message: `tests(capture): shared synthetic frame fixtures + e2e uploader/server roundtrip`.

## Cross-review checkpoint

When C18 + C19 land independently:

- **Claude reviews Codex's** DLL capture mode against the design's "Failure modes to design against" — especially the no-game-crash requirement and the delete-immediately guarantee.
- **Codex reviews Claude's** ingest server against the design's auth + dedup + rate-limit + R2-layout sections, plus the schema match between `tests/capture/test_fixtures.py` and the server's expected `<frame_uuid>.json`.
- File any cross-review findings under `## Open Findings` in `docs/superpowers/notes/2026-05-04-v5-rolling-review.md` with severity, exactly like prior rounds.

No user-facing release until both sides have passed cross-review.

## Out of scope for this round

- Per-game build config (separate task once Cyberpunk works)
- Signed installer (need to obtain EV cert; v1 ships unsigned with a SmartScreen warning)
- Contributor leaderboard UI (server endpoint stub OK; full UI is post-v1)
- DXGI hook on Vulkan games (Vulkan layer is a separate DLL, separate task)

## Live training context

The pixel training run is active on `<train-host>` (PID 2732, ~step 1000 of 80000). Capture-tool work doesn't conflict — it's pure code, no GPU, no compute. The remote 3080ti would be the testbed for the DLL once Cyberpunk is installed there; Cash has the box.
