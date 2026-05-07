# 2026-05-04 — Claude cross-review of Codex R6 (capture-tool client side)

Per Cash directive that Claude and Codex cross-review each other. Codex filed 3 findings against Claude's server commit (`2befaeb`), all valid and triaged. This is the symmetric pass: Claude's review of Codex's C18+C19+C20 commits.

**Verdict: 1 MEDIUM finding, 0 HIGH. Codex's client-side work is solid.**

## ✓ Approved with no findings

- **C18 game-stability:** all three hook entry points (`Present`, `ExecuteCommandLists`, `NVSDK_NGX_D3D12_EvaluateFeature`) are SEH-wrapped on MSVC and `try`/`catch`-wrapped on non-MSVC, with `safe_log_exception()` swallowing any escape. Game crash on a capture bug is structurally impossible. Reference: `oss/gaussian/interception/oss_capture.cpp:162-208`.
- **C18 sampling-policy ordering:** matches spec §"Capture sampling policy" rule order (temporal stride, motion bucket, perceptual dedup, G-buffer sanity, post-loading-screen guard). Confirmed via `oss_capture_consider_candidate` + the `g_sampler.Consider()` path.
- **C19 delete-immediately guarantee:** terminal responses (2xx + 4xx) call `delete_frame_pair()` synchronously (`oss/capture/uploader.py:206`); after exhausted retries the frame is also deleted (line 218). Files never persist past their upload disposition.
- **C19 2GB pending-dir cap:** `DEFAULT_MAX_PENDING_BYTES = 2 * 1024 * 1024 * 1024` enforced via `enforce_pending_cap()` called in `drain_once()` (line 229). Oldest-first eviction confirmed in test_uploader.py.
- **C20 metadata schema match:** test fixture metadata matches Claude's `server/oss_capture_ingest/schema.py` field-for-field (schema_version, game_id pattern, UUID4 session/frame fields, lr/hr_resolution arrays, jitter_offset_uv tuple, perceptual_hash_64 hex normalization). Wire-compatible.

## ✗ Open finding (MEDIUM)

### MEDIUM — Capture Uploader Treats 429 As Terminal-Delete Instead Of Backoff-And-Retry

Cross-review of Codex uploader commit `a238feb`.

The design memo's §"Local cache lifecycle (delete-immediately guarantee)" rule 5 specifies 5xx/network → exp-backoff retry, but ALSO explicitly distinguishes 429 from the rest of 4xx: "429 → backoff and retry after `window_seconds`". The current uploader collapses 429 into the 4xx terminal class:

- `oss/capture/uploader.py:189-190`:
  ```python
  if 400 <= status < 500:
      return UploadResult(status, terminal=True, retryable=False)
  ```

So the rate-limited frame is deleted on first 429 and never retried. Under server-side rate-limit pressure (which is also Codex's open MEDIUM finding "Capture Server Rate Limit Does Not Cover Rejected Upload Attempts"), a busy session would silently drop hundreds of frames at the limit boundary.

Fix direction: split out 429 → `terminal=False, retryable=True` with a longer initial backoff (e.g., 60–120s) before joining the standard exp-backoff schedule. Uploader keeps its own per-token request budget so it doesn't actively contribute to the rate-limit it's hitting.

## Triage of Codex's findings against Claude server (`2befaeb`)

All 3 valid. Fix priority order:

1. **HIGH process-local tokens** — fix BEFORE any external dogfood (production-blocker). Cheapest: SQLite-back the `TokenRegistry` (10–20 LOC). Or JSON file at a known path that the registry loads at startup.
2. **MEDIUM rate limit coverage** — fix BEFORE Phase 4 public release. Track auth-valid attempts (any), add per-game limiter once metadata parses.
3. **MEDIUM volatile dedup** — fix BEFORE Phase 4 public release. R2 hash sidecar (`<bucket>/_hash/<sha256[:2]>/<sha256>.txt`) + HEAD-check before write. R2 HEAD is sub-millisecond.

## Phase gate

For internal-dogfood (Phase 3 in the design memo's phasing): Codex's 1 MEDIUM + Claude's 3 findings (1 HIGH + 2 MEDIUM) all need fixing.

For public release (Phase 4): all 4 findings + signed installer + per-game allowlist + the consent dialog need to be done.
