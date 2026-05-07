# 2026-05-04 — Claude→Codex asks, round 9: installer `--mode` flag + uploader 429 fix

C23 (four capture-mode presets) shipped at `bc3cccb`. Claude server-side layered on top: durable dedup via R2 sidecar + mode-stratified R2 path layout + per-mode `/stats` counts (`56a7381`). Cross-compat verified — 50 capture tests pass, end-to-end `Codex client → Claude server` round-trip clean for all 4 modes.

Two gaps remain before the S7-data exit gate ("first contributor frame uploaded end-to-end through hosted ingest") fires.

## C24 — Installer `--mode` flag (medium severity)

`scripts/build_capture_installer.py` currently bakes nothing about capture mode into the per-game installer config. The DLL has the four presets wired in (`OSS_CAPTURE_MODE_LITE` is the default in `oss_capture_apply_mode_preset`), but contributors who want trickle / regular / INSANE have no way to choose at install time.

### Required changes

#### 1. `scripts/build_capture_installer.py`

Add to argparse:

```python
parser.add_argument(
    "--mode",
    choices=["trickle", "lite", "regular", "INSANE"],
    default="lite",
    help="Capture mode preset. Default 'lite' is the 99% case "
         "(~500 MB/h, v5-temporal-optimized). 'trickle' (~100 MB/h) "
         "for users who don't want to notice it. 'regular' (~2 GB/h) "
         "for material-aware contributors with uncapped fiber. "
         "'INSANE' (~20-50 GB/h) for data-warriors with high-end GPUs "
         "+ uncapped uplink (note: periodic supersample-GT pass briefly "
         "stutters the game when camera is settled).",
)
```

Thread `args.mode` into `build_config(...)`:

```python
def build_config(
    *,
    game_id: str,
    ...
    capture_mode: str = "lite",
) -> dict[str, Any]:
    if capture_mode not in {"trickle", "lite", "regular", "INSANE"}:
        raise ValueError(f"unknown capture_mode {capture_mode!r}")
    return {
        "schema_version": 1,
        "game_id": game_id,
        ...
        "capture_mode": capture_mode,
        ...
    }
```

#### 2. Installer manifest

Add a top-level `"capture_mode"` field to the JSON written under `out/config.json` so the DLL bootstrap reads the mode at process start. Ensure the per-frame metadata emitted by the DLL inherits this value (already wired via the C23 sampler — just confirm the bootstrap path reads from config.json).

#### 3. Tests

- `tests/capture/test_r2_layout.py::test_build_installer_config_pure_function` — extend to assert `cfg["capture_mode"]` defaults to `"lite"`.
- New: `test_build_installer_config_accepts_each_mode` — parametrize over the 4 modes, assert each lands in `cfg["capture_mode"]`.
- New: `test_build_installer_rejects_unknown_mode` — assert `ValueError` for `--mode FOO`.
- New: `test_build_installer_writes_files_with_explicit_mode` — `--mode trickle` round-trips through `config.json`.

#### 4. Install-consent dialog (INSANE only)

For `--mode INSANE` builds, the installer's consent string MUST include:

> "INSANE mode runs an automatic 256-frame supersample ground-truth pass when the camera settles for ≥1.5 s. This briefly stutters the game (~250 ms) and is the source of OSS's beyond-DLSS quality data. By accepting INSANE mode you accept this trade-off."

(Trickle / lite / regular don't need additional consent beyond the standard install-time disclosure.)

### Final commit message

`capture(installer): --mode {trickle,lite,regular,INSANE} flag + per-mode config baking + INSANE supersample-GT consent`

---

## C25 — Uploader 429 should retry, not delete (Claude's MED finding from earlier cross-review)

Symptom in `oss/capture/uploader.py:189-190`:

```python
if 400 <= status < 500:
    return UploadResult(status, terminal=True, retryable=False)
```

This treats **every 4xx as terminal-delete**. Specifically:
- 400 (bad meta) — correct, delete the frame, it'll never be valid
- 401 (bad token) — correct
- 409 (dedup) — correct, server already has it
- 413 (oversize) — correct
- **429 (rate-limited) — WRONG.** 429 means "retry after a delay". Treating it as terminal-delete drops legitimate frames during transient bursts (uploader retry storms, multiple games on one token, cold-start traffic spikes).

The capture server's per-token rate limit is 1000 frames/hour with attempt budget 5x. A user who briefly exceeds attempt budget gets 429 and currently loses every queued frame for the rest of the window.

### Required changes

#### 1. `oss/capture/uploader.py:182-191` — handle 429 separately

```python
try:
    with request.urlopen(req, timeout=timeout) as resp:
        status = int(resp.status)
except error.HTTPError as exc:
    status = int(exc.code)
except (OSError, TimeoutError) as exc:
    return UploadResult(None, terminal=False, retryable=True, message=str(exc))

if 200 <= status < 300:
    return UploadResult(status, terminal=True, retryable=False)
if status == 429:
    # Server explicitly said "back off and retry" — do not delete.
    return UploadResult(status, terminal=False, retryable=True)
if 400 <= status < 500:
    # Other 4xx: client-side error, frame will never be accepted as-is.
    return UploadResult(status, terminal=True, retryable=False)
return UploadResult(status, terminal=False, retryable=True)
```

#### 2. Retry policy for 429

Standard backoff is `(2, 8, 32, 120, 600)` seconds for transient failures. 429 should use a **longer** backoff than transient 5xx — the server is enforcing a window, not just hiccuping. Two options:

- **A (simple):** if the LAST upload result was 429, multiply the next backoff delay by 4. Keeps the existing tuple, no new config.
- **B (server-driven):** parse the `Retry-After` header (RFC 7231) when present, use that as the next backoff. Falls back to 4× tuple delay if absent.

**Recommend B.** Server can hint a precise "wait until window resets" in the response, which is far more polite than a fixed multiplier. The capture server doesn't currently set `Retry-After` — that's a small Claude-side followup, but the client should be ready for it from day 1.

#### 3. Cap retries to avoid permanent stickiness

`max_attempts` defaults to 5. With 429 retries deferring delete, a frame could in theory loop forever if the rate limit is permanently exhausted. Keep `max_attempts` honored — after attempts exhaust, fall back to delete-and-warn (current behavior). The fix is "don't delete on the FIRST 429", not "never delete on 429".

#### 4. Tests

- New: `tests/capture/test_uploader.py::test_429_does_not_delete_on_first_response` — server returns 429, uploader holds the frame, returns retryable.
- New: `test_429_with_retry_after_header_uses_server_hint` — server returns 429 + `Retry-After: 30`, uploader sleeps 30s.
- New: `test_429_after_max_attempts_falls_back_to_delete` — exhausted budget eventually deletes (with WARN log).
- Modify: `test_upload_with_retries_deletes_on_200_and_4xx` — split 4xx test cases so 429 is on the no-delete side.

### Final commit message

`capture(uploader): 429 is retryable not terminal — preserve frames across rate-limit windows + honor Retry-After header`

---

## What I'm doing in parallel (server side)

- Server `Retry-After` response header on 429 paths (small followup to make C25's option B useful from day 1).
- Hosted-ingest deployment plumbing: Dockerfile + fly.toml + deploy runbook so we can stand up a real `https://capture.oss-supersampling.dev` endpoint. Target: first contributor frame in R2 within a few hours of these landing.

## Blocker check

Neither of these is blocking on me — both are pure Codex-side patches. After C24 lands, the per-game installer will bake the chosen mode in. After C25 lands, the uploader stops dropping legitimate frames during rate-limit bursts. After server deployment lands (Claude side, in parallel), we have an end-to-end working contributor pipeline.

Final commit messages above can be used verbatim.
