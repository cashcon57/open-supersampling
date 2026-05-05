# 2026-05-04 — Claude→Codex asks, round 8: lite / regular / INSANE capture modes

C18 + C19 + C20 + C21 + C22 in. Cash directive (R8): **three capture modes** so 99% of users run lightweight (lite) but data warriors with high-end GPUs + uncapped fiber can opt into INSANE for the data that lets OSS exceed DLSS quality.

Spec patch landed at the design memo's new "Capture modes (lite / regular / INSANE)" section (commit pending in this round's first commit). Read that for the strategic rationale.

## C23 — Mode plumbing in DLL + uploader + installer

Severity: medium-high (foundational; informs everything downstream)

### Mode summary

| Mode | Bandwidth/h | Short bursts | Long bursts | Channels | Special |
|---|---|---|---|---|---|
| **lite** (default) | ~450 MB | N=2 / 80s | N=60 / 30 min | LR+HR+depth+motion+normals | none |
| **regular** | ~2 GB | N=4 / 40s | N=60 / 10 min | + albedo + roughness | none |
| **INSANE** | ~20–50 GB | N=8 / 20s | N=240 (4s @ 60fps) / 5 min | + albedo + roughness + metallic + emissive | supersample-GT, FP32 depth/motion, optional DLAA, every-DLSS-mode pairing |

### Required changes

#### 1. `oss/gaussian/interception/oss_capture.h`

Add `OssCaptureMode` enum + replace fixed config defaults with mode-driven preset selection:

```c
typedef enum OssCaptureMode {
    OSS_CAPTURE_MODE_LITE    = 0,
    OSS_CAPTURE_MODE_REGULAR = 1,
    OSS_CAPTURE_MODE_INSANE  = 2,
} OssCaptureMode;

typedef struct OssCaptureConfig {
    OssCaptureMode mode;            // default OSS_CAPTURE_MODE_LITE
    // ... existing burst fields, populated from mode preset on init ...
    // INSANE-only fields (zero-initialised in lite/regular):
    int    capture_albedo;          // 1 if mode >= regular
    int    capture_roughness;       // 1 if mode >= regular
    int    capture_metallic;        // 1 if mode == INSANE
    int    capture_emissive;        // 1 if mode == INSANE
    int    fp32_depth_motion;       // 1 if mode == INSANE
    int    enable_supersample_gt;   // 1 if mode == INSANE
    int    enable_dlaa_capture;     // 1 if mode == INSANE
    int    enable_multi_dlss_mode;  // 1 if mode == INSANE
} OssCaptureConfig;
```

Add a `oss_capture_apply_mode_preset(OssCaptureConfig*, OssCaptureMode)` helper that fills in the burst/period/channel fields from the mode.

#### 2. `oss/gaussian/interception/oss_capture.cpp` — sampler + capture logic

- Sampler tier-decision (already in place from C22) is unaffected; the mode just changes the burst-N + period defaults.
- EXR writer per-mode channel selection: lite/regular/INSANE map to distinct channel-set lists.
- INSANE-mode supersample-GT trigger: detect "stationary camera" (mean motion-vec magnitude < epsilon for ≥1s) → enqueue a 256-frame jittered-LR accumulation event. Encode the accumulated result as a separate `<frame_uuid>__supersample_gt.exr` next to the LR frames. Document the offline-reconstruction step in the dataset card; the server stores the accumulator output as-is.
- INSANE-mode DLAA capture: optional second EvaluateFeature interception that captures the game's DLAA output if user has it enabled. Skip silently if not.
- INSANE-mode every-DLSS-mode pairing: only attempted when the game exposes the live mode-swap path (per-game allowlist). Default off; per-game opt-in.

#### 3. `oss/capture/uploader.py`

No changes needed for the upload path itself. JSON metadata gets a `capture_mode: "lite" | "regular" | "INSANE"` field — server already accepts it (Claude server commit pending in this round). Uploader just passes through.

#### 4. `scripts/build_capture_installer.py` (Claude side)

I'll patch the build script to expose `--mode {lite,regular,INSANE}` and bake the chosen mode into the installer's default config. Per-install mode is also rewritable post-install via the tray-icon menu (out of scope for v1; v1 ships the mode fixed at install time).

#### 5. Tests

- `tests/capture/test_capture_unit.cpp`: `oss_capture_apply_mode_preset` produces the documented bandwidth profile for each mode.
- `tests/capture/test_e2e.py`: round-trip a synthetic INSANE-mode frame with all-channels EXR through uploader → server → R2 without rejection.

### Constraints

- **Mode is set at install time** for v1. Live mode swap (tray-icon menu) is post-v1.
- **Default `lite`** for any installer that ships without an explicit `--mode` flag. The 99% case never has to think about this.
- **INSANE mode performance:** capture-side worker thread budget is ~5 ms/frame at 60fps for the EXR encode + write. INSANE's larger channel set + supersample-GT accumulator may exceed this; document any frame-pacing impact and gate INSANE behind a one-time "your game may stutter" warning at install.
- **Symmetric server-side patch already pending:** the schema accepts `capture_mode` in this round's first server commit.

Final commit message: `capture(dll): lite/regular/INSANE mode presets + per-mode channel selection + INSANE supersample-GT trigger`.

## What I'm doing in parallel

- `server/oss_capture_ingest/schema.py`: add optional `capture_mode` field validation (commit pending).
- Update R2 path layout to include the mode: `<game_id>/<YYYY-MM>/<capture_mode>/<session_uuid>/<...>` so the daily index can stratify by mode without re-reading every file.
- `/stats` endpoint reports per-mode contribution counts.
- Update `scripts/build_capture_installer.py` to accept `--mode`.
