# 2026-05-04 — Claude→Codex asks, round 8: trickle / lite / regular / INSANE capture modes

C18 + C19 + C20 + C21 + C22 in. Cash directive (R8): **four capture modes** so 99% of users run lightweight (lite), data warriors with high-end GPUs + uncapped fiber can opt into INSANE for the data that lets OSS exceed DLSS, AND a "basically nothing" trickle mode catches static-frame data at near-zero burden for users who don't want to commit bandwidth.

Spec patched at the design memo's "Capture modes (trickle / lite / regular / INSANE)" section. Read that for the per-tier strategic rationale + optimization details.

## Mode summary (each is purpose-built for its budget, NOT just "more bytes = more of everything")

| Mode | Bandwidth/h | Optimized for | Capture strategy |
|---|---|---|---|
| **trickle** | ~50 MB | Single-frame SR + scene diversity | Static-camera-only single frames (LR + HR ONLY, no G-buffers). Min period 120s. Trigger: motion < 0.5 px for ≥1.5s. ~30 frames/h. |
| **lite** (default) | ~500 MB | v5 temporal SR | Short pairs (N=2/80s) + long sequences (N=60/30 min, no HR) + opportunistic trickle frames (~10% of budget). |
| **regular** | ~2 GB | Material-aware temporal SR | + albedo + roughness, denser bursts (N=4/40s, N=60/10 min), boost on mixed-material scenes. |
| **INSANE** | ~20–50 GB | Beyond-DLSS quality | Full BRDF + 4-second long bursts (N=240/5 min) + supersample-GT auto-trigger + FP32 depth/motion + DLAA + every-DLSS-mode pairing + scene-cut burst. |

## C23 — Mode plumbing in DLL + uploader + installer

Severity: medium-high (foundational; informs everything downstream)

### Required changes

#### 1. `oss/gaussian/interception/oss_capture.h` — enum + config

```c
typedef enum OssCaptureMode {
    OSS_CAPTURE_MODE_TRICKLE = 0,   // ~50 MB/h, single-frame static-camera only
    OSS_CAPTURE_MODE_LITE    = 1,   // ~500 MB/h, default, v5 temporal SR
    OSS_CAPTURE_MODE_REGULAR = 2,   // ~2 GB/h, material-aware
    OSS_CAPTURE_MODE_INSANE  = 3,   // ~20-50 GB/h, beyond-DLSS data
} OssCaptureMode;

typedef struct OssCaptureConfig {
    OssCaptureMode mode;            // default OSS_CAPTURE_MODE_LITE
    // Existing burst fields populated from mode preset on init via
    // oss_capture_apply_mode_preset(). Trickle sets burst_n=1 and uses
    // a static-camera trigger instead of a stride.
    int    burst_n;
    double stride_seconds;
    int    long_burst_n;
    double long_stride_seconds;
    // Channel selection (mode-driven):
    int    capture_lr;              // always 1
    int    capture_hr;              // 1 in trickle/lite/regular; 1 in INSANE except on long-burst frames
    int    capture_depth;           // 0 in trickle; 1 in lite/regular/INSANE
    int    capture_motion;          // 0 in trickle; 1 in lite/regular/INSANE
    int    capture_normals;         // 0 in trickle; 1 in lite/regular/INSANE
    int    capture_albedo;          // 0 in trickle/lite; 1 in regular/INSANE
    int    capture_roughness;       // 0 in trickle/lite; 1 in regular/INSANE
    int    capture_metallic;        // 0 in trickle/lite/regular; 1 in INSANE
    int    capture_emissive;        // 0 in trickle/lite/regular; 1 in INSANE
    // INSANE-only:
    int    fp32_depth_motion;       // 1 in INSANE
    int    enable_supersample_gt;   // 1 in INSANE
    int    enable_dlaa_capture;     // 1 in INSANE
    int    enable_multi_dlss_mode;  // 1 in INSANE (per-game opt-in still required)
    // trickle + lite:
    int    enable_static_frame_trigger;  // 1 in trickle (only mode), AND 1 in lite (opportunistic)
    double static_motion_threshold_px;   // default 0.5
    double static_dwell_seconds;         // default 1.5
    int    static_min_period_seconds;    // 120 in trickle, 600 in lite
} OssCaptureConfig;
```

Add a `oss_capture_apply_mode_preset(OssCaptureConfig*, OssCaptureMode)` helper that fills in all fields from the mode.

#### 2. `oss/gaussian/interception/oss_capture.cpp` — sampler + capture logic

- **Trickle mode** uses a NEW sampler path: only the static-frame trigger fires; no stride-based bursts. EXR writer drops everything but LR + HR.
- **Lite mode** retains the C22 short+long burst sampler AND adds the static-frame trigger as a third tier (with a 600s min period to keep its budget share to ~10%).
- **Regular mode** is lite + the regular preset's denser bursts + the new channels (albedo, roughness).
- **INSANE mode** is regular + metallic + emissive + the new INSANE-only triggers (supersample-GT, DLAA, multi-DLSS, scene-cut burst).

Per-frame metadata gets:
- `capture_mode`: "trickle" | "lite" | "regular" | "INSANE"
- For trickle frames: `burst_uuid` and `burst_index` and `burst_tier` are ALL absent (single-frame).
- For lite/regular/INSANE non-burst opportunistic static frames: `burst_uuid`/`burst_index`/`burst_tier` also absent — they're treated as trickle-equivalent samples.

#### 3. EXR writer per-mode channel set

The writer enumerates which channels to emit based on the `capture_*` flags in config. For trickle, the EXR has only `LR.{R,G,B}` + `HR.{R,G,B}` and is correspondingly tiny (~1.5 MB per frame after zlib).

#### 4. `oss/capture/uploader.py`

No changes. JSON metadata gets `capture_mode` field — server already accepts all 4 modes (Claude server commit `1714c16` accepts trickle/lite/regular/INSANE).

#### 5. `scripts/build_capture_installer.py` (Claude side)

I'll patch the build script to expose `--mode {trickle,lite,regular,INSANE}` and bake the chosen mode into the installer's default config. Per-install mode is rewritable post-install via the tray-icon menu (out of scope for v1).

#### 6. Tests

- `tests/capture/test_capture_unit.cpp`:
  - `oss_capture_apply_mode_preset` produces the documented bandwidth profile + channel set for each of the 4 modes.
  - Trickle sampler fires only on static-camera trigger, not on stride.
  - Lite sampler fires both on stride (bursts) AND on static-camera trigger (opportunistic).
- `tests/capture/test_e2e.py`: round-trip a synthetic frame from EACH mode (trickle single-frame with no G-buffers; INSANE full-BRDF) through uploader → server → R2 without rejection.

### Constraints

- **Mode is set at install time** for v1. Live mode swap (tray-icon menu) is post-v1.
- **Default `lite`** for any installer that ships without an explicit `--mode` flag. The 99% case never has to think about this.
- **Trickle drops G-buffers entirely.** Purity argument was considered: depth could be useful for future post-hoc training, but the whole point of trickle is "minimum useful". v3/v4 trained without G-buffers and hit 30 dB on this exact data shape — proven sufficient for single-frame SR.
- **Lite includes opportunistic trickle frames** (~10% of bandwidth budget). Adds maybe 50 MB/h to lite's total — bumps to ~500 MB/h. Worth it: every lite contributor automatically gives us single-frame coverage too.
- **INSANE supersample-GT** can stutter the game during the 256-frame accumulator pass. Document in the install consent dialog.
- **Symmetric server-side schema patch already pending:** server commit `1714c16` accepts the full set of 4 modes and enforces "trickle frames must omit burst_*" cross-validation.

Final commit message: `capture(dll): trickle/lite/regular/INSANE mode presets + per-mode channel selection + INSANE supersample-GT trigger + lite opportunistic trickle frames`.

## What I'm doing in parallel

- `server/oss_capture_ingest/schema.py`: trickle accepted + cross-validation that trickle frames omit burst fields (commit pending in this round).
- R2 path layout to include the mode: `<game_id>/<YYYY-MM>/<capture_mode>/<session_uuid>/<...>` so the daily index can stratify by mode without re-reading every file.
- `/stats` endpoint reports per-mode contribution counts.
- Update `scripts/build_capture_installer.py` to accept `--mode {trickle,lite,regular,INSANE}`.
