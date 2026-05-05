# 2026-05-04 — Claude→Codex asks, round 8: trickle / lite / regular / INSANE capture modes

C18 + C19 + C20 + C21 + C22 in. Cash directive (R8): **four capture modes** so 99% of users run lightweight (lite), data warriors with high-end GPUs + uncapped fiber can opt into INSANE for the data that lets OSS exceed DLSS, AND a "trickle" mode extracts the maximum-density data that's invisible to a user (~100 MB/h) for users who don't want to commit bandwidth.

Spec patched at the design memo's "Capture modes (trickle / lite / regular / INSANE)" section. Read that for the per-tier strategic rationale + optimization details.

## Mode summary (each is purpose-built for its budget, NOT just "more bytes = more of everything")

| Mode | Bandwidth/h | Optimized for | Capture strategy |
|---|---|---|---|
| **trickle** | ~100 MB | Single-frame SR + temporal pairs at MAX density that's invisible to user | Static singles (LR + HR + depth + motion + normals, every 5 min) + opportunistic motion pairs (HR on `t` only; `t+1` is LR + G-buffers, every 20 min). Trigger: motion < 0.5 px for ≥1.5 s. |
| **lite** (default) | ~500 MB | v5 temporal SR | Short pairs (N=2/80s) + long sequences (N=60/30 min, no HR) + opportunistic trickle frames (~10% of budget). |
| **regular** | ~2 GB | Material-aware temporal SR | + albedo + roughness, denser bursts (N=4/40s, N=60/10 min), boost on mixed-material scenes. |
| **INSANE** | ~20–50 GB | Beyond-DLSS quality | Full BRDF + 4-second long bursts (N=240/5 min) + supersample-GT auto-trigger + FP32 depth/motion + DLAA + every-DLSS-mode pairing + scene-cut burst. |

**Critical revision (fixes earlier draft):** trickle keeps ALL G-buffers (depth + motion + normals) because they compress to ~1.75 MB combined and are cheap. The earlier draft dropped them — wrong call. The expensive bit per frame is HR (~3 MB), so trickle drops HR strategically (only on the `t+1` of opportunistic pairs) to fit BOTH single-frame and temporal-pair training signal in the same invisible-burden budget.

## C23 — Mode plumbing in DLL + uploader + installer

Severity: medium-high (foundational; informs everything downstream)

### Required changes

#### 1. `oss/gaussian/interception/oss_capture.h` — enum + config

```c
typedef enum OssCaptureMode {
    OSS_CAPTURE_MODE_TRICKLE = 0,   // ~100 MB/h, static singles + occasional pairs
    OSS_CAPTURE_MODE_LITE    = 1,   // ~500 MB/h, default, v5 temporal SR
    OSS_CAPTURE_MODE_REGULAR = 2,   // ~2 GB/h, material-aware
    OSS_CAPTURE_MODE_INSANE  = 3,   // ~20-50 GB/h, beyond-DLSS data
} OssCaptureMode;

typedef struct OssCaptureConfig {
    OssCaptureMode mode;            // default OSS_CAPTURE_MODE_LITE
    // Existing burst fields populated from mode preset on init via
    // oss_capture_apply_mode_preset().
    int    burst_n;
    double stride_seconds;
    int    long_burst_n;
    double long_stride_seconds;
    // Channel selection (mode-driven):
    int    capture_lr;              // always 1
    int    capture_hr_on_t0;        // 1 in trickle/lite/regular; 1 in INSANE except on long-burst frames
    int    capture_hr_on_tplus;     // 0 in trickle (saves bytes on pair t+1); 1 in lite/regular/INSANE
    int    capture_depth;           // 1 in trickle/lite/regular/INSANE (G-buffers ARE in trickle now)
    int    capture_motion;          // 1 in trickle/lite/regular/INSANE
    int    capture_normals;         // 1 in trickle/lite/regular/INSANE
    int    capture_albedo;          // 0 in trickle/lite; 1 in regular/INSANE
    int    capture_roughness;       // 0 in trickle/lite; 1 in regular/INSANE
    int    capture_metallic;        // 0 in trickle/lite/regular; 1 in INSANE
    int    capture_emissive;        // 0 in trickle/lite/regular; 1 in INSANE
    // INSANE-only:
    int    fp32_depth_motion;       // 1 in INSANE
    int    enable_supersample_gt;   // 1 in INSANE
    int    enable_dlaa_capture;     // 1 in INSANE
    int    enable_multi_dlss_mode;  // 1 in INSANE (per-game opt-in still required)
    // trickle + lite static-frame trigger:
    int    enable_static_frame_trigger;  // 1 in trickle, 1 in lite
    double static_motion_threshold_px;   // default 0.5
    double static_dwell_seconds;         // default 1.5
    int    static_min_period_seconds;    // 300 in trickle, 600 in lite
    // trickle-only opportunistic-pair trigger:
    int    enable_opportunistic_pair;        // 1 in trickle ONLY
    double opportunistic_pair_motion_window_s;  // default 5.0 (player must start moving within this window after a static)
    int    opportunistic_pair_min_period_s;     // default 1200
} OssCaptureConfig;
```

Add `oss_capture_apply_mode_preset(OssCaptureConfig*, OssCaptureMode)` to fill all fields from the mode.

#### 2. `oss/gaussian/interception/oss_capture.cpp` — sampler + capture logic

- **Trickle mode** uses TWO sampler paths:
  - **Static singles:** static-frame trigger fires (motion < 0.5 px for ≥1.5 s). Single-frame capture with all G-buffers + HR. Min period 300 s.
  - **Opportunistic pairs:** when a static-frame candidate is followed by motion within 5 s (player just started moving from a settled position), capture frames `t` (full G-buffers + HR) and `t+1` (full G-buffers, NO HR). Min period 1200 s. Tag with shared `burst_uuid`, `burst_index ∈ {0,1}`, `burst_tier = "short"`.
- **Lite mode** retains the C22 short+long burst sampler AND adds the static-frame trigger as a third tier (with a 600 s min period to keep its budget share to ~10%). NO opportunistic-pair trigger in lite (lite already has dedicated short pairs every 80 s).
- **Regular mode** is lite + denser bursts + albedo + roughness.
- **INSANE mode** is regular + metallic + emissive + INSANE-only triggers (supersample-GT, DLAA, multi-DLSS, scene-cut burst).

Per-frame metadata gets:

- `capture_mode`: "trickle" | "lite" | "regular" | "INSANE"
- For trickle static singles: `burst_uuid` / `burst_index` / `burst_tier` are ALL absent
- For trickle opportunistic pairs: `burst_uuid` set, `burst_index ∈ {0,1}`, `burst_tier = "short"`
- For lite/regular/INSANE bursts: as per C21 + C22
- For lite/regular/INSANE opportunistic static frames: `burst_*` all absent (treated as trickle-equivalent samples)

#### 3. EXR writer per-mode channel set

The writer enumerates channels based on `capture_*` flags. Trickle's `t+1` of an opportunistic pair has HR dropped (per `capture_hr_on_tplus = 0` in trickle); all G-buffers retained. Frame size: ~5.75 MB for static single, ~8.5 MB total for pair (5.75 on `t` + 2.75 on `t+1`).

#### 4. `oss/capture/uploader.py`

No changes. JSON metadata gets `capture_mode` field — server already accepts all 4 modes (Claude server commit pending in this round).

#### 5. `scripts/build_capture_installer.py` (Claude side)

I'll patch the build script to expose `--mode {trickle,lite,regular,INSANE}` and bake the chosen mode into the installer's default config. Per-install mode is rewritable post-install via the tray-icon menu (out of scope for v1).

#### 6. Tests

- `tests/capture/test_capture_unit.cpp`:
  - `oss_capture_apply_mode_preset` produces the documented bandwidth profile + channel set for each of the 4 modes.
  - Trickle static-single trigger fires only when the player is settled AND min period has elapsed.
  - Trickle opportunistic-pair trigger fires only on a static→motion transition within the 5 s window AND min period.
  - Lite sampler fires both on stride (bursts) AND on static-camera trigger (opportunistic singles).
- `tests/capture/test_e2e.py`: round-trip a synthetic frame from EACH mode (trickle static, trickle pair, INSANE full-BRDF) through uploader → server → R2 without rejection.

### Constraints

- **Mode is set at install time** for v1. Live mode swap (tray-icon menu) is post-v1.
- **Default `lite`** for any installer that ships without an explicit `--mode` flag. The 99% case never has to think about this.
- **Trickle keeps all G-buffers.** Cheap to capture (~1.75 MB combined for depth + motion + normals); strictly more conditioning info than DLSS sees.
- **Trickle drops HR only on `t+1` of opportunistic pairs.** Frame `t` always has HR. Saves the ~3 MB that would otherwise blow the invisible-burden budget.
- **Lite includes opportunistic trickle SINGLES** (not pairs — lite already has dedicated short pairs every 80 s; opportunistic singles cover the static-camera DLSS-converged case at ~10% of lite's budget).
- **INSANE supersample-GT** can stutter the game during the 256-frame accumulator pass. Document in the install consent dialog.
- **Symmetric server-side schema patch already pending:** server commit accepts the full set of 4 modes and enforces trickle's two-path cross-validation (singles have no burst fields; pairs have `burst_index ∈ {0,1}` and `burst_tier = "short"`).

Final commit message: `capture(dll): trickle/lite/regular/INSANE mode presets + per-mode channel selection + INSANE supersample-GT trigger + lite opportunistic trickle frames + trickle opportunistic motion pairs`.

## What I'm doing in parallel

- `server/oss_capture_ingest/schema.py`: 4 modes accepted + trickle's two-path cross-validation (commit pending in this round).
- R2 path layout to include the mode: `<game_id>/<YYYY-MM>/<capture_mode>/<session_uuid>/<...>` so the daily index can stratify by mode without re-reading every file.
- `/stats` endpoint reports per-mode contribution counts.
- Update `scripts/build_capture_installer.py` to accept `--mode {trickle,lite,regular,INSANE}`.
