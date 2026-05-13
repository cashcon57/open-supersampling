# Capture data spec — Cyberpunk 2077 → OSS v7 training

**Date:** 2026-05-13
**Status:** Spec. The OSS capture tool (`oss/gaussian/interception/`) currently captures Tier 1 (DLSS NGX inputs only). Tiers 2–3 are gated on additional D3D12 hooks + post-processing.
**Audience:** Capture-tool engineers + dataset pipeline.

## What v7 needs to actually train

The model takes a **9-channel LR input** and supervises against an **HR ground truth**. Plus, for OSS-FX (intermediate-frame prediction), a **held-out α=0.5 HR frame** between frame N and frame N+1.

| v7 input channel | What it is | Shape at LR | Cyberpunk source |
|---|---|---|---|
| RGB (3) | Pre-tonemap HDR color, jittered, low-res | (3, H/2, W/2) | NGX `Color` input — already captured |
| Depth (1) | Linear view-space depth | (1, H/2, W/2) | NGX `Depth` input — already captured |
| Motion (2) | Screen-space velocity, frame N→N+1, in pixels | (2, H/2, W/2) | NGX `MotionVectors` × `mv_scale_x/y` — already captured |
| Normals (3) | World-space normals | (3, H/2, W/2) | **G-buffer pass — NOT yet captured** |

The matching HR target is the natively-rendered RGB frame at (3, H, W). For OSS-FX, that target is the *intermediate* frame at α=0.5 (the half-time between N and N+1).

## Three capture tiers

### Tier 1 — what the interception DLL already produces

From `oss/gaussian/interception/include/oss_gaussian_interception.h::OssGaussianFrame`:

| Resource | Type | Captured |
|---|---|---|
| Color (LR HDR) | `ID3D12Resource*` | ✅ |
| Output / HR target | `ID3D12Resource*` (UAV) | ✅ |
| Depth | `ID3D12Resource*` | ✅ |
| Motion vectors | `ID3D12Resource*` + `mv_scale_x/y` floats | ✅ |
| Exposure texture | `ID3D12Resource*` (optional) | ✅ |
| Jitter offset (subpixel) | `(float, float)` | ✅ |
| Render subrect | `(base_x, base_y, w, h)` + `(out_w, out_h)` | ✅ |
| Reset flag (camera cut) | `uint32` | ✅ |

This is **enough to retrain v6.x** (which is RGB+depth+motion = 6 channels, no normals). It is **not enough for v7** without normals.

### Tier 2 — what v7 additionally needs

| Channel | Why | How to get it from Cyberpunk |
|---|---|---|
| **Normals (world-space, 3-ch)** | v7's 9-ch input | Hook the G-buffer pass: REDengine 4 writes packed octahedral normals to one of the GBuffer SRVs. Identify the resource by format + usage during the deferred-shading pass and copy via a readback heap, same pattern as the existing depth grab. |
| **HR ground truth (clean)** | Supervised target | Already captured as the output UAV the game presents — but **disable DLSS Frame Generation** so it's a real render, not an interpolated frame. With DLSS Quality (67% LR), the "HR target" is DLSS-upscaled — acceptable as a soft target for v7-pico-005 but not ideal. Cleaner: run with DLAA (native res, AA only) → the output is true native HR. |
| **α=0.5 intermediate GT** | OSS-FX supervision | Run engine at 2× target framerate (e.g. 120 fps for a 60-fps target). Capture every frame. Frame indices (i, i+1, i+2): use i and i+2 as the model's input pair, hold out i+1 as the half-frame GT. Engine-side framerate cap via console or RivaTuner. |

### Tier 3 — what makes the dataset *good*, not just adequate

| Item | Why | Cost |
|---|---|---|
| **Albedo / base color (3-ch)** | Disentangles material from lighting; better disocclusion fill | Same G-buffer hook as normals |
| **Roughness / metallic / AO (3-ch)** | Reflectance-aware loss; helps with specular highlights | Same G-buffer hook |
| **Object ID buffer (1-ch uint)** | Per-instance segmentation → smarter foreground losses + reflection masks | Cyberpunk does have a stencil/ID buffer in REDengine; needs targeted reverse-engineering |
| **TAA jitter sequence** | Sub-pixel reconstruction lower-bound; lets us train with engine's *actual* sampling pattern | Already in `OssGaussianFrame.jitter_offset_*` — just need to log the running sequence not just the per-frame value |
| **Camera pose matrix (4×4)** | 3D-aware losses, parent-child Gaussian spawner conditioning | NGX doesn't expose this directly; grab from the constant buffer the game binds to its main vertex shader |
| **Per-frame timestamps (HW + present)** | OSS-FX α coordinate inference, frame-pacing diagnostics | `D3D12 GetCompletedFrameCount` + `IDXGISwapChain::GetLastPresentCount` |
| **Reflection mask (1-ch bool)** | Reflections and specular violate the "motion vector tracks pixel" assumption; masking them out of the temporal loss helps | If RT reflections enabled, mark RT-reflection pixels via an extra stencil tag. Otherwise: derive heuristically from roughness + view angle at training time. |
| **Sky/skybox mask (1-ch bool)** | Sky doesn't move with camera the same way ground does | Depth == max(far_plane) within tolerance |

## Capture conditions (must-disable / must-enable)

### Disable

- **DLSS Frame Generation** — must be OFF. We don't want interpolated frames in our training set.
- **Motion blur** (graphics settings → "Motion Blur" → off). Destroys per-pixel ground truth.
- **Depth of Field** (settings + photo mode override). Or capture DOF separately as a post-process mask.
- **Chromatic Aberration** (post-process).
- **Film Grain**.
- **Vignette / lens flares**.
- **HUD**: REDhud disable mod, or `Hide HUD` keybind in photo mode. UI overlays poison gradient targets.

### Enable / configure

- **DLAA** (if just doing SR training; renders at native HR, gives clean HR target).
- **DLSS Quality** with our DLSS-swap proxy (if doing α=0.5 OSS-FX; the proxy intercepts the call and gives us LR + jitter directly).
- **HDR output**: on. We want HDR linear color, not tonemapped sRGB.
- **Borderless windowed at native target res** (not exclusive fullscreen — exclusive complicates the hook).
- **V-sync off**, framerate cap at 2× target.
- **Console**: `Game.SetReflexLowLatencyMode(false)` — Reflex pacing can re-time the present and skew motion vectors.

## Scene diversity matrix

The dataset is only as good as its motion + lighting coverage. For Cyberpunk specifically, hit these categories:

| Category | Why | Approx. capture target |
|---|---|---|
| Day driving (Watson, Heywood) | Sustained linear motion, clean shadows | 5 trajectories × 30 s each |
| Night driving (rainy Night City) | Neon reflections, wet surfaces, dynamic lights | 5 × 30 s |
| First-person walk (interior) | Slow camera, lots of small object motion (NPCs, objects on tables) | 10 × 20 s |
| First-person sprint (exterior) | Fast camera, depth changes | 5 × 20 s |
| Combat (gunfight) | Particle effects, recoil shake, NPC ragdolls | 10 × 15 s |
| Cyberspace / braindance sequences | Stylized visuals; tests the model's robustness | 3 × 30 s |
| Photo mode static, varied camera | Diverse stationary scenes with controlled motion | 50 stills + slow camera dollies |
| Skybox / vista | Long-range depth, clouds, sun | 5 × 20 s |
| Reflective surfaces (chrome, glass, puddles) | Test the reflection-mask need | 10 × 15 s, hand-picked |
| Particles / volumetrics (smoke, fire, fog) | Hardest case for any temporal SR | 10 × 15 s |

Total: ~30 min raw 120 fps capture per category × ~50 categories × 2-3 sessions for variance → **~10–25 hours of capture** for a first dataset. After deduplication + cuts, expect ~50–100 GB packaged.

## Per-trajectory metadata

Each capture session writes a sidecar JSON / Parquet row with:

```json
{
  "trajectory_id": "cp2077-night-driving-rain-001",
  "game_version": "2.13",
  "redmod_patches": ["..."],
  "graphics_preset": "Ultra",
  "dlss_mode": "DLAA",          // or "Quality" + factor
  "dlss_frame_gen": false,
  "hdr_enabled": true,
  "resolution_render": [1920, 1080],
  "resolution_output": [1920, 1080],
  "framerate_capture_hz": 120,
  "scene_tags": ["night", "rain", "vehicle", "city", "neon"],
  "time_of_day_in_game": "23:47",
  "weather": "rain",
  "location": "Watson - Kabuki",
  "camera_mode": "vehicle_third_person",
  "capture_start_present_ts_ns": 17841234567,
  "duration_s": 32.5,
  "n_frames": 3900,
  "checksum_sha256": "..."
}
```

This is what the dataset loader joins against to filter trajectories at training time (e.g. "exclude all DLSS-Quality captures from the α=1 SR teacher run").

## Storage format

Per-frame, on disk:

| File | Format | Why |
|---|---|---|
| `frame_NNNNNN.lr.exr` | RGB16F | Pre-tonemap HDR linear |
| `frame_NNNNNN.hr.exr` | RGB16F | HR target (DLAA native or DLSS-upscaled) |
| `frame_NNNNNN.depth.exr` | R32F (linear) | Depth |
| `frame_NNNNNN.mv.exr` | RG16F | Motion vectors in pixels (post `mv_scale` multiply) |
| `frame_NNNNNN.normals.exr` | RGB16F (or R8G8 if octahedral packed) | World-space normals |
| `frame_NNNNNN.gbuffer.zst` | Custom multi-layer (Tier 3 only) | Albedo + roughness + metallic + AO + object ID, zstd-compressed |
| `trajectory.json` | JSON | Per-trajectory metadata above |
| `frame_NNNNNN.meta.json` | JSON, one per frame | jitter, mv_scale, exposure, camera matrix, present-ts |

EXR is the canonical industry choice for HDR + multi-channel float buffers. It's already what `oss/gaussian/interception/exr_writer.cpp` writes for the captures we have today.

## Sanity gates the capture tool must enforce

The capture session aborts (or flags the trajectory as discard) if any of these trip — these are not "would be nice," these are "the data is unusable if we ship it":

1. **DLSS Frame Generation detected ON** → abort. Our SR target must not be an interpolated frame.
2. **HUD detected on** (heuristic: gradient magnitude in the top-left 200×100 region exceeds a threshold for >3 consecutive frames) → flag.
3. **Motion vectors zero everywhere for >5 frames while RGB is changing** → flag (mv buffer not actually written by engine; we'd be training on garbage).
4. **Frame-time variance > 30% of mean for the trajectory** → flag (stuttering would skew the α=0.5 timing).
5. **Depth all-zero or all-max** → abort (depth buffer cleared and never written, or wrong target hooked).
6. **Jitter offset constant across frames** → flag (TAA disabled or jitter sequence broken).

## Phasing — what to build first

**Phase A (unlocks v7 retraining on captured data, not just TartanAir):**
1. Add G-buffer hook → normals export. Most of the work; ~2 weeks if REDengine internals cooperate.
2. Add `frame_NNNNNN.meta.json` writer with jitter sequence + camera matrix + present-ts.
3. Implement the 6 sanity gates above in the capture-loop supervisor.

**Phase B (unlocks Tier 3 quality):**
4. Albedo + roughness + metallic + AO export.
5. Object ID buffer export.
6. Reflection mask (RT-on or roughness-derived).

**Phase C (operational):**
7. Capture-session UI that drives the diversity matrix (preset trajectories, scene-tag tagging).
8. Automatic upload to R2 with the sidecar JSON.
9. Dataset-builder script that emits a TartanAir-shaped torch Dataset over the captured EXRs so the v7 trainer can consume them directly.

## What this spec does NOT cover

- **Anti-cheat compatibility**. CD Projekt has not bricked modders historically but no formal posture.
- **Multi-GPU / SLI**. Not a concern for current OSS targets.
- **Other engines** (Unreal, idTech, Frostbite). REDengine-specific G-buffer layouts won't port directly; each game needs its own hook profile.
- **Audio / dialogue / cutscene capture**. Out of scope.
- **Player anonymization** (game saves contain PII-shaped data). Out of scope for the capture; handle in the upload pipeline.
