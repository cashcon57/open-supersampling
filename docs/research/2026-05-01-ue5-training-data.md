# UE5 Path Tracer Training Data Pipeline

**Status:** Design  
**Date:** 2026-05-01  
**Scope:** AAA-quality game-engine RT training data via Unreal Engine 5 Path Tracer

---

## Why UE5

Mitsuba 3 gives us clean synthetic data with controlled diversity but path-tracer noise characteristics. Real games use hybrid rendering: rasterization + 1 ray/pixel via RTXDI or ReSTIR. The noise distribution from these algorithms differs fundamentally from path-tracer noise — different firefly patterns, different disocclusion artifacts, different temporal instability. A denoiser trained only on path-tracer noise will underperform on game content.

UE5 has two RT paths:

- **Lumen** — production RT, but the denoiser (Lumen HWRT denoiser) runs inline. We can't extract pre-denoised frames without engine source modifications.
- **Path Tracer** — reference renderer, configurable SPP, outputs clean G-buffers. This is what we use.

UE5 Path Tracer gives us:
- 1-SPP noisy frames matching the noise budget of a real game RT pass
- Full G-buffer access (depth, normals, albedo, motion vectors, roughness)
- AAA-quality geometry from Epic's free content
- Python automation via Unreal Editor scripting

The gap vs actual game RT: UE5 Path Tracer uses standard PT, not RTXDI/ReSTIR. The noise patterns are closer to Mitsuba than to real game RTXDI. However, the geometry, materials, and scene complexity are orders of magnitude more realistic than procedural Mitsuba scenes — and that matters for generalization.

For RTXDI/ReSTIR noise specifically: that requires live game capture with a render hook, which is deferred to v3.

---

## Content Sources

All free for non-commercial research (UE EULA permits research use of Marketplace content):

| Asset Pack | Geometry complexity | Notes |
|-----------|--------------------|----|
| City Sample (Matrix demo) | Street-level urban, 8km² | Nanite meshes, realistic urban RT |
| Valley of the Ancient | Indoor + outdoor, large cave | Complex GI, organic shapes |
| Lyra Starter Game | Indoor corridors, modular | Combat-oriented layouts |
| Fab Marketplace free tier | Various | Check licenses per asset |

These cover the scene archetypes that matter most: urban exteriors, interior rooms, complex organic environments.

---

## Pipeline

```
UE5 Editor (Python automation)
  ├── Iterate camera positions (N positions per level)
  ├── For each position, render sequence of T frames (camera + scene motion)
  │     ├── Path Tracer at 1 SPP → noisy color frame
  │     ├── Path Tracer at 512 SPP → GT reference
  │     └── Custom Render Passes → G-buffers (depth, normals, albedo, motion, roughness)
  └── Write EXR per frame → package into zarr ZipStore (NoiseBase schema)
```

### Camera Sampling

N=200 cameras per level, sampled as:
- 60% ground-level (1.6m height), random yaw, looking toward scene center ± 45°
- 20% elevated (5-15m), looking down at 20-45° depression angle  
- 20% interior positions (if available), randomized

Sequences of T=8 frames with smooth camera motion (translation 0.1-2m/frame, rotation 0-3°/frame). Include a few frames with fast motion to stress the temporal model.

### UE5 Python Automation

UE5 Editor supports Python scripting via `unreal` module (editor scripting utilities plugin). The automation script:

1. Loads each level
2. Sets Path Tracer as active renderer (`r.AntiAliasingMethod 4` in console, or via settings API)
3. Configures SPP (`r.PathTracing.MaxSPP 1` for noisy, `r.PathTracing.MaxSPP 512` for GT)
4. Iterates camera positions, calls `unreal.EditorLevelLibrary.set_level_viewport_camera_info()`
5. Exports via Movie Render Queue (Python API: `unreal.MoviePipelineQueueEngineSubsystem`)
6. MRQ outputs: each frame as EXR with separate passes for G-buffers

### G-Buffer Export via Movie Render Queue

MRQ supports per-pass EXR export natively:
- `MoviePipelineDeferredPassBase` for G-buffers (BaseColor/albedo, WorldNormal, SceneDepth, Velocity/motion)
- `MoviePipelinePathTracerPass` for noisy and GT color
- Custom passes: roughness via material parameter collection

Motion vectors from UE5: the `Velocity` pass outputs screen-space velocity in pixels/frame. Divide by resolution to get NDC motion vectors — same convention as NoiseBase.

Roughness: UE5's G-buffer `Roughness` channel is directly available as a separate MRQ pass.

### Output Format

Same zarr ZipStore schema as the Mitsuba pipeline. The EXR → zarr packaging script (`scripts/package_ue5_exr.py`) converts MRQ output to NoiseBase-compatible format. This means both Mitsuba and UE5 data load through the same `NoiseBaseDataset` without changes.

---

## Scale Target

| Level | Cameras | Sequences | Frames | GT SPP | Storage est. |
|-------|---------|-----------|--------|--------|--------------|
| City Sample (subset) | 200 | 200 | 1,600 | 512 | ~50 GB |
| Valley of the Ancient | 150 | 150 | 1,200 | 512 | ~38 GB |
| Lyra corridors | 100 | 100 | 800 | 512 | ~25 GB |
| **Total v1** | **450** | **450** | **3,600** | 512 | **~113 GB** |

450 sequences is modest — but UE5 geometry diversity is very high, so each sequence carries more training signal than a Mitsuba procedural one.

Render time at 512 SPP + 1 SPP per frame pair, on RTX 3080 Ti (Windows machine on Tailnet):
- 1080p, 512 SPP: ~45 seconds/frame on RTX 3080 Ti
- 1080p, 1 SPP: ~0.5 seconds/frame
- Per sequence (8 frames): ~8 × (45 + 0.5) = ~364 seconds ≈ 6 minutes
- 450 sequences: ~2,700 minutes ≈ 45 GPU-hours on the 3080 Ti

Running on the 3080 Ti locally overnight gives the full dataset in ~2 nights.

---

## Implementation Plan

### Phase 1: Pipeline setup (local, 3080 Ti Windows)
1. Install UE5 + City Sample on the 3080 Ti Windows machine
2. Enable Editor Python Scripting plugin
3. Write `scripts/ue5_capture.py` — automation script (runs inside UE5 Editor Python)
4. Write `scripts/package_ue5_exr.py` — EXR → zarr packager
5. Test on 10 sequences from City Sample

### Phase 2: Full dataset capture
1. Run City Sample capture (~15 GPU-hours)
2. Run Valley + Lyra (~15 + 8 GPU-hours)
3. Package to zarr, upload to R2
4. Validate with `NoiseBaseDataset` loader

### Phase 3: Training with mixed data
1. Combined dataloader: NoiseBase + Mitsuba synthetic + UE5
2. Curriculum: NoiseBase first (RT noise distribution), then UE5 (geometry generalization)
3. Track metrics separately on NoiseBase test set vs UE5 validation set

---

## UE5-Specific Caveats

**Path Tracer vs Lumen noise:** PT noise is different from Lumen HWRT noise. This data helps with geometry generalization but not with the specific noise signatures of RTXDI/ReSTIR used in real games. Improving on that requires live game capture (deferred).

**Motion vectors:** UE5 Velocity pass includes both camera motion and object motion. Nanite meshes with World Position Offset animations produce correct velocity. Static scenes will have camera-motion-only vectors.

**HDR range:** UE5 Path Tracer outputs in scene-referred linear HDR. City Sample has extremely high dynamic range (bright sky + dark interiors in same frame). The RGBE encoder must handle this gracefully — use the full per-frame [emin, emax] range.

**Roughness channel:** UE5 material roughness is in [0, 1], matching NoiseBase convention. Direct export to G-buffer roughness pass works.

**License reminder:** UE5 EULA allows research use but prohibits commercial redistribution of the assets themselves. We can publish weights trained on this data (transformative use). We cannot publish the raw EXR frames.

---

## Open Questions

1. **Can we automate MRQ from Python without opening the full editor UI?** UE5 supports headless rendering via `-game` mode + console commands, but Path Tracer support in headless mode is untested. Fallback: interactive editor with automation.

2. **Motion blur at 1 SPP:** UE5 PT does motion blur by distributing samples across the shutter interval. At 1 SPP, motion blur effectively disappears (single time sample). This means our 1-SPP noisy frames lack motion blur even though real game frames at 1 SPP would have it from temporal accumulation. Acceptable for v1.

3. **Nanite mesh density:** City Sample uses Nanite. At 1 SPP, the ray-triangle intersection cost varies wildly with Nanite LOD behavior. This may produce unusual noise patterns not present in non-Nanite scenes. Worth checking with a reference render.
