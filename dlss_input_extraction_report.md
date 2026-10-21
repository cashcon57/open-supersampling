# DLSS Input Extraction Feasibility Report

This document contains the three model answers and the synthesis regarding the feasibility of extracting DLSS-related inputs from games for inspection and training a Gaussian-splatting-based upscaler.

## GPT-5.5 Thinking

## Bottom line

Yes — it is technically feasible to extract the same class of buffers DLSS 4 uses, export them, inspect them, and train another temporal/geometry-aware upscaler around them. The highest-feasibility path is **not** “rip the whole G-buffer from arbitrary games”; it is either:

1. **Instrument an engine you control** and export color/depth/velocity/G-buffer/metadata directly, or  
2. **Intercept the game’s DLSS/Streamline/NGX integration** and copy the exact resources the game already tags for DLSS before DLSS consumes them.

The big caveat: **DLSS Super Resolution does not normally take a full G-buffer**. For DLSS SR/DLAA, the public interface is basically low-res color + depth + motion vectors + exposure/jitter/camera constants/history/control flags; DLSS uses motion data and prior-frame feedback to reconstruct high-res frames[web:26][web:49]. Full-ish G-buffer data — normals, albedo, roughness, hit distances, specular albedo — is more associated with **DLSS Ray Reconstruction**, not normal DLSS SR[web:29]. DLSS Frame Generation has its own input set: final/backbuffer color, depth, motion vectors, optional HUD-less/UI buffers, and optional bidirectional distortion field[page:1].

For a Gaussian-splatting upscaler, “DLSS inputs” should be treated as a **minimum viable feature set**, then optionally extended with full G-buffer channels when the renderer is under your control.

## What DLSS 4 actually exposes as inputs

### DLSS Super Resolution / DLAA

Public DLSS SR documentation describes DLSS as reconstructing high-resolution frames from lower-resolution images using motion data and feedback from previous frames[web:26][web:49]. NVIDIA’s DLSS programming-guide search result explicitly lists input buffers such as **color, motion vectors, depth, and optionally history/exposure**[web:71]. Streamline/DLSS integrations also require per-frame constants such as motion-vector scale, depth settings, camera matrices, frame index, and jitter-related data; the Streamline guide notes that motion vectors may be in `[-1,1]` or pixel units and must be normalized through `mvecScale`[page:1].

So the practical SR/DLAA export set is:

- Low-res rendered color / scene color
- Depth or linear depth
- Dense 2D motion vectors
- Exposure / pre-exposure / luminance metadata when available
- Jitter offset
- Render resolution and output resolution
- Motion-vector scale
- Camera matrices: current/previous view-projection, usually non-jittered
- Reset/disocclusion/history flags if exposed
- Optional masks depending on integration: reactive/transparency masks, sharpening flags, etc.

### DLSS Frame Generation / Multi Frame Generation

DLSS-G/Frame Generation is more explicit in Streamline. Its required buffers are **depth** and **motion vectors**, while final color is intercepted through the swapchain/present path[page:1]. For best UI quality, Streamline recommends integrating either **UI Color and Alpha** or **HUD-less color**, and it also supports an optional **bidirectional distortion field** for strong post-process distortion effects[page:1].

So for DLSS-G-style data capture:

- Final color / backbuffer
- HUD-less color, if available
- UI color + alpha, if available
- Depth
- Motion vectors
- Optional bidirectional distortion field
- Camera/common constants
- Frame index / presentation metadata

Important: NVIDIA’s optical flow used internally for frame generation is not generally handed to you as an exportable buffer. The game-provided motion-vector field can be captured, but not necessarily DLSS’s internal optical-flow/intermediate tensors.

### DLSS Ray Reconstruction

Ray Reconstruction is the one that looks more like “DLSS with a G-buffer.” NVIDIA’s Streamline RR guide says it can use depth, motion vectors, albedo, normal, roughness, specular/diffuse hit distance, specular albedo, and exposure[web:28][web:29]. If a Gaussian upscaler wants material/normal priors, RR-enabled games are a better target than SR-only games, but the buffers are still engine-specific and not guaranteed to map cleanly to a universal schema.

## Feasibility by approach

| Approach | Feasibility | Best use | Major problem |
|---|---:|---|---|
| Engine-side export in Unreal/Unity/custom renderer | Very high | Real training dataset | Requires control over renderer |
| Streamline/NGX interception in DLSS-enabled game | Medium-high | Exact DLSS input inspection | D3D12/Vulkan complexity, EULA/anti-cheat issues |
| RenderDoc/Nsight/PIX frame capture | Medium | Reverse-engineering buffer semantics | Not scalable for huge datasets |
| Generic post-process injector / ReShade-style capture | Low-medium | Color/depth experiments | Usually no reliable motion vectors/G-buffer |
| Arbitrary commercial online game | Low / not recommended | Avoid | Anti-cheat/legal/stability problems |

## Best technical path: intercept Streamline/NGX, not the whole renderer

If the game uses NVIDIA Streamline, the game has to explicitly tag the resources it wants DLSS/FG/RR to consume. The Streamline Frame Generation guide shows `slSetTagForFrame` being used to tag depth, motion vectors, HUD-less color, UI color/alpha, and optional distortion-field resources[page:1]. That is the cleanest “wiretap” point.

A proxy/shim could:

1. Intercept `slSetTagForFrame`.
2. Log each `ResourceTag`: buffer type, texture pointer/handle, format, extent, resource lifecycle.
3. Intercept `slSetConstants` / feature constants.
4. Copy tagged GPU resources into a ring buffer before DLSS evaluates or before `Present`.
5. Asynchronously read back and serialize to disk.

For direct NGX integrations, the equivalent interception point is the NGX parameter system. NGX’s programming guide says feature input buffers are passed as parameters, including `ID3D*` resources or CUDA buffers[page:2]. A proxy around `NVSDK_NGX_D3D12_EvaluateFeature` / `NVSDK_NGX_D3D11_EvaluateFeature` can inspect the parameter block and copy the resource handles before forwarding the call.

This would give you the **actual buffers DLSS sees**, which is much better than trying to infer which random render target in a RenderDoc capture is “the velocity buffer.”

## Why full G-buffer extraction is harder than DLSS-input extraction

A DLSS-enabled game reliably passes **some** semantic buffers to DLSS: color, depth, velocity, exposure/constants, and maybe masks. But a “G-buffer” is not standardized.

Different engines pack G-buffer channels differently:

- Normals may be octahedral encoded, view-space, world-space, or compressed.
- Roughness/metallic/specular may be packed into arbitrary channels.
- Motion vectors may be pixel-space, normalized-space, jittered, non-jittered, previous-to-current, or current-to-previous.
- Depth may be hardware depth, reversed-Z, linear eye depth, logarithmic, or custom.
- Forward renderers may not have a complete G-buffer at all.
- Deferred renderers often discard or alias G-buffer targets after lighting.

Streamline itself warns that depth and motion-vector interpretation depend on constants like depth inversion and `mvecScale`; motion vectors must be scaled into Streamline’s expected normalized range[page:1]. So even when the texture can be copied, its coordinate system and packing still need decoding.

For training, a canonical schema is better than trying to mimic each game’s native layout.

## Recommended canonical training schema for a Gaussian upscaler

For each frame `t`, export:

### Core DLSS-like inputs

- `color_lr_t`: low-res scene color, preferably pre-HUD
- `depth_lr_t`: linear depth in meters or normalized view depth
- `motion_lr_t`: dense 2D motion vectors, canonicalized to pixels or normalized UV
- `exposure_t`: exposure/pre-exposure value
- `jitter_t`: subpixel jitter offset
- `camera_t`: view, projection, view-projection, inverse matrices
- `camera_t_minus_1`: previous-frame matrices
- `render_size`: internal render resolution
- `display_size`: target output resolution
- `frame_index`
- `reset_history`: whether temporal history should be invalidated

### Strongly recommended for Gaussian splatting

- `normal_lr_t`: world-space or view-space normals
- `albedo_lr_t`
- `roughness_metallic_lr_t`
- `object_id` or `material_id`, if available
- `reactive_mask`: particles, transparency, animated texture regions
- `disocclusion_mask`: derived from depth/motion reprojection
- `confidence_mask`: where motion/depth are reliable
- `hud_mask` / `ui_alpha`
- Optional: optical flow estimate from a separate model, but keep it separate from game motion vectors

### Targets

- `color_hr_t`: high-resolution ground truth, preferably rendered natively or supersampled
- Optional `color_hr_t_plus_1` / `color_hr_t_minus_1` for temporal supervision
- Optional path-traced or high-sample reference for still scenes

For a Gaussian-canvas upscaler, the critical part is **not just color/depth/MV**. Enough metadata is needed to spawn/update/confidence-weight splats:

- Depth → approximate splat scale and occlusion.
- Normal → anisotropic footprint orientation/confidence.
- Motion vector → temporal propagation.
- Albedo/material → separate texture detail from lighting noise.
- Disocclusion/reactive masks → prevent ghost trails.
- Jitter → reconstruct subpixel sample positions correctly.

## Training-data problem: existing games usually do not give clean ground truth

Extracting DLSS inputs from a commercial game is feasible. Creating **good supervised training pairs** from that game is much harder.

The needed relationship is:

`low-res inputs at frame t` → `high-res ground truth for same exact frame t`

In an arbitrary game, if a 1080p DLSS input is captured and then the game is later run at 4K native, the frame may not match. TAA history, particles, random animation, LOD changes, dynamic resolution, streaming, shader stochasticity, and post-processing can all shift.

For serious training, the best options are:

1. **Controlled renderer / engine build**  
   Render the same deterministic frame twice: once low-res with exported inputs, once high-res/supersampled as target.

2. **Offline replay / photo-mode deterministic capture**  
   Useful if the game has a deterministic replay camera and time, RNG, LOD, exposure, weather, and animation can be locked.

3. **Synthetic scenes**  
   Build Unreal/Godot/custom scenes with known buffers and HR targets.

4. **Commercial-game capture as domain adaptation only**  
   Use real-game captures to test robustness or self-supervise, but do not rely on them as the primary supervised ground truth.

## Practical exporter architecture

### Phase 1: manual inspection

Use Nsight Graphics, RenderDoc, or PIX to inspect frames and identify buffer semantics. Nsight Graphics supports frame capture and examination of rendering calls, bound textures, geometry, and unordered access views[web:91][web:96]. Nsight’s Current Target view shows currently bound color, depth, and stencil targets as events are scrubbed[web:95]. RenderDoc-style workflows can export textures and buffers, but its own issue tracker reflects that bulk export/asset ripping is not really the intended workflow[web:121].

Goal of phase 1:

- Confirm where DLSS/Streamline evaluation happens.
- Identify the actual color/depth/velocity resources.
- Determine formats and scales.
- Validate motion-vector direction and units.
- Check whether HUD-less color exists.
- Check whether normals/albedo/roughness are still alive at DLSS time.

### Phase 2: real-time capture shim

For D3D12:

- Hook Streamline or NGX calls.
- For each tagged `ID3D12Resource`, copy to your own GPU texture.
- For color/MV/normal/albedo, use `CopyTextureRegion` when formats permit.
- For depth, often use a shader copy into `R32_FLOAT`, because depth-stencil formats can be awkward.
- Put readback copies into a ring buffer.
- Read back N frames later to avoid stalling.
- Compress/write on a background thread.

For Vulkan:

- Use a Vulkan layer or Streamline wrapper.
- Track `VkImage`, layout, queue family, extent, format.
- Copy or blit into export images.
- Use staging buffers and fences.
- Serialize with metadata.

For D3D11:

- Easier than D3D12 in some ways.
- Copy to staging textures.
- Map asynchronously where possible.
- Still be careful about stalling the GPU.

### Phase 3: canonicalization

Before saving, convert everything to a training format:

- Color: linear HDR if possible, EXR/FP16.
- Motion vectors: one convention, e.g. previous-frame source position minus current pixel in pixels.
- Depth: linear view-space depth, not raw hardware depth.
- Normals: decoded float3.
- Exposure: scalar JSON field.
- Matrices: row/column-major documented explicitly.
- Jitter: pixels or NDC, documented explicitly.
- Resolution: store both source and target.

Training directly on raw packed game buffers is discouraged unless the model also receives a per-game decoder, because otherwise it will learn engine-specific packing artifacts.

## Storage/bandwidth reality check

At 1920×1080, the data can become large quickly:

- `RGBA16F color`: ~16 MB/frame
- `RG16F motion`: ~8 MB/frame
- `R32F depth`: ~8 MB/frame
- `RGBA16F normals/albedo/material extras`: easily another 16–48 MB/frame

A rich dataset can therefore be **40–100+ MB per frame uncompressed**. At 60 FPS, raw capture can hit multiple GB/s. For training, practical mitigations include:

- Capture every Nth frame, not every frame.
- Use FP16 where possible.
- Store tiled chunks.
- Use EXR, Zarr, NPY, KTX2, or custom binary chunks.
- Keep metadata JSON/MessagePack/CBOR sidecars.
- Use a ring-buffered writer to avoid frame hitches.

## What can and cannot be learned from extracted DLSS inputs

You can inspect:

- The actual low-res color DLSS sees.
- Depth buffer values.
- Motion-vector density and direction.
- Jitter pattern.
- Exposure behavior.
- HUD/UI separation quality.
- Whether dynamic objects have proper velocity.
- Whether transparencies/reactive regions are represented.

You generally cannot inspect:

- DLSS model weights.
- Internal tensors.
- Internal history representation.
- NVIDIA’s internal optical-flow output.
- DLSS’s internal feature normalization.
- Proprietary transformer attention/state.
- Training targets NVIDIA used.

That is acceptable for this use case because a competing or alternative upscaler only needs a robust buffer schema and good supervised targets.

## Legal / operational caveats

Recommended environments:

- Your own renderer.
- Open-source games.
- Sample scenes.
- Modding-friendly single-player titles.
- Games where capture/debugging is allowed.

Avoid:

- Online games with anti-cheat.
- Circumventing anti-debug protections.
- Redistributing copyrighted captured assets/textures.
- Shipping a tool that injects into protected games.
- Using extracted commercial-game captures as a public dataset without permission.

For research/dev on a private machine, the safer path is to build the pipeline around **engine-side export first**, then use commercial games only for local qualitative inspection.

## Recommendation for a Gaussian-splatting upscaler

For OpenSuperSampling, scraping arbitrary games should not be the starting point. A two-track dataset plan is stronger:

### Track A — clean supervised dataset

Use Unreal, Unity HDRP, Godot, or a custom renderer to export:

`LR color + depth + MV + normals + albedo + roughness + jitter + matrices`  
paired with  
`HR native/supersampled target`.

This gives clean ground truth and lets the Gaussian canvas train correctly.

### Track B — DLSS-input compatibility dataset

Build a Streamline/NGX shim to capture real DLSS inputs from DLSS-enabled single-player games. Use this for:

- Distribution shift testing.
- Motion-vector weirdness.
- HUD/transparency cases.
- Real engine post-processing artifacts.
- Fine-tuning if reasonable pseudo-targets can be obtained.

The key architectural insight is to **target the DLSS integration boundary, not the whole game renderer**. If a game already supports DLSS, it already hands NVIDIA the buffers that matter. The task is to duplicate those resources, canonicalize them, and pair them with good targets. Full G-buffer capture is a bonus, not the foundation.

## Claude Opus 4.7 Thinking

I have enough. Let me write the feasibility analysis.

# Feasibility: Extracting DLSS 4 Inputs for a Gaussian-Splatting Upscaler Training Pipeline

**Short answer: Highly feasible, with a clear hierarchy of approaches ranked by effort vs. fidelity. The easiest, cleanest path is to bypass DLSS entirely and source data directly from an engine you control (UE5 with a custom capture pass), because intercepting a closed shipping game's buffers is technically possible but legally and engineering-wise painful. The Streamline SDK is your friend if you want to mimic what DLSS 4 actually sees.**

## 1. What DLSS 4 actually consumes

Per NVIDIA's Streamline integration guide and the DLSS Programming Guide, the SR (super-resolution) path consumes:

- **Jittered low-resolution color** (pre-TAA, pre-tonemap, ideally HDR linear)[web:13][web:71]
- **Depth buffer** (with an "inverted depth" flag if reversed-Z)[web:13][web:28]
- **Dense screen-space 2D motion vectors** (and a separate specular/reflection MV channel for Ray Reconstruction)[web:13][web:71]
- **Exposure / brightness scalar (or texture)**[web:71][web:28]
- **Jitter offset per frame**, **MV scale**, **frame index**, and a set of per-frame constants (view/proj matrices, near/far, dynamic-resolution rect)[web:13][page:1]

For DLSS-G (Frame Gen) you additionally need **HUD-less color** and a **UI-only color buffer**, so the model can interpolate without smearing the HUD[page:1]. DLSS 4's Ray Reconstruction adds normals, roughness, and other G-buffer attributes (albedo/specular albedo, world-space normals, hit distance) — these are what NVIDIA's research page describes as "intermediate inputs" that engines provide but that "are often inaccurate for effects like reflections"[web:11].

So the target capture set is roughly: `{color_jittered_LR, depth, MV, specular_MV, normals, roughness, albedo, exposure, jitter_xy, view_proj, prev_view_proj, frame_idx}` plus a high-resolution ground-truth color (rendered native or DLAA/SSAA) for supervised training.

## 2. Five viable extraction paths, ranked

### Path A — Engine-side capture in UE5/Unity (recommended for ML training)

Build a custom render pass that writes the buffers DLSS would receive *before* the upscaler runs. In UE5 the DLSS plugin already routes these through Streamline; you can fork the plugin or add a SceneViewExtension that dumps `FSceneTextures` (SceneDepth, GBufferA/B/C/D/E, Velocity, SceneColor pre-TAA) per frame[web:50][web:53]. Pair each LR jittered frame with a corresponding **high-res non-jittered DLAA/native render** as ground truth.

Pros: deterministic, perfectly aligned, no reverse engineering, jitter sequences are controllable, and the same path can be re-rendered at multiple resolutions for paired data. This is also the only path that legally scales to a published dataset.

### Path B — Streamline SDK "passthrough" or custom feature

Streamline is open-source and sits between the game and DLSS as a thin tagging layer[web:57][web:66]. You can:
1. Write a custom Streamline feature plugin that registers for the same tagged buffers DLSS consumes (`kBufferTypeDepth`, `kBufferTypeMotionVectors`, `kBufferTypeHUDLessColor`, etc.) and dumps them to disk instead of, or in addition to, calling DLSS[web:56][page:1].
2. Use Streamline's existing **debug visualization** (CTRL+SHIFT+INS) as a sanity check that the right frame-aligned buffers are being captured[web:56].

This is the cleanest “intercept DLSS 4 inputs” route for any title that already integrates Streamline.

### Path C — `nvngx_dlss.dll` proxy / wrapper DLL

DLSS Swapper, DLSS Enabler, and DLSSTweaks all demonstrate the pattern: replace `nvngx_dlss.dll` with a shim that forwards the NGX API while logging[web:156][web:157][web:158]. This can be extended to **dump the input resources** passed to `NVSDK_NGX_D3D12_EvaluateFeature` (or the Vulkan equivalent) — those calls receive D3D12 resource pointers for color, depth, MV, exposure, etc.[page:2] Caveat: this only works on titles without DLL signature checks, and modifying DLSS behavior in distributed form likely violates NVIDIA’s terms, so it should remain private/local research.

### Path D — RenderDoc / PIX captures, scripted

RenderDoc and PIX both record full D3D12/Vulkan command streams and let you inspect any resource at any draw[web:87][web:22]. RenderDoc has a Python API that can iterate captures, locate the DLSS dispatch, and export the bound SRVs (color/depth/MV) as EXR/DDS[web:121][web:125]. Limitation: capture is per-frame and *slow* — good for bootstrapping and validation, bad for large datasets.

### Path E — DXGI swapchain / D3D12 API hooking

Generic API hooks can grab the final color but not internal G-buffers reliably, because by present-time many buffers are gone. This is not recommended for training data quality.

## 3. Practical gotchas

- **Jitter convention**: DLSS expects jitter in pixel units; mismatched sign or scale creates soft or ghosty output[page:1].
- **Motion vector scale and direction**: DLSS expects MVs mapped correctly and normalized with `MVScale`[web:13][page:1].
- **Reversed-Z**: modern engines often use 1→0 depth, so the flag or normalization must be captured[web:28].
- **HDR vs tonemapped**: DLSS works on pre-tonemap linear HDR. Post-tonemap color changes the problem.
- **Dynamic resolution rects**: capture the subrect metadata per frame or training will be corrupted.
- **Frame alignment**: depth and MV at frame N must correspond to color at frame N. Streamline’s volatile tagging rules exist because games may destroy these buffers before Present[page:1].

## 4. Specific to a Gaussian-splatting upscaler

A 3DGS-based upscaler is an unusual target because splats are usually a scene representation, not a screen-space residual predictor.

- If splats are used as a **learned per-frame upsampling primitive** (predict a sparse set of 2D/2.5D Gaussians that reconstruct the HR frame), then the DLSS input set is exactly the right starting point: depth gives approximate 2.5D positions, MVs give temporal reprojection priors, and normals/albedo give appearance disentanglement.
- If the goal is instead a **3D Gaussian field** refined per frame, then camera intrinsics/extrinsics and `prev_view_proj` are also needed. Streamline already exposes those via constants[page:1].

For dataset scale, the training bottleneck for large Gaussian systems is often GPU memory rather than raw data count[web:109][web:110]. A few hundred GB of paired LR+HR captures from UE5 can therefore be enough.

## 5. Recommended pipeline

1. **Stand up Path A in UE5**. Fork the NVIDIA DLSS UE plugin or add a capture extension that dumps the Streamline-tagged buffers plus a ground-truth DLAA/native target.
2. **Validate with Path D (RenderDoc)** on a single frame: capture the DLSS dispatch and diff the dumped buffers against the SRVs bound to NGX.
3. **Optionally use Path B (Streamline custom feature)** if third-party DLSS-enabled single-player titles are later needed for robustness testing.
4. Avoid Path C for any public dataset because of legal and deployment risk.

## 6. Bottom line

Every input DLSS 4 takes is, by design, *already exposed* through Streamline’s tagging API, the UE5 plugin path, and less cleanly through NGX interception[web:56][web:57][page:2]. There is no cryptographic gate or proprietary buffer format at the API boundary — DLSS receives ordinary D3D12/Vulkan resources. The real work is (a) building a capture harness that respects jitter/MV/depth conventions exactly, and (b) generating aligned high-resolution ground truth, which engine-side rendering gives for free and game-side interception does not. For a Gaussian-splatting upscaler training corpus, the best path is engine-side export that mirrors the Streamline input contract, with RenderDoc used only as a correctness oracle.

## Gemini 3.1 Pro Thinking

It is **highly feasible** to extract the exact inputs a game sends to DLSS (G-buffers, motion vectors, depth, etc.) and dump them to disk to build a dataset. The modding community has already paved the way for this type of interception, although their usual goal is translating those inputs to other upscalers such as FSR or XeSS rather than saving them.

Because this is for a Gaussian splatting-based upscaler, pristine, frame-perfect floating-point data is required, so traditional screen-capture tools will not work.

### 1. The Interception Point (Proxy DLLs)

Modern NVIDIA features (DLSS 2, 3, and newer iterations like Ray Reconstruction and Frame Generation) are integrated into games via the **NVIDIA Streamline API** (`sl.interposer.dll`) or the traditional **NGX API** (`nvngx_dlss.dll`)[web:50][web:57].

To get the data, the game engine itself does not necessarily need to be hacked. Instead, a "wrapper" or "proxy" DLL can be created:
1. Rename the real `nvngx_dlss.dll` to something else.
2. Place a custom `nvngx_dlss.dll` in the game directory.
3. The game loads the custom DLL first.
4. Each frame, the game calls `NVSDK_NGX_D3D12_EvaluateFeature` (or the Vulkan/Streamline equivalent) and passes pointers to the exact inputs needed[page:2].

### 2. What data can be extracted?

When the game calls the Evaluate function, it passes a parameter structure containing pointers to GPU textures/resources. The standard data sent includes:

- **Unresolved color buffer:** the raw, aliased, low-resolution render.
- **Depth buffer:** typically high precision, often inverted depth depending on engine conventions[web:28].
- **2D motion vectors:** often FP16 textures representing per-pixel movement between frames[page:1].
- **Jitter offsets:** crucial for upscalers; these are sub-pixel camera shifts for the current frame[page:1].
- **Reactive / transparency masks:** when supported, they tell the upscaler to distrust certain regions.
- **Exposure textures or scalars:** used for auto-exposure handling[web:71].

### 3. The extraction pipeline

Because these inputs are in GPU VRAM, they cannot simply be read like ordinary CPU memory. A workable pipeline is:

1. **Intercept the pointers** inside the Evaluate call.
2. **Create staging/readback buffers** in DirectX 12 or the equivalent in Vulkan.
3. **Issue GPU copy commands** to move the depth, motion-vector, and color textures into CPU-visible buffers.
4. **Map and serialize** those buffers into a precision-preserving format such as OpenEXR.
5. **Optionally pass through** the original call to the real DLSS DLL so the game continues rendering while data is dumped.

### 4. Reusing existing open-source projects

A strong starting point is to fork **OptiScaler**, because it already contains much of the logic for intercepting DLSS-related inputs and remapping them to other upscalers[web:83][web:143].

It already handles:
- Hooking DirectX 11, DirectX 12, and Vulkan entry points.
- Finding the depth, color, and motion-vector resources.
- Working around engine quirks such as DLSS/FSR input translation.

Instead of passing those buffers into another upscaler, the evaluation loop can be modified to enqueue GPU readbacks and write the results to disk, alongside metadata such as jitter offsets and frame index.

### 5. Challenges for dataset generation

- **GPU/CPU stalls:** If readback is done synchronously every frame, performance collapses. A ring buffer or delayed readback queue is required.
- **Storage throughput:** Uncompressed EXR sequences for color, depth, and motion vectors will consume storage extremely quickly.
- **Engine-specific quirks:** Unreal Engine titles may send dilated motion vectors, some engines send different depth encodings, and dynamic resolution can change dimensions frame to frame[web:138].

Those buffers will need standardization before being fed into a Gaussian splatting model.

## Synthesis

### Where the answers agree

All three analyses agree that extracting the buffers a game feeds to DLSS is technically feasible, because DLSS receives ordinary GPU resources through Streamline or NGX rather than some inaccessible internal state[page:1][page:2]. They also agree that the best path for a serious training pipeline is not arbitrary game scraping, but a controlled renderer or engine where the same DLSS-style inputs can be exported deterministically alongside high-resolution targets.

Another point of agreement is that “DLSS inputs” should not automatically be equated with a full deferred-renderer G-buffer. For standard Super Resolution, the common minimum set is low-resolution color, depth, motion vectors, and per-frame metadata such as exposure, jitter, matrices, and scaling constants[web:26][web:49][page:1]. A fuller set of normals, albedo, roughness, and similar material channels is more characteristic of Ray Reconstruction or custom engine-side exports[web:29].

### Main differences between the answers

The biggest difference is emphasis. GPT and Claude both stress that **ground-truth alignment** is the real hard part, not interception. Extracting buffers from a shipped game is workable, but producing clean supervised pairs from that game is often non-deterministic because of TAA history, streaming, particles, LOD changes, and other frame-to-frame sources of mismatch. Gemini is more optimistic about using DLL-proxy techniques directly and points toward the existing modding ecosystem as a practical starting point.

They also differ slightly on preferred interception strategy:

- GPT favors the **Streamline tagging boundary** (`slSetTagForFrame`, constants) because it preserves semantics and avoids guessing buffer identity[page:1].
- Claude favors either **engine-side export** or a **custom Streamline feature** layered into the same path[web:56][web:57].
- Gemini leans toward **DLL proxying / OptiScaler-style interception** as the quickest way to get moving[web:83][web:143].

These are not contradictory. They mainly reflect different trade-offs between cleanliness, speed of implementation, and robustness across titles.

### Final recommendation

For a Gaussian-splatting-based upscaler, the best architecture is a **two-track data plan**:

1. **Primary supervised dataset:** Generate in a renderer you control, exporting low-resolution color, depth, motion vectors, jitter, exposure, and optionally normals/albedo/roughness, paired with deterministic high-resolution targets.
2. **Secondary real-game inspection dataset:** Intercept Streamline or NGX in DLSS-enabled single-player titles to inspect real-world motion-vector failures, HUD contamination, depth quirks, dynamic-resolution behavior, and other production anomalies.

That gives the cleanest training path and the most realistic validation path. The most important design decision is to intercept the **DLSS integration boundary**, not to treat “extracting the full game renderer state” as the main problem. The game is already handing DLSS the buffers that matter; the job is to copy, decode, canonicalize, and pair them with useful targets.[page:1][page:2]
