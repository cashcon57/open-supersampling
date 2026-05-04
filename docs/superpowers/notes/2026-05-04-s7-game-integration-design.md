# Sprint 7 — Game Integration Design Memo

**Date:** 2026-05-04
**Status:** Design note only. Sprint 7 is post-S6 (perf pass). No code, no integration stubs yet.
**Owner:** SR / runtime / game-integration
**Related:**
- `oss/sr/temporal/stateless_export.py::TemporalSRModelStateless` (5-input ONNX-exportable wrapper)
- `docs/superpowers/notes/2026-05-04-v5-pixel-temporal-onnx-export-design.md` (runtime contract)
- `docs/superpowers/notes/vendor-optimization-audit.md` (vendor matrix-accel ground truth)
- `docs/superpowers/notes/cuda-mega-kernel-design.md` (S6 fused-kernel target)

## Scope and Goals

Sprint 7 ships OSS-SR as a *swap-in* upscaler for games that already integrated DLSS / FSR / XeSS, plus a frame-extrapolation byproduct from the v5 temporal warp infrastructure. Four interception surfaces:

1. **Windows DXGI + NGX shim** — DLSS-API-compatible swap-in for any DX12 game shipping DLSS 2/3.
2. **Vulkan layer** — Linux + Steam Deck (Proton) interception via the Vulkan layer mechanism.
3. **Metal frame interception** — Apple Silicon native + CrossOver-translated D3D11/12 games.
4. **OSS-FX α-conditioned frame extrapolation** — derived from v5 temporal warp, intermediate-frame synthesis at fractional time α ∈ [0, 1].

The runtime contract for each surface is the same: feed the stateless wrapper its 5 explicit inputs (`lr_inputs`, `prev_hr`, `depth_hr_curr`, `depth_hr_prev`, `motion_lr`) and consume `(out_hr, disocclusion)`. The integration layer owns `prev_hr` lifetime, exactly as specified in the ONNX export memo.

## 1. DXGI Hook + NGX Shim (Windows, Primary Target)

### Goal

Runtime swap-in for any DX12 game that already calls `NVSDK_NGX_D3D12_EvaluateFeature_C` for DLSS upscaling. The game ships unaware of OSS; OSS provides the upscale by registering itself as the NGX feature provider for `NVSDK_NGX_Feature_SuperSampling`.

### Hook Surface

The actual NGX entry point we intercept is:

```
NVSDK_NGX_Result NVSDK_NGX_D3D12_EvaluateFeature_C(
    ID3D12GraphicsCommandList*  InCmdList,
    const NVSDK_NGX_Handle*     InFeatureHandle,
    const NVSDK_NGX_Parameter*  InParameters,
    PFN_NVSDK_NGX_ProgressCallback InCallback
);
```

The DLSS-relevant parameters are read out of `InParameters` by string key (NGX is a key/value parameter dictionary, not a typed struct). The mapping to the stateless wrapper's 5 inputs:

| NGX parameter key | NGX semantics | Stateless wrapper input | Notes |
|---|---|---|---|
| `NVSDK_NGX_Parameter_Color` (`ID3D12Resource*`) | Game-rendered LR color (linear, post-tonemap depending on game) | `lr_inputs[:, 0:3]` | RGB channels; OSS expects sRGB-linear. Tonemapping placement is per-game; documented quirks below. |
| `NVSDK_NGX_Parameter_Depth` (`ID3D12Resource*`) | LR depth buffer | `lr_inputs[:, 3:4]` (LR depth slot) | We bilinear-upsample to HR for `depth_hr_curr`; the LR depth also goes into the LR feature stack. |
| `NVSDK_NGX_Parameter_MotionVectors` (`ID3D12Resource*`) | LR motion vectors, NDC scale | `motion_lr` | OSS expects pixel-space motion at LR resolution. Convert NDC → pixels using `NVSDK_NGX_Parameter_MV_Scale_X/Y`. |
| `NVSDK_NGX_Parameter_Output` (`ID3D12Resource*`) | HR output target | `out_hr` destination | OSS writes here via `CopyTextureRegion` from the OSS-owned HR ring buffer. |
| `NVSDK_NGX_Parameter_DLSS_Exposure_Texture` | Optional 1×1 exposure scalar | (ignored for v5) | v5 does not condition on exposure; hold for future tonemap-aware variant. |
| `NVSDK_NGX_Parameter_Reset` (uint) | Scene-cut signal | First-frame reset of `prev_hr` | This is the host-supplied scene-cut signal mentioned in the export memo §3 — when set, OSS calls `make_first_frame_prev_hr(lr_rgb, scale)` and seeds `depth_hr_prev = depth_hr_curr`. |
| `NVSDK_NGX_Parameter_Jitter_Offset_X/Y` (float) | TAA sub-pixel jitter | (ignored for v5) | v5 was not trained against jittered LR. Future work; see open questions. |
| `NVSDK_NGX_Parameter_PerfQualityValue` (int) | DLSS quality preset (Quality / Balanced / Performance / UltraPerf) | Selects scale factor | Maps to OSS internal `scale` (2.0× / 2.3× / 2.5× / 3.0×). v5 trained at scale=2; other ratios use the closest trained ratio with bilinear pre-resize. |

### Resource Lifetime

Game-side D3D12 resources passed through `InParameters` are NGX-owned and only valid for the duration of the `EvaluateFeature` call. OSS must:

1. **Copy in:** `CopyTextureRegion` from each NGX input resource into OSS-owned staging textures on the same `ID3D12CommandList`. No UAV barriers held past the call.
2. **Run inference:** dispatch the OSS compute graph (TRT engine in S5/S6, fused mega-kernel in S6) on a separate OSS-owned command list / queue, using OSS-owned UAV buffers.
3. **Copy out:** `CopyTextureRegion` from the OSS HR ring buffer into the NGX output target before returning from `EvaluateFeature`.
4. **`prev_hr` ownership:** OSS retains a per-swap-chain HR ring (2-deep is sufficient — current and previous HR). Lifetime is bound to swap-chain creation/resize/destroy.

This matches the export memo's "Option A: game-engine integration layer owns it" decision: the integration DLL is the host, and `prev_hr` lives in OSS-owned UAV memory tied to swap-chain identity.

### Validation Target

**Cyberpunk 2077.** No anti-cheat (offline / single-player), well-documented hook patterns (multiple community modding frameworks already inject DXGI hooks), ships DLSS 2. We can validate end-to-end without anti-cheat false positives. Secondary candidates (only if needed): *The Witcher 3 Next-Gen* (same engine class, DLSS 3), *Control* (early DLSS adopter, simple render graph).

### Distribution Strategy

Ship as **a single DLL** (e.g., `oss_ngx_shim.dll`) plus a TRT engine blob and ONNX fallback. The DLL:

1. Hooks DXGI swap-chain creation (`CreateDXGIFactory*` → `CreateSwapChain*`) to discover the swap-chain handle and resolution. Method: MinHook or Detours-style trampoline patching of the IAT or VTable. We use whichever is permitted by the validation game's loader.
2. Registers itself as the NGX feature provider via the standard NGX feature-discovery path. Practical implementation: ship a `nvngx.dll` in the game directory that forwards everything *except* `NVSDK_NGX_D3D12_EvaluateFeature_C` for `Feature_SuperSampling` to the real NVIDIA `nvngx.dll` in `System32`. This is a known DLL-search-order pattern (game directory wins over System32).
3. Loads the OSS TRT engine at first-frame and reuses it across the swap-chain lifetime.

**We do NOT ship a modified NGX SDK.** The shim DLL implements the public NGX C ABI; we link against the redistributable headers, not the SDK source.

### Risks — Three Known DLSS API Gotchas

1. **NGX SDK version drift (NGX 2.x vs 3.x parameter keys).** DLSS 2 and DLSS 3 (Frame Generation) share the parameter dictionary but DLSS 3 adds keys for optical-flow inputs that don't exist in DLSS 2 builds. The shim must tolerate missing keys (game built against DLSS 2 SDK won't supply DLSS 3 keys) and missing values (game supplies key with a NULL resource pointer). Treat absent keys as "feature not in use," not as an error.
2. **Per-driver NGX implementation quirks.** NVIDIA driver releases occasionally change the internal ordering of NGX initialization calls (`NVSDK_NGX_D3D12_Init` vs `NVSDK_NGX_D3D12_AllocateParameters`). The shim must implement *all* of `Init`, `Shutdown`, `AllocateParameters`, `DestroyParameters`, `CreateFeature`, `ReleaseFeature`, and `EvaluateFeature` as a contiguous group, even though we only do real work in `CreateFeature` and `EvaluateFeature`. Forwarding the others to the real `nvngx.dll` is the safest pattern.
3. **MV_Scale and motion-vector orientation.** Different games supply motion vectors in different conventions: NDC vs pixels, "previous-to-current" vs "current-to-previous", Y-up vs Y-down. The NGX parameter dictionary exposes `NVSDK_NGX_Parameter_MV_Scale_X/Y` to communicate scale, but **not** orientation. v5 was trained on pixel-space, current-to-previous (i.e. "where did this pixel come from"), Y-down motion. The shim needs a per-game override table. Cyberpunk's convention is documented in the modding community; verify on first integration.

## 2. Vulkan Layer (Linux, Steam Deck Proton)

### Goal

Intercept any Vulkan game performing DLSS-style upscaling. On Steam Deck specifically: intercept upscale dispatches issued by Proton-translated DXVK/VKD3D-Proton games or by native Linux Vulkan titles using `VK_NV_low_latency2` / DLSS-Vulkan extensions.

### Layer Manifest

A standard JSON manifest installed under `$XDG_DATA_HOME/vulkan/implicit_layer.d/oss_upscale.json`:

```
{
  "file_format_version": "1.2.0",
  "layer": {
    "name": "VK_LAYER_OSS_upscale",
    "type": "GLOBAL",
    "library_path": "liboss_vk_layer.so",
    "api_version": "1.3.250",
    "implementation_version": 1,
    "description": "OpenSuperSampling Vulkan upscale layer",
    "functions": {
      "vkGetInstanceProcAddr": "oss_GetInstanceProcAddr",
      "vkGetDeviceProcAddr":   "oss_GetDeviceProcAddr"
    },
    "enable_environment": { "OSS_VK_LAYER_ENABLE": "1" },
    "disable_environment": { "OSS_VK_LAYER_DISABLE": "1" }
  }
}
```

Implicit layer means it auto-loads when `OSS_VK_LAYER_ENABLE=1` is set in the game's environment (Proton's `user_settings.py` or a Steam launch option).

### Interception Points

- `vkCreateInstance` / `vkCreateDevice` — chain initialization, feature/extension query.
- `vkCreateSwapchainKHR` — discover render-target resolution, format, color space. Track the swap-chain handle as our integration identity (analogue of the DXGI swap-chain in §1).
- `vkDestroySwapchainKHR` — release per-swap-chain resources (`prev_hr`, OSS engine).
- `vkQueueSubmit` — last point of interception before frame goes to compositor; this is where post-process upscale could be injected as an OSS-owned secondary command buffer.

### The "No NGX" Problem

Vulkan has no equivalent of NGX as a single API surface. DLSS-on-Vulkan exists but is exposed via vendor-specific extensions and is not present on Steam Deck (RDNA 2). FSR 2/3 ships as in-game source code, not an interceptable API. **There is no single function call that says "upscale this".**

Two architectural options:

**Option A — Replace the game's compute upscale dispatch.**
Detect the game's own upscale pass (FSR 2 source compiled into the game's pipelines) by shader-hash matching on `vkCreateComputePipelines` / `vkCreateGraphicsPipelines`. When a known FSR/DLSS shader hash is bound, swap in an OSS pipeline instead.
- **Pro:** Zero render-graph changes; the upscaled output lands in the same texture the game expected.
- **Con:** Brittle. Shader hashes change with every game patch. We'd ship a hash database keyed by build IDs; not maintainable for more than a few titles.

**Option B — Add an OSS post-process pass.**
Let the game render at LR all the way through. After the game's last present-blit but before `vkQueuePresentKHR`, inject an OSS compute dispatch that reads the LR swap-chain image, runs OSS upscale, and writes back to a new HR swap-chain we substituted at `vkCreateSwapchainKHR`.
- **Pro:** No shader-hash matching, works on any game without per-title patches, only needs swap-chain interception.
- **Con:** We lose access to native depth and motion buffers (post-tonemap, post-UI). Quality will be materially worse than NGX-style integration because the temporal head needs depth + motion *before* UI compositing.
- **Mitigation:** Sniff `vkCmdBindDescriptorSets` for descriptor sets containing `VK_FORMAT_D32_SFLOAT` / `VK_FORMAT_D24_UNORM_S8_UINT` images and `VK_FORMAT_R16G16_SFLOAT` motion-vector textures. Cache the most recently bound depth and MV textures per command buffer; assume the game's last-bound depth/MV before its tonemap pass are the right inputs. Heuristic, but tractable.

**Recommendation: Option B with the depth/MV sniffing mitigation.** Option A's per-title fragility is incompatible with a "ship one layer, works on many games" goal. Option B's quality penalty is acceptable for the Steam Deck tier (Pico model anyway; see below).

### Steam Deck Specifics

Per `vendor-optimization-audit.md`: RDNA 2 has **no matrix accelerator path comparable to RDNA 3 WMMA or CDNA MFMA**. The OSS Vulkan path on Steam Deck must:

- Run the **Pico tier** (~150K-param distilled model, per `2026-05-04-pico-distillation-design.md`) instead of the standard 626K-param v5.
- Use **`VK_KHR_cooperative_matrix`** only when the device advertises it (RDNA 3+, NVIDIA Turing+, Intel Arc). On RDNA 2 / Steam Deck this extension is absent — fall back to compute shaders using packed FP16 dot products (`VK_KHR_shader_float16_int8`, which the Steam Deck does support).
- Budget: the audit pins Steam Deck at "~25% of matrix-equipped vendor peak until measured otherwise." For a 150K-param model at 720p → 1280p (the Deck's native screen), this should still hit 60 fps, but the perf gate is the S6 measurement.

### Trickier Than DXGI

Two extra complications relative to the Windows path:

1. **No first-class scene-cut signal.** NGX has `Reset`. Vulkan upscale extensions vary; in Option B we never see the game's own scene-cut state. The layer falls back to the motion-magnitude heuristic from `TemporalSRInferenceEngine` (`scene_cut_motion_threshold=32.0`).
2. **DXVK / VKD3D-Proton frame timing.** Proton-translated D3D12 games go DX12 → VKD3D → Vulkan. Some draws appear out of source order in the resulting Vulkan stream. This is benign for end-of-frame post-process injection (Option B) but would break Option A's shader-hash matching.

## 3. Metal Frame Interception (macOS, CrossOver Target)

### Goal

Apple Silicon support for two scenarios:

1. Native macOS games using Metal directly.
2. D3D11/12 games running under CrossOver / Game Porting Toolkit (which translates D3D → Metal).

### MPSGraph Operator Surface

`MPSGraph` (Metal Performance Shaders Graph) exposes a high-level op graph that includes `resampleTensor:` for upscaling. Native macOS games using MetalFX upscaling (Apple's first-party temporal/spatial upscaler) call `MTLFXTemporalScalerDescriptor` → `MTLFXTemporalScaler.encodeToCommandBuffer:`. Hook surface:

- **Method swizzling** on `MTLFXTemporalScaler` to intercept `encodeToCommandBuffer:`. Objective-C runtime supports this without modifying the Metal binary.
- Read input/output `MTLTexture` references, the depth + motion textures from the temporal scaler descriptor, and the jitter offset.
- Replace the encoded work with OSS Metal compute dispatches (Pico or standard tier depending on GPU family detected via `MTLDevice.supportsFamily:`).

### CrossOver / GPTK Layer

CrossOver and Apple's Game Porting Toolkit translate D3D11/12 calls to Metal at the framework level. DLSS calls in the original D3D game become NGX calls into a stub `nvngx.dll` (no NVIDIA hardware to back them). CrossOver's translation layer either ignores DLSS entirely (game falls back to native upscale) or, in newer builds, routes to MetalFX.

Two integration points for OSS:

1. **OSS as the MetalFX replacement** — same swizzling approach as above; OSS executes instead of MetalFX once CrossOver routes DLSS → MetalFX.
2. **OSS as a CrossOver upscale plug-in** — if CrossOver exposes a plug-in interface for upscaler swap (this is unconfirmed and would require coordination with CodeWeavers). Cleaner but depends on third-party API availability.

The S7 plan starts with method-swizzling (option 1) because it doesn't depend on CrossOver internals.

### Apple Silicon Caveats

Per `vendor-optimization-audit.md` Apple section, plus the practical implications:

- **ANE is FP16-only and not generally programmable as a custom shader target.** Core ML can dispatch to ANE but only for whole models compiled through `coremltools`. Custom compute kernels run on GPU, not ANE. OSS-SR via Core ML is a possible path; OSS-SR via Metal compute is the controllable path.
- **AMX is CPU-side.** Useful for small CPU-resident matmuls but not for the inference forward pass; ignore for S7.
- **TBDR (Tile-Based Deferred Rendering) applies to render passes, not compute dispatches.** Our upscale is a compute dispatch, so TBDR tile memory is *not* automatically available; we use threadgroup memory explicitly via Metal Shading Language (`threadgroup` qualifier), not the tile-shading API.
- **Device family gating.** Metal Feature Set Tables differ across M1 / M2 / M3 / M4 families. SIMD-scoped matrix multiply (`simdgroup_matrix`) is M1+ but operand types and tile shapes vary. The Metal port must query `MTLDevice.supportsFamily(.apple7)` and pick the kernel variant.
- **Memory model.** Apple Silicon is unified memory, so `MTLStorageMode.shared` is essentially free for inputs/outputs that the game also touches on CPU. `prev_hr` should still be `MTLStorageMode.private` (GPU-only) for cache locality.

This section is **partial pending C5 (vendor opt audit) completion for Apple specifics.** When Codex finishes the audit, cross-reference the device-family table against MetalFX hook validation results.

## 4. OSS-FX α-Conditioned Frame Extrapolation

### Goal

Generate a synthetic frame at fractional time α ∈ [0, 1] between two real frames, using the v5 pixel-temporal warp infrastructure. Doubles or triples effective frame rate at the cost of per-extrapolated-frame inference latency.

### Architecture

Reuse `oss.sr.temporal.warp_prev_hr(prev_hr, motion_lr, scale)` but interpolate the motion field by α:

```
motion_lr_alpha   = α * motion_lr_curr
prev_hr_alpha     = warp_prev_hr(prev_hr, motion_lr_alpha, scale)
depth_hr_alpha    = (1 - α) * depth_hr_prev + α * depth_hr_curr
disocclusion_alpha = compute_synthetic_disocclusion(depth_hr_alpha, prev_hr_alpha, α)
out_hr_alpha       = temporal_head(prev_hr_alpha, disocclusion_alpha, α_embedding)
```

The temporal head (`oss.sr.temporal.gate` + `head` in v5) consumes the partially-warped prev plus a synthetic disocclusion mask computed from α-scaled depth disparity. α is injected as an embedding (small MLP from scalar α to a per-channel bias added to the gate input), trained jointly.

### Quality Cost

Likely worse than full-frame SR because:

- There is no LR ground-truth at the intermediate time α. The model is fully synthesizing pixels from prev + warp + α.
- Disocclusions at α are heuristic (depth-based), not data-supported.

**Mitigation:** train OSS-FX as a *separate distilled model* with α as an explicit input, not as a v5 head extension. The distillation target can be a high-cost teacher that interpolates between two real frames using optical-flow ground truth from the synthetic dataset (UE5 path-traced sequences with known per-pixel motion across sub-frame intervals — same dataset family as the v5 training data).

### Latency Budget

If standard-tier SR is **15 ms at 1080p → 4K** on RTX 3080 Ti, frame extrapolation at α = 0.5 doubles effective frame rate at the cost of 15 ms extra per generated frame. Worked example:

- Native 60 fps render → OSS-FX inserts one frame at α=0.5 → effective 120 fps at the cost of 15 ms extra latency per real frame.
- Net latency seen by the player: 16.6 ms (real frame) + 15 ms (extrapolated frame) = 31.6 ms wall-clock for two displayed frames vs. 33.3 ms for two native frames. Latency wins; perceived smoothness wins more.

For 30 fps → 60 fps extrapolation (α=0.5 between two 33.3 ms frames), the extrapolated frame delivers at 16.6 ms after the prior real frame, well within the 33.3 ms native frame interval. This is the marketed mode.

### Open Question — Inputs at α

Extrapolation at α ∈ (0, 1) needs `depth` and `motion` at the intermediate time. **These don't exist in the game's real frame stream** — the engine only renders depth/motion at frame N and frame N+1, not at N + α.

**Working assumption:** linear interpolation of depth and motion across the (N, N+1) interval is good enough for small α. For α near 0.5, this is plausible (motion is locally smooth in most game scenes); for α near 0 or 1 it degenerates to "use the nearest real frame's depth/motion." For multi-frame extrapolation (insert two synthetic frames at α=1/3 and α=2/3), the linearity assumption gets progressively shakier.

**Validation plan:** measure PSNR / LPIPS of OSS-FX output against the held-out synthetic dataset's intermediate frames (which have ground-truth depth + motion at every sub-frame). If linear interpolation degrades quality past a threshold, fall back to single-frame extrapolation only.

### Reuse vs. Fork

OSS-FX shares ~80% of v5 temporal infrastructure (`warp_prev_hr`, `gate`, `head`) but differs in:

- α-conditioning embedding (new module).
- Loss function (no LR-ground-truth term; only HR perceptual loss against teacher).
- Training data (sub-frame interpolation pairs, not (LR, HR) pairs).

S7 deliverable: design memo + training scaffold. Production OSS-FX model is post-S7.

## 5. Integration Order Recommendation

**DXGI/NGX → Vulkan layer → Metal.** Rationale:

1. **DXGI/NGX first.** Largest TAM (Windows DX12 games shipping DLSS 2/3 number in the hundreds and include the highest-revenue titles). Well-defined hook surface (single NGX entry point, well-documented parameter dictionary). Single validation target (Cyberpunk 2077). Lowest engineering risk.
2. **Vulkan layer second.** Steam Deck unlocks ~10M units of installed base + Linux desktop. Layer mechanism is standardized but the "no NGX equivalent" problem makes the integration genuinely harder than Windows. Pico-tier model dependency means Vulkan can't ship until S5 distill (C10) is done.
3. **Metal last.** Smallest user base for AAA games (most macOS gaming is via CrossOver / GPTK, not native). But: no other ML upscaler currently ships there, so OSS could be the only option. Metal port is also a brand statement for Apple Silicon coverage. Last on the list, not because it's unimportant, but because the Windows + Steam Deck deliverables are higher-leverage per engineering hour.

**OSS-FX α-extrapolation slots in parallel** with whichever surface is most mature at the time. Easiest to validate on Windows (real DLSS 3 games to A/B against) but not blocking any of the three platform integrations.

## Open Questions Summarized

1. **Jitter handling.** v5 was not trained against TAA-jittered LR. NGX always supplies jitter via `Jitter_Offset_X/Y`. Do we de-jitter in the shim (warp LR by negative jitter before inference) or retrain v5 with jitter augmentation?
2. **Per-game motion-vector convention table.** Where does this live and who maintains it? Likely a JSON config shipped with the DLL, community-extensible.
3. **Vulkan Option A fallback.** If Option B's depth/MV-sniffing heuristic underperforms, do we maintain a per-title shader-hash database (Option A)? This is a maintenance commitment.
4. **OSS-FX teacher model.** What architecture is the high-cost interpolation teacher? RAFT + SR + disocclusion-aware blender, or a published frame-interpolation model (FILM, RIFE, Praktical-RIFE)?
5. **CrossOver plug-in API.** Does CodeWeavers expose an upscale plug-in interface? If yes, that's a cleaner Metal path than method-swizzling MetalFX.
6. **NGX 4.x.** Future NGX revisions may change parameter keys or add typed structs. Forward-compat strategy?
7. **Anti-cheat.** All validation targets are AC-free. What is the path to AC-protected titles (BattlEye, EAC) that ship DLSS? This is post-S7.

## Constraints and Non-Goals for S7 Design Phase

- Docs only. No code, no integration stubs.
- Pixel training (`v0.2-dev` on `<train-host>`) and Codex's parallel work (C5/C6/C8/C9–C12) are untouched.
- OSS-FX is design-only; no training launch from this memo.
- Anti-cheat-protected titles are out of scope for S7.
- Mobile (iOS / Android) is out of scope for S7; revisit if/when Apple Silicon Metal port lands.
