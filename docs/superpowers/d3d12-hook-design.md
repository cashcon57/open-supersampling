# OSS-Gaussian: D3D12 Hook Design (Cyberpunk 2077, RED Engine 4)

**Sprint 2 target:** Windows / RTX 3080 Ti, Cyberpunk 2077 (DX12). Replace DLSS, harvest G-buffer for our neural upscaler, run side-by-side comparison vs native DLSS.

---

## 1. Recommended hook architecture

**One-line decision: ship a renamed `nvngx.dll` proxy that does (a) NGX API spoofing for DLSS replacement and (b) a Detours/MinHook DXGI+D3D12 inline hook for G-buffer capture, in a single in-process module.**

Rejected alternatives: pure `nvngx_dlss.dll` swap exposes only `EvaluateFeature` (no back-buffer access for compare overlay, no Frame-Gen path); pure DXGI proxy ends up needing NGX spoofing too. The combined approach is OptiScaler's production-proven pattern; for Cyberpunk it recommends `dxgi.dll` or `wininet.dll` ([OptiScaler wiki](https://github.com/optiscaler/OptiScaler/wiki/Cyberpunk-2077)).

**Concrete plan:** ship `oss_gaussian.dll` renamed to `dxgi.dll` in `Cyberpunk 2077\bin\x64\`. On `DllMain`: (1) load real system `dxgi.dll`, forward unrelated exports; (2) Detours-hook `D3D12CreateDevice`, `IDXGIFactory*::CreateSwapChain*`, `IDXGISwapChain::Present`, `ID3D12CommandQueue::ExecuteCommandLists`; (3) intercept the game's `LoadLibrary("nvngx_dlss.dll")` and return our module so our `NVSDK_NGX_D3D12_*` exports get the calls. Mirrors OptiScaler's "hooks factory creation, DXGI, D3D11, and D3D12 API calls" stack ([DeepWiki](https://deepwiki.com/optiscaler/OptiScaler/3.2-vulkan-hooking)).

## 2. DLL filenames + on-disk locations (Cyberpunk 2077)

Game install root example: `C:\Program Files (x86)\Steam\steamapps\common\Cyberpunk 2077\`

| File | Path (relative to game root) | Role |
| --- | --- | --- |
| `Cyberpunk2077.exe` | `bin\x64\` | Main executable, loads NGX |
| `nvngx_dlss.dll` | `bin\x64\` | DLSS Super Resolution (the one we replace/spoof) |
| `nvngx_dlssg.dll` | `bin\x64\` | DLSS Frame Generation (out of scope Sprint 2) |
| `nvngx_dlssd.dll` | `bin\x64\` (when RR enabled) | DLSS Ray Reconstruction (out of scope Sprint 2) |
| `dxgi.dll` (ours) | `bin\x64\` | Our proxy. Game-local DLL search wins over system32 |

DLSS file install path is confirmed: `Cyberpunk 2077\bin\x64\nvngx_dlss.dll` ([TechPowerUp DLSS DLL](https://www.techpowerup.com/download/nvidia-dlss-dll/), [PotatoOfDoom/CyberFSR2](https://github.com/PotatoOfDoom/CyberFSR2)). Always rename original to `nvngx_dlss.dll.bak` before deploy.

## 3. NGX function signatures we must implement

These are the D3D12 entry points exported by `nvngx_dlss.dll` that any DLSS-integrated game (Cyberpunk included) calls. Reproducing them is what makes DLSS substitution invisible to the game ([NVIDIA/DLSS headers](https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx.h), [NGX Programming Guide](https://docs.nvidia.com/ngx/ngx-archived/ngx-100/programming-guide/index.html)).

```cpp
NVSDK_NGX_Result NVSDK_NGX_D3D12_Init(
    unsigned long long InApplicationId, const wchar_t* InApplicationDataPath,
    ID3D12Device* InDevice, NVSDK_NGX_Version InSDKVersion);

NVSDK_NGX_Result NVSDK_NGX_D3D12_Init_Ext(/* + InFeatureInfo, InParameters */);

NVSDK_NGX_Result NVSDK_NGX_D3D12_Shutdown1(ID3D12Device* InDevice);

NVSDK_NGX_Result NVSDK_NGX_D3D12_GetCapabilityParameters(NVSDK_NGX_Parameter** OutParameters);
NVSDK_NGX_Result NVSDK_NGX_D3D12_AllocateParameters(NVSDK_NGX_Parameter** OutParameters);
NVSDK_NGX_Result NVSDK_NGX_D3D12_DestroyParameters(NVSDK_NGX_Parameter* InParameters);

NVSDK_NGX_Result NVSDK_NGX_D3D12_CreateFeature(
    ID3D12GraphicsCommandList* InCmdList, NVSDK_NGX_Feature InFeatureID,
    NVSDK_NGX_Parameter* InParameters, NVSDK_NGX_Handle** OutHandle);

NVSDK_NGX_Result NVSDK_NGX_D3D12_EvaluateFeature(
    ID3D12GraphicsCommandList* InCmdList, const NVSDK_NGX_Handle* InFeatureHandle,
    const NVSDK_NGX_Parameter* InParameters,
    PFN_NVSDK_NGX_ProgressCallback InCallback);

NVSDK_NGX_Result NVSDK_NGX_D3D12_ReleaseFeature(NVSDK_NGX_Handle* InHandle);
```

The interesting one is `EvaluateFeature` — that is where Cyberpunk hands us, via the `NVSDK_NGX_Parameter` key/value bag, **everything we need**: `Color`, `Output`, `Depth`, `MotionVectors`, `Jitter.Offset.{X,Y}`, `MV.Scale.{X,Y}`, `Exposure`, `Reset`, `Subrect.*`, plus the flag `NVSDK_NGX_DLSS_Feature_Flags_MVJittered` ([NVIDIA forum reference](https://forums.developer.nvidia.com/t/motion-vector-flag-nvsdk-ngx-dlss-feature-flags-mvjittered-seems-to-work-abnormally/294576)). Parameter keys are string-named in `nvsdk_ngx_defs.h`. **All G-buffer extraction for the upscaler runs through this single function** — we don't even need to scrape DXGI for it.

## 4. G-buffer extraction approach

Two-layer strategy:

**Layer A — NGX param scrape (primary, free):** Inside our spoofed `NVSDK_NGX_D3D12_EvaluateFeature` we read these `NVSDK_NGX_Parameter` keys and copy the resources we need:

- `NVSDK_NGX_Parameter_Color` → input low-res HDR color (`ID3D12Resource*`)
- `NVSDK_NGX_Parameter_Depth` → linear/non-linear depth (game tells us which via flags)
- `NVSDK_NGX_Parameter_MotionVectors` → 2-channel float MV, scaled by `MV.Scale.{X,Y}`
- `NVSDK_NGX_Parameter_Jitter_Offset_{X,Y}` → sub-pixel jitter for TAA
- `NVSDK_NGX_Parameter_Exposure_Texture` → optional autoexposure
- `NVSDK_NGX_Parameter_Output` → target we must write upscaled result to
- `NVSDK_NGX_Parameter_Reset` → camera-cut signal

This is the production path. OptiScaler does exactly this for FSR/XeSS substitution: "Input is the upscaler used in game settings, and Output the one selected in Opti Overlay" — pass-through from the same param dict ([OptiScaler README](https://github.com/optiscaler/OptiScaler)). Implement an internal copy queue + readback heaps for offline dumps; for live upscale we route the resources straight to our compute pipeline.

**Layer B — DXGI/D3D12 swapchain scrape (secondary, for back-buffer + final-frame compare):** Hook `IDXGISwapChain::Present` via vtable patch on the swapchain returned from `CreateSwapChainForHwnd`. This is the standard ReShade pattern ([reshade/source/dxgi/dxgi_swapchain.cpp](https://github.com/crosire/reshade/blob/main/source/dxgi/dxgi_swapchain.cpp), [UniversalHookX D3D12 backend](https://github.com/bruhmoment21/UniversalHookX/blob/main/UniversalHookX/src/hooks/backend/dx12/hook_directx12.cpp)). For D3D12 we additionally hook `ID3D12CommandQueue::ExecuteCommandLists` so we can fence and snapshot the post-Present back-buffer to a CPU readback heap for the side-by-side comparison overlay.

We do **not** try to identify motion-vector RTs by scraping `OMSetRenderTargets` / `Resource::SetName` — that path is fragile across RED Engine patches. Layer A makes it unnecessary.

## 5. Reference implementations to study

Read in this order:

1. **OptiScaler** — `OptiScaler/Hooks/HooksDx.cpp`, `OptiScaler/NVNGX_DLSS.cpp`, `OptiScaler/Inputs/DLSSFeatureDx12.cpp`, `OptiScaler/Inputs/FSR2FeatureDx12.cpp`. This is the closest analog to what we are building. Repo: <https://github.com/optiscaler/OptiScaler>. Cyberpunk-specific notes: <https://github.com/optiscaler/OptiScaler/wiki/Cyberpunk-2077>.
2. **PotatoOfDoom/CyberFSR2** — original Cyberpunk-targeted DLSS→FSR2 spoof; smaller, easier to read first. <https://github.com/PotatoOfDoom/CyberFSR2>.
3. **NVIDIA/DLSS public SDK** — `include/nvsdk_ngx.h`, `nvsdk_ngx_defs.h`, `nvsdk_ngx_helpers.h`, `nvsdk_ngx_params.h`. <https://github.com/NVIDIA/DLSS>.
4. **ReShade** — `source/dxgi/dxgi_swapchain.cpp`, `source/d3d12/d3d12_command_queue.cpp`. Reference for swapchain proxying done right. <https://github.com/crosire/reshade>.
5. **Nukem9/dlssg-to-fsr3** — production example of replacing `nvngx_dlssg.dll` (Frame Gen). Useful when we move past Sprint 2. <https://github.com/Nukem9/dlssg-to-fsr3>.
6. **NVIDIA-RTX/Streamline** — official DLSS integration reference, shows expected param layouts. <https://github.com/NVIDIA-RTX/Streamline>.

## 6. Risks

- **Anti-cheat: low risk for this title.** Cyberpunk 2077 is single-player and ships **without** EasyAntiCheat / BattlEye / Denuvo Anti-Cheat. The game has had Denuvo DRM stripped (DRM-free distribution). OptiScaler, CyberFSR2, and DLSS Enabler have all been used widely without bans. Standard mod-warning still applies for any future multiplayer addition. ([OptiScaler README warning](https://github.com/optiscaler/OptiScaler), [Steam guide](https://steamcommunity.com/sharedfiles/filedetails/?id=3363435021)).
- **Version drift.** RED Engine 4 + DLSS DLL upgrades happen on every patch (currently the DLSS family is in the **3.7+ / 310.x** range as of recent patches — [TechPowerUp DLSS 310.6.0](https://www.techpowerup.com/download/nvidia-dlss-dll/), [DSOGaming DLSS 3.7](https://www.dsogaming.com/videotrailer-news/nvidia-dlss-3-7-brings-visual-improvements-to-cyberpunk-2077/)). NGX param keys are stable; DLL exports are stable; but new flags (Ray Reconstruction, MVJittered behavior) appear. Pin the DLSS SDK header version we target (4.x SDK headers) and assert on unknown flags rather than silently ignoring.
- **Path Tracing path forces DLSS-RR (`nvngx_dlssd.dll`).** OptiScaler's wiki flags "the game tries to force DLSS RR, leading to a noisy image" on AMD/Intel ([Cyberpunk wiki page](https://github.com/optiscaler/OptiScaler/wiki/Cyberpunk-2077)). Sprint 2 should default to **PT off, RT Reflections OK, DLSS SR mode** to keep the input set clean. Disable DLSS-RR via param spoofing if we detect it.
- **D3D12 ownership/state transitions.** Resources arriving in `EvaluateFeature` are in NGX's expected state (`UNORDERED_ACCESS` for output, `NON_PIXEL_SHADER_RESOURCE` for inputs). We must transition back to whatever the game expects on exit, or the next frame's barriers assert ([MS docs on cross-queue resource ownership](https://learn.microsoft.com/en-us/windows/win32/direct3d12/executing-and-synchronizing-command-lists)).
- **Jitter sign convention.** RED Engine reports jitter in pixels with one specific sign convention; FSR/XeSS use opposite signs in some versions. CyberFSR2 has the empirically correct flip — copy from there rather than re-deriving.
- **Allocator / readback budget.** Each captured frame at 1080p input = ~30 MB across color+depth+MV. At 60 fps that's 1.8 GB/s into a ring of readback heaps. Cap retained frames or stream to NVMe.

## 7. Sprint 2 task list

1. Stand up `oss_gaussian.dll` Visual Studio 2022 project (C++20, /MT, x64). Add Microsoft Detours and `nvsdk_ngx.h` headers.
2. Implement `dxgi.dll` proxy: forward all real exports to the system DLL, detoured `CreateDXGIFactory{,1,2}`.
3. Implement the 10 `NVSDK_NGX_D3D12_*` exports listed in §3 as pass-throughs that log+forward to the real `nvngx_dlss.dll` first; verify Cyberpunk runs unchanged with our DLL in `bin\x64\`.
4. Hook `IDXGISwapChain::Present` and `ID3D12CommandQueue::ExecuteCommandLists` via vtable patch on swapchain creation. Add fence-based back-buffer snapshot path.
5. In `EvaluateFeature`, read all NGX params (Color/Depth/MV/Jitter/Exposure/Output/Reset/MV.Scale/Subrect) and dump first 100 frames to disk as DDS for offline inspection. Verify motion vectors visually match camera motion.
6. Wire G-buffer scrape into our neural upscaler's input pipeline (compute queue, no host roundtrip).
7. Replace DLSS path: bypass real `nvngx_dlss.dll`, run our network on Color+Depth+MV, write to `Output` resource with correct state transitions. A/B toggle via in-game overlay key.
8. Build comparison overlay (split-screen DLSS vs ours) using ImGui DX12 backend hooked through the swapchain.
9. Capture session recorder: time-aligned RGB + depth + MV + ground-truth-DLSS-output dumps for training/eval.
10. Per-frame perf telemetry (CPU + GPU timestamps via `ID3D12QueryHeap`), CSV export.
11. Robustness: handle device removal, swapchain resize, alt-tab, resolution changes, DLSS preset changes mid-session.
12. Packaging: signed `.dll`, install/uninstall script, `nvngx_dlss.dll.bak` rollback, README with the OptiScaler-style proxy-name fallback list (`dxgi.dll` → `winmm.dll` → `version.dll` → `dbghelp.dll`) per [OptiScaler manual install](https://github.com/optiscaler/OptiScaler/wiki/Manual-Installation).

---

## Sources

- [OptiScaler repo](https://github.com/optiscaler/OptiScaler) — substitution architecture, supported APIs, anti-cheat warning.
- [OptiScaler Cyberpunk 2077 wiki](https://github.com/optiscaler/OptiScaler/wiki/Cyberpunk-2077) — `dxgi.dll` / `wininet.dll` proxy names, RR forcing on path tracing.
- [OptiScaler Manual Installation](https://github.com/optiscaler/OptiScaler/wiki/Manual-Installation) — full proxy-DLL fallback list.
- [OptiScaler DirectX/Vulkan hooking — DeepWiki](https://deepwiki.com/optiscaler/OptiScaler/3.2-vulkan-hooking) — Detours-based DXGI/D3D12 hook list.
- [PotatoOfDoom/CyberFSR2](https://github.com/PotatoOfDoom/CyberFSR2) — predecessor; Cyberpunk-specific NGX param handling.
- [NVIDIA/DLSS SDK headers](https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx.h) — public NGX surface.
- [NVIDIA NGX Programming Guide](https://docs.nvidia.com/ngx/ngx-archived/ngx-100/programming-guide/index.html) — Init/CreateFeature/EvaluateFeature contract.
- [NGX SDK Integration on DeepWiki](https://deepwiki.com/NVIDIA/DLSS/4-ngx-sdk-integration) — function-signature reference.
- [NVIDIA forum — MVJittered flag](https://forums.developer.nvidia.com/t/motion-vector-flag-nvsdk-ngx-dlss-feature-flags-mvjittered-seems-to-work-abnormally/294576) — jitter convention notes.
- [TechPowerUp DLSS DLL 310.6.0](https://www.techpowerup.com/download/nvidia-dlss-dll/) — current DLSS DLL version, install path.
- [DSOGaming DLSS 3.7 Cyberpunk](https://www.dsogaming.com/videotrailer-news/nvidia-dlss-3-7-brings-visual-improvements-to-cyberpunk-2077/) — version baseline.
- [ReShade DXGI swapchain](https://github.com/crosire/reshade/blob/main/source/dxgi/dxgi_swapchain.cpp) — production swapchain proxy.
- [UniversalHookX D3D12](https://github.com/bruhmoment21/UniversalHookX/blob/main/UniversalHookX/src/hooks/backend/dx12/hook_directx12.cpp) — minimal D3D12 Present-hook reference.
- [eugen15/directx-present-hook](https://github.com/eugen15/directx-present-hook) — small Present-hook example.
- [Microsoft — ID3D12CommandQueue::ExecuteCommandLists](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12commandqueue-executecommandlists) — execution semantics.
- [Microsoft — D3D12 command list synchronization](https://learn.microsoft.com/en-us/windows/win32/direct3d12/executing-and-synchronizing-command-lists) — resource state ownership.
- [Nukem9/dlssg-to-fsr3](https://github.com/Nukem9/dlssg-to-fsr3) — DLSSG DLL replacement reference.
- [NVIDIA-RTX/Streamline](https://github.com/NVIDIA-RTX/Streamline) — DLSS integration reference.
- [Steam guide — DLSS/FSR upgrades for Cyberpunk](https://steamcommunity.com/sharedfiles/filedetails/?id=3363435021) — community-confirmed safe DLL swap workflow.
