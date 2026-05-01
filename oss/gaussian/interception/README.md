# oss-gaussian-interception

C++ DLL that intercepts D3D12 and NVIDIA NGX (DLSS) calls inside Cyberpunk 2077,
hands G-buffers (color, depth, motion vectors) to the OSS-Gaussian renderer,
and replaces DLSS with our own upscaler.

This is **Sprint 2 scaffolding only**. No hooks are active yet. See
`docs/superpowers/plans/2026-05-01-gaussian-sprint-2-plan.md` for tasks.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

Third-party components retain their own licenses (Microsoft Detours = MIT,
NVIDIA DLSS headers = NVIDIA license). Detours and DLSS headers are vendored
under `third_party/` in Sprint 2 tasks T2.2 / T2.4 — not in the scaffold.

## Layout

```
oss/gaussian/interception/
  CMakeLists.txt
  README.md
  LICENSE
  include/oss_gaussian_interception.h    # public C API for the renderer side
  src/
    dllmain.cpp                          # DLL entry, Detours attach/detach
    ngx_exports.cpp                      # 10 NVSDK_NGX_D3D12_* stubs
    g_buffer_extractor.h/.cpp            # NVSDK_NGX_Parameter dict reader
    log.h/.cpp                           # file logger (%LOCALAPPDATA%\oss-gaussian)
  third_party/
    Detours/.gitkeep                     # vendored in T2.2
```

## Build (3080 Ti box, Windows 11)

Prerequisites:
- Visual Studio 2026 with the "Desktop development with C++" workload
  (gives MSVC v143+, Windows 11 SDK, CMake)
- Either: open the **VS 2026 Native Tools Command Prompt for x64**, or run from
  any shell after invoking `vcvars64.bat`.

Build steps:

```bat
cd %REPO%\oss\gaussian\interception
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Output: `build\Release\dxgi.dll`.

To use clang-cl instead of MSVC:

```bat
cmake -S . -B build-clang -T ClangCL -A x64
cmake --build build-clang --config Release
```

## Verify the build

```bat
dumpbin /exports build\Release\dxgi.dll
```

Expected exports (Sprint 2 scaffold):

```
NVSDK_NGX_D3D12_Init
NVSDK_NGX_D3D12_Init_Ext
NVSDK_NGX_D3D12_Shutdown1
NVSDK_NGX_D3D12_GetCapabilityParameters
NVSDK_NGX_D3D12_AllocateParameters
NVSDK_NGX_D3D12_DestroyParameters
NVSDK_NGX_D3D12_CreateFeature
NVSDK_NGX_D3D12_EvaluateFeature
NVSDK_NGX_D3D12_ReleaseFeature
NVSDK_NGX_D3D12_GetScratchBufferSize
oss_gaussian_set_callback
oss_gaussian_set_render_mode
oss_gaussian_get_render_mode
oss_gaussian_version
```

The DXGI proxy exports (`CreateDXGIFactory*`, etc.) are added in **T2.3** via
`src/dxgi_proxy.def`.

## Install (Cyberpunk 2077)

Sprint 2 install/uninstall scripts land in T2.12 under `scripts/install.ps1`
and `scripts/uninstall.ps1`. Until then, manual steps:

1. Quit Cyberpunk.
2. Back up: `copy "<Cyberpunk root>\bin\x64\nvngx_dlss.dll" nvngx_dlss.dll.real`.
3. Drop our DLL: `copy build\Release\dxgi.dll "<Cyberpunk root>\bin\x64\dxgi.dll"`.
4. **Disable Path Tracing** in Settings → Graphics (Sprint 2 does not handle DLSS-RR).

To roll back:

1. Quit Cyberpunk.
2. Delete `<Cyberpunk root>\bin\x64\dxgi.dll`.
3. `move nvngx_dlss.dll.real nvngx_dlss.dll` (back to the original name).

If `dxgi.dll` collides with another mod (Special K, ReShade), rename our DLL
to one of the OptiScaler fallback names: `winmm.dll`, `version.dll`,
`dbghelp.dll` — see
https://github.com/optiscaler/OptiScaler/wiki/Manual-Installation.

## Logs

`%LOCALAPPDATA%\oss-gaussian\interception.log` (line-buffered).
EXR captures (T2.7 onward) live under `%LOCALAPPDATA%\oss-gaussian\captures\`.

## Reference implementations

We model this DLL closely on **OptiScaler**
(https://github.com/optiscaler/OptiScaler). Per-export references are listed
in the Sprint 2 plan under "OptiScaler reading list".

Other references:
- PotatoOfDoom/CyberFSR2 — original Cyberpunk DLSS spoof
- NVIDIA/DLSS — public NGX SDK headers
- crosire/ReShade — DXGI swapchain proxy
- bruhmoment21/UniversalHookX — minimal D3D12 Present hook
- microsoft/Detours — hook engine (MIT)

## Anti-cheat

Cyberpunk 2077 ships **without** EasyAntiCheat / BattlEye / Denuvo Anti-Cheat.
This DLL is single-player only. **Do not** use this code on any multiplayer
title without the user's explicit knowledge — a `dxgi.dll` proxy will get
flagged.
