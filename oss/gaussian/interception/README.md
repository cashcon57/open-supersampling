# oss-gaussian-interception

C++ DLL that intercepts D3D12 and NVIDIA NGX (DLSS) calls in DLSS-shipping
games, hands G-buffers (color, depth, motion vectors) to the OSS-Gaussian
renderer, and replaces DLSS with our own upscaler.

This is a **generic open-source DLSS replacement**, not a Cyberpunk-specific
mod. The DLL works wherever DLSS works.

This is **Sprint 2 scaffolding only**. No hooks are active yet. See
`docs/superpowers/plans/2026-05-01-gaussian-sprint-2-plan.md` for tasks.

## Compatibility

The DLL ships under the name `dxgi.dll` and relies on Windows' game-local DLL
search resolving it before `C:\Windows\System32\dxgi.dll`. Every DXGI export
the host process imports is forwarded to the real system dxgi.dll via a
runtime LoadLibrary + GetProcAddress trampoline (see `src/dxgi_proxy.cpp`),
so the game launches identically to vanilla until we explicitly intercept a
call.

| Game                | Status     | Notes                                  |
|---------------------|------------|----------------------------------------|
| Cyberpunk 2077      | Primary    | Sprint 2 validation target             |
| Hogwarts Legacy     | TBD        | Same DLL; not yet smoke-tested         |
| Portal RTX          | TBD        | Same DLL; not yet smoke-tested         |
| Alan Wake 2         | TBD        | Same DLL; not yet smoke-tested         |
| Other DLSS DX12     | Should work| File a report if exports go missing    |

The proxy implementation is **game-agnostic**: no per-title quirks, no
hardcoded paths beyond `C:\Windows\System32\dxgi.dll`. If a forwarder
encounters a symbol that the local Windows version's system dxgi.dll does
not export, the miss is logged to `interception.log` and the call returns
`E_NOTIMPL`, allowing graceful degradation rather than a crash.

### Fallback: `.def`-based forwarding

A `.def` file with `EXPORTS Foo=C:/Windows/System32/dxgi.Foo` lines is a
viable alternative to the runtime trampoline (see MSDN "Module-Definition
(.def) Files"). It is documented here for completeness but **not used** —
the runtime trampoline is more flexible: we can log misses, support new
symbols without re-linking, and never bake a static path into the binary.

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

## Build and Package for Cyberpunk 2077 (Windows 11)

Prerequisites:
- Visual Studio 2022 or newer with "Desktop development with C++"
- CMake 3.20+
- Python 3.11+
- PyInstaller (`python -m pip install pyinstaller`)
- WiX Toolset v3 (`candle.exe` and `light.exe` on `PATH`)
- PowerShell from a shell where MSVC x64 tools are active

The packaging flow uses a staging directory that must contain exactly these
WiX inputs:

| File | Producer | Install target |
|------|----------|----------------|
| `oss_capture.dll` | CMake `CaptureInstaller` component | `<Cyberpunk>\bin\x64\dxgi.dll` |
| `oss_capture_uploader.exe` | PyInstaller uploader spec | `%LOCALAPPDATA%\oss-capture\oss_capture_uploader.exe` |
| `config.json` | `scripts/build_capture_installer.py` | `%LOCALAPPDATA%\oss-capture\config.json` |
| `scheduled_task.xml` | `scripts/build_capture_installer.py` | `%LOCALAPPDATA%\oss-capture\scheduled_task.xml` |

PowerShell build/package commands:

```powershell
$Repo = "C:\src\OpenSuperSampling"
$Stage = "$Repo\dist\oss_capture_cyberpunk_v1.0.0"
$GameBin = "C:\Program Files (x86)\Steam\steamapps\common\Cyberpunk 2077\bin\x64"

cd $Repo
cmake -S oss\gaussian\interception -B build\interception -G "Visual Studio 17 2022" -A x64
cmake --build build\interception --config Release
cmake --install build\interception --config Release --component CaptureInstaller --prefix $Stage

python -m PyInstaller oss\capture\installer\oss_capture_uploader_pyinstaller.spec --distpath $Stage
Move-Item -Force "$Stage\oss_capture_uploader\oss_capture_uploader.exe" "$Stage\oss_capture_uploader.exe"

python scripts\build_capture_installer.py `
  --game cyberpunk-2077 `
  --game-exe-name Cyberpunk2077.exe `
  --proxy-dll-name dxgi.dll `
  --installer-version 1.0.0 `
  --mode lite `
  --output $Stage `
  --require-artifacts

candle.exe -nologo -dArtifactDir="$Stage" "$Stage\oss_capture_installer.wxs" -out "$Stage\oss_capture_installer.wixobj"
light.exe -nologo -ext WixUtilExtension "$Stage\oss_capture_installer.wixobj" -out "$Stage\oss_capture_cyberpunk_v1.0.0.msi"

msiexec.exe /i "$Stage\oss_capture_cyberpunk_v1.0.0.msi" GAME_BIN_X64_DIR="$GameBin"
```

The CMake build also leaves the game-local proxy at
`build\interception\Release\dxgi.dll`. The installer stage consumes the
canonical copy named `oss_capture.dll`, then installs it as `dxgi.dll`.

To use clang-cl instead of MSVC:

```powershell
cmake -S oss\gaussian\interception -B build\interception-clang -G "Visual Studio 17 2022" -T ClangCL -A x64
cmake --build build\interception-clang --config Release
```

## Verify the build

```powershell
dumpbin /exports build\interception\Release\dxgi.dll
python scripts\build_capture_installer.py --game cyberpunk-2077 --game-exe-name Cyberpunk2077.exe --proxy-dll-name dxgi.dll --output $Stage --require-artifacts
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

The DXGI proxy exports (`CreateDXGIFactory*`, `DXGIGetDebugInterface1`, the
PIX/Compat/DXGID3D10 surface, etc.) are added in **T2.3** via
`src/dxgi_proxy.cpp` (runtime LoadLibrary + GetProcAddress trampolines).

## Manual Install (Cyberpunk 2077)

Prefer the MSI flow above. For a direct smoke test:

1. Quit Cyberpunk.
2. Back up any existing `dxgi.dll` in `<Cyberpunk root>\bin\x64\`.
3. Drop our DLL: `copy build\interception\Release\dxgi.dll "<Cyberpunk root>\bin\x64\dxgi.dll"`.
4. **Disable Path Tracing** in Settings → Graphics (Sprint 2 does not handle DLSS-RR).

To roll back:

1. Quit Cyberpunk.
2. Delete `<Cyberpunk root>\bin\x64\dxgi.dll`.
3. Restore the previous `dxgi.dll` if one existed.

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
