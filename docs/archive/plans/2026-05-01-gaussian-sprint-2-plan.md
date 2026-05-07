# OSS-Gaussian — Sprint 2 Detailed Plan: D3D12 Frame Interception

**Spec:** `docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md`
**Hook research:** `docs/superpowers/d3d12-hook-design.md`
**Master plan reference:** `docs/superpowers/plans/2026-05-01-gaussian-master-plan.md` § Sprint 2
**Branch:** `v0.2-dev`
**Target hardware:** RTX 3080 Ti / Windows 11
**Target title:** Cyberpunk 2077 (RED Engine 4, DX12, no anti-cheat)
**Total estimate:** ~1.5 weeks (10–14 tasks; 7–9 working days of focused work)

---

## 0. Sprint 2 Outcome Definition

By end of Sprint 2 the following must be true on the 3080 Ti box:

1. A single combined DLL (`dxgi.dll`) sits in `Cyberpunk 2077\bin\x64\` and the game launches, runs, and exits cleanly with our DLL loaded.
2. All 10 NGX D3D12 entry points are intercepted; calls are logged; calls are forwarded to the real `nvngx_dlss.dll` so DLSS still functions normally end-to-end (pass-through mode).
3. Inside the spoofed `EvaluateFeature`, we read every G-buffer resource handed to us via `NVSDK_NGX_Parameter` (Color, Depth, MotionVectors, Jitter, Exposure, Output, Reset, MV.Scale, Subrect) and dump the first 100 frames as EXR to disk.
4. An A/B render-mode switch (hotkey) toggles between "DLSS untouched" and "OSS-Gaussian stub render", where the stub render is permitted to be a debug pattern (e.g. flat color or upscaled bilinear) — the goal is proving we can write `Output` ourselves with correct D3D12 resource state.
5. Per-frame telemetry CSV captured (CPU + GPU timings, frame index, hooked-path identifier).
6. A documented rollback procedure restores the original game in under 60 seconds.
7. Path Tracing is **defaulted off** in the playtest checklist; if PT is detected (`nvngx_dlssd.dll` loaded), we log a warning and refuse to engage write-back mode.

Anything beyond MVP — real-time IPC, neural inference, DLSS visual replacement — is **explicitly out of scope** and is Sprint 4 / Sprint 5 work.

---

## 1. Architecture Recap (5-line version)

- One DLL: `oss_gaussian_interception.dll` → renamed `dxgi.dll` on disk → game-local DLL search resolves it before `C:\Windows\System32\dxgi.dll`.
- DLL exports the 10 `NVSDK_NGX_D3D12_*` symbols + minimal forwarders for `dxgi.dll` symbols the game actually consumes.
- Inside the DLL: Microsoft Detours patches DXGI factory creation and (later) `IDXGISwapChain::Present` / `ID3D12CommandQueue::ExecuteCommandLists`.
- G-buffers come exclusively from the `NVSDK_NGX_Parameter` dictionary in `EvaluateFeature` — no heuristic RT scraping.
- Sprint 2 IPC = **file dump (EXR per frame)**. Live-IPC (shared memory) is deferred to Sprint 5.

---

## 2. Integration Design Notes — DLL ↔ Python IPC

### Decision: **File dump (EXR + sidecar JSON) for Sprint 2 MVP. Shared memory for Sprint 5.**

We considered four IPC options for Cyberpunk → Python:

| Option | Latency | Throughput | Complexity | Sprint 2 fit |
| --- | --- | --- | --- | --- |
| **stdio pipe** | low | low (~MB/s, single channel) | low | poor — no real stdio in game process; would require a forked child |
| **named pipe (`\\.\pipe\…`)** | low | medium | medium | viable but synchronous-style API surfaces don't match per-frame DX12 cadence |
| **shared memory (file mapping)** | very low | very high (no copy) | high | overkill for offline EXR; correct choice for live integration in Sprint 5 |
| **file dump (EXR + JSON sidecar)** | high (disk-bound) | high enough | very low | **chosen** — all of Sprint 4 (training) wants EXR on disk anyway |

Rationale for file dump in Sprint 2:
1. Sprint 4 (training data capture) wants EXR-on-disk regardless. Building the IPC twice is wasted effort.
2. EXR with sidecar JSON (per-frame jitter, MV scale, exposure scalars, reset flag) is exactly the format Sprint 4's data loader will consume.
3. Failures are diagnosable with `oiiotool` / Photoshop / Image Watch — no custom inspector.
4. Eliminates entire failure modes (deadlock, ringbuffer overflow, partial-write tearing) for the sprint where we are also debugging hooks.

For Sprint 5 (live integration): shared memory ringbuffer (3–4 slots × per-frame G-buffer set) + a `HANDLE`-event signaling protocol. Layout will live at `oss/gaussian/interception/include/oss_gaussian_ipc.h` and is intentionally not designed yet.

### Sprint 2 file layout

```
%LOCALAPPDATA%\oss-gaussian\
  interception.log                           # human-readable
  captures\
    session_20260501_142233\
      frame_00000000.exr                     # multi-channel: color, depth, mv.x, mv.y
      frame_00000000.json                    # jitter, mv_scale, exposure, reset, subrect
      frame_00000001.exr
      ...
      session.json                           # device info, DLSS version, hook init log
      telemetry.csv                          # frame_index,cpu_us,gpu_us,mode
```

EXR is written via TinyEXR (header-only, MIT). Color is RGB16F, depth is R32F, MV is RG16F → packed into a single multi-channel EXR per frame to keep filesystem ops cheap.

---

## 3. OptiScaler Reading List (per NGX export)

Read these files first, in order, **without cloning the repo**. Use GitHub web view; copy nothing verbatim.

Repository: https://github.com/optiscaler/OptiScaler

| Concern | OptiScaler reference | Why |
| --- | --- | --- |
| Overall NGX D3D12 surface | `OptiScaler/NVNGX_DLSS.cpp` | Canonical OptiScaler entry-point bag for the 10 exports we mirror |
| `NVSDK_NGX_D3D12_Init` / `Init_Ext` | `OptiScaler/NVNGX_DLSS.cpp` (search `Init_Ext`) | Application ID + data path handling |
| `NVSDK_NGX_D3D12_Shutdown1` | `OptiScaler/NVNGX_DLSS.cpp` (search `Shutdown1`) | Clean device-scoped teardown ordering |
| `GetCapabilityParameters` / `AllocateParameters` / `DestroyParameters` | `OptiScaler/NVNGX_Parameter.cpp` (or whichever file owns the `NVSDK_NGX_Parameter` impl in current main) | Param-dict lifecycle; how OptiScaler hands back its own parameter object |
| `NVSDK_NGX_D3D12_CreateFeature` | `OptiScaler/Inputs/DLSSFeatureDx12.cpp` and `OptiScaler/Inputs/FSR2FeatureDx12.cpp` | Feature object construction; per-feature dispatcher table |
| `NVSDK_NGX_D3D12_EvaluateFeature` | `OptiScaler/Inputs/DLSSFeatureDx12.cpp` (`Evaluate(...)`) and same for FSR2 | **The one that matters** — exactly which param keys to read for Color/Depth/MV/Jitter/Exposure/Output/Reset/MV.Scale and how to handle subrects |
| `NVSDK_NGX_D3D12_ReleaseFeature` | `OptiScaler/Inputs/DLSSFeatureDx12.cpp` (`Release(...)`) | Resource lifecycle / fence flush |
| `GetScratchBufferSize` | `OptiScaler/NVNGX_DLSS.cpp` (search `GetScratchBufferSize`) | Trivial; just a passthrough number |
| DXGI/D3D12 hook installation | `OptiScaler/Hooks/HooksDx.cpp` | The Detours patching pattern we copy for DXGI factory + swapchain |
| Cyberpunk-specific notes | https://github.com/optiscaler/OptiScaler/wiki/Cyberpunk-2077 | Confirms `dxgi.dll` proxy name; flags PT-forces-RR behavior |
| Manual install fallback names | https://github.com/optiscaler/OptiScaler/wiki/Manual-Installation | If `dxgi.dll` collides, fallback list: `winmm.dll` → `version.dll` → `dbghelp.dll` |

Secondary references (read after OptiScaler):

- `PotatoOfDoom/CyberFSR2` — the original Cyberpunk-targeted DLSS spoof, smaller and easier to read first: https://github.com/PotatoOfDoom/CyberFSR2
- `NVIDIA/DLSS` — public NGX headers to depend on (`include/nvsdk_ngx.h`, `nvsdk_ngx_defs.h`, `nvsdk_ngx_helpers.h`, `nvsdk_ngx_params.h`): https://github.com/NVIDIA/DLSS
- `crosire/reshade` — `source/dxgi/dxgi_swapchain.cpp` — production-grade DXGI swapchain proxy
- `bruhmoment21/UniversalHookX` — `UniversalHookX/src/hooks/backend/dx12/hook_directx12.cpp` — minimal D3D12 Present hook

---

## 4. Tasks (T2.1 → T2.13)

Estimates are honest. "Half day" = 4 hours of focused work. "Day" = 8 hours.

---

### T2.1 — Project scaffold (this PR)

**Goal:** C++ DLL project compiles to `oss_gaussian_interception.dll` (eventually renamed `dxgi.dll`) on the 3080 Ti box. No hooks active yet.

**Files (all under `oss/gaussian/interception/`):**

```
CMakeLists.txt
README.md
LICENSE
include/oss_gaussian_interception.h
src/dllmain.cpp
src/ngx_exports.cpp
src/g_buffer_extractor.h
src/g_buffer_extractor.cpp
src/log.h
src/log.cpp
third_party/Detours/.gitkeep
```

**Steps (3080 Ti):**
1. Open VS 2026 Native Tools x64 prompt.
2. `cd %REPO%\oss\gaussian\interception && cmake -S . -B build -G "Visual Studio 17 2022" -A x64`
3. `cmake --build build --config Release`
4. Copy `build\Release\oss_gaussian_interception.dll` → `bin\x64\dxgi.dll` (after backing up real one — see T2.12).

**Verify:** `dumpbin /exports build\Release\oss_gaussian_interception.dll` lists all 10 `NVSDK_NGX_D3D12_*` symbols.

**Acceptance:** DLL builds clean with `/W4` and no warnings on MSVC. Dropping it in `bin\x64\` as `dxgi.dll` lets Cyberpunk launch (will not run yet — needs T2.2 and T2.3 forwarders).

**Estimate:** half day. (this PR is the macOS-side scaffold; on-box build verification is the half day on 3080 Ti.)

**Done by:** this PR.

---

### T2.2 — Vendor Microsoft Detours

**Goal:** Detours sources sit at `oss/gaussian/interception/third_party/Detours/` and link into the DLL as a static lib.

**Steps:**
1. On 3080 Ti box: `git clone https://github.com/microsoft/Detours third_party/Detours --depth 1`
2. Pin commit; record SHA in `third_party/Detours/COMMIT.pin`.
3. Update root `CMakeLists.txt` to `add_subdirectory(third_party/Detours)` (Detours ships its own makefiles; we wrap with a tiny CMake `add_library(detours STATIC …)` over `src/*.cpp` excluding tools).
4. Confirm Detours is **MIT-licensed** (it is, as of microsoft/Detours main).

**Verify:** `cmake --build build --config Release` links `detours.lib` into `oss_gaussian_interception.dll`. `dumpbin /imports` shows no missing externals.

**Acceptance:** A `DetourTransactionBegin()` / `DetourTransactionCommit()` pair calling no detours compiles, links, and is a no-op at runtime.

**Estimate:** half day.

---

### T2.3 — DXGI export forwarding (proxy minimum)

**Goal:** When loaded as `dxgi.dll`, our DLL forwards the DXGI exports Cyberpunk actually imports to the real system `dxgi.dll`. The game must launch with our DLL present and run identically to vanilla.

**Files:** `src/dllmain.cpp` (extend); `src/dxgi_proxy.cpp` (new, ~80 LOC); `src/dxgi_proxy.def` (module-definition file with `EXPORTS` list).

**Steps:**
1. Generate `.def` listing every export from `C:\Windows\System32\dxgi.dll` (`dumpbin /exports` then strip ordinals; keep names). Forward each unmodified export to `system_dxgi!Symbol` via DEF `EXPORTS Symbol = system_dxgi.Symbol`. (Use a runtime LoadLibrary + GetProcAddress trampoline pattern, NOT static `.lib` linking against system32 — avoids load-order traps.)
2. `DllMain(DLL_PROCESS_ATTACH)` calls `LoadLibraryW(L"C:\\Windows\\System32\\dxgi.dll")`, stores HMODULE.
3. Each forwarder: `static auto fn = GetProcAddress(hSysDxgi, "CreateDXGIFactory"); return fn(...)`.
4. Sanity-test: launch Cyberpunk with our DLL renamed to `dxgi.dll`. Game must reach main menu.

**Verify:** Game launches; Performance overlay numbers identical to vanilla (within noise). `interception.log` shows `dxgi.dll proxy attached, forwarding to system32`.

**Acceptance:** Cyberpunk reaches main menu and starts a save. No visual or perf regression.

**Estimate:** 1 day. (Most of the effort is the export list — there are ~30 of them.)

**Risk:** If Cyberpunk imports a DXGI symbol we forgot to forward, game fails to start with a crisp `LoadLibrary` failure in event log. Easy to diagnose.

---

### T2.4 — NGX exports: stub + log only

**Goal:** All 10 `NVSDK_NGX_D3D12_*` exports exist as DLL exports, are called by Cyberpunk (verified), and log "<name> entered" via our file logger. Returns are conservative success codes that **do not** crash the game (return `NVSDK_NGX_Result_FAIL_FeatureNotSupported` everywhere → Cyberpunk falls back to non-DLSS).

This task ships the stubs in this PR (`src/ngx_exports.cpp`). Verification on the 3080 Ti box is the work of this task.

**Steps:**
1. Drop the DLL into `bin\x64\nvngx_dlss.dll` (yes, also rename to `nvngx_dlss.dll` for this test) — backing up the real one first.
2. Launch Cyberpunk. Set DLSS = on in Settings. Confirm log shows `Init_Ext` → `AllocateParameters` → `CreateFeature` → `EvaluateFeature` (per frame) → `ReleaseFeature` → `Shutdown1` over a session.
3. Toggle DLSS off/on; confirm log shows lifecycle.

**Verify:** `interception.log` contains all 10 export names called at least once during a 5-minute session that toggles DLSS on/off twice.

**Acceptance:** Cyberpunk does **not** crash with our DLL acting as `nvngx_dlss.dll`. Game falls back to no-DLSS rendering when our stub returns `FeatureNotSupported`. Then we **revert** — for the rest of Sprint 2 we ship as `dxgi.dll` and forward NGX calls to the real DLSS DLL (T2.5).

**Estimate:** half day.

---

### T2.5 — NGX pass-through to real `nvngx_dlss.dll`

**Goal:** When loaded as `dxgi.dll`, our 10 NGX stubs forward to the real DLSS DLL so DLSS works visually identically to vanilla while we observe every parameter in flight.

**Files:** `src/ngx_exports.cpp` (extend); `src/ngx_passthrough.h/.cpp` (new).

**Steps:**
1. On `Init` / `Init_Ext`, `LoadLibraryW(L"nvngx_dlss.dll.real")` — we will rename the real DLL to `.real` at install time (T2.12) so it is still loadable but Cyberpunk's `LoadLibrary("nvngx_dlss.dll")` resolves to us first.
2. Resolve all 10 entry points from the real DLL via `GetProcAddress`. Store in a function-pointer table.
3. Each of our 10 exports: log call → forward to real → log return → return.
4. Special handling: `EvaluateFeature` logs the param-dict contents (key names + resource pointers) before forwarding.

Wait — re-read: in our model the DLL is `dxgi.dll`, **not** `nvngx_dlss.dll`. So Cyberpunk loads the real `nvngx_dlss.dll` directly. To intercept its NGX exports we must hook `LoadLibraryW` / `GetProcAddress` (Detours IAT patch) and return our function pointers when the game asks for `NVSDK_NGX_D3D12_*` from `nvngx_dlss.dll`. This is exactly OptiScaler's `Hooks/HooksDx.cpp` pattern.

Updated steps:
1. Detour `LoadLibraryW`, `LoadLibraryExW`. When game requests `nvngx_dlss.dll`, return our HMODULE.
2. Detour `GetProcAddress`. When game asks for `NVSDK_NGX_D3D12_*` from our HMODULE, return our exported function. For any other symbol, defer to a hidden real-NGX HMODULE.
3. Real NGX HMODULE comes from us calling `LoadLibraryW(L"nvngx_dlss.dll.real")` once at attach, where `.real` is the renamed-by-installer original.
4. Each NGX stub forwards to the real fn-ptr from that hidden HMODULE.

**Verify:** With our `dxgi.dll` installed and real DLSS renamed to `nvngx_dlss.dll.real`, Cyberpunk DLSS quality is visually identical to vanilla. Frame timing within ±2%. Log shows full per-frame NGX activity.

**Acceptance:** A 10-minute Cyberpunk benchmark run with our DLL installed has the same average FPS as the vanilla run within noise (±2%).

**Estimate:** 1.5 days. (LoadLibrary/GetProcAddress IAT detouring is fiddly across UCRT versions.)

**Risk:** RED Engine 4 may use `LoadLibraryExW` with `LOAD_LIBRARY_SEARCH_*` flags; verify our detour catches both. Run with `gflags /i Cyberpunk2077.exe +ust` to dump load-library trace if uncertain.

---

### T2.6 — Param-dict reader (G-buffer extractor)

**Goal:** Inside our `EvaluateFeature`, before forwarding, read every G-buffer-relevant param key from `NVSDK_NGX_Parameter` into a typed struct. Log resource descriptions (format, dim, dim).

**Files:** `src/g_buffer_extractor.cpp` (extend), `src/g_buffer_extractor.h` (extend).

**Param keys to read** (string keys, names from `nvsdk_ngx_defs.h`):
- `Color` → `ID3D12Resource*` (input low-res color)
- `Output` → `ID3D12Resource*` (target we are supposed to fill)
- `Depth` → `ID3D12Resource*`
- `MotionVectors` → `ID3D12Resource*`
- `Jitter.Offset.X`, `Jitter.Offset.Y` → `float`
- `MV.Scale.X`, `MV.Scale.Y` → `float`
- `Exposure.Scale` → `float` (optional; some titles)
- `ExposureTexture` → `ID3D12Resource*` (optional)
- `Reset` → `unsigned int`
- `Subrect.Base.X/Y`, `Subrect.Width/Height`, `Subrect.Rendering.Width/Height` → `unsigned int`
- `DLSS.Feature.Create.Flags` → `unsigned int` (read MVJittered, depth-inverted bits)

**Steps:**
1. Define `struct GBufferFrame` mirroring those keys.
2. Implement `ReadFromNgxParameters(const NVSDK_NGX_Parameter*, GBufferFrame*)`.
3. Log struct contents at `INFO` level for first 5 frames, `TRACE` thereafter.

**Verify:** First 5 frames' log dumps show non-null resource pointers for Color/Depth/MV/Output, jitter values in pixel range (±1.0), MV.Scale matching the render resolution, sane subrect.

**Acceptance:** Zero null-pointer reads in a 1000-frame session. Reset flag fires on save-load (camera cut).

**Estimate:** 1 day.

---

### T2.7 — EXR dump pipeline

**Goal:** Write each frame's color/depth/MV to a multi-channel EXR + sidecar JSON. Dump first 100 frames of any session by default.

**Files:** `src/exr_dump.h/.cpp` (new, ~150 LOC), `third_party/tinyexr/` (header-only).

**Steps:**
1. Vendor `syoyo/tinyexr` (single-header MIT) at `third_party/tinyexr/tinyexr.h`. Pin commit.
2. For each `EvaluateFeature` while `frame_index < 100`:
   - Allocate readback heaps sized to each input resource.
   - Copy resources to readback via a dedicated copy command list on a copy queue we own.
   - Fence-wait with a 50ms timeout. (If we exceed the budget, drop the frame and log.)
   - Map readback, build EXR multi-channel image, `SaveEXRToFile()`.
   - Write sidecar JSON with jitter / mv_scale / exposure / reset.
3. Output path: `%LOCALAPPDATA%\oss-gaussian\captures\session_<timestamp>\frame_<8d>.{exr,json}`.

**Verify:** `oiiotool --info captures\…\frame_00000010.exr` shows expected channels (`R, G, B, depth, mv.x, mv.y`) and sane min/max. Open in Photoshop / DJV — image is visually a low-res Cyberpunk frame.

**Acceptance:** 100 frames captured in <30 seconds of gameplay with no game stutter exceeding 5ms. EXRs round-trip through Sprint 4's data loader (deferred check; for Sprint 2 just confirm format via `oiiotool`).

**Estimate:** 1 day.

**Risk:** Readback for color+depth+MV at 1080p ≈ 30MB/frame. 100 frames = 3 GB — fine. At 1440p input it's ~50MB/frame; still fine.

---

### T2.8 — DXGI Present hook (only what's needed)

**Goal:** Hook `IDXGISwapChain::Present` so we can (a) snapshot back-buffer for the A/B compare overlay, (b) detect resize / device-lost.

This is **secondary**; G-buffer capture works without it. Only do this once T2.7 is solid.

**Files:** `src/swapchain_hooks.h/.cpp` (new, ~120 LOC), extend `src/dllmain.cpp`.

**Steps:**
1. Detour `IDXGIFactory::CreateSwapChain*` and `IDXGIFactory2::CreateSwapChainForHwnd`.
2. On first swapchain creation, vtable-patch `Present` and `ResizeBuffers` on the returned `IDXGISwapChain*`.
3. Our `Present` hook: log frame count, dispatch any back-buffer copy if A/B mode demands, call original `Present`.
4. Our `ResizeBuffers` hook: invalidate any cached back-buffer copies.

**Verify:** Log shows one `CreateSwapChain` event at game launch; `Present` is called 60+ times per second. `ResizeBuffers` fires when toggling fullscreen ↔ windowed.

**Acceptance:** No crash on resolution change, alt-tab, or fullscreen toggle in a 10-minute session. Average FPS within ±2% of T2.5 baseline.

**Estimate:** 1 day.

**Risk:** Swapchain proxy + DLSS interaction is the most version-fragile part of OptiScaler. If RED Engine patches break us, fall back to NGX-only path (skip A/B overlay this sprint).

---

### T2.9 — A/B render-mode toggle (stub render)

**Goal:** Hotkey (default `F11`) toggles between:
- Mode A: pass-through DLSS (T2.5 behavior)
- Mode B: skip DLSS forwarding; fill `Output` resource ourselves with a debug pattern (we copy `Color` upscaled by `D3D12 CopyTextureRegion` with target subrect = `Subrect.Width × Subrect.Height`)

This proves we can write the `Output` UAV in the correct D3D12 state and the game continues rendering UI on top of it.

**Files:** `src/ab_toggle.h/.cpp` (new, ~80 LOC).

**Steps:**
1. Install a low-level keyboard hook (`SetWindowsHookExW(WH_KEYBOARD_LL, …)`) for F11.
2. Maintain `std::atomic<RenderMode> g_mode`.
3. In `EvaluateFeature`, branch on `g_mode`. Mode B path:
   - `ID3D12GraphicsCommandList::ResourceBarrier`(Output: `UNORDERED_ACCESS` → `COPY_DEST`)
   - Bilinear-blit Color → Output (compute shader, ~60 LOC HLSL — vendor in `src/shaders/upscale_bilinear.hlsl`).
   - Barrier back to whatever NGX expects on exit (`UNORDERED_ACCESS`).
4. On Mode B → A switch, do nothing special; next frame just goes through pass-through.

**Verify:** F11 in-game flips between sharp DLSS image and a soft bilinear upscale. UI overlay (HUD) still composites correctly on top in both modes. No D3D12 debug-layer warnings.

**Acceptance:** 10-minute session toggling F11 every 30 seconds — no crashes, no debug-layer errors when run with `dxcpl.exe` debug-layer enabled.

**Estimate:** 1 day.

---

### T2.10 — Telemetry / structured logging

**Goal:** Per-frame CSV with `frame_index, cpu_ms, gpu_ms, mode, dropped_capture`. Plus structured init/shutdown JSON with detected DLSS version, GPU adapter, resolutions.

**Files:** `src/telemetry.h/.cpp` (new, ~120 LOC), extend `src/log.cpp`.

**Steps:**
1. CPU timing: `QueryPerformanceCounter` around `EvaluateFeature` body.
2. GPU timing: `ID3D12QueryHeap` (timestamp queries) on the same command list, resolved + read on the next `EvaluateFeature` (one-frame-late).
3. Telemetry CSV opened for append at session start; flushed every 60 frames.
4. Session JSON written on `DLL_PROCESS_DETACH`.

**Verify:** CSV opens cleanly in Excel. CPU times should be sub-millisecond in pass-through mode; GPU times should be a few ms (DLSS work).

**Acceptance:** 1000-frame CSV with no NaN / negative values.

**Estimate:** half day.

---

### T2.11 — Path Tracing detection + safety refusal

**Goal:** If the game enables Path Tracing, NVIDIA loads `nvngx_dlssd.dll` (Ray Reconstruction) instead of `nvngx_dlss.dll`. Sprint 2 explicitly does not handle DLSS-RR. Detect the load and refuse Mode B engagement.

**Files:** extend `src/dllmain.cpp` and `src/ngx_exports.cpp`.

**Steps:**
1. In our `LoadLibrary` detour (T2.5), watch for `nvngx_dlssd.dll` request. If seen, set `g_path_tracing_detected = true` and **do not** intercept it — let the original load happen.
2. In `EvaluateFeature`, if `g_path_tracing_detected`, log a one-time WARN and force `g_mode = ModeA` ignoring user toggle.
3. Document in README: "disable Path Tracing in Settings → Graphics for Sprint 2 testing".

**Verify:** Toggle PT in-game; log shows the warning; F11 has no effect.

**Acceptance:** No crashes or visual corruption when Path Tracing is enabled or toggled mid-session.

**Estimate:** half day.

---

### T2.12 — Install / uninstall script + rollback

**Goal:** A one-click PowerShell script that installs and uninstalls cleanly. Rollback completes in under 60 seconds on demand.

**Files:** `oss/gaussian/interception/scripts/install.ps1`, `scripts/uninstall.ps1`, append README.

**Install steps:**
1. Validate game install path argument.
2. Back up `bin\x64\dxgi.dll` → `dxgi.dll.vanilla` (if it exists; usually doesn't).
3. Back up `bin\x64\nvngx_dlss.dll` → `nvngx_dlss.dll.real`.
4. Copy our built DLL → `bin\x64\dxgi.dll`.
5. Create `%LOCALAPPDATA%\oss-gaussian\captures\` if absent.
6. Print success + reminder to disable Path Tracing.

**Uninstall steps:**
1. Delete `bin\x64\dxgi.dll`.
2. If `dxgi.dll.vanilla` exists, restore it. Otherwise leave absent (game-local DXGI is optional; Windows resolves system32).
3. Restore `nvngx_dlss.dll.real` → `nvngx_dlss.dll`.
4. Optionally archive `%LOCALAPPDATA%\oss-gaussian\` to a timestamped zip.

**Verify:** Run install, launch game, exit, run uninstall, launch game. Game runs identically to vanilla pre-install.

**Acceptance:** Round-trip install→play→uninstall→play cycle leaves no residue. Verified by `Get-FileHash` on `bin\x64\nvngx_dlss.dll` matching pre-install hash.

**Estimate:** half day.

---

### T2.13 — Sprint 2 code review checkpoint

**Goal:** Run the cross-cutting review pipeline on Sprint 2 commits.

**Steps:**
1. `python -m oss.gaussian.review.run --sprint 2 --commit-range main..HEAD`
2. Review artifacts saved to `oss/gaussian/review/artifacts/sprint-2/`.
3. Resolve any REQUEST_CHANGES; escalate any BLOCK.

**Verify:** Judge verdict file exists and is APPROVE.

**Acceptance:** Approved → unblocks Sprint 3 / Sprint 4.

**Estimate:** half day (plus any rework).

---

## 5. Time roll-up

| Task | Estimate |
| --- | --- |
| T2.1 scaffold | 0.5 d |
| T2.2 Detours vendor | 0.5 d |
| T2.3 DXGI proxy | 1.0 d |
| T2.4 NGX stubs verified | 0.5 d |
| T2.5 NGX pass-through | 1.5 d |
| T2.6 param-dict reader | 1.0 d |
| T2.7 EXR dump | 1.0 d |
| T2.8 Present hook | 1.0 d |
| T2.9 A/B toggle | 1.0 d |
| T2.10 telemetry | 0.5 d |
| T2.11 PT safety | 0.5 d |
| T2.12 install/rollback | 0.5 d |
| T2.13 review | 0.5 d |
| **Total** | **10.0 days** |

10 working days = 2 weeks calendar. Master plan said 1.5 weeks; honest answer is "1.5 weeks if nothing breaks; 2 weeks with realistic debug overhead".

---

## 6. Risk register (Sprint 2 specific)

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| LoadLibrary/GetProcAddress detour misses an entry path Cyberpunk uses | M | H — DLSS replacement won't fire | Trace with `gflags +ust` first; OptiScaler's `HooksDx.cpp` already handles this — mirror exactly |
| RED Engine patch lands during sprint and changes NGX param keys | L | M | Pin to current Cyberpunk patch version in test plan; defer post-patch validation to a follow-up |
| Path Tracing user-enabled without us noticing → DLSS-RR DLL crashes our hooks | M | M | T2.11 explicit detection + refusal; documented in README |
| EXR write stalls game thread | L | M | Copy queue + readback on dedicated thread (T2.7); 50ms timeout drops frame instead of blocking |
| Resource state mismatch on Mode B exit → debug layer error / next frame breaks | M | H | Test continuously with D3D12 debug layer on; verify Mode B is sample-perfect on a dedicated test save |
| `dxgi.dll` proxy name collides with another mod (e.g. Special K) | L | H — game won't start | Document fallback name list (`winmm.dll`, `version.dll`, `dbghelp.dll`) per OptiScaler manual install wiki |
| Detours license / source not vendored cleanly | L | L | Confirmed MIT; vendor source as submodule with pinned SHA in T2.2 |
| Anti-cheat trips | Very L | H — ban | Cyberpunk has no AC. Document "do not use on multiplayer titles" warning in README |

---

## 7. Out of scope (do NOT build in Sprint 2)

- Real-time shared-memory IPC to Python
- Neural network inference inside the DLL
- DLSS Frame Generation (`nvngx_dlssg.dll`) interception
- DLSS Ray Reconstruction (`nvngx_dlssd.dll`) interception
- Vulkan path (Cyberpunk is DX12 only; no Vulkan target this sprint)
- Cross-game support — Cyberpunk only
- ImGui overlay (deferred — F11 toggle without UI is enough for MVP)
- Capture compression / codec (raw EXR only)

---

## 8. References

- `docs/superpowers/d3d12-hook-design.md` (this repo) — hook research output
- `docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md` — design spec
- `docs/superpowers/plans/2026-05-01-gaussian-master-plan.md` — Sprint 2 outline (this plan supersedes its detail section)
- OptiScaler — https://github.com/optiscaler/OptiScaler (read list § 3 above)
- NVIDIA DLSS SDK — https://github.com/NVIDIA/DLSS
- Microsoft Detours — https://github.com/microsoft/Detours
- TinyEXR — https://github.com/syoyo/tinyexr
- ReShade DXGI swapchain — https://github.com/crosire/reshade/blob/main/source/dxgi/dxgi_swapchain.cpp
