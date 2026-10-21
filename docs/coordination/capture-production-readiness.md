# OSS Capture Tool Production Readiness

**Date:** 2026-05-12
**Scope:** make the capture tool production-ready for real game training-data capture. Cyberpunk 2077 is the first validation target; the capture architecture stays profile-driven for other DLSS/FSR titles.

## Current State

The synthetic upload path is ready enough for operator automation: simulator frames can be generated, uploaded, validated by ingest, and indexed. The Windows DLL also builds and the native unit tests pass on the 3080 Ti host. OpenEXR is installed through `C:\tools\vcpkg`, and Release builds with `OSS_CAPTURE_HAS_OPENEXR=1` now pass native tests.

The real-game capture path is implemented but not accepted for production training yet. DllMain now defers heavyweight initialization to a worker thread and the native smoke test asserts that async initialization completes. NGX/DLSS pass-through exports are wired, `EvaluateFeature` reads the DLSS SR parameter bag for color, output, depth, motion vectors, jitter, MV scale, reset, render dimensions, and output dimensions without vendoring NVIDIA headers, and accepted frames are copied through GPU readback into EXR + JSON sidecars. The build stages both `dxgi.dll` and `nvngx_dlss.dll` installer artifacts, and repeat Windows builds no longer collide with the SDK `dxgi.lib`.

The current runtime blocker is validation, not compilation: Cyberpunk starts as a process on the 3080 Ti host, but the scheduled/direct launch path does not create a game window or reach a render/upscaler call path, even with the local mod-loader stack temporarily disabled. No real pending EXR files exist yet. Upload-mode code is implemented, but the available R2 credential set is not green: direct S3 credentials are missing and the Worker upload smoke currently returns HTTP 403.

Cyberpunk loader findings on the 3080 Ti host:

- The install already contains ReShade as `bin\x64\dxgi.dll`, Ultimate ASI Loader as `bin\x64\version.dll`, RED4ext as `bin\x64\winmm.dll`, Streamline, DLSS, FSR, and XeSS DLLs.
- Cyberpunk maps local `dxgi.dll`, and the OSS proxy reaches `DllMain` when installed as `dxgi.dll`. Marker evidence from both SSH and interactive scheduled-task launches shows `dllmain-attach`, `schedule-start`, `create-thread-start`, `create-thread-done`, and `dllmain-attach-done`; the worker thread does not reach its first `begin` marker within a 90 second launch window.
- The DXGI export lazy-init fallback has been added, but this Cyberpunk launch path did not call the forwarded DXGI exports within the same 90 second smoke window.
- Direct init under `DllMain` is now available only as an opt-in diagnostic (`OSS_GAUSSIAN_INIT_IN_DLLMAIN=1`). It proves the DLL can reach logger/config/capture setup, but it stalls before D3D12 hook-install completion while still under loader lock, so it is not a production path.
- NGX-only replacement (`nvngx_dlss.dll` without replacing the loader proxy) did not load the OSS NGX DLL during a 75 second smoke, despite the user settings selecting DLSS Performance. The automated/console launch is not yet reaching a live DLSS EvaluateFeature path.
- Cyberpunk imports and maps `sl.interposer.dll`. A pure Streamline-only smoke (`-NoLoaderProxy -NoNgxProxy -StreamlineProxy`) safely replaced only `sl.interposer.dll`, preserved the original as `oss_sl_real.dll`, launched benchmark mode, and restored the original. The proxy mapped and reached `DllMain`, but no `slInit`, `slSetD3DDevice`, `slSetTag`, `slGetFeatureFunction`, or `slEvaluateFeature` wrapper logged within 90 seconds.
- The installed settings file currently reports `ResolutionScaling=DLSS`, `DLSS=Performance`, `DLSS_BackendPreset=Transformer`, and frame generation off. The same process still maps FSR3 and XeSS DLLs (`ffx_fsr3upscaler_x64.dll`, `ffx_backend_dx12_x64.dll`, `libxess.dll`), so generalized compatibility needs FSR/XeSS adapter profiles rather than assuming all DLSS-visible titles reach NGX or Streamline exports.
- The Streamline proxy now exports the Cyberpunk import subset plus `slSetTagForFrame`, wraps function pointers returned by `slGetFeatureFunction` for tag/constants/evaluate calls, and lazy-triggers post-loader native init from every Streamline export wrapper. The expanded native export-surface test covers every `.def` export.
- The Streamline proxy now builds with lightweight/deferred native attach (`OSS_GAUSSIAN_NO_SYSTEM_PROXY_ATTACH`, `OSS_GAUSSIAN_DEFER_NATIVE_ATTACH`, `OSS_GAUSSIAN_PROXY_EXPORT_LIGHTWEIGHT_INIT`). The 3080 Ti smoke marker now shows `native-attach-deferred` from `DllMain`, so the proxy no longer starts full D3D12 hook setup during load.
- `scripts/cyberpunk_proxy_smoke.ps1` has an opt-in `-TemporarilyDisableMods` mode that renames/restores RED4ext, CET/`winmm.dll`, `version.dll`, and game-local `dxgi.dll` when `-NoLoaderProxy` is used. Clean smokes restore all touched files and show no lingering `.oss-disabled-*` files.
- The FSR3 direct proxy now builds as `ffx_fsr3_x64.dll`, exports exactly the 17 AMD symbols Cyberpunk imports/loads, forwards to the original DLL preserved as `ffx_fsr3_x64.dll.oss-backup` or `oss_ffx_fsr3_real.dll`, and has a native export-surface test. Pure FSR3 proxy smokes prove Cyberpunk maps the OSS DLL and restore is clean, but no `ffxFsr3ContextDispatchUpscale` or related FSR3 export call is observed in the benchmark window yet.
- A reversible settings smoke changed Cyberpunk `ResolutionScaling` from `DLSS` to `FSR3`, ran the FSR3 proxy benchmark smoke, and restored the original settings file. The process still mapped `ffx_fsr3_x64.dll` but emitted no FSR3 export-call evidence, so the active benchmark upscaler path remains unresolved.
- The native writer now honors `pending_dir` from DLL config/session config when present, persists `capture_storage_mode` in metadata, and the EXR writer no longer requires normals when `capture_normals=0`. This fixes the current training-ready color/depth/motion/output path before full material G-buffer capture lands.
- Replacing `version.dll` is not viable on this install because the existing Ultimate ASI Loader exports 809 compatibility symbols, not only the 17 System32 VERSION exports.
- Dropping `oss_capture.asi` into `bin\x64` and `bin\x64\plugins` was not picked up by the existing ASI loader in the automated smoke.
- Direct `CreateRemoteThread`/`LoadLibraryW` injection currently times out even for `C:\Windows\System32\version.dll`, so the tray injection path needs a different injection primitive or an interactive/manual validation path for this host.
- The native DLL was refactored so loader-profile artifacts no longer statically import `dxgi.dll`, `d3d12.dll`, or `version.dll`; these are resolved dynamically after attach. This prevents game-local proxy stacks from being pulled in during DLL load.
- `scripts/cyberpunk_proxy_smoke.ps1` now supports marker-enabled, NGX-only, pure Streamline-only, FSR3, FFX backend, clean-mod-disable, and direct-init diagnostic modes, records SHA-256 identity for swapped DLLs, pre-kills stale Cyberpunk processes before replacement, restores original DLLs and the previous `%LOCALAPPDATA%\oss-capture\config.json` after each run, and avoids Windows PowerShell 5.1 `ConvertTo-Json` hangs with a bounded smoke-report serializer.

## Production Definition

The tool is production-ready only when all gates below are green:

- A signed per-game installer can install, upgrade, and uninstall without breaking the game or leaving stale proxy DLLs.
- Cyberpunk 2077 launches normally with the proxy DLL installed and DLSS still works.
- The DLL captures LR color, DLSS output, depth, motion vectors, jitter, render/output dimensions, and frame timing from real `NVSDK_NGX_D3D12_EvaluateFeature` calls.
- Captured frames are written as valid training EXRs plus matching metadata sidecars under the uploader pending layout.
- The uploader drains pending frames with bounded disk use, retry/backoff, and terminal-error cleanup.
- Local capture mode leaves frames on disk and upload mode explicitly drains to ingest/R2. Session roots are separated by storage mode, native metadata marks `capture_storage_mode`, the uploader skips local-only frames even if an operator later switches to upload mode, and upload mode fails closed unless an installer-provided token is present.
- The ingest service authenticates, validates, deduplicates, rate-limits, stores, and indexes captures durably.
- A Cyberpunk benchmark run produces a reproducible, trainable dataset shard with frame counts, channel validation, and sample visual inspection.

## P0 Blockers

### Native D3D12/NGX Capture

- Verify Cyberpunk launches with both game-local proxies installed: `dxgi.dll` and `nvngx_dlss.dll`.
- Confirm the active upscaler pass-through loads the real vendor DLL and forwards all calls on the Cyberpunk benchmark path.
- Validate DLSS/NGX `NVSDK_NGX_D3D12_EvaluateFeature`, Streamline `slEvaluateFeature`, or FSR3 `ffxFsr3ContextDispatchUpscale` parameter extraction against live Cyberpunk logs and resource descriptors.
- Confirm the native readback worker emits non-degenerate EXR channels from live game resources.
- Extend the FSR adapter profile beyond the validated `ffx_fsr3_x64.dll` surface if Cyberpunk or other games route through `ffx_fsr3upscaler_x64.dll` or the newer FFX API loader. AMD's public SDK exposes dispatch descriptions with the same canonical training signals: command list, color, depth, motion vectors, exposure, reactive/transparency masks, output, jitter offset, motion-vector scale, render size, upscale size, reset, and camera timing.

### Cyberpunk Smoke

- Locate or install Cyberpunk 2077 on the 3080 Ti host and record the exact executable path, store platform, game build, DLSS DLL version, GPU driver, display mode, and benchmark launch args.
- Run with capture disabled first, then hooks enabled, then full capture enabled.
- Confirm no crash, no DLSS regression, and no material frame-time spike during benchmark.
- Resolve the current no-hook-evidence smoke result by launching through the same path a user uses, or by switching the Cyberpunk profile to a proxy filename the game definitely loads on this install.
- Resolve the current no-window launch result. A clean no-proxy launch from the interactive scheduled-task harness with `--launcher-skip --intro-skip -benchmark` starts `Cyberpunk2077.exe`, maps vendor upscaler DLLs, and restores cleanly, but `MainWindowHandle` remains `0`; this must be fixed before hook/capture evidence is meaningful.
- If the installed mod stack must stay in place, validate one of:
  - an installer profile that chains the existing ReShade/Ultimate ASI/RED4ext loader rather than replacing it;
  - an NGX/Streamline-only profile that activates after DLSS is enabled in the benchmark;
  - a tray injection implementation that works in Cyberpunk's current process context.
- Add or select a post-loader trigger that definitely runs in Cyberpunk before rendering. Current evidence says `CreateThread` from `DllMain` is insufficient on this host, direct loader-lock init is unsafe, and DXGI/NGX exports are not being called in the current launch path.

### Dataset Validity

- Validate a real capture shard contains non-degenerate depth, finite motion vectors, correctly scaled MV units, matching LR/HR dimensions, and stable jitter metadata.
- Add an inspection script that opens a captured EXR, reports channels/ranges/NaN counts, and emits thumbnail previews for color/depth/motion.
- Build a parquet index from the real shard and run a tiny training-loader smoke over it.

### Upload/R2

- Provide valid direct R2 S3 credentials or fix the Worker shared-secret mismatch returning HTTP 403.
- Re-run `scripts/capture_cyberpunk_harness.py --start-ingest --mint-local-token` with upload mode and verify the R2 object, sidecar, and index row.

## P1 Release Hardening

- Add native integration tests for staging layout, worker shutdown drain, and failure paths.
- Add a Windows CI or 3080 Ti scheduled job that builds `dxgi.dll`, `oss_capture.dll`, and native tests from a clean checkout.
- Build the MSI from generated WiX source and test install/upgrade/uninstall rollback.
- Add code signing or document the unsigned trusted-tester install flow.
- Add ingest deployment runbook with token-store persistence, R2 credentials, backup, log retention, and alerting.
- Add a supported-games promotion checklist before changing Cyberpunk from `Experimental` to `Known to work`.

## First Cyberpunk Acceptance Run

Minimum passing run:

1. Build Release DLLs on Windows.
2. Generate Cyberpunk installer inputs with `--require-artifacts`.
3. Install to a Cyberpunk test copy.
4. Launch benchmark once with capture disabled.
5. Launch benchmark once in `trickle` or `lite` capture mode.
6. Upload at least 100 accepted real frames.
7. Confirm ingest stores `.exr` and `.json` pairs, no duplicate storm, no unbounded pending-dir growth.
8. Run EXR/channel inspection on at least 20 frames.
9. Build capture index and run a loader/training smoke.

The tool is not production-ready until this run is reproducible from a clean machine or a documented 3080 Ti host state.
