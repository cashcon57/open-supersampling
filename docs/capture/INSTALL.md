# OSS Capture Tool Install Runbook

> Pre-release/operator-only. No public MSI is production-ready yet. Today the harness can validate ingest and synthetic upload; real D3D12/DLSS frame capture is not accepted as training data until the Cyberpunk readiness gate passes.

This runbook is for the per-game OSS Capture Tool MSI once the production readiness gate is green. The intended tool captures selected gameplay frames for OSS training data, uploads accepted captures automatically, and deletes local frame files after successful upload or terminal rejection.

Maintainer context: [capture next phases](../coordination/capture-tool-next-phases.md) and [capture tool design](../superpowers/specs/2026-05-04-oss-capture-tool-design.md).

Before installing, check the [supported games policy](SUPPORTED_GAMES.md).

## Production Readiness Checklist

- [ ] MSI built from WiX and installed on a clean Windows host.
- [ ] DLL loads through game-local proxy without game crash.
- [ ] Present/ExecuteCommandLists hooks active; hook status not failed.
- [ ] Active upscaler dispatch params decoded for LR, HR, depth, motion, and jitter.
- [ ] EXR output contains required channels, not `.raw` staging dumps.
- [ ] Uploader deletes 2xx and terminal 4xx, preserves 429/5xx/network retryables, and enforces the pending cap.
- [ ] R2 object, metadata sidecar, and `_index.parquet` row are visible.
- [ ] Uninstall restores prior proxy DLL and removes scheduled task/cache.

## Install

1. Download the MSI for your game.
2. Run the MSI.
3. When prompted, point the installer at the game's install directory.
   - Example shape: `...\SteamLibrary\steamapps\common\<Game Name>\`
   - The installer verifies the expected game executable under the game install tree.
4. Choose a capture mode. `lite` is the default.
5. Choose a storage mode:
   - `local`: capture to disk only; nothing is uploaded.
   - `upload`: send accepted captures to the configured ingest service/R2 bucket.
6. Click **Install**.
7. Done. Leave the OSS Capture tray app running while you play.

## What Gets Installed

The installer creates:

- `%LOCALAPPDATA%\oss-capture\`
  - `config.json`
  - uploader executable and support files
  - logs
  - pending capture files under `pending\<game_id>\<session_uuid>\` while they are waiting to upload
- A game-local proxy DLL in the game's binary directory, usually:
  - `<game install dir>\bin\x64\dxgi.dll`
  - `<game install dir>\bin\x64\nvngx_dlss.dll`
  - `<game install dir>\bin\x64\sl.interposer.dll`
  - `<game install dir>\bin\x64\ffx_fsr3_x64.dll`

`dxgi.dll` installs the DXGI/D3D12 hooks. Vendor upscaler proxies such as `nvngx_dlss.dll`, `sl.interposer.dll`, and `ffx_fsr3_x64.dll` install pass-through hooks and forward to the real vendor DLL. If a proxy DLL with the same name already exists, the installer backs it up before installing the OSS Capture proxy.

In `local` storage mode, captures stay on disk and the uploader leaves them alone. In `upload` storage mode, pending captures are temporary: successful uploads are deleted immediately, terminal rejected captures are deleted instead of retried, and rate limits/network/server failures stay pending for later scheduled passes. The pending-dir cap evicts oldest frames if the uploader cannot catch up.

## Tray Menu

Use the OSS Capture tray icon for routine controls:

- **Change capture mode**: switch between capture modes after install.
- **Change storage mode**: switch between local-only capture and upload-to-R2 for future sessions.
- **Open log dir**: open the local OSS Capture log directory.
- **Force flush pending**: immediately ask the uploader to drain pending captures instead of waiting for the next scheduled pass.

Mode changes apply to future capture work. If a game is already running, restart the game if the new mode does not take effect immediately.

## Cyberpunk Readiness Harness

For the Cyberpunk 2077 validation host, use the programmatic harness before installing or dogfooding a build. It verifies ingest, generates the Cyberpunk installer config inputs, optionally launches the game benchmark, runs a synthetic fallback capture when a token is available, and prints a JSON readiness report.

Local Windows run from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\cyberpunk_capture_harness.ps1 `
  -RepoPath C:\Users\<you>\OpenSuperSampling `
  -StartIngest `
  -MintLocalToken
```

SSH run through PowerShell remoting over an existing SSH config or tailnet hostname:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\cyberpunk_capture_harness.ps1 `
  -HostName <windows-3080ti-host> `
  -RepoPath C:\Users\<you>\OpenSuperSampling `
  -Server http://127.0.0.1:8080 `
  -StartIngest `
  -MintLocalToken
```

If Cyberpunk is not in a standard install path, add `-CyberpunkExe "D:\Games\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe"`. Add `-LaunchCyberpunk` only when you want the harness to start the game; otherwise it only reports whether the executable was found. To use an existing ingest token, pass `-Token` or set `OSS_CAPTURE_INSTALL_TOKEN` instead of `-MintLocalToken`.

## Bandwidth

Expected upload volume while actively playing:

| Mode | Expected bandwidth |
|---|---:|
| `lite` | ~500 MB/h |
| `regular` | ~2 GB/h |
| `full` | ~6 GB/h |

Actual usage can be lower when the sampler rejects menus, loading screens, duplicate scenes, or frames with unusable buffers.

## Uninstall

1. Open Windows **Control Panel**.
2. Go to **Programs and Features**.
3. Uninstall the OSS Capture Tool entry for the game.
4. Delete `%LOCALAPPDATA%\oss-capture\` if it still exists.

The uninstaller should restore any backed-up game-local proxy DLL and remove the uploader schedule. Deleting `%LOCALAPPDATA%\oss-capture\` removes local config, logs, and any pending captures.

## Anti-Cheat Disclaimer

Anti-cheat protection is not enabled in the tool. The tool does not try to detect, bypass, or work around anti-cheat systems.

Only use the OSS Capture Tool with games listed as supported by the project and tested for this capture flow. Compatibility testing reduces risk, but there is no warranty. You are responsible for deciding whether installing a game-local proxy DLL is acceptable for your game, account, and platform rules.
