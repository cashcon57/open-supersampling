# OSS Capture Tray App

Current tray status: operator/dev flow with Steam allowlist scan, process watcher, session config writer, and Win32 `LoadLibraryW` injection path. It is not a public safety boundary and has not been proven with real training payloads.

System-tray application that runs on a single Windows machine, captures
gameplay frames + G-buffers to a local drive (<train-host-data>\ or G:\, whichever has
more free space), and lets the user switch capture modes (trickle / lite
/ regular / INSANE) and storage mode (`local` vs `upload`) for future sessions.

## What this contains

- `app.py` — pystray-based system-tray icon + menu (mode switcher, pause toggle, drive status, captures-folder shortcut, quit).
- `config.py` — persistent JSON config (`%LOCALAPPDATA%\oss-capture\tray-config.json`).
- `storage.py` — drive picker (auto-picks the larger of the configured candidates) and the disk-cap janitor.
- `steam_library.py` — Steam library and app manifest discovery.
- `process_watcher.py` — game process polling.
- `dll_inject.py` — bounded Win32 remote-thread DLL injection.
- `allowlist.py` — supported-game and anti-cheat guardrail metadata.
- `session_config.py` — per-launch config writer for DLL/uploader handoff.

Mode and storage-mode changes apply to the next launch/session. `local` mode keeps captures on disk and disables uploader draining; `upload` mode lets the scheduled uploader send captures to ingest/R2. The production path still needs one chosen load strategy: game-local proxy DLL or tray injection, not both.

## Install (Windows)

```powershell
python -m pip install pystray pillow pywin32
python -m oss.capture.tray
```

The icon appears in the Windows system tray (the carrot menu in the bottom-right of the taskbar). Right-click for the menu.

## Tests

Platform-independent tests cover config round-trip, drive picker, janitor cleanup, Steam discovery, launch flow, session config, and the mocked Win32 injection path:

```bash
pytest tests/capture/test_tray_config.py tests/capture/test_tray_launch_flow.py -v
```
