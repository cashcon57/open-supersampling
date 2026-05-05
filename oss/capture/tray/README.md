# OSS Capture tray app — local-only POC

System-tray application that runs on a single Windows machine, captures
gameplay frames + G-buffers to a local drive (<train-host-data>\ or G:\, whichever has
more free space), and lets the user switch capture modes (trickle / lite
/ regular / INSANE) at runtime.

This is the v0 single-machine POC. The shipping multi-user version reuses
the same DLL hook with the upload pathway re-enabled and a server / token
system layered on. For now, captures stay local.

## What this v0 contains

- `app.py` — pystray-based system-tray icon + menu (mode switcher, pause toggle, drive status, captures-folder shortcut, quit).
- `config.py` — persistent JSON config (`%LOCALAPPDATA%\oss-capture\tray-config.json`).
- `storage.py` — drive picker (auto-picks the larger of the configured candidates) and the disk-cap janitor.

## What this v0 does NOT yet contain

- Steam library scanner (parsing `libraryfolders.vdf` + `appmanifest_*.acf`)
- Game-process watcher (Win32 `CreateToolhelp32Snapshot` polling)
- DLL injection (`LoadLibrary` via `CreateRemoteThread`)
- Anti-cheat allowlist enforcement before injection
- Live mode-switching while a game is running (current behavior: mode change takes effect on next game launch)

These ship in a follow-up commit.

## Install (Windows)

```powershell
python -m pip install pystray pillow pywin32
python -m oss.capture.tray
```

The icon appears in the Windows system tray (the carrot menu in the bottom-right of the taskbar). Right-click for the menu.

## Tests

Platform-independent tests covering config round-trip, drive picker, and janitor cleanup:

```bash
pytest tests/capture/test_tray_config.py -v
```

The Windows-specific bits (Win32 process watcher, DLL injector, Steam library scanner) get their own tests when those modules land.
