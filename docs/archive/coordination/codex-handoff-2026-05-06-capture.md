# Codex Capture Handoff - 2026-05-06

## Tray app pass

- Added local-only per-session config writing at `%LOCALAPPDATA%\oss-capture\config.json`.
- Added Win32 DLL injection path with a non-Windows no-op result for development and tests.
- Wired process-launch events through tray config checks: paused state, per-game enable flag, output-drive resolution, config write, then DLL injection.
- Added tray menu entries for installed allowlisted Steam games.
- Updated the process watcher to use Toolhelp32 polling.

## Integration risk

- I could not find the DLL-side JSON config reader in `oss/gaussian/interception/` with:
  `rg -n "config.json|LOCALAPPDATA|output_dir|capture_api_base|pending_dir|proxy_dll|install_token|schema_version|game_exe_name" oss/gaussian/interception -S`
- The tray writer follows the `scripts/build_capture_installer.py:build_config` schema where it applies, but intentionally omits upload-only keys: `capture_api_base`, `endpoints`, `install_token`, and uploader retry settings.
- If the DLL config reader exists in a generated or untracked Windows-only source file, no C++ change is needed. If it does not, the next C++ pass must add the reader before the tray-to-DLL bridge can work end-to-end.

## Test note

- `./venv-py312/bin/python -m pytest tests/capture/ -q` ran, but this sandbox blocks binding a localhost HTTP server, so the two uploader e2e tests in `tests/capture/test_e2e.py` failed with `PermissionError: [Errno 1] Operation not permitted`.
- `./venv-py312/bin/python -m pytest tests/capture/ -q --ignore=tests/capture/test_e2e.py` passed: 76 passed, 1 skipped.
