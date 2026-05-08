# OSS Capture Tool Install Runbook

This runbook is for the per-game OSS Capture Tool MSI. The tool captures selected gameplay frames for OSS training data, uploads accepted captures automatically, and deletes local frame files after upload or terminal rejection.

Maintainer context: [capture next phases](../coordination/capture-tool-next-phases.md) and [capture tool design](../superpowers/specs/2026-05-04-oss-capture-tool-design.md).

Before installing, check the [supported games policy](SUPPORTED_GAMES.md).

## Install

1. Download the MSI for your game.
2. Run the MSI.
3. When prompted, point the installer at the game's install directory.
   - Example shape: `...\SteamLibrary\steamapps\common\<Game Name>\`
   - The installer verifies the expected game executable under the game install tree.
4. Choose a capture mode. `lite` is the default.
5. Click **Install**.
6. Done. Leave the OSS Capture tray app running while you play.

## What Gets Installed

The installer creates:

- `%LOCALAPPDATA%\oss-capture\`
  - `config.json`
  - uploader executable and support files
  - logs
  - pending capture files under `pending\<game_id>\<session_uuid>\` while they are waiting to upload
- A game-local proxy DLL in the game's binary directory, usually:
  - `<game install dir>\bin\x64\dxgi.dll`

If a proxy DLL with the same name already exists, the installer backs it up before installing the OSS Capture proxy.

Pending captures are temporary. Successful uploads are deleted immediately. Rejected captures are deleted instead of retried. Network/server failures are retried with backoff, then dropped rather than filling the disk.

## Tray Menu

Use the OSS Capture tray icon for routine controls:

- **Change capture mode**: switch between capture modes after install.
- **Open log dir**: open the local OSS Capture log directory.
- **Force flush pending**: immediately ask the uploader to drain pending captures instead of waiting for the next scheduled pass.

Mode changes apply to future capture work. If a game is already running, restart the game if the new mode does not take effect immediately.

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
