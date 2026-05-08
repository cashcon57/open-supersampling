# OSS Capture Tool Supported Games

This page is the editorial allowlist for OSS Capture Tool installers. A game is not supported just because the DLL can be copied into its install directory. Each supported game needs an explicit per-game installer, an anti-cheat review, and a smoke test by Cash before it moves to `Known to work`.

Parent design: [OSS Capture Tool design](../superpowers/specs/2026-05-04-oss-capture-tool-design.md). Current implementation phase: [capture tool next phases](../coordination/capture-tool-next-phases.md). User install flow: [install runbook](INSTALL.md).

## Status Labels

| Status | Meaning |
|---|---|
| `Known to work` | Cash has smoke-tested the per-game installer, launch path, capture hook, uploader, uninstall/rollback, and version notes on the listed game build. |
| `Experimental` | Editorially approved as a target, but the hook, installer, or uploader path is still being validated. Use only for trusted testing. |
| `Unsupported` | Do not install the capture tool for this game. This includes anti-cheat-protected titles, games with unsuitable overlays/UIs, or games that have not passed review. |

## Supported Games

| Game | Game ID | Anti-cheat status | DLSS version detected | Tested DLL version | Status | Notes |
|---|---|---|---|---|---|---|
| Cyberpunk 2077 | `cyberpunk-2077` | `NONE` | DLSS SR present; exact version pending first smoke test | Pending first Windows D3D12 smoke test | `Experimental` | Initial validation target. Single-player/offline target with no EAC, BattlEye, VAC, Vanguard, or Ricochet dependency documented in the capture specs. |

## Anti-Cheat Policy

The capture DLL uses game-local DLL loading and D3D12/NGX hooks. That is expected for the supported single-player capture workflow, but it is exactly the kind of behavior anti-cheat systems may block or penalize.

Do not install the capture tool into games protected by kernel-level or competitive anti-cheat systems. Titles using systems such as `EAC`, `BattlEye`, `Vanguard`, or `Ricochet` are off the supported list permanently under this policy. `VAC` or other non-kernel anti-cheat labels still require review before support; the default status is `Unsupported`.

The tool does not try to detect anti-cheat at runtime and should not be treated as a safety mechanism. If a game adds anti-cheat in a patch, its support status should be downgraded until Cash re-reviews it.

## Editorial Process

### How Cash Adds a Game

1. Confirm the game is a good training-data source: real gameplay rendering, useful motion/depth/G-buffer signal, no chat-heavy or competitive UI as the primary capture surface.
2. Record the anti-cheat status using the vocabulary in this file, such as `NONE`, `EAC`, `BattlEye`, `VAC`, `Vanguard`, `Ricochet`, or `Unknown`.
3. Create the per-game build configuration for the installer: game ID, executable name, expected install subdirectory, proxy DLL name, and default capture mode.
4. Smoke-test install, launch, capture, upload, pause/uninstall, and rollback on the target game build.
5. Record the DLSS version detected and the OSS capture DLL version tested.
6. Set the status:
   - `Known to work` only after the smoke test passes.
   - `Experimental` for trusted testing before the full path is proven.
   - `Unsupported` for rejected, anti-cheat-protected, or unsafe targets.

### How Contributors Propose a Game

Open a GitHub issue or pull request proposing a row for this file. Include:

- Game name and store/platform.
- Game executable name and install subdirectory, if known.
- Upscaler support present in the game (`DLSS`, `FSR`, `XeSS`, or native-only).
- Anti-cheat status, with evidence from the game documentation, store page, or installed files.
- Whether the game has multiplayer, competitive modes, chat overlays, or launcher overlays that should exclude it from capture.
- Any known mod-loader, ReShade, Special K, or proxy-DLL conflicts.

Do not install the capture tool into an anti-cheat-protected game to gather evidence. A proposal is only a nomination; Cash decides whether a game enters the allowlist and when it is safe to publish a per-game installer.
