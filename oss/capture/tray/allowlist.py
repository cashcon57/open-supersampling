"""Anti-cheat / supported-games allowlist for the OSS Capture tray app.

OSS only injects into games where injection won't get the user banned.
This is editorial — maintained by the project, not auto-detected from a
running process. Games with kernel-level anti-cheat (Vanguard, BattlEye,
EAC, Ricochet) are off-list permanently because DLL injection on those
titles trips the anti-cheat and risks account action.

The allowlist ships with the app as a static dict keyed by Steam app-id,
mapping to the per-game metadata the tray needs (display name, expected
exe basename, supported-DLSS-class flag, capture-mode default, notes).

To add a game later: add an entry here and ship a tray-app update.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class AllowedGame:
    """One entry in the supported-games allowlist."""

    app_id: str
    """Steam app ID."""

    display_name: str
    """Human-facing name (e.g. 'Cyberpunk 2077')."""

    exe_basename: str
    """The game's main executable basename (e.g. 'Cyberpunk2077.exe').
    The process watcher matches on this name."""

    game_id: str
    """Project-internal ID (e.g. 'cyberpunk-2077', lowercase + hyphens)
    used for the captures-output subdirectory layout and for the per-game
    config the DLL reads."""

    supports_dlss: bool
    """If True, the game uses DLSS / FSR / XeSS so depth + motion vectors
    + jitter are already provided to the SR API. The OSS capture DLL hooks
    the SR call site to read those buffers. If False, we'd need on-the-fly
    optical flow (not implemented in v0)."""

    notes: Optional[str] = None


# v0 allowlist — verified Day-1 candidates from the README's S7 list. Each
# entry: no kernel anti-cheat, has DLSS / FSR / XeSS support, hook patterns
# documented or trivially derivable.
ALLOWED_GAMES: Dict[str, AllowedGame] = {
    "1091500": AllowedGame(
        app_id="1091500",
        display_name="Cyberpunk 2077",
        exe_basename="Cyberpunk2077.exe",
        game_id="cyberpunk-2077",
        supports_dlss=True,
        notes="Initial validation target. No anti-cheat. Documented hook patterns.",
    ),
    "2271480": AllowedGame(
        app_id="2271480",
        display_name="Alan Wake 2",
        exe_basename="AlanWake2.exe",
        game_id="alan-wake-2",
        supports_dlss=True,
        notes="Heavy ray-traced workload — good showcase. No anti-cheat.",
    ),
    "990080": AllowedGame(
        app_id="990080",
        display_name="Hogwarts Legacy",
        exe_basename="HogwartsLegacy.exe",
        game_id="hogwarts-legacy",
        supports_dlss=True,
    ),
    "1716740": AllowedGame(
        app_id="1716740",
        display_name="Starfield",
        exe_basename="Starfield.exe",
        game_id="starfield",
        supports_dlss=True,
        notes="Bethesda. Modding-friendly.",
    ),
    "1086940": AllowedGame(
        app_id="1086940",
        display_name="Baldur's Gate 3",
        exe_basename="bg3.exe",
        game_id="baldurs-gate-3",
        supports_dlss=True,
    ),
    "1649240": AllowedGame(
        app_id="1649240",
        display_name="Returnal",
        exe_basename="Returnal-Win64-Shipping.exe",
        game_id="returnal",
        supports_dlss=True,
    ),
    "2461850": AllowedGame(
        app_id="2461850",
        display_name="Senua's Saga: Hellblade II",
        exe_basename="HellbladeGame-Win64-Shipping.exe",
        game_id="hellblade-2",
        supports_dlss=True,
    ),
    "1551360": AllowedGame(
        app_id="1551360",
        display_name="Forza Horizon 5",
        exe_basename="ForzaHorizon5.exe",
        game_id="forza-horizon-5",
        supports_dlss=True,
    ),
    "2358720": AllowedGame(
        app_id="2358720",
        display_name="Black Myth: Wukong",
        exe_basename="b1.exe",
        game_id="black-myth-wukong",
        supports_dlss=True,
        notes="Dx12 + DLSS 3.5.",
    ),
    "2050650": AllowedGame(
        app_id="2050650",
        display_name="Resident Evil 4 Remake",
        exe_basename="re4.exe",
        game_id="resident-evil-4",
        supports_dlss=False,  # Uses RE Engine FSR2.
        notes="FSR2 path; OSS shims FSR2 instead of DLSS.",
    ),
}


# Hard blocklist of executables that ship with kernel-level anti-cheat
# loaded into the game process. Even if a user manually adds the game's
# Steam app-id, we refuse to inject when one of these processes is
# resident. The blocklist is the SECOND gate (after the per-game enable
# toggle); never bypass.
KERNEL_ANTICHEAT_PROCESSES = frozenset({
    # Riot Vanguard
    "vgc.exe",
    "vgk.sys",
    # BattlEye
    "BEService.exe",
    "BEDaisy.sys",
    # Easy Anti-Cheat
    "EasyAntiCheat.exe",
    "EasyAntiCheat_EOS.exe",
    "EasyAntiCheat.sys",
    # Ricochet (Activision)
    "Ricochet.sys",
    # nProtect GameGuard
    "GameGuard.des",
    "npggNT.des",
    # Hyperion (Apex)
    "Hyperion.sys",
    # FACEIT (kernel-mode AC)
    "faceit.sys",
})


def lookup_by_app_id(app_id: str) -> Optional[AllowedGame]:
    """Return the AllowedGame for a Steam app-id, or None."""
    return ALLOWED_GAMES.get(app_id)


def lookup_by_exe(exe_basename: str) -> Optional[AllowedGame]:
    """Return the AllowedGame whose exe matches, or None.

    Used by the process watcher: when a new process appears, we check
    whether its executable name matches any allowlisted entry.
    """
    for game in ALLOWED_GAMES.values():
        if game.exe_basename.lower() == exe_basename.lower():
            return game
    return None
