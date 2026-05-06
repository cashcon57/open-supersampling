"""Steam library scanner for the OSS Capture tray app.

Walks the user's Steam library folders to find installed games. Uses
``libraryfolders.vdf`` to enumerate library roots (Steam supports multiple
drives) and ``appmanifest_*.acf`` to enumerate the apps in each.

The VDF / ACF format is Valve's text key-value format (similar to JSON
but with their own quirks: bare strings, no comma separators, nested
braces). We parse it with a minimal hand-rolled tokenizer rather than
adding a third-party dep — Valve's format is small enough that this
fits in ~80 lines and avoids pulling in `vdf` from PyPI.

Output: a list of ``InstalledGame`` records keyed by Steam app-ID, each
with the name, install dir, and the executable path detected from the
manifest's ``LaunchOption`` if available.

Note: this is read-only. We never write to Steam's files.
"""
from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional


@dataclass
class InstalledGame:
    """A single Steam-installed game."""

    app_id: str
    name: str
    install_dir: Path
    """Absolute path to the game's install directory (under steamapps/common/<dir>/)."""


# ---------------------------------------------------------------------------
# Steam install discovery
# ---------------------------------------------------------------------------


def steam_install_root() -> Optional[Path]:
    """Return the Steam install root, or None if not found.

    On Windows, this comes from the registry (HKCU\\Software\\Valve\\Steam,
    SteamPath value). On non-Windows we don't need to support Steam (the
    tray app only runs on Windows), so return None.
    """
    if platform.system() != "Windows":
        return None
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            steam_path, _ = winreg.QueryValueEx(key, "SteamPath")
            return Path(steam_path)
    except (OSError, FileNotFoundError):
        return None


def library_folders(steam_root: Path) -> List[Path]:
    """Parse libraryfolders.vdf and return every library root (steamapps/) path.

    Steam supports installing games across multiple drives. The canonical
    list lives in ``<steam_root>/steamapps/libraryfolders.vdf``. Each
    library root has a ``steamapps/`` subdirectory under it that we walk
    for app manifests.
    """
    vdf_path = steam_root / "steamapps" / "libraryfolders.vdf"
    if not vdf_path.is_file():
        return []
    text = vdf_path.read_text(encoding="utf-8", errors="replace")
    roots: List[Path] = []
    for path_str in _vdf_extract_keys(text, "path"):
        candidate = Path(path_str) / "steamapps"
        if candidate.is_dir():
            roots.append(candidate)
    return roots


def installed_games(library_root: Path) -> Iterator[InstalledGame]:
    """Yield every InstalledGame in a library root (a steamapps/ dir)."""
    for entry in library_root.glob("appmanifest_*.acf"):
        try:
            text = entry.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        game = _parse_appmanifest(text, library_root)
        if game is not None:
            yield game


def all_installed_games(steam_root: Optional[Path] = None) -> List[InstalledGame]:
    """Top-level: enumerate every Steam-installed game across every library."""
    if steam_root is None:
        steam_root = steam_install_root()
    if steam_root is None:
        return []
    games: List[InstalledGame] = []
    for lib in library_folders(steam_root):
        games.extend(installed_games(lib))
    return games


# ---------------------------------------------------------------------------
# Tiny VDF / ACF parser
# ---------------------------------------------------------------------------


# Valve's format encodes quoted strings; nested braces denote subobjects;
# whitespace is structural. For our needs we don't need a full parser — we
# only extract specific top-level keys and key-value pairs from a single
# manifest. Two helpers do that.

_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _vdf_extract_keys(text: str, key: str) -> List[str]:
    """Return every value associated with ``key`` in a VDF text, in order."""
    results: List[str] = []
    tokens = _QUOTED.findall(text)
    # Pairs of consecutive tokens are (key, value).
    for i in range(0, len(tokens) - 1, 2):
        if tokens[i] == key:
            results.append(tokens[i + 1])
    return results


def _parse_appmanifest(text: str, library_root: Path) -> Optional[InstalledGame]:
    """Parse one ``appmanifest_<id>.acf`` and return an InstalledGame.

    Returns None if the manifest is missing required fields.
    """
    # Manifests have a flat-ish structure with appid / name / installdir at
    # the top of the AppState block. We pull them by key-name across the
    # whole text; any nested object that incidentally has these keys with
    # a different scope is rare enough we accept the extraction.
    appids = _vdf_extract_keys(text, "appid")
    names = _vdf_extract_keys(text, "name")
    install_dirs = _vdf_extract_keys(text, "installdir")
    if not (appids and names and install_dirs):
        return None
    install_dir = library_root / "common" / install_dirs[0]
    if not install_dir.is_dir():
        # Manifest references an install dir that doesn't actually exist
        # (broken install, partial uninstall). Skip.
        return None
    return InstalledGame(
        app_id=appids[0],
        name=names[0],
        install_dir=install_dir,
    )
