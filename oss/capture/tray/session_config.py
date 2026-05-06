"""Per-session config writer for the local-only OSS Capture tray flow.

The injected DLL reads ``%LOCALAPPDATA%\\oss-capture\\config.json`` at
startup. The tray rewrites that file immediately before injecting into a
game so the DLL sees the selected mode and output directory for that
specific launch.

This is intentionally local-only: no upload token, no API base, no
endpoints, and no uploader retry config.
"""
from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from oss.capture.tray import config as cfg_mod
from oss.capture.tray.allowlist import AllowedGame


DEFAULT_PROXY_DLL_NAME = "oss_capture.dll"
DEFAULT_INSTALLER_VERSION = "0.2.0-dev"


def _local_config_dir() -> Path:
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA")
        if base is None:
            base = os.path.expanduser("~\\AppData\\Local")
        path = Path(base) / "oss-capture"
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        path = Path(base) / "oss-capture"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_file_path() -> Path:
    return _local_config_dir() / "config.json"


@dataclass(frozen=True)
class SessionConfig:
    """Local-only config schema consumed by the injected capture DLL."""

    game_id: str
    game_exe_name: str
    capture_mode: str
    output_dir: str
    schema_version: int = 1
    proxy_dll_name: str = DEFAULT_PROXY_DLL_NAME
    installer_version: str = DEFAULT_INSTALLER_VERSION
    suggested_capture_rate_per_min: float = 3.0
    pending_dir_cap_bytes: int = 2 * 1024 * 1024 * 1024
    max_frame_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.capture_mode not in cfg_mod.CAPTURE_MODES:
            raise ValueError(
                f"capture_mode must be one of {cfg_mod.CAPTURE_MODES}; "
                f"got {self.capture_mode!r}"
            )
        if not self.game_id:
            raise ValueError("game_id must be non-empty")
        if not self.game_exe_name.lower().endswith(".exe"):
            raise ValueError(f"game_exe_name {self.game_exe_name!r} must end with .exe")
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["consent"] = {"mode": self.capture_mode}
        return payload


def build_session_config(
    *,
    game: AllowedGame,
    capture_mode: str,
    output_dir: Path | str,
) -> SessionConfig:
    """Build a local-only per-session config for an allowlisted game."""
    return SessionConfig(
        game_id=game.game_id,
        game_exe_name=game.exe_basename,
        capture_mode=capture_mode,
        output_dir=str(output_dir),
    )


def write_session_config(
    *,
    game: AllowedGame,
    capture_mode: str,
    output_dir: Path | str,
    path: Optional[Path] = None,
) -> Path:
    """Atomically write ``config.json`` and return its path."""
    target = path or config_file_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_session_config(
        game=game,
        capture_mode=capture_mode,
        output_dir=output_dir,
    ).to_dict()
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)
    return target
