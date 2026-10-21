"""Persistent config for the OSS Capture tray app.

The tray app writes a single JSON file at ``%LOCALAPPDATA%\\oss-capture\\
tray-config.json`` (Windows) or ``~/.local/share/oss-capture/tray-config.json``
(Linux/macOS, only used during cross-platform development on a non-Windows
machine).

The config holds:
  - active capture mode (trickle / lite / regular / INSANE)
  - capture storage mode (local only vs upload to ingest/R2)
  - capture API base URL used in the per-session DLL/uploader config
  - output drive override (None = auto-pick the drive with more free space)
  - paused flag (tray "Pause captures" toggle)
  - dry-run launch flag (write session config but do not inject)
  - per-game enable list (dict[str game_id, bool enabled])
  - disk-cap bytes for the oldest-first janitor

The config is read on every game launch (which is when the DLL is injected
and the chosen mode + output dir are baked into the per-session config the
DLL reads). Live mode-switching while a game is already running is a v1
followup; the POC behavior is "next launch picks up the new mode".
"""
from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

CAPTURE_MODES = ("trickle", "lite", "regular", "INSANE")
CAPTURE_STORAGE_MODES = ("local", "upload")
DEFAULT_CAPTURE_API_BASE = "https://capture.oss-supersampling.dev"

# 200 GB default soft-cap on captured-frame footprint per output drive.
# When the janitor sees the captures dir cross this, it deletes oldest
# frames first until back under cap.
DEFAULT_DISK_CAP_BYTES = 200 * 1024 * 1024 * 1024


def _config_dir() -> Path:
    """Return the OS-appropriate config directory, creating it if needed."""
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA")
        if base is None:
            # %LOCALAPPDATA% is always set on real Windows; this branch is
            # defensive for tests / CI mocking.
            base = os.path.expanduser("~\\AppData\\Local")
        path = Path(base) / "oss-capture"
    else:
        # Cross-platform dev path. Tray app does not actually FUNCTION on
        # non-Windows (no DLL injection, no game launch detect), but the
        # config + drive-picker code paths are testable here.
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        path = Path(base) / "oss-capture"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_file_path() -> Path:
    return _config_dir() / "tray-config.json"


@dataclass
class TrayConfig:
    """The on-disk tray-app config schema."""

    capture_mode: str = "lite"
    """One of trickle / lite / regular / INSANE. lite is the documented
    default in the v0 OSS Capture Tool spec."""

    output_drive_override: Optional[str] = None
    """If set, the tray uses this drive root (e.g. 'G:\\\\') as the captures
    output. If None, the tray auto-picks the drive with more free space."""

    paused: bool = False
    """If True, the tray-app menu is in 'Pause captures' state and no DLL
    injection is performed when a game launches."""

    enabled_games: Dict[str, bool] = field(default_factory=dict)
    """Per-game enable flags keyed by game_id (e.g. 'cyberpunk-2077').
    Default is opt-in: a newly-detected game is OFF until the user enables
    it from the tray menu."""

    disk_cap_bytes: int = DEFAULT_DISK_CAP_BYTES
    """Soft cap on total captures footprint per output drive. The janitor
    deletes oldest frames first when over."""

    capture_api_base: str = DEFAULT_CAPTURE_API_BASE
    """Base URL for capture ingest/session endpoints written into each
    per-session config. Kept in tray config so programmatic capture runs can
    point at staging/local endpoints without rebuilding installer artifacts."""

    dry_run_launch: bool = False
    """If True, launch handling writes the same per-session config but skips
    DLL injection. Used for smoke tests and scripted validation on non-game
    hosts."""

    schema_version: int = 1

    capture_storage_mode: str = "local"
    """'local' captures data only to disk and disables uploader draining.
    'upload' enables the uploader path to ingest/R2."""

    def __post_init__(self) -> None:
        if self.capture_mode not in CAPTURE_MODES:
            raise ValueError(
                f"capture_mode must be one of {CAPTURE_MODES}; got {self.capture_mode!r}"
            )
        if self.capture_storage_mode not in CAPTURE_STORAGE_MODES:
            raise ValueError(
                "capture_storage_mode must be one of "
                f"{CAPTURE_STORAGE_MODES}; got {self.capture_storage_mode!r}"
            )
        if not isinstance(self.capture_api_base, str) or not self.capture_api_base:
            raise ValueError("capture_api_base must be non-empty")
        self.capture_api_base = self.capture_api_base.rstrip("/")
        if self.disk_cap_bytes < 1024 * 1024 * 1024:
            raise ValueError(
                f"disk_cap_bytes must be >= 1 GiB; got {self.disk_cap_bytes}"
            )


def load() -> TrayConfig:
    """Load the tray config. Returns defaults if the file does not exist."""
    path = config_file_path()
    if not path.exists():
        return TrayConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Corrupt or partially-written config; fall back to defaults rather
        # than crash the tray app. The user can fix via menu actions.
        return TrayConfig()
    # Drop unknown keys so future schema-version bumps don't crash the
    # current code path; require_version-bump-first is enforced at write
    # time via schema_version.
    known = {f.name for f in TrayConfig.__dataclass_fields__.values()}
    raw = {k: v for k, v in raw.items() if k in known}
    try:
        return TrayConfig(**raw)
    except (TypeError, ValueError):
        return TrayConfig()


def save(cfg: TrayConfig) -> None:
    """Write the tray config atomically.

    Writes to a temp file then renames so a crash mid-write can never leave
    a half-written tray-config.json on disk.
    """
    path = config_file_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(asdict(cfg), indent=2, sort_keys=True)
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
