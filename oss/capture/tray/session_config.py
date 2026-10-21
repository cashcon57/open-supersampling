"""Per-session config writer for the local-only OSS Capture tray flow.

The injected DLL reads ``%LOCALAPPDATA%\\oss-capture\\config.json`` at
startup. The tray rewrites that file immediately before injecting into a
game so the DLL sees the selected mode and output directory for that
specific launch.

The config is session-scoped: every launch gets a fresh session UUID and
explicit output directories so scripted Cyberpunk capture runs can tell
which frames belong to which game launch.
"""
from __future__ import annotations

import json
import os
import platform
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from oss.capture.tray import config as cfg_mod
from oss.capture.tray.allowlist import AllowedGame


DEFAULT_PROXY_DLL_NAME = "oss_capture.dll"
DEFAULT_INSTALLER_VERSION = "0.2.0-dev"
DEFAULT_CAPTURE_API_BASE = cfg_mod.DEFAULT_CAPTURE_API_BASE


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
    """Config schema consumed by the injected capture DLL and uploader."""

    game_id: str
    game_exe_name: str
    session_uuid: str
    capture_mode: str
    capture_api_base: str
    output_root_dir: str
    game_output_dir: str
    session_output_dir: str
    output_dir: str
    pending_dir: str
    capture_storage_mode: str = "local"
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
        if self.capture_storage_mode not in cfg_mod.CAPTURE_STORAGE_MODES:
            raise ValueError(
                "capture_storage_mode must be one of "
                f"{cfg_mod.CAPTURE_STORAGE_MODES}; got {self.capture_storage_mode!r}"
            )
        if not self.game_id:
            raise ValueError("game_id must be non-empty")
        try:
            uuid.UUID(self.session_uuid)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"session_uuid {self.session_uuid!r} is not a UUID") from exc
        if not self.game_exe_name.lower().endswith(".exe"):
            raise ValueError(f"game_exe_name {self.game_exe_name!r} must end with .exe")
        if not isinstance(self.capture_api_base, str) or not self.capture_api_base:
            raise ValueError("capture_api_base must be non-empty")
        object.__setattr__(self, "capture_api_base", self.capture_api_base.rstrip("/"))
        if not self.output_root_dir:
            raise ValueError("output_root_dir must be non-empty")
        if not self.game_output_dir:
            raise ValueError("game_output_dir must be non-empty")
        if not self.session_output_dir:
            raise ValueError("session_output_dir must be non-empty")
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty")
        if not self.pending_dir:
            raise ValueError("pending_dir must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["ingest_url"] = self.capture_api_base + "/ingest"
        payload["endpoints"] = {
            "ingest": self.capture_api_base + "/ingest",
            "session_start": self.capture_api_base + "/session/start",
            "stats": self.capture_api_base + "/stats",
        }
        payload["consent"] = {"mode": self.capture_mode}
        return payload


def build_session_config(
    *,
    game: AllowedGame,
    capture_mode: str,
    output_dir: Path | str,
    capture_api_base: str = DEFAULT_CAPTURE_API_BASE,
    capture_storage_mode: str = "local",
    session_uuid: Optional[str] = None,
) -> SessionConfig:
    """Build a per-session config for an allowlisted game.

    ``output_dir`` is the per-game output directory. The session writer
    appends ``session_uuid`` and writes that session path both as
    ``session_output_dir`` and as the legacy ``output_dir`` key.
    """
    if capture_storage_mode not in cfg_mod.CAPTURE_STORAGE_MODES:
        raise ValueError(
            "capture_storage_mode must be one of "
            f"{cfg_mod.CAPTURE_STORAGE_MODES}; got {capture_storage_mode!r}"
        )
    session_id = session_uuid or str(uuid.uuid4())
    requested_game_output_dir = Path(output_dir)
    output_root_dir = requested_game_output_dir.parent / capture_storage_mode
    game_output_dir = output_root_dir / requested_game_output_dir.name
    session_output_dir = game_output_dir / session_id
    return SessionConfig(
        game_id=game.game_id,
        game_exe_name=game.exe_basename,
        session_uuid=session_id,
        capture_mode=capture_mode,
        capture_api_base=capture_api_base,
        capture_storage_mode=capture_storage_mode,
        output_root_dir=str(output_root_dir),
        game_output_dir=str(game_output_dir),
        session_output_dir=str(session_output_dir),
        output_dir=str(session_output_dir),
        pending_dir=str(output_root_dir),
    )


def _read_existing_config(path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def write_session_config(
    *,
    game: AllowedGame,
    capture_mode: str,
    output_dir: Path | str,
    capture_api_base: Optional[str] = None,
    capture_storage_mode: Optional[str] = None,
    session_uuid: Optional[str] = None,
    path: Optional[Path] = None,
) -> Path:
    """Atomically write ``config.json`` and return its path."""
    target = path or config_file_path()
    existing = _read_existing_config(target)
    resolved_capture_api_base = (
        capture_api_base
        or str(existing.get("capture_api_base") or DEFAULT_CAPTURE_API_BASE)
    )
    resolved_storage_mode = (
        capture_storage_mode
        or str(existing.get("capture_storage_mode") or "local")
    )
    existing_install_token = existing.get("install_token")
    if resolved_storage_mode == "upload" and not existing_install_token:
        raise ValueError(
            "upload capture_storage_mode requires an installer-provided install_token"
        )
    session = build_session_config(
        game=game,
        capture_mode=capture_mode,
        output_dir=output_dir,
        capture_api_base=resolved_capture_api_base,
        capture_storage_mode=resolved_storage_mode,
        session_uuid=session_uuid,
    )
    Path(session.session_output_dir).mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(existing)
    session_payload = session.to_dict()
    if isinstance(existing.get("consent"), dict) and isinstance(session_payload.get("consent"), dict):
        merged_consent = dict(existing["consent"])
        merged_consent.update(session_payload["consent"])
        session_payload["consent"] = merged_consent
    payload.update(session_payload)
    payload["install_token"] = str(existing_install_token or uuid.uuid4().hex)
    # Separate, non-secret id for capture metadata. The DLL must never copy
    # install_token (the upload credential) into capture files.
    payload["consent_id"] = str(existing.get("consent_id") or uuid.uuid4().hex)
    for key in (
        "proxy_dll_name",
        "installer_version",
        "pending_dir_cap_bytes",
        "max_frame_bytes",
        "max_attempts",
        "max_pending_bytes",
        "scan_interval_seconds",
        "uploader_retry_attempts",
        "uploader_retry_max_seconds",
        "incomplete_grace_seconds",
    ):
        if key in existing:
            payload[key] = existing[key]
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return target
