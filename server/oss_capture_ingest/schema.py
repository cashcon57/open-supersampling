"""Frame metadata schema (server-side validator).

This is the **source of truth** for the metadata JSON the Codex-side
uploader sends alongside each EXR. Any change here is a cross-team
break and must be coordinated.

Schema reference (matches design memo §"On-disk capture format"):

    {
      "schema_version": 1,
      "game_id": "cyberpunk-2077",
      "game_version": "2.13",
      "session_uuid": "...",
      "frame_uuid": "...",
      "captured_at_unix": 1777940000,
      "lr_resolution": [1920, 1080],
      "hr_resolution": [3840, 2160],
      "hr_source": "dlss-quality" | "dlss-balanced" | "native" | "fsr-..." | ...,
      "jitter_offset_uv": [0.234, 0.781],
      "motion_mean_magnitude_px": 12.4,
      "perceptual_hash_64": "0x...",
      "user_consent_token": "<opaque>",
      "uploader_version": "1.0.0"
    }

We deliberately use a hand-rolled validator (not pydantic) so this
module is import-cost-free for the ``--help`` path.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

SUPPORTED_SCHEMA_VERSIONS: Tuple[int, ...] = (1,)
"""All schema_version ints the server currently accepts."""

REQUIRED_FIELDS: Tuple[str, ...] = (
    "schema_version",
    "game_id",
    "game_version",
    "session_uuid",
    "frame_uuid",
    "captured_at_unix",
    "lr_resolution",
    "hr_resolution",
    "hr_source",
    "jitter_offset_uv",
    "motion_mean_magnitude_px",
    "perceptual_hash_64",
    "user_consent_token",
    "uploader_version",
)

# game_id is the bucket-key prefix; restrict it to filesystem-safe chars.
_GAME_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
# UUIDs: accept any 8-4-4-4-12 hex — matches uuid.uuid4() string form.
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class SchemaError(ValueError):
    """Raised by :func:`validate_metadata` for any rejection."""


def _require_int(meta: Dict[str, Any], key: str) -> int:
    v = meta[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise SchemaError(f"{key!r} must be an int, got {type(v).__name__}")
    return v


def _require_number(meta: Dict[str, Any], key: str) -> float:
    v = meta[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SchemaError(f"{key!r} must be a number, got {type(v).__name__}")
    return float(v)


def _require_str(meta: Dict[str, Any], key: str, *, max_len: int = 256) -> str:
    v = meta[key]
    if not isinstance(v, str):
        raise SchemaError(f"{key!r} must be a string")
    if not v:
        raise SchemaError(f"{key!r} must be non-empty")
    if len(v) > max_len:
        raise SchemaError(f"{key!r} too long (max {max_len})")
    return v


def _require_int_pair(meta: Dict[str, Any], key: str) -> List[int]:
    v = meta[key]
    if (
        not isinstance(v, list)
        or len(v) != 2
        or any(isinstance(x, bool) or not isinstance(x, int) for x in v)
        or any(x <= 0 for x in v)
    ):
        raise SchemaError(f"{key!r} must be [width, height] of positive ints")
    return list(v)


def _require_float_pair(meta: Dict[str, Any], key: str) -> List[float]:
    v = meta[key]
    if (
        not isinstance(v, list)
        or len(v) != 2
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v)
    ):
        raise SchemaError(f"{key!r} must be [x, y] of numbers")
    return [float(x) for x in v]


def validate_metadata(meta: Any) -> Dict[str, Any]:
    """Validate + lightly normalize a metadata dict.

    Raises :class:`SchemaError` with a human-readable message on any
    violation. Returns the normalized dict (e.g., float-cast resolution
    pair, uppercased perceptual_hash_64).

    The validator is intentionally strict on field presence and types
    but lenient on ``hr_source`` strings — we don't want to reject new
    upscalers as they ship.
    """
    if not isinstance(meta, dict):
        raise SchemaError("metadata must be a JSON object")

    missing = [f for f in REQUIRED_FIELDS if f not in meta]
    if missing:
        raise SchemaError(f"missing required field(s): {', '.join(missing)}")

    sv = _require_int(meta, "schema_version")
    if sv not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaError(
            f"unsupported schema_version {sv}; "
            f"server supports {SUPPORTED_SCHEMA_VERSIONS}"
        )

    game_id = _require_str(meta, "game_id", max_len=64)
    if not _GAME_ID_RE.match(game_id):
        raise SchemaError(
            "game_id must match [a-z0-9][a-z0-9._-]{0,63} "
            "(lowercase, fs-safe)"
        )

    game_version = _require_str(meta, "game_version", max_len=64)

    session_uuid = _require_str(meta, "session_uuid", max_len=64)
    if not _UUID_RE.match(session_uuid):
        raise SchemaError("session_uuid must be a UUID4 string")

    frame_uuid = _require_str(meta, "frame_uuid", max_len=64)
    if not _UUID_RE.match(frame_uuid):
        raise SchemaError("frame_uuid must be a UUID4 string")

    captured_at = _require_number(meta, "captured_at_unix")
    if captured_at <= 0:
        raise SchemaError("captured_at_unix must be > 0")

    lr_res = _require_int_pair(meta, "lr_resolution")
    hr_res = _require_int_pair(meta, "hr_resolution")

    hr_source = _require_str(meta, "hr_source", max_len=64)

    jitter = _require_float_pair(meta, "jitter_offset_uv")

    motion_mag = _require_number(meta, "motion_mean_magnitude_px")
    if motion_mag < 0 or motion_mag != motion_mag:  # NaN check
        raise SchemaError(
            "motion_mean_magnitude_px must be a finite, non-negative number"
        )

    phash = _require_str(meta, "perceptual_hash_64", max_len=32)
    # Accept either "0x..." hex or plain hex. Normalize to lowercase 0x form.
    raw_hex = phash[2:] if phash.lower().startswith("0x") else phash
    if not re.fullmatch(r"[0-9a-fA-F]{1,16}", raw_hex):
        raise SchemaError(
            "perceptual_hash_64 must be hex (optionally 0x-prefixed), "
            "≤ 16 nibbles"
        )

    consent = _require_str(meta, "user_consent_token", max_len=128)
    uploader_version = _require_str(meta, "uploader_version", max_len=32)

    # Burst-mode (post-ce9bf3b spec revision): each ACCEPT decision from the
    # sampler enqueues N consecutive Present-frame captures sharing a
    # ``burst_uuid`` with per-frame ``burst_index`` ∈ [0, N-1]. Optional for
    # back-compat with older single-frame DLLs; required-when-present is
    # enforced by Codex's C18+C21 wiring.
    burst_uuid = meta.get("burst_uuid")
    if burst_uuid is not None:
        if not isinstance(burst_uuid, str) or not _UUID_RE.match(burst_uuid):
            raise SchemaError("burst_uuid must be a UUID4 string when present")
    burst_index = meta.get("burst_index")
    if burst_index is not None:
        if not isinstance(burst_index, int) or isinstance(burst_index, bool):
            raise SchemaError("burst_index must be an int when present")
        if burst_index < 0 or burst_index > 64:
            raise SchemaError("burst_index must be in [0, 64]")
    # Cross-field constraint: if one is set, both must be set.
    if (burst_uuid is None) != (burst_index is None):
        raise SchemaError(
            "burst_uuid and burst_index must be set together (or both omitted)"
        )

    return {
        "schema_version": sv,
        "game_id": game_id,
        "game_version": game_version,
        "session_uuid": session_uuid,
        "frame_uuid": frame_uuid,
        "captured_at_unix": captured_at,
        "lr_resolution": lr_res,
        "hr_resolution": hr_res,
        "hr_source": hr_source,
        "jitter_offset_uv": jitter,
        "motion_mean_magnitude_px": motion_mag,
        "perceptual_hash_64": "0x" + raw_hex.lower(),
        "burst_uuid": burst_uuid,
        "burst_index": burst_index,
        "user_consent_token": consent,
        "uploader_version": uploader_version,
    }
