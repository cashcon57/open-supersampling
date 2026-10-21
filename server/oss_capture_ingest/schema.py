"""Frame metadata schema (server-side validator).

This is the **source of truth** for the metadata JSON the Codex-side
uploader sends alongside each EXR. Any change here is a cross-team
break and must be coordinated.

Schema reference (matches design memo §"On-disk capture format"):

    {
      "schema_version": 1,
      "game_id": "cyberpunk-2077",
      "game_version": "2.13",
      "capture_kind": "sr",
      "provider": "dlss",
      "session_uuid": "...",
      "frame_uuid": "...",
      "captured_at_unix": 1777940000,
      "captured_at_monotonic_seconds": 123.45,
      "sequence_index": 42,
      "lr_resolution": [1920, 1080],
      "hr_resolution": [3840, 2160],
      "render_resolution": [1920, 1080],
      "output_resolution": [3840, 2160],
      "dxgi_formats": {"color": "R16G16B16A16_FLOAT", "output": "R16G16B16A16_FLOAT"},
      "hr_source": "dlss-quality" | "dlss-balanced" | "native" | "fsr-..." | ...,
      "jitter_offset_uv": [0.234, 0.781],
      "jitter_offset_px": [0.234, 0.781],
      "exposure_scale": 1.0,
      "motion_mean_magnitude_px": 12.4,
      "channel_presence": {"lr": true, "hr": true, "depth": true, "motion": true},
      "perceptual_hash_64": "0x...",
      "user_consent_token": "<opaque>",
      "uploader_version": "1.0.0"
    }

We deliberately use a hand-rolled validator (not pydantic) so this
module is import-cost-free for the ``--help`` path.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

SUPPORTED_SCHEMA_VERSIONS: Tuple[int, ...] = (1,)
"""All schema_version ints the server currently accepts."""

REQUIRED_FIELDS: Tuple[str, ...] = (
    "schema_version",
    "game_id",
    "game_version",
    "capture_kind",
    "provider",
    "session_uuid",
    "frame_uuid",
    "captured_at_unix",
    "captured_at_monotonic_seconds",
    "sequence_index",
    "lr_resolution",
    "hr_resolution",
    "render_resolution",
    "output_resolution",
    "dxgi_formats",
    "hr_source",
    "jitter_offset_uv",
    "jitter_offset_px",
    "exposure_scale",
    "motion_mean_magnitude_px",
    "channel_presence",
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
_MAX_CAPTURE_DIMENSION = 16_384
_MAX_CAPTURE_PIXELS = 16_384 * 16_384


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
    value = float(v)
    if not math.isfinite(value):
        raise SchemaError(f"{key!r} must be a finite number")
    return value


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
        or any(x <= 0 or x > _MAX_CAPTURE_DIMENSION for x in v)
        or int(v[0]) * int(v[1]) > _MAX_CAPTURE_PIXELS
    ):
        raise SchemaError(
            f"{key!r} must be [width, height] of positive ints within "
            f"{_MAX_CAPTURE_DIMENSION}px and {_MAX_CAPTURE_PIXELS} pixels"
        )
    return list(v)


def _require_float_pair(meta: Dict[str, Any], key: str) -> List[float]:
    v = meta[key]
    if (
        not isinstance(v, list)
        or len(v) != 2
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v)
    ):
        raise SchemaError(f"{key!r} must be [x, y] of numbers")
    result = [float(x) for x in v]
    if any(not math.isfinite(x) for x in result):
        raise SchemaError(f"{key!r} values must be finite")
    return result


def _require_string_map(meta: Dict[str, Any], key: str) -> Dict[str, str]:
    v = meta[key]
    if not isinstance(v, dict):
        raise SchemaError(f"{key!r} must be an object")
    result: Dict[str, str] = {}
    for map_key, map_value in v.items():
        if not isinstance(map_key, str) or not map_key:
            raise SchemaError(f"{key!r} keys must be non-empty strings")
        if not isinstance(map_value, str) or not map_value:
            raise SchemaError(f"{key!r}.{map_key} must be a non-empty string")
        if len(map_key) > 64 or len(map_value) > 128:
            raise SchemaError(f"{key!r}.{map_key} is too long")
        result[map_key] = map_value
    return result


def _require_bool_map(meta: Dict[str, Any], key: str) -> Dict[str, bool]:
    v = meta[key]
    if not isinstance(v, dict):
        raise SchemaError(f"{key!r} must be an object")
    result: Dict[str, bool] = {}
    for map_key, map_value in v.items():
        if not isinstance(map_key, str) or not map_key:
            raise SchemaError(f"{key!r} keys must be non-empty strings")
        if not isinstance(map_value, bool):
            raise SchemaError(f"{key!r}.{map_key} must be a bool")
        if len(map_key) > 64:
            raise SchemaError(f"{key!r}.{map_key} is too long")
        result[map_key] = map_value
    return result


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
    capture_kind = _require_str(meta, "capture_kind", max_len=32)
    if capture_kind not in ("sr", "extrapolation", "supersample_gt"):
        raise SchemaError("capture_kind must be 'sr', 'extrapolation', or 'supersample_gt'")
    provider = _require_str(meta, "provider", max_len=64)

    session_uuid = _require_str(meta, "session_uuid", max_len=64)
    if not _UUID_RE.match(session_uuid):
        raise SchemaError("session_uuid must be a UUID4 string")

    frame_uuid = _require_str(meta, "frame_uuid", max_len=64)
    if not _UUID_RE.match(frame_uuid):
        raise SchemaError("frame_uuid must be a UUID4 string")

    captured_at = _require_number(meta, "captured_at_unix")
    if captured_at <= 0:
        raise SchemaError("captured_at_unix must be > 0")
    captured_at_mono = _require_number(meta, "captured_at_monotonic_seconds")
    if captured_at_mono < 0:
        raise SchemaError("captured_at_monotonic_seconds must be >= 0")
    sequence_index = _require_int(meta, "sequence_index")
    if sequence_index < 0:
        raise SchemaError("sequence_index must be >= 0")

    lr_res = _require_int_pair(meta, "lr_resolution")
    hr_res = _require_int_pair(meta, "hr_resolution")
    render_res = _require_int_pair(meta, "render_resolution")
    output_res = _require_int_pair(meta, "output_resolution")
    if render_res != lr_res:
        raise SchemaError("render_resolution must match lr_resolution")
    if output_res != hr_res:
        raise SchemaError("output_resolution must match hr_resolution")
    dxgi_formats = _require_string_map(meta, "dxgi_formats")
    for required_format in ("color", "output", "depth", "motion_vectors"):
        if required_format not in dxgi_formats:
            raise SchemaError(f"dxgi_formats missing {required_format!r}")

    hr_source = _require_str(meta, "hr_source", max_len=64)

    jitter = _require_float_pair(meta, "jitter_offset_uv")
    if any(value < -8.0 or value > 8.0 for value in jitter):
        raise SchemaError("jitter_offset_uv values must be in [-8, 8]")
    jitter_px = _require_float_pair(meta, "jitter_offset_px")
    if any(value < -64.0 or value > 64.0 for value in jitter_px):
        raise SchemaError("jitter_offset_px values must be in [-64, 64]")
    exposure_scale = _require_number(meta, "exposure_scale")
    if exposure_scale <= 0:
        raise SchemaError("exposure_scale must be > 0")

    motion_mag = _require_number(meta, "motion_mean_magnitude_px")
    if motion_mag < 0:
        raise SchemaError(
            "motion_mean_magnitude_px must be a finite, non-negative number"
        )
    channel_presence = _require_bool_map(meta, "channel_presence")
    for required_channel in ("lr", "hr", "depth", "motion", "normals"):
        if required_channel not in channel_presence:
            raise SchemaError(f"channel_presence missing {required_channel!r}")
    if hr_source == "none" and channel_presence.get("hr"):
        raise SchemaError("hr_source='none' requires channel_presence.hr=false")
    if hr_source != "none" and not channel_presence.get("hr"):
        raise SchemaError("hr_source requires channel_presence.hr=true unless hr_source='none'")

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
    burst_n = meta.get("burst_n")
    if burst_n is not None:
        if not isinstance(burst_n, int) or isinstance(burst_n, bool):
            raise SchemaError("burst_n must be an int when present")
        if burst_n <= 0 or burst_n > 256:
            raise SchemaError("burst_n must be in [1, 256]")
    # Cross-field constraint: if one is set, both must be set.
    if (burst_uuid is None) != (burst_index is None):
        raise SchemaError(
            "burst_uuid and burst_index must be set together (or both omitted)"
        )
    if burst_index is not None and burst_n is not None and burst_index >= burst_n:
        raise SchemaError("burst_index must be less than burst_n")

    # Two-tier burst classification (post-C22 spec). Optional for back-compat;
    # when present, must be one of the documented tiers, and "long" tier
    # additionally implies the HR channel is dropped (hr_source == "none").
    burst_tier = meta.get("burst_tier")
    if burst_tier is not None:
        if burst_tier not in ("short", "long"):
            raise SchemaError(
                "burst_tier must be 'short' or 'long' when present"
            )
        if burst_tier == "long" and hr_source != "none":
            raise SchemaError(
                "burst_tier='long' requires hr_source='none' "
                "(HR channels are dropped in long bursts to save bandwidth)"
            )

    # Capture mode (post-C23 spec). Optional for back-compat; defaults to
    # "lite" semantics on the server side. Stratifies training samples by
    # bandwidth/feature tier so the dataset card can analyze quality
    # contribution per mode.
    capture_mode = meta.get("capture_mode")
    if capture_mode is not None:
        if capture_mode not in ("trickle", "lite", "regular", "INSANE"):
            raise SchemaError(
                "capture_mode must be 'trickle', 'lite', 'regular', or "
                "'INSANE' when present"
            )
        # trickle has TWO capture paths: static singles (no burst fields)
        # and opportunistic motion pairs (burst_index ∈ {0,1},
        # burst_tier='short'). Reject anything else for trickle.
        if capture_mode == "trickle":
            if burst_index is not None and burst_index not in (0, 1):
                raise SchemaError(
                    "capture_mode='trickle' opportunistic pairs are 2 frames; "
                    "burst_index must be 0 or 1"
                )
            if burst_tier is not None and burst_tier != "short":
                raise SchemaError(
                    "capture_mode='trickle' supports only burst_tier='short' "
                    "(opportunistic motion pairs); long bursts are lite+ only"
                )

    return {
        "schema_version": sv,
        "game_id": game_id,
        "game_version": game_version,
        "capture_kind": capture_kind,
        "provider": provider,
        "session_uuid": session_uuid,
        "frame_uuid": frame_uuid,
        "captured_at_unix": captured_at,
        "captured_at_monotonic_seconds": captured_at_mono,
        "sequence_index": sequence_index,
        "lr_resolution": lr_res,
        "hr_resolution": hr_res,
        "render_resolution": render_res,
        "output_resolution": output_res,
        "dxgi_formats": dxgi_formats,
        "hr_source": hr_source,
        "jitter_offset_uv": jitter,
        "jitter_offset_px": jitter_px,
        "exposure_scale": exposure_scale,
        "motion_mean_magnitude_px": motion_mag,
        "channel_presence": channel_presence,
        "perceptual_hash_64": "0x" + raw_hex.lower(),
        "burst_uuid": burst_uuid,
        "burst_index": burst_index,
        "burst_n": burst_n,
        "burst_tier": burst_tier,
        "capture_mode": capture_mode,
        "user_consent_token": consent,
        "uploader_version": uploader_version,
    }
