"""Build monthly capture parquet indexes from R2 sidecar JSON.

Primary C.6 usage:

    python -m server.scripts.build_capture_index --month 2026-05 --game cyberpunk-2077

The command walks:

    <game_id>/<YYYY-MM>/

and uploads:

    <game_id>/<YYYY-MM>/_index.parquet

Legacy helper behavior for the older root-level daily index tests is kept
for compatibility: callers may still pass an explicit prefix to
``collect_rows`` or ``date_str`` to ``upload_index_to_bucket``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple


log = logging.getLogger(__name__)


REQUIRED_INDEX_COLUMNS: Tuple[str, ...] = (
    "game_id",
    "capture_mode",
    "session_uuid",
    "frame_uuid",
    "frame_idx",
    "ts",
    "contributor_hash",
    "bytes",
)

# Keep a small set of existing columns so older tests and downstream readers
# that already consumed the pre-C.6 index do not break.
INDEX_COLUMNS: Tuple[str, ...] = REQUIRED_INDEX_COLUMNS + (
    "game_version",
    "captured_at_unix",
    "lr_width",
    "lr_height",
    "hr_width",
    "hr_height",
    "hr_source",
    "motion_mean_magnitude_px",
    "perceptual_hash_64",
    "uploader_version",
    "exr_key",
    "json_key",
    "frame_bytes",
    "content_sha256",
)

_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_VALID_CAPTURE_MODES = {"trickle", "lite", "regular", "INSANE"}


def _month_prefix(game_id: str, month: str) -> str:
    if not game_id:
        raise ValueError("game_id must be non-empty")
    if not _MONTH_RE.fullmatch(month):
        raise ValueError("month must be YYYY-MM")
    return f"{game_id}/{month}/"


def _capture_mode_from_key(json_key: str) -> Optional[str]:
    parts = json_key.split("/")
    if len(parts) >= 5 and parts[2] in _VALID_CAPTURE_MODES:
        return parts[2]
    return None


def _stable_contributor_hash(meta: Dict[str, Any]) -> str:
    value = meta.get("contributor_hash")
    if value:
        return str(value)
    token = meta.get("user_consent_token")
    if not token:
        return ""
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def _int_or_default(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_from_meta(
    meta: Dict[str, Any],
    json_key: str,
    *,
    object_sizes: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Project one sidecar JSON dict to an index row."""
    lr = meta.get("lr_resolution") or [0, 0]
    hr = meta.get("hr_resolution") or [0, 0]
    exr_key = json_key[:-5] + ".exr" if json_key.endswith(".json") else ""
    frame_bytes = _int_or_default(meta.get("frame_bytes"))
    if frame_bytes <= 0 and object_sizes is not None:
        frame_bytes = int(object_sizes.get(exr_key, 0))

    captured_at = _float_or_default(meta.get("captured_at_unix"))
    frame_idx = _int_or_default(meta.get("frame_idx"), default=-1)
    if frame_idx < 0:
        frame_idx = _int_or_default(meta.get("burst_index"))

    capture_mode = meta.get("capture_mode") or _capture_mode_from_key(json_key) or "lite"

    return {
        "game_id": str(meta.get("game_id", "")),
        "capture_mode": str(capture_mode),
        "session_uuid": str(meta.get("session_uuid", "")),
        "frame_uuid": str(meta.get("frame_uuid", "")),
        "frame_idx": frame_idx,
        "ts": captured_at,
        "contributor_hash": _stable_contributor_hash(meta),
        "bytes": frame_bytes,
        "game_version": str(meta.get("game_version", "")),
        "captured_at_unix": captured_at,
        "lr_width": _int_or_default(lr[0] if len(lr) >= 1 else 0),
        "lr_height": _int_or_default(lr[1] if len(lr) >= 2 else 0),
        "hr_width": _int_or_default(hr[0] if len(hr) >= 1 else 0),
        "hr_height": _int_or_default(hr[1] if len(hr) >= 2 else 0),
        "hr_source": str(meta.get("hr_source", "")),
        "motion_mean_magnitude_px": _float_or_default(
            meta.get("motion_mean_magnitude_px")
        ),
        "perceptual_hash_64": str(meta.get("perceptual_hash_64", "")),
        "uploader_version": str(meta.get("uploader_version", "")),
        "exr_key": exr_key,
        "json_key": json_key,
        "frame_bytes": frame_bytes,
        "content_sha256": str(meta.get("content_sha256", "")),
    }


def collect_rows(
    r2_client: Any,
    *,
    game_id: Optional[str] = None,
    month: Optional[str] = None,
    prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Walk R2 and return one row per frame sidecar JSON.

    Passing ``game_id`` and ``month`` is the C.6 path and walks
    ``<game_id>/<YYYY-MM>/``. Passing ``prefix`` preserves the previous
    generic behavior used by older tests.
    """
    if prefix is None:
        prefix = _month_prefix(game_id, month) if game_id and month else ""

    objects = list(r2_client.iter_objects(prefix=prefix))
    object_sizes = {key: size for key, size in objects}
    rows: List[Dict[str, Any]] = []

    for key, _size in objects:
        if not key.endswith(".json"):
            continue
        if key.startswith("_index_") or "/_index" in key:
            continue
        if key.startswith("_dedup/"):
            continue
        try:
            body = r2_client.get_bytes(key)
            meta = json.loads(body.decode("utf-8"))
        except Exception as exc:
            log.warning("failed to read/parse %s: %s", key, exc)
            continue
        if not isinstance(meta, dict):
            log.warning("skipping %s: sidecar JSON is not an object", key)
            continue
        try:
            rows.append(_row_from_meta(meta, key, object_sizes=object_sizes))
        except Exception as exc:
            log.warning("failed to project %s: %s", key, exc)

    rows.sort(key=lambda row: (row["game_id"], row["ts"], row["frame_uuid"]))
    return rows


def _arrow_schema() -> Any:
    import pyarrow as pa  # noqa: PLC0415

    return pa.schema(
        [
            ("game_id", pa.string()),
            ("capture_mode", pa.string()),
            ("session_uuid", pa.string()),
            ("frame_uuid", pa.string()),
            ("frame_idx", pa.int64()),
            ("ts", pa.float64()),
            ("contributor_hash", pa.string()),
            ("bytes", pa.int64()),
            ("game_version", pa.string()),
            ("captured_at_unix", pa.float64()),
            ("lr_width", pa.int64()),
            ("lr_height", pa.int64()),
            ("hr_width", pa.int64()),
            ("hr_height", pa.int64()),
            ("hr_source", pa.string()),
            ("motion_mean_magnitude_px", pa.float64()),
            ("perceptual_hash_64", pa.string()),
            ("uploader_version", pa.string()),
            ("exr_key", pa.string()),
            ("json_key", pa.string()),
            ("frame_bytes", pa.int64()),
            ("content_sha256", pa.string()),
        ]
    )


def write_parquet(rows: List[Dict[str, Any]], *, out_path: str) -> bytes:
    """Serialize rows to a parquet bytestring written to ``out_path``."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    schema = _arrow_schema()
    arrays = []
    for field in schema:
        values = [row.get(field.name) for row in rows]
        if pa.types.is_string(field.type):
            values = ["" if value is None else str(value) for value in values]
        elif pa.types.is_integer(field.type):
            values = [_int_or_default(value) for value in values]
        elif pa.types.is_floating(field.type):
            values = [_float_or_default(value) for value in values]
        arrays.append(pa.array(values, type=field.type))

    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(table, out_path)
    with open(out_path, "rb") as f:
        return f.read()


def upload_index_to_bucket(
    r2_client: Any,
    *,
    parquet_bytes: bytes,
    game_id: Optional[str] = None,
    month: Optional[str] = None,
    date_str: Optional[str] = None,
) -> str:
    """Upload parquet bytes to the C.6 month path or legacy daily path."""
    if game_id is not None or month is not None:
        if not (game_id and month):
            raise ValueError("game_id and month must be supplied together")
        key = f"{_month_prefix(game_id, month)}_index.parquet"
    else:
        if not date_str:
            raise ValueError("date_str is required for legacy upload")
        key = f"_index_{date_str}.parquet"

    r2_client.put_bytes(
        key,
        parquet_bytes,
        content_type="application/x-parquet",
    )
    return key


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_capture_index",
        description="Walk R2 captures and write a parquet frame index.",
    )
    parser.add_argument(
        "--month",
        default=None,
        help="UTC month to index (YYYY-MM). Defaults to current UTC month.",
    )
    parser.add_argument(
        "--game",
        default=None,
        help="Game id to index, e.g. cyberpunk-2077.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Legacy root index date (YYYY-MM-DD). Use --month/--game for C.6.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Override bucket prefix to walk.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Local parquet path. Defaults to a tempfile.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Build the parquet locally but skip the R2 upload step.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    now = datetime.now(timezone.utc)
    month = args.month or now.strftime("%Y-%m")
    date_str = args.date or now.strftime("%Y-%m-%d")

    if args.game is None and args.prefix is None:
        parser.error("--game is required unless --prefix is supplied")

    from server.oss_capture_ingest.r2 import build_default_client  # noqa: PLC0415

    try:
        r2 = build_default_client()
    except RuntimeError as exc:
        print(f"R2 config error: {exc}", file=sys.stderr)
        return 2

    if args.prefix is None:
        log.info("walking bucket prefix=%r", _month_prefix(args.game, month))
        rows = collect_rows(r2, game_id=args.game, month=month)
    else:
        log.info("walking bucket prefix=%r", args.prefix)
        rows = collect_rows(r2, prefix=args.prefix)
    log.info("collected %d rows", len(rows))

    out_path = args.out
    if out_path is None:
        import os  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        fd, out_path = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)

    parquet_bytes = write_parquet(rows, out_path=out_path)
    log.info("wrote parquet to %s (%d bytes)", out_path, len(parquet_bytes))

    if not args.no_upload:
        if args.prefix is None:
            key = upload_index_to_bucket(
                r2, parquet_bytes=parquet_bytes, game_id=args.game, month=month
            )
        else:
            key = upload_index_to_bucket(
                r2, parquet_bytes=parquet_bytes, date_str=date_str
            )
        log.info("uploaded index to %s", key)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
