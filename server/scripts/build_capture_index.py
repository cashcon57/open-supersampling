"""Daily capture-index builder.

Walks the R2 capture bucket, gathers every ``<frame_uuid>.json`` companion
file, and writes a parquet index keyed by ``(game_id, frame_uuid)`` to:

    <bucket>/_index_<YYYY-MM-DD>.parquet

The training data loader (see Sprint S6+ pipelines) reads this parquet
to do motion-bucket-balanced sampling without touching individual JSON
sidecars.

Designed to be invoked from cron / GitHub Actions / a Cloudflare Worker
that schedules a small VM. Idempotent: running twice on the same day
overwrites the index for that day.

Usage:

    python -m server.scripts.build_capture_index --date 2026-05-04
    python -m server.scripts.build_capture_index   # defaults to today UTC

Heavy deps (``boto3``, ``pyarrow``) are imported lazily inside the
``main`` function so ``--help`` stays cheap.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


log = logging.getLogger(__name__)


# Columns we materialize into the parquet index. Subset of the metadata
# JSON schema — fields used by training-data sampling. Adding a column
# here is a non-breaking change as long as readers tolerate missing cols.
INDEX_COLUMNS: Tuple[str, ...] = (
    "game_id",
    "game_version",
    "session_uuid",
    "frame_uuid",
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


def _row_from_meta(meta: Dict[str, Any], json_key: str) -> Dict[str, Any]:
    """Project a metadata dict to one parquet row. Tolerates older schemas."""
    lr = meta.get("lr_resolution") or [0, 0]
    hr = meta.get("hr_resolution") or [0, 0]
    exr_key = json_key[:-5] + ".exr" if json_key.endswith(".json") else ""
    return {
        "game_id": str(meta.get("game_id", "")),
        "game_version": str(meta.get("game_version", "")),
        "session_uuid": str(meta.get("session_uuid", "")),
        "frame_uuid": str(meta.get("frame_uuid", "")),
        "captured_at_unix": float(meta.get("captured_at_unix", 0.0)),
        "lr_width": int(lr[0]) if len(lr) >= 1 else 0,
        "lr_height": int(lr[1]) if len(lr) >= 2 else 0,
        "hr_width": int(hr[0]) if len(hr) >= 1 else 0,
        "hr_height": int(hr[1]) if len(hr) >= 2 else 0,
        "hr_source": str(meta.get("hr_source", "")),
        "motion_mean_magnitude_px": float(
            meta.get("motion_mean_magnitude_px", 0.0)
        ),
        "perceptual_hash_64": str(meta.get("perceptual_hash_64", "")),
        "uploader_version": str(meta.get("uploader_version", "")),
        "exr_key": exr_key,
        "json_key": json_key,
        "frame_bytes": int(meta.get("frame_bytes", 0)),
        "content_sha256": str(meta.get("content_sha256", "")),
    }


def collect_rows(r2_client: Any, *, prefix: str = "") -> List[Dict[str, Any]]:
    """Walk the bucket and return one row per ``.json`` companion."""
    rows: List[Dict[str, Any]] = []
    for key, _size in r2_client.iter_objects(prefix=prefix):
        if not key.endswith(".json"):
            continue
        # Skip any bucket-root index files we wrote previously.
        if key.startswith("_index_") or "/_index_" in key:
            continue
        # Skip durable-dedup markers (post-MED-volatile-dedup fix). These
        # never have a .json suffix today, but the prefix-skip is cheap
        # insurance against future marker-format changes.
        if key.startswith("_dedup/"):
            continue
        try:
            body = r2_client.get_bytes(key)
            meta = json.loads(body.decode("utf-8"))
        except Exception as exc:
            log.warning("failed to read/parse %s: %s", key, exc)
            continue
        try:
            rows.append(_row_from_meta(meta, key))
        except Exception as exc:
            log.warning("failed to project %s: %s", key, exc)
    return rows


def write_parquet(rows: List[Dict[str, Any]], *, out_path: str) -> bytes:
    """Serialize rows to a parquet bytestring written to ``out_path``."""
    import pyarrow as pa  # noqa: PLC0415 — intentional lazy import
    import pyarrow.parquet as pq  # noqa: PLC0415

    # Ensure all columns exist on every row (fill with type-appropriate zero).
    type_zero = {
        "captured_at_unix": 0.0,
        "lr_width": 0,
        "lr_height": 0,
        "hr_width": 0,
        "hr_height": 0,
        "motion_mean_magnitude_px": 0.0,
        "frame_bytes": 0,
    }
    columns: Dict[str, List[Any]] = {c: [] for c in INDEX_COLUMNS}
    for row in rows:
        for c in INDEX_COLUMNS:
            if c in row:
                columns[c].append(row[c])
            else:
                columns[c].append(type_zero.get(c, ""))

    table = pa.table(columns)
    pq.write_table(table, out_path)
    with open(out_path, "rb") as f:
        return f.read()


def upload_index_to_bucket(
    r2_client: Any,
    *,
    parquet_bytes: bytes,
    date_str: str,
) -> str:
    """Upload the parquet bytes to ``_index_<date>.parquet`` in the bucket."""
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
        description=(
            "Walk the R2 capture bucket and write a daily parquet index."
        ),
    )
    parser.add_argument(
        "--date",
        default=None,
        help="UTC date for the index filename (YYYY-MM-DD). "
        "Defaults to today UTC.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Bucket prefix to walk. Empty = whole bucket.",
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

    if args.date is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        date_str = args.date

    # Lazy imports so --help is dep-free.
    from server.oss_capture_ingest.r2 import (  # noqa: PLC0415
        build_default_client,
    )

    try:
        r2 = build_default_client()
    except RuntimeError as exc:
        print(f"R2 config error: {exc}", file=sys.stderr)
        return 2

    log.info("walking bucket prefix=%r", args.prefix)
    rows = collect_rows(r2, prefix=args.prefix)
    log.info("collected %d rows", len(rows))

    out_path = args.out
    if out_path is None:
        import tempfile  # noqa: PLC0415

        fd, out_path = tempfile.mkstemp(suffix=".parquet")
        import os  # noqa: PLC0415

        os.close(fd)

    parquet_bytes = write_parquet(rows, out_path=out_path)
    log.info("wrote parquet to %s (%d bytes)", out_path, len(parquet_bytes))

    if not args.no_upload:
        key = upload_index_to_bucket(
            r2, parquet_bytes=parquet_bytes, date_str=date_str
        )
        log.info("uploaded index to %s", key)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
