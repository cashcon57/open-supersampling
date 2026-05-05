"""Tests for the daily capture-index parquet builder.

Populates a moto-backed bucket with synthetic EXR + JSON pairs, runs the
collect/write pipeline, and reads the parquet back to check rows + cols.
"""

from __future__ import annotations

import json
import os

import pytest


def _seed(r2_client, *, n_frames: int = 5) -> list:
    """Drop ``n_frames`` synthetic .exr/.json pairs into the bucket.

    Returns the list of expected metadata dicts (one per frame).
    """
    out = []
    for i in range(n_frames):
        meta = {
            "schema_version": 1,
            "game_id": "cyberpunk-2077" if i % 2 == 0 else "bg3",
            "game_version": "1.0",
            "session_uuid": f"00000000-0000-0000-0000-{i:012d}",
            "frame_uuid": f"11111111-1111-1111-1111-{i:012d}",
            "captured_at_unix": 1714867200.0 + i,
            "lr_resolution": [1920, 1080],
            "hr_resolution": [3840, 2160],
            "hr_source": "dlss-quality",
            "jitter_offset_uv": [0.1, 0.2],
            "motion_mean_magnitude_px": float(i),
            "perceptual_hash_64": f"0x{i:016x}",
            "user_consent_token": "ct",
            "uploader_version": "1.0.0",
            "content_sha256": f"deadbeef{i:056d}",
            "frame_bytes": 100 + i,
        }
        body = f"FRAME-BODY-{i}".encode() * 4
        from server.oss_capture_ingest.r2 import frame_key

        exr_key = frame_key(
            meta["game_id"],
            meta["captured_at_unix"],
            meta["session_uuid"],
            meta["frame_uuid"],
            suffix=".exr",
        )
        json_key = exr_key[:-4] + ".json"
        r2_client.put_bytes(exr_key, body, content_type="image/x-exr")
        r2_client.put_bytes(
            json_key,
            json.dumps(meta).encode("utf-8"),
            content_type="application/json",
        )
        out.append(meta)
    return out


def test_collect_rows_picks_up_all_companion_jsons(r2_client):
    metas = _seed(r2_client, n_frames=4)
    from server.scripts.build_capture_index import collect_rows

    rows = collect_rows(r2_client)
    assert len(rows) == 4
    fids = sorted(r["frame_uuid"] for r in rows)
    assert fids == sorted(m["frame_uuid"] for m in metas)


def test_collect_rows_skips_non_json(r2_client):
    _seed(r2_client, n_frames=2)
    # Drop a stray non-JSON file at bucket root that should be ignored.
    r2_client.put_bytes(
        "_index_2026-05-04.parquet",
        b"PARQ-MOCK",
        content_type="application/x-parquet",
    )
    from server.scripts.build_capture_index import collect_rows

    rows = collect_rows(r2_client)
    # Should still only have 2, not 3.
    assert len(rows) == 2


def test_write_parquet_roundtrip(tmp_path, r2_client):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    metas = _seed(r2_client, n_frames=3)

    from server.scripts.build_capture_index import (
        INDEX_COLUMNS,
        collect_rows,
        write_parquet,
    )

    rows = collect_rows(r2_client)
    out = str(tmp_path / "test_index.parquet")
    parquet_bytes = write_parquet(rows, out_path=out)
    assert os.path.exists(out)
    assert len(parquet_bytes) > 0

    table = pq.read_table(out)
    cols = set(table.column_names)
    for c in INDEX_COLUMNS:
        assert c in cols, f"missing column {c}"
    assert table.num_rows == 3

    # Spot-check a couple of values.
    fids = set(table.column("frame_uuid").to_pylist())
    assert fids == set(m["frame_uuid"] for m in metas)
    motion = sorted(table.column("motion_mean_magnitude_px").to_pylist())
    assert motion == [0.0, 1.0, 2.0]


def test_upload_index_to_bucket(r2_client):
    from server.scripts.build_capture_index import upload_index_to_bucket

    key = upload_index_to_bucket(
        r2_client, parquet_bytes=b"FAKE-PARQUET", date_str="2026-05-04"
    )
    assert key == "_index_2026-05-04.parquet"
    assert r2_client.get_bytes(key) == b"FAKE-PARQUET"
