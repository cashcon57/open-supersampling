"""Unit tests for the FastAPI ingest server.

Covers: auth, schema validation, dedup, rate limit, multipart parsing,
and R2 write side-effect (with R2 mocked via moto).
"""

from __future__ import annotations

import json
import uuid

import pytest


def _register(reset_state, *, token: str = "test-token-001"):
    registry, _ = reset_state
    registry.register_token(token, label="test")
    return token


# ---- auth ------------------------------------------------------------------


def test_ingest_missing_auth_returns_401(client, make_meta_fn, reset_state):
    _register(reset_state)
    resp = client.post(
        "/ingest",
        files={"frame": ("f.exr", b"FRAME", "image/x-exr")},
        data={"meta": json.dumps(make_meta_fn())},
    )
    assert resp.status_code == 401


def test_ingest_unknown_token_returns_401(client, make_meta_fn, reset_state):
    _register(reset_state)
    resp = client.post(
        "/ingest",
        headers={"Authorization": "Bearer not-a-real-token"},
        files={"frame": ("f.exr", b"FRAME", "image/x-exr")},
        data={"meta": json.dumps(make_meta_fn())},
    )
    assert resp.status_code == 401


def test_ingest_revoked_token_returns_401(client, make_meta_fn, reset_state):
    registry, _ = reset_state
    registry.register_token("token-x")
    registry.revoke_token("token-x")
    resp = client.post(
        "/ingest",
        headers={"Authorization": "Bearer token-x"},
        files={"frame": ("f.exr", b"FRAME", "image/x-exr")},
        data={"meta": json.dumps(make_meta_fn())},
    )
    assert resp.status_code == 401


# ---- happy path ------------------------------------------------------------


def test_ingest_happy_path_writes_to_r2(
    client, r2_client, make_meta_fn, post_ingest_fn, reset_state
):
    token = _register(reset_state)
    meta = make_meta_fn(
        game_id="cyberpunk-2077",
        session_uuid="11111111-1111-1111-1111-111111111111",
        frame_uuid="22222222-2222-2222-2222-222222222222",
        captured_at_unix=1714867200.0,  # 2024-05-05 UTC
    )
    body = b"\x00\x01\x02EXR-MOCK-BODY" * 100

    resp = post_ingest_fn(client, token=token, frame_body=body, meta=meta)
    assert resp.status_code == 200, resp.text
    j = resp.json()
    assert j["status"] == "ok"
    assert j["frame_uuid"] == meta["frame_uuid"]
    assert j["frame_bytes"] == len(body)

    # The two keys exist in the moto-backed bucket. Layout includes the
    # capture_mode segment (post-C23) — meta with no capture_mode falls
    # back to "lite" on the server side.
    expected_exr = "cyberpunk-2077/2024-05/lite/11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222.exr"
    expected_json = expected_exr[:-4] + ".json"
    assert j["exr_key"] == expected_exr
    assert j["json_key"] == expected_json

    # Server stored the actual bytes
    fetched = r2_client.get_bytes(expected_exr)
    assert fetched == body
    fetched_meta = json.loads(r2_client.get_bytes(expected_json))
    assert fetched_meta["frame_uuid"] == meta["frame_uuid"]
    assert "content_sha256" in fetched_meta


# ---- schema validation -----------------------------------------------------


def test_ingest_meta_not_json_returns_400(client, reset_state):
    token = _register(reset_state)
    resp = client.post(
        "/ingest",
        headers={"Authorization": f"Bearer {token}"},
        files={"frame": ("f.exr", b"X", "image/x-exr")},
        data={"meta": "{not json"},
    )
    assert resp.status_code == 400


def test_ingest_meta_missing_field_returns_400(
    client, make_meta_fn, post_ingest_fn, reset_state
):
    token = _register(reset_state)
    meta = make_meta_fn()
    del meta["hr_resolution"]
    resp = post_ingest_fn(client, token=token, frame_body=b"X" * 16, meta=meta)
    assert resp.status_code == 400
    assert "hr_resolution" in resp.json()["detail"]


def test_ingest_meta_bad_game_id_returns_400(
    client, make_meta_fn, post_ingest_fn, reset_state
):
    token = _register(reset_state)
    meta = make_meta_fn(game_id="UPPERCASE/slash")
    resp = post_ingest_fn(client, token=token, frame_body=b"X" * 16, meta=meta)
    assert resp.status_code == 400


# ---- frame size cap --------------------------------------------------------


def test_ingest_oversize_frame_returns_413(
    client, make_meta_fn, post_ingest_fn, reset_state, monkeypatch
):
    # Override the cap module-wide for this test, then rebuild the app
    # through a smaller cap. Easier path: hammer the existing 16 MB cap.
    from server.oss_capture_ingest import MAX_FRAME_BYTES

    token = _register(reset_state)
    big = b"X" * (MAX_FRAME_BYTES + 1)
    resp = post_ingest_fn(client, token=token, frame_body=big, meta=make_meta_fn())
    assert resp.status_code == 413


# ---- dedup -----------------------------------------------------------------


def test_ingest_duplicate_frame_returns_409(
    client, make_meta_fn, post_ingest_fn, reset_state
):
    token = _register(reset_state)
    body = b"DEDUP-BODY-XYZ" * 50
    r1 = post_ingest_fn(client, token=token, frame_body=body, meta=make_meta_fn())
    assert r1.status_code == 200, r1.text
    # Second upload of the *same body* (different frame_uuid) is dedup'd.
    r2 = post_ingest_fn(client, token=token, frame_body=body, meta=make_meta_fn())
    assert r2.status_code == 409


def test_ingest_dedup_survives_lru_reset_via_durable_backend(
    client, r2_client, make_meta_fn, post_ingest_fn, reset_state
):
    """Closes Codex's MED 'volatile dedup' finding: even if the in-memory
    LRU is wiped (simulating a process restart), the next upload of the
    same content still returns 409 because the dedup marker persists in
    R2 and the LRU falls back to it on miss."""
    from server.oss_capture_ingest.dedup import get_dedup, reset_dedup_for_tests

    # Wire moto-backed R2 as the durable backend (the test app fixture
    # doesn't auto-wire it because configure_r2_from_env=False).
    get_dedup().set_durable_backend(r2_client)

    token = _register(reset_state)
    body = b"DEDUP-PERSISTS" * 40

    r1 = post_ingest_fn(client, token=token, frame_body=body, meta=make_meta_fn())
    assert r1.status_code == 200, r1.text

    # Simulate process restart — fresh LRU, no in-memory hash entries.
    fresh = reset_dedup_for_tests()
    fresh.set_durable_backend(r2_client)
    assert len(fresh) == 0

    # The marker in R2 must still cause a 409.
    r2 = post_ingest_fn(client, token=token, frame_body=body, meta=make_meta_fn())
    assert r2.status_code == 409
    # And the LRU should now be hot for that hash (hydrated on the
    # contains-call fallback).
    assert len(fresh) == 1


# ---- rate limit ------------------------------------------------------------


def test_ingest_rate_limit_returns_429(
    client, make_meta_fn, post_ingest_fn, reset_state
):
    registry, _ = reset_state
    registry.rate_limit = 3  # tiny limit for the test
    token = _register(reset_state, token="rl-token")
    # 3 successful uploads, distinct bodies
    for i in range(3):
        body = f"BODY-{i}".encode() * 32
        r = post_ingest_fn(client, token=token, frame_body=body, meta=make_meta_fn())
        assert r.status_code == 200, r.text
    # 4th is throttled
    r4 = post_ingest_fn(
        client,
        token=token,
        frame_body=b"BODY-4" * 32,
        meta=make_meta_fn(),
    )
    assert r4.status_code == 429
    # Retry-After is set so the client can honor RFC 7231 backoff (closes
    # the day-one server-side prep for the uploader's MED 429 fix).
    assert "retry-after" in {h.lower() for h in r4.headers}
    assert int(r4.headers["retry-after"]) >= 1


# ---- multipart parsing -----------------------------------------------------


def test_ingest_missing_frame_part_returns_422_or_400(
    client, make_meta_fn, reset_state
):
    token = _register(reset_state)
    resp = client.post(
        "/ingest",
        headers={"Authorization": f"Bearer {token}"},
        data={"meta": json.dumps(make_meta_fn())},
    )
    # FastAPI returns 422 for missing required form/file params.
    assert resp.status_code in (400, 422)


def test_ingest_empty_frame_returns_400(
    client, make_meta_fn, post_ingest_fn, reset_state
):
    token = _register(reset_state)
    resp = post_ingest_fn(client, token=token, frame_body=b"", meta=make_meta_fn())
    assert resp.status_code == 400


# ---- session/start ---------------------------------------------------------


def test_session_start_returns_uuid_and_rate(client, reset_state):
    token = _register(reset_state, token="session-token")
    resp = client.post(
        "/session/start",
        headers={"Authorization": f"Bearer {token}"},
        json={"game_id": "cyberpunk-2077", "game_version": "2.13"},
    )
    assert resp.status_code == 200
    j = resp.json()
    assert "session_uuid" in j
    assert j["server_time_unix"] > 0
    assert j["suggested_capture_rate"] > 0


def test_session_start_rejects_bad_game_id(client, reset_state):
    token = _register(reset_state)
    resp = client.post(
        "/session/start",
        headers={"Authorization": f"Bearer {token}"},
        json={"game_id": "BAD ID"},
    )
    assert resp.status_code == 400


def test_session_start_rejects_missing_token(client, reset_state):
    _register(reset_state)
    resp = client.post(
        "/session/start", json={"game_id": "cyberpunk-2077"}
    )
    assert resp.status_code == 401


# ---- /stats ---------------------------------------------------------------


def test_stats_global(client, reset_state):
    _register(reset_state)
    resp = client.get("/stats")
    assert resp.status_code == 200
    j = resp.json()
    assert "global" in j
    assert j["global"]["total_frames"] == 0


def test_stats_per_token_after_uploads(
    client, make_meta_fn, post_ingest_fn, reset_state
):
    token = _register(reset_state, token="stats-token")
    for i in range(2):
        r = post_ingest_fn(
            client,
            token=token,
            frame_body=f"BODY-{i}".encode() * 32,
            meta=make_meta_fn(),
        )
        assert r.status_code == 200, r.text
    resp = client.get(f"/stats?token={token}")
    assert resp.status_code == 200
    j = resp.json()
    assert j["token"]["frames_uploaded"] == 2
    assert j["token"]["total_bytes"] > 0
    assert j["token"]["contributor_rank"] == 1


def test_stats_per_mode_counts_global_and_per_token(
    client, make_meta_fn, post_ingest_fn, reset_state
):
    """Closes the 'no per-mode contribution stratification' gap on /stats.
    Dataset card consumes these counts to report per-mode contribution."""
    token = _register(reset_state, token="mode-stats-token")
    # Two trickle, one regular — distinct bodies to avoid dedup.
    for i, mode in enumerate(("trickle", "trickle", "regular")):
        body = f"MODE-BODY-{i}".encode() * 32
        r = post_ingest_fn(
            client,
            token=token,
            frame_body=body,
            meta=make_meta_fn(capture_mode=mode),
        )
        assert r.status_code == 200, r.text

    j = client.get(f"/stats?token={token}").json()
    assert j["global"]["frames_by_mode"] == {"trickle": 2, "regular": 1}
    assert j["token"]["frames_by_mode"] == {"trickle": 2, "regular": 1}
    # Bytes are mode-stratified too.
    assert j["token"]["bytes_by_mode"]["trickle"] > 0
    assert j["token"]["bytes_by_mode"]["regular"] > 0
    # Bytes by mode roll up to total.
    by_mode_total = sum(j["token"]["bytes_by_mode"].values())
    assert by_mode_total == j["token"]["total_bytes"]


# ---- healthz ---------------------------------------------------------------


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
