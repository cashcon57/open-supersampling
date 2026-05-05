"""Shared fixtures for capture-server tests.

Stands up a moto-backed S3 endpoint, builds an :class:`R2Client` pointed
at it, and hands a freshly-built FastAPI app to each test with the in-memory
auth + dedup state reset between tests.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Iterator, Optional, Tuple

import pytest


# Moto needs *some* AWS-shaped creds in env to satisfy boto3's defaults.
_FAKE_ENV = {
    "AWS_ACCESS_KEY_ID": "moto-test-key",
    "AWS_SECRET_ACCESS_KEY": "moto-test-secret",
    "AWS_DEFAULT_REGION": "us-east-1",
    "R2_ACCESS_KEY_ID": "moto-test-key",
    "R2_SECRET_ACCESS_KEY": "moto-test-secret",
    "R2_ENDPOINT": "https://moto-fake-endpoint.invalid",
    "R2_BUCKET": "ors-captures-test",
}


@pytest.fixture(autouse=True)
def _fake_env(monkeypatch):
    for k, v in _FAKE_ENV.items():
        monkeypatch.setenv(k, v)
    yield


@pytest.fixture
def moto_s3():
    """Start a mock_aws context that backs all boto3 S3 calls in-memory."""
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        import boto3

        client = boto3.client("s3", region_name="us-east-1")
        yield client


@pytest.fixture
def r2_client(moto_s3):
    """An :class:`R2Client` wired up to the moto-backed S3 client."""
    from server.oss_capture_ingest.r2 import R2Client, R2Config

    cfg = R2Config(
        access_key_id="moto-test-key",
        secret_access_key="moto-test-secret",
        endpoint_url="https://moto-fake-endpoint.invalid",
        bucket="ors-captures-test",
    )
    client = R2Client(cfg, _client=moto_s3)
    moto_s3.create_bucket(Bucket=cfg.bucket)
    return client


@pytest.fixture
def reset_state():
    """Reset the in-memory token registry + dedup LRU between tests."""
    from server.oss_capture_ingest.auth import reset_registry_for_tests
    from server.oss_capture_ingest.dedup import reset_dedup_for_tests

    reg = reset_registry_for_tests()
    dedup = reset_dedup_for_tests()
    return reg, dedup


@pytest.fixture
def app(r2_client, reset_state):
    """Build the FastAPI app with the moto-backed R2 client injected."""
    from server.oss_capture_ingest.main import create_app

    app = create_app(configure_r2_from_env=False)
    app.state.r2_client = r2_client
    return app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


# ---- helpers ---------------------------------------------------------------


def make_meta(
    *,
    game_id: str = "cyberpunk-2077",
    captured_at_unix: float = 1777940000.0,
    session_uuid: Optional[str] = None,
    frame_uuid: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a valid metadata dict for tests."""
    meta = {
        "schema_version": 1,
        "game_id": game_id,
        "game_version": "2.13",
        "session_uuid": session_uuid or str(uuid.uuid4()),
        "frame_uuid": frame_uuid or str(uuid.uuid4()),
        "captured_at_unix": captured_at_unix,
        "lr_resolution": [1920, 1080],
        "hr_resolution": [3840, 2160],
        "hr_source": "dlss-quality",
        "jitter_offset_uv": [0.234, 0.781],
        "motion_mean_magnitude_px": 12.4,
        "perceptual_hash_64": "0x0123456789abcdef",
        "user_consent_token": "test-consent",
        "uploader_version": "1.0.0",
    }
    meta.update(overrides)
    return meta


def post_ingest(
    client,
    *,
    token: str,
    frame_body: bytes,
    meta: Dict[str, Any],
):
    return client.post(
        "/ingest",
        headers={"Authorization": f"Bearer {token}"},
        files={"frame": (f"{meta.get('frame_uuid', 'frame')}.exr", frame_body, "image/x-exr")},
        data={"meta": json.dumps(meta)},
    )


@pytest.fixture
def make_meta_fn():
    return make_meta


@pytest.fixture
def post_ingest_fn():
    return post_ingest
