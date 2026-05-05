"""``POST /ingest`` — accept a single EXR frame + metadata.

Auth, schema validation, dedup, rate-limit, and R2 upload all happen
here. On success we write **two** keys to R2:

- ``<game_id>/<YYYY-MM>/<session_uuid>/<frame_uuid>.exr``  (the EXR body)
- ``<game_id>/<YYYY-MM>/<session_uuid>/<frame_uuid>.json`` (the metadata)

The EXR object also carries a small ``Metadata`` map (S3 user metadata)
with the content SHA256 + game_id + session_uuid for fast in-bucket
forensic lookup without having to download the JSON sidecar.

Status codes match the design memo §"Server-side ingestion":

* 200 — accepted
* 400 — malformed metadata or multipart parts
* 401 — bad/missing token
* 409 — duplicate (content hash already seen)
* 413 — frame > 16 MB
* 429 — rate limit exceeded
* 500 — server error (uploader will retry with backoff)

``fastapi`` is imported lazily from :mod:`server.oss_capture_ingest.main`,
so this module does its own lazy import for the ``APIRouter`` symbol it
uses at decorator time.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

# These modules are only imported transitively from create_app(), which
# is itself a lazy entrypoint; importing fastapi here is safe — the
# --help path in main.py never reaches this module.
from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)

from .. import MAX_FRAME_BYTES
from ..auth import extract_bearer_token, get_registry
from ..dedup import get_dedup
from ..r2 import R2Client, build_default_client, frame_key
from ..schema import SchemaError, validate_metadata

log = logging.getLogger(__name__)


def _r2_client(app_state: Any) -> R2Client:
    """Resolve the R2Client from app.state, falling back to env."""
    client = getattr(app_state, "r2_client", None)
    if client is None:
        client = build_default_client()
        app_state.r2_client = client
    return client


async def _read_upload_body(upload, max_bytes: int) -> bytes:
    """Read a starlette UploadFile body, capped at ``max_bytes``.

    Returns the bytes if within cap; raises ``ValueError("too_large")``
    if the cap is exceeded mid-stream.
    """
    chunks = []
    total = 0
    chunk_size = 1 << 20  # 1 MiB
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("too_large")
        chunks.append(chunk)
    return b"".join(chunks)


def build_router(*, max_frame_bytes: int = MAX_FRAME_BYTES) -> APIRouter:
    """Construct the FastAPI ``APIRouter`` for the ingest endpoint."""
    router = APIRouter()

    @router.post("/ingest")
    async def ingest(
        request: Request,
        frame: UploadFile = File(...),
        meta: str = Form(...),
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        # ---- auth --------------------------------------------------------
        token = extract_bearer_token(authorization)
        if token is None:
            raise HTTPException(status_code=401, detail="missing bearer token")
        registry = get_registry()
        rec = registry.get(token)
        if rec is None or rec.revoked:
            raise HTTPException(status_code=401, detail="unknown or revoked token")

        # ---- rate limit --------------------------------------------------
        if not registry.check_rate(token):
            raise HTTPException(
                status_code=429,
                detail=(
                    f"rate limit exceeded "
                    f"({registry.rate_limit} frames/{registry.window_seconds}s)"
                ),
            )

        # ---- metadata ----------------------------------------------------
        try:
            meta_obj = json.loads(meta)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"meta is not valid JSON: {exc}")
        try:
            normalized = validate_metadata(meta_obj)
        except SchemaError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # ---- frame body --------------------------------------------------
        try:
            frame_bytes = await _read_upload_body(frame, max_frame_bytes)
        except ValueError as exc:
            if str(exc) == "too_large":
                raise HTTPException(
                    status_code=413,
                    detail=f"frame body exceeds {max_frame_bytes} bytes",
                )
            raise
        if not frame_bytes:
            raise HTTPException(status_code=400, detail="empty frame body")

        content_hash = hashlib.sha256(frame_bytes).hexdigest()

        # ---- dedup -------------------------------------------------------
        dedup = get_dedup()
        if dedup.contains(content_hash):
            raise HTTPException(
                status_code=409,
                detail="duplicate frame (content hash already seen)",
            )

        # ---- write to R2 -------------------------------------------------
        try:
            r2 = _r2_client(request.app.state)
        except RuntimeError as exc:
            # Missing creds is a server-config bug, not a client issue.
            log.error("R2 config missing: %s", exc)
            raise HTTPException(status_code=500, detail="R2 not configured")

        exr_key = frame_key(
            normalized["game_id"],
            normalized["captured_at_unix"],
            normalized["session_uuid"],
            normalized["frame_uuid"],
            suffix=".exr",
        )
        json_key = frame_key(
            normalized["game_id"],
            normalized["captured_at_unix"],
            normalized["session_uuid"],
            normalized["frame_uuid"],
            suffix=".json",
        )

        meta_with_hash = dict(normalized)
        meta_with_hash["content_sha256"] = content_hash
        meta_with_hash["frame_bytes"] = len(frame_bytes)
        meta_bytes = json.dumps(meta_with_hash, sort_keys=True).encode("utf-8")

        try:
            try:
                r2.put_bytes(
                    exr_key,
                    frame_bytes,
                    content_type="image/x-exr",
                    metadata={
                        "content-sha256": content_hash,
                        "game-id": normalized["game_id"],
                        "session-uuid": normalized["session_uuid"],
                    },
                )
            except Exception:
                # First-time path: bucket may not exist yet on a fresh deploy.
                try:
                    r2.ensure_bucket()
                except Exception as exc:  # pragma: no cover — backend dep
                    log.exception("ensure_bucket failed: %s", exc)
                    raise
                r2.put_bytes(
                    exr_key,
                    frame_bytes,
                    content_type="image/x-exr",
                    metadata={
                        "content-sha256": content_hash,
                        "game-id": normalized["game_id"],
                        "session-uuid": normalized["session_uuid"],
                    },
                )

            r2.put_bytes(
                json_key,
                meta_bytes,
                content_type="application/json",
            )
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("R2 put failed: %s", exc)
            raise HTTPException(status_code=500, detail="R2 write failed")

        # ---- accounting --------------------------------------------------
        dedup.add(content_hash)
        registry.record_upload(token, len(frame_bytes))

        return {
            "status": "ok",
            "frame_uuid": normalized["frame_uuid"],
            "session_uuid": normalized["session_uuid"],
            "exr_key": exr_key,
            "json_key": json_key,
            "content_sha256": content_hash,
            "frame_bytes": len(frame_bytes),
        }

    return router
