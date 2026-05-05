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

        # Estimated time until the rate-limit window frees a slot. A
        # rough but useful Retry-After hint for the client — clients that
        # honor RFC 7231 ``Retry-After`` will back off precisely instead
        # of hammering. Capped at the full window so a permanently
        # exhausted budget still produces a finite hint.
        retry_after_attempt = max(
            1, registry.window_seconds // max(1, registry.attempt_limit)
        )
        retry_after_upload = max(
            1, registry.window_seconds // max(1, registry.rate_limit)
        )
        retry_after_per_game = max(
            1, registry.window_seconds // max(1, registry.per_game_attempt_limit)
        )

        # ---- attempt rate limit (cheap gate, BEFORE multipart parsing) ---
        # Closes Codex's MED finding: every authenticated request (success
        # or rejection) charges against this budget so a misbehaving client
        # can't hammer the parse/validate paths for free.
        if not registry.check_attempt(token):
            raise HTTPException(
                status_code=429,
                detail=(
                    f"attempt rate limit exceeded "
                    f"({registry.attempt_limit} attempts/"
                    f"{registry.window_seconds}s)"
                ),
                headers={"Retry-After": str(retry_after_attempt)},
            )
        # ---- successful-upload rate limit (also cheap; hard cap) ---------
        if not registry.check_rate(token):
            raise HTTPException(
                status_code=429,
                detail=(
                    f"upload rate limit exceeded "
                    f"({registry.rate_limit} frames/{registry.window_seconds}s)"
                ),
                headers={"Retry-After": str(retry_after_upload)},
            )

        # Charge the attempt before any further work — even a 400/409/413
        # below should consume budget so we shed load fairly.
        registry.record_attempt(token)

        # ---- metadata ----------------------------------------------------
        try:
            meta_obj = json.loads(meta)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"meta is not valid JSON: {exc}")
        try:
            normalized = validate_metadata(meta_obj)
        except SchemaError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # ---- per-game attempt rate limit --------------------------------
        # game_id is now validated; check the per-(token, game) budget.
        # Closes Codex's 'no per-game limiter' gap.
        game_id_val = normalized["game_id"]
        if not registry.check_per_game_attempt(token, game_id_val):
            raise HTTPException(
                status_code=429,
                detail=(
                    f"per-game attempt rate limit exceeded for game "
                    f"'{game_id_val}' "
                    f"({registry.per_game_attempt_limit} attempts/"
                    f"{registry.window_seconds}s)"
                ),
                headers={"Retry-After": str(retry_after_per_game)},
            )
        registry.record_per_game_attempt(token, game_id_val)

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

        # Mode-stratified path layout — pre-C23 uploads with no
        # capture_mode in metadata default to "lite" (server's documented
        # back-compat assumption from schema.py).
        capture_mode = normalized.get("capture_mode") or "lite"
        exr_key = frame_key(
            normalized["game_id"],
            normalized["captured_at_unix"],
            normalized["session_uuid"],
            normalized["frame_uuid"],
            suffix=".exr",
            capture_mode=capture_mode,
        )
        json_key = frame_key(
            normalized["game_id"],
            normalized["captured_at_unix"],
            normalized["session_uuid"],
            normalized["frame_uuid"],
            suffix=".json",
            capture_mode=capture_mode,
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
        # Durable dedup marker — survives process restarts. Best-effort:
        # the LRU is authoritative within the process, the marker only
        # becomes load-bearing post-restart.
        dedup.add_durable(content_hash)
        registry.record_upload(
            token, len(frame_bytes), capture_mode=capture_mode
        )

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
