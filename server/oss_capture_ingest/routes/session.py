"""``POST /session/start`` — issue a session UUID for a play session.

The uploader calls this once at game start with the install token.
The server validates the token and returns:

- ``session_uuid`` — UUID4 the client uses for all frames in this session
- ``server_time_unix`` — current server time, for clock-skew detection
- ``suggested_capture_rate`` — frames per minute the client should target
  (server-side knob so we can throttle globally during dataset rebuilds)

Default suggested rate is **3 frames/minute** (≈ 1 every 20 s, matches
the design memo's network-respect target).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Header, HTTPException

from .. import API_VERSION
from ..auth import extract_bearer_token, get_registry
from ..schema import _GAME_ID_RE  # type: ignore[attr-defined]


DEFAULT_SUGGESTED_CAPTURE_RATE_PER_MIN = 3.0


def build_router(
    *, suggested_rate_per_min: float = DEFAULT_SUGGESTED_CAPTURE_RATE_PER_MIN
) -> APIRouter:
    router = APIRouter()

    @router.post("/session/start")
    async def session_start(
        body: Dict[str, Any] = Body(...),
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        token = extract_bearer_token(authorization)
        # The body may also carry an ``install_token`` for clients that
        # prefer not to use the Authorization header. Header wins if both.
        body_token = body.get("install_token")
        effective_token = token or (
            body_token if isinstance(body_token, str) else None
        )
        if not effective_token:
            raise HTTPException(status_code=401, detail="missing install token")

        registry = get_registry()
        rec = registry.get(effective_token)
        if rec is None or rec.revoked:
            raise HTTPException(status_code=401, detail="unknown or revoked token")

        game_id = body.get("game_id")
        if not isinstance(game_id, str) or not _GAME_ID_RE.match(game_id):
            raise HTTPException(
                status_code=400,
                detail="game_id missing or malformed (lowercase, fs-safe)",
            )

        game_version = body.get("game_version")
        if game_version is not None and not isinstance(game_version, str):
            raise HTTPException(status_code=400, detail="game_version must be string")

        return {
            "session_uuid": str(uuid.uuid4()),
            "server_time_unix": time.time(),
            "suggested_capture_rate": float(suggested_rate_per_min),
            "api_version": API_VERSION,
        }

    return router
