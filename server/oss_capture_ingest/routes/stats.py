"""``GET /stats`` — per-token + global dataset statistics.

Two modes:

- ``GET /stats?token=<install-token>`` — returns counters for that one
  token (frames uploaded, total bytes, optional contributor rank).
- ``GET /stats`` (no token) — returns the global aggregate only.

Counters are sourced from the in-memory :class:`TokenRegistry`. A v2
follow-up will reconcile with the parquet index for crash-survivable
stats.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from .. import API_VERSION
from ..auth import get_registry


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/stats")
    async def stats(token: Optional[str] = Query(default=None)) -> Dict[str, Any]:
        registry = get_registry()

        # global rollup
        all_tokens = registry.all_tokens()
        global_frames = sum(r.total_frames for r in all_tokens.values())
        global_bytes = sum(r.total_bytes for r in all_tokens.values())

        result: Dict[str, Any] = {
            "api_version": API_VERSION,
            "global": {
                "total_frames": global_frames,
                "total_bytes": global_bytes,
                "contributor_count": sum(
                    1 for r in all_tokens.values() if r.total_frames > 0
                ),
            },
        }

        if token is not None:
            rec = registry.get(token)
            if rec is None or rec.revoked:
                raise HTTPException(
                    status_code=401, detail="unknown or revoked token"
                )
            # Contributor rank: how many tokens have strictly more frames.
            rank = 1 + sum(
                1
                for r in all_tokens.values()
                if r.total_frames > rec.total_frames
            )
            result["token"] = {
                "frames_uploaded": rec.total_frames,
                "total_bytes": rec.total_bytes,
                "contributor_rank": rank,
            }

        return result

    return router
