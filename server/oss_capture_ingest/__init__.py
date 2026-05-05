"""OSS Capture Tool — server-side ingestion service.

FastAPI application that accepts EXR frame uploads from the per-game
capture installer (DLL + uploader daemon, see Codex's tandem half), runs
schema/dedup/auth/rate-limit validation, and writes accepted frames to
the R2 ``ors-captures`` bucket.

Heavy dependencies (``fastapi``, ``boto3``, ``pyarrow``) are imported
lazily at call sites so a vanilla Python interpreter without these
installed can still ``import server.oss_capture_ingest`` for help and
tooling purposes.
"""

__all__ = [
    "API_VERSION",
    "MAX_FRAME_BYTES",
    "DEFAULT_R2_BUCKET",
]

API_VERSION = "1.0.0"
"""Semver string returned in /stats and /session/start responses."""

MAX_FRAME_BYTES = 16 * 1024 * 1024  # 16 MB hard cap per spec
"""Hard cap on a single uploaded EXR frame body (matches design memo)."""

DEFAULT_R2_BUCKET = "ors-captures"
"""Default R2 bucket name. Overridable via the ``R2_BUCKET`` env var."""
