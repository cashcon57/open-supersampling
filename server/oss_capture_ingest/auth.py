"""Bearer-token auth + per-token rate limiting.

The token registry is in-memory for v1. The interface is deliberately
SQLite/Postgres-shaped (``register_token``, ``revoke_token``,
``check_token``) so that swapping the backing store is a one-file change.

Tokens are minted by ``scripts/build_capture_installer.py`` and baked into
each per-game installer. They are opaque from the client's perspective:
not user-identifying, just rate-limitable.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


@dataclass
class TokenRecord:
    """Server-side state for one install token."""

    token: str
    label: str = ""
    created_at_unix: float = field(default_factory=lambda: time.time())
    revoked: bool = False
    # rolling window of frame-upload timestamps for rate-limit accounting
    upload_times: Deque[float] = field(default_factory=deque)
    # cumulative counters (do not reset; used by /stats)
    total_frames: int = 0
    total_bytes: int = 0


class TokenRegistry:
    """In-memory bearer-token store with sliding-window rate limiting.

    The default rate limit is **1000 frames/hour** per token. This matches
    the design memo's network-respect target (<500 MB/hour at ~3 MB/frame
    ≈ 170 frames/hour, so 1000/hour leaves headroom for retries).

    Thread-safe via a single ``RLock``.
    """

    def __init__(
        self,
        rate_limit_frames_per_hour: int = 1000,
        window_seconds: int = 3600,
    ) -> None:
        self.rate_limit = int(rate_limit_frames_per_hour)
        self.window_seconds = int(window_seconds)
        self._tokens: Dict[str, TokenRecord] = {}
        self._lock = threading.RLock()

    # ---- registry management ------------------------------------------------

    def register_token(self, token: str, label: str = "") -> TokenRecord:
        """Register a new token (idempotent)."""
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None:
                rec = TokenRecord(token=token, label=label)
                self._tokens[token] = rec
            elif label and not rec.label:
                rec.label = label
            return rec

    def revoke_token(self, token: str) -> bool:
        """Mark a token as revoked. Returns True if the token existed."""
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None:
                return False
            rec.revoked = True
            return True

    def get(self, token: str) -> Optional[TokenRecord]:
        with self._lock:
            return self._tokens.get(token)

    def all_tokens(self) -> Dict[str, TokenRecord]:
        with self._lock:
            return dict(self._tokens)

    # ---- rate limiting + accounting ----------------------------------------

    def _prune_window(self, rec: TokenRecord, now: float) -> None:
        cutoff = now - self.window_seconds
        while rec.upload_times and rec.upload_times[0] < cutoff:
            rec.upload_times.popleft()

    def check_rate(self, token: str, now: Optional[float] = None) -> bool:
        """Return True if a frame upload is allowed under the rate limit.

        Does not record the upload — call :meth:`record_upload` after a
        successful write.
        """
        if now is None:
            now = time.time()
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None or rec.revoked:
                return False
            self._prune_window(rec, now)
            return len(rec.upload_times) < self.rate_limit

    def record_upload(
        self,
        token: str,
        frame_bytes: int,
        now: Optional[float] = None,
    ) -> None:
        """Record a successful upload for rate-limit + /stats accounting."""
        if now is None:
            now = time.time()
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None:
                return
            rec.upload_times.append(now)
            rec.total_frames += 1
            rec.total_bytes += int(frame_bytes)
            self._prune_window(rec, now)


# ---- header parsing ---------------------------------------------------------


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """Extract ``Bearer <token>`` from an ``Authorization`` header value.

    Returns ``None`` for missing or malformed headers.
    """
    if not authorization_header:
        return None
    parts = authorization_header.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


# ---- module-level singleton -------------------------------------------------

_REGISTRY: Optional[TokenRegistry] = None


def get_registry() -> TokenRegistry:
    """Return the process-wide :class:`TokenRegistry` singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = TokenRegistry()
    return _REGISTRY


def reset_registry_for_tests() -> TokenRegistry:
    """Replace the process-wide registry with a fresh one. Tests only."""
    global _REGISTRY
    _REGISTRY = TokenRegistry()
    return _REGISTRY
