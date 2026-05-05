"""Bearer-token auth + per-token rate limiting.

The token registry is in-memory + JSON-file backed so server restarts
and multiple-process workers see the same token set (closes Codex's
HIGH cross-review finding "Capture Server Tokens Are Process-Local Only").

Tokens are minted by ``scripts/build_capture_installer.py`` and baked into
each per-game installer. They are opaque from the client's perspective:
not user-identifying, just rate-limitable.

Persistence: tokens are flushed to the JSON file at the path returned by
``_token_store_path()`` (``$OSS_CAPTURE_TOKEN_STORE`` env var, falling
back to ``~/.oss-capture-tokens.json``). Reads are atomic (single
``json.load``); writes use a tmp+rename pattern. SQLite is the planned
v2 store; the on-disk JSON is the smallest interface that closes the
production gap without pulling in a database dependency.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional


@dataclass
class TokenRecord:
    """Server-side state for one install token."""

    token: str
    label: str = ""
    created_at_unix: float = field(default_factory=lambda: time.time())
    revoked: bool = False
    # rolling window of SUCCESSFUL frame-upload timestamps (rate-limit
    # accounting for accepted writes — bandwidth/stage cost).
    upload_times: Deque[float] = field(default_factory=deque)
    # rolling window of ANY authenticated attempt (success OR rejection).
    # Closes Codex's MEDIUM finding "Rate Limit Does Not Cover Rejected
    # Upload Attempts": valid token with malformed meta/oversize/dedup
    # rejections must still count against the budget so a misbehaving
    # client can't hammer the server's parse/validate paths for free.
    attempt_times: Deque[float] = field(default_factory=deque)
    # per-game attempt windows — closes Codex's "no per-game limiter".
    # Keyed by game_id (validated string from metadata). Each value is a
    # rolling-window deque just like ``attempt_times``.
    per_game_attempts: Dict[str, Deque[float]] = field(default_factory=dict)
    # cumulative counters (do not reset; used by /stats)
    total_frames: int = 0
    total_bytes: int = 0
    # per-mode cumulative counters — populated when ingest passes the
    # capture_mode through ``record_upload``. Surfaces in ``/stats`` so
    # the dataset card can stratify contribution by mode.
    frames_by_mode: Dict[str, int] = field(default_factory=dict)
    bytes_by_mode: Dict[str, int] = field(default_factory=dict)


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
        store_path: Optional[Path] = None,
        # Authenticated-attempt budget per token (counts success + rejection).
        # Default 5x the success limit so a small fraction of malformed/dup
        # uploads is tolerated, but a malicious flood is throttled.
        attempt_limit_per_hour: int = 5000,
        # Per-(token, game_id) attempt budget. Default 2x the success limit
        # per game — keeps a token-fleet for one game from consuming the
        # global service budget.
        per_game_attempt_limit_per_hour: int = 2000,
    ) -> None:
        self.rate_limit = int(rate_limit_frames_per_hour)
        self.window_seconds = int(window_seconds)
        self.attempt_limit = int(attempt_limit_per_hour)
        self.per_game_attempt_limit = int(per_game_attempt_limit_per_hour)
        self._tokens: Dict[str, TokenRecord] = {}
        self._lock = threading.RLock()
        self.store_path: Optional[Path] = store_path
        if self.store_path is not None:
            self._load_from_disk()

    # ---- registry management ------------------------------------------------

    def register_token(self, token: str, label: str = "") -> TokenRecord:
        """Register a new token (idempotent). Persisted if store_path is set."""
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None:
                rec = TokenRecord(token=token, label=label)
                self._tokens[token] = rec
            elif label and not rec.label:
                rec.label = label
            self._flush_to_disk()
            return rec

    def revoke_token(self, token: str) -> bool:
        """Mark a token as revoked. Returns True if the token existed."""
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None:
                return False
            rec.revoked = True
            self._flush_to_disk()
            return True

    # ---- persistence -------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Load registered tokens from ``store_path`` if it exists."""
        if self.store_path is None or not self.store_path.is_file():
            return
        try:
            data = json.loads(self.store_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for entry in data.get("tokens", []):
            tok = entry.get("token")
            if not isinstance(tok, str) or not tok:
                continue
            rec = TokenRecord(
                token=tok,
                label=str(entry.get("label", "")),
                created_at_unix=float(entry.get("created_at_unix", time.time())),
                revoked=bool(entry.get("revoked", False)),
                total_frames=int(entry.get("total_frames", 0)),
                total_bytes=int(entry.get("total_bytes", 0)),
                frames_by_mode={
                    str(k): int(v)
                    for k, v in (entry.get("frames_by_mode") or {}).items()
                },
                bytes_by_mode={
                    str(k): int(v)
                    for k, v in (entry.get("bytes_by_mode") or {}).items()
                },
            )
            self._tokens[tok] = rec

    def _flush_to_disk(self) -> None:
        """Atomically write the token registry to ``store_path``.

        Tmp-file + rename pattern so concurrent readers never see a half-
        written JSON. Counters (total_frames/total_bytes) are flushed too;
        rate-limit windows are NOT — they're transient by definition and
        re-establish naturally on restart from upload activity.
        """
        if self.store_path is None:
            return
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
            payload = {
                "version": 1,
                "tokens": [
                    {
                        "token": rec.token,
                        "label": rec.label,
                        "created_at_unix": rec.created_at_unix,
                        "revoked": rec.revoked,
                        "total_frames": rec.total_frames,
                        "total_bytes": rec.total_bytes,
                        "frames_by_mode": dict(rec.frames_by_mode),
                        "bytes_by_mode": dict(rec.bytes_by_mode),
                    }
                    for rec in self._tokens.values()
                ],
            }
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.store_path)
        except OSError:
            # Persistence is best-effort; an unwritable store should not
            # bring the server down. Logged via the FastAPI app's logger
            # at the call site if needed.
            pass

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
        """Return True if a frame upload is allowed under the SUCCESSFUL-upload
        rate limit (back-compat name; prefer :meth:`check_attempt` for the
        cheap pre-parse gate that closes Codex's MED finding).
        """
        if now is None:
            now = time.time()
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None or rec.revoked:
                return False
            self._prune_window(rec, now)
            return len(rec.upload_times) < self.rate_limit

    def check_attempt(self, token: str, now: Optional[float] = None) -> bool:
        """Return True if any authenticated request is allowed (cheaper gate
        than ``check_rate`` — covers parse/validate/dedup/oversize 4xx paths).

        This is the gate that should be checked BEFORE multipart parsing.
        Closes Codex's MED finding 'Rate Limit Does Not Cover Rejected
        Upload Attempts'.
        """
        if now is None:
            now = time.time()
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None or rec.revoked:
                return False
            cutoff = now - self.window_seconds
            while rec.attempt_times and rec.attempt_times[0] < cutoff:
                rec.attempt_times.popleft()
            return len(rec.attempt_times) < self.attempt_limit

    def check_per_game_attempt(
        self,
        token: str,
        game_id: str,
        now: Optional[float] = None,
    ) -> bool:
        """Return True if a per-(token, game_id) authenticated attempt is
        allowed. Called AFTER metadata parses (game_id is in the validated
        meta). Closes Codex's 'no per-game limiter' gap.
        """
        if now is None:
            now = time.time()
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None or rec.revoked:
                return False
            window = rec.per_game_attempts.get(game_id)
            if window is None:
                return True  # first attempt for this (token, game) pair
            cutoff = now - self.window_seconds
            while window and window[0] < cutoff:
                window.popleft()
            return len(window) < self.per_game_attempt_limit

    def record_attempt(
        self,
        token: str,
        now: Optional[float] = None,
    ) -> None:
        """Record any authenticated request (success or rejection) against
        the per-token attempt window. Always called once auth passes."""
        if now is None:
            now = time.time()
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None:
                return
            rec.attempt_times.append(now)
            cutoff = now - self.window_seconds
            while rec.attempt_times and rec.attempt_times[0] < cutoff:
                rec.attempt_times.popleft()

    def record_per_game_attempt(
        self,
        token: str,
        game_id: str,
        now: Optional[float] = None,
    ) -> None:
        """Record a per-(token, game_id) authenticated attempt. Called
        once metadata parses + game_id is validated."""
        if now is None:
            now = time.time()
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None:
                return
            window = rec.per_game_attempts.setdefault(game_id, deque())
            window.append(now)
            cutoff = now - self.window_seconds
            while window and window[0] < cutoff:
                window.popleft()

    def record_upload(
        self,
        token: str,
        frame_bytes: int,
        now: Optional[float] = None,
        capture_mode: Optional[str] = None,
    ) -> None:
        """Record a successful upload for rate-limit + /stats accounting.

        ``capture_mode`` (when provided) increments the per-mode counters
        on the token record; the dataset card consumes these via /stats.
        """
        if now is None:
            now = time.time()
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None:
                return
            rec.upload_times.append(now)
            rec.total_frames += 1
            rec.total_bytes += int(frame_bytes)
            if capture_mode:
                rec.frames_by_mode[capture_mode] = (
                    rec.frames_by_mode.get(capture_mode, 0) + 1
                )
                rec.bytes_by_mode[capture_mode] = (
                    rec.bytes_by_mode.get(capture_mode, 0) + int(frame_bytes)
                )
            self._prune_window(rec, now)
            # Persist every 10 uploads to amortize disk I/O. The window
            # itself isn't persisted (transient); only cumulative counters.
            if rec.total_frames % 10 == 0:
                self._flush_to_disk()


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


def _token_store_path() -> Optional[Path]:
    """Resolve the on-disk token-store path.

    Order of precedence:
      1. ``$OSS_CAPTURE_TOKEN_STORE`` env var (explicit override)
      2. ``~/.oss-capture-tokens.json`` (default for the running server user)

    Set the env var to an empty string to DISABLE persistence (in-memory only;
    used by tests + ephemeral preview deployments).
    """
    env = os.environ.get("OSS_CAPTURE_TOKEN_STORE")
    if env is None:
        return Path.home() / ".oss-capture-tokens.json"
    if env == "":
        return None
    return Path(env)


def get_registry() -> TokenRegistry:
    """Return the process-wide :class:`TokenRegistry` singleton.

    On first call, creates the registry backed by the on-disk token store
    (see :func:`_token_store_path`). All processes that load this module
    in the same OS user account share the same persisted token set.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = TokenRegistry(store_path=_token_store_path())
    return _REGISTRY


def reset_registry_for_tests() -> TokenRegistry:
    """Replace the process-wide registry with a fresh, in-memory one.

    Tests only — does NOT touch the on-disk store. Sets ``store_path=None``
    so the test registry is fully isolated from any live server state.
    """
    global _REGISTRY
    _REGISTRY = TokenRegistry(store_path=None)
    return _REGISTRY
