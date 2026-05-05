"""Recent-hash LRU + R2-backed durable store for frame-content deduplication.

The design memo specifies dedup on the **SHA256 of the EXR frame body**
(plus the ``perceptual_hash_64`` from metadata for near-duplicate
detection — that lives client-side in the sampling policy). On the
server we only deduplicate exact bit-identical re-uploads, which catches
uploader retry storms after a 5xx hiccup.

**Durability (closes Codex's MED 'volatile dedup' finding):**
The in-memory LRU is now a *cache* in front of an R2-backed durable
store. On a cache miss, we ``head_object`` a tiny dedup-marker key in R2
(see :func:`server.oss_capture_ingest.r2.dedup_key`). After a successful
upload, the ingest path writes the marker so subsequent restarts /
multi-process workers / region-failover deployments still catch dupes.

The LRU still does most of the work — the R2 head only fires on misses
and is bounded by the standard rate limiter, so a flood of unique hashes
costs at most one R2 head per attempt that the rate limiter has already
green-lit.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Optional

log = logging.getLogger(__name__)


class HashLRU:
    """Thread-safe LRU set of recently-seen content hashes.

    Optionally backed by an R2 store via :meth:`set_durable_backend` so
    ``contains`` survives process restarts. The backend is a duck-typed
    object exposing ``head_dedup(hash) -> bool`` and
    ``put_dedup(hash) -> None`` — concretely the
    :class:`server.oss_capture_ingest.r2.R2Client`.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: "OrderedDict[str, None]" = OrderedDict()
        self._lock = threading.Lock()
        self._backend: Optional[Any] = None

    def set_durable_backend(self, backend: Optional[Any]) -> None:
        """Wire (or unset) the R2-backed durable dedup store."""
        with self._lock:
            self._backend = backend

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def contains(self, content_hash: str) -> bool:
        """Return True if ``content_hash`` was previously added.

        On a hot-cache hit, touches LRU ordering. On a miss, falls back
        to the durable backend (if wired); a hit there hydrates the LRU
        so subsequent checks short-circuit.

        Backend errors are swallowed and logged — the LRU answer is
        authoritative as a fail-open fallback. A flaky R2 must NOT cause
        unique frames to be silently rejected.
        """
        with self._lock:
            if content_hash in self._items:
                self._items.move_to_end(content_hash)
                return True
            backend = self._backend
        if backend is None:
            return False
        try:
            hit = bool(backend.head_dedup(content_hash))
        except Exception as exc:  # pragma: no cover — backend dep
            log.warning("dedup backend head_dedup failed: %s", exc)
            return False
        if hit:
            with self._lock:
                self._items[content_hash] = None
                self._items.move_to_end(content_hash)
                if len(self._items) > self.capacity:
                    self._items.popitem(last=False)
        return hit

    def add(self, content_hash: str) -> bool:
        """Add a hash to the hot LRU. Returns True if newly inserted.

        Does NOT write to the durable backend — the ingest path calls
        :meth:`add_durable` separately so a partial R2 write doesn't
        leave the cache out of sync with the bucket.
        """
        with self._lock:
            if content_hash in self._items:
                self._items.move_to_end(content_hash)
                return False
            self._items[content_hash] = None
            if len(self._items) > self.capacity:
                self._items.popitem(last=False)
            return True

    def add_durable(self, content_hash: str) -> None:
        """Persist ``content_hash`` to the durable backend (if wired).

        Best-effort: backend errors are logged and swallowed. The LRU is
        always authoritative within a single process; the durable store
        only becomes load-bearing after a restart.
        """
        with self._lock:
            backend = self._backend
        if backend is None:
            return
        try:
            backend.put_dedup(content_hash)
        except Exception as exc:  # pragma: no cover — backend dep
            log.warning("dedup backend put_dedup failed: %s", exc)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


# ---- module-level singleton -------------------------------------------------

_DEDUP: Optional[HashLRU] = None


def get_dedup() -> HashLRU:
    """Return the process-wide :class:`HashLRU` singleton."""
    global _DEDUP
    if _DEDUP is None:
        _DEDUP = HashLRU()
    return _DEDUP


def reset_dedup_for_tests(capacity: int = 100_000) -> HashLRU:
    """Replace the process-wide LRU with a fresh one. Tests only."""
    global _DEDUP
    _DEDUP = HashLRU(capacity=capacity)
    return _DEDUP
