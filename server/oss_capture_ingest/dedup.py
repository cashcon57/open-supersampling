"""Recent-hash LRU for frame-content deduplication.

The design memo specifies dedup on the **SHA256 of the EXR frame body**
(plus the ``perceptual_hash_64`` from metadata for near-duplicate
detection — that lives client-side in the sampling policy). On the
server we only deduplicate exact bit-identical re-uploads, which catches
uploader retry storms after a 5xx hiccup.

Persistence is a v2 follow-up — this in-memory LRU is sized at 100k
entries by default which is several days of dedup at the rate-limited
cap (1000 frames/hour × 24h × 4 days ≈ 96k).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional


class HashLRU:
    """Thread-safe LRU set of recently-seen content hashes."""

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: "OrderedDict[str, None]" = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def contains(self, content_hash: str) -> bool:
        """Return True if ``content_hash`` was previously added.

        Touches the LRU ordering on hit (so popular hashes stay alive).
        """
        with self._lock:
            if content_hash in self._items:
                self._items.move_to_end(content_hash)
                return True
            return False

    def add(self, content_hash: str) -> bool:
        """Add a hash. Returns True if newly inserted, False if it existed.

        Evicts the oldest entry if capacity would be exceeded.
        """
        with self._lock:
            if content_hash in self._items:
                self._items.move_to_end(content_hash)
                return False
            self._items[content_hash] = None
            if len(self._items) > self.capacity:
                self._items.popitem(last=False)
            return True

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
