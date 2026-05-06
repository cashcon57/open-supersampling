"""v6 model orchestrator stub.

The full V6Model wires together HAT spatial backbone, persistent
Gaussian canvas, analytical warp with covariance resampling,
cross-attention between pixel features and Gaussian tokens,
score-based pruning, and the key-frame active-Gaussian mask. See
``docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md``
section 2.

This file is the import target of ``scripts/sr_train_v6.py``.
Construction raises NotImplementedError until the integration commit
that follows the parallel-tier work; the training script's --smoke flag
exits cleanly when this is the case rather than crashing with an
ImportError.
"""
from __future__ import annotations

from typing import Any


class V6Model:
    """Orchestrator stub. Constructing this raises NotImplementedError so
    that callers (e.g. ``scripts/sr_train_v6.py``) can detect the unfinished
    integration and exit cleanly instead of crashing with an attribute or
    import error.
    """

    NOT_IMPLEMENTED_MESSAGE = (
        "V6Model is not yet implemented. The orchestrator wiring HAT + canvas "
        "+ cross-attention + covariance-resampled warp + ST-score pruning + "
        "active-mask rasterizer lands in a follow-up commit after the "
        "individual components are merged. See "
        "docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md "
        "section 2 for the integration plan."
    )

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(self.NOT_IMPLEMENTED_MESSAGE)


__all__ = ["V6Model"]
