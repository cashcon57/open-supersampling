"""Metal compute port of the OSS-Gaussian renderer.

Sprint 7 / Track M scaffold. The MSL kernel + Swift host harness live as
sibling files (`rasterizer.metal`, `rasterizer.swift`); CoreML export of the
Sprint 4 ``GaussianParamNetwork`` lives in ``export_coreml``.

This package is import-safe on every platform — heavy SDK imports (CoreML,
PyObjC) are deferred to the modules that need them and guarded behind
``try/except ImportError``.
"""

from __future__ import annotations

import platform


def is_supported() -> bool:
    """Return True iff the current host can run the Metal port.

    The check is conservative: macOS arm64 only. Intel Macs technically have
    Metal but lack the M-series ``simdgroup_matrix`` / unified-memory path
    the kernel assumes.
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


__all__ = ["is_supported"]
