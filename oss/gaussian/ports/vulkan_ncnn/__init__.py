"""Vulkan + ncnn port of the OSS-Gaussian renderer (Steam Deck target).

Sprint 7 / Track V scaffold. The GLSL compute kernel + C++ host harness live
as sibling files (`rasterizer.comp`, `rasterizer.cpp`); ncnn export of the
Sprint 4 ``GaussianParamNetwork`` lives in ``export_ncnn``.

This package is import-safe on every platform — heavy SDK imports (``ncnn``,
``pnnx``) are deferred to the modules that need them and guarded behind
``try/except ImportError``.
"""

from __future__ import annotations

import shutil


def has_vulkan_toolchain() -> bool:
    """Return True iff `glslangValidator` is on PATH.

    The renderer port (T7.V.1) needs the Vulkan SDK installed; this is the
    cheapest portable check that doesn't require importing any Python
    Vulkan binding.
    """
    return shutil.which("glslangValidator") is not None


def has_ncnn() -> bool:
    """Return True iff the ``ncnn`` Python module imports.

    ncnn ships pre-built wheels under the ``[vulkan]`` extra; this check
    lets tests skip cleanly when the extra has not been installed.
    """
    try:
        import ncnn  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = ["has_vulkan_toolchain", "has_ncnn"]
