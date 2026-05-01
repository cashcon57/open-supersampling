"""Sprint 1 / T1.7 — integration smoke test.

Verifies the new Gaussian renderer module coexists with the existing
pixel-based OSS modules without import collisions or interference. Existing
OSS pixel-based tests must remain green after the Gaussian module lands.
"""

from __future__ import annotations

import pytest


def test_gaussian_renderer_importable() -> None:
    """The renderer module must import cleanly on any machine, regardless of
    CUDA availability."""
    from oss.gaussian.renderer import GaussianBatch, Rasterizer, TILE_SIZE
    assert TILE_SIZE == 16
    assert callable(Rasterizer)
    assert callable(GaussianBatch)


def test_pixel_based_oss_still_importable() -> None:
    """Importing the new module must not break the existing pixel-based OSS
    classes."""
    # If any of these fails, the Gaussian track is leaking into the existing
    # namespace.
    from oss.model.oss import OSS  # noqa: F401
    from oss.model.oss_pico import OSSPico  # noqa: F401
    from oss.model.oss_rg import OSSRG  # noqa: F401
    from oss.model.oss_fx import OSSFx  # noqa: F401


def test_no_namespace_collision_between_tracks() -> None:
    """`oss.gaussian` is a sibling of `oss.model`, not a parent. Confirm
    namespace isolation."""
    import oss
    import oss.gaussian
    import oss.model

    assert oss.gaussian.__name__ == "oss.gaussian"
    assert oss.model.__name__ == "oss.model"
    # Submodules don't shadow each other.
    assert oss.gaussian is not oss.model


def test_review_pipeline_module_importable() -> None:
    """The cross-cutting code review pipeline must be importable for sprint
    checkpoints."""
    from oss.gaussian.review import schema, reviewers, judge, run  # noqa: F401


def test_renderer_default_construction_works() -> None:
    """Default Rasterizer config must construct without errors on any
    machine."""
    from oss.gaussian.renderer import Rasterizer
    r = Rasterizer()
    assert r.tile_size == 16
    assert r.topk_norm is True
    assert r.force_backend is None


@pytest.mark.gpu
def test_gpu_marker_exists() -> None:
    """The `gpu` marker must be registered so CUDA-only Gaussian tests can be
    selected/deselected via -m."""
    # Self-test: this test itself uses the marker.
    pass
