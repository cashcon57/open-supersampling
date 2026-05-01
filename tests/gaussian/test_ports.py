"""Sprint 7 cross-platform port scaffold tests.

These are placeholders — real validation runs on actual hardware in the
Sprint 7 implementation phase. The goal here is:

1. Confirm the scaffold imports cleanly on every platform.
2. Run ``--dry-run`` style export paths that build the traceable model
   without invoking ``coremltools`` / ``pnnx`` (so CI on Linux can still
   exercise the export plumbing).
3. Skip tests that genuinely need the hardware (Metal compiler, Vulkan SDK,
   ncnn runtime) when those aren't present, rather than fail.
"""

from __future__ import annotations

import platform
import shutil

import pytest

from oss.gaussian.ports.metal import is_supported as metal_supported
from oss.gaussian.ports.vulkan_ncnn import has_ncnn, has_vulkan_toolchain


# ---------------------------------------------------------------------------
# Scaffold importability — every platform.
# ---------------------------------------------------------------------------

def test_metal_package_imports():
    """The metal port package imports on every host (no Metal SDK required)."""
    from oss.gaussian.ports import metal  # noqa: F401

    assert callable(metal.is_supported)
    # is_supported() returns a bool regardless of host.
    assert isinstance(metal.is_supported(), bool)


def test_vulkan_ncnn_package_imports():
    """The Vulkan/ncnn port package imports on every host."""
    from oss.gaussian.ports import vulkan_ncnn  # noqa: F401

    assert callable(vulkan_ncnn.has_vulkan_toolchain)
    assert callable(vulkan_ncnn.has_ncnn)


def test_metal_export_module_imports():
    """The CoreML export module imports without coremltools installed."""
    from oss.gaussian.ports.metal import export_coreml

    assert hasattr(export_coreml, "export")
    assert hasattr(export_coreml, "build_traceable")
    assert hasattr(export_coreml, "parity_smoke_test")


def test_ncnn_export_module_imports():
    """The ncnn export module imports without ncnn/pnnx installed."""
    from oss.gaussian.ports.vulkan_ncnn import export_ncnn

    assert hasattr(export_ncnn, "export")
    assert hasattr(export_ncnn, "build_traceable")
    assert hasattr(export_ncnn, "parity_smoke_test")


# ---------------------------------------------------------------------------
# Dry-run export paths — every platform.
# ---------------------------------------------------------------------------

def test_metal_export_coreml_dry_run():
    """Dry-run CoreML export builds the traceable model + dummy input.

    No coremltools dependency. Verifies the exporter wires the network
    correctly even before Sprint 4 produces a checkpoint.
    """
    from oss.gaussian.ports.metal.export_coreml import export

    traced = export(tier="pico", lr_hw=(64, 64), dry_run=True)
    assert traced is not None
    # Traced module should be callable.
    assert callable(traced)


def test_metal_export_coreml_parity_smoke():
    """Inference-mode determinism smoke test for the CoreML export plumbing."""
    from oss.gaussian.ports.metal.export_coreml import parity_smoke_test

    diff = parity_smoke_test(tier="pico", lr_hw=(64, 64))
    # Inference-mode forward should be exactly deterministic.
    assert diff == pytest.approx(0.0, abs=1e-6)


def test_ncnn_export_dry_run():
    """Dry-run ncnn export builds the traceable model + dummy input."""
    from oss.gaussian.ports.vulkan_ncnn.export_ncnn import export

    traced = export(tier="pico", lr_hw=(64, 64), dry_run=True)
    assert traced is not None
    assert callable(traced)


def test_ncnn_export_parity_smoke():
    """Inference-mode determinism smoke test for the ncnn export plumbing."""
    from oss.gaussian.ports.vulkan_ncnn.export_ncnn import parity_smoke_test

    diff = parity_smoke_test(tier="pico", lr_hw=(64, 64))
    assert diff == pytest.approx(0.0, abs=1e-6)


def test_metal_build_traceable_shape():
    """Traced model produces the expected raw output shape per param_net."""
    from oss.gaussian.network.param_net import param_net_for_tier
    from oss.gaussian.ports.metal.export_coreml import build_traceable

    model, dummy = build_traceable("pico", (64, 64))
    expected = param_net_for_tier("pico").output_shape(64, 64)
    out = model(dummy)
    assert out.shape[1:] == expected, f"got {tuple(out.shape[1:])}, expected {expected}"


def test_ncnn_build_traceable_shape():
    """Same shape contract on the ncnn side — it's the same network."""
    from oss.gaussian.network.param_net import param_net_for_tier
    from oss.gaussian.ports.vulkan_ncnn.export_ncnn import build_traceable

    model, dummy = build_traceable("pico", (64, 64))
    expected = param_net_for_tier("pico").output_shape(64, 64)
    out = model(dummy)
    assert out.shape[1:] == expected


# ---------------------------------------------------------------------------
# Hardware-conditional tests — skip when the toolchain isn't present.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not metal_supported(),
    reason="Metal port is macOS arm64 only; current host is "
           f"{platform.system()} {platform.machine()}",
)
def test_metal_compiler_available():
    """On macOS arm64, `xcrun metal` should resolve.

    Skips on every other host. Exists as a Sprint 7 readiness probe rather
    than a real test.
    """
    metal_bin = shutil.which("xcrun")
    assert metal_bin is not None, "xcrun not on PATH on a macOS arm64 host"


@pytest.mark.skipif(
    not has_vulkan_toolchain(),
    reason="Vulkan SDK (glslangValidator) not on PATH",
)
def test_vulkan_toolchain_available():
    """When glslangValidator is present, the scaffold should be buildable."""
    assert shutil.which("glslangValidator") is not None


@pytest.mark.skipif(
    not has_ncnn(),
    reason="ncnn Python module not installed (need `[vulkan]` extra)",
)
def test_ncnn_runtime_imports():
    """When the [vulkan] extra is installed, ncnn should import cleanly."""
    import ncnn  # type: ignore[import-not-found]

    assert hasattr(ncnn, "Net")
