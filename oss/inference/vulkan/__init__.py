"""Vulkan compute inference backend for ORS.

Public surface (see ``runtime`` for full docs):

- ``run_pico_vulkan(...)``  -- one-shot functional API mirroring
  ``OSSPico.forward(...)``.
- ``VulkanPicoRuntime``     -- persistent runtime for sequence inference.
- ``runtime_available()``   -- True iff ncnn + pnnx are importable.
- ``vulkan_available()``    -- True iff NCNN can see a Vulkan-capable GPU.

The v0.2-alpha implementation is NCNN + PNNX based; see
``docs/research/2026-04-30-vulkan-runtime-eval.md`` for the runtime
selection rationale.
"""
from .runtime import (
    VulkanPicoRuntime,
    run_pico_vulkan,
    runtime_available,
    vulkan_available,
)

__all__ = [
    "VulkanPicoRuntime",
    "run_pico_vulkan",
    "runtime_available",
    "vulkan_available",
]
