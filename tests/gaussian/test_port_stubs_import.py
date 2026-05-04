"""Smoke-import vendor port scaffolds.

These tests intentionally do not validate runtime SDK availability. The port
packages are scaffolds and should at least remain import-safe on a generic
developer machine. Optional heavy dependencies are expected to be imported only
inside export/runtime functions, not at module import time.
"""
from __future__ import annotations

import importlib
import pkgutil
import platform

import pytest

import oss.gaussian.ports


def _discover_port_modules() -> list[str]:
    roots = [oss.gaussian.ports]
    try:
        sr_ports = importlib.import_module("oss.sr.ports")
    except ImportError:
        pass
    else:
        roots.append(sr_ports)

    modules: list[str] = []
    for root in roots:
        prefix = root.__name__ + "."
        for info in pkgutil.walk_packages(root.__path__, prefix=prefix):
            modules.append(info.name)
    return sorted(modules)


PORT_MODULES = _discover_port_modules()


def _skip_reason(module_name: str) -> str | None:
    if "coreml" in module_name and platform.system() != "Darwin":
        return "CoreML export scaffolding is macOS-only"
    return None


@pytest.mark.parametrize("module_name", PORT_MODULES)
def test_vendor_port_stub_imports(module_name: str) -> None:
    reason = _skip_reason(module_name)
    if reason:
        pytest.skip(reason)
    mod = importlib.import_module(module_name)
    assert mod is not None
