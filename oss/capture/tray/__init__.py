"""OSS Capture local-only tray app for Windows.

Runs in the system tray. Picks the larger of the configured output drives,
writes captured frames there directly (no upload, no server), and lets the
user switch capture modes (trickle / lite / regular / INSANE) at runtime.

Targets the local-only single-machine workflow described in the README's
v0 Windows-PC scope. The shipping multi-user upload version reuses the
same DLL hook with the upload pathway re-enabled.

Entry point: ``python -m oss.capture.tray``.
"""
from __future__ import annotations

__all__ = []
