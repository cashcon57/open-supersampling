"""Output-drive picker + disk-space janitor for the OSS Capture tray app.

Two responsibilities:

1. ``pick_output_drive`` — given a list of candidate drive letters (e.g.
   ['E:', 'G:']), return the one with the most free space. The tray app
   uses this when ``output_drive_override`` is None in the config.

2. ``cleanup_to_cap`` — walk the captures directory, sum frame sizes, and
   delete oldest frames first until total footprint is under the configured
   cap. Called periodically by the tray app on a low-priority timer.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

log = logging.getLogger("oss.capture.tray.storage")


# Default candidate drives on Cash's machine. The tray app surfaces a
# settings entry to add more, but on a single-user POC this list is fine.
DEFAULT_CANDIDATE_DRIVES = ("G:\\", "<train-host-data>\\")

# Captures live under this subdirectory off the chosen drive root.
CAPTURES_SUBDIR = "oss-captures"


@dataclass
class DriveInfo:
    """Free-space + total-space for a single drive root."""

    root: str
    total_bytes: int
    free_bytes: int

    @property
    def free_gib(self) -> float:
        return self.free_bytes / (1024**3)


def list_candidate_drives(candidates: Iterable[str] = DEFAULT_CANDIDATE_DRIVES) -> List[DriveInfo]:
    """Return DriveInfo for every candidate drive that exists.

    Candidates that don't exist are silently skipped — letting the tray
    app run on a machine with only one of {E:, G:} mounted is the right
    behavior; the user can manually override via settings if neither is
    present.
    """
    out: List[DriveInfo] = []
    for root in candidates:
        if not _drive_exists(root):
            continue
        try:
            usage = shutil.disk_usage(root)
        except OSError as exc:
            log.warning("disk_usage(%s) failed: %s", root, exc)
            continue
        out.append(DriveInfo(root=root, total_bytes=usage.total, free_bytes=usage.free))
    return out


def _drive_exists(root: str) -> bool:
    """Cross-platform check that a drive letter / mount root is present."""
    if platform.system() == "Windows":
        # On Windows, os.path.exists is reliable for drive roots.
        return os.path.exists(root)
    # Non-Windows: drive letters don't exist; treat as missing so dev
    # machines cleanly skip the Windows path.
    return False


def pick_output_drive(
    override: Optional[str] = None,
    candidates: Iterable[str] = DEFAULT_CANDIDATE_DRIVES,
) -> Optional[str]:
    """Pick the output drive root, honoring an override or auto-picking the
    drive with the most free space.

    Returns the drive root string (e.g. ``'G:\\\\'``) or ``None`` if no
    candidate exists and no override is set. The tray UI surfaces the None
    case as "no output drive available — please plug one in or override
    via settings".
    """
    if override is not None:
        return override
    drives = list_candidate_drives(candidates)
    if not drives:
        return None
    drives.sort(key=lambda d: d.free_bytes, reverse=True)
    return drives[0].root


def captures_dir(drive_root: str) -> Path:
    """Return the captures dir under a drive root, creating it if needed."""
    path = Path(drive_root) / CAPTURES_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def total_captures_bytes(drive_root: str) -> int:
    """Sum the size of every file under the captures dir."""
    base = captures_dir(drive_root)
    total = 0
    for root, _dirs, files in os.walk(base):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                # File could be deleted mid-walk by a concurrent
                # writer (the DLL); skip and continue.
                continue
    return total


def cleanup_to_cap(drive_root: str, cap_bytes: int) -> int:
    """Delete oldest frames first until footprint <= cap_bytes.

    Returns the number of files deleted. Walks the captures dir, sorts
    files by mtime (oldest first), and deletes until under cap.

    The DLL writes per-frame .exr files; the cap deliberately operates at
    the file granularity rather than at the session-directory granularity
    so a long single-game session can be partially trimmed without
    discarding the whole session.
    """
    base = captures_dir(drive_root)
    files = []
    for root, _dirs, names in os.walk(base):
        for name in names:
            full = os.path.join(root, name)
            try:
                stat = os.stat(full)
            except OSError:
                continue
            files.append((stat.st_mtime, stat.st_size, full))

    files.sort(key=lambda t: t[0])  # oldest first
    total = sum(size for _mtime, size, _path in files)
    if total <= cap_bytes:
        return 0

    deleted = 0
    for _mtime, size, path in files:
        if total <= cap_bytes:
            break
        try:
            os.remove(path)
            total -= size
            deleted += 1
        except OSError as exc:
            log.warning("cleanup_to_cap: failed to delete %s: %s", path, exc)
            continue

    if deleted:
        log.info(
            "cleanup_to_cap(%s): deleted %d files, footprint now %.2f GiB",
            drive_root, deleted, total / (1024**3),
        )
    return deleted
