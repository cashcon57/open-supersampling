"""Win32 process watcher for the OSS Capture tray app.

Polls the running-process list once per second using
``CreateToolhelp32Snapshot`` (via pywin32 / win32process). When a new
process matches an allowlisted game executable AND no kernel-anti-cheat
process is also resident on the system, fires a callback so the tray app
can decide whether to inject.

This is a polling watcher rather than an event-driven one because:
  - WMI process-creation event subscriptions are noisy and add a WMI
    service dependency (sometimes flaky on Windows).
  - 1 Hz is fast enough for a "user just launched a game" UX latency
    target (0-1 second between game-window-shown and tray-app-injects).
  - Polling can be implemented in pure pywin32 with no extra deps.
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from oss.capture.tray import allowlist

log = logging.getLogger("oss.capture.tray.process_watcher")


@dataclass
class GameLaunchEvent:
    """Fired when an allowlisted game is detected as newly running."""

    pid: int
    exe_basename: str
    allowed: allowlist.AllowedGame
    """The allowlist entry that matched the running exe."""


@dataclass
class AntiCheatBlockEvent:
    """Fired when an allowlisted game is detected BUT a kernel anti-cheat
    process is also resident on the system. The tray app surfaces this as
    an explicit notification ("Cyberpunk launched, but Vanguard is also
    running — refusing to inject for your account safety") and DOES NOT
    inject."""

    pid: int
    exe_basename: str
    blocking_processes: tuple
    """The kernel-AC process names that triggered the block."""


GameLaunchCallback = Callable[[GameLaunchEvent], None]
AntiCheatBlockCallback = Callable[[AntiCheatBlockEvent], None]


def _list_processes_win32() -> Dict[int, str]:
    """Return {pid: exe_basename_lower} for every running process.

    Uses pywin32's win32process + win32api. Requires the tray app to be
    running on Windows; raises ImportError on other platforms.
    """
    try:
        import win32process  # type: ignore[import-not-found]
        import win32api      # type: ignore[import-not-found]
        import win32con      # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "process_watcher requires pywin32; install with: pip install pywin32"
        ) from exc

    pids = win32process.EnumProcesses()
    out: Dict[int, str] = {}
    for pid in pids:
        try:
            handle = win32api.OpenProcess(
                win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
            )
        except Exception:
            # Many system PIDs cannot be opened by a non-elevated tray app.
            # That's fine — those PIDs are not games we'd inject into
            # anyway. Skip silently.
            continue
        try:
            exe_path = win32process.GetModuleFileNameEx(handle, 0)
            basename = exe_path.rsplit("\\", 1)[-1].lower()
            out[pid] = basename
        except Exception:
            continue
        finally:
            try:
                win32api.CloseHandle(handle)
            except Exception:
                pass
    return out


def _list_processes_stub() -> Dict[int, str]:
    """Non-Windows fallback: returns empty dict so the watcher can be
    instantiated for unit tests on macOS / Linux without crashing."""
    return {}


class ProcessWatcher:
    """Background thread that polls the process list and fires callbacks.

    Lifecycle:
      watcher = ProcessWatcher(on_game_launch, on_anticheat_block)
      watcher.start()
      ...  # tray app runs
      watcher.stop()
    """

    def __init__(
        self,
        on_game_launch: GameLaunchCallback,
        on_anticheat_block: AntiCheatBlockCallback,
        poll_interval_sec: float = 1.0,
    ) -> None:
        self._on_game_launch = on_game_launch
        self._on_anticheat_block = on_anticheat_block
        self._poll_interval = poll_interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._known_game_pids: set = set()
        # Pick the right enumerator at construction time so unit tests on
        # non-Windows hosts can run without pywin32.
        self._enumerate = (
            _list_processes_win32 if platform.system() == "Windows"
            else _list_processes_stub
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="oss-capture-process-watcher", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        log.info("process watcher started (poll=%ss)", self._poll_interval)
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("process watcher tick failed; continuing")
            self._stop.wait(timeout=self._poll_interval)
        log.info("process watcher stopped")

    def _tick(self) -> None:
        procs = self._enumerate()  # {pid: exe_lower}
        if not procs:
            return

        # Detect kernel-anti-cheat processes resident system-wide. If any
        # are present, refuse to inject into ANY game on this poll cycle.
        ac_lower = {n.lower() for n in allowlist.KERNEL_ANTICHEAT_PROCESSES}
        ac_hits = {basename for basename in procs.values() if basename in ac_lower}

        # Find matching game launches that we haven't seen yet.
        for pid, basename in procs.items():
            if pid in self._known_game_pids:
                continue
            game = allowlist.lookup_by_exe(basename)
            if game is None:
                continue
            self._known_game_pids.add(pid)
            if ac_hits:
                log.warning(
                    "%s detected (pid=%d) but kernel anti-cheat is resident: %s. "
                    "Refusing to inject.",
                    basename, pid, sorted(ac_hits),
                )
                self._on_anticheat_block(AntiCheatBlockEvent(
                    pid=pid, exe_basename=basename, blocking_processes=tuple(sorted(ac_hits)),
                ))
                continue
            log.info("game launch: %s (pid=%d, %s)", basename, pid, game.display_name)
            self._on_game_launch(GameLaunchEvent(
                pid=pid, exe_basename=basename, allowed=game,
            ))

        # Forget PIDs that have exited so a re-launch fires the callback again.
        self._known_game_pids &= set(procs.keys())
