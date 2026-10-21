"""Win32 LoadLibrary injector for the OSS Capture tray app.

On Windows this injects a DLL into an already-running game process using
``CreateRemoteThread`` with ``LoadLibraryW``. On non-Windows hosts the
module returns a skipped result so the tray code and tests can run during
cross-platform development.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_DLL_NAME = "oss_capture.dll"
DEFAULT_INJECTION_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class InjectionResult:
    pid: int
    dll_path: Path
    injected: bool
    skipped: bool = False
    dry_run: bool = False
    message: str = ""


def default_dll_path() -> Path:
    override = os.environ.get("OSS_CAPTURE_DLL_PATH")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA")
    if base is None:
        base = os.path.expanduser("~\\AppData\\Local")
    return Path(base) / "oss-capture" / DEFAULT_DLL_NAME


def _env_dry_run() -> bool:
    return os.environ.get("OSS_CAPTURE_INJECTION_DRY_RUN", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def inject_dll(
    pid: int,
    dll_path: Optional[Path | str] = None,
    *,
    dry_run: bool = False,
    timeout_seconds: float = DEFAULT_INJECTION_TIMEOUT_SECONDS,
) -> InjectionResult:
    """Inject ``dll_path`` into ``pid``.

    Returns a structured result instead of exiting so the tray app can log
    failures and keep running.
    """
    path = Path(dll_path) if dll_path is not None else default_dll_path()
    if dry_run or _env_dry_run():
        return InjectionResult(
            pid=pid,
            dll_path=path,
            injected=False,
            skipped=True,
            dry_run=True,
            message="DLL injection dry-run",
        )
    if platform.system() != "Windows":
        return InjectionResult(
            pid=pid,
            dll_path=path,
            injected=False,
            skipped=True,
            message="DLL injection is Windows-only",
        )
    if not path.is_file():
        return InjectionResult(
            pid=pid,
            dll_path=path,
            injected=False,
            message=f"DLL not found: {path}",
        )
    return _inject_windows(pid, path, timeout_seconds=timeout_seconds)


def _inject_windows(pid: int, dll_path: Path, *, timeout_seconds: float) -> InjectionResult:
    try:
        import win32api  # type: ignore[import-not-found]
        import win32con  # type: ignore[import-not-found]
        import win32event  # type: ignore[import-not-found]
        import win32process  # type: ignore[import-not-found]
    except ImportError as exc:
        return InjectionResult(
            pid=pid,
            dll_path=dll_path,
            injected=False,
            message=f"pywin32 is required for DLL injection: {exc}",
        )

    access = (
        win32con.PROCESS_CREATE_THREAD
        | win32con.PROCESS_QUERY_INFORMATION
        | win32con.PROCESS_VM_OPERATION
        | win32con.PROCESS_VM_WRITE
        | win32con.PROCESS_VM_READ
    )
    encoded = (str(dll_path.resolve()) + "\0").encode("utf-16-le")
    process_handle = None
    thread_handle = None
    remote_mem = None

    try:
        process_handle = win32api.OpenProcess(access, False, pid)
        load_library = win32api.GetProcAddress(
            win32api.GetModuleHandle("kernel32.dll"),
            "LoadLibraryW",
        )
        remote_mem = win32process.VirtualAllocEx(
            process_handle,
            0,
            len(encoded),
            win32con.MEM_RESERVE | win32con.MEM_COMMIT,
            win32con.PAGE_READWRITE,
        )
        win32process.WriteProcessMemory(process_handle, remote_mem, encoded)
        created = win32process.CreateRemoteThread(
            process_handle,
            None,
            0,
            load_library,
            remote_mem,
            0,
        )
        if isinstance(created, tuple):
            thread_handle = created[0]
        else:
            thread_handle = created
        timeout_ms = max(1, int(timeout_seconds * 1000.0))
        wait_result = win32event.WaitForSingleObject(thread_handle, timeout_ms)
        wait_object_0 = getattr(win32event, "WAIT_OBJECT_0", 0)
        wait_timeout = getattr(win32event, "WAIT_TIMEOUT", 0x00000102)
        if wait_result == wait_timeout:
            return InjectionResult(
                pid=pid,
                dll_path=dll_path,
                injected=False,
                message=f"LoadLibraryW timed out after {timeout_seconds:g}s",
            )
        if wait_result != wait_object_0:
            return InjectionResult(
                pid=pid,
                dll_path=dll_path,
                injected=False,
                message=f"LoadLibraryW wait failed: {wait_result}",
            )
        exit_code = win32process.GetExitCodeThread(thread_handle)
        if exit_code == 0:
            return InjectionResult(
                pid=pid,
                dll_path=dll_path,
                injected=False,
                message="LoadLibraryW returned NULL",
            )
        return InjectionResult(
            pid=pid,
            dll_path=dll_path,
            injected=True,
            message="injected",
        )
    except Exception as exc:
        return InjectionResult(
            pid=pid,
            dll_path=dll_path,
            injected=False,
            message=str(exc),
        )
    finally:
        if remote_mem is not None and process_handle is not None:
            try:
                win32process.VirtualFreeEx(
                    process_handle,
                    remote_mem,
                    0,
                    win32con.MEM_RELEASE,
                )
            except Exception:
                pass
        if thread_handle is not None:
            try:
                win32api.CloseHandle(thread_handle)
            except Exception:
                pass
        if process_handle is not None:
            try:
                win32api.CloseHandle(process_handle)
            except Exception:
                pass
