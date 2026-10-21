#!/usr/bin/env python3
"""Inject one DLL into one Windows process using LoadLibraryW."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def inject(pid: int, dll_path: Path, timeout_seconds: float) -> dict[str, object]:
    import win32api  # type: ignore[import-not-found]
    import win32con  # type: ignore[import-not-found]
    import win32event  # type: ignore[import-not-found]
    import win32process  # type: ignore[import-not-found]

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
        thread_handle = created[0] if isinstance(created, tuple) else created
        wait_result = win32event.WaitForSingleObject(
            thread_handle,
            max(1, int(timeout_seconds * 1000.0)),
        )
        if wait_result == getattr(win32event, "WAIT_TIMEOUT", 0x00000102):
            return {"injected": False, "message": "LoadLibraryW timed out"}
        if wait_result != getattr(win32event, "WAIT_OBJECT_0", 0):
            return {"injected": False, "message": f"LoadLibraryW wait failed: {wait_result}"}
        exit_code = int(win32process.GetExitCodeThread(thread_handle))
        return {
            "injected": exit_code != 0,
            "message": "injected" if exit_code != 0 else "LoadLibraryW returned NULL",
            "load_library_return": exit_code,
        }
    finally:
        if remote_mem is not None and process_handle is not None:
            try:
                win32process.VirtualFreeEx(process_handle, remote_mem, 0, win32con.MEM_RELEASE)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    dll = args.dll
    if not dll.is_file():
        print(json.dumps({"injected": False, "message": f"DLL not found: {dll}"}))
        return 2
    result = inject(args.pid, dll, args.timeout)
    result.update({"pid": args.pid, "dll": str(dll)})
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("injected") else 1


if __name__ == "__main__":
    raise SystemExit(main())
