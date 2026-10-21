"""Tests for tray game detection, config writing, and DLL injection glue."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from oss.capture.tray import allowlist
from oss.capture.tray import app as tray_app
from oss.capture.tray import config as cfg_mod
from oss.capture.tray import dll_inject
from oss.capture.tray import process_watcher
from oss.capture.tray import session_config
from oss.capture.tray import steam_library


def test_allowlist_lookup_by_app_id_and_exe() -> None:
    game = allowlist.lookup_by_app_id("1091500")
    assert game is not None
    assert game.game_id == "cyberpunk-2077"
    assert game.display_name == "Cyberpunk 2077"
    assert game.exe_basename == "Cyberpunk2077.exe"
    assert game.supports_dlss is True
    assert allowlist.lookup_by_exe("cyberpunk2077.exe") == game
    assert allowlist.lookup_by_exe("not-supported.exe") is None


def test_steam_library_parses_manifest(tmp_path: Path) -> None:
    steam_root = tmp_path / "Steam"
    library = tmp_path / "Library"
    steamapps = library / "steamapps"
    install_dir = steamapps / "common" / "Cyberpunk 2077"
    (steam_root / "steamapps").mkdir(parents=True)
    install_dir.mkdir(parents=True)
    (steam_root / "steamapps" / "libraryfolders.vdf").write_text(
        '"libraryfolders"\n{\n  "1"\n  {\n    "path"\t"'
        + str(library).replace("\\", "\\\\")
        + '"\n  }\n}\n',
        encoding="utf-8",
    )
    (steamapps / "appmanifest_1091500.acf").write_text(
        '"AppState"\n{\n'
        '  "appid" "1091500"\n'
        '  "name" "Cyberpunk 2077"\n'
        '  "installdir" "Cyberpunk 2077"\n'
        '}\n',
        encoding="utf-8",
    )

    games = steam_library.all_installed_games(steam_root)
    assert len(games) == 1
    assert games[0].app_id == "1091500"
    assert games[0].install_dir == install_dir


def test_process_watcher_launch_block_and_relaunch() -> None:
    launched = []
    blocked = []
    watcher = process_watcher.ProcessWatcher(launched.append, blocked.append)
    watcher._enumerate = lambda: {100: "Cyberpunk2077.exe".lower()}

    watcher._tick()
    watcher._tick()

    assert len(launched) == 1
    assert launched[0].pid == 100
    assert launched[0].allowed.game_id == "cyberpunk-2077"

    watcher._enumerate = lambda: {}
    watcher._tick()
    watcher._enumerate = lambda: {
        101: "Cyberpunk2077.exe".lower(),
        7: "EasyAntiCheat.exe".lower(),
    }
    watcher._tick()

    assert len(launched) == 1
    assert len(blocked) == 1
    assert blocked[0].pid == 101
    assert blocked[0].blocking_processes == ("easyanticheat.exe",)


def test_session_config_writes_cyberpunk_session_payload(tmp_path: Path) -> None:
    game = allowlist.lookup_by_app_id("1091500")
    assert game is not None
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"install_token": "installer-token"}), encoding="utf-8")
    session_uuid = "11111111-2222-3333-4444-555555555555"

    written = session_config.write_session_config(
        game=game,
        capture_mode="regular",
        output_dir=tmp_path / "captures" / game.game_id,
        capture_api_base="https://capture.example.test/",
        capture_storage_mode="upload",
        session_uuid=session_uuid,
        path=path,
    )

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["game_id"] == "cyberpunk-2077"
    assert payload["game_exe_name"] == "Cyberpunk2077.exe"
    assert payload["session_uuid"] == session_uuid
    assert payload["capture_mode"] == "regular"
    assert payload["capture_storage_mode"] == "upload"
    assert payload["capture_api_base"] == "https://capture.example.test"
    assert payload["output_root_dir"].endswith("captures/upload")
    assert payload["game_output_dir"].endswith("captures/upload/cyberpunk-2077")
    assert payload["session_output_dir"].endswith(
        "captures/upload/cyberpunk-2077/11111111-2222-3333-4444-555555555555"
    )
    assert payload["output_dir"] == payload["session_output_dir"]
    assert payload["pending_dir"] == payload["output_root_dir"]
    assert payload["ingest_url"] == "https://capture.example.test/ingest"
    assert payload["endpoints"]["ingest"] == "https://capture.example.test/ingest"
    assert payload["consent"]["mode"] == "regular"
    assert payload["install_token"] == "installer-token"
    assert Path(payload["session_output_dir"]).is_dir()


def test_session_config_upload_requires_installer_token(tmp_path: Path) -> None:
    game = allowlist.lookup_by_app_id("1091500")
    assert game is not None

    try:
        session_config.write_session_config(
            game=game,
            capture_mode="lite",
            output_dir=tmp_path / "captures" / game.game_id,
            capture_storage_mode="upload",
            path=tmp_path / "config.json",
        )
    except ValueError as exc:
        assert "install_token" in str(exc)
    else:
        raise AssertionError("expected upload mode without installer token to fail")


def test_session_config_preserves_installer_uploader_fields(tmp_path: Path) -> None:
    game = allowlist.lookup_by_app_id("1091500")
    assert game is not None
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "install_token": "installer-token",
                "pending_dir_cap_bytes": 123456789,
                "uploader_retry_attempts": 7,
                "capture_api_base": "http://127.0.0.1:18080",
                "capture_storage_mode": "local",
                "proxy_dll_name": "dxgi.dll",
                "consent_disclosure": "operator-only test",
                "future_compatible_knob": {"keep": True},
            }
        ),
        encoding="utf-8",
    )

    session_config.write_session_config(
        game=game,
        capture_mode="lite",
        output_dir=tmp_path / "captures" / game.game_id,
        session_uuid="11111111-2222-3333-4444-555555555555",
        path=path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["install_token"] == "installer-token"
    assert payload["pending_dir_cap_bytes"] == 123456789
    assert payload["uploader_retry_attempts"] == 7
    assert payload["capture_api_base"] == "http://127.0.0.1:18080"
    assert payload["capture_storage_mode"] == "local"
    assert payload["ingest_url"] == "http://127.0.0.1:18080/ingest"
    assert payload["proxy_dll_name"] == "dxgi.dll"
    assert payload["consent_disclosure"] == "operator-only test"
    assert payload["future_compatible_knob"] == {"keep": True}
    assert payload["pending_dir"].endswith("captures/local")


def test_session_config_rejects_bad_session_uuid(tmp_path: Path) -> None:
    game = allowlist.lookup_by_app_id("1091500")
    assert game is not None

    try:
        session_config.write_session_config(
            game=game,
            capture_mode="lite",
            output_dir=tmp_path / "captures" / game.game_id,
            session_uuid="not-a-uuid",
            path=tmp_path / "config.json",
        )
    except ValueError as exc:
        assert "session_uuid" in str(exc)
    else:
        raise AssertionError("expected bad session UUID to fail")


def test_dll_inject_non_windows_is_no_op(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dll_inject.platform, "system", lambda: "Darwin")
    result = dll_inject.inject_dll(1234, tmp_path / "missing.dll")
    assert result.skipped is True
    assert result.injected is False


def test_dll_inject_dry_run_skips_before_windows_checks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dll_inject.platform, "system", lambda: "Windows")
    result = dll_inject.inject_dll(1234, tmp_path / "missing.dll", dry_run=True)
    assert result.skipped is True
    assert result.dry_run is True
    assert result.injected is False


def test_dll_inject_windows_uses_loadlibrary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dll_inject.platform, "system", lambda: "Windows")
    dll = tmp_path / "oss_capture.dll"
    dll.write_bytes(b"dll")
    calls = []

    fake_api = types.SimpleNamespace(
        OpenProcess=lambda access, inherit, pid: calls.append(("OpenProcess", access, inherit, pid)) or "process",
        GetModuleHandle=lambda name: calls.append(("GetModuleHandle", name)) or "kernel32",
        GetProcAddress=lambda handle, name: calls.append(("GetProcAddress", handle, name)) or 0x1234,
        CloseHandle=lambda handle: calls.append(("CloseHandle", handle)),
    )
    fake_con = types.SimpleNamespace(
        PROCESS_CREATE_THREAD=0x0002,
        PROCESS_QUERY_INFORMATION=0x0400,
        PROCESS_VM_OPERATION=0x0008,
        PROCESS_VM_WRITE=0x0020,
        PROCESS_VM_READ=0x0010,
        MEM_RESERVE=0x2000,
        MEM_COMMIT=0x1000,
        MEM_RELEASE=0x8000,
        PAGE_READWRITE=0x04,
    )
    fake_event = types.SimpleNamespace(
        INFINITE=0xFFFFFFFF,
        WAIT_OBJECT_0=0,
        WAIT_TIMEOUT=0x00000102,
        WaitForSingleObject=lambda handle, timeout: calls.append(("WaitForSingleObject", handle, timeout)) or 0,
    )
    fake_process = types.SimpleNamespace(
        VirtualAllocEx=lambda process, addr, size, flags, protect: calls.append(
            ("VirtualAllocEx", process, addr, size, flags, protect)
        ) or 0xCAFE,
        WriteProcessMemory=lambda process, addr, data: calls.append(
            ("WriteProcessMemory", process, addr, data)
        ),
        CreateRemoteThread=lambda process, sec, stack, start, param, flags: calls.append(
            ("CreateRemoteThread", process, sec, stack, start, param, flags)
        ) or ("thread", 55),
        GetExitCodeThread=lambda thread: calls.append(("GetExitCodeThread", thread)) or 1,
        VirtualFreeEx=lambda process, addr, size, free_type: calls.append(
            ("VirtualFreeEx", process, addr, size, free_type)
        ),
    )
    monkeypatch.setitem(sys.modules, "win32api", fake_api)
    monkeypatch.setitem(sys.modules, "win32con", fake_con)
    monkeypatch.setitem(sys.modules, "win32event", fake_event)
    monkeypatch.setitem(sys.modules, "win32process", fake_process)

    result = dll_inject.inject_dll(4242, dll)

    assert result.injected is True
    assert ("GetProcAddress", "kernel32", "LoadLibraryW") in calls
    assert any(call[0] == "CreateRemoteThread" and call[4] == 0x1234 for call in calls)
    assert ("WaitForSingleObject", "thread", 10_000) in calls


def test_dll_inject_windows_times_out_remote_thread(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dll_inject.platform, "system", lambda: "Windows")
    dll = tmp_path / "oss_capture.dll"
    dll.write_bytes(b"dll")
    calls = []

    fake_api = types.SimpleNamespace(
        OpenProcess=lambda access, inherit, pid: "process",
        GetModuleHandle=lambda name: "kernel32",
        GetProcAddress=lambda handle, name: 0x1234,
        CloseHandle=lambda handle: calls.append(("CloseHandle", handle)),
    )
    fake_con = types.SimpleNamespace(
        PROCESS_CREATE_THREAD=0x0002,
        PROCESS_QUERY_INFORMATION=0x0400,
        PROCESS_VM_OPERATION=0x0008,
        PROCESS_VM_WRITE=0x0020,
        PROCESS_VM_READ=0x0010,
        MEM_RESERVE=0x2000,
        MEM_COMMIT=0x1000,
        MEM_RELEASE=0x8000,
        PAGE_READWRITE=0x04,
    )
    fake_event = types.SimpleNamespace(
        WAIT_OBJECT_0=0,
        WAIT_TIMEOUT=0x00000102,
        WaitForSingleObject=lambda handle, timeout: calls.append(("WaitForSingleObject", handle, timeout)) or 0x00000102,
    )
    fake_process = types.SimpleNamespace(
        VirtualAllocEx=lambda process, addr, size, flags, protect: 0xCAFE,
        WriteProcessMemory=lambda process, addr, data: None,
        CreateRemoteThread=lambda process, sec, stack, start, param, flags: ("thread", 55),
        GetExitCodeThread=lambda thread: 1,
        VirtualFreeEx=lambda process, addr, size, free_type: calls.append(("VirtualFreeEx", process, addr, size, free_type)),
    )
    monkeypatch.setitem(sys.modules, "win32api", fake_api)
    monkeypatch.setitem(sys.modules, "win32con", fake_con)
    monkeypatch.setitem(sys.modules, "win32event", fake_event)
    monkeypatch.setitem(sys.modules, "win32process", fake_process)

    result = dll_inject.inject_dll(4242, dll, timeout_seconds=0.25)

    assert result.injected is False
    assert "timed out" in result.message
    assert ("WaitForSingleObject", "thread", 250) in calls


def test_tray_game_launch_writes_config_and_injects(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setattr(cfg_mod, "_config_dir", lambda: config_dir)
    game = allowlist.lookup_by_app_id("1091500")
    assert game is not None
    cfg_mod.save(cfg_mod.TrayConfig(
        capture_mode="INSANE",
        capture_storage_mode="upload",
        output_drive_override=str(tmp_path / "drive"),
        enabled_games={game.game_id: True},
    ))
    local_config_dir = tmp_path / "local" / "oss-capture"
    local_config_dir.mkdir(parents=True)
    (local_config_dir / "config.json").write_text(
        json.dumps({"install_token": "installer-token"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(session_config, "_local_config_dir", lambda: local_config_dir)
    injected = []
    monkeypatch.setattr(
        dll_inject,
        "inject_dll",
        lambda pid, **_kwargs: injected.append((pid, _kwargs)) or dll_inject.InjectionResult(
            pid=pid,
            dll_path=tmp_path / "oss_capture.dll",
            injected=True,
        ),
    )

    app = tray_app.TrayApp()
    app._on_game_launch(process_watcher.GameLaunchEvent(
        pid=777,
        exe_basename=game.exe_basename,
        allowed=game,
    ))

    payload = json.loads((tmp_path / "local" / "oss-capture" / "config.json").read_text(encoding="utf-8"))
    assert payload["game_id"] == game.game_id
    assert payload["capture_mode"] == "INSANE"
    assert payload["capture_storage_mode"] == "upload"
    assert payload["install_token"] == "installer-token"
    assert payload["session_uuid"]
    assert payload["game_output_dir"].endswith("oss-captures/upload/cyberpunk-2077")
    assert payload["output_dir"].endswith(f"oss-captures/upload/cyberpunk-2077/{payload['session_uuid']}")
    assert injected == [(777, {"dry_run": False})]


def test_tray_game_launch_dry_run_writes_config_without_real_injection(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setattr(cfg_mod, "_config_dir", lambda: config_dir)
    game = allowlist.lookup_by_app_id("1091500")
    assert game is not None
    cfg_mod.save(cfg_mod.TrayConfig(
        output_drive_override=str(tmp_path / "drive"),
        enabled_games={game.game_id: True},
        dry_run_launch=True,
    ))
    monkeypatch.setattr(session_config, "_local_config_dir", lambda: tmp_path / "local" / "oss-capture")
    calls = []
    monkeypatch.setattr(
        dll_inject,
        "inject_dll",
        lambda pid, **kwargs: calls.append((pid, kwargs)) or dll_inject.InjectionResult(
            pid=pid,
            dll_path=tmp_path / "oss_capture.dll",
            injected=False,
            skipped=True,
            dry_run=kwargs["dry_run"],
        ),
    )

    app = tray_app.TrayApp()
    app._on_game_launch(process_watcher.GameLaunchEvent(
        pid=888,
        exe_basename=game.exe_basename,
        allowed=game,
    ))

    payload = json.loads((tmp_path / "local" / "oss-capture" / "config.json").read_text(encoding="utf-8"))
    assert payload["game_id"] == game.game_id
    assert calls == [(888, {"dry_run": True})]


def test_tray_game_launch_config_write_failure_skips_injection(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setattr(cfg_mod, "_config_dir", lambda: config_dir)
    game = allowlist.lookup_by_app_id("1091500")
    assert game is not None
    cfg_mod.save(cfg_mod.TrayConfig(
        output_drive_override=str(tmp_path / "drive"),
        enabled_games={game.game_id: True},
    ))
    monkeypatch.setattr(
        session_config,
        "write_session_config",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        dll_inject,
        "inject_dll",
        lambda _pid, **_kwargs: (_ for _ in ()).throw(AssertionError("should not inject")),
    )

    app = tray_app.TrayApp()
    app._on_game_launch(process_watcher.GameLaunchEvent(
        pid=999,
        exe_basename=game.exe_basename,
        allowed=game,
    ))


def test_tray_game_launch_disabled_skips_config_and_inject(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setattr(cfg_mod, "_config_dir", lambda: config_dir)
    game = allowlist.lookup_by_app_id("1091500")
    assert game is not None
    cfg_mod.save(cfg_mod.TrayConfig(enabled_games={game.game_id: False}))
    monkeypatch.setattr(
        session_config,
        "write_session_config",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not write")),
    )
    monkeypatch.setattr(
        dll_inject,
        "inject_dll",
        lambda _pid: (_ for _ in ()).throw(AssertionError("should not inject")),
    )

    app = tray_app.TrayApp()
    app._on_game_launch(process_watcher.GameLaunchEvent(
        pid=777,
        exe_basename=game.exe_basename,
        allowed=game,
    ))
