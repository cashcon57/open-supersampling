"""Tests for the R2 bucket key layout.

Verifies that ``frame_key`` produces the spec'd
``<game_id>/<YYYY-MM>/<session_uuid>/<frame_uuid>.{exr,json}`` layout
under realistic conditions, and that a synthetic ingest lands at the
expected key in the moto-backed bucket.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest


# ---- pure-function layout tests --------------------------------------------


def test_frame_key_basic():
    from server.oss_capture_ingest.r2 import frame_key

    # 2026-05-04 00:00:00 UTC = 1777881600
    ts = datetime(2026, 5, 4, tzinfo=timezone.utc).timestamp()
    key = frame_key(
        "cyberpunk-2077",
        ts,
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    )
    assert (
        key
        == "cyberpunk-2077/2026-05/11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222.exr"
    )


def test_frame_key_json_suffix():
    from server.oss_capture_ingest.r2 import frame_key

    ts = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    key = frame_key("game-x", ts, "s-uuid", "f-uuid", suffix=".json")
    assert key == "game-x/2025-12/s-uuid/f-uuid.json"


def test_frame_key_rejects_bad_suffix():
    from server.oss_capture_ingest.r2 import frame_key

    with pytest.raises(ValueError):
        frame_key("g", 0.0, "s", "f", suffix=".png")


# ---- mode-stratified path layout (post-C23) --------------------------------


def test_frame_key_with_capture_mode_inserts_segment():
    from server.oss_capture_ingest.r2 import frame_key

    ts = datetime(2026, 5, 4, tzinfo=timezone.utc).timestamp()
    key = frame_key(
        "cyberpunk-2077",
        ts,
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
        capture_mode="trickle",
    )
    assert (
        key
        == "cyberpunk-2077/2026-05/trickle/11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222.exr"
    )


def test_frame_key_capture_mode_none_is_legacy_layout():
    from server.oss_capture_ingest.r2 import frame_key

    ts = datetime(2026, 5, 4, tzinfo=timezone.utc).timestamp()
    legacy = frame_key("g", ts, "s", "f", capture_mode=None)
    moded = frame_key("g", ts, "s", "f", capture_mode="lite")
    assert "/lite/" in moded
    assert "/lite/" not in legacy


def test_frame_key_rejects_unknown_capture_mode():
    from server.oss_capture_ingest.r2 import frame_key

    ts = datetime(2026, 5, 4, tzinfo=timezone.utc).timestamp()
    with pytest.raises(ValueError):
        frame_key("g", ts, "s", "f", capture_mode="ULTRAINSANE")


def test_dedup_key_layout():
    from server.oss_capture_ingest.r2 import dedup_key

    h = "deadbeefcafebabe" * 4  # 64 hex
    assert dedup_key(h) == f"_dedup/de/{h}"
    # Case-normalized.
    assert dedup_key(h.upper()) == f"_dedup/de/{h}"


def test_ingest_with_trickle_lands_under_trickle_segment(
    client, r2_client, make_meta_fn, post_ingest_fn, reset_state
):
    registry, _ = reset_state
    registry.register_token("trickle-token", label="trickle-test")

    session_uuid = "cccccccc-dddd-eeee-ffff-000000000000"
    frame_uuid = "11111111-2222-3333-4444-555555555555"
    captured_at = datetime(2026, 4, 1, tzinfo=timezone.utc).timestamp()

    meta = make_meta_fn(
        game_id="cyberpunk-2077",
        session_uuid=session_uuid,
        frame_uuid=frame_uuid,
        captured_at_unix=captured_at,
        capture_mode="trickle",  # static single — no burst fields
    )
    body = b"TRICKLE-EXR" * 200

    r = post_ingest_fn(client, token="trickle-token", frame_body=body, meta=meta)
    assert r.status_code == 200, r.text
    assert r.json()["exr_key"] == (
        f"cyberpunk-2077/2026-04/trickle/{session_uuid}/{frame_uuid}.exr"
    )


def test_month_partition_boundary():
    from server.oss_capture_ingest.r2 import month_partition

    # Last second of January UTC stays in 01.
    ts = datetime(2026, 1, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    assert month_partition(ts) == "2026-01"
    # First second of February crosses to 02.
    ts2 = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    assert month_partition(ts2) == "2026-02"


# ---- end-to-end via client + moto ------------------------------------------


def test_ingest_lands_at_expected_key(
    client, r2_client, make_meta_fn, post_ingest_fn, reset_state
):
    registry, _ = reset_state
    registry.register_token("layout-token", label="layout-test")

    session_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    frame_uuid = "ffffffff-0000-1111-2222-333333333333"
    captured_at = datetime(2026, 3, 15, 12, tzinfo=timezone.utc).timestamp()

    meta = make_meta_fn(
        game_id="bg3",
        session_uuid=session_uuid,
        frame_uuid=frame_uuid,
        captured_at_unix=captured_at,
    )
    body = b"BG3-EXR-MOCK" * 1000

    r = post_ingest_fn(client, token="layout-token", frame_body=body, meta=meta)
    assert r.status_code == 200, r.text
    # Layout includes capture_mode segment (defaults to "lite" when meta
    # omits it — the conftest's make_meta() does).
    expected_exr = f"bg3/2026-03/lite/{session_uuid}/{frame_uuid}.exr"
    expected_json = f"bg3/2026-03/lite/{session_uuid}/{frame_uuid}.json"
    assert r.json()["exr_key"] == expected_exr
    assert r.json()["json_key"] == expected_json

    # Both objects exist in the bucket and round-trip correctly.
    assert r2_client.get_bytes(expected_exr) == body
    sidecar = json.loads(r2_client.get_bytes(expected_json))
    assert sidecar["game_id"] == "bg3"
    assert sidecar["session_uuid"] == session_uuid
    assert sidecar["frame_uuid"] == frame_uuid


# ---- installer config builder ----------------------------------------------


def test_build_installer_config_pure_function():
    from scripts.build_capture_installer import build_config

    cfg = build_config(
        game_id="cyberpunk-2077",
        game_exe_name="Cyberpunk2077.exe",
        proxy_dll_name="dxgi.dll",
        installer_version="1.0.0",
        capture_api_base="https://capture.oss-supersampling.dev",
    )
    assert cfg["schema_version"] == 1
    assert cfg["game_id"] == "cyberpunk-2077"
    assert cfg["proxy_dll_name"] == "dxgi.dll"
    assert cfg["capture_mode"] == "lite"
    assert len(cfg["install_token"]) == 32  # uuid4 hex
    assert cfg["endpoints"]["ingest"].endswith("/ingest")
    assert cfg["consent"]["mode"] == "lite"
    assert cfg["consent"]["insane_supersample_gt_disclosure"] == ""


@pytest.mark.parametrize("mode", ["trickle", "lite", "regular", "INSANE"])
def test_build_installer_config_accepts_each_mode(mode):
    from scripts.build_capture_installer import (
        INSANE_SUPERSAMPLE_GT_CONSENT,
        build_config,
    )

    cfg = build_config(
        game_id="cyberpunk-2077",
        game_exe_name="Cyberpunk2077.exe",
        proxy_dll_name="dxgi.dll",
        installer_version="1.0.0",
        capture_api_base="https://capture.oss-supersampling.dev",
        capture_mode=mode,
    )

    assert cfg["capture_mode"] == mode
    assert cfg["consent"]["mode"] == mode
    if mode == "INSANE":
        assert cfg["consent"]["insane_supersample_gt_disclosure"] == INSANE_SUPERSAMPLE_GT_CONSENT
        assert "256-frame supersample ground-truth pass" in cfg["consent"]["insane_supersample_gt_disclosure"]
    else:
        assert cfg["consent"]["insane_supersample_gt_disclosure"] == ""


def test_build_installer_config_rejects_unknown_mode():
    from scripts.build_capture_installer import build_config

    with pytest.raises(ValueError, match="unknown capture_mode"):
        build_config(
            game_id="cyberpunk-2077",
            game_exe_name="Cyberpunk2077.exe",
            proxy_dll_name="dxgi.dll",
            installer_version="1.0.0",
            capture_api_base="https://capture.oss-supersampling.dev",
            capture_mode="FOO",
        )


def test_build_installer_config_rejects_bad_inputs():
    from scripts.build_capture_installer import build_config

    with pytest.raises(ValueError):
        build_config(
            game_id="UPPER",
            game_exe_name="x.exe",
            proxy_dll_name="x.dll",
            installer_version="1.0.0",
            capture_api_base="https://x",
        )
    with pytest.raises(ValueError):
        build_config(
            game_id="ok-id",
            game_exe_name="not-an-exe",
            proxy_dll_name="x.dll",
            installer_version="1.0.0",
            capture_api_base="https://x",
        )
    with pytest.raises(ValueError):
        build_config(
            game_id="ok-id",
            game_exe_name="x.exe",
            proxy_dll_name="x.notdll",
            installer_version="1.0.0",
            capture_api_base="https://x",
        )
    with pytest.raises(ValueError):
        build_config(
            game_id="ok-id",
            game_exe_name="x.exe",
            proxy_dll_name="x.dll",
            installer_version="not-semver",
            capture_api_base="https://x",
        )


def test_build_installer_writes_files(tmp_path):
    from scripts.build_capture_installer import main as installer_main

    out = tmp_path / "out"
    rc = installer_main(
        [
            "--game",
            "cyberpunk-2077",
            "--game-exe-name",
            "Cyberpunk2077.exe",
            "--proxy-dll-name",
            "dxgi.dll",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    cfg = json.loads((out / "config.json").read_text())
    manifest = json.loads((out / "installer_manifest.json").read_text())
    assert cfg["game_id"] == "cyberpunk-2077"
    assert cfg["install_token"]
    assert manifest["scheduled_task"]["name"] == "OSS-Capture-Uploader-cyberpunk-2077"
    assert any(
        f["src"] == "oss_capture.dll" for f in manifest["files"]
    )


def test_build_installer_writes_files_with_explicit_mode(tmp_path):
    from scripts.build_capture_installer import main as installer_main

    out = tmp_path / "out"
    rc = installer_main(
        [
            "--game",
            "cyberpunk-2077",
            "--game-exe-name",
            "Cyberpunk2077.exe",
            "--proxy-dll-name",
            "dxgi.dll",
            "--mode",
            "trickle",
            "--output",
            str(out),
        ]
    )

    assert rc == 0
    cfg = json.loads((out / "config.json").read_text())
    manifest = json.loads((out / "installer_manifest.json").read_text())
    assert cfg["capture_mode"] == "trickle"
    assert cfg["consent"]["mode"] == "trickle"
    assert manifest["consent"]["mode"] == "trickle"
