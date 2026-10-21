from __future__ import annotations

import json

from scripts.capture_cyberpunk_harness import (
    KNOWN_3080TI_CYBERPUNK_EXE,
    StepResult,
    build_readiness_report,
    build_runtime_env,
    candidate_cyberpunk_paths,
    credential_preflight,
    generate_installer_inputs,
    health_url,
    verify_worker_upload_credentials,
    worker_upload_url,
)
import scripts.capture_cyberpunk_harness as harness
from scripts.capture_simulator import UploadEvent, build_simulation_report, ingest_url


def test_ingest_url_accepts_base_or_endpoint() -> None:
    assert ingest_url("http://127.0.0.1:18080") == "http://127.0.0.1:18080/ingest"
    assert ingest_url("http://127.0.0.1:18080/ingest") == "http://127.0.0.1:18080/ingest"
    assert health_url("http://127.0.0.1:18080/ingest") == "http://127.0.0.1:18080/healthz"


def test_simulation_report_shape_counts_accepts_and_dedup() -> None:
    events = [
        UploadEvent(
            frame_uuid="11111111-1111-4111-8111-111111111111",
            status_code=200,
            duration_ms=12.3456,
            terminal=True,
            retryable=False,
            response={
                "status": "ok",
                "exr_key": "cyberpunk-2077/2026-05/lite/session/frame.exr",
            },
        ),
        UploadEvent(
            frame_uuid="22222222-2222-4222-8222-222222222222",
            status_code=409,
            duration_ms=4.0,
            terminal=True,
            retryable=False,
            response={"detail": "duplicate frame (content hash already seen)"},
        ),
    ]

    report = build_simulation_report(
        game="cyberpunk-2077",
        frames_requested=2,
        mode="lite",
        target_url="http://127.0.0.1:18080/ingest",
        frames_sent=2,
        events=events,
        elapsed_ms=20.25,
    )

    assert report["frames_sent"] == 2
    assert report["accepts"] == 1
    assert report["dedup_hits"] == 1
    assert report["server_timings"]["total_ms"] == 20.25
    assert report["server_timings"]["requests"][0]["duration_ms"] == 12.346
    assert report["accepted_exr_keys"] == [
        "cyberpunk-2077/2026-05/lite/session/frame.exr"
    ]


def test_cyberpunk_candidates_prefer_explicit_and_env() -> None:
    candidates = candidate_cyberpunk_paths(
        explicit=r"D:\Games\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe",
        env={
            "CYBERPUNK_EXE": r"E:\Steam\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe",
            "ProgramFiles(x86)": r"C:\Program Files (x86)",
            "ProgramFiles": r"C:\Program Files",
        },
        system="Windows",
    )

    assert str(candidates[0]) == r"D:\Games\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe"
    assert str(candidates[1]) == r"E:\Steam\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe"
    assert str(candidates[2]) == KNOWN_3080TI_CYBERPUNK_EXE
    assert any("Steam" in str(candidate) for candidate in candidates)


def test_generate_cyberpunk_installer_inputs_redacts_report_token(tmp_path) -> None:
    step = generate_installer_inputs(
        tmp_path,
        server="http://127.0.0.1:8080",
        token="abc123token",
        mode="lite",
        installer_version="1.2.3",
    )

    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "installer_manifest.json").read_text(encoding="utf-8"))

    assert step.status == "ready"
    assert cfg["game_id"] == "cyberpunk-2077"
    assert cfg["game_exe_name"] == "Cyberpunk2077.exe"
    assert cfg["install_token"] == "abc123token"
    assert cfg["ingest_url"] == "http://127.0.0.1:8080/ingest"
    assert step.details["install_token"] == {
        "present": True,
        "length": 11,
        "preview": "...oken",
    }
    assert manifest["verify"]["expected_game_exe"] == "Cyberpunk2077.exe"
    assert any(
        file["src"] == "dxgi.dll" and file["dst_name"] == "dxgi.dll"
        for file in manifest["files"]
    )
    assert any(
        file["src"] == "nvngx_dlss.dll" and file["dst_name"] == "nvngx_dlss.dll"
        for file in manifest["files"]
    )
    assert (tmp_path / "scheduled_task.xml").is_file()
    assert (tmp_path / "oss_capture_installer.wxs").is_file()


def test_readiness_report_allows_skipped_synthetic_when_no_token(tmp_path) -> None:
    report = build_readiness_report(
        steps=[
            StepResult("verify_ingest", "ready"),
            StepResult("generate_installer_config", "ready"),
            StepResult("synthetic_capture", "skipped"),
        ],
        server="http://127.0.0.1:8080",
        output_dir=tmp_path,
        token=None,
    )

    assert report["ready"] is True
    assert report["ready_paths"]["local_ingest"] is True
    assert report["token"] == {"present": False}


def test_readiness_report_fails_on_installer_smoke_error(tmp_path) -> None:
    report = build_readiness_report(
        steps=[
            StepResult("verify_ingest", "ready"),
            StepResult("generate_installer_config", "ready"),
            StepResult("installer_fake_cyberpunk_smoke", "error"),
            StepResult("synthetic_capture", "skipped"),
        ],
        server="http://127.0.0.1:8080",
        output_dir=tmp_path,
        token=None,
    )

    assert report["ready"] is False


def test_runtime_env_loads_worker_credentials_and_derives_r2_endpoint(tmp_path, monkeypatch) -> None:
    creds = tmp_path / "r2-credentials.env"
    creds.write_text(
        "\n".join(
            [
                "CF_ACCOUNT_ID=abc123",
                "R2_BUCKET=ors-captures",
                "WORKER_UPLOAD_URL=https://upload.example.test",
                "WORKER_SHARED_SECRET=worker-secret",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("R2_ENDPOINT", raising=False)
    monkeypatch.delenv("WORKER_UPLOAD_URL", raising=False)
    monkeypatch.delenv("WORKER_SHARED_SECRET", raising=False)

    env = build_runtime_env(base_env={}, credentials_env=creds)
    preflight = credential_preflight(env=env, credentials_env=creds)

    assert env["R2_ENDPOINT"] == "https://abc123.r2.cloudflarestorage.com"
    assert preflight.status == "worker_upload"
    assert preflight.details["worker_upload"]["ready"] is True
    assert preflight.details["direct_r2"]["ready"] is False
    assert "worker-secret" not in json.dumps(preflight.to_json())


def test_worker_upload_smoke_redacts_secret_and_uses_allowlisted_test_key(monkeypatch) -> None:
    seen = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _limit):
            return b'{"ok":true,"key":"test/capture-ingest-smoke.json"}'

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.headers.get("Authorization")
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(harness.request, "urlopen", fake_urlopen)
    step = verify_worker_upload_credentials(
        env={
            "WORKER_UPLOAD_URL": "https://upload.example.test",
            "WORKER_SHARED_SECRET": "super-secret",
        },
        timeout=3.0,
        smoke_key="test/capture-ingest-smoke.json",
    )

    assert step.status == "ready"
    assert seen["url"] == "https://upload.example.test/upload/test/capture-ingest-smoke.json"
    assert seen["auth"] == "Bearer super-secret"
    assert seen["timeout"] == 3.0
    assert step.details["shared_secret"] == {"present": True}
    assert "super-secret" not in json.dumps(step.to_json())


def test_worker_upload_url_preserves_base_path() -> None:
    assert (
        worker_upload_url("https://upload.example.test/base", "test/smoke.json")
        == "https://upload.example.test/base/upload/test/smoke.json"
    )


def test_readiness_report_accepts_verified_worker_upload_path(tmp_path) -> None:
    report = build_readiness_report(
        steps=[
            StepResult("credential_preflight", "worker_upload"),
            StepResult("worker_upload_credentials", "ready"),
            StepResult("verify_ingest", "unavailable"),
        ],
        server="http://127.0.0.1:8080",
        output_dir=tmp_path,
        token=None,
    )

    assert report["ready"] is True
    assert report["ready_paths"] == {"local_ingest": False, "worker_upload": True}
