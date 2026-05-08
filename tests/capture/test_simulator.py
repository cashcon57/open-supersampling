from __future__ import annotations

from scripts.capture_simulator import UploadEvent, build_simulation_report, ingest_url


def test_ingest_url_accepts_base_or_endpoint() -> None:
    assert ingest_url("http://127.0.0.1:18080") == "http://127.0.0.1:18080/ingest"
    assert ingest_url("http://127.0.0.1:18080/ingest") == "http://127.0.0.1:18080/ingest"


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
