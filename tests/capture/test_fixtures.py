from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CAPTURE_CHANNELS = (
    "LR.R",
    "LR.G",
    "LR.B",
    "HR.R",
    "HR.G",
    "HR.B",
    "Depth.Z",
    "Motion.X",
    "Motion.Y",
    "Normals.X",
    "Normals.Y",
    "Normals.Z",
)
LONG_CAPTURE_CHANNELS = tuple(channel for channel in CAPTURE_CHANNELS if not channel.startswith("HR."))
REGULAR_CAPTURE_CHANNELS = CAPTURE_CHANNELS + (
    "Albedo.R",
    "Albedo.G",
    "Albedo.B",
    "Roughness.R",
)
INSANE_CAPTURE_CHANNELS = REGULAR_CAPTURE_CHANNELS + (
    "Metallic.R",
    "Emissive.R",
    "Emissive.G",
    "Emissive.B",
)


@dataclass(frozen=True)
class SyntheticCapture:
    frame_path: Path
    meta_path: Path
    metadata: dict


def make_synthetic_capture(
    pending_dir: Path,
    *,
    game_id: str = "synthetic-game",
    game_version: str = "test",
    session_uuid: str | None = None,
    frame_uuid: str | None = None,
    burst_uuid: str | None = None,
    burst_index: int | None = 0,
    burst_tier: str | None = "short",
    capture_mode: str = "lite",
    include_hr: bool | None = None,
    include_burst: bool | None = None,
    lr_resolution: tuple[int, int] = (16, 9),
    scale: int = 2,
    payload_bytes: int | None = None,
    captured_at_unix: int | None = None,
) -> SyntheticCapture:
    session_uuid = session_uuid or str(uuid.uuid4())
    frame_uuid = frame_uuid or str(uuid.uuid4())
    if include_burst is None:
        include_burst = not (capture_mode in {"trickle", "lite", "regular", "INSANE"} and burst_tier is None)
    if include_burst:
        burst_uuid = burst_uuid or str(uuid.uuid4())
        burst_index = 0 if burst_index is None else int(burst_index)
    session_dir = pending_dir / game_id / session_uuid
    session_dir.mkdir(parents=True, exist_ok=True)

    frame_path = session_dir / f"{frame_uuid}.exr"
    meta_path = session_dir / f"{frame_uuid}.json"
    hr_resolution = (lr_resolution[0] * scale, lr_resolution[1] * scale)
    if include_hr is None:
        include_hr = not (
            burst_tier == "long" or (capture_mode == "trickle" and include_burst and burst_index == 1)
        )
    include_materials = capture_mode in {"regular", "INSANE"}
    include_insane_materials = capture_mode == "INSANE"
    write_synthetic_exr(
        frame_path,
        lr_resolution=lr_resolution,
        hr_resolution=hr_resolution,
        include_hr=include_hr,
        include_materials=include_materials,
        include_insane_materials=include_insane_materials,
    )
    if payload_bytes is not None:
        _resize_payload(frame_path, payload_bytes)

    metadata = make_metadata(
        game_id=game_id,
        game_version=game_version,
        session_uuid=session_uuid,
        frame_uuid=frame_uuid,
        burst_uuid=burst_uuid,
        burst_index=burst_index,
        burst_tier=burst_tier,
        capture_mode=capture_mode,
        include_hr=include_hr,
        include_burst=include_burst,
        lr_resolution=lr_resolution,
        hr_resolution=hr_resolution,
        captured_at_unix=captured_at_unix,
    )
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return SyntheticCapture(frame_path=frame_path, meta_path=meta_path, metadata=metadata)


def make_metadata(
    *,
    game_id: str,
    game_version: str,
    session_uuid: str,
    frame_uuid: str,
    burst_uuid: str | None,
    burst_index: int | None,
    burst_tier: str | None = "short",
    capture_mode: str = "lite",
    include_hr: bool = True,
    include_burst: bool = True,
    lr_resolution: tuple[int, int] = (16, 9),
    hr_resolution: tuple[int, int] = (32, 18),
    captured_at_unix: int | None = None,
) -> dict:
    meta = {
        "schema_version": 1,
        "game_id": game_id,
        "game_version": game_version,
        "session_uuid": session_uuid,
        "frame_uuid": frame_uuid,
        "captured_at_unix": int(captured_at_unix if captured_at_unix is not None else time.time()),
        "lr_resolution": [int(lr_resolution[0]), int(lr_resolution[1])],
        "hr_resolution": [int(hr_resolution[0]), int(hr_resolution[1])],
        "capture_mode": capture_mode,
        "hr_source": "dlss-quality" if include_hr else "none",
        "jitter_offset_uv": [0.25, 0.75],
        "motion_mean_magnitude_px": 4.5,
        "perceptual_hash_64": "0x0123456789abcdef",
        "user_consent_token": "synthetic-consent-token",
        "uploader_version": "1.0.0",
    }
    if include_burst:
        meta["burst_uuid"] = burst_uuid
        meta["burst_index"] = int(0 if burst_index is None else burst_index)
        meta["burst_tier"] = burst_tier
    return meta


def write_synthetic_exr(
    path: Path,
    *,
    lr_resolution: tuple[int, int] = (16, 9),
    hr_resolution: tuple[int, int] = (32, 18),
    include_hr: bool = True,
    include_materials: bool = False,
    include_insane_materials: bool = False,
) -> None:
    """Write deterministic channel data using the capture EXR channel names.

    EXR channels share one data window, so LR channels are nearest-upsampled to
    the HR canvas while metadata records the original LR resolution.
    """

    pyexr = _require_pyexr()
    hr_w, hr_h = hr_resolution
    lr_w, lr_h = lr_resolution
    x = np.linspace(0.0, 1.0, hr_w, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, hr_h, dtype=np.float32)[:, None]
    hr_rgb = np.stack(
        [
            np.broadcast_to(x, (hr_h, hr_w)),
            np.broadcast_to(y, (hr_h, hr_w)),
            np.full((hr_h, hr_w), 0.25, dtype=np.float32),
        ],
        axis=-1,
    )
    lr_small = hr_rgb[:: max(1, hr_h // lr_h), :: max(1, hr_w // lr_w), :][:lr_h, :lr_w]
    lr_rgb = np.repeat(np.repeat(lr_small, max(1, hr_h // lr_h), axis=0), max(1, hr_w // lr_w), axis=1)
    lr_rgb = lr_rgb[:hr_h, :hr_w, :]
    if lr_rgb.shape[:2] != (hr_h, hr_w):
        padded = np.zeros_like(hr_rgb)
        padded[: lr_rgb.shape[0], : lr_rgb.shape[1], :] = lr_rgb
        lr_rgb = padded

    channels = {
        "LR": lr_rgb.astype(np.float32),
        "Depth": (1.0 + 99.0 * y.repeat(hr_w, axis=1))[..., None].astype(np.float32),
        "Motion": np.stack(
            [
                np.full((hr_h, hr_w), 0.5, dtype=np.float32),
                np.full((hr_h, hr_w), -0.25, dtype=np.float32),
            ],
            axis=-1,
        ),
        "Normals": np.dstack(
            [
                np.zeros((hr_h, hr_w), dtype=np.float32),
                np.zeros((hr_h, hr_w), dtype=np.float32),
                np.ones((hr_h, hr_w), dtype=np.float32),
            ]
        ),
    }
    if include_hr:
        channels["HR"] = hr_rgb.astype(np.float32)
    if include_materials:
        channels["Albedo"] = np.dstack(
            [
                np.broadcast_to(x, (hr_h, hr_w)),
                np.full((hr_h, hr_w), 0.2, dtype=np.float32),
                np.broadcast_to(y, (hr_h, hr_w)),
            ]
        ).astype(np.float32)
        channels["Roughness"] = np.full((hr_h, hr_w, 1), 0.65, dtype=np.float32)
    if include_insane_materials:
        channels["Metallic"] = np.full((hr_h, hr_w, 1), 0.15, dtype=np.float32)
        channels["Emissive"] = np.dstack(
            [
                np.zeros((hr_h, hr_w), dtype=np.float32),
                np.full((hr_h, hr_w), 0.05, dtype=np.float32),
                np.full((hr_h, hr_w), 0.1, dtype=np.float32),
            ]
        )
    channel_names = {
        "LR": ["R", "G", "B"],
        "Depth": ["Z"],
        "Motion": ["X", "Y"],
        "Normals": ["X", "Y", "Z"],
    }
    if include_hr:
        channel_names["HR"] = ["R", "G", "B"]
    if include_materials:
        channel_names["Albedo"] = ["R", "G", "B"]
        channel_names["Roughness"] = ["R"]
    if include_insane_materials:
        channel_names["Metallic"] = ["R"]
        channel_names["Emissive"] = ["R", "G", "B"]
    pyexr.write(path, channels, channel_names=channel_names, compression=pyexr.ZIP_COMPRESSION, compression_level=5)


def _resize_payload(path: Path, size_bytes: int) -> None:
    with path.open("ab") as handle:
        handle.truncate(size_bytes)


def _require_pyexr():
    try:
        import pyexr
    except Exception as exc:  # pragma: no cover - dependency is in pyproject.
        raise RuntimeError("pyexr is required for capture EXR fixtures") from exc
    return pyexr


def test_synthetic_capture_fixture_matches_pending_layout_and_schema(tmp_path: Path) -> None:
    capture = make_synthetic_capture(
        tmp_path,
        game_id="cyberpunk-2077",
        session_uuid="session-a",
        frame_uuid="frame-a",
        captured_at_unix=1_777_940_000,
    )

    assert capture.frame_path == tmp_path / "cyberpunk-2077" / "session-a" / "frame-a.exr"
    assert capture.meta_path == capture.frame_path.with_suffix(".json")
    assert capture.frame_path.stat().st_size > 0

    metadata = json.loads(capture.meta_path.read_text(encoding="utf-8"))
    assert metadata == capture.metadata
    assert set(metadata) == {
        "schema_version",
        "game_id",
        "game_version",
        "session_uuid",
        "frame_uuid",
        "burst_uuid",
        "burst_index",
        "burst_tier",
        "capture_mode",
        "captured_at_unix",
        "lr_resolution",
        "hr_resolution",
        "hr_source",
        "jitter_offset_uv",
        "motion_mean_magnitude_px",
        "perceptual_hash_64",
        "user_consent_token",
        "uploader_version",
    }


def test_synthetic_exr_contains_capture_channels(tmp_path: Path) -> None:
    pyexr = _require_pyexr()
    capture = make_synthetic_capture(tmp_path)

    with pyexr.open(capture.frame_path) as exr:
        assert set(exr.channels) == set(CAPTURE_CHANNELS)


def test_long_synthetic_exr_omits_hr_channels_and_marks_metadata(tmp_path: Path) -> None:
    pyexr = _require_pyexr()
    capture = make_synthetic_capture(tmp_path, burst_tier="long", burst_index=12)

    metadata = json.loads(capture.meta_path.read_text(encoding="utf-8"))
    assert metadata["burst_tier"] == "long"
    assert metadata["burst_index"] == 12
    assert metadata["hr_source"] == "none"
    with pyexr.open(capture.frame_path) as exr:
        assert set(exr.channels) == set(LONG_CAPTURE_CHANNELS)


def test_trickle_static_single_has_no_burst_fields_and_keeps_gbuffers(tmp_path: Path) -> None:
    pyexr = _require_pyexr()
    capture = make_synthetic_capture(tmp_path, capture_mode="trickle", burst_tier=None)

    metadata = json.loads(capture.meta_path.read_text(encoding="utf-8"))
    assert metadata["capture_mode"] == "trickle"
    assert "burst_uuid" not in metadata
    assert "burst_index" not in metadata
    assert "burst_tier" not in metadata
    assert metadata["hr_source"] == "dlss-quality"
    with pyexr.open(capture.frame_path) as exr:
        assert {"Depth.Z", "Motion.X", "Motion.Y", "Normals.X", "Normals.Y", "Normals.Z"} <= set(exr.channels)


def test_trickle_pair_tplus_omits_hr_but_keeps_gbuffers(tmp_path: Path) -> None:
    pyexr = _require_pyexr()
    capture = make_synthetic_capture(
        tmp_path,
        capture_mode="trickle",
        burst_uuid="33333333-3333-4333-8333-333333333333",
        burst_index=1,
        burst_tier="short",
    )

    metadata = json.loads(capture.meta_path.read_text(encoding="utf-8"))
    assert metadata["capture_mode"] == "trickle"
    assert metadata["burst_tier"] == "short"
    assert metadata["burst_index"] == 1
    assert metadata["hr_source"] == "none"
    with pyexr.open(capture.frame_path) as exr:
        channels = set(exr.channels)
    assert "HR.R" not in channels
    assert {"LR.R", "Depth.Z", "Motion.X", "Motion.Y", "Normals.X", "Normals.Y", "Normals.Z"} <= channels


def test_insane_synthetic_exr_contains_full_brdf_channels(tmp_path: Path) -> None:
    pyexr = _require_pyexr()
    capture = make_synthetic_capture(tmp_path, capture_mode="INSANE")

    metadata = json.loads(capture.meta_path.read_text(encoding="utf-8"))
    assert metadata["capture_mode"] == "INSANE"
    with pyexr.open(capture.frame_path) as exr:
        assert set(exr.channels) == set(INSANE_CAPTURE_CHANNELS)
