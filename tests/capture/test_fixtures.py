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
    burst_index: int = 0,
    lr_resolution: tuple[int, int] = (16, 9),
    scale: int = 2,
    payload_bytes: int | None = None,
    captured_at_unix: int | None = None,
) -> SyntheticCapture:
    session_uuid = session_uuid or str(uuid.uuid4())
    frame_uuid = frame_uuid or str(uuid.uuid4())
    burst_uuid = burst_uuid or str(uuid.uuid4())
    session_dir = pending_dir / game_id / session_uuid
    session_dir.mkdir(parents=True, exist_ok=True)

    frame_path = session_dir / f"{frame_uuid}.exr"
    meta_path = session_dir / f"{frame_uuid}.json"
    hr_resolution = (lr_resolution[0] * scale, lr_resolution[1] * scale)
    write_synthetic_exr(frame_path, lr_resolution=lr_resolution, hr_resolution=hr_resolution)
    if payload_bytes is not None:
        _resize_payload(frame_path, payload_bytes)

    metadata = make_metadata(
        game_id=game_id,
        game_version=game_version,
        session_uuid=session_uuid,
        frame_uuid=frame_uuid,
        burst_uuid=burst_uuid,
        burst_index=burst_index,
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
    burst_uuid: str,
    burst_index: int,
    lr_resolution: tuple[int, int] = (16, 9),
    hr_resolution: tuple[int, int] = (32, 18),
    captured_at_unix: int | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "game_id": game_id,
        "game_version": game_version,
        "session_uuid": session_uuid,
        "frame_uuid": frame_uuid,
        "burst_uuid": burst_uuid,
        "burst_index": int(burst_index),
        "captured_at_unix": int(captured_at_unix if captured_at_unix is not None else time.time()),
        "lr_resolution": [int(lr_resolution[0]), int(lr_resolution[1])],
        "hr_resolution": [int(hr_resolution[0]), int(hr_resolution[1])],
        "hr_source": "dlss-quality",
        "jitter_offset_uv": [0.25, 0.75],
        "motion_mean_magnitude_px": 4.5,
        "perceptual_hash_64": "0x0123456789abcdef",
        "user_consent_token": "synthetic-consent-token",
        "uploader_version": "1.0.0",
    }


def write_synthetic_exr(
    path: Path,
    *,
    lr_resolution: tuple[int, int] = (16, 9),
    hr_resolution: tuple[int, int] = (32, 18),
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
        "HR": hr_rgb.astype(np.float32),
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
    channel_names = {
        "LR": ["R", "G", "B"],
        "HR": ["R", "G", "B"],
        "Depth": ["Z"],
        "Motion": ["X", "Y"],
        "Normals": ["X", "Y", "Z"],
    }
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
