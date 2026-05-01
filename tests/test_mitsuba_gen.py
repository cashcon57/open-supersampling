"""Tests for oss.data.mitsuba_gen procedural dataset pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _mitsuba_available() -> bool:
    try:
        import mitsuba  # noqa: F401
        return True
    except ImportError:
        return False


_REQUIRES_MITSUBA = pytest.mark.skipif(
    not _mitsuba_available(),
    reason="requires mitsuba",
)


# ---------------------------------------------------------------------------
# RGBE roundtrip (no mitsuba needed)
# ---------------------------------------------------------------------------

def test_rgbe_roundtrip() -> None:
    from oss.data.mitsuba_gen.render_worker import encode_rgbe
    from oss.data.noisebase import _decompress_rgbe

    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.0, 10.0, size=(32, 32, 3)).astype(np.float32)

    rgbe, exp_pair = encode_rgbe(rgb)
    assert rgbe.shape == (32, 32, 4)
    assert rgbe.dtype == np.uint8
    assert exp_pair.dtype == np.float32

    color_5d = rgbe.transpose(2, 0, 1)[None, ..., None]
    exposure_2d = exp_pair[None, :]
    decoded = _decompress_rgbe(color_5d, exposure_2d)

    decoded_hw3 = decoded[0, :, :, :, 0].transpose(1, 2, 0)

    # NoiseBase RGBE uses a single per-pixel exponent (not per-channel), so
    # channels more than ~8 stops below the pixel's dominant channel fall
    # below the 1/255 mantissa floor and decode to zero by design. We only
    # check relative error for channels that are representable: those above
    # 1/255 of the per-pixel max channel value.
    per_pixel_max = np.max(rgb, axis=-1, keepdims=True)
    representable = rgb > (per_pixel_max / 255.0)
    rel_err = np.abs(decoded_hw3 - rgb) / (np.abs(rgb) + 1e-6)
    max_rep_err = rel_err[representable].max() if representable.any() else 0.0
    assert max_rep_err < 0.01, (
        f"Max relative RGBE error on representable channels {max_rep_err:.4f} >= 1%"
    )


def test_rgbe_roundtrip_dark() -> None:
    """Very dark images (near zero) must not produce NaN or excessive error."""
    from oss.data.mitsuba_gen.render_worker import encode_rgbe
    from oss.data.noisebase import _decompress_rgbe

    rgb = np.full((4, 4, 3), 1e-5, dtype=np.float32)
    rgb[0, 0, 0] = 1.0

    rgbe, exp_pair = encode_rgbe(rgb)
    color_5d = rgbe.transpose(2, 0, 1)[None, ..., None]
    decoded = _decompress_rgbe(color_5d, exp_pair[None, :])
    assert np.isfinite(decoded).all()


def test_rgbe_zero_image() -> None:
    """All-zero image must not crash."""
    from oss.data.mitsuba_gen.render_worker import encode_rgbe

    rgb = np.zeros((8, 8, 3), dtype=np.float32)
    rgbe, exp_pair = encode_rgbe(rgb)
    assert rgbe.dtype == np.uint8
    assert exp_pair.shape == (2,)


# ---------------------------------------------------------------------------
# Scene builder (no mitsuba needed — only builds dicts)
# ---------------------------------------------------------------------------

def test_scene_builder_returns_correct_frame_count() -> None:
    from oss.data.mitsuba_gen.scene_builder import build_scene

    rng = np.random.default_rng(1)
    spec = build_scene(rng, scene_type="room", seq_len=4, resolution=(64, 64))
    assert len(spec.frames) == 4


def test_scene_builder_all_types() -> None:
    from oss.data.mitsuba_gen.scene_builder import build_scene

    for stype in ("room", "corridor", "outdoor"):
        rng = np.random.default_rng(2)
        spec = build_scene(rng, scene_type=stype, seq_len=2, resolution=(32, 32))
        assert len(spec.frames) == 2


def test_scene_builder_random_type() -> None:
    from oss.data.mitsuba_gen.scene_builder import build_scene

    rng = np.random.default_rng(3)
    spec = build_scene(rng, scene_type=None, seq_len=3, resolution=(32, 32))
    assert len(spec.frames) == 3


def test_scene_builder_frame_spec_has_matrices() -> None:
    from oss.data.mitsuba_gen.scene_builder import build_scene

    rng = np.random.default_rng(4)
    spec = build_scene(rng, seq_len=2, resolution=(32, 32))
    for frame in spec.frames:
        assert frame.view_mat.shape == (4, 4)
        assert frame.proj_mat.shape == (4, 4)
        assert frame.camera_origin.shape == (3,)


# ---------------------------------------------------------------------------
# Zarr schema test (requires mitsuba)
# ---------------------------------------------------------------------------

@pytest.mark.mitsuba
@_REQUIRES_MITSUBA
def test_zarr_schema(tmp_path: Path) -> None:
    """Generate one sequence, verify NoiseBaseDataset can load it with correct shapes."""
    import numpy as np
    from oss.data.mitsuba_gen.scene_builder import build_scene
    from oss.data.mitsuba_gen.render_worker import render_sequence
    from oss.data.mitsuba_gen.zarr_writer import write_sequence
    from oss.data.noisebase import NoiseBaseDataset

    seq_len = 2
    res = (64, 64)
    spp_gt = 4

    rng = np.random.default_rng(42)
    spec = build_scene(rng, scene_type="room", seq_len=seq_len, resolution=res)
    buffers = render_sequence(spec, spp_noisy=1, spp_gt=spp_gt, seed_base=0)

    out_path = tmp_path / "scene0000.zip"
    write_sequence(buffers, out_path)
    assert out_path.exists()

    ds = NoiseBaseDataset(
        root=tmp_path,
        sequence_length=seq_len,
        resolution=(64, 64),
        scale_factor=2.0,
        split="train",
    )
    assert len(ds) == 1

    item = ds[0]
    expected_keys = {"color_lr", "gt_hr", "motion_lr", "depth_lr", "normals_lr", "albedo_lr"}
    assert set(item.keys()) == expected_keys

    import torch
    for k, v in item.items():
        assert isinstance(v, torch.Tensor), f"{k} is not a Tensor"
        assert v.dtype == torch.float32, f"{k} dtype is {v.dtype}"
        assert v.ndim == 4, f"{k} shape {tuple(v.shape)} is not 4D"
        assert v.shape[0] == seq_len, f"{k} T={v.shape[0]} != {seq_len}"
        assert torch.isfinite(v).all(), f"{k} contains non-finite values"

    assert item["gt_hr"].shape[1] == 3
    assert item["color_lr"].shape[1] == 3
    assert item["motion_lr"].shape[1] == 2
    assert item["depth_lr"].shape[1] == 1
    assert item["normals_lr"].shape[1] == 3
    assert item["albedo_lr"].shape[1] == 3
