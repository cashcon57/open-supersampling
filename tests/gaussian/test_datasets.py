"""Tests for OSS-Gaussian Sprint 4 dataset loaders.

These tests build tiny synthetic on-disk fixtures matching each dataset's
canonical layout, then verify the loader returns the expected
``GaussianTrainingExample`` shapes. No real datasets are downloaded.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from oss.gaussian.data import (
    CANVAS_CHANNELS,
    DEPTH_CHANNELS,
    DEFAULT_WEIGHTS,
    EngineAliasedLRSynth,
    GaussianDataset,
    GaussianTrainingExample,
    HyperSimGaussianDataset,
    LR_CHANNELS,
    MOTION_CHANNELS,
    MixedGaussianDataset,
    NORMAL_CHANNELS,
    SRGDGaussianDataset,
    SintelGaussianDataset,
    TOTAL_INPUT_CHANNELS,
    TartanAirGaussianDataset,
    collate_examples,
)

# ---- Constants used by all fixtures. Small but multiple of all scales. -----

HR_H, HR_W = 64, 96   # multiple of 2, 3, 4 → safe for box downsample
SCALE = 2.0


# ---- Helpers to write canonical-format files ------------------------------


def _save_png(path: Path, tensor: torch.Tensor) -> None:
    """tensor: (3, H, W), float in [0,1]."""
    from torchvision.utils import save_image
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.clamp(0, 1), str(path))


def _write_flo(path: Path, flow: torch.Tensor) -> None:
    """flow: (2, H, W) float32 → Sintel .flo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _, H, W = flow.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<f", 202021.25))
        f.write(struct.pack("<ii", W, H))
        # .flo is row-major (H, W, 2)
        arr = flow.permute(1, 2, 0).contiguous().numpy().astype("float32")
        f.write(arr.tobytes())


def _write_dpt(path: Path, depth: torch.Tensor) -> None:
    """depth: (1, H, W) float32 → Sintel .dpt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _, H, W = depth.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<f", 202021.25))
        f.write(struct.pack("<ii", W, H))
        arr = depth.squeeze(0).contiguous().numpy().astype("float32")
        f.write(arr.tobytes())


def _save_jpg(path: Path, tensor: torch.Tensor) -> None:
    from torchvision.utils import save_image
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.clamp(0, 1), str(path))


# ---- Fixtures: build a tiny on-disk dataset of each kind ------------------


@pytest.fixture
def sintel_root(tmp_path: Path) -> Path:
    root = tmp_path / "sintel"
    pass_dir = root / "training" / "clean"
    flow_dir = root / "training" / "flow"
    depth_dir = root / "training" / "depth"
    seq = "alley_1"
    n_frames = 4
    for i in range(1, n_frames + 1):
        rgb = torch.rand(3, HR_H, HR_W)
        _save_png(pass_dir / seq / f"frame_{i:04d}.png", rgb)
        _write_dpt(depth_dir / seq / f"frame_{i:04d}.dpt", torch.rand(1, HR_H, HR_W) * 10.0)
        # flow not provided for last frame in real Sintel; mimic that.
        if i < n_frames:
            _write_flo(flow_dir / seq / f"frame_{i:04d}.flo", torch.randn(2, HR_H, HR_W) * 2.0)
    return root


@pytest.fixture
def tartanair_root(tmp_path: Path) -> Path:
    root = tmp_path / "tartanair"
    traj = root / "abandonedfactory" / "Easy" / "P000"
    img_dir = traj / "image_left"
    depth_dir = traj / "depth_left"
    flow_dir = traj / "flow"
    for i in range(3):
        idx = f"{i:06d}"
        _save_png(img_dir / f"{idx}_left.png", torch.rand(3, HR_H, HR_W))
        depth = (np.random.rand(HR_H, HR_W) * 5.0).astype("float32")
        depth_dir.mkdir(parents=True, exist_ok=True)
        np.save(depth_dir / f"{idx}_left_depth.npy", depth)
        # forward flow points to next frame; last frame has no flow.
        if i < 2:
            nxt = f"{i+1:06d}"
            flow = (np.random.randn(HR_H, HR_W, 2) * 2.0).astype("float32")
            flow_dir.mkdir(parents=True, exist_ok=True)
            np.save(flow_dir / f"{idx}_{nxt}_flow.npy", flow)
    return root


@pytest.fixture
def hypersim_root(tmp_path: Path) -> Path:
    root = tmp_path / "hypersim"
    scene = root / "ai_001_001" / "images"
    cam_color = scene / "scene_cam_00_final_preview"
    cam_geom = scene / "scene_cam_00_geometry_hdf5"
    for i in range(3):
        fid = f"{i:04d}"
        _save_jpg(cam_color / f"frame.{fid}.tonemap.jpg", torch.rand(3, HR_H, HR_W))
        depth = (np.random.rand(HR_H, HR_W) * 8.0).astype("float32")
        cam_geom.mkdir(parents=True, exist_ok=True)
        np.save(cam_geom / f"frame.{fid}.depth_meters.npy", depth)
    return root


@pytest.fixture
def srgd_root(tmp_path: Path) -> Path:
    root = tmp_path / "srgd"
    hr_dir = root / "hr"
    lr_dir = root / "lr"
    for i in range(3):
        hr = torch.rand(3, HR_H, HR_W)
        _save_png(hr_dir / f"img_{i:03d}.png", hr)
        # Provide an LR for half of them; the loader should box-downsample for the rest.
        if i % 2 == 0:
            lr = torch.nn.functional.avg_pool2d(hr.unsqueeze(0), int(SCALE)).squeeze(0)
            _save_png(lr_dir / f"img_{i:03d}.png", lr)
    return root


# ---- GaussianTrainingExample dataclass tests ------------------------------


def _example(H: int = 32, W: int = 48, hr_scale: int = 2) -> GaussianTrainingExample:
    return GaussianTrainingExample(
        lr_frame=torch.rand(LR_CHANNELS, H, W),
        depth=torch.rand(DEPTH_CHANNELS, H, W),
        motion=torch.randn(MOTION_CHANNELS, H, W),
        normals=torch.randn(NORMAL_CHANNELS, H, W),
        canvas_hint=torch.zeros(CANVAS_CHANNELS, H, W),
        gt_hr_frame=torch.rand(LR_CHANNELS, H * hr_scale, W * hr_scale),
        metadata={"source": "unit-test"},
    )


def test_training_example_roundtrip() -> None:
    e = _example()
    d = e.to_dict()
    e2 = GaussianTrainingExample.from_dict(d)
    assert torch.equal(e.lr_frame, e2.lr_frame)
    assert torch.equal(e.depth, e2.depth)
    assert torch.equal(e.motion, e2.motion)
    assert torch.equal(e.normals, e2.normals)
    assert torch.equal(e.canvas_hint, e2.canvas_hint)
    assert torch.equal(e.gt_hr_frame, e2.gt_hr_frame)
    assert e.metadata == e2.metadata


def test_training_example_stack_input_matches_param_net() -> None:
    e = _example()
    x = e.stack_input()
    assert x.shape == (TOTAL_INPUT_CHANNELS, e.lr_frame.shape[1], e.lr_frame.shape[2])
    # When normals=None, stack_input must still produce the right channel count.
    e2 = GaussianTrainingExample(
        lr_frame=e.lr_frame, depth=e.depth, motion=e.motion,
        canvas_hint=e.canvas_hint, gt_hr_frame=e.gt_hr_frame, normals=None,
    )
    x2 = e2.stack_input()
    assert x2.shape == x.shape


def test_training_example_validation_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        GaussianTrainingExample(
            lr_frame=torch.rand(LR_CHANNELS, 32, 48),
            depth=torch.rand(DEPTH_CHANNELS, 16, 48),  # mismatched
            motion=torch.zeros(MOTION_CHANNELS, 32, 48),
            canvas_hint=torch.zeros(CANVAS_CHANNELS, 32, 48),
            gt_hr_frame=torch.rand(LR_CHANNELS, 64, 96),
        )
    with pytest.raises(ValueError):
        GaussianTrainingExample(
            lr_frame=torch.rand(LR_CHANNELS, 32, 48),
            depth=torch.rand(2, 32, 48),  # wrong channel count
            motion=torch.zeros(MOTION_CHANNELS, 32, 48),
            canvas_hint=torch.zeros(CANVAS_CHANNELS, 32, 48),
            gt_hr_frame=torch.rand(LR_CHANNELS, 64, 96),
        )


def test_collate_examples_handles_missing_normals() -> None:
    a = _example()
    b = GaussianTrainingExample(
        lr_frame=a.lr_frame, depth=a.depth, motion=a.motion,
        canvas_hint=a.canvas_hint, gt_hr_frame=a.gt_hr_frame, normals=None,
    )
    batch = collate_examples([a, b])
    assert batch["lr_frame"].shape[0] == 2
    assert batch["normals"].shape == (2, NORMAL_CHANNELS, a.lr_frame.shape[1], a.lr_frame.shape[2])
    # Second normals slot should be all zeros.
    assert torch.equal(batch["normals"][1], torch.zeros_like(batch["normals"][1]))


# ---- Per-loader smoke tests -----------------------------------------------


def _check_example_shapes(e: GaussianTrainingExample, lr_h: int, lr_w: int, hr_h: int, hr_w: int) -> None:
    assert e.lr_frame.shape == (LR_CHANNELS, lr_h, lr_w)
    assert e.depth.shape == (DEPTH_CHANNELS, lr_h, lr_w)
    assert e.motion.shape == (MOTION_CHANNELS, lr_h, lr_w)
    assert e.canvas_hint.shape == (CANVAS_CHANNELS, lr_h, lr_w)
    assert e.gt_hr_frame.shape == (LR_CHANNELS, hr_h, hr_w)
    if e.normals is not None:
        assert e.normals.shape == (NORMAL_CHANNELS, lr_h, lr_w)
    # Stacked input matches the 12-channel contract.
    assert e.stack_input().shape == (TOTAL_INPUT_CHANNELS, lr_h, lr_w)
    # All tensors float32.
    for t in (e.lr_frame, e.depth, e.motion, e.canvas_hint, e.gt_hr_frame):
        assert t.dtype == torch.float32


def test_sintel_loader_smoke(sintel_root: Path) -> None:
    ds = SintelGaussianDataset(root=sintel_root, scale=SCALE)
    assert len(ds) > 0
    e = ds[0]
    _check_example_shapes(e, lr_h=HR_H // int(SCALE), lr_w=HR_W // int(SCALE), hr_h=HR_H, hr_w=HR_W)
    assert e.metadata["source"] == "sintel"


def test_tartanair_loader_smoke(tartanair_root: Path) -> None:
    ds = TartanAirGaussianDataset(root=tartanair_root, scale=SCALE)
    assert len(ds) > 0
    e = ds[0]
    _check_example_shapes(e, lr_h=HR_H // int(SCALE), lr_w=HR_W // int(SCALE), hr_h=HR_H, hr_w=HR_W)
    assert e.metadata["source"] == "tartanair"


def test_tartanair_loader_skips_corrupt_flow(tartanair_root: Path, caplog) -> None:
    bad_flow = (
        tartanair_root
        / "abandonedfactory"
        / "Easy"
        / "P000"
        / "flow"
        / "000000_000001_flow.npy"
    )
    bad_flow.write_bytes(b"not a valid npy")
    ds = TartanAirGaussianDataset(root=tartanair_root, scale=SCALE)
    # New policy (commit d483833): skip corrupt flow files instead of raising,
    # so a single bad sample doesn't kill long training runs. The skip should
    # be logged at WARNING level mentioning the bad file path.
    with caplog.at_level("WARNING"):
        try:
            ds[0]
        except (IndexError, StopIteration):
            pass
    assert any("000000_000001_flow.npy" in rec.message for rec in caplog.records), \
        "expected a WARNING log mentioning the corrupt flow file"


def test_hypersim_loader_smoke(hypersim_root: Path) -> None:
    ds = HyperSimGaussianDataset(root=hypersim_root, scale=SCALE)
    assert len(ds) > 0
    e = ds[0]
    _check_example_shapes(e, lr_h=HR_H // int(SCALE), lr_w=HR_W // int(SCALE), hr_h=HR_H, hr_w=HR_W)
    assert e.metadata["source"] == "hypersim"
    assert e.metadata.get("static") is True
    # HyperSim motion is always zero.
    assert torch.equal(e.motion, torch.zeros_like(e.motion))


def test_srgd_loader_smoke(srgd_root: Path) -> None:
    ds = SRGDGaussianDataset(root=srgd_root, scale=SCALE)
    assert len(ds) > 0
    e = ds[0]
    _check_example_shapes(e, lr_h=HR_H // int(SCALE), lr_w=HR_W // int(SCALE), hr_h=HR_H, hr_w=HR_W)
    assert e.metadata["source"] == "srgd"
    assert torch.equal(e.depth, torch.zeros_like(e.depth))
    assert torch.equal(e.motion, torch.zeros_like(e.motion))


# ---- Failure modes --------------------------------------------------------


def test_missing_root_raises_helpful_error(tmp_path: Path) -> None:
    bogus = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="Sintel"):
        SintelGaussianDataset(root=bogus)
    with pytest.raises(FileNotFoundError, match="TartanAir"):
        TartanAirGaussianDataset(root=bogus)
    with pytest.raises(FileNotFoundError, match="HyperSim"):
        HyperSimGaussianDataset(root=bogus)
    with pytest.raises(FileNotFoundError, match="SRGD"):
        SRGDGaussianDataset(root=bogus)


# ---- Mixer tests ----------------------------------------------------------


def test_mixed_dataset_honors_weights(sintel_root: Path, tartanair_root: Path,
                                      hypersim_root: Path, srgd_root: Path) -> None:
    sin = SintelGaussianDataset(sintel_root, scale=SCALE)
    tar = TartanAirGaussianDataset(tartanair_root, scale=SCALE)
    hyp = HyperSimGaussianDataset(hypersim_root, scale=SCALE)
    srgd = SRGDGaussianDataset(srgd_root, scale=SCALE)
    ds = MixedGaussianDataset(
        datasets={"sintel": sin, "tartanair": tar, "hypersim": hyp, "srgd": srgd},
        weights=DEFAULT_WEIGHTS,
        seed=0,
    )
    assert len(ds) > 0
    # Sample a bunch and verify the empirical mix is within tolerance.
    emp = ds.empirical_distribution(samples=4000)
    for k, target in DEFAULT_WEIGHTS.items():
        assert k in emp
        assert abs(emp[k] - target) < 0.05, f"{k}: emp {emp[k]:.3f} vs target {target:.3f}"

    # __getitem__ returns a valid example.
    e = ds[0]
    assert isinstance(e, GaussianTrainingExample)
    assert e.metadata["source"] in {"sintel", "tartanair", "hypersim", "srgd"}


def test_mixed_dataset_skips_none_entries(sintel_root: Path) -> None:
    sin = SintelGaussianDataset(sintel_root, scale=SCALE)
    ds = MixedGaussianDataset(
        datasets={"sintel": sin, "tartanair": None, "hypersim": None, "srgd": None},
    )
    assert len(ds) == len(sin)
    e = ds[0]
    assert e.metadata["source"] == "sintel"


def test_mixed_dataset_describe(sintel_root: Path) -> None:
    sin = SintelGaussianDataset(sintel_root, scale=SCALE)
    ds = MixedGaussianDataset(datasets={"sintel": sin})
    desc = ds.describe()
    assert "sintel" in desc and "MixedGaussianDataset" in desc


def test_mixed_dataset_rejects_empty() -> None:
    with pytest.raises(ValueError):
        MixedGaussianDataset(datasets={"sintel": None, "tartanair": None})


# ---- lr_synth integration: SintelGaussianDataset with EngineAliasedLRSynth --


def test_sintel_with_lr_synth_correct_shape(sintel_root: Path) -> None:
    """SintelGaussianDataset with a non-None lr_synth must still return
    GaussianTrainingExample with the correct LR spatial shape.

    This verifies that wiring the engine-aliased synthesis path does not break
    the dataset contract (shapes, dtypes, metadata).
    """
    synth = EngineAliasedLRSynth(
        scale=SCALE,
        enable_jitter=True,
        enable_taa_blur=True,
        enable_jpeg=False,
    )
    ds = SintelGaussianDataset(root=sintel_root, scale=SCALE, lr_synth=synth)
    assert len(ds) > 0
    e = ds[0]
    _check_example_shapes(
        e,
        lr_h=HR_H // int(SCALE),
        lr_w=HR_W // int(SCALE),
        hr_h=HR_H,
        hr_w=HR_W,
    )
    assert e.metadata["source"] == "sintel"


# ---- Helper: depth → normals -----------------------------------------------


def test_depth_to_normals_unit_length() -> None:
    depth = torch.rand(1, 16, 24)
    n = GaussianDataset._depth_to_normals(depth)
    norms = n.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
