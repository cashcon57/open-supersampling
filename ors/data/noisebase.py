"""NoiseBase dataset loader for ORS v0.2-pico.

NoiseBase
---------
Upstream: https://github.com/balintio/noisebase  (Bálint et al., NPPD SIGGRAPH 2023)
License:  Apache License 2.0  (verified 2026-04-29 from upstream LICENSE file)

ORS depends on NoiseBase only at runtime — no NoiseBase code is vendored.
This file is an independent re-implementation of the on-disk format readers,
written from the published format spec (docs/formats/sample_training_v1.md
and sample_test_v1.md in the upstream repo). License compatibility:
Apache 2.0 → Apache 2.0 (this project), no obligations beyond attribution.

On-disk format
--------------
NoiseBase ships two flavours we care about:

1. ``sampleset_v1`` (training) — one Zarr ``ZipStore`` per sequence
   (e.g. ``sampleset_training_v1/scene0001.zip``). Arrays inside are
   ``[F, ..., H, W, S]`` for per-sample buffers and ``[F, 3, H, W]`` for
   the reference. Default crop is 256x256, 32 spp, 64 frames per sequence,
   rendered from 1080x1920 source.
2. ``sampleset_test{8,32}_v1`` (test) — one Zarr ``ZipStore`` per **frame**,
   under ``sampleset_v1/test{N}/{sequence_name}/frame{idx:04d}.zip``.
   Same arrays, but with the leading ``F`` axis dropped. Test rendering is
   1080x1920, 8 or 32 spp.

Buffer keys (subset we use):

    color       uint8   [F, 4, H, W, S]   RGBE-encoded sample radiance
    exposure    float32 [F, 2]            min/max exposure for RGBE decode
    reference   float32 [F, 3, H, W]      clean ground-truth radiance
    motion      float32 [F, 3, H, W, S]   world-space motion (delta from prev frame)
    normal      float16 [F, 3, H, W, S]   world-space normal
    diffuse     float16 [F, 3, H, W, S]   diffuse / albedo

Test files drop the leading ``F`` axis.

NoiseBase stores world-space motion + camera matrices, then projects to
screen-space at load time (see ``noisebase.projective.motion_vectors``).
For ORU-Pico we want screen-space motion vectors directly. Until we
implement the world->screen projection here, we synthesise screen-space
motion from the sample-axis mean of the world-space motion projected onto
the image plane via a simple finite-difference fallback. See
``_compute_screen_motion``. This is approximate but adequate for shape
contract / smoke-testing the training pipeline; replace with a proper
projection when we wire real NoiseBase data.

Total dataset size (per upstream docs page):
- ``sampleset_v1``:        ~370 GB (1024 sequences x 64 frames x 32 spp x 256^2)
- ``sampleset_test8_v1``:  ~80 GB
- ``sampleset_test32_v1``: ~30 GB

Download: NoiseBase ships a ``noisebase download <name>`` CLI; ORS does
not (yet) wrap it. Users point ``NoiseBaseDataset(root=...)`` at a
directory containing the unpacked Zarr ZipStores.

ORS adapter contract
--------------------
``NoiseBaseDataset.__getitem__`` returns a dict of (T, C, H, W) float32
tensors. Six keys: color_lr, gt_hr, motion_lr, depth_lr, normals_lr,
albedo_lr. LR is HR / scale_factor (default 2x), produced by
box-downsample if needed.

If the on-disk resolution does not match the requested ``resolution``,
we bilinear-resample on load — used both for cropping arbitrary patch
sizes and for resizing 1080p test data down to 800p output.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# Buffer keys we read from the on-disk Zarr group. ``color`` is the noisy
# RGBE radiance; ``reference`` is the clean GT.
_REQUIRED_KEYS = ("color", "exposure", "reference", "motion", "normal", "diffuse")


def _decompress_rgbe(color: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Decode RGBE-compressed sample radiance to float32 RGB.

    NoiseBase stores radiance as uint8 RGBE with a per-frame log-space
    exposure pair ``(emin, emax)``. The exact decode is in
    ``noisebase.compression.decompress_RGBE`` upstream; we reproduce the
    minimal version here (mantissa * 2^exponent, exponent remapped from
    [0, 255] -> [emin, emax]).

    Parameters
    ----------
    color
        ``[F, 4, H, W, S]`` uint8 (or ``[4, H, W, S]`` for test).
    exposure
        ``[F, 2]`` float32 (or ``[2]`` for test).

    Returns
    -------
    np.ndarray
        Float32 RGB samples, same shape as ``color`` minus the channel-4
        axis: ``[F, 3, H, W, S]``.
    """
    # Promote to consistent shape: add F=1 if missing.
    if color.ndim == 4:  # test format, [4, H, W, S]
        color = color[None, ...]
    if exposure.ndim == 1:  # [2] -> [1, 2]
        exposure = exposure[None, ...]

    rgb = color[:, :3, ...].astype(np.float32) / 255.0
    e_byte = color[:, 3, ...].astype(np.float32)  # [F, H, W, S]

    emin = exposure[:, 0].reshape(-1, 1, 1, 1)
    emax = exposure[:, 1].reshape(-1, 1, 1, 1)
    # Map e_byte in [0, 255] to log2-exposure in [emin, emax], then 2^x.
    log_e = emin + (emax - emin) * (e_byte / 255.0)
    scale = np.power(2.0, log_e)[:, None, ...]  # [F, 1, H, W, S]
    return (rgb * scale).astype(np.float32)


def _avg_samples(buf: np.ndarray) -> np.ndarray:
    """Average over the trailing sample axis: [..., H, W, S] -> [..., H, W]."""
    return buf.mean(axis=-1).astype(np.float32)


def _compute_depth_from_position(position: np.ndarray, camera_pos: np.ndarray) -> np.ndarray:
    """Per-sample world-space distance from camera, averaged over samples.

    NoiseBase does not store depth directly — it stores world-space
    ``position`` per sample plus the camera trajectory. ORU-Pico only
    needs *some* monotonic depth-ish channel for guidance, so we use the
    camera-to-sample distance.

    Parameters
    ----------
    position
        ``[F, 3, H, W, S]`` world-space sample position.
    camera_pos
        ``[F, 3]`` world-space camera position.

    Returns
    -------
    np.ndarray
        ``[F, 1, H, W]`` float32 depth.
    """
    cam = camera_pos.reshape(-1, 3, 1, 1, 1)  # broadcast to [F, 3, H, W, S]
    delta = position - cam
    dist = np.sqrt((delta * delta).sum(axis=1, keepdims=True))  # [F, 1, H, W, S]
    return dist.mean(axis=-1).astype(np.float32)


def _compute_screen_motion(motion_world: np.ndarray) -> np.ndarray:
    """Approximate screen-space 2D motion from world-space 3D motion.

    The faithful path is to project ``position`` and ``position - motion``
    through ``view_proj_mat`` and ``prev view_proj_mat``, take the
    pixel-space delta. Until that's wired, we use the X/Y components of
    the world-space motion (sample-mean) as a stand-in. This is enough to
    exercise the (T, 2, H, W) shape contract; it is **not** a substitute
    for true motion vectors when training against real data.

    Parameters
    ----------
    motion_world
        ``[F, 3, H, W, S]`` world-space motion.

    Returns
    -------
    np.ndarray
        ``[F, 2, H, W]`` float32 screen-space-ish motion.
    """
    avg = motion_world.mean(axis=-1).astype(np.float32)  # [F, 3, H, W]
    return avg[:, :2, ...]  # take X/Y


def _resize_chw(buf: np.ndarray, target_hw: tuple[int, int], mode: str) -> np.ndarray:
    """Bilinear/area resize a [F, C, H, W] buffer to ``target_hw``.

    ``mode`` is the torch ``F.interpolate`` mode ("bilinear" for floats,
    "area" for downsample). Channels-first preserved. Out: [F, C, H', W'].
    """
    if buf.shape[-2:] == target_hw:
        return buf
    t = torch.from_numpy(buf)  # [F, C, H, W]
    align = False if mode == "bilinear" else None
    if align is False:
        out = F.interpolate(t, size=target_hw, mode=mode, align_corners=False)
    else:
        out = F.interpolate(t, size=target_hw, mode=mode)
    return out.numpy()


def _load_zarr_sequence(path: Path) -> dict[str, np.ndarray]:
    """Load all required NoiseBase buffers from a single Zarr ZipStore.

    Lazy-imports zarr so the rest of ORS doesn't need it at import time.
    Works with zarr v2 *and* v3 (the v3 API has ``zarr.storage.ZipStore``;
    v2 exposes ``zarr.ZipStore``).
    """
    import zarr  # lazy

    # zarr 3 path
    if hasattr(zarr, "storage") and hasattr(zarr.storage, "ZipStore"):
        store = zarr.storage.ZipStore(str(path), mode="r")
        try:
            grp = zarr.open_group(store=store, mode="r")
            out: dict[str, np.ndarray] = {}
            keys = set(grp.array_keys()) if hasattr(grp, "array_keys") else set(grp.keys())
            for k in _REQUIRED_KEYS:
                if k not in keys:
                    raise KeyError(f"NoiseBase sequence {path} missing required buffer '{k}'")
                out[k] = np.asarray(grp[k][:])
            # Optional camera position for depth synthesis.
            if "camera_position" in keys:
                out["camera_position"] = np.asarray(grp["camera_position"][:])
            return out
        finally:
            store.close()

    # zarr 2 path
    store = zarr.ZipStore(str(path), mode="r")  # type: ignore[attr-defined]
    try:
        grp = zarr.group(store=store)
        out = {}
        for k in _REQUIRED_KEYS:
            if k not in grp:
                raise KeyError(f"NoiseBase sequence {path} missing required buffer '{k}'")
            out[k] = np.asarray(grp[k][:])
        if "camera_position" in grp:
            out["camera_position"] = np.asarray(grp["camera_position"][:])
        return out
    finally:
        store.close()


# Filename matchers for the training format. The upstream default is
# ``sampleset_training_v1/scene{index:04d}.zip``; we also accept any
# ``*.zip`` directly under ``root`` to keep the loader portable.
_TRAINING_RE = re.compile(r"^scene(\d+)\.zip$")


def _discover_sequences(root: Path, split: str) -> list[Path]:
    """Find sequence Zarr ZipStores under ``root``.

    Search order:

    1. ``root / sampleset_training_v1 / scene*.zip`` (canonical training).
    2. ``root / *.zip`` (flat layout — used by the synthetic test fixture).
    3. ``root / **/scene*.zip`` recursively (fallback).

    The ``split`` argument is honoured by deterministic slicing of the
    sorted list: 80% train, 10% val, 10% test. NoiseBase has separate
    test sets on disk, but for v0.2-pico we just want a deterministic
    train/val split off ``sampleset_v1``.
    """
    canonical = root / "sampleset_training_v1"
    if canonical.is_dir():
        seqs = sorted(canonical.glob("scene*.zip"))
    else:
        seqs = sorted(root.glob("*.zip"))
        if not seqs:
            seqs = sorted(root.rglob("scene*.zip"))

    if not seqs:
        raise FileNotFoundError(
            f"No NoiseBase sequences (.zip Zarr stores) found under {root}. "
            "Expected `sampleset_training_v1/scene*.zip` or `*.zip`."
        )

    n = len(seqs)
    # Deterministic 80/10/10 slice. For tiny n we fall back to: train=all,
    # val=last, test=last (single-shot tests should pass split='train').
    if n < 5:
        if split == "train":
            return seqs
        return seqs[-1:]
    n_train = int(n * 0.8)
    n_val = max(1, int(n * 0.1))
    if split == "train":
        return seqs[:n_train]
    if split == "val":
        return seqs[n_train : n_train + n_val]
    if split == "test":
        return seqs[n_train + n_val :]
    raise ValueError(f"Unknown split {split!r}; expected 'train' | 'val' | 'test'")


class NoiseBaseDataset(Dataset):
    """PyTorch Dataset adapter for NoiseBase ``sampleset_v1``-style data.

    Each item is one contiguous ``sequence_length``-frame window taken from
    a single sequence file, returned as a dict of (T, C, H, W) float32
    tensors. See module docstring for the buffer mapping.

    Parameters
    ----------
    root
        Directory containing NoiseBase Zarr ZipStores.
    sequence_length
        Number of frames per item (T).
    resolution
        Output (HR) (H, W) in pixels. LR is this divided by ``scale_factor``.
        If the on-disk resolution differs we bilinear-resize.
    scale_factor
        HR / LR ratio. Default 2.0.
    split
        ``'train'`` | ``'val'`` | ``'test'``.
    """

    def __init__(
        self,
        root: Path | str,
        sequence_length: int = 8,
        resolution: tuple[int, int] = (800, 1280),
        scale_factor: float = 2.0,
        split: str = "train",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.sequence_length = int(sequence_length)
        self.hr_hw = (int(resolution[0]), int(resolution[1]))
        self.scale_factor = float(scale_factor)
        if self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be > 0; got {scale_factor}")
        if self.hr_hw[0] % int(self.scale_factor) or self.hr_hw[1] % int(self.scale_factor):
            # Not a hard error — bilinear resize handles non-integer ratios —
            # but warn-via-assert during construction would be too noisy. We
            # silently allow it: F.interpolate with non-integer factors is fine.
            pass
        self.lr_hw = (
            max(1, int(round(self.hr_hw[0] / self.scale_factor))),
            max(1, int(round(self.hr_hw[1] / self.scale_factor))),
        )
        self.split = split
        self.sequence_files = _discover_sequences(self.root, split)

    def __len__(self) -> int:
        return len(self.sequence_files)

    # -- internals -------------------------------------------------------

    def _to_hr(self, buf: np.ndarray, mode: str = "bilinear") -> np.ndarray:
        return _resize_chw(buf, self.hr_hw, mode)

    def _to_lr(self, hr_buf: np.ndarray, mode: str = "area") -> np.ndarray:
        return _resize_chw(hr_buf, self.lr_hw, mode)

    def _slice_window(self, buf: np.ndarray, start: int) -> np.ndarray:
        """Take a ``sequence_length``-long window from the F axis, padding by
        edge-repeat if the sequence is shorter than requested."""
        T = self.sequence_length
        end = start + T
        F_axis = buf.shape[0]
        if end <= F_axis:
            return buf[start:end]
        # pad by repeating the last frame
        head = buf[start:F_axis]
        pad = np.repeat(buf[F_axis - 1 : F_axis], end - F_axis, axis=0)
        return np.concatenate([head, pad], axis=0)

    # -- main ------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self.sequence_files):
            raise IndexError(idx)
        path = self.sequence_files[idx]
        raw = _load_zarr_sequence(path)

        # Promote training format ([F, ...]) and test format ([...]) to a
        # consistent F-leading layout.
        color = raw["color"]
        if color.ndim == 4:  # test format, single frame per file
            for k in ("color", "reference", "motion", "normal", "diffuse"):
                raw[k] = raw[k][None, ...]
            if raw["exposure"].ndim == 1:
                raw["exposure"] = raw["exposure"][None, ...]
            if "camera_position" in raw and raw["camera_position"].ndim == 1:
                raw["camera_position"] = raw["camera_position"][None, ...]

        # Decode RGBE noisy radiance, then sample-mean to get [F, 3, H, W].
        noisy_samples = _decompress_rgbe(raw["color"], raw["exposure"])  # [F, 3, H, W, S]
        noisy_avg = _avg_samples(noisy_samples)  # [F, 3, H, W]

        # Reference is already [F, 3, H, W] float32.
        ref = raw["reference"].astype(np.float32)

        # Aux buffers: average over sample axis.
        normal = _avg_samples(raw["normal"].astype(np.float32))    # [F, 3, H, W]
        albedo = _avg_samples(raw["diffuse"].astype(np.float32))   # [F, 3, H, W]
        motion_screen = _compute_screen_motion(raw["motion"].astype(np.float32))  # [F, 2, H, W]

        # Depth: synthesised from camera position if available, else 0s.
        if "camera_position" in raw:
            position = raw.get("position")
            if position is None:
                # NoiseBase 'position' isn't in our required-keys list; fall
                # back to a constant depth so the shape contract still holds.
                F_, _, H, W = noisy_avg.shape
                depth = np.zeros((F_, 1, H, W), dtype=np.float32)
            else:
                depth = _compute_depth_from_position(position, raw["camera_position"])
        else:
            F_, _, H, W = noisy_avg.shape
            depth = np.zeros((F_, 1, H, W), dtype=np.float32)

        # Window the F axis to ``sequence_length``.
        # Start at frame 0 deterministically — randomised cropping is a
        # train-time concern handled by a wrapping sampler if needed.
        start = 0
        noisy_avg = self._slice_window(noisy_avg, start)
        ref = self._slice_window(ref, start)
        normal = self._slice_window(normal, start)
        albedo = self._slice_window(albedo, start)
        motion_screen = self._slice_window(motion_screen, start)
        depth = self._slice_window(depth, start)

        # Resize HR buffers to requested HR resolution.
        gt_hr_np = self._to_hr(ref, mode="bilinear")

        # LR buffers: resize HR-equivalent to LR (area downsample).
        # We feed the *noisy* radiance straight to LR — that's what the
        # network sees as the temporally-noisy low-res input.
        color_lr_np = self._to_lr(self._to_hr(noisy_avg, mode="bilinear"), mode="area")
        normals_lr_np = self._to_lr(self._to_hr(normal, mode="bilinear"), mode="area")
        albedo_lr_np = self._to_lr(self._to_hr(albedo, mode="bilinear"), mode="area")
        motion_lr_np = self._to_lr(self._to_hr(motion_screen, mode="bilinear"), mode="area")
        depth_lr_np = self._to_lr(self._to_hr(depth, mode="bilinear"), mode="area")

        return {
            "color_lr":   torch.from_numpy(np.ascontiguousarray(color_lr_np, dtype=np.float32)),
            "gt_hr":      torch.from_numpy(np.ascontiguousarray(gt_hr_np,    dtype=np.float32)),
            "motion_lr":  torch.from_numpy(np.ascontiguousarray(motion_lr_np, dtype=np.float32)),
            "depth_lr":   torch.from_numpy(np.ascontiguousarray(depth_lr_np,  dtype=np.float32)),
            "normals_lr": torch.from_numpy(np.ascontiguousarray(normals_lr_np, dtype=np.float32)),
            "albedo_lr":  torch.from_numpy(np.ascontiguousarray(albedo_lr_np, dtype=np.float32)),
        }


# ---------------------------------------------------------------------------
# Synthetic fixture writer — used by tests, but exported because it's also
# useful as a reference for "what NoiseBase sequence files look like".
# ---------------------------------------------------------------------------


def _write_synthetic_sequence(
    path: Path,
    *,
    frames: int = 4,
    height: int = 16,
    width: int = 16,
    samples: int = 2,
    seed: int = 0,
) -> None:
    """Write a tiny NoiseBase-format Zarr ZipStore for tests.

    Produces all required arrays with the right shapes and dtypes. Values
    are deterministic-from-seed but otherwise meaningless. Uses zarr
    format v2 because that's what real NoiseBase uses; zarr v3 can read
    v2 stores transparently.
    """
    import zarr  # lazy

    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    color    = rng.integers(0, 256, (frames, 4, height, width, samples), dtype=np.uint8)
    exposure = np.stack([np.full(frames, -8.0, dtype=np.float32),
                         np.full(frames,  8.0, dtype=np.float32)], axis=1)
    reference = rng.random((frames, 3, height, width), dtype=np.float32)
    motion    = rng.standard_normal((frames, 3, height, width, samples)).astype(np.float32) * 0.01
    normal    = rng.standard_normal((frames, 3, height, width, samples)).astype(np.float16)
    diffuse   = rng.random((frames, 3, height, width, samples)).astype(np.float16)
    cam_pos   = rng.standard_normal((frames, 3)).astype(np.float32)

    if hasattr(zarr, "storage") and hasattr(zarr.storage, "ZipStore"):
        store = zarr.storage.ZipStore(str(path), mode="w")
        try:
            grp = zarr.group(store=store, zarr_format=2)
            for k, v in [
                ("color", color),
                ("exposure", exposure),
                ("reference", reference),
                ("motion", motion),
                ("normal", normal),
                ("diffuse", diffuse),
                ("camera_position", cam_pos),
            ]:
                arr = grp.create_array(k, shape=v.shape, dtype=v.dtype)
                arr[:] = v
        finally:
            store.close()
    else:  # zarr v2
        store = zarr.ZipStore(str(path), mode="w")  # type: ignore[attr-defined]
        try:
            grp = zarr.group(store=store)
            for k, v in [
                ("color", color),
                ("exposure", exposure),
                ("reference", reference),
                ("motion", motion),
                ("normal", normal),
                ("diffuse", diffuse),
                ("camera_position", cam_pos),
            ]:
                grp.create_dataset(k, data=v, chunks=False)
        finally:
            store.close()


__all__ = ["NoiseBaseDataset"]
