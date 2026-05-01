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
_REQUIRED_KEYS = ("color", "exposure", "reference", "motion", "normal", "diffuse", "position")


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


def _halton(i: int, base: int) -> float:
    """One-dimensional Halton sequence sample (Halton 1964).

    Standard low-discrepancy sub-pixel jitter source used by FSR2 / DLSS.
    With base=2 for X and base=3 for Y you get well-distributed offsets in
    [0, 1)^2. We subtract 0.5 in the caller to center on [-0.5, 0.5].
    """
    f, q = 1.0, 0.0
    while i > 0:
        f /= base
        q += f * (i % base)
        i //= base
    return q


def _jitter_offsets(num_frames: int) -> np.ndarray:
    """Per-frame Halton(2,3) sub-pixel offsets in [-0.5, 0.5] HR-pixel units.

    Returns shape (num_frames, 2) — (jitter_x, jitter_y) per frame. FSR2
    uses Halton(2,3) for its jitter; DLSS uses similar quasi-random
    sequences. Centred on zero so accumulated mean = 0 across many frames.
    """
    out = np.zeros((num_frames, 2), dtype=np.float32)
    for t in range(num_frames):
        # Halton index 0 → 0; start at 1 for proper distribution
        out[t, 0] = _halton(t + 1, 2) - 0.5
        out[t, 1] = _halton(t + 1, 3) - 0.5
    return out


def _apply_jitter_to_lr(hr_tensor: np.ndarray, jitter_hr_pixels: np.ndarray, scale_factor: float) -> np.ndarray:
    """Resample HR tensor to LR with per-frame sub-pixel jitter offsets.

    Args:
        hr_tensor: shape (F, C, H_hr, W_hr) — frames at HR resolution.
        jitter_hr_pixels: shape (F, 2) — (jx, jy) per frame in HR pixel units.
        scale_factor: HR / LR ratio.
    Returns:
        (F, C, H_lr, W_lr) LR tensor where each frame samples HR at jittered
        sub-pixel positions. Each frame uses bilinear interp via grid_sample.

    LR pixel (x, y) at frame t samples HR at (x*scale + jx_t, y*scale + jy_t).
    """
    import torch
    F_, C, H_hr, W_hr = hr_tensor.shape
    H_lr = max(1, int(H_hr / scale_factor))
    W_lr = max(1, int(W_hr / scale_factor))

    hr = torch.from_numpy(np.ascontiguousarray(hr_tensor, dtype=np.float32))

    # Build a normalized [-1, 1] grid for grid_sample. Each LR pixel (x, y)
    # samples HR at (x*scale + jx, y*scale + jy) in HR pixel-center units.
    # grid_sample uses align_corners=False convention: pixel center i ↔ (2i+1)/N - 1.
    base_y = (np.arange(H_lr, dtype=np.float32) + 0.5) * scale_factor  # HR pixel-center y
    base_x = (np.arange(W_lr, dtype=np.float32) + 0.5) * scale_factor

    out = np.zeros((F_, C, H_lr, W_lr), dtype=np.float32)
    for t in range(F_):
        jx, jy = float(jitter_hr_pixels[t, 0]), float(jitter_hr_pixels[t, 1])
        # Per-pixel HR sample positions for this frame
        ys = base_y + jy
        xs = base_x + jx
        # Normalize to [-1, 1] in grid_sample convention
        norm_y = (ys * 2.0 / H_hr) - 1.0
        norm_x = (xs * 2.0 / W_hr) - 1.0
        gy, gx = np.meshgrid(norm_y, norm_x, indexing="ij")
        grid = np.stack([gx, gy], axis=-1)[None, ...]  # (1, H_lr, W_lr, 2)
        grid_t = torch.from_numpy(np.ascontiguousarray(grid, dtype=np.float32))
        sampled = torch.nn.functional.grid_sample(
            hr[t : t + 1], grid_t, mode="bilinear", padding_mode="border", align_corners=False
        )
        out[t] = sampled[0].numpy()
    return out


def _compute_screen_motion(
    positions: np.ndarray,
    view_proj_mats: np.ndarray,
) -> np.ndarray:
    """Compute screen-space motion vectors via view-projection matrix transforms.

    Projects world-space positions through the view-projection matrix at
    consecutive frames, then computes the NDC-space pixel displacement.
    This yields proper screen-space motion vectors for temporal tasks.

    Parameters
    ----------
    positions
        ``[F, 3, H, W, S]`` world-space hit positions per sample.
    view_proj_mats
        ``[F, 4, 4]`` view-projection matrices (or ``[F, 3, 4]`` if stored
        in homogeneous form). The matrix at frame t transforms world coords
        to the clip space of frame t.

    Returns
    -------
    np.ndarray
        ``[F, 2, H, W]`` float32 screen-space motion vectors in NDC space
        (range ~[-1, 1]). Frame 0 motion is zero (no prior frame).
    """
    F, _, H, W, S = positions.shape
    motion = np.zeros((F, 2, H, W), dtype=np.float32)

    if F < 2:
        return motion  # single frame, no motion

    # Pad world positions to homogeneous coordinates: [F, 3, H, W, S] -> [F, 4, H, W, S]
    ones = np.ones((F, 1, H, W, S), dtype=positions.dtype)
    pos_homo = np.concatenate([positions, ones], axis=1)  # [F, 4, H, W, S]

    # Apply view-projection for prev and curr frames.
    for f in range(1, F):
        # Reshape for matrix multiply: [4, 4] @ [4, H*W*S] -> [4, H*W*S]
        prev_mat = view_proj_mats[f - 1]  # [4, 4]
        curr_mat = view_proj_mats[f]      # [4, 4]

        pos_f_flat = pos_homo[f].reshape(4, -1)  # [4, H*W*S]
        pos_prev_flat = pos_homo[f - 1].reshape(4, -1)  # [4, H*W*S]

        # Transform to clip space.
        prev_clip = prev_mat @ pos_prev_flat  # [4, H*W*S]
        curr_clip = curr_mat @ pos_f_flat    # [4, H*W*S]

        # Perspective divide to NDC.
        prev_ndc = prev_clip[:2] / (prev_clip[3:4] + 1e-6)  # [2, H*W*S]
        curr_ndc = curr_clip[:2] / (curr_clip[3:4] + 1e-6)  # [2, H*W*S]

        # Motion in NDC space, then sample-mean, reshape to spatial.
        delta = curr_ndc - prev_ndc  # [2, H*W*S]
        delta_mean = delta.mean(axis=1)  # [2]
        motion[f] = delta.reshape(2, H, W, S).mean(axis=-1)  # [2, H, W]

    return motion


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

        # Depth: world-space distance from camera, derived from position AOV.
        position = raw["position"].astype(np.float32)  # [F, 3, H, W, S]
        depth = _compute_depth_from_position(position, raw["camera_position"])  # [F, 1, H, W]

        # Normalize depth to [0, 1] range for stable FP16 training.
        depth_max = np.maximum(depth.max(), 1e-6)
        depth = depth / depth_max

        # Screen-space motion via view-projection matrix transforms.
        # Requires view_proj_mat from raw data (if available; else fallback to zeros).
        if "view_proj_mat" in raw:
            view_proj = raw["view_proj_mat"].astype(np.float32)  # [F, 4, 4]
            motion_screen = _compute_screen_motion(position, view_proj)  # [F, 2, H, W]
        else:
            # Fallback: zeros (no view-proj data available). Frame 0 motion is
            # always zero (no prior frame); frames 1+ would need true projection.
            F_, _, H, W = position.shape[:4]
            motion_screen = np.zeros((F_, 2, H, W), dtype=np.float32)

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

        # ---- Per-frame Halton(2,3) sub-pixel jitter -----------------------
        # Critical for the model to learn temporal sub-pixel accumulation:
        # without jitter, every frame's LR samples HR at the same grid
        # positions, so accumulating frames over time can't recover detail
        # below the LR Nyquist limit. With jitter, consecutive frames sample
        # at slightly shifted sub-pixel positions, and the recurrent state
        # learns to integrate across them — same trick FSR2 / DLSS use.
        F_seq = gt_hr_np.shape[0]
        jitter_hr_pixels = _jitter_offsets(F_seq)  # (F, 2) in HR-pixel units

        # color_lr is the only buffer the network treats as the "noisy LR
        # input you actually have to recover from" — apply jitter here so
        # successive frames carry distinct sub-pixel info.
        # Other LR buffers (depth, normals, albedo, motion) are aux signals;
        # using the same jittered grid keeps them aligned with color_lr.
        color_lr_np = _apply_jitter_to_lr(
            self._to_hr(noisy_avg, mode="bilinear"),
            jitter_hr_pixels, self.scale_factor,
        )
        normals_lr_np = _apply_jitter_to_lr(
            self._to_hr(normal, mode="bilinear"),
            jitter_hr_pixels, self.scale_factor,
        )
        albedo_lr_np = _apply_jitter_to_lr(
            self._to_hr(albedo, mode="bilinear"),
            jitter_hr_pixels, self.scale_factor,
        )
        motion_lr_np = _apply_jitter_to_lr(
            self._to_hr(motion_screen, mode="bilinear"),
            jitter_hr_pixels, self.scale_factor,
        )
        depth_lr_np = _apply_jitter_to_lr(
            self._to_hr(depth, mode="bilinear"),
            jitter_hr_pixels, self.scale_factor,
        )

        # Add jitter delta (jitter[t] - jitter[t-1]) to motion_lr.
        # This represents the additional sub-pixel displacement between
        # consecutive frames' LR sample grids, beyond the camera motion
        # already encoded in motion_screen. Frame 0 has no prior, so its
        # delta is 0. Units: convert HR-pixel delta to NDC delta.
        H_hr_size = gt_hr_np.shape[-2]
        W_hr_size = gt_hr_np.shape[-1]
        for t in range(1, F_seq):
            dx_hr = jitter_hr_pixels[t, 0] - jitter_hr_pixels[t - 1, 0]
            dy_hr = jitter_hr_pixels[t, 1] - jitter_hr_pixels[t - 1, 1]
            # NDC delta = HR-pixel delta * 2 / dim (NDC spans [-1, 1])
            motion_lr_np[t, 0] += np.float32(dx_hr * 2.0 / W_hr_size)
            motion_lr_np[t, 1] += np.float32(dy_hr * 2.0 / H_hr_size)

        # G-buffer normalization for stable FP16 training.
        # Clamp normals to [-1, 1] range (sh_normal style, already in range but enforce).
        normals_lr_np = np.clip(normals_lr_np, -1.0, 1.0)
        # Clamp albedo to [0, 1] range (typical reflectance).
        albedo_lr_np = np.clip(albedo_lr_np, 0.0, 1.0)
        # Motion vectors in NDC [-1, 1] space; clamp to bounds.
        motion_lr_np = np.clip(motion_lr_np, -1.0, 1.0)
        # Depth in [0, 1] (normalized in the depth synthesis block above; re-clamp for safety).
        depth_lr_np = np.clip(depth_lr_np, 0.0, 1.0)
        # color_lr (noisy) and gt_hr (ground truth) stay HDR linear; no clamping.

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
    position  = rng.standard_normal((frames, 3, height, width, samples)).astype(np.float32) * 10.0
    cam_pos   = rng.standard_normal((frames, 3)).astype(np.float32)
    # View-projection matrices: identity + small random perturbations per frame.
    view_proj = np.zeros((frames, 4, 4), dtype=np.float32)
    for f in range(frames):
        view_proj[f] = np.eye(4, dtype=np.float32) + rng.standard_normal((4, 4)).astype(np.float32) * 0.01

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
                ("position", position),
                ("camera_position", cam_pos),
                ("view_proj_mat", view_proj),
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
                ("position", position),
                ("camera_position", cam_pos),
                ("view_proj_mat", view_proj),
            ]:
                grp.create_dataset(k, data=v, chunks=False)
        finally:
            store.close()


__all__ = ["NoiseBaseDataset"]
