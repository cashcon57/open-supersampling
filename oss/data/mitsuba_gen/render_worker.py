"""Core render logic: one sequence -> SequenceBuffers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene_builder import SceneSpec


@dataclass
class SequenceBuffers:
    color: np.ndarray
    exposure: np.ndarray
    reference: np.ndarray
    motion: np.ndarray
    normal: np.ndarray
    diffuse: np.ndarray
    position: np.ndarray
    camera_position: np.ndarray
    view_proj_mat: np.ndarray


def _select_variant() -> str:
    import mitsuba as mi
    for v in ("cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"):
        try:
            mi.set_variant(v)
            return v
        except (ImportError, AttributeError):
            continue
    raise RuntimeError("No usable Mitsuba variant found")


def encode_rgbe(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode linear HDR (H, W, 3) float32 to NoiseBase RGBE format.

    The NoiseBase decoder (_decompress_rgbe in noisebase.py) does NOT use
    standard Ward RGBE (per-pixel exponent in [−128, 127]). Instead it maps
    the e_byte channel linearly:

        log_e = emin + (emax - emin) * (e_byte / 255.0)
        scale = 2 ** log_e

    So we must encode the per-pixel log2(max_channel) into e_byte using that
    same linear map, and store the mantissa as rgb / 2^log_e.
    """
    H, W, _ = rgb.shape
    safe_rgb = np.maximum(rgb, 0.0)

    max_val = float(safe_rgb.max())
    if max_val < 1e-30:
        rgbe = np.zeros((H, W, 4), dtype=np.uint8)
        exp_pair = np.array([-128.0, -128.0], dtype=np.float32)
        return rgbe, exp_pair

    pos_mask = safe_rgb > 0
    if pos_mask.any():
        min_val = float(safe_rgb[pos_mask].min())
    else:
        min_val = 1e-9

    # ceil ensures 2^exp_max >= max_val so the dominant-channel mantissa fits in [0,1].
    exp_max = float(np.ceil(np.log2(max_val + 1e-9)))
    exp_min = float(np.floor(np.log2(max(min_val, 1e-9))))

    exp_range = exp_max - exp_min
    if exp_range < 1e-6:
        exp_min = exp_max - 1.0
        exp_range = 1.0

    max_ch = np.max(safe_rgb, axis=-1)
    log2_max_ch = np.where(max_ch > 0, np.log2(np.maximum(max_ch, 1e-32)), exp_min)
    log2_max_ch = np.clip(log2_max_ch, exp_min, exp_max)

    e_byte_f = (log2_max_ch - exp_min) / exp_range * 255.0
    e_byte = np.clip(np.round(e_byte_f), 0, 255).astype(np.uint8)

    # Re-derive scale from the quantized e_byte to match the decoder's computation exactly.
    decoded_log_e = exp_min + exp_range * (e_byte.astype(np.float64) / 255.0)
    scale = np.power(2.0, decoded_log_e)[..., None]
    mantissa = np.clip(np.round(safe_rgb / np.maximum(scale, 1e-32) * 255.0), 0, 255).astype(np.uint8)

    rgbe = np.concatenate([mantissa, e_byte[..., None]], axis=-1)
    return rgbe, np.array([exp_min, exp_max], dtype=np.float32)


def _render_frame(
    frame_spec,
    spp_noisy: int,
    spp_gt: int,
    seed_noisy: int,
    seed_gt: int,
    prev_position: np.ndarray | None,
) -> dict:
    import mitsuba as mi

    scene = mi.load_dict(frame_spec.scene_dict)
    H = frame_spec.scene_dict["camera"]["film"]["height"]
    W = frame_spec.scene_dict["camera"]["film"]["width"]

    noisy_integrator = mi.load_dict({"type": "path", "max_depth": 8})
    noisy_img = mi.render(scene, integrator=noisy_integrator, spp=spp_noisy, seed=seed_noisy)
    noisy_arr = np.array(noisy_img, dtype=np.float32)

    gt_integrator = mi.load_dict({"type": "path", "max_depth": 8})
    gt_img = mi.render(scene, integrator=gt_integrator, spp=spp_gt, seed=seed_gt)
    gt_arr = np.array(gt_img, dtype=np.float32)

    aov_integrator = mi.load_dict({
        "type": "aov",
        "aovs": "albedo:albedo,normal:sh_normal,depth:depth,pos:position",
        "integrator": {"type": "path", "max_depth": 1},
    })
    aov_img = mi.render(scene, integrator=aov_integrator, spp=1, seed=seed_noisy + 99999)
    aov_arr = np.array(aov_img, dtype=np.float32)

    albedo_hw3 = aov_arr[..., 3:6]
    normal_hw3 = aov_arr[..., 6:9]
    position_hw3 = aov_arr[..., 10:13]

    if prev_position is not None:
        motion_hw3 = position_hw3 - prev_position
    else:
        motion_hw3 = np.zeros((H, W, 3), dtype=np.float32)

    return {
        "noisy_hw3": noisy_arr[..., :3],
        "gt_hw3": gt_arr[..., :3],
        "albedo_hw3": albedo_hw3,
        "normal_hw3": normal_hw3,
        "position_hw3": position_hw3,
        "motion_hw3": motion_hw3,
    }


def render_sequence(
    spec: SceneSpec,
    spp_noisy: int = 1,
    spp_gt: int = 1024,
    seed_base: int = 0,
) -> SequenceBuffers:
    _select_variant()

    F = len(spec.frames)
    H = spec.height
    W = spec.width
    S = 1

    color_arr = np.zeros((F, 4, H, W, S), dtype=np.uint8)
    exposure_arr = np.zeros((F, 2), dtype=np.float32)
    reference_arr = np.zeros((F, 3, H, W), dtype=np.float32)
    motion_arr = np.zeros((F, 3, H, W, S), dtype=np.float32)
    normal_arr = np.zeros((F, 3, H, W, S), dtype=np.float16)
    diffuse_arr = np.zeros((F, 3, H, W, S), dtype=np.float16)
    position_arr = np.zeros((F, 3, H, W, S), dtype=np.float32)
    camera_pos_arr = np.zeros((F, 3), dtype=np.float32)
    view_proj_arr = np.zeros((F, 4, 4), dtype=np.float32)

    prev_position: np.ndarray | None = None

    for f, frame_spec in enumerate(spec.frames):
        seed_noisy = seed_base + f * 2
        seed_gt = seed_base + f * 2 + 1

        buffers = _render_frame(frame_spec, spp_noisy, spp_gt, seed_noisy, seed_gt, prev_position)

        noisy_hw3 = buffers["noisy_hw3"]
        rgbe, exp_pair = encode_rgbe(noisy_hw3)
        color_arr[f, :, :, :, 0] = rgbe.transpose(2, 0, 1)
        exposure_arr[f] = exp_pair

        reference_arr[f] = buffers["gt_hw3"].transpose(2, 0, 1)

        motion_hw3 = buffers["motion_hw3"]
        motion_arr[f, :, :, :, 0] = motion_hw3.transpose(2, 0, 1)

        normal_hw3 = buffers["normal_hw3"].astype(np.float16)
        normal_arr[f, :, :, :, 0] = normal_hw3.transpose(2, 0, 1)

        albedo_hw3 = buffers["albedo_hw3"].astype(np.float16)
        diffuse_arr[f, :, :, :, 0] = albedo_hw3.transpose(2, 0, 1)

        position_hw3 = buffers["position_hw3"]
        position_arr[f, :, :, :, 0] = position_hw3.transpose(2, 0, 1)
        prev_position = position_hw3

        camera_pos_arr[f] = frame_spec.camera_origin.astype(np.float32)

        vp = frame_spec.proj_mat @ frame_spec.view_mat
        view_proj_arr[f] = vp

    return SequenceBuffers(
        color=color_arr,
        exposure=exposure_arr,
        reference=reference_arr,
        motion=motion_arr,
        normal=normal_arr,
        diffuse=diffuse_arr,
        position=position_arr,
        camera_position=camera_pos_arr,
        view_proj_mat=view_proj_arr,
    )
