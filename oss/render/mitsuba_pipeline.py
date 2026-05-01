"""Mitsuba 3 paired rendering pipeline (noisy + GT + G-buffer)."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import drjit as dr
import mitsuba as mi
import numpy as np


def _select_variant():
    """Pick the best available Mitsuba variant. Try CUDA, fall back to LLVM."""
    # Mitsuba raises ImportError ("Requested an unsupported variant ...") when an
    # unbuilt variant is requested, and AttributeError if the variant name is
    # unknown to the build. Catching only those two keeps unrelated bugs loud.
    for v in ("cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"):
        try:
            mi.set_variant(v)
            return v
        except (ImportError, AttributeError):
            continue
    raise RuntimeError("No usable Mitsuba variant found")


_VARIANT = _select_variant()


def _render_radiance(scene, spp: int, seed: int) -> np.ndarray:
    integrator = mi.load_dict({"type": "path", "max_depth": 8})
    img = mi.render(scene, integrator=integrator, spp=spp, seed=seed)
    return np.array(img, dtype=np.float32)


def _render_aovs(scene) -> dict[str, np.ndarray]:
    """Render aux G-buffer via AOV integrator wrapping a 1-bounce path tracer.

    AOV channel layout (Mitsuba 3 convention):
      [0:3]   primary RGB from inner integrator
      [3:6]   albedo
      [6:9]   sh_normal
      [9:10]  depth

    NOTE: Mitsuba 3.7/3.8 AOV integrator does not expose a `motion` channel
    (only albedo/sh_normal/geo_normal/position/uv/depth/prim_index/shape_index
    are supported). Motion vectors are synthesized as zeros here, which is
    correct for the static-camera/static-scene case (cbox). For Bistro and
    multi-frame training data, motion will need to be computed manually by
    re-projecting world-space hit points (from `position` AOV) through a
    previous-frame view-projection matrix — see scenes.py for the per-view
    camera transforms that make this feasible.
    """
    aov = mi.load_dict({
        "type": "aov",
        "aovs": "albedo:albedo,normal:sh_normal,depth:depth",
        "integrator": {"type": "path", "max_depth": 1},
    })
    img = mi.render(scene, integrator=aov, spp=1)
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    return {
        "albedo": arr[..., 3:6],
        "normal": arr[..., 6:9],
        "depth":  arr[..., 9:10],
        "motion": np.zeros((h, w, 2), dtype=np.float32),
    }


def render_pair(
    scene_name: str,
    view_index: int,
    spp_noisy: int = 1,
    spp_gt: int = 4096,
    resolution: tuple[int, int] = (1920, 1080),
    out_dir: Optional[Path] = None,
) -> dict[str, np.ndarray]:
    """Render a paired (noisy + ground truth + G-buffer) image triplet.

    Returns a dict with keys ``noisy, ground_truth, albedo, normal, depth, motion``;
    all values are ``np.float32`` arrays in linear HDR (not tonemapped). The two
    radiance renders use deterministic but distinct seeds derived from
    ``view_index`` so re-runs are reproducible and the noisy/GT pair is
    decorrelated.

    Parameters
    ----------
    scene_name
        ``"cbox"`` (Mitsuba bundled, used by the test) or ``"bistro"`` (loaded
        via the ``ORS_BISTRO_XML`` environment variable). See ``scenes.py``.
    view_index
        Index into the scene's ``_VIEWS`` list (0 for cbox; 0–3 for bistro).
    spp_noisy, spp_gt
        Samples per pixel for the noisy and ground-truth renders.
    resolution
        Output image resolution as ``(width, height)``.
    out_dir
        If provided, every buffer is also written to disk as
        ``<scene>_v<view:04d>_<key>.exr`` under this directory. The directory
        is created if missing. When ``None``, the function only returns the
        in-memory dict.
    """
    from .scenes import load_scene
    scene = load_scene(scene_name, view_index, resolution)
    noisy = _render_radiance(scene, spp=spp_noisy, seed=view_index * 7919 + 1)
    gt    = _render_radiance(scene, spp=spp_gt,    seed=view_index * 7919 + 2)
    aovs  = _render_aovs(scene)
    result = {"noisy": noisy, "ground_truth": gt, **aovs}

    if out_dir is not None:
        import pyexr
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in result.items():
            pyexr.write(str(out_dir / f"{scene_name}_v{view_index:04d}_{k}.exr"), v)

    return result
