"""Procedural Mitsuba 3 scene generation for dataset synthesis."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class FrameSpec:
    scene_dict: dict
    camera_origin: np.ndarray
    camera_target: np.ndarray
    camera_up: np.ndarray
    view_mat: np.ndarray
    proj_mat: np.ndarray


@dataclass
class SceneSpec:
    frames: list[FrameSpec]
    roughness_map_fn: Callable[[np.ndarray], np.ndarray]
    width: int
    height: int


def _random_color(rng: np.random.Generator) -> list[float]:
    return rng.uniform(0.1, 0.9, size=3).tolist()


def _principled_bsdf(rng: np.random.Generator) -> dict:
    metallic = float(rng.uniform(0.0, 1.0)) if rng.random() < 0.2 else 0.0
    return {
        "type": "principled",
        "base_color": {"type": "rgb", "value": _random_color(rng)},
        "roughness": float(rng.uniform(0.05, 0.95)),
        "metallic": metallic,
    }


def _perspective_camera(origin: np.ndarray, target: np.ndarray, up: np.ndarray,
                         fov_deg: float, width: int, height: int) -> dict:
    return {
        "type": "perspective",
        "fov": fov_deg,
        "fov_axis": "x",
        "to_world": {
            "type": "lookat",
            "origin": origin.tolist(),
            "target": target.tolist(),
            "up": up.tolist(),
        },
        "film": {
            "type": "hdrfilm",
            "width": width,
            "height": height,
            "pixel_format": "rgb",
        },
        "sampler": {"type": "independent"},
    }


def _view_matrix(origin: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - origin
    norm = np.linalg.norm(f)
    if norm < 1e-9:
        f = np.array([0.0, 0.0, -1.0])
    else:
        f = f / norm
    r = np.cross(f, up)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-9:
        r = np.array([1.0, 0.0, 0.0])
    else:
        r = r / r_norm
    u = np.cross(r, f)
    mat = np.eye(4, dtype=np.float32)
    mat[0, :3] = r
    mat[1, :3] = u
    mat[2, :3] = -f
    mat[0, 3] = -float(np.dot(r, origin))
    mat[1, 3] = -float(np.dot(u, origin))
    mat[2, 3] = float(np.dot(f, origin))
    return mat


def _proj_matrix(fov_deg: float, aspect: float, near: float = 0.1, far: float = 1000.0) -> np.ndarray:
    fov_rad = math.radians(fov_deg)
    f = 1.0 / math.tan(fov_rad / 2.0)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2.0 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def _rotate_around_up(vec: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a vector around the Y axis by angle_deg degrees."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return (rot @ vec.reshape(3, 1)).reshape(3).astype(np.float32)


def _build_room_scene(rng: np.random.Generator, width: int, height: int) -> tuple[dict, dict]:
    room_size = float(rng.uniform(4.0, 8.0))
    half = room_size / 2.0

    walls = []
    faces = [
        # floor
        (np.array([0, 1, 0]), -half, room_size),
        # ceiling
        (np.array([0, -1, 0]), -half, room_size),
        # back wall
        (np.array([0, 0, 1]), -half, room_size),
        # left wall
        (np.array([1, 0, 0]), -half, room_size),
        # right wall
        (np.array([-1, 0, 0]), -half, room_size),
    ]
    for i, (normal, offset, size) in enumerate(faces):
        walls.append({
            "type": "rectangle",
            "to_world": {
                "type": "scale",
                "value": [size / 2, size / 2, 1.0],
            },
            "bsdf": _principled_bsdf(rng),
        })

    scene_dict: dict = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 8},
    }
    for i, w in enumerate(walls):
        scene_dict[f"wall_{i}"] = w

    n_objects = int(rng.integers(2, 6))
    for i in range(n_objects):
        pos = rng.uniform(-half * 0.6, half * 0.6, size=3).astype(np.float32)
        pos[1] = -half + float(rng.uniform(0.2, 1.2))
        if rng.random() < 0.5:
            scale = float(rng.uniform(0.2, 0.8))
            scene_dict[f"obj_{i}"] = {
                "type": "sphere",
                "center": pos.tolist(),
                "radius": scale,
                "bsdf": _principled_bsdf(rng),
            }
        else:
            sx = float(rng.uniform(0.2, 1.0))
            sy = float(rng.uniform(0.2, 1.0))
            sz = float(rng.uniform(0.2, 1.0))
            scene_dict[f"obj_{i}"] = {
                "type": "cube",
                "to_world": {
                    "type": "translate",
                    "value": pos.tolist(),
                },
                "bsdf": _principled_bsdf(rng),
            }

    n_lights = int(rng.integers(1, 4))
    for i in range(n_lights):
        lpos = rng.uniform(-half * 0.5, half * 0.5, size=3).astype(np.float32)
        lpos[1] = half - 0.1
        scene_dict[f"light_{i}"] = {
            "type": "rectangle",
            "to_world": {
                "type": "translate",
                "value": lpos.tolist(),
            },
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "rgb",
                    "value": rng.uniform(2.0, 10.0, size=3).tolist(),
                },
            },
        }

    cam_origin = np.array([0.0, 0.0, half * 0.7], dtype=np.float32)
    cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    motion_state = {
        "room_size": room_size,
        "cam_origin": cam_origin,
        "cam_target": cam_target,
        "cam_up": cam_up,
    }
    return scene_dict, motion_state


def _build_corridor_scene(rng: np.random.Generator, width: int, height: int) -> tuple[dict, dict]:
    length = float(rng.uniform(10.0, 20.0))
    w = float(rng.uniform(2.0, 4.0))
    h = float(rng.uniform(2.5, 4.0))

    scene_dict: dict = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 8},
        "light_end": {
            "type": "rectangle",
            "to_world": {
                "type": "translate",
                "value": [0.0, h / 2.0 - 0.05, -length / 2.0],
            },
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "rgb",
                    "value": rng.uniform(5.0, 20.0, size=3).tolist(),
                },
            },
        },
    }

    n_objects = int(rng.integers(1, 3))
    for i in range(n_objects):
        z = float(rng.uniform(-length * 0.4, length * 0.4))
        x = float(rng.uniform(-w * 0.3, w * 0.3))
        pos = np.array([x, -h / 2.0 + 0.3, z], dtype=np.float32)
        if rng.random() < 0.5:
            scene_dict[f"obj_{i}"] = {
                "type": "sphere",
                "center": pos.tolist(),
                "radius": float(rng.uniform(0.2, 0.5)),
                "bsdf": _principled_bsdf(rng),
            }
        else:
            scene_dict[f"obj_{i}"] = {
                "type": "cube",
                "to_world": {
                    "type": "translate",
                    "value": pos.tolist(),
                },
                "bsdf": _principled_bsdf(rng),
            }

    cam_origin = np.array([0.0, 0.0, length * 0.4], dtype=np.float32)
    cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    motion_state = {
        "room_size": min(w, h),
        "cam_origin": cam_origin,
        "cam_target": cam_target,
        "cam_up": cam_up,
    }
    return scene_dict, motion_state


def _build_outdoor_scene(rng: np.random.Generator, width: int, height: int) -> tuple[dict, dict]:
    ground_size = float(rng.uniform(20.0, 40.0))

    scene_dict: dict = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 8},
        "ground": {
            "type": "rectangle",
            "to_world": {
                "type": "scale",
                "value": [ground_size / 2, ground_size / 2, 1.0],
            },
            "bsdf": _principled_bsdf(rng),
        },
        "sky": {
            "type": "constant",
            "radiance": {
                "type": "rgb",
                "value": rng.uniform(0.5, 2.0, size=3).tolist(),
            },
        },
    }

    n_columns = int(rng.integers(3, 9))
    for i in range(n_columns):
        x = float(rng.uniform(-ground_size * 0.4, ground_size * 0.4))
        z = float(rng.uniform(-ground_size * 0.4, ground_size * 0.4))
        col_height = float(rng.uniform(1.0, 5.0))
        scene_dict[f"col_{i}"] = {
            "type": "cube",
            "to_world": {
                "type": "translate",
                "value": [x, col_height / 2.0, z],
            },
            "bsdf": _principled_bsdf(rng),
        }

    cam_origin = np.array([0.0, 3.0, ground_size * 0.35], dtype=np.float32)
    cam_target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    motion_state = {
        "room_size": ground_size * 0.1,
        "cam_origin": cam_origin,
        "cam_target": cam_target,
        "cam_up": cam_up,
    }
    return scene_dict, motion_state


_SCENE_BUILDERS = {
    "room": _build_room_scene,
    "corridor": _build_corridor_scene,
    "outdoor": _build_outdoor_scene,
}

_SCENE_TYPES = list(_SCENE_BUILDERS.keys())


def build_scene(
    rng: np.random.Generator,
    scene_type: str | None = None,
    seq_len: int = 8,
    resolution: tuple[int, int] = (512, 512),
) -> SceneSpec:
    """Build a procedural multi-frame SceneSpec.

    Parameters
    ----------
    rng
        Seeded RNG; all randomness flows through this.
    scene_type
        One of 'room', 'corridor', 'outdoor', or None for random selection.
    seq_len
        Number of frames in the sequence.
    resolution
        (width, height) in pixels.
    """
    width, height = resolution
    if scene_type is None:
        scene_type = _SCENE_TYPES[int(rng.integers(len(_SCENE_TYPES)))]
    if scene_type not in _SCENE_BUILDERS:
        raise ValueError(f"Unknown scene_type {scene_type!r}; expected one of {_SCENE_TYPES}")

    base_dict, motion_state = _SCENE_BUILDERS[scene_type](rng, width, height)

    fov_deg = float(rng.uniform(45.0, 75.0))
    aspect = width / height
    room_size = motion_state["room_size"]
    max_translate = room_size * 0.05
    max_rotate_deg = 2.0

    cam_origin = motion_state["cam_origin"].copy()
    cam_target = motion_state["cam_target"].copy()
    cam_up = motion_state["cam_up"].copy()

    frames: list[FrameSpec] = []
    for f in range(seq_len):
        if f > 0:
            delta = rng.uniform(-max_translate, max_translate, size=3).astype(np.float32)
            cam_origin = cam_origin + delta
            rot_deg = float(rng.uniform(-max_rotate_deg, max_rotate_deg))
            direction = cam_target - cam_origin
            direction = _rotate_around_up(direction, rot_deg)
            cam_target = cam_origin + direction

        frame_dict = dict(base_dict)
        frame_dict["camera"] = _perspective_camera(cam_origin, cam_target, cam_up, fov_deg, width, height)

        view_mat = _view_matrix(cam_origin, cam_target, cam_up)
        proj_mat = _proj_matrix(fov_deg, aspect)

        frames.append(FrameSpec(
            scene_dict=frame_dict,
            camera_origin=cam_origin.copy(),
            camera_target=cam_target.copy(),
            camera_up=cam_up.copy(),
            view_mat=view_mat,
            proj_mat=proj_mat,
        ))

    roughness_scalar = float(rng.uniform(0.05, 0.95))

    def roughness_map_fn(shape_hw: np.ndarray) -> np.ndarray:
        h, w = shape_hw
        return np.full((h, w), roughness_scalar, dtype=np.float32)

    return SceneSpec(frames=frames, roughness_map_fn=roughness_map_fn, width=width, height=height)
