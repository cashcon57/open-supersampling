"""Scene loaders. cbox = Mitsuba bundled (test). bistro = local install via $ORS_BISTRO_XML."""
from __future__ import annotations
from pathlib import Path
import os

import mitsuba as mi


_VIEWS: dict[str, list[dict]] = {
    "cbox": [
        {"origin": [0, 1, 6.8], "target": [0, 1, 0], "up": [0, 1, 0]},
    ],
    "bistro": [
        {"origin": [-3.0, 1.6, 5.0], "target": [0, 1.5, 0], "up": [0, 1, 0]},
        {"origin": [4.0, 1.7, -2.0], "target": [0, 1.5, 0], "up": [0, 1, 0]},
        {"origin": [0.0, 4.0, 6.0],  "target": [0, 1.5, 0], "up": [0, 1, 0]},
        {"origin": [-5.0, 1.5, 0.0], "target": [0, 1.5, 0], "up": [0, 1, 0]},
    ],
}


def _build_sensor(view: dict, resolution: tuple[int, int]):
    width, height = resolution
    return {
        "type": "perspective",
        "fov": 45.0,
        "to_world": mi.ScalarTransform4f.look_at(
            origin=view["origin"], target=view["target"], up=view["up"],
        ),
        "film": {
            "type": "hdrfilm",
            "width": width, "height": height,
            "pixel_format": "rgb",
            "rfilter": {"type": "box"},
        },
    }


def load_scene(name: str, view_index: int, resolution: tuple[int, int]):
    if name == "cbox":
        # Mitsuba's cornell_box is a dict-shaped scene description.
        scene_dict = mi.cornell_box()
        scene_dict["sensor"] = _build_sensor(_VIEWS["cbox"][view_index], resolution)
        return mi.load_dict(scene_dict)
    elif name == "bistro":
        scene_path = os.environ.get("ORS_BISTRO_XML")
        if not scene_path or not Path(scene_path).exists():
            raise FileNotFoundError(
                "Set ORS_BISTRO_XML to the Bistro Mitsuba XML path. "
                "Download from https://developer.nvidia.com/orca/amazon-lumberyard-bistro "
                "and convert to Mitsuba format."
            )
        # File-based scene — sensor override would require pre-edit; for MVP load as-is.
        return mi.load_file(scene_path)
    else:
        raise ValueError(f"Unknown scene: {name}")
