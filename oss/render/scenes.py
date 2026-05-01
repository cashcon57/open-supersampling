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
    """Load a registered scene with the requested per-view camera and resolution.

    Supported names: ``"cbox"`` (Mitsuba bundled cornell box, 1 view) and
    ``"bistro"`` (loaded via ``ORS_BISTRO_XML`` env var, 4 views).

    For ``cbox`` the sensor is overridden in-memory from ``_VIEWS["cbox"]``.
    For ``bistro`` the per-view sensor override is **not yet applied** — the
    scene loads with whatever camera the XML defines. Multi-view Bistro
    rendering therefore produces identical view pairs in v0.1; per-view
    sensor patching (re-emit XML or load via dict + patch) is a v0.2 item
    tracked in the README "known limitations" section.

    Raises ``FileNotFoundError`` if Bistro is requested without the env var,
    ``ValueError`` for unknown scene names, and ``IndexError`` if
    ``view_index`` is out of range for the chosen scene.
    """
    if name == "cbox":
        scene_dict = mi.cornell_box()
        scene_dict["sensor"] = _build_sensor(_VIEWS["cbox"][view_index], resolution)
        return mi.load_dict(scene_dict)
    elif name == "bistro":
        # Validate view_index even though we can't yet honor it on load_file().
        _ = _VIEWS["bistro"][view_index]
        scene_path = os.environ.get("ORS_BISTRO_XML")
        if not scene_path or not Path(scene_path).exists():
            raise FileNotFoundError(
                "Set ORS_BISTRO_XML to the Bistro Mitsuba XML path. "
                "Download from https://developer.nvidia.com/orca/amazon-lumberyard-bistro "
                "and convert to Mitsuba format."
            )
        return mi.load_file(scene_path)
    else:
        raise ValueError(f"Unknown scene: {name}")
