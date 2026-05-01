"""Write SequenceBuffers to a NoiseBase-compatible zarr ZipStore."""
from __future__ import annotations

from pathlib import Path

from .render_worker import SequenceBuffers


def write_sequence(buffers: SequenceBuffers, out_path: Path) -> None:
    import zarr

    arrays = [
        ("color", buffers.color),
        ("exposure", buffers.exposure),
        ("reference", buffers.reference),
        ("motion", buffers.motion),
        ("normal", buffers.normal),
        ("diffuse", buffers.diffuse),
        ("position", buffers.position),
        ("camera_position", buffers.camera_position),
        ("view_proj_mat", buffers.view_proj_mat),
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(zarr, "storage") and hasattr(zarr.storage, "ZipStore"):
        store = zarr.storage.ZipStore(str(out_path), mode="w")
        try:
            grp = zarr.group(store=store, zarr_format=2)
            for k, v in arrays:
                arr = grp.create_array(k, shape=v.shape, dtype=v.dtype)
                arr[:] = v
        finally:
            store.close()
    else:
        store = zarr.ZipStore(str(out_path), mode="w")  # type: ignore[attr-defined]
        try:
            grp = zarr.group(store=store)
            for k, v in arrays:
                grp.create_dataset(k, data=v, chunks=False)
        finally:
            store.close()
