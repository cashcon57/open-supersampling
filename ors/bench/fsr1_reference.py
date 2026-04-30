"""NumPy FSR1-style EASU + RCAS reference baseline.

This is a compact CPU reference derived from AMD's open FSR1 EASU/RCAS
pipeline, not a shader-instruction-equivalent port.
"""
from __future__ import annotations

import numpy as np


def _reflect(img: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


def easu_upscale(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
    src = np.asarray(image, dtype=np.float32)
    h, w, c = src.shape
    oh, ow = int(round(h * scale_factor)), int(round(w * scale_factor))
    pad = _reflect(src, 2)
    lum = np.tensordot(pad, np.array([0.299, 0.587, 0.114], dtype=np.float32), axes=([2], [0]))
    out = np.empty((oh, ow, c), dtype=np.float32)
    for y in range(oh):
        sy = (y + 0.5) / scale_factor - 0.5
        iy = int(np.floor(sy))
        fy = sy - iy
        for x in range(ow):
            sx = (x + 0.5) / scale_factor - 0.5
            ix = int(np.floor(sx))
            fx = sx - ix
            py, px = iy + 2, ix + 2
            patch = pad[py - 1:py + 3, px - 1:px + 3]
            lpatch = lum[py - 1:py + 3, px - 1:px + 3]
            gx = float((lpatch[1:3, 2:4] - lpatch[1:3, 0:2]).mean())
            gy = float((lpatch[2:4, 1:3] - lpatch[0:2, 1:3]).mean())
            mag = min(1.0, np.hypot(gx, gy) * 4.0)
            ex, ey = (gx, gy) if mag > 1e-6 else (1.0, 0.0)
            n = np.hypot(ex, ey)
            ex, ey = ex / n, ey / n
            yy = np.arange(-1, 3, dtype=np.float32) - fy
            xx = np.arange(-1, 3, dtype=np.float32) - fx
            dx, dy = np.meshgrid(xx, yy)
            along = dx * ex + dy * ey
            across = -dx * ey + dy * ex
            wgt = 1.0 / (1.0 + along * along + (1.0 + 2.0 * mag) * across * across)
            out[y, x] = (patch * wgt[..., None]).sum((0, 1)) / wgt.sum()
    return np.clip(out, 0.0, None)


def rcas_sharpen(image: np.ndarray, sharpness: float = 0.2) -> np.ndarray:
    src = np.asarray(image, dtype=np.float32)
    pad = _reflect(src, 1)
    cen = pad[1:-1, 1:-1]
    nbr = (
        pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        + pad[:-2, :-2] + pad[:-2, 2:] + pad[2:, :-2] + pad[2:, 2:]
    ) / 8.0
    high = cen - nbr
    limit = np.minimum(cen, 1.0 - np.clip(cen, 0.0, 1.0)) / (np.abs(high) + 1e-4)
    gain = (1.0 - float(sharpness)) * np.clip(limit, 0.0, 1.0)
    return np.clip(cen + gain * high, 0.0, None)


def fsr1_upscale(image: np.ndarray, scale_factor: float = 2.0, sharpness: float = 0.2) -> np.ndarray:
    return rcas_sharpen(easu_upscale(image, scale_factor), sharpness=sharpness)
