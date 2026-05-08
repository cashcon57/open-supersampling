#!/usr/bin/env python3
"""Technique K: Gaussian state quantization bounds."""
from __future__ import annotations

import json
import math


def main() -> None:
    width = 1920
    height = 1080
    xy_step = max(width, height) / 32767.0
    rot_step = 2.0 * math.pi / 256.0
    fp16_rel = 2.0 ** -10
    print(json.dumps({
        "technique": "K",
        "xy_int16_step_px": xy_step,
        "xy_half_step_px": 0.5 * xy_step,
        "rot_uint8_step_rad": rot_step,
        "rot_uint8_step_deg": rot_step * 180.0 / math.pi,
        "rot_half_step_rad": 0.5 * rot_step,
        "fp16_relative_precision": fp16_rel,
        "first_order_weight_error_xy": "|delta w| <= 0.5 w |grad_q dot delta_xy|",
        "verdict": "xy int16 and fp16 scale are plausible; int8 rotation is flag-only for anisotropic Gaussians until ckpt stats bound sx/sy.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
