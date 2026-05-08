#!/usr/bin/env python3
"""Technique A: [4/4] Pade approximation for exp(-0.5 q)."""
from __future__ import annotations

import json
import numpy as np


def pade44_exp(x: np.ndarray) -> np.ndarray:
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    num = 1.0 + 0.5 * x + (3.0 / 28.0) * x2 + (1.0 / 84.0) * x3 + (1.0 / 1680.0) * x4
    den = 1.0 - 0.5 * x + (3.0 / 28.0) * x2 - (1.0 / 84.0) * x3 + (1.0 / 1680.0) * x4
    return num / den


def main() -> None:
    q = np.linspace(0.0, 9.0, 1_000_001, dtype=np.float64)
    x = -0.5 * q
    approx = pade44_exp(x)
    truth = np.exp(x)
    err = np.abs(approx - truth)
    i = int(np.argmax(err))
    print(json.dumps({
        "technique": "A",
        "pade44_exp_coefficients": {
            "numerator": [1.0, 0.5, 3.0 / 28.0, 1.0 / 84.0, 1.0 / 1680.0],
            "denominator": [1.0, -0.5, 3.0 / 28.0, -1.0 / 84.0, 1.0 / 1680.0],
        },
        "q_range": [0.0, 9.0],
        "samples": int(q.size),
        "max_abs_error": float(err[i]),
        "argmax_q": float(q[i]),
        "approx_at_argmax": float(approx[i]),
        "truth_at_argmax": float(truth[i]),
        "ship_bar": 1.0e-3,
        "passes_ship_bar": bool(err[i] < 1.0e-3),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
