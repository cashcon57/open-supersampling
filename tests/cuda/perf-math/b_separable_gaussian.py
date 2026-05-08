#!/usr/bin/env python3
"""Technique B: separable axis-aligned Gaussian and rotation threshold."""
from __future__ import annotations

import json
import math


def theta_threshold_for_ratio(ratio: float, eps_weight: float = 1.0e-5) -> float:
    """Conservative threshold for ignoring only the cross term 2b dx dy.

    Bound uses |dx| <= 3 sx, |dy| <= 3 sy and |delta weight| <= 0.5 |delta q|.
    With r = sx/sy, |delta q| <= 18 |sin(theta) cos(theta)| |r - 1/r|.
    """
    anis = abs(ratio - 1.0 / ratio)
    if anis == 0.0:
        return math.inf
    # For small theta, sin(theta) cos(theta) ~= theta.
    return eps_weight / (9.0 * anis)


def main() -> None:
    ratios = [1.0, 1.25, 1.5, 2.0, 4.0, 8.0, 16.0]
    table = []
    for r in ratios:
        th = theta_threshold_for_ratio(r)
        table.append({
            "sx_over_sy": r,
            "abs_r_minus_inv_r": abs(r - 1.0 / r),
            "threshold_rad": "inf" if math.isinf(th) else th,
            "threshold_deg": None if math.isinf(th) else th * 180.0 / math.pi,
        })
    print(json.dumps({
        "technique": "B",
        "axis_aligned_identity": {
            "rot": 0.0,
            "a": "1/sx^2",
            "b": "0",
            "d": "1/sy^2",
            "q": "dx^2/sx^2 + dy^2/sy^2",
            "factorization": "exp(-0.5*q)=exp(-0.5*dx^2/sx^2)*exp(-0.5*dy^2/sy^2)",
        },
        "cross_term_bound": "|delta weight| <= 9 |sin(theta)cos(theta)| |sx/sy - sy/sx|",
        "thresholds": table,
        "verdict": "Only exactly/semi-exactly axis-aligned or nearly isotropic Gaussians are safe; otherwise keep full conic.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
