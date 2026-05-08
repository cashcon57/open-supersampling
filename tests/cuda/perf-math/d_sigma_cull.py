#!/usr/bin/env python3
"""Technique D: 2D Gaussian mass retained by radial sigma culls."""
from __future__ import annotations

import json
import math


def retained_mass(radius_sigma: float) -> float:
    return 1.0 - math.exp(-0.5 * radius_sigma * radius_sigma)


def main() -> None:
    rows = []
    for r in [2.0, 2.5, 3.0, math.sqrt(12.0)]:
        retained = retained_mass(r)
        dropped = 1.0 - retained
        rows.append({
            "radius_sigma": r,
            "retained_mass": retained,
            "dropped_mass": dropped,
            "linf_error_bound_feat3": dropped * 3.0,
        })
    print(json.dumps({
        "technique": "D",
        "cdf_2d_isotropic": "F(r)=1-exp(-r^2/(2 sigma^2))",
        "rows": rows,
        "verdict": "reject 2sigma; 2.5sigma is flag-worthy; 3sigma remains default",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
