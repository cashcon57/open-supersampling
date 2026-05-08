#!/usr/bin/env python3
"""Technique E: q > 12 far-field contribution bound."""
from __future__ import annotations

import json
import math


def main() -> None:
    weight = math.exp(-6.0)
    print(json.dumps({
        "technique": "E",
        "q_cut": 12.0,
        "weight_at_cut": weight,
        "per_gaussian_linf_bound_feat3": 3.0 * weight,
        "mass_outside_radius_sqrt12": weight,
        "note": "Not bit-exact if contributions are zeroed. It is redundant when a 3sigma AABB (q<=9 for isotropic axis-aligned support) is already enforced, but useful for the current full-frame native CUDA path.",
        "verdict": "ship-with-flag or pair with 3sigma pair construction",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
