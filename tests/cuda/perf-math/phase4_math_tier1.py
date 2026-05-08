#!/usr/bin/env python3
"""Run the Phase 4 Tier-1 math scripts with the report's stated domains."""
from __future__ import annotations

import runpy
from pathlib import Path


SCRIPTS = [
    "a_pade_exp.py",
    "b_separable_gaussian.py",
    "c_lut_exp.py",
    "d_sigma_cull.py",
    "e_far_field_skip.py",
    "k_quantized_state.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    for name in SCRIPTS:
        print(f"\n# {name}")
        runpy.run_path(str(here / name), run_name="__main__")


if __name__ == "__main__":
    main()
