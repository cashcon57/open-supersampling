#!/usr/bin/env python3
"""Technique M: static audit findings recorded for Phase 4 report."""
from __future__ import annotations

import json


FINDINGS = [
    ("M1", "topk_norm passed through Python/native binding, then discarded in native CUDA forward"),
    ("M2", "forward preprocess writes aabb/pair_count scratch that full-frame forward ignores"),
    ("M3", "deterministic full-frame gid/tile_offsets buffers can be derived in-kernel"),
    ("M4", "hot-path buffers are zero-filled before every element is overwritten"),
    ("M5", "raster weight loop recomputes row-only pixel coordinates for each column"),
    ("M6", "backward recomputes dx2/dxdy/dy2 instead of CSE"),
    ("M7", "forward conic preprocess repeats c*c, s*s, c*s while backward already CSEs"),
    ("M8", "d_rot expression can be factored around diff and two_cs"),
    ("M9", "canvas warp samples Jacobians for Gaussians filtered out afterward"),
    ("M10", "identity active-mask view matrix allocation can be replaced by None"),
    ("M11", "ST update allocates/multiplies all-ones transmittance"),
    ("M12", "shape/device constants are recomputed instead of cached"),
]


def main() -> None:
    print(json.dumps({
        "technique": "M",
        "finding_count": len(FINDINGS),
        "findings": [{"id": k, "summary": v} for k, v in FINDINGS],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
