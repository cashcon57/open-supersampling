#!/usr/bin/env python3
"""Technique C: LUT linear interpolation bound for exp(-0.5 q)."""
from __future__ import annotations

import json
import numpy as np


def main() -> None:
    entries = 256
    q0, q1 = 0.0, 9.0
    q_nodes = np.linspace(q0, q1, entries, dtype=np.float64)
    y_nodes = np.exp(-0.5 * q_nodes)
    dq = float(q_nodes[1] - q_nodes[0])
    q = np.linspace(q0, q1, 1_000_001, dtype=np.float64)
    idx = np.minimum(((q - q0) / dq).astype(np.int64), entries - 2)
    t = (q - q_nodes[idx]) / dq
    interp = y_nodes[idx] * (1.0 - t) + y_nodes[idx + 1] * t
    truth = np.exp(-0.5 * q)
    err = np.abs(interp - truth)
    i = int(np.argmax(err))
    print(json.dumps({
        "technique": "C",
        "entries": entries,
        "storage_fp16_bytes": entries * 2,
        "delta_q": dq,
        "analytical_bound": (dq * dq / 8.0) * 0.25,
        "measured_max_abs_error": float(err[i]),
        "argmax_q": float(q[i]),
        "verdict": "ship if LUT is resident in constant/shared memory",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
