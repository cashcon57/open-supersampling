"""Parse nvidia-smi --query-gpu CSV output into the dashboard's gpu_status.json.

Usage:
  python3 scripts/parse_nvidia_smi_to_json.py <csv_in> <json_out> <captured_at_iso>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <csv_in> <json_out> <captured_at_iso>", file=sys.stderr)
        return 2

    csv_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    captured_at = sys.argv[3]

    rows = list(csv.reader(csv_path.read_text(encoding="utf-8", errors="replace").splitlines()))
    if not rows or len(rows[0]) < 4:
        print("nvidia-smi returned no GPU rows", file=sys.stderr)
        return 3

    name = rows[0][0].strip()
    used = int(float(rows[0][1].strip()))
    total = int(float(rows[0][2].strip()))
    util = int(float(rows[0][3].strip()))

    payload = {
        "captured_at": captured_at,
        "gpu_name": name,
        "memory_used_mib": used,
        "memory_total_mib": total,
        "memory_used_pct": round((used / total) * 100, 1) if total else 0.0,
        "utilization_pct": util,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
