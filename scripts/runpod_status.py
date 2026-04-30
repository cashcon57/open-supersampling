"""List currently-active RunPod pods + cumulative cost.

    python -m scripts.runpod_status

Reports every running pod under your API key, the elapsed wall-time, and
the running cost. Useful for verifying nothing is orphaned.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

from ors.cloud import RunPodClient


def main():
    client = RunPodClient(live_pricing=True)
    instances = client.list_instances()
    if not instances:
        print("No RunPod pods active.")
        return 0

    print(f"{'POD_ID':<24} {'GPU':<32} {'STATUS':<12} {'IP':<16} {'$/HR':<6} {'ELAPSED':<10} {'COST_USD':<10}")
    total_cost = 0.0
    for inst in instances:
        rate = client.hourly_rate(inst.instance_type)
        elapsed_str = "?"
        cost = 0.0
        if inst.launched_at:
            try:
                ms = int(inst.launched_at)
                t0 = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
                elapsed_s = (datetime.now(timezone.utc) - t0).total_seconds()
                hours = elapsed_s / 3600.0
                elapsed_str = f"{hours:.2f}h"
                cost = hours * rate
                total_cost += cost
            except Exception:
                pass
        print(
            f"{inst.instance_id:<24} "
            f"{(inst.instance_type or '-')[:32]:<32} "
            f"{inst.status:<12} "
            f"{(inst.ip or '-'):<16} "
            f"${rate:<5.2f} "
            f"{elapsed_str:<10} "
            f"${cost:<9.2f}"
        )
    print(f"\nTotal running cost: ${total_cost:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
