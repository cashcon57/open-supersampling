"""List currently-active Lambda instances + cumulative cost.

    python -m scripts.lambda_status

Reports every running instance under your API key, the elapsed wall-time,
and the running cost. Useful for verifying nothing is orphaned.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

from oss.cloud import LambdaClient, INSTANCE_PRICING


def main():
    client = LambdaClient()
    instances = client.list_instances()
    if not instances:
        print("No Lambda instances active.")
        return 0

    print(f"{'INSTANCE_ID':<32} {'TYPE':<22} {'STATUS':<12} {'IP':<16} {'$/HR':<6} {'ELAPSED':<10} {'COST_USD':<10}")
    total_cost = 0.0
    for inst in instances:
        rate = INSTANCE_PRICING.get(inst.instance_type, 0.0)
        elapsed_str = "?"
        cost = 0.0
        if inst.launched_at:
            try:
                t0 = datetime.fromisoformat(inst.launched_at.replace("Z", "+00:00"))
                elapsed_s = (datetime.now(timezone.utc) - t0).total_seconds()
                hours = elapsed_s / 3600.0
                elapsed_str = f"{hours:.2f}h"
                cost = hours * rate
                total_cost += cost
            except Exception:
                pass
        print(
            f"{inst.instance_id:<32} "
            f"{inst.instance_type:<22} "
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
