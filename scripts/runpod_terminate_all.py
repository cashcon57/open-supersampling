"""Emergency kill switch — terminate every RunPod pod under this API key.

    python -m scripts.runpod_terminate_all          # interactive confirm
    python -m scripts.runpod_terminate_all --force  # no prompt

Use this when:
- A SafetyHarness exit failed
- You want to verify nothing is orphaned (lists pods first)
- You see unexpected charges and need a panic button
"""
from __future__ import annotations

import argparse
import sys

from oss.cloud import RunPodClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="skip confirmation prompt")
    args = p.parse_args()

    client = RunPodClient(live_pricing=False)
    instances = client.list_instances()
    if not instances:
        print("No active pods. Nothing to terminate.")
        return 0

    print(f"Found {len(instances)} active pod(s):")
    for inst in instances:
        print(f"  {inst.instance_id}  ({inst.instance_type}, status={inst.status}, ip={inst.ip or '-'})")

    if not args.force:
        try:
            answer = input("\nTerminate ALL of these? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if answer != "y":
            print("Aborted.")
            return 1

    ids = [i.instance_id for i in instances]
    result = client.terminate(ids)
    print(f"\nTerminate response: {result}")

    # Verify
    remaining = client.list_instances()
    still_active = [i for i in remaining if i.status not in ("terminated", "terminating")]
    if still_active:
        print(f"\nWARNING: {len(still_active)} pod(s) still showing as active. Retry shortly.")
        return 2
    print("\nAll pods terminated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
