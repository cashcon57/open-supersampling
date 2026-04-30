"""Emergency kill switch — terminate every Lambda instance under this API key.

    python -m scripts.lambda_terminate_all          # interactive confirm
    python -m scripts.lambda_terminate_all --force  # no prompt

Use this when:
- A SafetyHarness exit failed
- You want to verify nothing is orphaned (lists instances first)
- You see unexpected charges and need a panic button
"""
from __future__ import annotations

import argparse
import sys

from ors.cloud import LambdaClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="skip confirmation prompt")
    args = p.parse_args()

    client = LambdaClient()
    instances = client.list_instances()
    if not instances:
        print("No active instances. Nothing to terminate.")
        return 0

    print(f"Found {len(instances)} active instance(s):")
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
    if remaining:
        print(f"\nWARNING: {len(remaining)} instance(s) still showing as active. Retry shortly.")
        return 2
    print("\nAll instances terminated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
