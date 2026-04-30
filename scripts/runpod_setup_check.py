"""Verify RunPod setup before any launch.

    python -m scripts.runpod_setup_check

Runs read-only API calls and reports:
- API key resolves OK
- Available H100 / A100 / 4090 GPU types with current capacity (community + secure)
- Whether `.secrets/runpod-api-key.txt` is present and 0600
- Whether any pods are currently active (and their cost)

Run this BEFORE attempting any training launch to confirm everything is wired.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from ors.cloud import RunPodClient


# GPUs we actually care about for ORU-Pico training. Order = preference.
_RELEVANT = [
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 NVL",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
]


def main():
    print("=== RunPod setup check ===\n")

    # 1. API key
    try:
        client = RunPodClient()
        print(f"✓ API key resolved (length {len(client._api_key)})")
    except Exception as e:
        print(f"✗ API key failed: {e}")
        return 1

    # 2. Local key file permissions
    repo_root = Path(__file__).resolve().parents[1]
    key_path = repo_root / ".secrets" / "runpod-api-key.txt"
    if key_path.exists():
        mode = oct(key_path.stat().st_mode & 0o777)
        if mode != "0o600":
            print(f"⚠ Local API key file permissions are {mode}, expected 0o600. Run:")
            print(f"    chmod 600 {key_path}")
        else:
            print(f"✓ Local API key file at .secrets/runpod-api-key.txt (mode {mode})")
    else:
        print(f"  (no local key file at {key_path}; using env var instead)")

    # 3. GPU availability + pricing
    print()
    try:
        gpus = client.list_gpus()
        gpus_by_id = {g["id"]: g for g in gpus if "id" in g}
        print(f"Relevant GPU availability ({client._cloud_type} cloud, "
              f"{'spot' if client._spot else 'on-demand'}):")
        for gid in _RELEVANT:
            g = gpus_by_id.get(gid)
            if g is None:
                print(f"  — {gid:<32}  not in catalog")
                continue
            secure = g.get("secureCloud")
            community = g.get("communityCloud")
            sp = g.get("securePrice") or 0.0
            cp = g.get("communityPrice") or 0.0
            avail_marks = []
            if secure:
                avail_marks.append(f"secure ${sp:.2f}/hr")
            if community:
                avail_marks.append(f"community ${cp:.2f}/hr")
            mark = "✓" if avail_marks else "—"
            availability = ", ".join(avail_marks) if avail_marks else "NONE LISTED"
            canonical = client.canonicalize_gpu_id(gid) or "—"
            print(f"  {mark} {gid:<32}  {availability:<40}  canonical={canonical}")
    except Exception as e:
        print(f"✗ Failed to list GPU types: {e}")
        return 4

    # 4. Active pods (anything currently billing?)
    print()
    try:
        instances = client.list_instances()
        if instances:
            print(f"⚠ {len(instances)} pod(s) currently active (still billing):")
            total_cost = 0.0
            for inst in instances:
                rate = client.hourly_rate(inst.instance_type)
                elapsed_str = "?"
                cost = 0.0
                if inst.launched_at:
                    try:
                        # `lastStatusChange` is millis since epoch in RunPod's schema.
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
                    f"    - {inst.instance_id}  {inst.instance_type:<32}  "
                    f"status={inst.status:<11}  ${rate}/hr  elapsed={elapsed_str}  cost=${cost:.2f}"
                )
            print(f"\n  Total running cost: ${total_cost:.2f}")
            print("\n  → Run `python -m scripts.runpod_status` for a focused report; or")
            print("    `python -m scripts.runpod_terminate_all` to kill them.")
        else:
            print("✓ No pods currently active.")
    except Exception as e:
        print(f"✗ Failed to list pods: {e}")
        return 5

    print("\nSetup check complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
