"""Verify Lambda setup before any launch.

    python -m scripts.lambda_setup_check

Runs read-only API calls and reports:
- API key resolves OK
- SSH keys registered in Lambda UI
- Available instance types in each region
- Whether `.secrets/lambda-ssh.pem` exists

Run this BEFORE attempting any training launch to confirm everything is wired.
"""
from __future__ import annotations

import sys
from pathlib import Path

from oss.cloud import LambdaClient, INSTANCE_PRICING


def main():
    print("=== Lambda setup check ===\n")

    # 1. API key
    try:
        client = LambdaClient()
        print(f"✓ API key resolved (length {len(client._api_key)})")
    except Exception as e:
        print(f"✗ API key failed: {e}")
        return 1

    # 2. SSH keys in Lambda
    try:
        keys = client.list_ssh_keys()
        if not keys:
            print("✗ No SSH keys registered in Lambda. Add one in the Lambda dashboard.")
            return 2
        print(f"✓ {len(keys)} SSH key(s) registered:")
        for k in keys:
            name = k.get("name", "?")
            pubkey = k.get("public_key", "")
            preview = (pubkey[:20] + "..." + pubkey[-20:]) if len(pubkey) > 50 else pubkey
            print(f"    - {name!r}  ({preview})")
    except Exception as e:
        print(f"✗ Failed to list SSH keys: {e}")
        return 3

    # 3. Local private key file
    repo_root = Path(__file__).resolve().parents[1]
    pem_path = repo_root / ".secrets" / "lambda-ssh.pem"
    if pem_path.exists():
        mode = oct(pem_path.stat().st_mode & 0o777)
        size = pem_path.stat().st_size
        if mode != "0o600":
            print(f"⚠ Local SSH private key permissions are {mode}, expected 0o600. Run:")
            print(f"    chmod 600 {pem_path}")
        else:
            print(f"✓ Local SSH private key at .secrets/lambda-ssh.pem ({size} bytes, mode {mode})")
    else:
        print(f"✗ Missing .secrets/lambda-ssh.pem. SSH-into-instance idle detection will not work.")

    # 4. Instance types + region availability
    print()
    try:
        types = client.list_instance_types()
        relevant = ["gpu_1x_a100", "gpu_1x_a6000", "gpu_1x_a10", "gpu_1x_h100_pcie", "gpu_1x_h100_sxm5"]
        print("Relevant instance availability:")
        for tname in relevant:
            entry = types.get(tname, {})
            regions = entry.get("regions_with_capacity_available", []) or entry.get("regions", [])
            region_names = [r.get("name", r) if isinstance(r, dict) else r for r in regions]
            price = INSTANCE_PRICING.get(tname, "?")
            mark = "✓" if region_names else "—"
            print(f"  {mark} {tname:<22} (${price}/hr)  regions: {region_names if region_names else 'NONE AVAILABLE'}")
    except Exception as e:
        print(f"✗ Failed to list instance types: {e}")
        return 4

    # 5. Active instances (anything currently billing?)
    print()
    try:
        instances = client.list_instances()
        if instances:
            print(f"⚠ {len(instances)} instance(s) currently active (still billing):")
            for inst in instances:
                rate = INSTANCE_PRICING.get(inst.instance_type, 0.0)
                print(f"    - {inst.instance_id}  {inst.instance_type}  status={inst.status}  ${rate}/hr")
            print("\n  → Run `python -m scripts.lambda_status` for cost; or")
            print("    `python -m scripts.lambda_terminate_all` to kill them.")
        else:
            print("✓ No instances currently active.")
    except Exception as e:
        print(f"✗ Failed to list instances: {e}")
        return 5

    print("\nSetup check complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
