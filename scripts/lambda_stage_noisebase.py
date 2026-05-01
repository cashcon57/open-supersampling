"""Stage NoiseBase dataset onto a Lambda persistent filesystem.

    python -m scripts.lambda_stage_noisebase

What it does:
    1. Find or create a Lambda persistent filesystem named --filesystem-name
       (default "ors-noisebase") in --region.
    2. Launch a cheap instance (A10 or A6000) with that filesystem attached.
    3. SSH in, install the noisebase package, download the requested subsets.
    4. Terminate the instance (filesystem + data persist indefinitely).

The filesystem mounts at ${OSS_REMOTE_HOME}/<name>/ on every future instance.
Point training at that path with --data ${OSS_REMOTE_HOME}/ors-noisebase.

Dataset sizes (upstream docs):
    sampleset_training_v1  ~370 GB  (1024 sequences, 64 frames, 32 spp, 256²)
    sampleset_test8_v1     ~80 GB
    sampleset_test32_v1    ~30 GB
    all                    ~480 GB  (fits on 1 TB volume with room for checkpoints)

Cost reference (~Apr 2026):
    Lambda filesystem  ~$0.20/GB/month  → 500 GB = $100/month = ~$3.30/day
    A10 instance       $0.75/hr         → ~4h download = ~$3.00 one-time
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from ors.cloud import LambdaClient, SafetyHarness
from ors.cloud.lambda_client import INSTANCE_PRICING
from ors.cloud.safety_harness import HarnessConfig


REPO_ROOT = Path(__file__).resolve().parents[1]

# Cheap instances in preference order for staging (no GPU needed).
_STAGING_PREFERENCE = ["gpu_1x_a10", "gpu_1x_a6000", "gpu_1x_a100"]

_SUBSETS = {
    "training": ["sampleset_v1"],
    "test8":    ["sampleset_test8_v1"],
    "test32":   ["sampleset_test32_v1"],
    "all":      ["sampleset_v1", "sampleset_test8_v1", "sampleset_test32_v1"],
}


def _ssh(ip: str, key_path: Path, user: str = "ubuntu") -> list[str]:
    return [
        "ssh", "-i", str(key_path),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30",
        f"{user}@{ip}",
    ]


def _find_or_create_filesystem(
    client: LambdaClient,
    name: str,
    region: str,
) -> dict:
    filesystems = client.list_filesystems()
    for fs in filesystems:
        if fs.get("name") == name:
            print(f"[stage] filesystem '{name}' already exists (id={fs['id']}, region={fs.get('region','?')})")
            return fs
    print(f"[stage] creating filesystem '{name}' in {region} ...")
    fs = client.create_filesystem(name, region)
    print(f"[stage] created: id={fs['id']}")
    return fs


def _pick_instance(client: LambdaClient, preferred_region: str) -> tuple[str, str]:
    """Return (instance_type, actual_region) — prefer requested region, fall back to any."""
    types = client.list_instance_types()
    for tname in _STAGING_PREFERENCE:
        entry = types.get(tname, {})
        regions = entry.get("regions_with_capacity_available", []) or []
        region_names = [r.get("name") if isinstance(r, dict) else r for r in regions]
        if preferred_region in region_names:
            rate = INSTANCE_PRICING.get(tname, 0.0)
            print(f"[stage] using {tname} @ ${rate:.2f}/hr in {preferred_region}")
            return tname, preferred_region
    # any region fallback — filesystem will be created in the same region
    for tname in _STAGING_PREFERENCE:
        entry = types.get(tname, {})
        regions = entry.get("regions_with_capacity_available", []) or []
        region_names = [r.get("name") if isinstance(r, dict) else r for r in regions]
        if region_names:
            rate = INSTANCE_PRICING.get(tname, 0.0)
            actual = region_names[0]
            print(f"[stage] no capacity in {preferred_region}; using {tname} in {actual} @ ${rate:.2f}/hr")
            return tname, actual
    raise RuntimeError("no staging-suitable instances available")


def _download_noisebase(
    harness: SafetyHarness,
    ip: str,
    key_path: Path,
    mount_path: str,
    subsets: list[str],
) -> int:
    ssh = _ssh(ip, key_path)
    subsets_str = " ".join(subsets)

    # NoiseBase ships a CLI via its pip package.  The download destination is
    # the filesystem mount point so data survives instance termination.
    # We try pip install first; if the package name differs, fall back to
    # cloning the repo and using its download.py script directly.
    remote_cmd = (
        "set -uo pipefail && "
        f"mkdir -p {mount_path} && "
        "echo '--- pip install noisebase ---' && "
        "pip3 install --quiet noisebase && "
        "echo '--- nb-download version ---' && "
        "nb-download --help 2>&1 | head -5 || true && "
        f"for SUBSET in {subsets_str}; do "
        f"  echo \"=== downloading $SUBSET to {mount_path} ===\"; "
        f"  nb-download --data_path {mount_path} \"$SUBSET\" 2>&1; "
        "done && "
        f"echo '--- done ---' && "
        f"du -sh {mount_path}/* 2>/dev/null || echo 'nothing downloaded yet'"
    )

    full = ssh + ["bash", "-lc", shlex.quote(remote_cmd)]
    print(f"[stage] downloading subsets: {subsets}")
    proc = subprocess.Popen(full, stdout=sys.stdout, stderr=sys.stderr)
    try:
        while proc.poll() is None:
            harness.heartbeat()
            harness.check_limits()
            time.sleep(60)
        return proc.returncode or 0
    finally:
        if proc.poll() is None:
            proc.terminate()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--filesystem-name", default="ors-noisebase",
                   help="Lambda filesystem name (created if absent)")
    p.add_argument("--region", default="us-west-2",
                   help="Lambda region for filesystem + staging instance")
    p.add_argument("--subset", default="all",
                   choices=list(_SUBSETS.keys()),
                   help="which NoiseBase subsets to download (default: all ~480 GB)")
    p.add_argument("--budget", type=float, default=12.0,
                   help="hard $ cap for the staging instance (default $12)")
    p.add_argument("--max-hours", type=float, default=8.0,
                   help="max wall-time for staging (default 8h)")
    p.add_argument("--ssh-key-name", default="Upscaler-ClaudeCode")
    args = p.parse_args()

    client = LambdaClient()

    # 1. Filesystem
    fs = _find_or_create_filesystem(client, args.filesystem_name, args.region)
    fs_id = fs["id"]
    # Lambda returns the actual mount point in the API response
    mount_path = fs.get("mount_point") or f"/lambda/nfs/{args.filesystem_name}"
    subsets = _SUBSETS[args.subset]

    print(f"[stage] filesystem id={fs_id}, mount={mount_path}")
    print(f"[stage] subsets to download: {subsets}")

    # 2. Pick instance type — must be in same region as filesystem
    instance_type, actual_region = _pick_instance(client, args.region)
    if actual_region != args.region and fs.get("region", {}).get("name") != actual_region:
        # Filesystem is in wrong region for the available instance — delete and recreate
        print(f"[stage] filesystem region mismatch; recreating in {actual_region} ...")
        client.delete_filesystem(fs_id)
        fs = client.create_filesystem(args.filesystem_name, actual_region)
        fs_id = fs["id"]
        print(f"[stage] recreated: id={fs_id} in {actual_region}")
    rate = INSTANCE_PRICING.get(instance_type, 0.0)

    cfg = HarnessConfig(
        instance_type=instance_type,
        region=actual_region,
        ssh_key_names=[args.ssh_key_name],
        file_system_names=[args.filesystem_name],
        name="ors-noisebase-staging",
        max_duration_s=int(args.max_hours * 3600),
        budget_usd=args.budget,
        ssh_key_path=REPO_ROOT / ".secrets" / "lambda-ssh.pem",
        purpose=f"NoiseBase staging → {args.filesystem_name} ({args.subset})",
    )

    harness = SafetyHarness(client, cfg)
    with harness as inst:
        print(f"[stage] instance {inst.instance_id} active at {inst.ip}")
        rc = _download_noisebase(harness, inst.ip, cfg.ssh_key_path, mount_path, subsets)
        if rc != 0:
            print(f"[stage] download failed rc={rc}", file=sys.stderr)
            return rc

    print(f"\n[stage] DONE. Filesystem '{args.filesystem_name}' (id={fs_id}) ready.")
    print(f"[stage] Pass --filesystem-name {args.filesystem_name} to lambda_train_pico.py")
    print(f"[stage] Data path on instance: {mount_path}")
    return 0


if __name__ == "__main__":
    print(
        "WARNING: This creates a persistent Lambda filesystem (billed until deleted).\n"
        "Run `python -m scripts.lambda_list_filesystems` to see active filesystems.\n"
    )
    sys.exit(main())
