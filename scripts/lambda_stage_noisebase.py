"""Stage NoiseBase dataset to Cloudflare R2 via a Lambda instance.

    python -m scripts.lambda_stage_noisebase

What it does:
    1. Launch a cheap Lambda instance (A10 or A6000) — no filesystem needed.
    2. SSH in, install noisebase + rclone, download each subset to /tmp.
    3. rclone sync each subset to R2 bucket ors-noisebase/ with 32 parallel transfers.
    4. Terminate the instance.  R2 data persists.

Lambda has no egress fees → R2 upload is free.
R2 has no ingress fees.
Training pulls R2 → Lambda filesystem at ~1 GB/s (sub-minute warm-up).

Dataset sizes (upstream docs):
    sampleset_v1            ~370 GB  (1024 sequences, 64 frames, 32 spp, 256²)
    sampleset_test8_v1      ~80 GB
    sampleset_test32_v1     ~30 GB
    all                     ~480 GB

Cost reference (~Apr 2026):
    A10 instance  $0.75/hr  →  ~4h download+upload ≈ $3.00 one-time
    R2 storage    ~$0.015/GB/month  →  500 GB ≈ $7.50/month
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


def _load_r2_creds() -> dict[str, str]:
    creds_path = Path(__file__).resolve().parents[1] / ".secrets" / "r2-credentials.env"
    creds: dict[str, str] = {}
    for line in creds_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip()
    return creds


def _download_noisebase_to_r2(
    harness: SafetyHarness,
    ip: str,
    key_path: Path,
    subsets: list[str],
) -> int:
    """Download NoiseBase subsets to Cloudflare R2 via rclone.

    Flow on instance:
      nb-download → /tmp/noisebase/<subset>/  →  rclone sync → R2:ors-noisebase/<subset>/

    Lambda has no egress fees so upload to R2 is free.
    R2 has no ingress fees.
    Subsequent training pulls R2 → Lambda filesystem at ~1 GB/s.
    """
    ssh = _ssh(ip, key_path)
    subsets_str = " ".join(subsets)
    creds = _load_r2_creds()
    r2_access = creds["R2_ACCESS_KEY_ID"]
    r2_secret = creds["R2_SECRET_ACCESS_KEY"]
    r2_endpoint = creds["R2_ENDPOINT"]
    r2_bucket = creds["R2_BUCKET"]

    remote_cmd = (
        "set -euo pipefail && "
        # numpy<2.0: nb-download's pyfvvdp dep uses removed numpy.lib.shape_base
        "echo '--- installing tools ---' && "
        "pip3 install --quiet 'numpy<2.0' noisebase && "
        "curl -fsSL https://rclone.org/install.sh | sudo bash 2>&1 | tail -3 && "
        # Configure rclone R2 remote
        "rclone config create r2 s3 "
        f"  provider=Cloudflare "
        f"  access_key_id={r2_access} "
        f"  secret_access_key={r2_secret} "
        f"  endpoint={r2_endpoint} "
        "  no_check_bucket=true 2>&1 | tail -3 && "
        "echo '--- tools ready ---' && "
        f"for SUBSET in {subsets_str}; do "
        f"  echo \"=== nb-download $SUBSET ===\"; "
        f"  mkdir -p /tmp/noisebase && "
        f"  nb-download --data_path /tmp/noisebase \"$SUBSET\" && "
        f"  echo \"=== rclone sync $SUBSET -> r2:{r2_bucket}/$SUBSET ===\"; "
        f"  rclone sync /tmp/noisebase/$SUBSET r2:{r2_bucket}/$SUBSET "
        f"    --transfers=32 --checkers=16 --progress && "
        f"  rm -rf /tmp/noisebase/$SUBSET && "
        f"  echo \"=== $SUBSET done ===\"; "
        "done && "
        f"echo '--- all subsets uploaded to R2 ---' && "
        f"rclone ls r2:{r2_bucket} | tail -5"
    )

    full = ssh + ["bash", "-lc", shlex.quote(remote_cmd)]
    print(f"[stage] downloading {subsets} → R2 bucket {r2_bucket}")
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
    p.add_argument("--region", default="us-east-1",
                   help="Lambda region for staging instance (default: us-east-1)")
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
    subsets = _SUBSETS[args.subset]

    print(f"[stage] subsets to upload to R2: {subsets}")

    instance_type, actual_region = _pick_instance(client, args.region)

    cfg = HarnessConfig(
        instance_type=instance_type,
        region=actual_region,
        ssh_key_names=[args.ssh_key_name],
        name="ors-noisebase-r2-staging",
        max_duration_s=int(args.max_hours * 3600),
        budget_usd=args.budget,
        ssh_key_path=REPO_ROOT / ".secrets" / "lambda-ssh.pem",
        purpose=f"NoiseBase → R2 staging ({args.subset})",
    )

    harness = SafetyHarness(client, cfg)
    with harness as inst:
        print(f"[stage] instance {inst.instance_id} active at {inst.ip}")
        rc = _download_noisebase_to_r2(harness, inst.ip, cfg.ssh_key_path, subsets)
        if rc != 0:
            print(f"[stage] upload failed rc={rc}", file=sys.stderr)
            return rc

    creds = _load_r2_creds()
    print(f"\n[stage] DONE. NoiseBase uploaded to R2 bucket '{creds['R2_BUCKET']}'.")
    print(f"[stage] Pull to Lambda filesystem with:")
    print(f"[stage]   rclone sync r2:{creds['R2_BUCKET']} /lambda/nfs/ors-noisebase --transfers=32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
