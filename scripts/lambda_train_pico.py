"""End-to-end Lambda launcher for ORU-Pico training.

    python -m scripts.lambda_train_pico --epochs 50 --budget 8

What it does, in order:
    1. Pick the optimal available instance type (fastest first, biased toward
       time savings — see `select_optimal_instance` in lambda_client.py).
    2. Print a cost preview + estimated wall-time for the chosen tier.
    3. Wrap the launch in `SafetyHarness` which will:
         - require interactive 'launch' confirmation (cost preview again)
         - terminate at budget cap (default $20)
         - terminate at max duration (default 6h)
         - terminate on idle (<5% GPU util for 15min)
         - terminate on signal / atexit / watchdog stale heartbeat
    4. SSH into the instance, rsync the repo, run the training command.
    5. Stream training logs back; heartbeat to the watchdog every minute.
    6. On any exit path (success, failure, ctrl-C, kill -9), terminate the
       instance and verify it's gone.

NEVER call `LambdaClient.launch()` directly outside of SafetyHarness.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from ors.cloud import LambdaClient, SafetyHarness
from ors.cloud.lambda_client import (
    INSTANCE_PRICING,
    INSTANCE_EFFECTIVE_FP16_TFLOPS,
    SINGLE_GPU_PREFERENCE_ORDER,
    select_optimal_instance,
)
from ors.cloud.safety_harness import HarnessConfig


# Estimated total compute for ORU-Pico training (~250K params, 30K sequences,
# seq_len=8, batch 8, ~100 epochs). Used for instance selection only — actual
# wall-time and cost are tracked live by the harness.
PICO_WORKLOAD_TFLOPS = 200.0  # rough estimate of total FP16 compute needed

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ssh_command(ip: str, key_path: Path, user: str = "ubuntu") -> list[str]:
    return [
        "ssh",
        "-i", str(key_path),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=20",
        f"{user}@{ip}",
    ]


def _rsync_repo(ip: str, key_path: Path, user: str = "ubuntu") -> int:
    """Push the repo (excluding venv/data/results/secrets) to the instance."""
    cmd = [
        "rsync", "-az", "--delete",
        "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
        "--exclude", "venv*/",
        "--exclude", "data/",
        "--exclude", "results/",
        "--exclude", ".secrets/",
        "--exclude", ".git/",
        "--exclude", "__pycache__/",
        "--exclude", "*.pth",
        f"{REPO_ROOT}/",
        f"{user}@{ip}:~/ors/",
    ]
    print(f"[lambda_train_pico] rsync repo to {ip}:~/ors/ ...")
    return subprocess.run(cmd).returncode


def _run_training(harness: SafetyHarness, ip: str, key_path: Path, args) -> int:
    """SSH in, install deps, run training. Heartbeat regularly."""
    ssh = _ssh_command(ip, key_path)

    # Install + run as a single shell pipeline; heartbeat by polling exit code.
    remote_cmd = (
        "cd ~/ors && "
        "python3 -m venv venv-cloud --upgrade-deps && "
        "source venv-cloud/bin/activate && "
        "pip install -e .[dev] 2>&1 | tail -20 && "
        f"python -m ors.train.train_pico "
        f"--data ${OSS_REMOTE_HOME}/noisebase "
        f"--out results/pico-cloud "
        f"--epochs {args.epochs} "
        f"--sequence-length {args.sequence_length} "
        f"--batch-size {args.batch_size} "
        f"--scale-factor {args.scale_factor}"
    )
    full = ssh + ["bash", "-lc", shlex.quote(remote_cmd)]

    print(f"[lambda_train_pico] starting training:\n  {remote_cmd}\n")
    proc = subprocess.Popen(full, stdout=sys.stdout, stderr=sys.stderr)
    try:
        while proc.poll() is None:
            harness.heartbeat()
            harness.check_limits()
            time.sleep(30)
        return proc.returncode or 0
    finally:
        if proc.poll() is None:
            proc.terminate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--sequence-length", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--scale-factor", type=float, default=2.0)
    p.add_argument("--budget", type=float, default=10.0,
                   help="hard $-cap for this run (auto-terminate if exceeded)")
    p.add_argument("--max-hours", type=float, default=4.0,
                   help="hard wall-time cap in hours")
    p.add_argument("--purpose", type=str, default="ORU-Pico training",
                   help="short description shown in the approval preview")
    p.add_argument("--ssh-key-name", type=str, default="Upscaler-ClaudeCode",
                   help="name of the SSH key registered in Lambda dashboard")
    p.add_argument("--instance-type", type=str, default=None,
                   help="override auto-select; e.g. 'gpu_1x_h100_pcie'")
    p.add_argument("--region", type=str, default=None,
                   help="override auto-select; required if --instance-type is set")
    args = p.parse_args()

    client = LambdaClient()

    # 1. Pick instance
    if args.instance_type:
        if not args.region:
            print("error: --region required when --instance-type is set", file=sys.stderr)
            return 2
        instance_type, region = args.instance_type, args.region
        print(f"[lambda_train_pico] using user-specified {instance_type} in {region}")
    else:
        try:
            instance_type, region = select_optimal_instance(
                client,
                preference=SINGLE_GPU_PREFERENCE_ORDER,
                workload_tflops=PICO_WORKLOAD_TFLOPS,
                max_extra_per_run_usd=5.0,
            )
        except RuntimeError as e:
            print(f"[lambda_train_pico] {e}", file=sys.stderr)
            return 3
        rate = INSTANCE_PRICING.get(instance_type, 0.0)
        eff = INSTANCE_EFFECTIVE_FP16_TFLOPS.get(instance_type, 30.0)
        wall_s = PICO_WORKLOAD_TFLOPS / max(eff, 1e-6)
        est_cost = (wall_s / 3600.0) * rate
        print(f"[lambda_train_pico] auto-selected {instance_type} in {region}")
        print(f"  estimated wall time : ~{wall_s/60:.1f} minutes")
        print(f"  estimated cost      : ~${est_cost:.2f}")
        print(f"  hourly rate         : ${rate:.2f}/hr")

    # 2. Build HarnessConfig — pre-launch approval will print full preview + prompt
    cfg = HarnessConfig(
        instance_type=instance_type,
        region=region,
        ssh_key_names=[args.ssh_key_name],
        name="ors-pico-training",
        max_duration_s=int(args.max_hours * 3600),
        budget_usd=args.budget,
        ssh_key_path=REPO_ROOT / ".secrets" / "lambda-ssh.pem",
        purpose=args.purpose,
    )

    # 3. Launch + train. Keep harness ref so the training loop can heartbeat
    # and check limits during the long-running SSH command.
    harness = SafetyHarness(client, cfg)
    with harness as inst:
        print(f"[lambda_train_pico] instance {inst.instance_id} active at {inst.ip}")

        rc = _rsync_repo(inst.ip, cfg.ssh_key_path)
        if rc != 0:
            print(
                f"[lambda_train_pico] rsync failed (rc={rc}); aborting "
                f"(instance will be terminated by the harness)",
                file=sys.stderr,
            )
            return rc

        rc = _run_training(harness, inst.ip, cfg.ssh_key_path, args)
        if rc != 0:
            print(f"[lambda_train_pico] training exited rc={rc}", file=sys.stderr)
            return rc

        # Pull the trained checkpoint back before the harness terminates.
        ckpt_remote = "~/ors/results/pico-cloud/oru_pico.pth"
        ckpt_local = REPO_ROOT / "results" / "pico-cloud" / "oru_pico.pth"
        ckpt_local.parent.mkdir(parents=True, exist_ok=True)
        scp_cmd = [
            "scp",
            "-i", str(cfg.ssh_key_path),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"ubuntu@{inst.ip}:{ckpt_remote}",
            str(ckpt_local),
        ]
        print(f"[lambda_train_pico] pulling checkpoint to {ckpt_local}")
        subprocess.run(scp_cmd, check=False)

    return 0


if __name__ == "__main__":
    print(
        "WARNING: This launcher is the wrapper around real Lambda spend. Read it\n"
        "before running. The SafetyHarness will print a cost preview and require\n"
        "interactive approval before any instance is created.\n"
    )
    sys.exit(main())
