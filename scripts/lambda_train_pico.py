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

# Realistic per-instance recommended max epochs that fit within --max-hours
# given the model + dataset scale. Keeps us from booking 50 epochs on an
# instance that can only finish 25 in the time we cap.
INSTANCE_EPOCH_CAPS_AT_4H: dict[str, int] = {
    "gpu_1x_a10":       25,   # ~3.5h for 25 epochs at our scale
    "gpu_1x_a6000":     35,
    "gpu_1x_a100":      50,
    "gpu_1x_a100_sxm4": 60,
    "gpu_1x_h100_pcie": 100,
    "gpu_1x_h100_sxm5": 120,
}


def _recommended_epochs(instance_type: str, requested: int, max_hours: float) -> int:
    """Cap epochs to what the chosen instance can realistically finish in
    `max_hours`, scaled from the 4h reference table. Always honors `requested`
    if it's already smaller."""
    base = INSTANCE_EPOCH_CAPS_AT_4H.get(instance_type, 25)
    scaled = int(base * (max_hours / 4.0))
    return min(requested, max(1, scaled))


def _expand_wait_target(short: str) -> list[str]:
    """Map shorthand --wait-for value to the list of acceptable instance type names."""
    mapping = {
        "h100":       ["gpu_1x_h100_sxm5", "gpu_1x_h100_pcie"],  # accept either flavor
        "h100_sxm5":  ["gpu_1x_h100_sxm5"],
        "h100_pcie":  ["gpu_1x_h100_pcie"],
        "a100":       ["gpu_1x_a100_sxm4", "gpu_1x_a100"],
        "a100_sxm4":  ["gpu_1x_a100_sxm4"],
    }
    return mapping[short]


def _wait_for_capacity(
    client: LambdaClient,
    targets: list[str],
    max_minutes: int = 180,
    poll_interval_s: int = 60,
) -> tuple[Optional[str], Optional[str]]:
    """Poll Lambda's instance-types endpoint until any of `targets` has capacity.

    Returns (instance_type, region) when capacity is found, or (None, None)
    on timeout. Prints progress every poll. Ctrl-C to cancel.
    """
    print(f"[lambda_train_pico] polling for capacity in {targets} (max {max_minutes}min) ...")
    started = time.time()
    deadline = started + max_minutes * 60
    poll_n = 0
    while time.time() < deadline:
        poll_n += 1
        try:
            types = client.list_instance_types()
            for tname in targets:
                entry = types.get(tname, {})
                regions = entry.get("regions_with_capacity_available", []) or []
                region_names = [r.get("name") if isinstance(r, dict) else r for r in regions]
                if region_names:
                    elapsed_min = (time.time() - started) / 60.0
                    print(
                        f"[lambda_train_pico] ✓ {tname} now available in {region_names[0]} "
                        f"after {elapsed_min:.1f}min ({poll_n} polls)"
                    )
                    return tname, region_names[0]
            elapsed_min = (time.time() - started) / 60.0
            remaining_min = (deadline - time.time()) / 60.0
            print(
                f"[lambda_train_pico] poll {poll_n}: still no capacity for {targets}. "
                f"Elapsed {elapsed_min:.1f}min, remaining {remaining_min:.0f}min. "
                f"Ctrl-C to abort.",
                flush=True,
            )
        except Exception as e:
            sys.stderr.write(f"[lambda_train_pico] poll {poll_n} failed: {e}\n")
        try:
            time.sleep(poll_interval_s)
        except KeyboardInterrupt:
            print("\n[lambda_train_pico] poll cancelled by user.")
            return None, None
    return None, None


def _ssh_command(ip: str, key_path: Path, user: str = "ubuntu") -> list[str]:
    return [
        "ssh",
        "-i", str(key_path),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=20",
        f"{user}@{ip}",
    ]


def _install_guest_agent(ip: str, key_path: Path, user: str = "ubuntu") -> int:
    """Install Lambda's Guest Agent for GPU/VRAM/system metrics in the dashboard.

    Per https://docs.lambda.ai/public-cloud/guest-agent/. Idempotent: safe to
    re-run; the install script handles existing installs.
    """
    ssh = _ssh_command(ip, key_path, user)
    cmd = (
        "curl -fsSL https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh "
        "| sudo bash 2>&1 | tail -20 && "
        "sudo systemctl --no-pager status 'lambda-guest-agent*' 2>&1 | head -20 || "
        "echo '[lambda_train_pico] guest-agent status check failed (non-fatal)'"
    )
    print(f"[lambda_train_pico] installing Lambda Guest Agent on {ip} ...")
    rc = subprocess.run(ssh + ["bash", "-c", cmd]).returncode
    if rc != 0:
        print(
            "[lambda_train_pico] Guest Agent install returned non-zero (non-fatal); "
            "metrics dashboard may be unavailable for this run.",
            file=sys.stderr,
        )
    return 0  # never block training on guest-agent install


def _rsync_repo(ip: str, key_path: Path, user: str = "ubuntu") -> int:
    """Push the repo (excluding venv/data/results/secrets) to the instance."""
    # IMPORTANT: rsync `--exclude foo/` matches `foo/` AT ANY DEPTH. Anchoring
    # with `/foo/` matches only the top-level directory. We need this for
    # `data/` because `ors/data/` is a real Python package we MUST ship.
    # Same for `results/` — keep it anchored to root only.
    cmd = [
        "rsync", "-az", "--delete",
        "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
        "--exclude", "venv*/",
        "--exclude", "/data/",            # top-level dataset dir only — NOT ors/data/
        "--exclude", "/results/",
        "--exclude", "/.secrets/",
        "--exclude", "/.git/",
        "--exclude", "__pycache__/",
        "--exclude", "*.pth",
        f"{REPO_ROOT}/",
        f"{user}@{ip}:~/ors/",
    ]
    print(f"[lambda_train_pico] rsync repo to {ip}:~/ors/ ...")
    return subprocess.run(cmd).returncode


def _provision_noisebase(harness: "SafetyHarness", ip: str, key_path: Path, user: str = "ubuntu") -> int:
    """Clone NoiseBase repo on the instance + fetch a sample sequence.

    NoiseBase data is hosted separately from the github repo (Zenodo / mirror).
    The repo ships utilities to fetch sample data.

    For our smoke / first-real run we don't need the full ~370 GB; a few
    sample sequences suffice. The user can scale up later.
    """
    ssh = _ssh_command(ip, key_path, user)
    # Clone repo + try to fetch sample data via its own scripts. If the repo's
    # own download tooling fails, we fall back to a minimal manual fetch and
    # report so the user can intervene.
    remote = (
        "set -euo pipefail && "
        "if [ ! -d ~/noisebase ]; then "
        "  git clone --depth 1 https://github.com/balintio/noisebase ~/noisebase || exit 11; "
        "fi && "
        "mkdir -p ~/data && "
        "cd ~/noisebase && "
        # Attempt to use the repo's own download utility if present
        "if [ -f download.py ]; then "
        "  python3 download.py --out ~/data/noisebase --subset sample 2>&1 | tail -20 || true; "
        "fi && "
        # Fallback: pull a tiny set of test data shipped with the repo, if any
        "if [ -d test-data ]; then "
        "  cp -r test-data ~/data/noisebase-test 2>/dev/null || true; "
        "fi && "
        # Report what landed
        "echo '---noisebase contents---' && "
        "(ls -la ~/data/ 2>&1 | head -20) && "
        "(du -sh ~/data/* 2>/dev/null || echo 'empty')"
    )
    print(f"[lambda_train_pico] provisioning NoiseBase on instance ...")
    proc = subprocess.Popen(ssh + ["bash", "-lc", remote], stdout=sys.stdout, stderr=sys.stderr)
    try:
        while proc.poll() is None:
            if harness is not None:
                harness.heartbeat()
                harness.check_limits()
            time.sleep(15)
        rc = proc.returncode or 0
    finally:
        if proc.poll() is None:
            proc.terminate()
    if rc != 0:
        print(
            f"[lambda_train_pico] NoiseBase provisioning failed (rc={rc}). "
            "You may need to extend the launcher with a known-good download URL.",
            file=sys.stderr,
        )
    return rc


def _run_training(harness: SafetyHarness, ip: str, key_path: Path, args) -> int:
    """SSH in, install deps, run training. Heartbeat regularly."""
    ssh = _ssh_command(ip, key_path)

    # Install + run as a single shell pipeline; heartbeat by polling exit code.
    # Stream pip install via tee so we see errors in real-time AND keep the
    # full log on the instance for post-mortem if it fails.
    #
    # Lambda's stock Ubuntu 22.04 image ships Python 3.10. Our pyproject
    # requires >=3.11. Install python3.11 via deadsnakes PPA and use it for
    # the cloud venv. This adds ~30s to first-boot install but is fully
    # reproducible across Lambda images.
    smoke_flag = "--smoke-test" if args.smoke_test else ""
    if args.smoke_test:
        data_arg = ""
    elif args.filesystem_name:
        data_arg = f"--data /lambda/nfs/{args.filesystem_name}"
    else:
        data_arg = "--data ${OSS_DATA_DIR}/noisebase"
    remote_cmd = (
        "set -uo pipefail && "
        "cd ~/ors && "
        # Python 3.11+ bootstrap (idempotent — skip if already present)
        "if ! command -v python3.11 >/dev/null 2>&1; then "
        "  echo '--- bootstrapping python3.11 (Lambda image ships 3.10) ---' && "
        "  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
        "  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "    software-properties-common 2>&1 | tail -3 && "
        "  sudo add-apt-repository -y ppa:deadsnakes/ppa 2>&1 | tail -3 && "
        "  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "    python3.11 python3.11-venv python3.11-dev 2>&1 | tail -3; "
        "fi && "
        "python3.11 --version && "
        "python3.11 -m venv venv-cloud --upgrade-deps 2>&1 | tail -3 && "
        "source venv-cloud/bin/activate && "
        "python --version && "
        "echo '--- pip install starting ---' && "
        "pip install -e .[dev] 2>&1 | tee /tmp/ors-pip-install.log | tail -100 && "
        "echo '--- pip install OK ---' && "
        f"python -m ors.train.train_pico "
        f"{smoke_flag} {data_arg} "
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
    p.add_argument("--smoke-test", action="store_true",
                   help="run synthetic-data smoke training only (no NoiseBase). "
                        "Validates pip install + ssh + harness + checkpoint + scp pipeline. "
                        "Cheap (~$0.40 / 30 min on A10).")
    p.add_argument("--no-epoch-cap", action="store_true",
                   help="Don't cap requested epochs to what the instance can finish in "
                        "max_hours. Off by default.")
    p.add_argument("--wait-for", type=str, default=None,
                   choices=["h100", "h100_sxm5", "h100_pcie", "a100", "a100_sxm4"],
                   help="Poll Lambda capacity until the requested tier opens, then launch. "
                        "Polls every 60s. Useful when H100 is out of capacity but you don't "
                        "want to settle for A10. Combine with --wait-mins to set a timeout.")
    p.add_argument("--wait-mins", type=int, default=180,
                   help="Max minutes to poll for --wait-for tier. Default 180 (3 hours). "
                        "Aborts if capacity doesn't open in that window.")
    p.add_argument("--filesystem-name", type=str, default=None,
                   help="Lambda persistent filesystem to attach (created by lambda_stage_noisebase.py). "
                        "Mounts at ${OSS_REMOTE_HOME}/<name>. When set, skips in-run NoiseBase provisioning "
                        "and points training at ${OSS_REMOTE_HOME}/<name> directly.")
    args = p.parse_args()

    client = LambdaClient()

    # 1. Pick instance — with optional capacity poll if --wait-for set
    if args.wait_for:
        wait_targets = _expand_wait_target(args.wait_for)
        instance_type, region = _wait_for_capacity(
            client, wait_targets, max_minutes=args.wait_mins
        )
        if instance_type is None:
            print(
                f"[lambda_train_pico] capacity for {args.wait_for!r} did not open within "
                f"{args.wait_mins} minutes; aborting.",
                file=sys.stderr,
            )
            return 4
    elif args.instance_type:
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

    # 1.5 Cap epochs to what fits in max_hours on the chosen instance
    if not args.no_epoch_cap and not args.smoke_test:
        capped = _recommended_epochs(instance_type, args.epochs, args.max_hours)
        if capped < args.epochs:
            print(
                f"[lambda_train_pico] capping epochs {args.epochs} -> {capped} "
                f"(realistic for {instance_type} in {args.max_hours:.1f}h). "
                f"Pass --no-epoch-cap to override."
            )
            args.epochs = capped

    # 2. Build HarnessConfig — pre-launch approval will print full preview + prompt
    cfg = HarnessConfig(
        instance_type=instance_type,
        region=region,
        ssh_key_names=[args.ssh_key_name],
        file_system_names=[args.filesystem_name] if args.filesystem_name else [],
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

        # Install Lambda Guest Agent (best-effort; metrics in the dashboard).
        _install_guest_agent(inst.ip, cfg.ssh_key_path)

        rc = _rsync_repo(inst.ip, cfg.ssh_key_path)
        if rc != 0:
            print(
                f"[lambda_train_pico] rsync failed (rc={rc}); aborting "
                f"(instance will be terminated by the harness)",
                file=sys.stderr,
            )
            return rc

        # Provision NoiseBase — skipped if a pre-staged filesystem is attached
        if not args.smoke_test and not args.filesystem_name:
            rc = _provision_noisebase(harness, inst.ip, cfg.ssh_key_path)
            if rc != 0:
                print(
                    f"[lambda_train_pico] NoiseBase provisioning failed (rc={rc}); aborting.",
                    file=sys.stderr,
                )
                return rc

        rc = _run_training(harness, inst.ip, cfg.ssh_key_path, args)
        if rc != 0:
            print(f"[lambda_train_pico] training exited rc={rc}", file=sys.stderr)
            # Pull remote logs back for post-mortem before harness terminates.
            log_local = REPO_ROOT / "results" / "pico-cloud" / "ors-pip-install.log"
            log_local.parent.mkdir(parents=True, exist_ok=True)
            scp_log = [
                "scp",
                "-i", str(cfg.ssh_key_path),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                f"ubuntu@{inst.ip}:/tmp/ors-pip-install.log",
                str(log_local),
            ]
            subprocess.run(scp_log, check=False, capture_output=True)
            if log_local.exists():
                print(f"[lambda_train_pico] saved pip install log to {log_local}", file=sys.stderr)
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
