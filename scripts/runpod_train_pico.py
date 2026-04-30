"""End-to-end RunPod launcher for ORU-Pico training.

    python -m scripts.runpod_train_pico --epochs 50 --budget 8

What it does, in order:
    1. Pick a RunPod GPU type (default H100 PCIe; override with --gpu-type).
    2. Print a cost preview + estimated wall-time for the chosen tier.
    3. Wrap the launch in `SafetyHarness` which will:
         - require interactive 'launch' confirmation (cost preview again)
         - terminate at budget cap (default $20)
         - terminate at max duration (default 6h)
         - terminate on idle (<5% GPU util for 15min)
         - terminate on signal / atexit / watchdog stale heartbeat
    4. SSH into the pod (via RunPod's ssh proxy), rsync the repo, run training.
    5. Stream training logs back; heartbeat to the watchdog every minute.
    6. On any exit path, terminate the pod and verify it's gone.

NEVER call `RunPodClient.launch()` directly outside of SafetyHarness.

SSH note: RunPod pods get a public IP + port 22 when launched with
`start_ssh=True` and a port mapping. The actual SSH key has to be uploaded
to your RunPod account in advance — RunPod injects all of your account
SSH keys into every pod automatically.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from ors.cloud import RunPodClient, SafetyHarness, HarnessConfig


# Default preference for ORU-Pico — fastest H100 first, falling through.
RUNPOD_PREFERENCE = [
    "NVIDIA H100 80GB HBM3",   # H100 SXM, ~$2.99/hr secure
    "NVIDIA H100 PCIe",        # ~$2.79/hr secure
    "NVIDIA A100 80GB PCIe",   # ~$1.89/hr secure
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 4090", # ~$0.69/hr secure
    "NVIDIA A40",              # ~$0.39/hr secure
]


REPO_ROOT = Path(__file__).resolve().parents[1]


def _select_gpu(client: RunPodClient, preference: list[str]) -> str:
    """Walk `preference` and return the first GPU type with capacity."""
    gpus = client.list_gpus()
    by_id = {g["id"]: g for g in gpus}
    for gid in preference:
        g = by_id.get(gid)
        if not g:
            continue
        # `secureCloud` / `communityCloud` flags = at least one DC has it listed.
        if g.get("secureCloud") or g.get("communityCloud"):
            return gid
    raise RuntimeError(
        f"None of the preferred GPU types are listed at all: {preference}"
    )


_RUNPOD_SSH_KEY = REPO_ROOT / ".secrets" / "runpod-ssh.pem"


def _ssh_command(ip: str, port: int = 22, user: str = "root") -> list[str]:
    return [
        "ssh",
        "-i", str(_RUNPOD_SSH_KEY),
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=20",
        "-o", "BatchMode=yes",          # never prompt for password — fail fast
        f"{user}@{ip}",
    ]


def _rsync_repo(ip: str, port: int = 22, user: str = "root") -> int:
    """Push the committed repo to the pod via `git archive HEAD | ssh tar`.

    Why git archive instead of plain tar:
    - bsdtar (macOS) `--exclude=./data` matches BASENAME `data` anywhere,
      not just top-level — strips out `ors/data/` too. We tried `--anchored`
      but bsdtar doesn't support GNU's anchored mode.
    - git archive packages exactly what's tracked in HEAD: respects
      .gitignore (so venv/, data/, results/, .secrets/ never ship), and
      includes packages we DO want like ors/data/.
    - Bonus: deterministic. Whatever was committed is what goes to the pod.

    Note: untracked work-in-progress files won't ship until committed. For
    smoke runs that's fine — and a feature, since we want training to use
    the version we'd push to GitHub, not work-in-progress changes.
    """
    print(f"[runpod_train_pico] git-archive+ssh push of HEAD to {ip}:~/ors/ ...")
    ssh_cmd = [
        "ssh",
        "-i", str(_RUNPOD_SSH_KEY),
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        f"{user}@{ip}",
        "mkdir -p ~/ors && cd ~/ors && tar xf -",
    ]
    git_cmd = ["git", "-C", str(REPO_ROOT), "archive", "--format=tar", "HEAD"]
    g = subprocess.Popen(git_cmd, stdout=subprocess.PIPE)
    ssh = subprocess.Popen(ssh_cmd, stdin=g.stdout, stdout=sys.stdout, stderr=sys.stderr)
    if g.stdout:
        g.stdout.close()
    rc_ssh = ssh.wait()
    rc_g = g.wait()
    if rc_g != 0 or rc_ssh != 0:
        print(f"[runpod_train_pico] git-archive+ssh failed (git={rc_g} ssh={rc_ssh})", file=sys.stderr)
        return rc_ssh or rc_g
    return 0


def _rsync_repo_orig_keep(ip: str, port: int = 22, user: str = "root") -> int:
    ssh_args = (
        f"ssh -i {_RUNPOD_SSH_KEY} -p {port} "
        f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-o BatchMode=yes"
    )
    cmd = [
        "rsync", "-az", "--delete",
        "-e", ssh_args,
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
    print(f"[runpod_train_pico] rsync repo to {ip}:~/ors/ ...")
    return subprocess.run(cmd).returncode


def _run_training(harness: SafetyHarness, ip: str, port: int, args) -> int:
    ssh = _ssh_command(ip, port)
    smoke_flag = "--smoke-test" if args.smoke_test else ""
    data_arg = "" if args.smoke_test else "--data /root/data/noisebase"
    remote_cmd = (
        "set -uo pipefail && "
        "cd ~/ors && "
        "python3 -m venv venv-cloud --upgrade-deps 2>&1 | tail -3 && "
        "source venv-cloud/bin/activate && "
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

    print(f"[runpod_train_pico] starting training:\n  {remote_cmd}\n")
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


def _wait_for_ssh_endpoint(client: RunPodClient, pod_id: str, timeout_s: int = 300) -> tuple[str, int]:
    """Wait for the pod's runtime ports to become available; return (ip, port).

    RunPod's `runtime.ports` is empty until the pod's container is fully up.
    We poll `get_instance` every 5s for `timeout_s`.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        inst = client.get_instance(pod_id)
        # `inst.ip` is the public IP for port 22 if exposed. The actual port
        # is in the underlying SDK response — fetch it via the raw query.
        raw = client._runpod.get_pod(pod_id)
        runtime = (raw or {}).get("runtime") or {}
        for prt in (runtime.get("ports") or []):
            if prt.get("privatePort") == 22 and prt.get("isIpPublic"):
                ip = prt.get("ip")
                port = int(prt.get("publicPort") or 22)
                if ip:
                    return ip, port
        time.sleep(5)
    raise RuntimeError(f"pod {pod_id} did not expose SSH within {timeout_s}s")


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
    p.add_argument("--purpose", type=str, default="ORU-Pico training (RunPod)",
                   help="short description shown in the approval preview")
    p.add_argument("--gpu-type", type=str, default=None,
                   help="override auto-select; e.g. 'NVIDIA H100 PCIe'")
    p.add_argument("--cloud-type", type=str, default="SECURE",
                   choices=["SECURE", "COMMUNITY", "ALL"])
    p.add_argument("--region", type=str, default="",
                   help="optional data_center_id (e.g. US-CA-1). Empty = RunPod picks.")
    p.add_argument("--image", type=str, default=None,
                   help="override docker image (default: runpod/pytorch CUDA 12.4)")
    p.add_argument("--container-disk-gb", type=int, default=40)
    p.add_argument("--volume-gb", type=int, default=0,
                   help="persistent volume size in GB (0 = no volume)")
    p.add_argument("--smoke-test", action="store_true",
                   help="run synthetic-data smoke training only (no NoiseBase).")
    args = p.parse_args()

    client = RunPodClient(cloud_type=args.cloud_type)

    if args.gpu_type:
        gpu_type = args.gpu_type
        print(f"[runpod_train_pico] using user-specified gpu_type={gpu_type!r}")
    else:
        gpu_type = _select_gpu(client, RUNPOD_PREFERENCE)
        print(f"[runpod_train_pico] auto-selected gpu_type={gpu_type!r}")

    rate = client.hourly_rate(gpu_type)
    if rate == 0.0:
        print(f"[runpod_train_pico] WARNING: no pricing entry for {gpu_type!r}; "
              f"add to RUNPOD_DEFAULT_PRICING in runpod_client.py.", file=sys.stderr)
        return 3
    print(f"  hourly rate         : ${rate:.2f}/hr  (cloud_type={args.cloud_type})")
    print(f"  worst-case duration : {args.max_hours:.1f}h  ->  ${args.max_hours*rate:.2f}")

    cfg = HarnessConfig(
        instance_type=gpu_type,
        region=args.region,
        ssh_key_names=[],   # RunPod manages SSH at account level
        name="ors-pico-training",
        max_duration_s=int(args.max_hours * 3600),
        budget_usd=args.budget,
        # SSH key path needed so harness can install on-instance self-terminate
        # cron + run idle detection. RunPod uses dedicated key in .secrets/.
        ssh_key_path=_RUNPOD_SSH_KEY,
        ssh_user="root",
        purpose=args.purpose,
    )

    harness = SafetyHarness(client, cfg)
    with harness as inst:
        print(f"[runpod_train_pico] pod {inst.instance_id} active")

        # Wait for the SSH port to become reachable.
        try:
            ip, port = _wait_for_ssh_endpoint(client, inst.instance_id, timeout_s=600)
        except Exception as e:
            print(f"[runpod_train_pico] SSH endpoint never appeared: {e}", file=sys.stderr)
            return 4
        print(f"[runpod_train_pico] SSH endpoint: {ip}:{port}")

        # No rsync prep needed — _rsync_repo uses tar+ssh which works on any
        # container with a shell.
        rc = _rsync_repo(ip, port)
        if rc != 0:
            print(f"[runpod_train_pico] rsync failed (rc={rc}); aborting", file=sys.stderr)
            return rc

        rc = _run_training(harness, ip, port, args)
        if rc != 0:
            print(f"[runpod_train_pico] training exited rc={rc}", file=sys.stderr)
            return rc

        # Pull checkpoint back before harness terminates.
        ckpt_local = REPO_ROOT / "results" / "pico-cloud-runpod" / "oru_pico.pth"
        ckpt_local.parent.mkdir(parents=True, exist_ok=True)
        scp_cmd = [
            "scp",
            "-P", str(port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"root@{ip}:~/ors/results/pico-cloud/oru_pico.pth",
            str(ckpt_local),
        ]
        print(f"[runpod_train_pico] pulling checkpoint to {ckpt_local}")
        subprocess.run(scp_cmd, check=False)

    return 0


if __name__ == "__main__":
    print(
        "WARNING: This launcher is the wrapper around real RunPod spend. Read it\n"
        "before running. The SafetyHarness will print a cost preview and require\n"
        "interactive approval before any pod is created.\n"
    )
    sys.exit(main())
