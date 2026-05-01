"""Submit an OSS-RG training job to RunPod H100 with NoiseBase data from R2.

Usage:
    python scripts/train_rg_runpod.py [--runpod-key KEY] [--r2-bucket NAME]
                                      [--epochs N] [--batch-size N]
"""
from __future__ import annotations

import argparse
import os
import sys

from oss.cloud import RunPodClient


_IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel"
_DISK_GB = 100

_GPU_PREFERENCE = [
    "NVIDIA H100 SXM",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
]


def _select_gpu(client: RunPodClient, preference: list[str]) -> str:
    gpus = client.list_gpus()
    by_id = {g["id"]: g for g in gpus}
    for gid in preference:
        g = by_id.get(gid)
        if g and (g.get("secureCloud") or g.get("communityCloud")):
            return gid
    raise RuntimeError(f"No preferred H100 GPU type available: {preference}")


def _build_start_cmd(r2_bucket: str, epochs: int, batch_size: int) -> str:
    return (
        "set -euo pipefail && "
        "pip install 'oss[cuda]' && "
        "mkdir -p /data/noisebase && "
        f"rclone copy r2:{r2_bucket}/ /data/noisebase/ "
        "--config /root/.config/rclone/rclone.conf "
        "--transfers 8 --checkers 16 --progress && "
        "mkdir -p /out/rg_run && "
        f"python -m oss.train.train_rg "
        f"--data-root /data/noisebase "
        f"--out /out/rg_run "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} && "
        f"rclone copy /out/rg_run/ r2:{r2_bucket}/checkpoints/rg/ "
        "--config /root/.config/rclone/rclone.conf "
        "--transfers 4 --progress"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runpod-key", default=None)
    p.add_argument("--r2-bucket", default="oss-noisebase")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    api_key = args.runpod_key or os.environ.get("RUNPOD_API_KEY")
    client = RunPodClient(api_key=api_key)

    gpu_type = _select_gpu(client, _GPU_PREFERENCE)
    rate = client.hourly_rate(gpu_type)
    print(f"Selected GPU : {gpu_type}")
    print(f"Hourly rate  : ${rate:.2f}/hr")

    start_cmd = _build_start_cmd(args.r2_bucket, args.epochs, args.batch_size)

    pod_ids = client.launch(
        instance_type_name=gpu_type,
        region_name="",
        ssh_key_names=[],
        name="oss-rg-training",
        image=_IMAGE,
        container_disk_in_gb=_DISK_GB,
        env={"START_CMD": start_cmd},
    )

    print(f"Pod launched : {pod_ids[0]}")
    print(f"R2 bucket    : {args.r2_bucket}")
    print(f"Epochs       : {args.epochs}  batch-size: {args.batch_size}")
    print(
        "NOTE: terminate via `runpod_terminate_all.py` or the RunPod dashboard "
        "when training is complete."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
