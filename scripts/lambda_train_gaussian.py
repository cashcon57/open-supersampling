"""Launch Gaussian param network training on Lambda Cloud.

Mirrors `scripts/lambda_train_pico.py`'s structure for the existing pixel-based
OSSPico, but trains the Gaussian param network from `oss/gaussian/network/` against
the renderer in `oss/gaussian/renderer/`.

Usage:
    python scripts/lambda_train_gaussian.py --tier standard --hours 24 --dry-run
    python scripts/lambda_train_gaussian.py --tier standard --hours 24

Sprint 4 / T4.5. Reads its training config from
`oss/gaussian/train/config.py` (which Sprint 4 implementation produces).

The launcher:
1. Provisions an H100 80GB instance via the Lambda API.
2. SCPs the repo (sans data, venv, .git submodule cache) and the staged dataset
   pointer to the instance.
3. Installs the env with `pip install -e .[cuda,review]` + `pip install -e
   oss/gaussian/renderer/vendor/image_gs/` to build gsplat.
4. Runs `python -m oss.gaussian.train.train --tier {tier} --max-steps {steps}
   --output-dir ${OSS_REMOTE_HOME}/checkpoints/{run_id}`.
5. Polls progress; on completion or timeout, downloads checkpoints to
   `./checkpoints/lambda/{run_id}/`.
6. Terminates the instance.

DOES NOT START until you confirm — `--dry-run` prints what would happen.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """One row of the training matrix."""
    tier: str           # pico | lite | standard | ultra
    max_steps: int
    batch_size: int
    learning_rate: float
    expected_hours: float
    expected_cost_usd: float


# Per-tier defaults. Tune via Sprint 4 ablation runs.
TRAIN_CONFIGS: dict[str, TrainConfig] = {
    "pico": TrainConfig("pico", 80_000, 16, 5e-4, 6.0, 18.0),
    "lite": TrainConfig("lite", 100_000, 8, 4e-4, 12.0, 36.0),
    "standard": TrainConfig("standard", 150_000, 4, 3e-4, 24.0, 72.0),
    "ultra": TrainConfig("ultra", 200_000, 2, 2e-4, 36.0, 108.0),
}


def _check_lambda_creds() -> None:
    if not os.environ.get("LAMBDA_API_KEY"):
        sys.exit("LAMBDA_API_KEY not set; export it before launching")


def _print_plan(cfg: TrainConfig, run_id: str) -> None:
    print(f"=== Lambda training plan ===")
    print(f"  run_id:           {run_id}")
    print(f"  tier:             {cfg.tier}")
    print(f"  max_steps:        {cfg.max_steps:,}")
    print(f"  batch_size:       {cfg.batch_size}")
    print(f"  learning_rate:    {cfg.learning_rate}")
    print(f"  expected_hours:   {cfg.expected_hours}")
    print(f"  expected_cost:    ${cfg.expected_cost_usd:.0f}")
    print(f"  GPU:              1x H100 80GB PCIe")


def _provision_instance() -> str:
    """Provision an H100 instance via Lambda API. Returns the SSH host."""
    raise NotImplementedError(
        "Real Lambda provision needs LAMBDA_API_KEY + the Lambda Cloud REST API "
        "client. See scripts/lambda_train_pico.py for reference. Sprint 4 will "
        "wire this up — for now use --dry-run to validate the plan."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", choices=list(TRAIN_CONFIGS), required=True)
    p.add_argument("--hours", type=float, default=None,
                   help="Override expected hours (still subject to max-steps)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without provisioning")
    p.add_argument("--run-id", default=None,
                   help="Run identifier; default: gaussian-{tier}-{timestamp}")
    args = p.parse_args(argv)

    cfg = TRAIN_CONFIGS[args.tier]
    if args.hours is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "expected_hours": args.hours})

    run_id = args.run_id or f"gaussian-{args.tier}-{int(time.time())}"
    _print_plan(cfg, run_id)

    if args.dry_run:
        print("\n--dry-run set; not provisioning.")
        return 0

    _check_lambda_creds()

    print("\nProvisioning instance...")
    host = _provision_instance()
    print(f"Provisioned. SSH host: {host}")
    # TODO(sprint-4): scp + remote install + remote train + checkpoint download
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
