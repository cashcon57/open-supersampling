"""External watchdog process — terminates a cloud GPU instance if the parent
training process dies without orderly shutdown.

This is a SEPARATE process spawned by `SafetyHarness`. It is the last line
of defense against orphaned billable instances. Even if the harness's
context-manager exit, signal handlers, and atexit handler all fail to fire
(e.g., the parent process is `kill -9`'d, or the machine reboots, or Python
crashes), the watchdog independently polls a heartbeat file and calls
terminate via the same vendor API.

Invoked via:
    python -m ors.cloud.watchdog --vendor <name> --instance-id <id> \\
                                 --heartbeat <path> ...

The watchdog reads the API key from a vendor-specific env var so the key
never appears on the command line:
    --vendor lambda    -> LAMBDA_API_KEY
    --vendor runpod    -> RUNPOD_API_KEY
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _make_client(vendor: str):
    """Construct the matching CloudClient for `vendor`. Lazy imports keep the
    runpod SDK optional for Lambda-only setups."""
    if vendor == "lambda":
        from .lambda_client import LambdaClient
        return LambdaClient()  # picks up LAMBDA_API_KEY
    if vendor == "runpod":
        from .runpod_client import RunPodClient
        # Skip live pricing fetch in the watchdog — we don't need it for
        # terminate, and avoiding the extra API call keeps the watchdog
        # resilient when the vendor's read API is briefly down.
        return RunPodClient(live_pricing=False)
    raise ValueError(f"unknown vendor: {vendor!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vendor", default="lambda",
                   help="cloud vendor: lambda | runpod (default: lambda for back-compat)")
    p.add_argument("--instance-id", required=True)
    p.add_argument("--heartbeat", required=True, type=Path)
    p.add_argument("--stale-s", type=int, default=120,
                   help="terminate if heartbeat older than this many seconds")
    p.add_argument("--max-duration-s", type=int, default=6 * 3600,
                   help="absolute hard cap from watchdog start")
    p.add_argument("--poll-interval-s", type=int, default=15)
    args = p.parse_args()

    started = time.time()
    client = _make_client(args.vendor)

    def terminate(reason: str):
        try:
            client.terminate([args.instance_id])
            sys.stderr.write(
                f"[watchdog {args.vendor} {args.instance_id}] TERMINATED: {reason}\n"
            )
        except Exception as e:
            sys.stderr.write(
                f"[watchdog {args.vendor} {args.instance_id}] terminate failed: {e}\n"
            )

    while True:
        elapsed = time.time() - started
        if elapsed > args.max_duration_s:
            terminate(f"watchdog max_duration ({elapsed:.0f}s)")
            return

        if not args.heartbeat.exists():
            # Heartbeat file removed = parent did orderly shutdown. Exit cleanly.
            return

        try:
            beat_t = float(args.heartbeat.read_text().strip())
        except Exception:
            beat_t = 0.0

        if beat_t > 0 and (time.time() - beat_t) > args.stale_s:
            terminate(
                f"heartbeat stale ({(time.time() - beat_t):.0f}s > {args.stale_s}s)"
            )
            return

        time.sleep(args.poll_interval_s)


if __name__ == "__main__":
    main()
