"""Thin REST wrapper around the Lambda Cloud API.

Auth: HTTP Basic with API key as username, no password.
Docs: https://cloud.lambda.ai/api/v1/docs

This module is intentionally minimal — *all* business logic about lifecycle,
auto-termination, idle detection, and budget tracking lives in
`safety_harness.py`. This file only knows how to talk to the API.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


_API_BASE = "https://cloud.lambda.ai/api/v1"

# Hourly USD pricing for common Lambda SKUs.
# Source: lambda.ai pricing page; verify before relying on for budget caps.
# Conservative — round up to absorb occasional billing rounding.
INSTANCE_PRICING: dict[str, float] = {
    "gpu_1x_a6000":    0.80,
    "gpu_1x_a10":      0.75,
    "gpu_1x_a100":     1.29,
    "gpu_1x_a100_sxm4": 1.79,
    "gpu_1x_h100_pcie": 2.49,
    "gpu_1x_h100_sxm5": 2.99,
    "gpu_2x_a100":     2.58,
    "gpu_4x_a100":     5.16,
    "gpu_8x_a100":     10.32,
    "gpu_8x_h100":     23.92,
}

# Effective FP16 throughput (TFLOPs) for small CNNs at typical 25-35% of peak.
# Used by `select_optimal_instance` to estimate wall-time and pick "fastest
# available" rather than "cheapest available". Numbers are conservative.
INSTANCE_EFFECTIVE_FP16_TFLOPS: dict[str, float] = {
    "gpu_1x_a10":       35.0,
    "gpu_1x_a6000":     45.0,
    "gpu_1x_a100":      90.0,
    "gpu_1x_a100_sxm4": 110.0,
    "gpu_1x_h100_pcie": 250.0,
    "gpu_1x_h100_sxm5": 320.0,
    "gpu_2x_a100":      180.0,
    "gpu_4x_a100":      360.0,
    "gpu_8x_a100":      720.0,
    "gpu_8x_h100":      2560.0,
}

# Default preference ordering for a single-GPU small-CNN training workload
# (~250K params, ~1-3 hour total compute on a single instance).
# Order: fastest first; selector falls through to next available.
SINGLE_GPU_PREFERENCE_ORDER: list[str] = [
    "gpu_1x_h100_sxm5",   # fastest, $2.99/hr — usually scarce
    "gpu_1x_h100_pcie",   # nearly as fast, $2.49/hr
    "gpu_1x_a100_sxm4",   # $1.79/hr
    "gpu_1x_a100",        # $1.29/hr — common mid-tier
    "gpu_1x_a6000",       # $0.80/hr — solid fallback
    "gpu_1x_a10",         # $0.75/hr — slow but always available
]


def select_optimal_instance(
    client: "LambdaClient",
    preference: Optional[list[str]] = None,
    workload_tflops: float = 200.0,
    max_extra_per_run_usd: float = 5.0,
) -> tuple[str, str]:
    """Pick the most time-efficient available instance, biased toward speed.

    Policy:
        1. Walk `preference` in order (fastest first).
        2. For each tier, if it has capacity in any region:
             a. Compute its estimated wall-time and run-cost for `workload_tflops`.
             b. If we already have a "cheapest acceptable" candidate and this
                tier's run-cost is more than `max_extra_per_run_usd` higher,
                prefer the cheaper one (still finishing in reasonable time).
                Otherwise pick the faster one — time savings win.
        3. If nothing in preference is available, raise.

    Returns (instance_type_name, region_name).
    """
    if preference is None:
        preference = SINGLE_GPU_PREFERENCE_ORDER

    types = client.list_instance_types()
    candidates = []
    for tname in preference:
        entry = types.get(tname, {})
        regions = entry.get("regions_with_capacity_available", []) or []
        region_names = [r.get("name") if isinstance(r, dict) else r for r in regions]
        if not region_names:
            continue
        rate = INSTANCE_PRICING.get(tname, 0.0)
        eff_tflops = INSTANCE_EFFECTIVE_FP16_TFLOPS.get(tname, 30.0)
        # Wall time in hours = TFLOPs of work / TFLOPs/sec / 3600
        wall_hr = workload_tflops * 1000.0 / (eff_tflops * 1000.0 * 3600.0) if eff_tflops > 0 else 9999.0
        # Simpler: workload_tflops is "thousands of GFLOPs"; eff is TFLOPs/sec
        # workload_seconds = workload_tflops / eff_tflops
        wall_s = workload_tflops / eff_tflops if eff_tflops > 0 else 1e9
        cost = (wall_s / 3600.0) * rate
        candidates.append((tname, region_names[0], wall_s, cost, rate))

    if not candidates:
        raise RuntimeError(
            "No instances of any preferred tier are available. "
            "Try again later, or expand `preference` list."
        )

    # Candidates are already in fastest-first order via `preference`. Walk them
    # and apply the cost-extra cap: if a faster tier costs >cap more than the
    # cheapest, fall back to cheapest. Else pick the fastest.
    fastest = candidates[0]
    cheapest = min(candidates, key=lambda c: c[3])  # by total run cost
    if fastest[3] - cheapest[3] > max_extra_per_run_usd:
        return cheapest[0], cheapest[1]
    return fastest[0], fastest[1]


@dataclass
class LambdaInstance:
    instance_id: str
    instance_type: str
    region: str
    status: str  # "booting" | "active" | "terminated" | "unhealthy" | ...
    ip: Optional[str]
    hostname: Optional[str]
    launched_at: Optional[str]  # ISO8601 string from Lambda API

    @property
    def hourly_rate_usd(self) -> float:
        return INSTANCE_PRICING.get(self.instance_type, 0.0)


# `LambdaInstance` happens to be field-compatible with `CloudInstance` from
# `ors.cloud.protocol` — the dataclasses share the same field names and types.
# The harness treats either one structurally, so no conversion layer is needed.


class LambdaClient:
    """Implements the `CloudClient` Protocol from `ors.cloud.protocol`."""

    vendor_name: str = "lambda"
    """Lambda Cloud REST API client.

    Reads the API key from one of (in priority order):
      1. The `api_key` constructor argument
      2. The `LAMBDA_API_KEY` environment variable
      3. The contents of the file at `key_path`
      4. The default file at `<repo>/.secrets/lambda-api-key.txt`
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        key_path: Optional[Path] = None,
        timeout_s: float = 30.0,
    ):
        self._api_key = self._resolve_api_key(api_key, key_path)
        self._timeout = timeout_s
        self._session = requests.Session()
        self._session.auth = (self._api_key, "")

    @staticmethod
    def _resolve_api_key(api_key: Optional[str], key_path: Optional[Path]) -> str:
        if api_key:
            return api_key.strip()
        env = os.environ.get("LAMBDA_API_KEY")
        if env:
            return env.strip()
        if key_path is None:
            # default repo location
            key_path = Path(__file__).resolve().parents[2] / ".secrets" / "lambda-api-key.txt"
        if key_path.exists():
            text = key_path.read_text().strip()
            if not text:
                raise RuntimeError(f"empty Lambda API key file: {key_path}")
            return text
        raise RuntimeError(
            "no Lambda API key available — set LAMBDA_API_KEY env var, "
            "pass api_key=, or place key at .secrets/lambda-api-key.txt"
        )

    # ----- read endpoints (cheap, idempotent) -----

    def list_instances(self) -> list[LambdaInstance]:
        r = self._session.get(f"{_API_BASE}/instances", timeout=self._timeout)
        r.raise_for_status()
        data = r.json().get("data", [])
        return [self._parse_instance(item) for item in data]

    def get_instance(self, instance_id: str) -> LambdaInstance:
        r = self._session.get(f"{_API_BASE}/instances/{instance_id}", timeout=self._timeout)
        r.raise_for_status()
        return self._parse_instance(r.json().get("data", {}))

    def list_instance_types(self) -> dict:
        r = self._session.get(f"{_API_BASE}/instance-types", timeout=self._timeout)
        r.raise_for_status()
        return r.json().get("data", {})

    def list_ssh_keys(self) -> list[dict]:
        r = self._session.get(f"{_API_BASE}/ssh-keys", timeout=self._timeout)
        r.raise_for_status()
        return r.json().get("data", [])

    # ----- write endpoints (do not call directly — use SafetyHarness) -----

    def launch(
        self,
        instance_type_name: str,
        region_name: str,
        ssh_key_names: list[str],
        file_system_names: Optional[list[str]] = None,
        name: Optional[str] = None,
        quantity: int = 1,
    ) -> list[str]:
        """Launch instance(s). Returns list of new instance IDs.

        WARNING: this starts billing. Always call from inside SafetyHarness
        so that termination is guaranteed.

        Lambda's launch API can be slow (seen 60-90s server-side processing
        when capacity is tight). We use a longer timeout here than for
        read endpoints. If we timeout but the launch succeeded server-side,
        a stale instance may end up unowned — see `_recover_orphaned_launch`.
        """
        body = {
            "region_name": region_name,
            "instance_type_name": instance_type_name,
            "ssh_key_names": ssh_key_names,
            "quantity": quantity,
        }
        if file_system_names:
            body["file_system_names"] = file_system_names
        if name:
            body["name"] = name

        # Snapshot active instances BEFORE the launch so we can detect any
        # instance Lambda created even if our POST times out.
        try:
            pre_existing_ids = {i.instance_id for i in self.list_instances()}
        except Exception:
            pre_existing_ids = set()

        try:
            r = self._session.post(
                f"{_API_BASE}/instance-operations/launch",
                json=body,
                timeout=180.0,  # generous; launch can be slow when capacity tight
            )
            r.raise_for_status()
            return r.json().get("data", {}).get("instance_ids", [])
        except requests.exceptions.RequestException as e:
            # Did Lambda create an instance anyway despite our timeout?
            recovered = self._recover_orphaned_launch(
                pre_existing_ids, instance_type_name, region_name
            )
            if recovered:
                sys.stderr.write(
                    f"[LambdaClient] launch POST raised {type(e).__name__} but found new "
                    f"instance(s) {recovered} — recovered.\n"
                )
                return recovered
            raise

    def _recover_orphaned_launch(
        self,
        pre_existing_ids: set[str],
        instance_type_name: str,
        region_name: str,
        retries: int = 6,
        delay_s: float = 10.0,
    ) -> list[str]:
        """After a launch POST raises, poll list_instances for up to ~60s
        to detect whether Lambda created an instance server-side that we
        weren't told about. Returns its ID(s) so the harness can track it
        properly (and terminate it on exit if needed)."""
        for _ in range(retries):
            try:
                current = self.list_instances()
                new_matching = [
                    i.instance_id for i in current
                    if i.instance_id not in pre_existing_ids
                    and i.instance_type == instance_type_name
                    and i.region == region_name
                    and i.status not in ("terminated", "terminating")
                ]
                if new_matching:
                    return new_matching
            except Exception as poll_err:
                sys.stderr.write(
                    f"[LambdaClient] orphan-recovery poll failed: {poll_err}\n"
                )
            time.sleep(delay_s)
        return []

    def terminate(self, instance_ids: list[str]) -> dict:
        """Terminate instance(s). Idempotent — safe to call multiple times.

        Uses a generous timeout because terminate is the safety operation
        that MUST succeed and we don't want to give up early.
        """
        if not instance_ids:
            return {}
        r = self._session.post(
            f"{_API_BASE}/instance-operations/terminate",
            json={"instance_ids": instance_ids},
            timeout=120.0,  # generous; terminate must succeed
        )
        # do NOT raise_for_status here — terminate must succeed best-effort
        # even if some instances are already gone
        return r.json() if r.status_code < 500 else {"error": r.text}

    @staticmethod
    def _parse_instance(d: dict) -> LambdaInstance:
        return LambdaInstance(
            instance_id=d.get("id", ""),
            instance_type=(d.get("instance_type") or {}).get("name", ""),
            region=(d.get("region") or {}).get("name", ""),
            status=d.get("status", "unknown"),
            ip=d.get("ip"),
            hostname=d.get("hostname"),
            launched_at=d.get("created_at") or d.get("launched_at"),
        )

    # ----- CloudClient protocol surface (vendor-agnostic) -----

    def hourly_rate(self, instance_type_name: str) -> float:
        return INSTANCE_PRICING.get(instance_type_name, 0.0)

    def terminate_endpoint(self) -> str:
        return f"{_API_BASE}/instance-operations/terminate"

    def terminate_auth_header(self, api_key: str) -> str:
        # Lambda uses HTTP Basic via curl -u, no Authorization header.
        return ""

    def terminate_curl_auth_flag(self, api_key: str) -> Optional[str]:
        # Lambda Cloud accepts API key as username, empty password.
        return f"{api_key}:"

    def terminate_request_body(self, instance_id: str) -> str:
        import json as _json
        return _json.dumps({"instance_ids": [instance_id]})
