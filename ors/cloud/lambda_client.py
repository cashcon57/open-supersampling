"""Thin REST wrapper around the Lambda Cloud API.

Auth: HTTP Basic with API key as username, no password.
Docs: https://cloud.lambda.ai/api/v1/docs

This module is intentionally minimal — *all* business logic about lifecycle,
auto-termination, idle detection, and budget tracking lives in
`safety_harness.py`. This file only knows how to talk to the API.
"""
from __future__ import annotations

import os
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


class LambdaClient:
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
        r = self._session.post(
            f"{_API_BASE}/instance-operations/launch",
            json=body,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json().get("data", {}).get("instance_ids", [])

    def terminate(self, instance_ids: list[str]) -> dict:
        """Terminate instance(s). Idempotent — safe to call multiple times."""
        if not instance_ids:
            return {}
        r = self._session.post(
            f"{_API_BASE}/instance-operations/terminate",
            json={"instance_ids": instance_ids},
            timeout=self._timeout,
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
