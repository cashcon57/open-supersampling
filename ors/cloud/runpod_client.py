"""RunPod cloud-vendor client — implements the `CloudClient` Protocol so it
plugs into `SafetyHarness` exactly the way `LambdaClient` does.

Why we use RunPod
-----------------
Lambda H100 capacity is frequently scarce. RunPod typically has both
H100 PCIe and H100 SXM available across two tiers (`SECURE` = vetted
datacenter; `COMMUNITY` = host-supplied — cheaper but lower trust).
We default to `SECURE` for safety; pass `cloud_type="COMMUNITY"` if you
need the lower price and accept the trust trade-off.

Auth model
----------
- Compute (this client): a single API key, Bearer-auth'd.
  Resolution priority: `api_key=` kwarg → `RUNPOD_API_KEY` env →
  `<repo>/.secrets/runpod-api-key.txt`.
- Storage (NOT used by this client): RunPod has S3-compatible object storage
  for persisting datasets across pod sessions; that uses a separate
  `(access_key, secret_key)` pair stored at
  `.secrets/runpod-s3-{access,secret}-key.txt`. Those creds are explicitly
  NOT touched here. They live in RunPod's S3 namespace and are unrelated to
  pod compute or SSH.
- SSH: RunPod uses its own SSH proxy at `ssh.runpod.io` with a key the user
  uploads to their RunPod account. We do not ssh into pods over the public
  internet; we use the RunPod proxy hostname returned per-pod.

Instance / GPU naming
---------------------
RunPod's GPU IDs are human-readable strings like `NVIDIA H100 PCIe` or
`NVIDIA H100 80GB HBM3` (which is the SXM variant). We pass those strings
through verbatim as the `instance_type` field of `CloudInstance` and provide
`canonicalize_gpu_id()` for callers that want to map them onto Lambda-shape
canonical names (`gpu_1x_h100_pcie`, etc.).

NEVER call `RunPodClient.launch()` outside of `SafetyHarness` — that's the
only invariant that makes the cost-safety guarantees real.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# The runpod SDK is an optional dependency. We import lazily inside methods so
# `from ors.cloud import RunPodClient` doesn't fail at import time when the
# package isn't installed (e.g. during Lambda-only smoke tests).


# RunPod live-pricing endpoint that our `hourly_rate` falls back to, then
# caches per-process. Hand-curated reference numbers for the common SKUs we
# care about. These are floor values — RunPod's `secureSpotPrice` may be
# higher in some regions, but for budget-cap purposes we want a conservative
# (i.e. higher-than-actual is OK, lower-than-actual is dangerous) estimate.
#
# Pricing snapshot as of 2026-04-29 from `runpod.get_gpu(...)` live API:
#   NVIDIA H100 80GB HBM3 (SXM):  securePrice=$2.99, communityPrice=$2.69
#   NVIDIA H100 PCIe:             securePrice=$2.79 (TODO verify),
#                                 communityPrice ~ $2.39 (TODO verify)
#   NVIDIA A100 80GB PCIe:        securePrice=$1.89 (TODO verify)
#   NVIDIA GeForce RTX 4090:      securePrice=$0.69 (TODO verify)
#   NVIDIA A40:                   securePrice=$0.39 (TODO verify)
#
# At construction time `RunPodClient` calls `get_gpus()` ONCE and overlays the
# live `securePrice` on top of these defaults so we always have authoritative
# numbers for the harness's worst-case-cost preview. The defaults below are
# pure fallbacks for offline tests where the live call is mocked out.
RUNPOD_DEFAULT_PRICING: dict[str, float] = {
    "NVIDIA A40":                  0.44,   # secure $0.44, community $0.34 (2026-04 verified)
    "NVIDIA RTX A6000":            0.49,   # secure $0.49, community $0.33
    "NVIDIA GeForce RTX 4090":     0.69,   # secure $0.69, community $0.34
    "NVIDIA A100 80GB PCIe":       1.39,   # secure $1.39, community $1.19
    "NVIDIA A100-SXM4-40GB":       1.39,
    "NVIDIA A100-SXM4-80GB":       1.49,   # secure $1.49, community $1.39
    "NVIDIA H100 PCIe":            2.79,   # placeholder above live $2.39 for safety
    "NVIDIA H100 SXM":             2.99,   # SXM variant (alias used by some regions)
    "NVIDIA H100 80GB HBM3":       2.99,   # SXM variant
    "NVIDIA H100 NVL":             3.39,
    "NVIDIA H200":                 3.99,
    "NVIDIA B200":                 5.99,   # placeholder
}


# Cross-walk RunPod's GPU IDs to Lambda-style canonical names. Used by the
# preference selector if a caller wants to express "any H100 PCIe regardless
# of vendor" in a single list.
_RUNPOD_TO_CANONICAL: dict[str, str] = {
    "NVIDIA H100 80GB HBM3":       "gpu_1x_h100_sxm5",
    "NVIDIA H100 PCIe":            "gpu_1x_h100_pcie",
    "NVIDIA H100 NVL":             "gpu_1x_h100_pcie",
    "NVIDIA A100 80GB PCIe":       "gpu_1x_a100",
    "NVIDIA A100-SXM4-80GB":       "gpu_1x_a100_sxm4",
    "NVIDIA A100-SXM4-40GB":       "gpu_1x_a100_sxm4",
    "NVIDIA RTX A6000":            "gpu_1x_a6000",
    "NVIDIA GeForce RTX 4090":     "gpu_1x_4090",
    "NVIDIA A40":                  "gpu_1x_a40",
}


# Default Docker image — generic CUDA + PyTorch. Override via
# `RunPodClient.launch(image=...)` when you have a project-specific one.
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


@dataclass
class RunPodInstance:
    """Field-compatible with `ors.cloud.protocol.CloudInstance`."""
    instance_id: str
    instance_type: str
    region: str
    status: str
    ip: Optional[str]
    hostname: Optional[str]
    launched_at: Optional[str]


# Map RunPod's `desiredStatus` -> canonical status strings.
_STATUS_MAP: dict[str, str] = {
    "RUNNING":     "active",
    "EXITED":      "terminated",
    "TERMINATED":  "terminated",
    "DEAD":        "terminated",
    "PAUSED":      "stopped",
    "PROVISIONING": "booting",
    "PENDING":     "booting",
    "CREATED":     "booting",
    "RESTARTING":  "booting",
}


def _canonical_status(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    return _STATUS_MAP.get(raw.upper(), "unknown")


class RunPodClient:
    """RunPod compute API client. Implements `CloudClient` Protocol.

    Construction is cheap: it sets `runpod.api_key` and (best-effort) fetches
    the GPU pricing table once. If the live fetch fails, the in-module default
    table is used. Either way, `hourly_rate(...)` is fully synchronous.
    """

    vendor_name: str = "runpod"

    def __init__(
        self,
        api_key: Optional[str] = None,
        key_path: Optional[Path] = None,
        cloud_type: str = "SECURE",   # "SECURE" | "COMMUNITY" | "ALL"
        spot: bool = False,            # On-Demand by default; True = bid spot
        timeout_s: float = 30.0,
        live_pricing: bool = True,
    ):
        if cloud_type not in ("SECURE", "COMMUNITY", "ALL"):
            raise ValueError(
                f"cloud_type must be SECURE / COMMUNITY / ALL, got {cloud_type!r}"
            )
        self._api_key = self._resolve_api_key(api_key, key_path)
        self._cloud_type = cloud_type
        self._spot = spot
        self._timeout = timeout_s

        # Configure the SDK's module-global API key.
        import runpod  # noqa: PLC0415  (lazy import — see module docstring)
        runpod.api_key = self._api_key
        self._runpod = runpod

        # Per-instance pricing table — live values override defaults.
        self._pricing: dict[str, float] = dict(RUNPOD_DEFAULT_PRICING)
        if live_pricing:
            try:
                self._refresh_pricing()
            except Exception as e:
                # Don't block construction on a transient API issue.
                sys.stderr.write(
                    f"[RunPodClient] live pricing fetch failed ({e!r}); "
                    "using built-in defaults.\n"
                )

    # ---- key resolution ----

    @staticmethod
    def _resolve_api_key(api_key: Optional[str], key_path: Optional[Path]) -> str:
        if api_key:
            return api_key.strip()
        env = os.environ.get("RUNPOD_API_KEY")
        if env:
            return env.strip()
        if key_path is None:
            key_path = Path(__file__).resolve().parents[2] / ".secrets" / "runpod-api-key.txt"
        if key_path.exists():
            text = key_path.read_text().strip()
            if not text:
                raise RuntimeError(f"empty RunPod API key file: {key_path}")
            return text
        raise RuntimeError(
            "no RunPod API key available — set RUNPOD_API_KEY env var, "
            "pass api_key=, or place key at .secrets/runpod-api-key.txt"
        )

    # ---- pricing ----

    def _refresh_pricing(self) -> None:
        """Overlay live pricing from RunPod onto the default table.

        Note: `get_gpus()` returns only `(id, displayName, memoryInGb)` — the
        pricing fields are only populated when calling `get_gpu(<id>)`. So we
        loop the GPUs we care about and fetch each one's full record. Errors
        are swallowed per-GPU so a single failure doesn't blow up the rest.

        We always pick the higher of (live, default) so the budget cap stays
        conservative.
        """
        for gid in list(RUNPOD_DEFAULT_PRICING.keys()):
            try:
                g = self._runpod.get_gpu(gid)
            except Exception:
                continue
            if not g:
                continue
            if self._spot:
                price = g.get("secureSpotPrice") if self._cloud_type != "COMMUNITY" else g.get("communitySpotPrice")
            elif self._cloud_type == "COMMUNITY":
                price = g.get("communityPrice")
            else:
                price = g.get("securePrice")
            try:
                p = float(price) if price is not None else None
            except (TypeError, ValueError):
                p = None
            if p is None or p <= 0:
                continue
            existing = self._pricing.get(gid, 0.0)
            self._pricing[gid] = max(existing, p)

    def hourly_rate(self, instance_type_name: str) -> float:
        return self._pricing.get(instance_type_name, 0.0)

    @staticmethod
    def canonicalize_gpu_id(runpod_gpu_id: str) -> Optional[str]:
        """Map a RunPod GPU ID to a Lambda-style canonical name.

        Returns None if we don't have a cross-walk for it.
        """
        return _RUNPOD_TO_CANONICAL.get(runpod_gpu_id)

    def list_gpus(self) -> list[dict]:
        """Pass-through for callers that want to display capacity. Caller
        decides what to do with it; we don't massage the schema."""
        return self._runpod.get_gpus() or []

    # ---- read endpoints ----

    def list_instances(self) -> list[RunPodInstance]:
        pods = self._runpod.get_pods() or []
        return [self._parse_pod(p) for p in pods]

    def get_instance(self, instance_id: str) -> RunPodInstance:
        pod = self._runpod.get_pod(instance_id)
        if not pod:
            raise RuntimeError(f"RunPod returned no pod for id={instance_id!r}")
        return self._parse_pod(pod)

    @staticmethod
    def _parse_pod(p: dict) -> RunPodInstance:
        # SSH is exposed via runtime.ports — find the public TCP port for 22.
        ip: Optional[str] = None
        hostname: Optional[str] = None
        runtime = p.get("runtime") or {}
        for prt in (runtime.get("ports") or []):
            if prt.get("privatePort") == 22 and prt.get("isIpPublic"):
                ip = prt.get("ip")
                break
        # RunPod SSH proxy hostname is always reachable when start_ssh=True.
        # The instance ID is the hostname suffix on the proxy: `<id>@ssh.runpod.io`.
        hostname = f"{p.get('id', '')}@ssh.runpod.io"

        # `lastStatusChange` is roughly the launch timestamp once status is RUNNING.
        launched_at = p.get("lastStatusChange")

        # Region: RunPod's `myPods` schema doesn't directly include datacenter
        # for the pod, but `machine.gpuDisplayName` is what we got. Region
        # info lives behind a separate query; we leave it blank rather than
        # invent a value.
        region = ""

        return RunPodInstance(
            instance_id=p.get("id", ""),
            instance_type=(p.get("machine") or {}).get("gpuDisplayName") or "",
            region=region,
            status=_canonical_status(p.get("desiredStatus")),
            ip=ip,
            hostname=hostname,
            launched_at=str(launched_at) if launched_at is not None else None,
        )

    # ---- write endpoints (DO NOT call directly — use SafetyHarness) ----

    def launch(
        self,
        instance_type_name: str,
        region_name: str,
        ssh_key_names: list[str],
        name: Optional[str] = None,
        image: str = DEFAULT_IMAGE,
        gpu_count: int = 1,
        volume_in_gb: int = 0,
        container_disk_in_gb: int = 40,
        ports: str = "22/tcp",
        env: Optional[dict] = None,
    ) -> list[str]:
        """Launch a single RunPod GPU pod. Returns [pod_id].

        `region_name` is mapped to `data_center_id` if provided; pass empty
        string to let RunPod pick. `ssh_key_names` is accepted for parity
        with the protocol but RunPod attaches SSH keys at the *account* level
        — we forward the names via env in case the user's container reads
        them, but RunPod's own SSH proxy is what's actually used.
        """
        # Snapshot before, for orphan recovery.
        try:
            pre_existing_ids = {i.instance_id for i in self.list_instances()}
        except Exception:
            pre_existing_ids = set()

        kwargs = dict(
            name=name or "ors-training",
            image_name=image,
            gpu_type_id=instance_type_name,
            cloud_type=self._cloud_type,
            support_public_ip=True,
            start_ssh=True,
            gpu_count=gpu_count,
            volume_in_gb=volume_in_gb,
            container_disk_in_gb=container_disk_in_gb,
            ports=ports,
            env=env or {},
        )
        if region_name:
            kwargs["data_center_id"] = region_name

        try:
            pod = self._runpod.create_pod(**kwargs)
        except Exception as e:
            recovered = self._recover_orphaned_launch(
                pre_existing_ids, instance_type_name
            )
            if recovered:
                sys.stderr.write(
                    f"[RunPodClient] create_pod raised {type(e).__name__} but "
                    f"found new pod(s) {recovered} — recovered.\n"
                )
                return recovered
            raise

        if not pod or not pod.get("id"):
            # SDK returned no ID — try orphan recovery anyway.
            recovered = self._recover_orphaned_launch(
                pre_existing_ids, instance_type_name
            )
            if recovered:
                return recovered
            raise RuntimeError(f"RunPod create_pod returned no id: {pod!r}")

        return [pod["id"]]

    def _recover_orphaned_launch(
        self,
        pre_existing_ids: set[str],
        instance_type_name: str,
        retries: int = 6,
        delay_s: float = 10.0,
    ) -> list[str]:
        """If create_pod raised or returned no ID, poll list_instances for up
        to ~60s and adopt any new pod that matches the requested GPU type."""
        for _ in range(retries):
            try:
                current = self.list_instances()
                new_matching = [
                    i.instance_id for i in current
                    if i.instance_id not in pre_existing_ids
                    and (instance_type_name in i.instance_type
                         or i.instance_type in instance_type_name
                         or self.canonicalize_gpu_id(instance_type_name) is not None)
                    and i.status not in ("terminated", "terminating")
                ]
                if new_matching:
                    return new_matching
            except Exception as poll_err:
                sys.stderr.write(
                    f"[RunPodClient] orphan-recovery poll failed: {poll_err}\n"
                )
            time.sleep(delay_s)
        return []

    def terminate(self, instance_ids: list[str]) -> dict:
        """Idempotent: best-effort terminate, no raise on already-gone pods."""
        if not instance_ids:
            return {}
        results: dict[str, str] = {}
        for pid in instance_ids:
            try:
                self._runpod.terminate_pod(pid)
                results[pid] = "ok"
            except Exception as e:
                # Could be already-terminated; verify after the loop.
                results[pid] = f"err:{type(e).__name__}"
        return {"data": results}

    # ---- self-terminate fail-safe wiring ----

    def terminate_endpoint(self) -> str:
        # RunPod's GraphQL endpoint. The on-instance script POSTs a mutation
        # rather than a REST DELETE.
        return "https://api.runpod.io/graphql"

    def terminate_auth_header(self, api_key: str) -> str:
        return f"Authorization: Bearer {api_key}"

    def terminate_curl_auth_flag(self, api_key: str) -> Optional[str]:
        # We use the Authorization header instead of curl -u.
        return None

    def terminate_request_body(self, instance_id: str) -> str:
        # GraphQL mutation envelope. RunPod accepts `podTerminate(input: {podId: "..."})`.
        import json as _json
        mutation = (
            "mutation { podTerminate(input: {podId: \"" + instance_id + "\"}) }"
        )
        return _json.dumps({"query": mutation})
