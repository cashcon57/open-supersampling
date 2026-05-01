"""Vendor-agnostic cloud client protocol for SafetyHarness.

`SafetyHarness` deliberately does not depend on any vendor's SDK. Both
`LambdaClient` and `RunPodClient` implement the `CloudClient` Protocol below,
and the harness is typed against it. Adding a third vendor (e.g. Vast.ai,
CoreWeave, Paperspace) is a matter of writing one more class that satisfies
this Protocol — no harness changes required.

Canonical status strings (used by every vendor implementation):
    "booting"      — instance is provisioning / not yet usable
    "active"       — instance is running and reachable
    "stopped"      — instance is paused (RunPod-only; Lambda has no equivalent)
    "terminating"  — termination in progress
    "terminated"   — instance is gone
    "unhealthy"    — vendor reports a problem
    "unknown"      — fall-through

Canonical instance-type names: each vendor maps its own SKU strings to its
own canonical names. The harness only ever pattern-matches on the Lambda
naming convention (e.g. `gpu_1x_h100_pcie`) for the *built-in* SKU lists.
RunPod's `RunPodClient.canonicalize_gpu_id` provides a cross-walk to those
same canonical names so a single pricing table can serve both vendors when
appropriate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class CloudInstance:
    """Vendor-neutral instance representation passed back to the caller.

    `instance_type` is the *vendor's* SKU string (whatever they returned). Use
    `client.hourly_rate(instance_type)` to look up pricing — that is where the
    vendor-specific mapping lives.
    """
    instance_id: str
    instance_type: str
    region: str
    status: str  # canonical: "booting" | "active" | "stopped" | "terminating" | "terminated" | "unhealthy" | "unknown"
    ip: Optional[str]
    hostname: Optional[str]
    launched_at: Optional[str]


@runtime_checkable
class CloudClient(Protocol):
    """Minimal interface SafetyHarness needs from any cloud provider.

    Implementations MUST:
      - never log the API key
      - make `terminate(...)` idempotent and best-effort (no raise on already-gone)
      - return canonical status strings (see module docstring)
      - implement orphan recovery in `launch(...)` so a timed-out POST that
        nevertheless created an instance server-side returns its ID
    """

    @property
    def vendor_name(self) -> str:
        """Lower-case vendor identifier, e.g. "lambda" or "runpod"."""
        ...

    @property
    def _api_key(self) -> str:  # noqa: D401  (intentional protected name)
        """Resolved API key. Used by SafetyHarness to forward to the watchdog
        and on-instance self-terminate scripts via stdin / env. NEVER logged."""
        ...

    def list_instances(self) -> list[CloudInstance]: ...

    def get_instance(self, instance_id: str) -> CloudInstance: ...

    def launch(
        self,
        instance_type_name: str,
        region_name: str,
        ssh_key_names: list[str],
        name: Optional[str] = None,
    ) -> list[str]:
        """Returns the list of new instance IDs the launch produced."""
        ...

    def terminate(self, instance_ids: list[str]) -> dict: ...

    def hourly_rate(self, instance_type_name: str) -> float:
        """USD/hour for `instance_type_name`. Returns 0.0 if unknown — the
        harness REFUSES to launch in that case, on the theory that silent
        $0/hr would defeat the budget cap."""
        ...

    def terminate_endpoint(self) -> str:
        """HTTPS URL the on-instance self-terminate cron should POST to.

        The script POSTs `{"instance_ids": ["<id>"]}` with whatever auth the
        client implementation declares via `terminate_auth_header(api_key)`.
        """
        ...

    def terminate_auth_header(self, api_key: str) -> str:
        """Single HTTP header line for the self-terminate curl, e.g.
        'Authorization: Bearer <key>' or '' (Lambda uses HTTP Basic via curl -u).

        Return empty string for "use curl's -u user:pass instead"; in that case
        the on-instance script must also know the auth flavor — see
        `terminate_curl_auth_flag`.
        """
        ...

    def terminate_curl_auth_flag(self, api_key: str) -> Optional[str]:
        """Return a fully-formed `-u "user:pass"` string for curl, or None to
        rely on the header from `terminate_auth_header`. Lambda uses Basic auth
        with key-as-username; RunPod uses Bearer.
        """
        ...

    def terminate_request_body(self, instance_id: str) -> str:
        """JSON body for the on-instance terminate POST. Lambda: REST shape
        (`{"instance_ids":[...]}`); RunPod: GraphQL mutation envelope.
        """
        ...
