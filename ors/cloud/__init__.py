"""Cloud orchestration with mandatory cost-safety harness.

NEVER launch a GPU instance outside of `SafetyHarness` — every launch must
be wrapped so that termination is guaranteed via context-manager exit, signal
handlers, watchdog timeout, idle detection, or budget cap.

Two vendor clients ship out of the box; both implement the `CloudClient`
Protocol so a single `SafetyHarness` instance works against either:

    from ors.cloud import LambdaClient, RunPodClient, SafetyHarness, HarnessConfig

The harness is the only API a training script should care about. The
`*_client.py` modules are intentionally small and only know how to talk to
their respective vendor APIs.
"""
from .lambda_client import LambdaClient, LambdaInstance, INSTANCE_PRICING
from .protocol import CloudClient, CloudInstance
from .safety_harness import (
    SafetyHarness,
    BudgetExceeded,
    MaxDurationExceeded,
    IdleTimeout,
    HarnessConfig,
)

# RunPod is an optional import: only available when `runpod` is installed.
try:  # pragma: no cover  (import-time guard, exercised by both branches in tests)
    from .runpod_client import RunPodClient, RunPodInstance, RUNPOD_DEFAULT_PRICING
except ImportError:  # pragma: no cover
    RunPodClient = None  # type: ignore[assignment,misc]
    RunPodInstance = None  # type: ignore[assignment,misc]
    RUNPOD_DEFAULT_PRICING = {}  # type: ignore[assignment]


__all__ = [
    "LambdaClient",
    "LambdaInstance",
    "INSTANCE_PRICING",
    "RunPodClient",
    "RunPodInstance",
    "RUNPOD_DEFAULT_PRICING",
    "CloudClient",
    "CloudInstance",
    "SafetyHarness",
    "HarnessConfig",
    "BudgetExceeded",
    "MaxDurationExceeded",
    "IdleTimeout",
]
