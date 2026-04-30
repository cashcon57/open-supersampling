"""Cloud orchestration with mandatory cost-safety harness.

NEVER launch a Lambda instance outside of `SafetyHarness` — every launch must
be wrapped so that termination is guaranteed via context-manager exit, signal
handlers, watchdog timeout, idle detection, or budget cap.
"""
from .lambda_client import LambdaClient, LambdaInstance, INSTANCE_PRICING
from .safety_harness import SafetyHarness, BudgetExceeded, MaxDurationExceeded, IdleTimeout

__all__ = [
    "LambdaClient",
    "LambdaInstance",
    "INSTANCE_PRICING",
    "SafetyHarness",
    "BudgetExceeded",
    "MaxDurationExceeded",
    "IdleTimeout",
]
