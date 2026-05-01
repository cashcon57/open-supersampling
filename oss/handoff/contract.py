"""Frozen feature-handoff contract between OSSRG and OSS.

This is a load-bearing public API. The tensor shape and dtype defined here
MUST be honored by every shipping OSSRG weight set and every shipping OSS
weight set. Both networks may evolve internally; the contract may not, except
via a major version bump (HandoffContract.VERSION).
"""
from __future__ import annotations
import torch


class HandoffContract:
    """V1 feature-handoff contract."""
    VERSION: int = 1
    CHANNELS: int = 32
    DTYPE: torch.dtype = torch.float16


class HandoffContractError(ValueError):
    """Raised when a handoff tensor violates the contract."""


def validate_handoff(features: torch.Tensor) -> None:
    """Validate a tensor against the v1 handoff contract.

    Raises HandoffContractError on any mismatch.
    """
    if features.dtype != HandoffContract.DTYPE:
        raise HandoffContractError(
            f"handoff dtype mismatch: expected {HandoffContract.DTYPE}, got {features.dtype}"
        )
    if features.ndim != 4:
        raise HandoffContractError(
            f"handoff must be 4D (B, C, H, W); got {features.ndim}D shape={tuple(features.shape)}"
        )
    if features.shape[1] != HandoffContract.CHANNELS:
        raise HandoffContractError(
            f"handoff channels mismatch: expected {HandoffContract.CHANNELS}, got {features.shape[1]}"
        )
