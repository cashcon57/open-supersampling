# SPDX-License-Identifier: Apache-2.0
"""Activation functions used by v6.2 architecture.

SqrSwish is a softsign-based Swish variant. We first saw it in AMD's FSR 4
shader source from the FidelityFX SDK 2.0.0 push of 2025-08-18, which AMD
withdrew as "published in error". Cheaper than ``x * sigmoid(x)`` on FP16
hardware: needs only mul, add, sqrt, div -- no exp.

    SqrSwish(v) = 0.5 * v * (1 + v / sqrt(v^2 + 1))

This module implements the one-line formula independently under Apache 2.0.
No AMD code is included or required. See ``docs/legal/fsr4-status.md``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SqrSwish(nn.Module):
    """Softsign-based Swish variant used by the v6.2 conv blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + x / torch.sqrt(x * x + 1.0))


__all__ = ["SqrSwish"]
