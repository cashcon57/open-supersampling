# SPDX-License-Identifier: Apache-2.0
"""Activation functions used by v6.2 architecture.

SqrSwish is a softsign-based Swish variant from AMD's FSR 4 (MIT-licensed at
``oss/third_party/fidelityfx-sdk-2.0.0-mit/Kits/FidelityFX/upscalers/fsr4/dx12/
ml2code_runtime/scalar_functions.hlsli`` line 27). Cheaper than
``x * sigmoid(x)`` on FP16 hardware: needs only mul, add, sqrt, div -- no exp.

    SqrSwish(v) = 0.5 * v * (1 + v / sqrt(v^2 + 1))

Independent reimplementation under Apache 2.0; see
``fsr4-architecture-observations.md`` for full attribution and analysis.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SqrSwish(nn.Module):
    """Softsign-based Swish variant used by the v6.2 conv blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + x / torch.sqrt(x * x + 1.0))


__all__ = ["SqrSwish"]
