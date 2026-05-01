"""Paired-mode wrapper: ORD + ORU coupled via the handoff contract.

When paired, ORD runs at the low (input) resolution and emits the 32-ch FP16
feature tensor as a side output. ORU consumes those features directly via its
`features` input mode. The result is a joint denoise+upscale architecture that
delivers near-DLSS-RR perf without coupling either component to the other.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..handoff import validate_handoff
from .oss_rg import ORD
from .oss import ORU


class PairedORS(nn.Module):
    def __init__(self, ord_model: ORD, oru_model: ORU):
        super().__init__()
        if oru_model.input_mode != "features":
            raise ValueError(
                "PairedORS requires ORU configured with input_mode='features', "
                f"got input_mode='{oru_model.input_mode}'"
            )
        self.ord = ord_model
        self.oru = oru_model

    def forward(
        self,
        *,
        noisy: torch.Tensor,
        aux: torch.Tensor,
        history: torch.Tensor,
        depth: torch.Tensor,
        motion: torch.Tensor,
    ):
        rgb_lo, features = self.ord(noisy, aux, history)
        validate_handoff(features)
        rgb_hi = self.oru(features=features, depth=depth, motion=motion)
        return rgb_lo, rgb_hi
