"""Paired-mode wrapper: OSSRG + OSS coupled via the handoff contract.

When paired, OSSRG runs at the low (input) resolution and emits the 32-ch FP16
feature tensor as a side output. OSS consumes those features directly via its
`features` input mode. The result is a joint denoise+upscale architecture that
delivers near-DLSS-RR perf without coupling either component to the other.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..handoff import validate_handoff
from .oss_rg import OSSRG
from .oss import OSS


class PairedOSS(nn.Module):
    def __init__(self, ossrg_model: OSSRG, oss_model: OSS):
        super().__init__()
        if oss_model.input_mode != "features":
            raise ValueError(
                "PairedOSS requires OSS configured with input_mode='features', "
                f"got input_mode='{oss_model.input_mode}'"
            )
        self.ord = ossrg_model
        self.oru = oss_model

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
