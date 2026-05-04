"""Small temporal-head conv stack for the v5 pixel temporal track.

Architecture (per spec §Architecture point 5):
    Input (8ch HR): concat(current_sr, warped_prev, disocclusion, depth_hr)
    Conv(8 → 32, 3x3) + ReLU
    Conv(32 → 32, 3x3) + ReLU
    Conv(32 → 32, 3x3) + ReLU
    Conv(32 → 3,  3x3)            # residual on top of current_sr

Final output = current_sr + small residual. The final conv is initialized
with small weights and zero bias so the head starts as a near-identity on
current_sr — training only has to learn the temporal correction.

Param budget (8*32 + 32*32*3) channels worth of 3x3 convs + biases ~= 28K.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalHead(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(8, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden, 3, 3, padding=1)

        for m in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        # Tiny init on output residual so initial output ~= current_sr.
        nn.init.normal_(self.conv_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        current_sr: torch.Tensor,
        warped_prev: torch.Tensor,
        disocclusion: torch.Tensor,
        depth_hr: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([current_sr, warped_prev, disocclusion, depth_hr], dim=1)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        residual = self.conv_out(x)
        return current_sr + residual


__all__ = ["TemporalHead"]
