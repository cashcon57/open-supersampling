"""Recurrent latent primitives for ORU-Pico.

The recurrent latent cell threads a hidden state through bottleneck features
across frames so the network can amortize denoising/upsampling work over time.
We use a GRU-style gating scheme implemented with 1x1 convolutions to keep the
parameter count down (the cell is sized for Steam Deck / RDNA 2 inference).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentLatentCell(nn.Module):
    """GRU-style recurrent cell over a 4D feature tensor.

    Cheaper than full ConvGRU - uses 1x1 convs for the update / reset / candidate
    gates so the param count stays under ~5K at the (channels=32, hidden=24)
    bottleneck. Returns a refined feature tensor at the same shape as the input
    plus a propagated hidden state for the next frame.
    """

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        assert channels % 8 == 0, f"channels={channels} must be multiple of 8"
        assert hidden_channels % 8 == 0, (
            f"hidden_channels={hidden_channels} must be multiple of 8"
        )
        self.channels = channels
        self.hidden_channels = hidden_channels

        # GRU gates: input is concat(x, hidden) -> hidden_channels.
        gate_in = channels + hidden_channels
        self.gate_z = nn.Conv2d(gate_in, hidden_channels, kernel_size=1)
        self.gate_r = nn.Conv2d(gate_in, hidden_channels, kernel_size=1)
        self.gate_h = nn.Conv2d(gate_in, hidden_channels, kernel_size=1)

        # Project the new hidden state back into feature space so the bottleneck
        # downstream sees `channels` channels (no shape change to the encoder).
        self.refine = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        assert C == self.channels, (
            f"Expected {self.channels} channels, got {C}"
        )

        if hidden is None:
            hidden = x.new_zeros((B, self.hidden_channels, H, W))
        else:
            assert hidden.shape == (B, self.hidden_channels, H, W), (
                f"hidden shape {tuple(hidden.shape)} != "
                f"{(B, self.hidden_channels, H, W)}"
            )

        combined = torch.cat([x, hidden], dim=1)
        z = torch.sigmoid(self.gate_z(combined))
        r = torch.sigmoid(self.gate_r(combined))

        # Candidate hidden uses x and (r * hidden).
        cand_in = torch.cat([x, r * hidden], dim=1)
        h_tilde = torch.tanh(self.gate_h(cand_in))

        new_hidden = (1.0 - z) * hidden + z * h_tilde
        # Residual refinement - additive so init (zero hidden) is the identity
        # on the very first frame and the encoder still learns standalone.
        x_refined = x + self.refine(new_hidden)
        x_refined = F.silu(x_refined, inplace=True)
        return x_refined, new_hidden
