"""ORU-FX: G-buffer-assisted frame extrapolation network.

Inputs per forward pass:
  warped   (B, 3, H, W)  — color frame warped to target time via flow extrapolation
  depth    (B, 1, H, W)  — depth buffer at render time, upscaled to HR; zeros if unavailable
  history  (B, C_h, H, W) — accumulated temporal features from prior frames
  alpha    (B,)           — temporal offset in (0, 1]; 1.0 = one full render interval ahead

Output:
  frame    (B, 3, H, W)  — extrapolated full-res frame at t+alpha
  history  (B, C_h, H, W) — updated temporal features for next call

Alpha embedding: sinusoidal, dim=32, injected at encoder output and bottleneck.
Residual formulation: output = warped + residual, keeping warp as strong prior.

History channels C_h=32 — hidden state propagated across frames, detached between
steps to cap BPTT at 1 (same convention as ORU-Pico).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ors.model.blocks import ConvBlock, DownBlock, UpBlock

HISTORY_CH = 32
ALPHA_DIM = 32


def alpha_embed(alpha: torch.Tensor, dim: int = ALPHA_DIM) -> torch.Tensor:
    """Sinusoidal embedding for scalar alpha in (0, 1].

    Args:
        alpha: (B,) tensor
        dim:   embedding dimension (must be even)

    Returns:
        (B, dim) tensor
    """
    half = dim // 2
    i = torch.arange(half, dtype=torch.float32, device=alpha.device)
    freq = 1.0 / (10000.0 ** (2.0 * i / dim))
    x = alpha.unsqueeze(1) * freq.unsqueeze(0)  # (B, half)
    return torch.cat([x.sin(), x.cos()], dim=-1)  # (B, dim)


class _AlphaInject(nn.Module):
    """Project alpha embedding to channel count and add to spatial feature map."""
    def __init__(self, ch: int):
        super().__init__()
        self.proj = nn.Linear(ALPHA_DIM, ch)

    def forward(self, feat: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        # feat: (B, ch, H, W), a_emb: (B, ALPHA_DIM)
        bias = self.proj(a_emb)[:, :, None, None]  # (B, ch, 1, 1)
        return feat + bias


class ORUFx(nn.Module):
    """Shading Correction Network for guided frame extrapolation.

    Architecture: shallow U-Net (3 levels), residual output.
    Input channels: 3 (warped) + 1 (depth) + HISTORY_CH = 3+1+32 = 36
    Alpha injected at encoder output (64ch) and bottleneck (128ch).
    """

    def __init__(self, history_ch: int = HISTORY_CH):
        super().__init__()
        self.history_ch = history_ch
        in_ch = 3 + 1 + history_ch  # warped RGB + depth + history

        # Encoder
        self.enc1 = ConvBlock(in_ch, 32)
        self.down1 = DownBlock(32, 64)
        self.enc2 = ConvBlock(64, 64)
        self.down2 = DownBlock(64, 128)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 128)
        self.alpha_bot = _AlphaInject(128)

        # Decoder
        self.up2 = UpBlock(128, 64)
        self.dec2 = ConvBlock(128, 64)   # 64 up + 64 skip
        self.alpha_dec2 = _AlphaInject(64)
        self.up1 = UpBlock(64, 32)
        self.dec1 = ConvBlock(64, 32)    # 32 up + 32 skip

        # Output heads
        self.out_residual = nn.Conv2d(32, 3, 1)   # color residual
        self.out_history = nn.Conv2d(32, history_ch, 1)  # updated hidden state

    def forward(
        self,
        warped: torch.Tensor,
        depth: torch.Tensor,
        history: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            warped:  (B, 3, H, W)
            depth:   (B, 1, H, W)   zeros OK when unavailable
            history: (B, history_ch, H, W)
            alpha:   (B,)            temporal offset in (0, 1]

        Returns:
            frame:   (B, 3, H, W)   extrapolated frame
            history: (B, history_ch, H, W)  updated, to be detached by caller
        """
        a_emb = alpha_embed(alpha)  # (B, ALPHA_DIM)

        x = torch.cat([warped, depth, history], dim=1)  # (B, 36, H, W)

        # Encoder
        s1 = self.enc1(x)           # (B, 32, H, W)
        s2 = self.enc2(self.down1(s1))  # (B, 64, H/2, W/2)

        # Bottleneck
        b = self.bottleneck(self.down2(s2))   # (B, 128, H/4, W/4)
        b = self.alpha_bot(b, a_emb)

        # Decoder
        d2 = self.dec2(torch.cat([self.up2(b), s2], dim=1))   # (B, 64, H/2, W/2)
        d2 = self.alpha_dec2(d2, a_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))  # (B, 32, H, W)

        residual = self.out_residual(d1)
        new_history = self.out_history(d1)

        frame = warped + residual
        return frame, new_history
