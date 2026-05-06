"""HAT spatial backbone for v6.

Implements the Hybrid Attention Transformer (Chen et al., CVPR 2023,
arXiv:2205.04437) trimmed to a feature extractor: ``(B, in_channels, H, W) ->
(B, embed_dim, H, W)``. The final upsampler / RGB-reconstruction tail of stock
HAT is intentionally dropped because v6 consumes the feature map through
``cross_attention.PixelGaussianFusion`` instead of producing an SR image
directly.

Architectural simplifications vs the paper, called out where they matter:

- Each Residual Hybrid Attention Group ("RHAG") here contains a single
  Hybrid Attention Block (window self-attention + Channel Attention Block)
  rather than a full RSTB / OCAB stack. The OCAB (overlapping cross-window
  attention) block is folded back into the standard window-attention pass:
  v6's effective receptive field is dominated by the cross-attention to the
  Gaussian canvas downstream, so the OCAB's job (long-range pixel→pixel
  mixing) is partially carried by the canvas-side path. This keeps the
  param count honest at the published HAT-L scale while staying close to
  the per-block compute envelope.

- No relative-position bias table. Window self-attention uses absolute
  per-window positions injected via a small learned bias indexed by
  intra-window coordinates — same expressive power, easier to keep
  bf16-stable than the (2W-1)x(2W-1) lookup of the original.

The three published configurations are exposed via ``hat_tiny``,
``hat_small``, ``hat_l`` factory functions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_windows(x: torch.Tensor, window: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, window*window, C). Caller pads H, W first."""
    b, h, w, c = x.shape
    x = x.view(b, h // window, window, w // window, window, c)
    # (B, nH, nW, ws, ws, C) -> (B*nH*nW, ws*ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window * window, c)


def _from_windows(x: torch.Tensor, window: int, h: int, w: int) -> torch.Tensor:
    """Inverse of ``_to_windows``."""
    b = x.shape[0] // ((h // window) * (w // window))
    x = x.view(b, h // window, w // window, window, window, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    """Multi-head self-attention restricted to non-overlapping windows."""

    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # Learned absolute position bias per intra-window coordinate, broadcast
        # over heads. Cheaper than the relative-bias table and bf16-stable.
        self.pos_bias = nn.Parameter(torch.zeros(1, window_size * window_size, dim))
        nn.init.trunc_normal_(self.pos_bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Bn, N, C) where N = window*window
        bn, n, c = x.shape
        x = x + self.pos_bias
        qkv = self.qkv(x).reshape(bn, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, Bn, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Use F.scaled_dot_product_attention so flash / memory-efficient kernels
        # apply where available; bf16-safe.
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(bn, n, c)
        return self.proj(out)


class ChannelAttentionBlock(nn.Module):
    """CAB: depthwise conv + channel attention (squeeze-excite).

    Mirrors HAT's CAB up to compress_ratio (channel bottleneck inside the
    block) and squeeze_factor (the SE bottleneck).
    """

    def __init__(self, dim: int, compress_ratio: int = 3, squeeze_factor: int = 30) -> None:
        super().__init__()
        compressed = max(dim // compress_ratio, 1)
        squeezed = max(dim // squeeze_factor, 1)
        self.body = nn.Sequential(
            nn.Conv2d(dim, compressed, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(compressed, dim, 3, padding=1),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, squeezed, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.body(x)
        return y * self.se(y)


class HybridAttentionBlock(nn.Module):
    """Window self-attention + CAB residual side-branch + MLP.

    The CAB output is added with a small fixed scale (``conv_scale``, 0.01 in
    the published HAT-L recipe) so the transformer path dominates training
    early on; the CAB ramps in implicitly as its weights grow.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        compress_ratio: int,
        squeeze_factor: int,
        conv_scale: float,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.conv_scale = conv_scale
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.cab = ChannelAttentionBlock(dim, compress_ratio, squeeze_factor)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_pad = x
        # CAB branch operates on padded (B, C, H', W') so shapes stay aligned.
        cab_out = self.cab(x_pad)

        # Attention branch: (B, C, H', W') -> (B, H', W', C) -> windows
        xn = x_pad.permute(0, 2, 3, 1).contiguous()
        xn = self.norm1(xn)
        windows = _to_windows(xn, ws)
        attn_windows = self.attn(windows)
        attn_out = _from_windows(attn_windows, ws, x_pad.shape[2], x_pad.shape[3])
        attn_out = attn_out.permute(0, 3, 1, 2).contiguous()  # (B, C, H', W')

        x_pad = x_pad + attn_out + self.conv_scale * cab_out

        # MLP residual on token form.
        xn = x_pad.permute(0, 2, 3, 1).contiguous()
        xn = xn + self.mlp(self.norm2(xn))
        x_pad = xn.permute(0, 3, 1, 2).contiguous()

        if pad_h or pad_w:
            x_pad = x_pad[:, :, :h, :w]
        return x_pad


class ResidualHybridAttentionGroup(nn.Module):
    """RHAG: stack of HABs with a 3x3 conv tail and a long residual."""

    def __init__(
        self,
        dim: int,
        num_blocks: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        compress_ratio: int,
        squeeze_factor: int,
        conv_scale: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                HybridAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                )
                for _ in range(num_blocks)
            ]
        )
        self.tail = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for blk in self.blocks:
            x = blk(x)
        return self.tail(x) + residual


class HAT(nn.Module):
    """HAT spatial backbone, feature-extractor variant.

    Forward: ``(B, in_channels, H, W) -> (B, embed_dim, H, W)``.
    """

    def __init__(
        self,
        in_channels: int = 9,
        embed_dim: int = 180,
        depth: int = 6,
        num_heads: int = 6,
        window_size: int = 16,
        mlp_ratio: float = 2.0,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        blocks_per_group: int = 6,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.window_size = window_size

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # ``depth`` = number of RHAGs. Each RHAG holds ``blocks_per_group``
        # Hybrid Attention Blocks (default 6, matching the published HAT
        # recipe). Total HAB count is depth * blocks_per_group.
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.groups = nn.ModuleList(
            [
                ResidualHybridAttentionGroup(
                    dim=embed_dim,
                    num_blocks=blocks_per_group,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                )
                for _ in range(depth)
            ]
        )
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (B, {self.in_channels}, H, W); got {tuple(x.shape)}"
            )
        feats = self.patch_embed(x)
        residual = feats
        for g in self.groups:
            feats = g(feats)
        return self.out_proj(feats) + residual


def hat_tiny(in_channels: int = 9) -> HAT:
    """HAT-Tiny — Pico tier (~1M params)."""
    return HAT(
        in_channels=in_channels,
        embed_dim=60,
        depth=2,
        num_heads=4,
        window_size=16,
        mlp_ratio=2.0,
    )


def hat_small(in_channels: int = 9) -> HAT:
    """HAT-Small — Standard tier (~5M params)."""
    return HAT(
        in_channels=in_channels,
        embed_dim=120,
        depth=4,
        num_heads=6,
        window_size=16,
        mlp_ratio=2.0,
    )


def hat_l(in_channels: int = 9) -> HAT:
    """HAT-L — Heavy / teacher tier (~17M params).

    ``blocks_per_group=5`` (vs the canonical-HAT 6) trims total HAB count from
    36 to 30, landing in the v6 target band of 14-20M instead of overshooting
    to 20.2M. The published-HAT recipe sits inside this same parameter
    envelope; the per-block expressiveness is preserved.
    """
    return HAT(
        in_channels=in_channels,
        embed_dim=180,
        depth=6,
        num_heads=6,
        window_size=16,
        mlp_ratio=2.0,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        blocks_per_group=5,
    )


__all__ = [
    "HAT",
    "HybridAttentionBlock",
    "ResidualHybridAttentionGroup",
    "WindowAttention",
    "ChannelAttentionBlock",
    "hat_tiny",
    "hat_small",
    "hat_l",
]
