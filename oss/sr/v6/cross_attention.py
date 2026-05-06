"""Pixel-Gaussian fusion: window cross-attention with rotary positional encoding.

Pixel features (queries) attend to Gaussian-canvas tokens (keys / values).
Mirrors GSASR's ``fea2gsropeamp_arch.py`` in shape but kept self-contained:
no GSASR imports, no dependency on the canvas package, and bf16-friendly via
``F.scaled_dot_product_attention``.

Layout per forward pass:

  pixel_features  (B, feat_dim, H, W)
        |
        |  partition into non-overlapping (window, window) blocks
        v
  Q  (B*nW, ws*ws, feat_dim)         <- with 2D rotary positional encoding
        ^
        |  cross-attention against the same K Gaussian tokens for every window
        |
  K, V (B,    K,    feat_dim)        <- broadcast across windows of the same B

The Gaussian set is global to the frame (no per-window assignment), so every
window sees every Gaussian token. This matches GSASR's design and keeps the
layer agnostic to canvas size — ``K`` may vary frame to frame.

Edge case: if ``K == 0`` (empty canvas, typical on the very first frame
before any Gaussians have been emitted) the fusion short-circuits to identity
and returns ``pixel_features`` unchanged. The cross-attention residual is
zero in that case anyway, but skipping the math also avoids creating
zero-shaped tensors that some kernels reject.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rope_freqs(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """RoPE inverse frequencies; ``dim`` is the per-axis half-dim (must be even)."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE half-dim must be even; got {dim}")
    half = dim // 2
    freqs = torch.arange(0, half, device=device, dtype=torch.float32)
    return 1.0 / (10000.0 ** (freqs / half)).to(dtype)


def _apply_rope_2d(q: torch.Tensor, ws: int) -> torch.Tensor:
    """Apply 2D rotary positional encoding to per-window query tokens.

    ``q`` shape: (Bn, heads, ws*ws, head_dim). RoPE rotates the largest
    prefix ``rope_dim`` of ``head_dim`` divisible by 4 (first quarter by
    row, second quarter by column). Any trailing channels are passed
    through unrotated, so this works for HAT-L's head_dim=30 where exact
    divisibility doesn't hold.
    """
    bn, h, n, d = q.shape
    assert n == ws * ws, f"window size mismatch: n={n}, ws={ws}"
    rope_dim = (d // 4) * 4
    if rope_dim == 0:
        return q
    half = rope_dim // 2
    inv = _rope_freqs(half, q.device, q.dtype)  # (half/2,)

    coords = torch.arange(ws, device=q.device, dtype=q.dtype)
    rows = coords.view(ws, 1).expand(ws, ws).reshape(-1)  # (ws*ws,)
    cols = coords.view(1, ws).expand(ws, ws).reshape(-1)

    def _rot(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x: (Bn, heads, n, half); pos: (n,); inv: (half/2,)
        angles = pos[None, None, :, None] * inv[None, None, None, :]  # (1,1,n,half/2)
        cos = angles.cos()
        sin = angles.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated_even = x1 * cos - x2 * sin
        rotated_odd = x1 * sin + x2 * cos
        out = torch.stack([rotated_even, rotated_odd], dim=-1)
        return out.flatten(-2)

    q_rope = q[..., :rope_dim]
    q_pass = q[..., rope_dim:]
    q_row, q_col = q_rope[..., :half], q_rope[..., half:]
    q_row = _rot(q_row, rows)
    q_col = _rot(q_col, cols)
    rotated = torch.cat([q_row, q_col], dim=-1)
    if q_pass.shape[-1] == 0:
        return rotated
    return torch.cat([rotated, q_pass], dim=-1)


class PixelGaussianFusion(nn.Module):
    """Window cross-attention from pixel queries to Gaussian K/V."""

    def __init__(
        self,
        feat_dim: int = 180,
        token_dim: int = 64,
        num_heads: int = 6,
        window_size: int = 16,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if feat_dim % num_heads != 0:
            raise ValueError(f"feat_dim={feat_dim} not divisible by num_heads={num_heads}")
        head_dim = feat_dim // num_heads
        self.feat_dim = feat_dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.scale = head_dim ** -0.5

        self.norm_q = nn.LayerNorm(feat_dim)
        self.norm_kv = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=True)
        self.k_proj = nn.Linear(token_dim, feat_dim, bias=True)
        self.v_proj = nn.Linear(token_dim, feat_dim, bias=True)
        self.out_proj = nn.Linear(feat_dim, feat_dim)

        self.norm_mlp = nn.LayerNorm(feat_dim)
        hidden = int(feat_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(
        self,
        pixel_features: torch.Tensor,
        gaussian_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if pixel_features.dim() != 4 or pixel_features.shape[1] != self.feat_dim:
            raise ValueError(
                f"pixel_features must be (B, {self.feat_dim}, H, W); "
                f"got {tuple(pixel_features.shape)}"
            )
        if gaussian_tokens.dim() != 3 or gaussian_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"gaussian_tokens must be (B, K, {self.token_dim}); "
                f"got {tuple(gaussian_tokens.shape)}"
            )
        b, c, h, w = pixel_features.shape
        b2, k, _ = gaussian_tokens.shape
        if b != b2:
            raise ValueError(
                f"batch mismatch: pixel B={b} vs gaussian B={b2}"
            )
        # Empty canvas (e.g. first frame before any Gaussians exist): identity.
        if k == 0:
            return pixel_features

        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(pixel_features, (0, pad_w, 0, pad_h))
        else:
            x = pixel_features
        h_p, w_p = x.shape[2], x.shape[3]
        n_win_h, n_win_w = h_p // ws, w_p // ws
        n_win = n_win_h * n_win_w

        # Q path: tokenize pixel windows.
        xn = x.permute(0, 2, 3, 1).contiguous()  # (B, H', W', C)
        xn = self.norm_q(xn)
        # (B, nH, ws, nW, ws, C) -> (B*nW, ws*ws, C)
        q_tokens = (
            xn.view(b, n_win_h, ws, n_win_w, ws, c)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(b * n_win, ws * ws, c)
        )

        # K, V path: project Gaussian tokens once, then expand to per-window batch.
        kv_norm = self.norm_kv(gaussian_tokens)  # (B, K, token_dim)
        k_tokens = self.k_proj(kv_norm)  # (B, K, feat_dim)
        v_tokens = self.v_proj(kv_norm)
        k_tokens = (
            k_tokens.unsqueeze(1)
            .expand(b, n_win, k, c)
            .reshape(b * n_win, k, c)
        )
        v_tokens = (
            v_tokens.unsqueeze(1)
            .expand(b, n_win, k, c)
            .reshape(b * n_win, k, c)
        )

        q = self.q_proj(q_tokens)  # (Bn, ws*ws, C)
        # (Bn, n, heads, head_dim) -> (Bn, heads, n, head_dim)
        q = q.view(b * n_win, ws * ws, self.num_heads, self.head_dim).transpose(1, 2)
        kk = k_tokens.view(b * n_win, k, self.num_heads, self.head_dim).transpose(1, 2)
        vv = v_tokens.view(b * n_win, k, self.num_heads, self.head_dim).transpose(1, 2)

        # 2D RoPE on Q only — keys are Gaussian tokens, not on the pixel grid,
        # so they don't carry a window position. Q gets its position injected
        # before the dot product.
        q = _apply_rope_2d(q, ws)

        attn = F.scaled_dot_product_attention(q, kk, vv, scale=self.scale)
        attn = attn.transpose(1, 2).reshape(b * n_win, ws * ws, c)
        attn = self.out_proj(attn)

        # Reassemble windows back to (B, H', W', C) and add residual.
        attn = attn.view(b, n_win_h, n_win_w, ws, ws, c)
        attn = attn.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h_p, w_p, c)

        out = x.permute(0, 2, 3, 1).contiguous() + attn  # residual on the attention path
        out = out + self.mlp(self.norm_mlp(out))  # residual on the MLP
        out = out.permute(0, 3, 1, 2).contiguous()

        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


__all__ = ["PixelGaussianFusion"]
