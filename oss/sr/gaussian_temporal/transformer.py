"""Multi-frame transformer over Gaussian tokens for v5 Gaussian-temporal SR.

Architecture
------------
Tokens fed to the transformer are the concatenation of:

    [ tile_feat tokens (current frame G-buffer)         ]
    [ Gaussian tokens from current field (alive only)   ]
    [ Gaussian tokens from each history frame (alive)   ]

Each token gets a learned linear input projection to ``d_model``. A learned
**frame-id embedding** is added (current=0, history t-1=1, ..., t-N=N, and a
distinct id for tile tokens). NOTE: the frame-id embedding is invariant to
token order WITHIN a frame -- it does NOT break permutation equivariance over
the per-frame Gaussian set.

Positional information is injected ONLY via 2D-RoPE applied to the (q, k) of
each self-attention layer. Positions:
    - Gaussian tokens: their μ ∈ R^2.
    - Tile tokens: tile-grid centers in (x, y) pixel-coord units.

There are NO learned positional embeddings; ``nn.MultiheadAttention`` is
order-equivariant by default, so shuffling Gaussian tokens of the current
field shuffles the corresponding outputs identically (within numerical
precision). This is exercised by ``test_permutation_equivariance``.

Output heads (linear from d_model) read only the current-field Gaussian token
slice and produce ``(dmu, dlog_scale, drot, dcolor)`` aligned to
``field_curr.alive``.

Parameter budget
----------------
With ``d_model=128, n_heads=4, n_layers=4, ffn_hidden=256`` the module sits in
the 400K--600K band required by the spec. The FFN hidden width is the main
knob; if the budget tightens we tune it here. Current FFN width: **256**.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from oss.sr.gaussian_temporal.gaussian_field import GaussianField


# --------------------------------------------------------------------------- #
# 2D Rotary positional embedding.                                             #
# --------------------------------------------------------------------------- #


def _build_inv_freq(half_dim: int, base: float = 10000.0, device=None, dtype=None) -> torch.Tensor:
    """Standard RoPE inverse-frequency vector of length ``half_dim``.

    Each entry corresponds to a 2D rotation pair (cos, sin) inside a single
    axis (x or y). For 2D RoPE we use HALF the head_dim per axis, so the
    inverse-freq vector has length ``head_dim // 4`` per axis.
    """
    idx = torch.arange(0, half_dim, dtype=torch.float32, device=device)
    inv = 1.0 / (base ** (idx / half_dim))
    if dtype is not None:
        inv = inv.to(dtype=dtype)
    return inv


def _rope_2d(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Apply 2D RoPE to a (B, H, T, D) tensor given (B, T, 2) positions.

    Splits the head_dim D into two halves; the first half is rotated by
    x-frequencies, the second half by y-frequencies. Within each half, pairs
    of consecutive channels (2k, 2k+1) form (cos, sin) rotation pairs.

    Args:
        x:         (B, n_heads, T, D) — q or k.
        positions: (B, T, 2)         — (x, y) in pixel coords.
        base:      RoPE frequency base.

    Returns:
        Rotated tensor with the same shape as ``x``.
    """
    b, h, t, d = x.shape
    if d % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE; got {d}")
    half = d // 2
    quarter = d // 4
    inv_freq = _build_inv_freq(quarter, base=base, device=x.device, dtype=x.dtype)  # (D/4,)

    # x-axis rotation for first half --------------------------------------- #
    px = positions[..., 0].unsqueeze(-1)  # (B, T, 1)
    py = positions[..., 1].unsqueeze(-1)
    angle_x = px * inv_freq                         # (B, T, D/4)
    angle_y = py * inv_freq                         # (B, T, D/4)
    cos_x = torch.cos(angle_x).unsqueeze(1).expand(b, h, t, quarter)
    sin_x = torch.sin(angle_x).unsqueeze(1).expand(b, h, t, quarter)
    cos_y = torch.cos(angle_y).unsqueeze(1).expand(b, h, t, quarter)
    sin_y = torch.sin(angle_y).unsqueeze(1).expand(b, h, t, quarter)

    x_first = x[..., :half]    # (B, H, T, D/2)
    x_second = x[..., half:]   # (B, H, T, D/2)

    # Within each half, channels split into even/odd pairs that form (cos, sin)
    # rotation operands. Standard RoPE pair-rotation:
    #     (a, b) -> (a*cos - b*sin, a*sin + b*cos)
    a1 = x_first[..., 0::2]
    b1 = x_first[..., 1::2]
    rot_first_a = a1 * cos_x - b1 * sin_x
    rot_first_b = a1 * sin_x + b1 * cos_x
    rotated_first = torch.stack([rot_first_a, rot_first_b], dim=-1).reshape(b, h, t, half)

    a2 = x_second[..., 0::2]
    b2 = x_second[..., 1::2]
    rot_second_a = a2 * cos_y - b2 * sin_y
    rot_second_b = a2 * sin_y + b2 * cos_y
    rotated_second = torch.stack([rot_second_a, rot_second_b], dim=-1).reshape(b, h, t, half)

    return torch.cat([rotated_first, rotated_second], dim=-1)


# --------------------------------------------------------------------------- #
# Attention layer with RoPE on (q, k).                                        #
# --------------------------------------------------------------------------- #


class _RoPEMultiheadAttention(nn.Module):
    """Custom MHA so we can apply 2D-RoPE on q and k before the dot product."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 4 != 0:
            raise ValueError(
                f"head_dim={self.head_dim} must be divisible by 4 for 2D RoPE; "
                f"adjust d_model or n_heads."
            )
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Self-attention with 2D-RoPE on q and k (NOT on v).

        Args:
            x:         (B, T, D).
            positions: (B, T, 2) — pixel-coord (x, y) per token.
        """
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = _rope_2d(q, positions)
        k = _rope_2d(k, positions)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(out)


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block: RoPE-MHA + FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _RoPEMultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.ffn(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Top-level module.                                                           #
# --------------------------------------------------------------------------- #


# Per-Gaussian "raw" feature dim (log_scale 2 + rotation 1 + color 3 + opacity 1).
_GAUSS_FEAT_DIM = 7


class GaussianMultiFrameTransformer(nn.Module):
    """Transformer attending over current tile features + multi-frame Gaussians.

    Forward signature::

        forward(field_curr, history, tile_features) -> dict[str, Tensor]

    Returns a dict aligned to ``field_curr.alive`` (size ``N_alive``):
        - ``dmu``        : (N_alive, 2)
        - ``dlog_scale`` : (N_alive, 2)
        - ``drot``       : (N_alive,)
        - ``dcolor``     : (N_alive, 3)

    Permutation equivariance: shuffling alive Gaussian rows of ``field_curr``
    produces an identically shuffled output. RoPE keyed on μ guarantees this
    even though positions matter -- as long as μ is permuted together with the
    feature vector, the attention dot products are unchanged up to the same
    permutation.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        history_len: int = 5,
        ffn_hidden: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.history_len = history_len

        # Input projections.
        self.gauss_in = nn.Linear(_GAUSS_FEAT_DIM, d_model)
        self.tile_in = nn.Linear(d_model, d_model)  # tile feats already at d_model

        # Frame-id embedding: id 0 = tile token; id 1 = current Gaussians;
        # id 2..(2 + history_len - 1) = history slots t-1, t-2, ...
        self.frame_embed = nn.Embedding(2 + history_len, d_model)

        self.blocks = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, ffn_hidden, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(d_model)

        # Output heads.
        self.head_mu = nn.Linear(d_model, 2)
        self.head_log_scale = nn.Linear(d_model, 2)
        self.head_rot = nn.Linear(d_model, 1)
        self.head_color = nn.Linear(d_model, 3)

        # Initialize output heads near-zero so initial dynamics are gentle.
        for h in (self.head_mu, self.head_log_scale, self.head_rot, self.head_color):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    # ------------------------------------------------------------------ #
    # Helpers.                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gauss_token_features(field: GaussianField, alive_idx: torch.Tensor) -> torch.Tensor:
        """Pack per-Gaussian features into (N_alive, _GAUSS_FEAT_DIM)."""
        log_scale = field.log_scale[alive_idx]                 # (N, 2)
        rotation = field.rotation[alive_idx].unsqueeze(-1)     # (N, 1)
        color = field.color[alive_idx]                         # (N, 3)
        opacity = field.opacity[alive_idx].unsqueeze(-1)       # (N, 1)
        return torch.cat([log_scale, rotation, color, opacity], dim=-1)

    @staticmethod
    def _tile_positions(h_t: int, w_t: int, device, dtype) -> torch.Tensor:
        """Tile-center positions in (x, y) order, flattened row-major. (h_t*w_t, 2)."""
        ys = torch.arange(h_t, device=device, dtype=dtype) + 0.5
        xs = torch.arange(w_t, device=device, dtype=dtype) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    # ------------------------------------------------------------------ #
    # Forward.                                                           #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        field_curr: GaussianField,
        history: List[GaussianField],
        tile_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if tile_features.dim() != 4:
            raise ValueError(
                f"tile_features must be (B, F, h, w); got shape {tuple(tile_features.shape)}"
            )
        b, f, h_t, w_t = tile_features.shape
        if b != 1:
            raise ValueError(f"This module is per-sample (B=1); got B={b}.")
        if f != self.d_model:
            raise ValueError(
                f"tile_features feature dim ({f}) must equal d_model ({self.d_model})."
            )

        device = tile_features.device
        dtype = tile_features.dtype

        # ---- tokens: tile features -------------------------------------- #
        tile_tokens = tile_features.flatten(2).transpose(1, 2)   # (1, h_t*w_t, F)
        tile_tokens = self.tile_in(tile_tokens)                  # (1, T_tile, D)
        tile_pos = self._tile_positions(h_t, w_t, device=device, dtype=dtype).unsqueeze(0)  # (1, T_tile, 2)
        tile_frame_id = torch.zeros(tile_tokens.shape[1], dtype=torch.long, device=device)  # frame id 0
        tile_tokens = tile_tokens + self.frame_embed(tile_frame_id).unsqueeze(0)

        token_chunks: List[torch.Tensor] = [tile_tokens]
        pos_chunks: List[torch.Tensor] = [tile_pos]

        # ---- tokens: current-frame alive Gaussians ---------------------- #
        alive_idx_curr = field_curr.alive.nonzero(as_tuple=True)[0]
        n_alive_curr = int(alive_idx_curr.numel())
        if n_alive_curr > 0:
            feats = self._gauss_token_features(field_curr, alive_idx_curr).to(dtype=dtype, device=device)
            tok = self.gauss_in(feats).unsqueeze(0)   # (1, N, D)
            mu_pos = field_curr.mu[alive_idx_curr].to(dtype=dtype, device=device).unsqueeze(0)
            frame_ids = torch.full((n_alive_curr,), 1, dtype=torch.long, device=device)
            tok = tok + self.frame_embed(frame_ids).unsqueeze(0)
            token_chunks.append(tok)
            pos_chunks.append(mu_pos)

        # ---- tokens: history Gaussians (newest first) ------------------- #
        for h_idx, prev in enumerate(history[: self.history_len]):
            alive_idx_prev = prev.alive.nonzero(as_tuple=True)[0]
            n_alive_prev = int(alive_idx_prev.numel())
            if n_alive_prev == 0:
                continue
            feats = self._gauss_token_features(prev, alive_idx_prev).to(dtype=dtype, device=device)
            tok = self.gauss_in(feats).unsqueeze(0)
            mu_pos = prev.mu[alive_idx_prev].to(dtype=dtype, device=device).unsqueeze(0)
            frame_ids = torch.full((n_alive_prev,), 2 + h_idx, dtype=torch.long, device=device)
            tok = tok + self.frame_embed(frame_ids).unsqueeze(0)
            token_chunks.append(tok)
            pos_chunks.append(mu_pos)

        x = torch.cat(token_chunks, dim=1)       # (1, T, D)
        positions = torch.cat(pos_chunks, dim=1)  # (1, T, 2)

        # ---- transformer body ----------------------------------------- #
        for block in self.blocks:
            x = block(x, positions)
        x = self.norm_out(x)

        # ---- output heads on current-frame alive slice only ----------- #
        n_alive_total_in_field = int(field_curr.alive.shape[0])
        # Allocate full-size outputs (rows for dead slots stay zero).
        out_dmu = x.new_zeros((n_alive_total_in_field, 2))
        out_dlog = x.new_zeros((n_alive_total_in_field, 2))
        out_drot = x.new_zeros((n_alive_total_in_field,))
        out_dcolor = x.new_zeros((n_alive_total_in_field, 3))

        if n_alive_curr > 0:
            tile_count = tile_tokens.shape[1]
            curr_slice = x[0, tile_count : tile_count + n_alive_curr]   # (N_alive, D)
            dmu = self.head_mu(curr_slice)
            dlog = self.head_log_scale(curr_slice)
            drot = self.head_rot(curr_slice).squeeze(-1)
            dcolor = self.head_color(curr_slice)
            out_dmu[alive_idx_curr] = dmu
            out_dlog[alive_idx_curr] = dlog
            out_drot[alive_idx_curr] = drot
            out_dcolor[alive_idx_curr] = dcolor

        # The acceptance contract says outputs are aligned to alive (i.e.
        # shape (N_alive, *)). Test asserts shape == capacity when ALL slots
        # are alive (capacity == N_alive in tests). We index by alive mask
        # to honor both: if every slot is alive the result is (capacity, *);
        # otherwise it's (N_alive, *) -- consistent with the spec.
        alive_mask = field_curr.alive
        return {
            "dmu": out_dmu[alive_mask],
            "dlog_scale": out_dlog[alive_mask],
            "drot": out_drot[alive_mask],
            "dcolor": out_dcolor[alive_mask],
        }


__all__ = ["GaussianMultiFrameTransformer"]
