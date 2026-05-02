"""OutputHead — Sprint 4 / T4.2.

Converts the GaussianParamNetwork's raw tile-wise output tensor into a
``GaussianBatch`` ready for the Sprint-1 Rasterizer.

Layout of the raw tensor (per the param_net spec):
    (B, K * per_gauss_ch, H_tile, W_tile)

with per-Gaussian channels:
    [Δμx, Δμy, log_scale, rot_offset, bank_logits[bank_size], color[3]]

Decoded per-Gaussian quantities:
    μx = (tile_x + 0.5) * tile_size + tile_size * tanh(Δμx)
    μy = (tile_y + 0.5) * tile_size + tile_size * tanh(Δμy)
        → centers stay inside or just outside the tile (±tile_size).
    scale_factor = exp(clamp(log_scale, ±ln(8)))   ∈ [1/8, 8]
    rotation = bank_theta + rot_offset_clipped     (small rotational refinement)
    bank_weights = softmax(bank_logits + gbuf_bias)  (G-buffer-conditioned)
    sx, sy, theta = CovariancePriorBank(bank_weights) ∗ scale_factor (sx, sy)
                    + rot_offset (theta)
    color = sigmoid(color)                         ∈ [0, 1] (LDR mode)

NOTE: HDR mode is selectable via ``color_activation`` ("sigmoid" or "softplus").
softplus is unbounded above for HDR training.

Anisotropic G-buffer-conditioned covariance (validation memo 2026-05-01,
Decision 2): when ``enable_gbuffer_bias=True`` and the caller passes per-pixel
``depth`` + ``normals`` to ``decode``, a small linear head maps per-tile
(mean normal, mean depth gradient) → an additive bias on the bank logits.
The bias is shared across all K Gaussians in the same tile. This biases the
softmax toward elongated bank entries aligned with edges/silhouettes, which
the D1 denoising test showed reduces over-smoothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from oss.gaussian.network.param_net import per_gaussian_channels
from oss.gaussian.network.prior_bank import CovariancePriorBank
from oss.gaussian.renderer import GaussianBatch


# Channel widths of the G-buffer feature vector fed into the bias head:
# 3 channels of mean normal + 2 channels of mean depth gradient (∂z/∂x, ∂z/∂y).
_GBUF_FEATURE_CHANNELS: int = 5


@dataclass(frozen=True)
class DecodedParams:
    """All decoded per-Gaussian tensors. Shapes are (B, N) or (B, N, ...)."""
    xy: torch.Tensor            # (B, N, 2)
    scale: torch.Tensor         # (B, N, 2)
    rot: torch.Tensor           # (B, N)
    feat: torch.Tensor          # (B, N, F)
    bank_weights: torch.Tensor  # (B, N, K)


class GBufferCovarianceBias(nn.Module):
    """Per-tile G-buffer → additive bank-logit bias.

    Input  per tile: (mean_nx, mean_ny, mean_nz, mean_dz_dx, mean_dz_dy)
    Output per tile: (bank_size,) additive logit bias

    Implemented as a single linear layer initialised to zero so that an
    untrained model behaves identically to ``enable_gbuffer_bias=False``
    (graceful initialisation — bias contribution starts at 0 and the
    network learns when/how to use it).
    """

    def __init__(self, bank_size: int) -> None:
        super().__init__()
        if bank_size < 2:
            raise ValueError(f"bank_size must be >=2; got {bank_size}")
        self.bank_size = bank_size
        # Single linear head: (5,) → (bank_size,).
        self.proj = nn.Linear(_GBUF_FEATURE_CHANNELS, bank_size, bias=True)
        # Zero init — see class docstring. Network learns the bias from scratch.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Map per-tile (B, Ht, Wt, 5) features → (B, Ht, Wt, bank_size) bias."""
        if feat.shape[-1] != _GBUF_FEATURE_CHANNELS:
            raise ValueError(
                f"expected last dim {_GBUF_FEATURE_CHANNELS}; got {feat.shape[-1]}"
            )
        return self.proj(feat)


def _depth_gradient(depth: torch.Tensor) -> torch.Tensor:
    """Central-difference depth gradient with edge replication.

    Args:
        depth: (B, 1, H, W).
    Returns:
        (B, 2, H, W) — channel 0 is ∂z/∂x, channel 1 is ∂z/∂y.
    """
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError(f"expected (B,1,H,W); got {tuple(depth.shape)}")
    d = depth[:, 0]  # (B, H, W)
    dx = torch.zeros_like(d)
    dy = torch.zeros_like(d)
    dx[:, :, 1:-1] = d[:, :, 2:] - d[:, :, :-2]
    dy[:, 1:-1, :] = d[:, 2:, :] - d[:, :-2, :]
    return torch.stack([dx, dy], dim=1)


def _per_tile_avg(x: torch.Tensor, tile_size: int) -> torch.Tensor:
    """Average pool ``x`` (B, C, H, W) over non-overlapping ``tile_size`` blocks.

    Output shape: (B, C, H // tile_size, W // tile_size).
    """
    if x.dim() != 4:
        raise ValueError(f"expected (B,C,H,W); got {tuple(x.shape)}")
    B, C, H, W = x.shape
    if H % tile_size or W % tile_size:
        raise ValueError(
            f"H={H}, W={W} must be exact multiples of tile_size={tile_size}"
        )
    return F.avg_pool2d(x, kernel_size=tile_size, stride=tile_size)


class OutputHead(nn.Module):
    """Decode raw network output into renderer-ready Gaussian parameters.

    Args:
        bank: a ``CovariancePriorBank`` (must have the same bank_size that
            produced the raw logits).
        tile_size: 16 (must match renderer.TILE_SIZE).
        k_per_tile: number of Gaussians per tile (must match the param net).
        color_activation: "sigmoid" → LDR colors in [0, 1].
                          "softplus" → unbounded HDR-friendly.
        log_scale_clip: absolute clip on log_scale; final scale_factor is
            in [exp(-clip), exp(+clip)]. Default ln(8) ≈ 2.08.
        enable_gbuffer_bias: when True, ``decode`` accepts optional ``depth``
            and ``normals`` G-buffers; a small linear head maps per-tile
            (mean normal, mean depth gradient) → additive bias on the bank
            logits, biasing the softmax toward edge-aligned anisotropic
            bank entries. Default False (backward compat). Even when True,
            if neither depth nor normals are passed to ``decode``, the bias
            contribution is zero.
    """

    def __init__(
        self,
        bank: CovariancePriorBank,
        tile_size: int = 16,
        k_per_tile: int = 5,
        color_activation: str = "sigmoid",
        log_scale_clip: float = math.log(8.0),
        enable_gbuffer_bias: bool = False,
    ) -> None:
        super().__init__()
        if color_activation not in ("sigmoid", "softplus"):
            raise ValueError(
                f"color_activation must be 'sigmoid' or 'softplus'; "
                f"got {color_activation!r}"
            )
        self.bank = bank
        self.tile_size = tile_size
        self.k_per_tile = k_per_tile
        self.color_activation = color_activation
        self.log_scale_clip = float(log_scale_clip)
        self.enable_gbuffer_bias = enable_gbuffer_bias
        if enable_gbuffer_bias:
            self.gbuffer_bias = GBufferCovarianceBias(bank_size=bank.bank_size)
        else:
            self.gbuffer_bias = None

    def _compute_gbuffer_bias(
        self,
        depth: torch.Tensor | None,
        normals: torch.Tensor | None,
        B: int,
        Ht: int,
        Wt: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return per-tile bank-logit bias of shape (B, Ht, Wt, bank_size).

        Returns zeros when the bias is disabled or no G-buffer was provided.
        """
        if not self.enable_gbuffer_bias or self.gbuffer_bias is None:
            return torch.zeros(B, Ht, Wt, self.bank.bank_size,
                               device=device, dtype=dtype)
        if depth is None and normals is None:
            return torch.zeros(B, Ht, Wt, self.bank.bank_size,
                               device=device, dtype=dtype)

        ts = self.tile_size
        H_lr = Ht * ts
        W_lr = Wt * ts

        # Mean normal per tile, defaults to 0 if not provided.
        if normals is not None:
            if normals.dim() != 4 or normals.shape[1] != 3:
                raise ValueError(
                    f"normals must be (B,3,H,W); got {tuple(normals.shape)}"
                )
            if normals.shape[-2:] != (H_lr, W_lr):
                raise ValueError(
                    f"normals spatial {tuple(normals.shape[-2:])} != "
                    f"(H_lr={H_lr}, W_lr={W_lr})"
                )
            normal_tile = _per_tile_avg(normals.to(dtype), ts)  # (B, 3, Ht, Wt)
        else:
            normal_tile = torch.zeros(B, 3, Ht, Wt, device=device, dtype=dtype)

        # Mean depth gradient per tile, defaults to 0 if not provided.
        if depth is not None:
            if depth.dim() != 4 or depth.shape[1] != 1:
                raise ValueError(
                    f"depth must be (B,1,H,W); got {tuple(depth.shape)}"
                )
            if depth.shape[-2:] != (H_lr, W_lr):
                raise ValueError(
                    f"depth spatial {tuple(depth.shape[-2:])} != "
                    f"(H_lr={H_lr}, W_lr={W_lr})"
                )
            grad = _depth_gradient(depth.to(dtype))            # (B, 2, H, W)
            grad_tile = _per_tile_avg(grad, ts)                # (B, 2, Ht, Wt)
        else:
            grad_tile = torch.zeros(B, 2, Ht, Wt, device=device, dtype=dtype)

        # (B, 5, Ht, Wt) → (B, Ht, Wt, 5) → linear → (B, Ht, Wt, bank_size)
        feat = torch.cat([normal_tile, grad_tile], dim=1)
        feat = feat.permute(0, 2, 3, 1).contiguous()
        return self.gbuffer_bias(feat)

    def decode(
        self,
        raw: torch.Tensor,
        depth: torch.Tensor | None = None,
        normals: torch.Tensor | None = None,
    ) -> DecodedParams:
        """Decode the raw output tensor into ``DecodedParams``.

        Args:
            raw: (B, K * per_g, Ht, Wt) network output.
            depth: optional (B, 1, H_lr, W_lr) G-buffer; used only when
                ``enable_gbuffer_bias=True``.
            normals: optional (B, 3, H_lr, W_lr) G-buffer; used only when
                ``enable_gbuffer_bias=True``.
        """
        if raw.dim() != 4:
            raise ValueError(f"expected (B, C, H, W); got {tuple(raw.shape)}")
        B, C, Ht, Wt = raw.shape
        per_g = per_gaussian_channels(self.bank.bank_size)
        if C != self.k_per_tile * per_g:
            raise ValueError(
                f"raw channels ({C}) != k_per_tile ({self.k_per_tile}) * "
                f"per_gaussian_channels ({per_g})"
            )

        # Reshape to (B, K, per_g, Ht, Wt) → (B, Ht, Wt, K, per_g)
        x = raw.view(B, self.k_per_tile, per_g, Ht, Wt).permute(0, 3, 4, 1, 2)
        # Slice channels.
        d_xy = x[..., 0:2]                      # (B, Ht, Wt, K, 2)
        log_scale = x[..., 2]                   # (B, Ht, Wt, K)
        rot_off = x[..., 3]                     # (B, Ht, Wt, K)
        bank_logits = x[..., 4:4 + self.bank.bank_size]   # (B, Ht, Wt, K, K_bank)
        color = x[..., 4 + self.bank.bank_size:4 + self.bank.bank_size + 3]
        # color shape: (B, Ht, Wt, K, 3)

        # ---- Position decode -------------------------------------------------
        # Tile centers in pixel space.
        device = raw.device
        dtype = raw.dtype
        ts = float(self.tile_size)
        ys = (torch.arange(Ht, device=device, dtype=dtype) + 0.5) * ts
        xs = (torch.arange(Wt, device=device, dtype=dtype) + 0.5) * ts
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")  # (Ht, Wt)
        # Broadcast tile centers across (B, K).
        cx = cx[None, :, :, None].expand(B, Ht, Wt, self.k_per_tile)
        cy = cy[None, :, :, None].expand(B, Ht, Wt, self.k_per_tile)

        # ±tile_size offset envelope; tanh keeps the center within ±1 tile.
        dx = torch.tanh(d_xy[..., 0]) * ts
        dy = torch.tanh(d_xy[..., 1]) * ts
        mu_x = cx + dx
        mu_y = cy + dy

        # ---- Scale decode ----------------------------------------------------
        scale_factor = torch.exp(log_scale.clamp(-self.log_scale_clip,
                                                 self.log_scale_clip))

        # ---- Bank softmax (G-buffer-biased when enabled) --------------------
        # Per-tile bias is shared across the K Gaussians in that tile.
        bias = self._compute_gbuffer_bias(
            depth=depth, normals=normals,
            B=B, Ht=Ht, Wt=Wt, device=device, dtype=dtype,
        )  # (B, Ht, Wt, bank_size)
        # Broadcast to (B, Ht, Wt, K, bank_size).
        bank_w = F.softmax(bank_logits + bias.unsqueeze(-2), dim=-1)
        sx_bank, sy_bank, theta_bank, _cov = self.bank(bank_w)
        # Apply scale_factor multiplicatively to sx/sy; rot_offset adds in.
        sx = sx_bank * scale_factor
        sy = sy_bank * scale_factor
        # Clamp rotation offset to ±π/4 so it can't undo the bank's selection.
        rot = theta_bank + torch.tanh(rot_off) * (math.pi / 4.0)

        # ---- Color -----------------------------------------------------------
        if self.color_activation == "sigmoid":
            feat = torch.sigmoid(color)
        else:
            feat = F.softplus(color)

        # ---- Flatten (Ht, Wt, K) → N -----------------------------------------
        N = Ht * Wt * self.k_per_tile
        mu_x = mu_x.reshape(B, N)
        mu_y = mu_y.reshape(B, N)
        xy = torch.stack([mu_x, mu_y], dim=-1)            # (B, N, 2)
        scale = torch.stack([sx.reshape(B, N), sy.reshape(B, N)], dim=-1)  # (B, N, 2)
        rot = rot.reshape(B, N)
        feat = feat.reshape(B, N, 3)
        bank_weights = bank_w.reshape(B, N, self.bank.bank_size)
        return DecodedParams(xy=xy, scale=scale, rot=rot, feat=feat,
                             bank_weights=bank_weights)

    # ------------------------------------------------------------------------
    def to_gaussian_batch(
        self,
        raw: torch.Tensor,
        batch_index: int = 0,
        depth: torch.Tensor | None = None,
        normals: torch.Tensor | None = None,
    ) -> GaussianBatch:
        """Decode a single sample of the raw tensor into a ``GaussianBatch``.

        The Sprint 1 ``GaussianBatch`` is per-image (no batch dimension).
        Callers wanting B>1 should iterate or write a batched renderer wrapper.
        """
        if not (0 <= batch_index < raw.shape[0]):
            raise IndexError(f"batch_index {batch_index} out of range for B={raw.shape[0]}")
        d = self.decode(raw, depth=depth, normals=normals)
        return GaussianBatch(
            xy=d.xy[batch_index],
            scale=d.scale[batch_index],
            rot=d.rot[batch_index],
            feat=d.feat[batch_index],
        )


__all__ = ["OutputHead", "DecodedParams", "GBufferCovarianceBias"]
