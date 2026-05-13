"""Backbone -> Canvas spawner for v7.

Decodes backbone HR feature maps into K Gaussian parameter predictions
per frame. Tile-based sampling: per HR tile of size `tile_size`,
predict `k_per_tile` Gaussians' (xy_offset, cov_raw, feature, opacity)
parameters. The xy_offset is sigmoid-bounded within the tile and added
to the tile anchor, so Gaussians distribute spatially without
clustering at the image's top-left.

Currently single-batch only (B=1). v7 training treats each
trajectory's canvas as a per-rank state, matching v6.x conventions.
A future refactor can extend to batched canvases.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Per-Gaussian output channel count: 2 xy + 6 cov_raw + R feature + 1 opacity = 9 + R
def _params_per_gaussian(latent_rank: int) -> int:
    return 9 + int(latent_rank)


class BackboneSpawner(nn.Module):
    """Tile-based learnable spawner.

    Args:
        feat_dim:     backbone HR feature channel count
        latent_rank:  R (canvas feature dim)
        k_per_tile:   how many Gaussians to predict per HR tile
        tile_size:    HR tile side length in pixels
        hidden_dim:   intermediate channel width
        opacity_init_bias:  init bias on the opacity logit (default
                             -3 so opacities start ~0.05; gentle on
                             the canvas at training start)
    """

    def __init__(
        self,
        feat_dim: int,
        latent_rank: int,
        k_per_tile: int = 4,
        tile_size: int = 16,
        hidden_dim: int = 64,
        opacity_init_bias: float = -3.0,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.latent_rank = int(latent_rank)
        self.k_per_tile = int(k_per_tile)
        self.tile_size = int(tile_size)
        out_per_gaussian = _params_per_gaussian(latent_rank)
        out_channels = self.k_per_tile * out_per_gaussian

        # Per-tile pooling + small MLP via 1x1 convs.
        self.pool = nn.AvgPool2d(kernel_size=tile_size, stride=tile_size)
        self.mix = nn.Conv2d(feat_dim, hidden_dim, kernel_size=1)
        self.out = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

        # Initialize the output bias so opacities start near zero.
        # The opacity sits at the last position of each per-Gaussian
        # output slot. Walk the bias and set those slots.
        with torch.no_grad():
            bias = self.out.bias.view(self.k_per_tile, out_per_gaussian)
            bias[:, -1] = opacity_init_bias  # opacity logit
            # Cholesky diagonal params (l00, l11, l22) live at indices 2,
            # 4 (l00, l11) and 7 (l22) of cov_raw -> within per-Gaussian
            # slot indices 4, 6, 9 (after 2 xy offsets). Set them to log(2)
            # so initial covariance diagonals are exp(log(2)) = 2 -> scale 2 px.
            # This avoids degenerate near-zero scales at initialization.
            bias[:, 4] = float(torch.tensor(2.0).log())  # l00 raw -> diag entry exp(log(2)) = 2
            bias[:, 6] = float(torch.tensor(2.0).log())  # l11 raw
            bias[:, 9] = float(torch.tensor(2.0).log())  # l22 raw (t-axis sigma)
            self.out.bias.copy_(bias.view(-1))

    def forward(self, refined_hr: torch.Tensor, t: float) -> dict:
        """Decode (B, feat_dim, H, W) -> Gaussian params for B=1.

        Returns dict with:
          positions  (K, 3)    HR xy coords + t
          cov_raw    (K, 6)    Cholesky raw params
          features   (K, R)    per-Gaussian feature vec
          opacity    (K,)      sigmoid-bounded in (0, 1)

        Where K = k_per_tile * (H / tile_size) * (W / tile_size).
        """
        if refined_hr.shape[0] != 1:
            raise ValueError(
                f"BackboneSpawner currently supports B=1 only; got B={refined_hr.shape[0]}. "
                f"Multi-batch canvases land in a future refactor."
            )

        b, f, h, w = refined_hr.shape
        if h % self.tile_size != 0 or w % self.tile_size != 0:
            raise ValueError(
                f"refined_hr shape ({h}, {w}) must be divisible by tile_size {self.tile_size}"
            )
        n_tiles_h = h // self.tile_size
        n_tiles_w = w // self.tile_size

        x = self.pool(refined_hr)         # (1, F, n_h, n_w)
        x = F.gelu(self.mix(x))            # (1, hidden, n_h, n_w)
        x = self.out(x)                    # (1, k * params_per_g, n_h, n_w)

        # Reshape to per-Gaussian rows.
        params_per_g = _params_per_gaussian(self.latent_rank)
        x = x.view(1, self.k_per_tile, params_per_g, n_tiles_h, n_tiles_w)
        # (n_h, n_w, k, params)
        x = x.permute(0, 3, 4, 1, 2).reshape(-1, params_per_g)
        K = x.shape[0]

        # Decode each parameter slot.
        xy_offset = torch.sigmoid(x[:, 0:2]) * float(self.tile_size)
        cov_raw = x[:, 2:8]
        feature = x[:, 8 : 8 + self.latent_rank]
        opacity = torch.sigmoid(x[:, 8 + self.latent_rank])

        # Tile anchors at HR coords. Each Gaussian within tile (i, j)
        # gets anchor = (j * tile_size, i * tile_size).
        device = x.device
        dtype = x.dtype
        ys = torch.arange(n_tiles_h, device=device, dtype=dtype) * float(self.tile_size)
        xs = torch.arange(n_tiles_w, device=device, dtype=dtype) * float(self.tile_size)
        anchor_y, anchor_x = torch.meshgrid(ys, xs, indexing="ij")
        # Each tile contributes k_per_tile rows; tile_anchor repeats k times.
        anchor_xy = torch.stack([anchor_x.flatten(), anchor_y.flatten()], dim=-1)
        # (n_h*n_w, 2) -> (n_h*n_w*k, 2) by repeating
        anchor_xy = anchor_xy.unsqueeze(1).expand(-1, self.k_per_tile, -1).reshape(-1, 2)

        xy_abs = anchor_xy + xy_offset       # (K, 2)
        t_vec = torch.full((K, 1), float(t), device=device, dtype=dtype)
        positions = torch.cat([xy_abs, t_vec], dim=-1)   # (K, 3)

        return {
            "positions": positions,
            "cov_raw": cov_raw,
            "features": feature,
            "opacity": opacity,
        }
