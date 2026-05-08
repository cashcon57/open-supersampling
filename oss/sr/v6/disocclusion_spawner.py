# SPDX-License-Identifier: Apache-2.0
"""v6.2 disocclusion-driven Gaussian spawner.

Disocclusion mask: ``D(p) = 1[ |Z_t(p) - Z_{t-1}(p - MV(p))| > tau_z ]``.

Spawn logic:
  - For each pixel where ``D(p) = 1`` and ``residual > tau_R``:
    - ``xy_g = (px + 0.5, py + 0.5)``: exact pixel center, no learned offset.
    - ``velocity_g = MV(p)``.
    - ``feat_g, conic_abd_g = DGPDictionary(lr_features at p)``.

This decouples spawn-time grid alignment from sub-pixel positioning. The
canvas warp and motion-vector advection move Gaussians off-grid naturally over
time, avoiding the integer-pixel-aligned local minimum that produced the
lambda=2 px stippling artifact in v6.1-pico-001.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from oss.sr.v6.dgp_dictionary import DGPDictionary


class DisocclusionSpawner(nn.Module):
    """Hard-spawn Gaussians at disoccluded pixel centers."""

    def __init__(
        self,
        feat_dim: int = 64,
        dgp_M: int = 16,
        max_births_per_frame: int = 256,
        max_births_per_tile: int = 4,
        disocclusion_depth_threshold: float = 0.05,
        residual_threshold: float = 0.02,
    ) -> None:
        super().__init__()
        if feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive; got {feat_dim}")
        if max_births_per_frame < 0:
            raise ValueError(
                "max_births_per_frame must be non-negative; "
                f"got {max_births_per_frame}"
            )
        if max_births_per_tile <= 0:
            raise ValueError(
                f"max_births_per_tile must be positive; got {max_births_per_tile}"
            )

        self.dgp = DGPDictionary(M=dgp_M, feat_dim=feat_dim)
        self.max_births_per_frame = int(max_births_per_frame)
        self.max_births_per_tile = int(max_births_per_tile)
        self.tau_z = float(disocclusion_depth_threshold)
        self.tau_r = float(residual_threshold)

    def compute_disocclusion_mask(
        self,
        depth_t: torch.Tensor,
        depth_prev: torch.Tensor,
        MV: torch.Tensor,
    ) -> torch.Tensor:
        """Return a binary ``(B, 1, H, W)`` mask where one means disoccluded."""
        if depth_t.shape != depth_prev.shape:
            raise ValueError(
                "depth_t and depth_prev must have the same shape; "
                f"got {tuple(depth_t.shape)} and {tuple(depth_prev.shape)}"
            )
        if depth_t.ndim != 4 or depth_t.shape[1] != 1:
            raise ValueError(
                "depth tensors must have shape (B, 1, H, W); "
                f"got {tuple(depth_t.shape)}"
            )
        if MV.shape != (depth_t.shape[0], 2, depth_t.shape[2], depth_t.shape[3]):
            raise ValueError(
                "MV must have shape "
                f"{(depth_t.shape[0], 2, depth_t.shape[2], depth_t.shape[3])}; "
                f"got {tuple(MV.shape)}"
            )
        depth_prev_warped = self._warp_prev_depth(depth_prev, MV)
        return ((depth_t - depth_prev_warped).abs() > self.tau_z).to(depth_t.dtype)

    def _warp_prev_depth(
        self,
        depth_prev: torch.Tensor,
        MV: torch.Tensor,
    ) -> torch.Tensor:
        """Sample previous depth at ``p - MV(p)`` with zero for offscreen history."""
        B, _, H, W = depth_prev.shape
        y = torch.arange(H, device=depth_prev.device, dtype=depth_prev.dtype)
        x = torch.arange(W, device=depth_prev.device, dtype=depth_prev.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(B, H, W, 2)
        flow = MV.permute(0, 2, 3, 1).to(
            device=depth_prev.device,
            dtype=depth_prev.dtype,
        )
        src = base - flow
        if W > 1:
            grid_x = 2.0 * src[..., 0] / float(W - 1) - 1.0
        else:
            grid_x = torch.zeros_like(src[..., 0])
        if H > 1:
            grid_y = 2.0 * src[..., 1] / float(H - 1) - 1.0
        else:
            grid_y = torch.zeros_like(src[..., 1])
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return F.grid_sample(
            depth_prev,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def forward(
        self,
        depth_t: torch.Tensor,
        depth_prev: torch.Tensor,
        MV: torch.Tensor,
        lr_features: torch.Tensor,
        residual: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return new-Gaussian state tensors selected from disoccluded pixels.

        Args:
            depth_t: ``(B, 1, H, W)`` current depth.
            depth_prev: ``(B, 1, H, W)`` previous depth.
            MV: ``(B, 2, H, W)`` per-pixel motion vector.
            lr_features: ``(B, feat_dim, H, W)`` per-pixel features.
            residual: ``(B, 1, H, W)`` LR-vs-canvas-render residual.
        """
        if lr_features.ndim != 4:
            raise ValueError(
                f"lr_features must have shape (B, C, H, W); got {tuple(lr_features.shape)}"
            )
        B, feat_dim, H, W = lr_features.shape
        if depth_t.shape != (B, 1, H, W):
            raise ValueError(
                f"depth_t must have shape {(B, 1, H, W)}; got {tuple(depth_t.shape)}"
            )
        if depth_prev.shape != (B, 1, H, W):
            raise ValueError(
                "depth_prev must have shape "
                f"{(B, 1, H, W)}; got {tuple(depth_prev.shape)}"
            )
        if MV.shape != (B, 2, H, W):
            raise ValueError(f"MV must have shape {(B, 2, H, W)}; got {tuple(MV.shape)}")
        if residual.shape != (B, 1, H, W):
            raise ValueError(
                f"residual must have shape {(B, 1, H, W)}; got {tuple(residual.shape)}"
            )

        device = lr_features.device
        dtype = lr_features.dtype
        disocc = self.compute_disocclusion_mask(depth_t, depth_prev, MV)
        residual_gate = (residual > self.tau_r).to(disocc.dtype)
        priority = (disocc * residual_gate).squeeze(1)

        K = min(self.max_births_per_frame, H * W)
        if K == 0 or B == 0:
            feat = torch.zeros(0, feat_dim, device=device, dtype=dtype)
            conic_abd = torch.zeros(0, 3, device=device, dtype=dtype)
            scale = torch.zeros(0, device=device, dtype=dtype)
            return {
                "xy": torch.zeros(0, 2, device=device, dtype=dtype),
                "velocity": torch.zeros(0, 2, device=device, dtype=MV.dtype),
                "conic_abd": conic_abd,
                "feat": feat,
                "scale": scale,
                "n_births": torch.tensor(0, device=device, dtype=torch.long),
            }

        flat = priority.reshape(B, -1)
        topk_vals, topk_idx = flat.topk(K, dim=-1)
        valid = topk_vals > 0
        py_all = topk_idx // W
        px_all = topk_idx % W

        all_xy: list[torch.Tensor] = []
        all_vel: list[torch.Tensor] = []
        all_feat: list[torch.Tensor] = []
        for b in range(B):
            py = py_all[b][valid[b]]
            px = px_all[b][valid[b]]
            xy = torch.stack((px.to(dtype) + 0.5, py.to(dtype) + 0.5), dim=-1)
            velocity = MV[b, :, py, px].transpose(0, 1)
            feat = lr_features[b, :, py, px].transpose(0, 1)
            all_xy.append(xy)
            all_vel.append(velocity)
            all_feat.append(feat)

        xy = torch.cat(all_xy, dim=0)
        velocity = torch.cat(all_vel, dim=0)
        feat = torch.cat(all_feat, dim=0)
        if feat.numel() > 0:
            conic_abd, scale = self.dgp(feat)
        else:
            conic_abd = torch.zeros(0, 3, device=device, dtype=dtype)
            scale = torch.zeros(0, device=device, dtype=dtype)

        return {
            "xy": xy,
            "velocity": velocity,
            "conic_abd": conic_abd,
            "feat": feat,
            "scale": scale,
            "n_births": torch.tensor(xy.shape[0], device=device, dtype=torch.long),
        }
