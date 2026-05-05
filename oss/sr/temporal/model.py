"""TemporalSRModel — v4 backbone + temporal warp + disocclusion + head.

Wires together:
- v4 SRCNNSimple backbone (in_channels=12, scale=2).
- Backward warp of prev-HR by motion vectors.
- DisocclusionGate (alpha, beta, gamma).
- TemporalHead conv stack.

Plus utilities: warm-start from v4 checkpoint, backbone-freeze toggle,
and ``make_first_frame_prev_hr`` for sequence boundary handling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from oss.sr import build_sr_model
from oss.sr.temporal.disocclusion import DisocclusionGate
from oss.sr.temporal.temporal_head import TemporalHead
from oss.sr.temporal.warp import warp_prev_hr


def make_first_frame_prev_hr(lr_rgb: torch.Tensor, scale: int) -> torch.Tensor:
    """Bilinear-upscale LR RGB to use as the synthetic prev-HR on frame 0."""
    if lr_rgb.dim() != 4 or lr_rgb.shape[1] != 3:
        raise ValueError(f"lr_rgb must be (B, 3, H, W); got {tuple(lr_rgb.shape)}")
    return F.interpolate(lr_rgb, scale_factor=float(scale), mode="bilinear", align_corners=False)


class TemporalSRModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        scale: int = 2,
        tier: str = "standard",
        backbone_kind: str = "simple",
        zero_gbuffer_into_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        # Zero non-RGB channels into the backbone only when the backbone was
        # trained on SRGD (which had no real G-buffers). For from-scratch
        # training on data WITH real depth/motion/normals (e.g. TartanAir),
        # leave this False so the backbone can actually learn from those
        # channels. ``load_v4_warm_start`` flips this to True automatically.
        self.zero_gbuffer_into_backbone = zero_gbuffer_into_backbone
        self.backbone = build_sr_model(
            model_kind=backbone_kind, tier=tier, in_channels=in_channels, scale=scale
        )
        self.gate = DisocclusionGate()
        self.head = TemporalHead()

    def forward(
        self,
        lr_inputs: torch.Tensor,
        prev_hr: torch.Tensor,
        depth_hr_curr: torch.Tensor,
        depth_hr_prev: torch.Tensor,
        motion_lr: torch.Tensor,
    ) -> torch.Tensor:
        if self.zero_gbuffer_into_backbone:
            # Match the SRGD training distribution v4 was warm-started from:
            # backbone sees only RGB + a constant prior (normals[2]=1.0
            # default-up). Real G-buffers still flow into warp + gate + head
            # via their dedicated arguments.
            lr_for_backbone = torch.zeros_like(lr_inputs)
            lr_for_backbone[:, :3] = lr_inputs[:, :3]
            if lr_for_backbone.shape[1] >= 7:
                lr_for_backbone[:, 6] = 1.0
            current_sr = self.backbone(lr_for_backbone)
        else:
            current_sr = self.backbone(lr_inputs)
        warped_prev = warp_prev_hr(prev_hr, motion_lr, scale=self.scale)
        disoccl = self.gate(
            depth_curr=depth_hr_curr, depth_prev=depth_hr_prev,
            motion_lr=motion_lr, scale=self.scale,
        )
        return self.head(
            current_sr=current_sr, warped_prev=warped_prev,
            disocclusion=disoccl, depth_hr=depth_hr_curr,
        )

    def freeze_backbone(self, freeze: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)

    @classmethod
    def load_v4_warm_start(
        cls,
        ckpt_path: Path,
        in_channels: int = 12,
        scale: int = 2,
        device: str | torch.device = "cpu",
    ) -> "TemporalSRModel":
        ck: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ck.get("args", {})
        tier = saved.get("tier", "standard")
        backbone_kind = "rrdb" if saved.get("sr_backbone") == "rrdb" else "simple"
        model = cls(
            in_channels=in_channels, scale=scale, tier=tier, backbone_kind=backbone_kind,
            zero_gbuffer_into_backbone=True,
        )
        missing, unexpected = model.backbone.load_state_dict(ck["sr_model"], strict=True)
        if missing or unexpected:
            raise RuntimeError(f"v4 warm-start mismatch: missing={missing}, unexpected={unexpected}")
        return model


__all__ = ["TemporalSRModel", "make_first_frame_prev_hr"]
