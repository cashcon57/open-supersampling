"""Stateless export wrapper for v5 pixel-temporal SR.

The runtime inference engine carries previous-frame state internally, which is
not convenient for ONNX/TensorRT-style export. This wrapper exposes every
temporal input explicitly and returns both the HR output and the disocclusion
mask so deployment code can visualize/reset around scene-cut behavior without
changing the model math.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from oss.sr.temporal.model import TemporalSRModel
from oss.sr.temporal.warp import warp_prev_hr


class TemporalSRModelStateless(nn.Module):
    """Thin stateless wrapper around :class:`TemporalSRModel`.

    Forward is intentionally the same computation as ``TemporalSRModel.forward``
    with one extra returned tensor: the disocclusion mask produced by the gate.
    """

    def __init__(self, model: TemporalSRModel) -> None:
        super().__init__()
        self.model = model

    @property
    def scale(self) -> int:
        return int(self.model.scale)

    @property
    def in_channels(self) -> int:
        return int(self.model.in_channels)

    def forward(
        self,
        lr_inputs: torch.Tensor,
        prev_hr: torch.Tensor,
        depth_hr_curr: torch.Tensor,
        depth_hr_prev: torch.Tensor,
        motion_lr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model.zero_gbuffer_into_backbone:
            lr_for_backbone = torch.zeros_like(lr_inputs)
            lr_for_backbone[:, :3] = lr_inputs[:, :3]
            if lr_for_backbone.shape[1] >= 7:
                lr_for_backbone[:, 6] = 1.0
            current_sr = self.model.backbone(lr_for_backbone)
        else:
            current_sr = self.model.backbone(lr_inputs)
        warped_prev = warp_prev_hr(prev_hr, motion_lr, scale=self.model.scale)
        disoccl = self.model.gate(
            depth_curr=depth_hr_curr,
            depth_prev=depth_hr_prev,
            motion_lr=motion_lr,
            scale=self.model.scale,
        )
        out_hr = self.model.head(
            current_sr=current_sr,
            warped_prev=warped_prev,
            disocclusion=disoccl,
            depth_hr=depth_hr_curr,
        )
        return out_hr, disoccl

    @classmethod
    def from_temporal_checkpoint(
        cls,
        ckpt_path: str | Path,
        device: str | torch.device = "cuda",
    ) -> "TemporalSRModelStateless":
        """Load a temporal checkpoint into a stateless export wrapper."""
        ck: dict[str, Any] = torch.load(
            Path(ckpt_path), map_location=device, weights_only=False
        )
        saved = ck.get("args", {})
        in_channels = int(saved.get("in_channels", 12))
        scale = int(saved.get("scale", 2))
        tier = saved.get("tier", "standard")
        backbone_kind = saved.get("backbone_kind")
        if backbone_kind is None:
            backbone_kind = "rrdb" if saved.get("sr_backbone") == "rrdb" else "simple"
        # New ckpts persist the flag explicitly. Legacy ckpts: infer from
        # whether warm-start was used (warm-started runs need zeroing to
        # match the v4-on-SRGD distribution; from-scratch runs do not).
        if "zero_gbuffer_into_backbone" in saved:
            zero_flag = bool(saved["zero_gbuffer_into_backbone"])
        else:
            zero_flag = bool(saved.get("warm_start"))
        model = TemporalSRModel(
            in_channels=in_channels,
            scale=scale,
            tier=tier,
            backbone_kind=backbone_kind,
            zero_gbuffer_into_backbone=zero_flag,
        )
        model.load_state_dict(ck["temporal_model"])
        model = model.to(device).train(False)
        return cls(model)


__all__ = ["TemporalSRModelStateless"]
