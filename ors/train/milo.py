"""MILO perceptual loss for HDR renderings.

This module prefers the official MILO reference architecture when the caller
provides an upstream ``MILO.pth`` via ``ORS_MILO_WEIGHTS``. ORS does not vendor
those pretrained weights, so the default path is a clearly documented
approximation: differentiable ACES tonemapping followed by LPIPS-VGG in the
display-referred domain.
"""
from __future__ import annotations

import os
from pathlib import Path

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F


def _aces_tonemap(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    num = x * (2.51 * x + 0.03)
    den = x * (2.43 * x + 0.59) + 0.14
    return (num / den.clamp(min=1e-6)).clamp(0.0, 1.0)


class _ScalerNetwork(nn.Module):
    def __init__(self, chn_mid: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, chn_mid, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(chn_mid, chn_mid, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(chn_mid, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _MaskFinder(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class _OfficialMILO(nn.Module):
    def __init__(self, weight_path: str | os.PathLike[str]):
        super().__init__()
        self.mask_finder_1 = _MaskFinder(7)
        self.scaler_network = _ScalerNetwork()
        state = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(state, strict=True)
        for param in self.parameters():
            param.requires_grad = False
        self.train(False)

    def _mask_generator(self, ref: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        ref_scale = [ref]
        dist_scale = [dist]
        for _ in range(3):
            ref_scale.insert(0, F.avg_pool2d(ref_scale[0], 2, stride=2, count_include_pad=False))
            dist_scale.insert(0, F.avg_pool2d(dist_scale[0], 2, stride=2, count_include_pad=False))
        mask = ref_scale[0].new_zeros(
            ref_scale[0].shape[0], 1, ref_scale[0].shape[2] // 2, ref_scale[0].shape[3] // 2
        )
        for ref_level, dist_level in zip(ref_scale, dist_scale):
            up = F.interpolate(mask, scale_factor=2, mode="bilinear", align_corners=True)
            if up.shape[-2:] != ref_level.shape[-2:]:
                up = F.pad(up, (0, ref_level.shape[-1] - up.shape[-1], 0, ref_level.shape[-2] - up.shape[-2]), mode="replicate")
            mask = self.mask_finder_1(torch.cat([ref_level, dist_level, up], dim=1)) + up
        return mask

    def forward(self, pred_ldr: torch.Tensor, target_ldr: torch.Tensor) -> torch.Tensor:
        mask = self._mask_generator(target_ldr, pred_ldr)
        return (mask * (target_ldr - pred_ldr).abs()).mean()


class MILOLoss(nn.Module):
    def __init__(self, milo_weights: str | os.PathLike[str] | None = None):
        super().__init__()
        weight_path = milo_weights or os.environ.get("ORS_MILO_WEIGHTS")
        self._official = _OfficialMILO(weight_path) if weight_path and Path(weight_path).exists() else None
        self._lpips: nn.Module | None = None

    def _init_lpips(self) -> nn.Module:
        if self._lpips is None:
            self._lpips = lpips.LPIPS(net="vgg", verbose=False)
            for param in self._lpips.parameters():
                param.requires_grad = False
            self._lpips.train(False)
        return self._lpips

    def forward(self, pred_hdr: torch.Tensor, target_hdr: torch.Tensor) -> torch.Tensor:
        pred_ldr = _aces_tonemap(pred_hdr)
        target_ldr = _aces_tonemap(target_hdr)
        if self._official is not None:
            self._official = self._official.to(pred_hdr.device)
            return self._official(pred_ldr, target_ldr)
        self._lpips = self._init_lpips().to(pred_hdr.device)
        return self._lpips(pred_ldr * 2.0 - 1.0, target_ldr * 2.0 - 1.0).mean()


_DEFAULT_LOSS = MILOLoss()


def milo_loss(pred_hdr: torch.Tensor, target_hdr: torch.Tensor) -> torch.Tensor:
    return _DEFAULT_LOSS(pred_hdr, target_hdr)


__all__ = ["MILOLoss", "milo_loss"]
