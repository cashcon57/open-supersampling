"""NoiseBase adapter for OSS-RG (denoiser) training.

Maps NoiseBaseDataset output to the (noisy, aux, history, gt) schema
expected by train_rg.py. NoiseBase stores everything at LR resolution;
GT for the denoiser is the reference downsampled to match.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .noisebase import NoiseBaseDataset


class NoiseBaseRGDataset(Dataset):
    """Wraps NoiseBaseDataset to yield per-frame OSSRG training batches.

    Each item is a single frame (not a sequence). Sequences are flattened
    so len() == n_sequences * sequence_length.

    Returns
    -------
    dict with keys:
        noisy    (3, H_lr, W_lr)  — 1-spp noisy radiance
        aux      (11, H_lr, W_lr) — [albedo(3), normal(3), depth(1),
                                      roughness(1)=0, spec_hit(1)=0, motion(2)]
        history  (3, H_lr, W_lr)  — prev frame GT downsampled (zeros at t=0)
        gt       (3, H_lr, W_lr)  — clean reference at LR resolution
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        resolution: tuple[int, int] = (800, 1280),
        scale_factor: float = 2.0,
        sequence_length: int = 8,
    ):
        self._base = NoiseBaseDataset(
            root=root,
            split=split,
            resolution=resolution,
            scale_factor=scale_factor,
            sequence_length=sequence_length,
        )
        self._T = sequence_length
        self._scale = scale_factor

    def __len__(self) -> int:
        return len(self._base) * self._T

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx = idx // self._T
        frame_t = idx % self._T

        seq = self._base[seq_idx]
        # seq values: (T, C, H_lr, W_lr) or (T, C, H_hr, W_hr) for gt_hr

        color_lr   = seq["color_lr"][frame_t]    # (3, H_lr, W_lr)
        albedo_lr  = seq["albedo_lr"][frame_t]   # (3, H_lr, W_lr)
        normals_lr = seq["normals_lr"][frame_t]  # (3, H_lr, W_lr)
        depth_lr   = seq["depth_lr"][frame_t]    # (1, H_lr, W_lr)
        motion_lr  = seq["motion_lr"][frame_t]   # (2, H_lr, W_lr)
        gt_hr      = seq["gt_hr"][frame_t]       # (3, H_hr, W_hr)

        H_lr, W_lr = color_lr.shape[-2], color_lr.shape[-1]

        gt_lr = F.interpolate(
            gt_hr.unsqueeze(0), size=(H_lr, W_lr), mode="bilinear", align_corners=False
        ).squeeze(0)

        zeros_1 = torch.zeros(1, H_lr, W_lr, dtype=torch.float32)
        aux = torch.cat([albedo_lr, normals_lr, depth_lr, zeros_1, zeros_1, motion_lr], dim=0)

        if frame_t == 0:
            history = torch.zeros_like(gt_lr)
        else:
            prev_gt_hr = seq["gt_hr"][frame_t - 1]
            history = F.interpolate(
                prev_gt_hr.unsqueeze(0), size=(H_lr, W_lr), mode="bilinear", align_corners=False
            ).squeeze(0)

        return {
            "noisy":   color_lr,
            "aux":     aux,
            "history": history,
            "gt":      gt_lr,
        }
