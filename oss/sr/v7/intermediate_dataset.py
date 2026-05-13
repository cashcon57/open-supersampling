"""Intermediate-frame dataset adapter for v7 OSS-FX training.

TartanAir provides RGB + depth + motion + normals frames at fixed
timesteps but NO intermediate-frame ground truth at, say, t = N + 0.5.
We construct alpha=0.5 supervision by SUBSAMPLING the source dataset:

  Pick indices i, i+1, i+2 within a single TartanAir sub-trajectory.
  Treat (i, i+2) as the model's "consecutive frames" (alpha=1 endpoints)
  and (i+1) as the held-out alpha=0.5 ground truth that the model
  is supervised to predict via OSS-FX.

The adapter filters the source dataset's _items to triplets where all
three indices share the same trajectory path. Triplets that span a
trajectory boundary are silently skipped.

Output dict shape (one sample):
  {
    "n":      {lr, depth, motion, normals, gt_hr},   # frame i
    "n_half": {gt_hr},                                  # frame i+1 -- alpha=0.5 GT
    "n_plus_1": {lr, depth, motion, normals, gt_hr},  # frame i+2
    "motion_n_to_np1": (2, H_lr, W_lr) -- combined i -> i+2 flow,
        computed as flow(i, i+1) + flow_warp(i+1, i+2). Used by the
        V7 model to seed canvas Gaussians with proper V_xt.
  }
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


class TartanAirIntermediateTriplets:
    """torch.utils.data.Dataset compatible. Wraps a
    TartanAirGaussianDataset and exposes valid (i, i+1, i+2) triplets.
    """

    def __init__(self, base, max_triplets: Optional[int] = None):
        """
        Args:
            base: a TartanAirGaussianDataset instance (or compatible
                object exposing __getitem__ returning
                GaussianTrainingExample, and an _items list of
                (image_path, depth_path, flow_path) tuples).
            max_triplets: if given, cap the number of returned triplets
                (for fast iteration during testing).
        """
        self.base = base
        # Build the list of valid triplets up-front. A "valid triplet"
        # has indices (i, i+1, i+2) sharing the same trajectory path,
        # which we infer from the image_path prefix up to image_left/.
        items = getattr(base, "_items", None)
        if items is None:
            raise ValueError(
                "Base dataset has no _items attribute -- "
                "TartanAirIntermediateTriplets requires direct access "
                "to the frame index for triplet validation."
            )
        triplet_indices: list[tuple[int, int, int]] = []
        for i in range(len(items) - 2):
            p_i = _trajectory_root(items[i][0])
            p_i1 = _trajectory_root(items[i + 1][0])
            p_i2 = _trajectory_root(items[i + 2][0])
            if p_i == p_i1 == p_i2:
                triplet_indices.append((i, i + 1, i + 2))
                if max_triplets is not None and len(triplet_indices) >= max_triplets:
                    break
        self._triplet_indices = triplet_indices

    def __len__(self) -> int:
        return len(self._triplet_indices)

    def __getitem__(self, idx: int) -> dict:
        i_n, i_half, i_np1 = self._triplet_indices[idx]
        ex_n = self.base[i_n]
        ex_half = self.base[i_half]
        ex_np1 = self.base[i_np1]

        # Compose i->i+2 motion field at LR:
        # motion(i, i+2) approx = motion(i, i+1) + warp(motion(i+1, i+2), motion(i, i+1))
        # For a first cut we simply double motion(i, i+1) -- works
        # exactly for linear motion, approximation for non-linear.
        # A future commit can swap in the more rigorous warp-composition.
        motion_n_to_np1 = ex_n.motion * 2.0

        return {
            "n": {
                "lr": ex_n.lr_frame,
                "depth": ex_n.depth,
                "motion": ex_n.motion,
                "normals": ex_n.normals,
                "gt_hr": ex_n.gt_hr_frame,
            },
            "n_half": {
                "gt_hr": ex_half.gt_hr_frame,
            },
            "n_plus_1": {
                "lr": ex_np1.lr_frame,
                "depth": ex_np1.depth,
                "motion": ex_np1.motion,
                "normals": ex_np1.normals,
                "gt_hr": ex_np1.gt_hr_frame,
            },
            "motion_n_to_np1": motion_n_to_np1,
        }


def _trajectory_root(image_path) -> str:
    """Extract the trajectory root from an image path.

    A TartanAir image_left path looks like:
       .../oldtown/Easy/P000/image_left/000123_left.png
    The trajectory root is everything up to image_left.
    """
    p = str(image_path)
    idx = p.find("image_left")
    if idx < 0:
        return p   # fall back to raw path; equality test still works
    return p[:idx]
