# Copyright 2026 OpenSuperSampling contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deep Gaussian Prior covariance dictionary for v6.2 spawning.

The DGP idea follows ContinuousSR / Pinilla et al. 2025: natural-image local
Gaussian covariances occupy a constrained region, so a learned soft assignment
over positive-definite prototype covariances is a better birth prior than
directly regressing unconstrained scale and rotation deltas.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DGPDictionary"]


_NATURAL_VAR_X_MAX = 2.4
_NATURAL_VAR_Y_MAX = 2.2
_NATURAL_COV_MIN = -0.9
_NATURAL_COV_MAX = 1.5
_MIN_VARIANCE = 0.05
_SPD_MARGIN = 0.95
_MIN_CONIC_DET = 1.0e-10
_MIN_SCALE = 1.0e-4


class DGPDictionary(nn.Module):
    """Deep Gaussian Prior covariance dictionary.

    Replaces direct ``(delta scale, delta rotation)`` regression with a
    softmax over ``M`` positive-definite inverse-covariance prototypes.
    Prototypes are initialized across the ContinuousSR-reported natural-image
    covariance range: ``sigma_x^2 in [0, 2.4]``, ``sigma_y^2 in [0, 2.2]``,
    and cross-covariance in ``[-0.9, 1.5]``.

    Args:
        M: Number of covariance prototypes. Expected v6.2 range is 8-16, but
            any positive value is supported.
        feat_dim: Input feature dimension.
    """

    def __init__(self, M: int = 16, feat_dim: int = 64) -> None:
        super().__init__()
        if M <= 0:
            raise ValueError(f"M must be positive; got {M}")
        if feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive; got {feat_dim}")

        self.M = int(M)
        self.feat_dim = int(feat_dim)

        proto_init = self._sample_natural_prototypes(self.M)
        self._assert_positive_definite_conics(proto_init)
        self.register_buffer("prototypes_abd", proto_init)

        self.weight_head = nn.Linear(self.feat_dim, self.M)
        self.scale_head = nn.Linear(self.feat_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize heads to an unbiased dictionary mean and unit scale."""
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        unit_scale_bias = math.log(math.expm1(1.0 - _MIN_SCALE))
        nn.init.constant_(self.scale_head.bias, unit_scale_bias)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode per-Gaussian features to conic coefficients.

        Args:
            feat: Tensor with shape ``(..., feat_dim)``.

        Returns:
            ``(conic_abd, scale)`` where ``conic_abd`` has shape ``(..., 3)``
            and stores inverse-covariance entries ``(a, b, d)`` for
            ``[[a, b], [b, d]]``. ``scale`` has shape ``(...)`` and is the
            positive scalar multiplier ``lambda`` in
            ``Lambda_g = lambda_g * sum_m softmax(w)_m * Lambda_m``.
        """
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(
                f"expected last dim feat_dim={self.feat_dim}, got {feat.shape[-1]}"
            )

        original_shape = feat.shape[:-1]
        flat = feat.reshape(-1, self.feat_dim)
        logits = self.weight_head(flat)
        weights = torch.softmax(logits, dim=-1)

        prototypes = self.prototypes_abd.to(device=flat.device, dtype=flat.dtype)
        conic = weights @ prototypes
        scale = F.softplus(self.scale_head(flat)).squeeze(-1) + _MIN_SCALE
        conic = conic * scale.unsqueeze(-1)

        return conic.reshape(*original_shape, 3), scale.reshape(*original_shape)

    @staticmethod
    def _sample_natural_prototypes(M: int) -> torch.Tensor:
        """Deterministically cover the natural-image covariance envelope."""
        covariances = []
        low_var_x = _MIN_VARIANCE
        low_var_y = _MIN_VARIANCE
        for i in range(M):
            if M == 1:
                t = 0.5
                phase = 0.5
            else:
                t = float(i) / float(M - 1)
                phase = float((i * 5) % M) / float(M - 1)

            var_x = low_var_x + (_NATURAL_VAR_X_MAX - low_var_x) * t
            var_y = low_var_y + (_NATURAL_VAR_Y_MAX - low_var_y) * (1.0 - t)
            cov_target = _NATURAL_COV_MIN + (_NATURAL_COV_MAX - _NATURAL_COV_MIN) * phase

            # The reported covariance range includes values that can be
            # indefinite for small variance pairs. Clip only to the SPD cone;
            # the unclipped endpoints are represented where the variances allow.
            cov_limit = _SPD_MARGIN * math.sqrt(var_x * var_y)
            cov_xy = max(-cov_limit, min(cov_target, cov_limit))
            sigma = torch.tensor(
                [[var_x, cov_xy], [cov_xy, var_y]],
                dtype=torch.float32,
            )
            covariances.append(sigma)

        sigma = torch.stack(covariances, dim=0)
        inv = torch.linalg.inv(sigma)
        return torch.stack((inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]), dim=-1)

    @staticmethod
    def _assert_positive_definite_conics(conic_abd: torch.Tensor) -> None:
        if conic_abd.ndim != 2 or conic_abd.shape[-1] != 3:
            raise ValueError(f"conic_abd must be (M, 3); got {tuple(conic_abd.shape)}")
        a, b, d = conic_abd.unbind(dim=-1)
        det = a * d - b * b
        is_spd = (a > 0.0) & (d > 0.0) & (det > _MIN_CONIC_DET)
        if not bool(is_spd.all().item()):
            bad = torch.nonzero(~is_spd, as_tuple=False).flatten().tolist()
            raise AssertionError(
                "DGPDictionary prototype conics must be positive-definite; "
                f"bad indices={bad}"
            )
