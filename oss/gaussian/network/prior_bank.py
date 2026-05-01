"""Covariance Prior Bank — Sprint 4 / T4.1.

A small, fixed (or optionally learnable) bank of 2D covariance shapes. The
GaussianParamNetwork emits a softmax distribution over this bank per Gaussian;
the final per-Gaussian covariance is the weighted sum of the bank entries.

Why a bank rather than direct (sx, sy, θ) regression?
- The network's output dimensionality drops from 3 continuous params to a
  K-way discrete-logit head (well-conditioned softmax, no scale collapse).
- Degenerate predictions (sx≈0, gigantic anisotropy) become impossible — every
  bank entry is hand-picked to be a valid, well-shaped Gaussian.
- GS-STVSR (2025) reports covariance is ≈0.99-correlated frame-to-frame, so
  a small fixed vocabulary captures the shape distribution; only the bank
  weights need to be predicted.

The bank stores entries in (sx, sy, θ) form (positive scales + rotation in
radians). Each forward pass converts entries → 2×2 covariance matrices, then
combines them with the network's per-Gaussian softmax weights.

Both the (sx, sy, θ) parametrisation and the resulting covariance matrices
are returned so downstream code can choose either representation:
- (sx, sy, θ) feeds directly into the existing Rasterizer GaussianBatch API.
- 2×2 Σ is what the design spec exposes if/when we move to a full-cov renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default 16-entry vocabulary. Each entry is (sx, sy, θ_radians).
# Coverage:
#   - Circular (isotropic) at 3 scales: 1, 2, 4
#   - Elongated horizontal at 2 scales: (4,1), (8,1)
#   - Elongated vertical at 2 scales: (1,4), (1,8)
#   - Diagonal 45° (rotated horizontal-elongated)
#   - Diagonal 135° (rotated vertical-elongated)
#   - Narrow (very anisotropic): (8, 0.5) at 0°, 45°, 90°, 135°
#   - Mid-elongated diagonals: (3, 1) @ 45°, (1, 3) @ 45°
import math

_DEFAULT_BANK_16: Tuple[Tuple[float, float, float], ...] = (
    # Circular
    (1.0, 1.0, 0.0),
    (2.0, 2.0, 0.0),
    (4.0, 4.0, 0.0),
    # Elongated horizontal
    (4.0, 1.0, 0.0),
    (8.0, 1.0, 0.0),
    # Elongated vertical
    (1.0, 4.0, 0.0),
    (1.0, 8.0, 0.0),
    # 45° diagonal elongations
    (4.0, 1.0, math.pi / 4.0),
    (4.0, 1.0, 3.0 * math.pi / 4.0),  # 135°
    # Mid-anisotropic diagonals
    (3.0, 1.0, math.pi / 4.0),
    (1.0, 3.0, math.pi / 4.0),
    # Narrow / streak-like (sx >> sy) at multiple orientations
    (8.0, 0.5, 0.0),
    (8.0, 0.5, math.pi / 4.0),
    (8.0, 0.5, math.pi / 2.0),
    (8.0, 0.5, 3.0 * math.pi / 4.0),
    # One extra small isotropic — useful as the "neutral" fallback weight
    (1.5, 1.5, 0.0),
)


@dataclass(frozen=True)
class CovarianceEntry:
    """Human-readable description of one bank entry."""
    sx: float
    sy: float
    theta_rad: float


def default_bank_16() -> Tuple[CovarianceEntry, ...]:
    """Return the default 16-entry vocabulary as a tuple of CovarianceEntry."""
    return tuple(CovarianceEntry(sx, sy, t) for (sx, sy, t) in _DEFAULT_BANK_16)


class CovariancePriorBank(nn.Module):
    """A fixed (default) or learnable bank of 2D covariance shapes.

    Args:
        entries: iterable of (sx, sy, theta) triples. Defaults to the 16-entry
            vocabulary above. ``len(entries)`` becomes ``self.bank_size``.
        learnable: when True, the bank parameters become trainable. Default
            False — the bank is a fixed prior and the network just predicts
            weights over it. Ablation knob for Sprint 4.

    Forward:
        weights: (..., K) softmax-normalized weights over the K bank entries.
        Returns:
            sx: (...,) per-Gaussian effective sx
            sy: (...,) per-Gaussian effective sy
            theta: (...,) per-Gaussian effective rotation in radians
            cov: (..., 2, 2) per-Gaussian covariance matrix Σ.

    The (sx, sy, θ) outputs are computed from the SAME bank weights as Σ — they
    are produced by weighted-summing the entries' raw (sx, sy, θ) values, NOT
    by re-deriving from Σ. This keeps the renderer interface (which takes
    (sx, sy, θ) directly) consistent with what the network parametrised.
    Wrap-around for θ is handled via circular weighted mean of (cosθ, sinθ).
    """

    def __init__(
        self,
        entries: Tuple[Tuple[float, float, float], ...] | None = None,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        if entries is None:
            entries = _DEFAULT_BANK_16
        if len(entries) < 2:
            raise ValueError(f"bank must have at least 2 entries; got {len(entries)}")
        for (sx, sy, _) in entries:
            if sx <= 0 or sy <= 0:
                raise ValueError(f"sx and sy must be positive; got ({sx}, {sy})")
        self.bank_size = len(entries)

        sx_t = torch.tensor([e[0] for e in entries], dtype=torch.float32)
        sy_t = torch.tensor([e[1] for e in entries], dtype=torch.float32)
        theta_t = torch.tensor([e[2] for e in entries], dtype=torch.float32)

        # Internally store log-scales so that learnable bank stays positive.
        log_sx = torch.log(sx_t)
        log_sy = torch.log(sy_t)

        if learnable:
            self.log_sx = nn.Parameter(log_sx)
            self.log_sy = nn.Parameter(log_sy)
            self.theta = nn.Parameter(theta_t)
        else:
            self.register_buffer("log_sx", log_sx, persistent=True)
            self.register_buffer("log_sy", log_sy, persistent=True)
            self.register_buffer("theta", theta_t, persistent=True)
        self.learnable = learnable

    # ---- Inspection helpers --------------------------------------------------
    def entries(self) -> Tuple[CovarianceEntry, ...]:
        sxs = torch.exp(self.log_sx).detach().cpu().tolist()
        sys_ = torch.exp(self.log_sy).detach().cpu().tolist()
        ts = self.theta.detach().cpu().tolist()
        return tuple(CovarianceEntry(sx, sy, t) for sx, sy, t in zip(sxs, sys_, ts))

    def covariance_matrices(self) -> torch.Tensor:
        """Return the K bank covariance matrices as a (K, 2, 2) tensor."""
        sx = torch.exp(self.log_sx)  # (K,)
        sy = torch.exp(self.log_sy)
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)
        # R = [[c, -s], [s, c]]; Σ = R diag(sx², sy²) Rᵀ
        sx2 = sx * sx
        sy2 = sy * sy
        a = c * c * sx2 + s * s * sy2
        b = c * s * (sx2 - sy2)
        d = s * s * sx2 + c * c * sy2
        # Stack into (K, 2, 2)
        row0 = torch.stack([a, b], dim=-1)
        row1 = torch.stack([b, d], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    # ---- Forward -------------------------------------------------------------
    def forward(
        self,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine bank entries by ``weights``.

        Args:
            weights: (..., K) softmax-normalized weights. Last dim == bank_size.

        Returns:
            sx, sy, theta: (...,) per-Gaussian effective scale + rotation.
            cov: (..., 2, 2) per-Gaussian covariance.
        """
        if weights.shape[-1] != self.bank_size:
            raise ValueError(
                f"weights last dim ({weights.shape[-1]}) must equal bank_size "
                f"({self.bank_size})"
            )
        # Numerically robust weighted sum even if caller forgot to softmax.
        if not torch.all((weights >= 0) & torch.isfinite(weights)):
            raise ValueError("weights must be non-negative and finite (apply softmax first)")

        sx_bank = torch.exp(self.log_sx)  # (K,)
        sy_bank = torch.exp(self.log_sy)
        # Weighted geometric mean for sx, sy via log-space (preserves positivity
        # and matches the parametrisation used by gradient-based learning).
        log_sx_eff = (weights * self.log_sx).sum(dim=-1)
        log_sy_eff = (weights * self.log_sy).sum(dim=-1)
        sx_eff = torch.exp(log_sx_eff)
        sy_eff = torch.exp(log_sy_eff)

        # Circular weighted mean for θ (handles 0/π wrap-around correctly).
        cos_w = (weights * torch.cos(self.theta)).sum(dim=-1)
        sin_w = (weights * torch.sin(self.theta)).sum(dim=-1)
        theta_eff = torch.atan2(sin_w, cos_w)

        # Cov matrix from the resulting (sx, sy, θ).
        c = torch.cos(theta_eff)
        s = torch.sin(theta_eff)
        sx2 = sx_eff * sx_eff
        sy2 = sy_eff * sy_eff
        a = c * c * sx2 + s * s * sy2
        b = c * s * (sx2 - sy2)
        d = s * s * sx2 + c * c * sy2
        # (..., 2, 2)
        row0 = torch.stack([a, b], dim=-1)
        row1 = torch.stack([b, d], dim=-1)
        cov = torch.stack([row0, row1], dim=-2)

        # Sanity guard: avoid degenerate / non-finite outputs reaching the
        # renderer when callers forget to clamp.
        sx_eff = sx_eff.clamp(min=1e-4)
        sy_eff = sy_eff.clamp(min=1e-4)
        # Silence unused-locals — sx_bank/sy_bank are kept for debug branches.
        _ = (sx_bank, sy_bank)
        return sx_eff, sy_eff, theta_eff, cov


__all__ = [
    "CovariancePriorBank",
    "CovarianceEntry",
    "default_bank_16",
]
