"""Regularization losses for the v5 Gaussian-temporal track.

Composite term:

    L_gaussian_reg = w_pos   * ||mu_drift||_2
                   + w_cov   * sum_alive max(0, det(Sigma_t) - max_area)
                   + w_count * max(0, count_alive(t) - max_count)

Where:
    mu_drift = field_t.mu[alive_intersection] - field_t_minus_1.mu[alive_intersection]
    alive_intersection = field_t.alive & field_t_minus_1.alive  (same SoA index = same Gaussian slot)
    Sigma_t = covariance reconstructed from (field_t.log_scale, field_t.rotation)

Gradient policy:
    - Flows to ``field_t.mu`` and ``field_t.log_scale`` (and ``field_t.rotation`` via Sigma).
    - Does NOT flow into ``field_t_minus_1`` — its tensors are detached at the boundary.
    - The count term is non-differentiable (depends on the bool ``alive`` mask), but
      its scalar value still contributes to the total loss for monitoring/clamping.
"""
from __future__ import annotations

from typing import Optional

import torch

from oss.sr.gaussian_temporal.analytical_warp import _decompose_covariance
from oss.sr.gaussian_temporal.gaussian_field import GaussianField


_DEFAULT_WEIGHTS: dict[str, float] = {"pos": 1.0, "cov": 1.0, "count": 1.0}


def gaussian_regularization_loss(
    field_t: GaussianField,
    field_t_minus_1: GaussianField,
    max_area: float,
    max_count: int,
    weights: Optional[dict] = None,
) -> torch.Tensor:
    """Composite drift / area / count regularizer for the Gaussian field.

    Args:
        field_t:           Current Gaussian field (gradient source).
        field_t_minus_1:   Previous Gaussian field. Detached internally.
        max_area:          Hinge knee for the covariance area term: only
                           ``det(Sigma) > max_area`` contributes.
        max_count:         Hinge knee for the count term: only
                           ``count_alive() > max_count`` contributes.
        weights:           Optional override for ``{'pos', 'cov', 'count'}``.
                           Defaults to ``{'pos': 1.0, 'cov': 1.0, 'count': 1.0}``.

    Returns:
        Scalar tensor on the same device/dtype as ``field_t.mu``.
    """
    w = dict(_DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)

    mu_t = field_t.mu
    log_scale_t = field_t.log_scale
    rotation_t = field_t.rotation
    alive_t = field_t.alive

    # Detach the previous field — gradient must NOT flow there.
    mu_prev = field_t_minus_1.mu.detach()
    alive_prev = field_t_minus_1.alive.detach()

    device = mu_t.device
    dtype = mu_t.dtype
    zero = torch.zeros((), device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Drift term — only Gaussians alive in BOTH fields contribute.
    # ------------------------------------------------------------------
    # Indices align: alive[i] in field_t and alive_prev[i] both true.
    # Capacity of the two fields can differ; intersect over the shared range.
    n_shared = min(alive_t.shape[0], alive_prev.shape[0])
    if n_shared > 0:
        intersect = alive_t[:n_shared] & alive_prev[:n_shared]
    else:
        intersect = torch.zeros((0,), dtype=torch.bool, device=device)

    if intersect.any():
        mu_drift = mu_t[:n_shared][intersect] - mu_prev[:n_shared][intersect]
        drift_term = torch.linalg.vector_norm(mu_drift)
    else:
        drift_term = zero

    # ------------------------------------------------------------------
    # Area term — sum over alive Gaussians of max(0, det(Sigma) - max_area).
    # ------------------------------------------------------------------
    if alive_t.any():
        sigma = _decompose_covariance(log_scale_t, rotation_t)  # (N, 2, 2)
        det_sigma = torch.linalg.det(sigma)  # (N,)
        # Mask dead slots out of the hinge sum.
        alive_f = alive_t.to(dtype)
        area_excess = torch.clamp(det_sigma - max_area, min=0.0) * alive_f
        area_term = area_excess.sum()
    else:
        area_term = zero

    # ------------------------------------------------------------------
    # Count term — hinged at max_count. Non-differentiable (alive is bool).
    # ------------------------------------------------------------------
    count_alive = float(field_t.count_alive())
    count_excess = max(0.0, count_alive - float(max_count))
    count_term = torch.tensor(count_excess, device=device, dtype=dtype)

    loss = (
        w["pos"] * drift_term
        + w["cov"] * area_term
        + w["count"] * count_term
    )
    return loss


__all__ = ["gaussian_regularization_loss"]
