"""Tests for ``gaussian_regularization_loss``.

Composite regularizer:
    L = w_pos * ||mu_drift||_2
      + w_cov * sum max(0, det(Sigma_t) - max_area)
      + w_count * max(0, count_alive - max_count)

Where ``mu_drift`` is over Gaussians alive in BOTH ``field_t`` and
``field_t_minus_1`` (same SoA index = same Gaussian slot). Gradient must
NOT flow into ``field_t_minus_1``.
"""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import GaussianField, gaussian_regularization_loss


def _live_field(n: int, capacity: int | None = None) -> GaussianField:
    cap = capacity if capacity is not None else n
    f = GaussianField(capacity=cap)
    f.alive[:n] = True
    f.mu = torch.zeros(cap, 2)
    f.log_scale = torch.zeros(cap, 2)
    f.rotation = torch.zeros(cap)
    f.color = torch.zeros(cap, 3)
    f.opacity = torch.ones(cap)
    return f


# ---------------------------------------------------------------------------
# AC 1: zero on steady state
# ---------------------------------------------------------------------------

def test_zero_when_steady_state() -> None:
    """Identical fields, det(Sigma) <= max_area, count <= max_count -> 0."""
    f = _live_field(4)
    f_prev = _live_field(4)
    # log_scale = 0 -> scale = 1, det(Sigma) = 1.0
    loss = gaussian_regularization_loss(
        f, f_prev, max_area=64.0, max_count=8
    )
    assert loss.shape == ()
    assert loss.item() < 1e-5


def test_zero_with_default_weights_uses_unit_weights() -> None:
    f = _live_field(2)
    f_prev = _live_field(2)
    loss = gaussian_regularization_loss(f, f_prev, max_area=64.0, max_count=8)
    assert loss.item() < 1e-5


# ---------------------------------------------------------------------------
# AC 2: drift term linear in ||mu_t - mu_{t-1}||
# ---------------------------------------------------------------------------

def test_drift_grows_linearly() -> None:
    f_prev = _live_field(4)
    # Configure: huge max_area, huge max_count, only drift term active.
    weights = {"pos": 1.0, "cov": 0.0, "count": 0.0}

    f1 = _live_field(4)
    f1.mu = f1.mu.clone()
    f1.mu[:, 0] = 1.0  # drift of 1 unit in x for each of 4 alive Gaussians

    f2 = _live_field(4)
    f2.mu = f2.mu.clone()
    f2.mu[:, 0] = 2.0

    l1 = gaussian_regularization_loss(
        f1, f_prev, max_area=1e6, max_count=10_000, weights=weights
    )
    l2 = gaussian_regularization_loss(
        f2, f_prev, max_area=1e6, max_count=10_000, weights=weights
    )
    assert torch.isfinite(l1) and torch.isfinite(l2)
    # ||mu_drift||_2 over (4,2) tensor: f1 -> sqrt(4*1) = 2, f2 -> sqrt(4*4) = 4.
    assert abs(l1.item() - 2.0) < 1e-5
    assert abs(l2.item() - 4.0) < 1e-5
    # Linearity: doubling drift doubles the loss.
    assert abs((l2 / l1).item() - 2.0) < 1e-5


# ---------------------------------------------------------------------------
# AC 3: area term hinged at max_area
# ---------------------------------------------------------------------------

def test_area_term_hinged_below_max() -> None:
    """det(Sigma) <= max_area -> area contribution is zero."""
    f = _live_field(3)
    f_prev = _live_field(3)
    f.log_scale = torch.zeros(3, 2)  # det(Sigma) = 1
    weights = {"pos": 0.0, "cov": 1.0, "count": 0.0}
    loss = gaussian_regularization_loss(
        f, f_prev, max_area=10.0, max_count=10, weights=weights
    )
    assert loss.item() < 1e-6


def test_area_term_active_above_max() -> None:
    """det(Sigma) > max_area -> hinge term equals (det - max_area)."""
    f = _live_field(2)
    f_prev = _live_field(2)
    # log_scale = log(2) on both axes: scale = 2, scale^2 = 4 -> det(Sigma) = 16
    f.log_scale = torch.full((2, 2), float(torch.log(torch.tensor(2.0)).item()))
    weights = {"pos": 0.0, "cov": 1.0, "count": 0.0}
    loss = gaussian_regularization_loss(
        f, f_prev, max_area=10.0, max_count=10, weights=weights
    )
    # Each Gaussian: max(0, 16 - 10) = 6; sum over 2 alive -> 12
    assert abs(loss.item() - 12.0) < 1e-4


def test_area_term_only_alive_contribute() -> None:
    """Dead Gaussians' covariances do not contribute to area term."""
    f = _live_field(1, capacity=4)
    f_prev = _live_field(1, capacity=4)
    # Set log_scale on a DEAD slot to a huge value; should be ignored.
    f.log_scale = torch.zeros(4, 2)
    f.log_scale[2] = 5.0  # dead slot — det would be exp(20) if counted
    weights = {"pos": 0.0, "cov": 1.0, "count": 0.0}
    loss = gaussian_regularization_loss(
        f, f_prev, max_area=10.0, max_count=10, weights=weights
    )
    assert loss.item() < 1e-5


# ---------------------------------------------------------------------------
# AC 4: count term hinged at max_count
# ---------------------------------------------------------------------------

def test_count_term_hinged_below_max() -> None:
    f = _live_field(3, capacity=8)
    f_prev = _live_field(3, capacity=8)
    weights = {"pos": 0.0, "cov": 0.0, "count": 1.0}
    loss = gaussian_regularization_loss(
        f, f_prev, max_area=1e6, max_count=10, weights=weights
    )
    assert loss.item() < 1e-6


def test_count_term_active_above_max() -> None:
    f = _live_field(7, capacity=8)
    f_prev = _live_field(7, capacity=8)
    weights = {"pos": 0.0, "cov": 0.0, "count": 1.0}
    loss = gaussian_regularization_loss(
        f, f_prev, max_area=1e6, max_count=4, weights=weights
    )
    # max(0, 7 - 4) = 3
    assert abs(loss.item() - 3.0) < 1e-5


# ---------------------------------------------------------------------------
# AC 5: gradient flows to field_t.mu and field_t.log_scale
# ---------------------------------------------------------------------------

def test_grad_flows_to_field_t_mu() -> None:
    f_prev = _live_field(3)
    f = _live_field(3)
    mu = torch.full((3, 2), 0.5, requires_grad=True)
    f.mu = mu
    loss = gaussian_regularization_loss(
        f, f_prev, max_area=1e6, max_count=10
    )
    loss.backward()
    assert mu.grad is not None
    assert torch.isfinite(mu.grad).all()
    # Drift is non-zero -> gradient should be non-zero somewhere.
    assert mu.grad.abs().sum().item() > 0.0


def test_grad_flows_to_field_t_log_scale() -> None:
    f_prev = _live_field(2)
    f = _live_field(2)
    # log_scale that produces det(Sigma) > max_area so area term is active.
    log_scale = torch.full((2, 2), float(torch.log(torch.tensor(2.0)).item()),
                           requires_grad=True)
    f.log_scale = log_scale
    weights = {"pos": 0.0, "cov": 1.0, "count": 0.0}
    loss = gaussian_regularization_loss(
        f, f_prev, max_area=1.0, max_count=10, weights=weights
    )
    loss.backward()
    assert log_scale.grad is not None
    assert torch.isfinite(log_scale.grad).all()
    assert log_scale.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# AC 6: field_t_minus_1 is detached
# ---------------------------------------------------------------------------

def test_grad_does_not_flow_to_prev_mu() -> None:
    f_prev = _live_field(3)
    prev_mu = torch.full((3, 2), 0.1, requires_grad=True)
    f_prev.mu = prev_mu

    f = _live_field(3)
    cur_mu = torch.full((3, 2), 1.0, requires_grad=True)
    f.mu = cur_mu

    loss = gaussian_regularization_loss(f, f_prev, max_area=1e6, max_count=10)
    loss.backward()
    # Current field receives gradient.
    assert cur_mu.grad is not None
    assert cur_mu.grad.abs().sum().item() > 0.0
    # Gradient must NOT flow into the previous field.
    assert prev_mu.grad is None or prev_mu.grad.abs().sum().item() == 0.0
