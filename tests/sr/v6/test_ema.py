"""Tests for ``oss.sr.v6.ema.EMAModel``."""
from __future__ import annotations

import torch
import torch.nn as nn

from oss.sr.v6.ema import EMAModel


def _make_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 3, 3, padding=1),
    )


def test_init_snapshots_source_params():
    m = _make_model()
    ema = EMAModel(m, decay=0.999)
    for name, p in m.named_parameters():
        assert name in ema.shadow_params
        assert torch.equal(ema.shadow_params[name], p.detach())


def test_update_is_deterministic_given_fixed_inputs():
    """Same EMA, same updates -> same shadow tensors."""
    m1 = _make_model(seed=42)
    m2 = _make_model(seed=42)
    ema1 = EMAModel(m1, decay=0.99)
    ema2 = EMAModel(m2, decay=0.99)

    # Apply identical perturbations to both models.
    torch.manual_seed(7)
    perturbations = [torch.randn_like(p) * 0.1 for p in m1.parameters()]
    with torch.no_grad():
        for p1, p2, dp in zip(m1.parameters(), m2.parameters(), perturbations):
            p1.add_(dp)
            p2.add_(dp)

    ema1.update(m1)
    ema2.update(m2)

    for name in ema1.shadow_params:
        assert torch.equal(ema1.shadow_params[name], ema2.shadow_params[name])


def test_update_math():
    """One update from initial state with decay=0.5 gives 0.5*init + 0.5*new."""
    m = _make_model(seed=1)
    ema = EMAModel(m, decay=0.5)

    # Snapshot pre-update shadow values (== init params).
    init_shadow = {name: p.detach().clone() for name, p in m.named_parameters()}

    # Set new param values to 1.0 everywhere.
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.fill_(1.0)

    ema.update(m)

    for name, init_val in init_shadow.items():
        expected = 0.5 * init_val + 0.5 * torch.ones_like(init_val)
        assert torch.allclose(ema.shadow_params[name], expected, atol=1e-6)


def test_state_dict_round_trip():
    m = _make_model()
    ema = EMAModel(m, decay=0.999)
    # Mutate.
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.add_(0.5)
    ema.update(m)

    state = ema.state_dict()
    ema2 = EMAModel(_make_model(), decay=0.5)  # different decay on init
    ema2.load_state_dict(state)
    assert ema2.decay == 0.999
    for name in ema.shadow_params:
        assert torch.equal(ema.shadow_params[name], ema2.shadow_params[name])


def test_swap_into_restores_on_exit():
    m = _make_model()
    ema = EMAModel(m, decay=0.999)
    # Make EMA distinct from live params.
    with torch.no_grad():
        for shadow in ema.shadow_params.values():
            shadow.fill_(0.0)

    live_snapshot = {name: p.detach().clone() for name, p in m.named_parameters()}

    with ema.swap_into(m):
        # Inside context: model has EMA values (zeros for trainable params).
        for name, p in m.named_parameters():
            if name in ema.shadow_params:
                assert torch.allclose(p, torch.zeros_like(p))

    # After context: model has live values restored.
    for name, p in m.named_parameters():
        assert torch.equal(p, live_snapshot[name])


def test_swap_into_restores_on_exception():
    m = _make_model()
    ema = EMAModel(m, decay=0.999)
    with torch.no_grad():
        for shadow in ema.shadow_params.values():
            shadow.fill_(0.0)
    live_snapshot = {name: p.detach().clone() for name, p in m.named_parameters()}

    try:
        with ema.swap_into(m):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    for name, p in m.named_parameters():
        assert torch.equal(p, live_snapshot[name])
