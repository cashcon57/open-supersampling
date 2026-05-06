"""Tests for ``oss.sr.v6.schedules.CosineLRWithWarmRestarts``."""
from __future__ import annotations

import math

import pytest
import torch

from oss.sr.v6.schedules import CosineLRWithWarmRestarts


def _opt(lr: float = 1e-4) -> torch.optim.Optimizer:
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.Adam([p], lr=lr)


def test_first_step_is_base_lr():
    opt = _opt()
    sched = CosineLRWithWarmRestarts(opt, base_lr=2e-4, T_0=100, T_mult=1.0, num_restarts=3)
    sched.step(0)
    assert sched.get_last_lr() == pytest.approx(2e-4, rel=1e-9)
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-4, rel=1e-9)


def test_mid_cycle_value():
    opt = _opt()
    sched = CosineLRWithWarmRestarts(opt, base_lr=1.0, T_0=100, T_mult=1.0, num_restarts=3)
    # Halfway through cycle: cos(pi/2) = 0 -> lr = 0.5 * base.
    sched.step(50)
    assert sched.get_last_lr() == pytest.approx(0.5, abs=1e-6)


def test_restart_boundary_resets_to_base():
    """At step == T_0 we enter cycle 1 at progress=0 -> lr == base_lr again."""
    opt = _opt()
    sched = CosineLRWithWarmRestarts(opt, base_lr=2e-4, T_0=100, T_mult=1.0, num_restarts=3)

    # Just before restart (progress -> 1): lr -> 0.
    sched.step(99)
    just_before = sched.get_last_lr()
    assert just_before < 2e-4 * 0.05  # near zero but not exactly zero

    # At restart: back to peak.
    sched.step(100)
    assert sched.get_last_lr() == pytest.approx(2e-4, rel=1e-9)


def test_after_final_cycle_lr_is_zero():
    opt = _opt()
    sched = CosineLRWithWarmRestarts(opt, base_lr=2e-4, T_0=100, T_mult=1.0, num_restarts=3)
    # 4 cycles total of length 100 -> end_step = 400.
    sched.step(500)
    assert sched.get_last_lr() == 0.0
    assert opt.param_groups[0]["lr"] == 0.0


def test_t_mult_lengthens_cycles():
    opt = _opt()
    sched = CosineLRWithWarmRestarts(opt, base_lr=1.0, T_0=10, T_mult=2.0, num_restarts=2)
    # cycles: [0, 10), [10, 30), [30, 70). End = 70.
    sched.step(0)
    assert sched.get_last_lr() == pytest.approx(1.0)
    sched.step(10)
    assert sched.get_last_lr() == pytest.approx(1.0)  # restart
    sched.step(30)
    assert sched.get_last_lr() == pytest.approx(1.0)  # restart
    sched.step(70)
    assert sched.get_last_lr() == 0.0


def test_v6_default_layout():
    """v6 memo §6: T_0=50K, T_mult=1, 3 restarts -> 4 cycles of 50K = 200K total.

    The memo also says 300K total steps. The schedule covers 200K of that;
    after step 200K, lr stays at 0 (which is what 'cosine to zero with warm
    restarts' means at end-of-schedule)."""
    opt = _opt()
    sched = CosineLRWithWarmRestarts(
        opt, base_lr=2e-4, T_0=50_000, T_mult=1.0, num_restarts=3
    )
    # Boundary checks at the four restart points.
    for boundary in (0, 50_000, 100_000, 150_000):
        sched.step(boundary)
        assert sched.get_last_lr() == pytest.approx(2e-4, rel=1e-9)
    # End of schedule.
    sched.step(200_000)
    assert sched.get_last_lr() == 0.0


def test_cosine_curve_shape():
    """Verify the cosine curve at quartile points within a cycle."""
    opt = _opt()
    sched = CosineLRWithWarmRestarts(opt, base_lr=1.0, T_0=100, T_mult=1.0, num_restarts=0)
    # progress=0.25: lr = 0.5 * (1 + cos(pi/4)) = 0.5 * (1 + 0.7071) ~= 0.8536
    sched.step(25)
    assert sched.get_last_lr() == pytest.approx(0.5 * (1.0 + math.cos(math.pi * 0.25)), abs=1e-6)
    # progress=0.75:
    sched.step(75)
    assert sched.get_last_lr() == pytest.approx(0.5 * (1.0 + math.cos(math.pi * 0.75)), abs=1e-6)


def test_invalid_args_raise():
    opt = _opt()
    with pytest.raises(ValueError):
        CosineLRWithWarmRestarts(opt, base_lr=1e-3, T_0=0)
    with pytest.raises(ValueError):
        CosineLRWithWarmRestarts(opt, base_lr=1e-3, T_0=100, T_mult=0.0)
    with pytest.raises(ValueError):
        CosineLRWithWarmRestarts(opt, base_lr=1e-3, T_0=100, num_restarts=-1)
