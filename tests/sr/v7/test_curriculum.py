"""Tests for the v7-pico-005 alpha-curriculum step-conditional lambda schedule."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sr_train_v7 import curriculum_lambdas


KW = dict(
    stage1_end=20_000,
    stage2_end=60_000,
    fg_ramp_steps=5_000,
    target_fg=1.0,
    target_fg_lpips=0.5,
    target_temp=0.1,
)


def test_stage_1_returns_all_zero_FG_and_temp():
    """Pure-SR stage: FG, FG-LPIPS, temp all zero (only SR Charbonnier + LPIPS
    drive gradients via the unchanged --lambda-charbonnier / --lambda-lpips)."""
    for s in (1, 1_000, 10_000, 20_000):
        fg, fg_lpips, temp = curriculum_lambdas(step=s, **KW)
        assert fg == 0.0
        assert fg_lpips == 0.0
        assert temp == 0.0


def test_stage_2_ramps_FG_linearly_from_zero_to_target():
    # Start of stage 2: ramp starts at 0
    fg, fg_lpips, temp = curriculum_lambdas(step=20_001, **KW)
    assert 0.0 < fg < 0.001          # tiny but nonzero
    assert fg_lpips == 0.0
    assert temp == 0.0
    # Half-way through the ramp (2500 steps in)
    fg_mid, *_ = curriculum_lambdas(step=22_500, **KW)
    assert abs(fg_mid - 0.5) < 0.01
    # End of ramp
    fg_full, fg_lpips_end, temp_end = curriculum_lambdas(step=25_000, **KW)
    assert abs(fg_full - 1.0) < 1e-6
    assert fg_lpips_end == 0.0
    assert temp_end == 0.0


def test_stage_2_post_ramp_holds_FG_at_target_until_stage_3():
    """After the FG ramp completes (~step 25K) but before stage 3 (60K),
    FG-LPIPS and temp must still be 0."""
    fg, fg_lpips, temp = curriculum_lambdas(step=40_000, **KW)
    assert abs(fg - 1.0) < 1e-6
    assert fg_lpips == 0.0
    assert temp == 0.0


def test_stage_3_activates_FG_LPIPS_and_temp_consistency():
    fg, fg_lpips, temp = curriculum_lambdas(step=60_001, **KW)
    assert abs(fg - 1.0) < 1e-6
    assert abs(fg_lpips - 0.5) < 1e-6
    assert abs(temp - 0.1) < 1e-6
    # And way past the end of training
    fg, fg_lpips, temp = curriculum_lambdas(step=100_000, **KW)
    assert abs(fg - 1.0) < 1e-6
    assert abs(fg_lpips - 0.5) < 1e-6
    assert abs(temp - 0.1) < 1e-6


def test_curriculum_preserves_target_values_at_full_strength():
    """Stage 3 must respect whatever the user passed as targets, not hardcode."""
    custom = dict(KW)
    custom["target_fg"] = 2.5
    custom["target_fg_lpips"] = 1.5
    custom["target_temp"] = 0.25
    fg, fg_lpips, temp = curriculum_lambdas(step=100_000, **custom)
    assert abs(fg - 2.5) < 1e-6
    assert abs(fg_lpips - 1.5) < 1e-6
    assert abs(temp - 0.25) < 1e-6
