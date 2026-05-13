"""Tests for canvas_health_metrics — the snapshot helper that emits
canvas count / mean opacity / mean L_diag to history.jsonl during
v7 training."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sr_train_v7 import canvas_health_metrics
from oss.sr.v7.model import V7Config, V7Model


def _model(capacity: int = 64) -> V7Model:
    cfg = V7Config(
        in_channels=9, scale=2, feat_dim=8, latent_rank=4,
        canvas_capacity=capacity, backbone_kind="placeholder",
        backbone_blocks=1, enable_spawner=False,
    )
    m = V7Model(cfg).train(False)
    m.allocate_canvas("cpu")
    return m


def test_empty_canvas_returns_all_zero():
    m = _model()
    h = canvas_health_metrics(m)
    assert h["canvas_count"] == 0
    assert h["canvas_mean_opacity"] == 0.0
    assert h["canvas_mean_L_diag"] == 0.0


def test_canvas_with_two_gaussians_returns_correct_summary():
    m = _model()
    # L00=exp(0)=1, L11=exp(0)=1, L22=exp(0)=1 -> mean diag = 1
    # Opacities 0.4 and 0.8 -> mean 0.6
    m.canvas.add(
        positions=torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]]),
        cov_raw=torch.zeros((2, 6)),
        features=torch.zeros((2, 4)),
        opacity=torch.tensor([0.4, 0.8]),
    )
    h = canvas_health_metrics(m)
    assert h["canvas_count"] == 2
    assert abs(h["canvas_mean_opacity"] - 0.6) < 1e-6
    assert abs(h["canvas_mean_L_diag"] - 1.0) < 1e-6


def test_canvas_L_diag_reflects_exp_of_raw_diagonals():
    """If cov_raw diagonals = log(2), log(3), log(4), the actual L_diag
    values are 2, 3, 4 -> mean = 3."""
    m = _model()
    cov_raw = torch.zeros((1, 6))
    cov_raw[0, 0] = math.log(2.0)
    cov_raw[0, 2] = math.log(3.0)
    cov_raw[0, 5] = math.log(4.0)
    m.canvas.add(
        positions=torch.zeros((1, 3)),
        cov_raw=cov_raw,
        features=torch.zeros((1, 4)),
        opacity=torch.tensor([0.5]),
    )
    h = canvas_health_metrics(m)
    assert h["canvas_count"] == 1
    assert abs(h["canvas_mean_L_diag"] - 3.0) < 1e-5


def test_health_metrics_skip_pruned_gaussians():
    """Pruned (dormant) entries must NOT contribute to the means."""
    m = _model()
    m.canvas.add(
        positions=torch.zeros((3, 3)),
        cov_raw=torch.zeros((3, 6)),
        features=torch.zeros((3, 4)),
        opacity=torch.tensor([1.0, 0.0, 1.0]),
    )
    # Prune the middle one
    m.canvas.prune(torch.tensor([True, False, True]))
    h = canvas_health_metrics(m)
    assert h["canvas_count"] == 2
    assert abs(h["canvas_mean_opacity"] - 1.0) < 1e-6
