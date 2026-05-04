"""GaussianField container tests."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import GaussianField


def test_default_shapes() -> None:
    f = GaussianField(capacity=8)
    assert f.mu.shape == (8, 2)
    assert f.log_scale.shape == (8, 2)
    assert f.rotation.shape == (8,)
    assert f.color.shape == (8, 3)
    assert f.opacity.shape == (8,)
    assert f.alive.shape == (8,) and f.alive.dtype == torch.bool


def test_count_alive() -> None:
    f = GaussianField(capacity=8)
    f.alive[:5] = True
    assert f.count_alive() == 5


def test_history_window_capped_at_5() -> None:
    f = GaussianField(capacity=4)
    for _ in range(7):
        f.push_history(f.clone())
    assert len(f.history) == 5


def test_to_device_moves_all() -> None:
    f = GaussianField(capacity=4)
    f2 = f.to("cpu")  # no-op but exercises the path
    assert f2.mu.device.type == "cpu"


def test_clone_is_deep() -> None:
    f = GaussianField(capacity=4)
    f.mu.fill_(1.0)
    g = f.clone()
    g.mu.fill_(2.0)
    assert f.mu.mean().item() == 1.0
    assert g.mu.mean().item() == 2.0
