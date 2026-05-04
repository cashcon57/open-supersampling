"""Tests for opacity + count pruning."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import GaussianField, prune


def test_low_opacity_removed() -> None:
    f = GaussianField(capacity=8)
    f.alive[:] = True
    f.opacity = torch.tensor([0.01, 0.5, 0.02, 0.9, 0.09, 0.5, 0.05, 0.5])
    g = prune(f, opacity_threshold=0.1, max_count=8)
    # Threshold 0.1: indices with opacity < 0.1 (0, 2, 4, 6) become dead.
    assert g.alive.tolist() == [False, True, False, True, False, True, False, True]


def test_count_cap_evicts_lowest_opacity_first() -> None:
    f = GaussianField(capacity=8)
    f.alive[:] = True
    f.opacity = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.4, 0.3, 0.2, 0.1])
    g = prune(f, opacity_threshold=0.0, max_count=4)
    assert int(g.alive.sum().item()) == 4
    # The 4 lowest-opacity (indices 4..7 with opacities 0.4, 0.3, 0.2, 0.1) should be evicted.
    assert g.alive.tolist() == [True, True, True, True, False, False, False, False]


def test_no_change_when_within_caps() -> None:
    f = GaussianField(capacity=4)
    f.alive[:] = True
    f.opacity = torch.tensor([0.5, 0.5, 0.5, 0.5])
    g = prune(f, opacity_threshold=0.1, max_count=4)
    assert int(g.alive.sum().item()) == 4
