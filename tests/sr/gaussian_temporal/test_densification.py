"""Densification tests."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import GaussianField, densify


def test_no_spawn_below_threshold() -> None:
    f = GaussianField(capacity=8)
    lr_target = torch.zeros(1, 3, 16, 16)
    rendered = torch.zeros(1, 3, 16, 16)  # zero residual
    g = densify(f, lr_target=lr_target, rendered=rendered,
                tile_size=8, residual_threshold=0.01, max_new=4)
    assert g.count_alive() == 0


def test_spawn_inserts_into_free_slots() -> None:
    f = GaussianField(capacity=8)
    lr_target = torch.full((1, 3, 16, 16), 0.5)
    rendered = torch.zeros(1, 3, 16, 16)  # residual=0.5 everywhere → exceed threshold
    g = densify(f, lr_target=lr_target, rendered=rendered,
                tile_size=8, residual_threshold=0.01, max_new=2)
    assert g.count_alive() == 2
    # Inserted color matches tile mean.
    inserted_idx = g.alive.nonzero(as_tuple=True)[0]
    assert torch.allclose(g.color[inserted_idx], torch.full((2, 3), 0.5), atol=1e-3)


def test_color_grad_flows() -> None:
    f = GaussianField(capacity=4)
    lr_target = torch.full((1, 3, 8, 8), 0.5, requires_grad=True)
    rendered = torch.zeros(1, 3, 8, 8)
    g = densify(f, lr_target=lr_target, rendered=rendered,
                tile_size=4, residual_threshold=0.01, max_new=4)
    g.color.sum().backward()
    assert lr_target.grad is not None
