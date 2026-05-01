"""Tests for the recurrent latent cell used at the OSS-Pico bottleneck."""
import torch

from oss.model.temporal import RecurrentLatentCell


def test_recurrent_cell_shapes():
    cell = RecurrentLatentCell(channels=32, hidden_channels=24)
    x = torch.randn(2, 32, 16, 16)
    refined, new_h = cell(x, hidden=None)
    assert refined.shape == x.shape
    assert new_h.shape == (2, 24, 16, 16)


def test_recurrent_cell_state_evolution():
    cell = RecurrentLatentCell(channels=32, hidden_channels=24)
    x = torch.randn(1, 32, 8, 8)
    _, h1 = cell(x, hidden=None)
    _, h2 = cell(x, hidden=h1)
    assert not torch.allclose(h1, h2)


def test_recurrent_cell_param_budget():
    cell = RecurrentLatentCell(channels=32, hidden_channels=24)
    n = sum(p.numel() for p in cell.parameters())
    assert n < 8_000, f"RecurrentLatentCell exceeds 8K params: {n}"
