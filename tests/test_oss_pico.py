"""Tests for the Pico-tier ORU (Steam Deck / RDNA 2 target)."""
import pytest
import torch

from oss.model.oru_pico import OSSPico


@pytest.mark.parametrize("use_wavelet", [False, True])
def test_oru_pico_forward_shapes(use_wavelet):
    m = OSSPico(use_wavelet=use_wavelet).train(False)
    B, H_lr, W_lr = 2, 64, 64
    H_hr, W_hr = 128, 128
    color_lr = torch.randn(B, 3, H_lr, W_lr)
    depth_lr = torch.randn(B, 1, H_lr, W_lr)
    motion_lr = torch.randn(B, 2, H_lr, W_lr)
    normals_lr = torch.randn(B, 3, H_lr, W_lr)
    albedo_lr = torch.randn(B, 3, H_lr, W_lr)
    history_hr = torch.randn(B, 3, H_hr, W_hr)
    rgb_hr, new_hidden = m(
        color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_state=None
    )
    assert rgb_hr.shape == (B, 3, H_hr, W_hr)
    assert new_hidden.shape == (B, 24, H_lr // 4, W_lr // 4)


def test_oru_pico_param_budget():
    """Param-count test runs against the ship config (use_wavelet=True).

    The wavelet head adds ~50K params on top of the ~270K U-Net trunk, so the
    upper bound is relaxed to 350K. Ablation runs (use_wavelet=False) drop
    well under this bound.
    """
    n = sum(p.numel() for p in OSSPico(use_wavelet=True).parameters())
    assert 200_000 <= n <= 350_000, f"ORU-Pico params {n} out of [200K, 350K]"


def test_oru_pico_hidden_state_propagation():
    m = OSSPico().train(False)
    color_lr = torch.randn(1, 3, 32, 32)
    depth_lr = torch.randn(1, 1, 32, 32)
    motion_lr = torch.randn(1, 2, 32, 32)
    normals_lr = torch.randn(1, 3, 32, 32)
    albedo_lr = torch.randn(1, 3, 32, 32)
    history_hr = torch.randn(1, 3, 64, 64)
    # First frame.
    _, h1 = m(
        color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_state=None
    )
    # Second frame uses h1.
    _, h2 = m(
        color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_state=h1
    )
    # h2 should differ from h1 (state evolves).
    assert not torch.allclose(h1, h2)


def test_oru_pico_backward():
    m = OSSPico().train(True)
    color_lr = torch.randn(1, 3, 32, 32, requires_grad=True)
    depth_lr = torch.randn(1, 1, 32, 32)
    motion_lr = torch.randn(1, 2, 32, 32)
    normals_lr = torch.randn(1, 3, 32, 32)
    albedo_lr = torch.randn(1, 3, 32, 32)
    history_hr = torch.randn(1, 3, 64, 64)
    rgb_hr, _ = m(
        color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_state=None
    )
    rgb_hr.mean().backward()
    assert color_lr.grad is not None
