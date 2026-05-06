"""Tests for ``oss.sr.v6.losses``.

Covers:
    - shape correctness on representative inputs
    - gradient flow (loss.backward() works)
    - GAN warmup is respected (step < warmup_until -> GAN weight is 0)
    - multi-scale VGG handles (B, 3, 32, 32) minimum input size
    - wavelet / Sobel / Charbonnier scalar outputs
    - composite returns parts dict
"""
from __future__ import annotations

import pytest
import torch

from oss.sr.v6.losses import (
    V6CompositeLoss,
    charbonnier_loss,
    gan_hinge_d_loss,
    gan_hinge_g_loss,
    multi_scale_vgg_loss,
    sobel_edge_loss,
    temporal_consistency_loss,
    wavelet_l1_loss,
)


def _rand(shape, dtype=torch.float32, requires_grad: bool = False) -> torch.Tensor:
    t = torch.rand(*shape, dtype=dtype)
    if requires_grad:
        t.requires_grad_(True)
    return t


def test_charbonnier_scalar_and_gradient():
    pred = _rand((2, 3, 16, 16), requires_grad=True)
    target = _rand((2, 3, 16, 16))
    loss = charbonnier_loss(pred, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_charbonnier_zero_when_equal():
    pred = _rand((1, 3, 8, 8))
    target = pred.clone()
    loss = charbonnier_loss(pred, target, eps=1e-3)
    # eps floor: sqrt(0 + eps^2) = eps.
    assert abs(loss.item() - 1e-3) < 1e-5


def test_sobel_edge_loss_scalar_and_gradient():
    pred = _rand((2, 3, 16, 16), requires_grad=True)
    target = _rand((2, 3, 16, 16))
    loss = sobel_edge_loss(pred, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_wavelet_l1_loss_scalar_and_gradient():
    pred = _rand((2, 3, 16, 16), requires_grad=True)
    target = _rand((2, 3, 16, 16))
    loss = wavelet_l1_loss(pred, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_gan_hinge_losses():
    real = torch.randn(4, 1, 8, 8, requires_grad=True)
    fake = torch.randn(4, 1, 8, 8, requires_grad=True)
    d_loss = gan_hinge_d_loss(real, fake)
    g_loss = gan_hinge_g_loss(fake)
    assert d_loss.dim() == 0
    assert g_loss.dim() == 0
    d_loss.backward()
    assert real.grad is not None
    # g_loss separately: fresh fake.
    fake2 = torch.randn(4, 1, 8, 8, requires_grad=True)
    gan_hinge_g_loss(fake2).backward()
    assert fake2.grad is not None


def test_temporal_consistency_re_export_runs():
    pred_t = _rand((2, 3, 16, 16))
    pred_prev = _rand((2, 3, 16, 16))
    motion = torch.zeros(2, 2, 8, 8)
    loss = temporal_consistency_loss(pred_t, pred_prev, motion, scale_factor=2.0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


@pytest.fixture(scope="module")
def composite() -> V6CompositeLoss:
    """Single instance to amortize VGG / LPIPS setup."""
    torch.manual_seed(0)
    return V6CompositeLoss(gan_warmup_until_step=20_000)


def test_multi_scale_vgg_min_input_size():
    """Multi-scale VGG handles (B, 3, 32, 32) — the documented minimum.

    32x32 -> after relu5_1 (4 maxpools downstream) -> 2x2 feature map. Below
    32 the relu5_1 head would degenerate to <1px and break.
    """
    pred = _rand((2, 3, 32, 32), requires_grad=True)
    target = _rand((2, 3, 32, 32))
    loss = multi_scale_vgg_loss(pred, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_vgg_and_lpips_state_is_fp32(composite: V6CompositeLoss):
    """Frozen perceptual backbones stay fp32 even when callers use autocast."""
    vgg_state = [
        t for t in composite.vgg.state_dict().values()
        if torch.is_floating_point(t)
    ]
    assert vgg_state
    assert {t.dtype for t in vgg_state} == {torch.float32}

    assert composite._lpips is not None
    lpips_state = [
        t for t in composite._lpips.state_dict().values()
        if torch.is_floating_point(t)
    ]
    assert lpips_state
    assert {t.dtype for t in lpips_state} == {torch.float32}


def test_composite_can_opt_out_of_lpips():
    loss = V6CompositeLoss(gan_warmup_until_step=20_000, use_lpips=False)
    assert loss._lpips is None


def test_composite_forward_returns_parts(composite: V6CompositeLoss):
    pred = _rand((2, 3, 32, 32), requires_grad=True)
    target = _rand((2, 3, 32, 32))
    total, parts = composite(pred, target, fake_logits=None, step=0)
    assert total.dim() == 0
    assert torch.isfinite(total)
    for key in ("charbonnier", "vgg", "lpips", "wavelet", "sobel", "gan", "temporal", "total"):
        assert key in parts
        assert isinstance(parts[key], float)


def test_composite_gradient_flow(composite: V6CompositeLoss):
    pred = _rand((1, 3, 32, 32), requires_grad=True)
    target = _rand((1, 3, 32, 32))
    total, _ = composite(pred, target, fake_logits=None, step=0)
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


def test_composite_gan_warmup(composite: V6CompositeLoss):
    """Below warmup: GAN contribution is zero regardless of fake_logits."""
    pred = _rand((1, 3, 32, 32), requires_grad=True)
    target = _rand((1, 3, 32, 32))
    fake_logits = torch.randn(1, 1, 32, 32, requires_grad=True)

    # step < warmup -> gan part recorded as 0.
    _, parts_pre = composite(pred, target, fake_logits=fake_logits, step=19_999)
    assert parts_pre["gan"] == 0.0

    # step >= warmup -> gan contribution is non-zero (negative of mean fake_logits).
    _, parts_post = composite(pred, target, fake_logits=fake_logits, step=20_000)
    expected = float(-fake_logits.detach().mean())
    assert parts_post["gan"] == pytest.approx(expected, rel=1e-5, abs=1e-7)


def test_composite_temporal_only_when_provided(composite: V6CompositeLoss):
    pred = _rand((1, 3, 32, 32), requires_grad=True)
    target = _rand((1, 3, 32, 32))

    _, parts_no_temp = composite(pred, target, fake_logits=None, step=0)
    assert parts_no_temp["temporal"] == 0.0

    pwp = _rand((1, 3, 32, 32))
    twp = _rand((1, 3, 32, 32))
    _, parts_temp = composite(
        pred, target, fake_logits=None, step=0,
        pred_warped_prev=pwp, target_warped_prev=twp,
    )
    expected = float((pwp - twp).abs().mean())
    assert parts_temp["temporal"] == pytest.approx(expected, rel=1e-5, abs=1e-7)
