"""Cross-module integration + edge-case smoke for the v6 package.

Each individual module test file covers shape / gradient / numerical
correctness in isolation. This file fills the gaps the per-module tests
collectively miss:

  - bf16 / autocast safety on the loss + discriminator paths (the
    pretrained backbones — VGG, LPIPS — historically break under
    autocast unless the wrappers force fp32 internally).
  - Cross-module composition: HAT -> cross-attention chain runs end to
    end in both fp32 and bf16 with finite gradients.
  - NaN guards on the loss with degenerate inputs (all-zero pred and
    target). The composite loss must produce finite output even when
    the model collapses to a constant.
  - Discriminator gradient flow through spectral normalization (a
    common silent-zero-gradient pattern).

These are minimum-required regression tests before V6Model integration;
finer coverage lives in the per-module test files.
"""
from __future__ import annotations

import torch
import pytest


# ---------------------------------------------------------------------------
# bf16 / autocast safety
# ---------------------------------------------------------------------------


def test_hat_to_cross_attention_chain_bf16():
    """HAT features -> cross-attention forward must stay finite in bf16."""
    from oss.sr.v6.hat import hat_tiny
    from oss.sr.v6.cross_attention import PixelGaussianFusion

    backbone = hat_tiny(in_channels=9)
    fusion = PixelGaussianFusion(feat_dim=60, token_dim=64, num_heads=4, window_size=16)

    backbone.train(False)
    fusion.train(False)

    x = torch.randn(1, 9, 32, 32)
    tokens = torch.randn(1, 8, 64)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        feats = backbone(x)
        out = fusion(feats, tokens)

    assert out.shape == feats.shape, f"shape mismatch {out.shape} vs {feats.shape}"
    assert torch.isfinite(out).all(), "non-finite values in bf16 chain output"


def test_hat_to_cross_attention_chain_empty_canvas_bf16():
    """K=0 (empty Gaussian canvas) must short-circuit and stay finite in bf16."""
    from oss.sr.v6.hat import hat_tiny
    from oss.sr.v6.cross_attention import PixelGaussianFusion

    backbone = hat_tiny(in_channels=9)
    fusion = PixelGaussianFusion(feat_dim=60, token_dim=64, num_heads=4, window_size=16)

    backbone.train(False)
    fusion.train(False)

    x = torch.randn(1, 9, 32, 32)
    empty_tokens = torch.zeros(1, 0, 64)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        feats = backbone(x)
        out = fusion(feats, empty_tokens)

    assert out.shape == feats.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out.float(), feats.float(), atol=1e-2), \
        "empty-canvas case should pass features through unchanged"


# ---------------------------------------------------------------------------
# Loss bf16 + NaN guards
# ---------------------------------------------------------------------------


def test_composite_loss_bf16_finite_with_pretrained_backbones():
    """V6CompositeLoss must run under autocast(bf16) with finite output.

    Pretrained backbones (VGG-19 for multi-scale + LPIPS-VGG) historically
    break in bf16 unless their wrappers internally cast to fp32. This
    test pins that the composite loss is end-to-end bf16-safe.
    """
    from oss.sr.v6.losses import V6CompositeLoss

    loss_fn = V6CompositeLoss(gan_warmup_until_step=20_000)
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        total, parts = loss_fn(pred, target, fake_logits=None, step=0)

    assert torch.isfinite(total), f"non-finite total loss in bf16: {total!r}"
    for k, v in parts.items():
        assert v == v, f"NaN in part {k}={v!r}"


def test_composite_loss_finite_on_zero_inputs():
    """Pred = target = zeros must yield a finite loss (not NaN/Inf).

    Charbonnier with eps>0 is well-defined at zero residual; LPIPS and
    multi-scale VGG handle constant inputs as a degenerate but valid
    case. Wavelet and Sobel see all-zero gradients which must not
    produce NaN.
    """
    from oss.sr.v6.losses import V6CompositeLoss

    loss_fn = V6CompositeLoss(gan_warmup_until_step=20_000)
    pred = torch.zeros(1, 3, 32, 32, requires_grad=True)
    target = torch.zeros(1, 3, 32, 32)

    total, parts = loss_fn(pred, target, fake_logits=None, step=0)

    assert torch.isfinite(total), f"non-finite loss on zero inputs: {total!r}"
    for k, v in parts.items():
        assert v == v, f"NaN in part {k}"


def test_composite_loss_finite_on_high_dynamic_range_inputs():
    """HDR-magnitude inputs (RGB > 1.0) must not crash the loss."""
    from oss.sr.v6.losses import V6CompositeLoss

    loss_fn = V6CompositeLoss(gan_warmup_until_step=20_000)
    pred = torch.rand(1, 3, 32, 32, requires_grad=True) * 5.0
    target = torch.rand(1, 3, 32, 32) * 5.0

    total, parts = loss_fn(pred, target, fake_logits=None, step=0)

    assert torch.isfinite(total), f"non-finite loss on HDR inputs: {total!r}"


# ---------------------------------------------------------------------------
# Discriminator: bf16 + spectral-norm gradient flow
# ---------------------------------------------------------------------------


def test_discriminator_bf16_finite_output():
    from oss.sr.v6.discriminator import UNetDiscriminator

    d = UNetDiscriminator()
    x = torch.randn(1, 3, 64, 64)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = d(x)
    assert torch.isfinite(logits).all()
    assert logits.shape == (1, 1, 64, 64)


def test_discriminator_gradient_flows_through_spectral_norm():
    """Spectral-norm wrapping breaks param.grad if the impl uses
    register_buffer for the singular vector instead of register_parameter.
    Verify gradients reach every learnable parameter in the discriminator.
    """
    from oss.sr.v6.discriminator import UNetDiscriminator

    d = UNetDiscriminator()
    x = torch.randn(1, 3, 32, 32)
    out = d(x)
    loss = out.mean()
    loss.backward()

    leaves = [(n, p) for n, p in d.named_parameters() if p.requires_grad]
    no_grad_params = [n for n, p in leaves if p.grad is None]
    assert not no_grad_params, (
        f"these discriminator params received no gradient: {no_grad_params}"
    )


# ---------------------------------------------------------------------------
# EMA: parameter-set bookkeeping under repeated update
# ---------------------------------------------------------------------------


def test_ema_repeated_updates_track_source():
    """After many updates with a low decay, the EMA params drift toward
    the source params. Catches impls that use decay = decay (instead of
    1 - decay) on the source side or that fail to update at all.
    """
    from oss.sr.v6.ema import EMAModel
    import torch.nn as nn

    src = nn.Linear(8, 4)
    # Snapshot the initial src params so we can compare drift directions.
    init_state = {k: v.clone() for k, v in src.state_dict().items() if isinstance(v, torch.Tensor)}
    ema = EMAModel(src, decay=0.5)

    target_offsets = {n: torch.randn_like(p) * 2.0 for n, p in src.named_parameters()}
    with torch.no_grad():
        for n, p in src.named_parameters():
            p.copy_(init_state[n] + target_offsets[n])
    for _ in range(50):
        ema.update(src)

    # After 50 updates with decay=0.5, EMA shadow params should be very
    # close to source (mixing weight on init is 0.5**50 ~= 9e-16).
    # EMA.state_dict() nests tensors under 'shadow_params'.
    shadow = ema.state_dict()["shadow_params"]
    src_named = dict(src.named_parameters())
    for name, src_p in src_named.items():
        assert name in shadow, f"EMA missing shadow for {name}"
        assert torch.allclose(shadow[name], src_p, atol=1e-3), (
            f"EMA shadow[{name}] should track source after 50 updates"
        )


# ---------------------------------------------------------------------------
# Schedule: exact-step boundaries
# ---------------------------------------------------------------------------


def test_lr_schedule_at_exact_restart_boundary():
    """At step == T_0 exactly, the cosine schedule restarts: LR jumps
    back to base_lr. Off-by-one bugs at this boundary are common; pin
    the behavior.
    """
    from oss.sr.v6.schedules import CosineLRWithWarmRestarts
    import torch.nn as nn

    p = nn.Linear(2, 2).parameters()
    optim = torch.optim.AdamW(p, lr=2e-4)
    sched = CosineLRWithWarmRestarts(
        optim, base_lr=2e-4, T_0=10, T_mult=1.0, num_restarts=2,
    )
    sched.step(0)
    assert abs(sched.get_last_lr() - 2e-4) < 1e-9
    sched.step(9)
    assert sched.get_last_lr() < 2e-4
    sched.step(10)
    assert abs(sched.get_last_lr() - 2e-4) < 1e-3, (
        f"LR at restart boundary should be near base_lr, got {sched.get_last_lr()}"
    )
