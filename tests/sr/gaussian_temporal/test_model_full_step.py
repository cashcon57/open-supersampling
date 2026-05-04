"""Task 8 acceptance tests: GaussianTemporalSRModel — full pipeline.

Covers:
- Constructs successfully with default args.
- Forward signature: (lr_inputs (B,12,h,w), motion_lr (B,2,h,w), prev_field|None)
  -> (out_hr, new_field, debug).
- First-frame seed (prev_field=None): count_alive() > 0 AND
  rendered_hr.abs().max() > 0.
- Synthetic moving-rectangle 2-frame sequence: full forward, full loss
  (L1 + 0.05*temporal_consistency + 0.01*gaussian_regularization_loss),
  loss.backward() produces finite gradients on encoder, transformer, and
  the rendered output.
- new_field.alive is consistent (no NaN, count >= 0).
"""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import (
    GaussianTemporalSRModel,
    GaussianField,
    gaussian_regularization_loss,
)
from oss.train.losses import temporal_consistency_loss


def _make_lr_inputs(h_lr: int = 32, w_lr: int = 32, seed: int = 0) -> torch.Tensor:
    """12-ch LR input — first 3 channels are an RGB rectangle on dark background."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 12, h_lr, w_lr, generator=g) * 0.05
    # Stamp a bright rectangle into the RGB channels.
    x[:, :3] = 0.0
    x[:, :3, 8:24, 8:24] = 0.7
    return x


def test_construct_default() -> None:
    """GaussianTemporalSRModel() constructs with documented default args."""
    model = GaussianTemporalSRModel(in_channels=12, scale=2, max_count=16384)
    assert isinstance(model, torch.nn.Module)
    assert model.scale == 2
    assert model.max_count == 16384
    # Constants required by the plan.
    assert model.initial_seed_count == 4096
    assert model.densify_threshold == 0.05
    assert model.densify_max_new == 256
    assert model.opacity_threshold == 0.05


def test_forward_signature_and_first_frame_seed() -> None:
    """First-frame (prev_field=None): count_alive > 0 AND rendered_hr.abs().max() > 0."""
    torch.manual_seed(0)
    model = GaussianTemporalSRModel(in_channels=12, scale=2, max_count=4096)

    h_lr, w_lr = 32, 32
    lr = _make_lr_inputs(h_lr, w_lr)
    motion_lr = torch.zeros(1, 2, h_lr, w_lr)

    out_hr, new_field, debug = model(lr, motion_lr, prev_field=None)

    # Shape contract.
    assert out_hr.shape == (1, 3, h_lr * 2, w_lr * 2), (
        f"expected (1, 3, {h_lr*2}, {w_lr*2}); got {tuple(out_hr.shape)}"
    )
    # Field state.
    assert isinstance(new_field, GaussianField)
    assert new_field.count_alive() > 0, "first-frame seed must populate the field"
    # Debug returned.
    assert isinstance(debug, dict) and "count_alive" in debug
    assert debug["count_alive"] == new_field.count_alive()
    # Critical: rendered_hr must be a real image, not the pre-densify zero render.
    assert out_hr.abs().max().item() > 0.0, (
        "rendered_hr must be non-zero on first frame (re-render after densify required)"
    )


def test_alive_mask_consistent_no_nan() -> None:
    """new_field.alive has no NaN equivalent (bool); count_alive >= 0 and within capacity."""
    torch.manual_seed(1)
    model = GaussianTemporalSRModel(in_channels=12, scale=2, max_count=2048)
    h_lr, w_lr = 32, 32
    lr = _make_lr_inputs(h_lr, w_lr, seed=1)
    motion_lr = torch.zeros(1, 2, h_lr, w_lr)
    out_hr, new_field, _ = model(lr, motion_lr, prev_field=None)

    assert new_field.alive.dtype == torch.bool
    assert new_field.count_alive() >= 0
    assert new_field.count_alive() <= new_field.alive.numel()
    # Tensors must be finite.
    assert torch.isfinite(new_field.mu).all()
    assert torch.isfinite(new_field.log_scale).all()
    assert torch.isfinite(new_field.rotation).all()
    assert torch.isfinite(new_field.color).all()
    assert torch.isfinite(new_field.opacity).all()
    assert torch.isfinite(out_hr).all()


def test_two_frame_full_loss_backward_finite_grads() -> None:
    """2-frame moving-rectangle: forward both frames, full loss, backward, finite grads."""
    torch.manual_seed(2)
    model = GaussianTemporalSRModel(in_channels=12, scale=2, max_count=2048)

    h_lr, w_lr = 32, 32
    # Frame 0 — rectangle in upper-left region.
    lr_t0 = _make_lr_inputs(h_lr, w_lr, seed=2)
    # Frame 1 — rectangle shifted right by 4 LR pixels.
    lr_t1 = torch.zeros_like(lr_t0)
    lr_t1[:, :3, 8:24, 12:28] = 0.7
    # Motion field: LR displacement of 4 px in x for the rectangle region; zero else.
    # We use a uniform constant motion to make temporal_consistency well-defined.
    motion_lr = torch.zeros(1, 2, h_lr, w_lr)
    motion_lr[:, 0] = 4.0  # x-displacement in LR pixels

    # Frame 0.
    out_t0, field_t0, _ = model(lr_t0, motion_lr, prev_field=None)
    # Frame 1 — prev_field is frame-0 field. Detach to mirror BPTT-detach in train loop;
    # the only inter-frame gradient path is via the temporal_consistency_loss.
    field_t0_detached = field_t0.clone()
    field_t0_detached.mu = field_t0_detached.mu.detach()
    field_t0_detached.log_scale = field_t0_detached.log_scale.detach()
    field_t0_detached.rotation = field_t0_detached.rotation.detach()
    field_t0_detached.color = field_t0_detached.color.detach()
    field_t0_detached.opacity = field_t0_detached.opacity.detach()

    out_t1, field_t1, _ = model(lr_t1, motion_lr, prev_field=field_t0_detached)

    # Synthesize a coarse HR ground truth — bilinear-upscaled LR rgb.
    gt_t1 = torch.nn.functional.interpolate(
        lr_t1[:, :3], size=(h_lr * 2, w_lr * 2), mode="bilinear", align_corners=False
    )

    # Composite loss per plan.
    l1 = (out_t1 - gt_t1).abs().mean()
    tc = temporal_consistency_loss(out_t1, out_t0.detach(), motion_lr, scale_factor=2.0)
    reg = gaussian_regularization_loss(
        field_t=field_t1,
        field_t_minus_1=field_t0_detached,
        max_area=64.0,
        max_count=model.max_count,
    )
    loss = l1 + 0.05 * tc + 0.01 * reg

    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()

    # Encoder grads — at least one parameter should have a finite, non-trivial grad.
    enc_has_grad = False
    for p in model.encoder.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "non-finite gradient in encoder"
            if p.grad.abs().sum() > 0:
                enc_has_grad = True
    assert enc_has_grad, "no non-zero gradient observed on encoder parameters"

    # Transformer grads — at least one parameter should have a finite, non-trivial grad.
    tfm_has_grad = False
    for p in model.transformer.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "non-finite gradient in transformer"
            if p.grad.abs().sum() > 0:
                tfm_has_grad = True
    assert tfm_has_grad, "no non-zero gradient observed on transformer parameters"


def test_export_in_package_namespace() -> None:
    """GaussianTemporalSRModel is importable from oss.sr.gaussian_temporal."""
    from oss.sr.gaussian_temporal import GaussianTemporalSRModel as M2

    assert M2 is GaussianTemporalSRModel
