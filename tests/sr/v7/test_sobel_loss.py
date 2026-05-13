"""Tests for the Sobel high-frequency edge loss added to oss_fx_loss.

The loss is opt-in (lambda_sobel=0 default) and intended for Standard /
Heavy teacher training where preserving thin geometry matters more than
straight pixel L1.
"""
from __future__ import annotations

import torch

from oss.sr.v7.losses import sobel_grad_l1, oss_fx_loss


def test_sobel_zero_when_pred_equals_target():
    torch.manual_seed(0)
    x = torch.rand((1, 3, 32, 32))
    assert sobel_grad_l1(x, x).item() < 1e-6


def test_sobel_responds_to_edge_intensity_differences():
    """A blurred version of the target should have a strictly smaller
    sobel gradient magnitude, so sobel_grad_l1(blur, target) > 0."""
    torch.manual_seed(1)
    target = torch.zeros((1, 3, 32, 32))
    target[:, :, :, 16:] = 1.0   # sharp vertical edge in the middle
    blurred = target.clone()
    # 5x5 mean-blur of the edge -> softer transition -> smaller gradient
    blurred = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(blurred, (2, 2, 2, 2), mode="replicate"),
        kernel_size=5, stride=1,
    )
    sobel_diff = sobel_grad_l1(blurred, target)
    assert sobel_diff.item() > 0.01, (
        f"Blurring an edge should drop its Sobel magnitude measurably; got {sobel_diff.item()}"
    )


def test_sobel_gradient_flows_backward():
    """Make sure the Sobel kernels work as a differentiable loss term."""
    torch.manual_seed(2)
    pred = torch.rand((1, 3, 16, 16), requires_grad=True)
    target = torch.rand((1, 3, 16, 16))
    loss = sobel_grad_l1(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0


def test_oss_fx_loss_sobel_term_appears_in_parts_when_enabled():
    torch.manual_seed(3)
    pred = torch.rand((1, 3, 16, 16))
    gt = torch.rand((1, 3, 16, 16))
    _, parts = oss_fx_loss(
        out_main=pred, gt_main=gt,
        lambda_charbonnier=1.0, lambda_lpips=0.0, lambda_fg=0.0,
        lambda_fg_lpips=0.0, lambda_temp_consistency=0.0,
        lambda_sobel=0.1,
    )
    assert "sr_sobel" in parts
    assert parts["sr_sobel"] > 0


def test_oss_fx_loss_sobel_term_absent_when_disabled():
    torch.manual_seed(4)
    pred = torch.rand((1, 3, 16, 16))
    gt = torch.rand((1, 3, 16, 16))
    _, parts = oss_fx_loss(
        out_main=pred, gt_main=gt,
        lambda_charbonnier=1.0, lambda_lpips=0.0, lambda_fg=0.0,
        lambda_fg_lpips=0.0, lambda_temp_consistency=0.0,
        lambda_sobel=0.0,
    )
    assert "sr_sobel" not in parts
