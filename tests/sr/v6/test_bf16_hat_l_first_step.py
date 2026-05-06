"""Regression for the v6 HAT-L bf16 first-step training path."""
from __future__ import annotations

import torch
import pytest

from oss.sr.v6.losses import V6CompositeLoss
from oss.sr.v6.model import V6Config, V6Model


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="HAT-L 128x128 bf16 first-step regression is CUDA-only in the suite",
)
def test_hat_l_bf16_first_step_forward_backward_finite() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = V6Model(V6Config(backbone="hat-l")).to(device)
    loss_fn = V6CompositeLoss(
        gan_warmup_until_step=20_000,
        use_lpips=False,
    ).to(device)

    lr_inputs = torch.randn(1, 9, 128, 128, device=device)
    target = torch.rand(1, 3, 256, 256, device=device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        pred = model(lr_inputs, motion_lr=None, frame_index=0)
        loss, parts = loss_fn(pred, target, fake_logits=None, step=1)

    assert torch.isfinite(pred).all()
    assert torch.isfinite(loss), parts

    loss.backward()

    grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
