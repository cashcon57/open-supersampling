"""Unit tests for v6.2 low-rank splat decode path."""
from __future__ import annotations

import torch

from oss.sr.v6.activations import SqrSwish
from oss.sr.v6.composite_head import LatentDecoder
from oss.sr.v6.model import V6Config, V6Model


def test_latent_decoder_shapes():
    B, R, H, W = 2, 4, 32, 64
    dec = LatentDecoder(latent_R=R)
    Z = torch.randn(B, R, H, W)
    m = torch.rand(B, 1, H, W)
    I_base = torch.rand(B, 3, H, W)
    out = dec(Z, m, I_base)
    assert out.shape == (B, 3, H, W)
    assert isinstance(dec.act, SqrSwish)
    assert (
        dec.depthwise.groups
        == dec.depthwise.in_channels
        == dec.depthwise.out_channels
    )


def test_latent_decoder_zero_init_means_zero_delta():
    """At init, conv2 weights are zero, so DeltaI must be zero."""
    dec = LatentDecoder(latent_R=4)
    Z = torch.randn(1, 4, 8, 8)
    m = torch.rand(1, 1, 8, 8)
    I_base = torch.rand(1, 3, 8, 8)
    delta = dec(Z, m, I_base)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6), (
        "LatentDecoder DeltaI must be zero at init"
    )


def test_v6model_fresh_forward_identity_at_init_matches_ground_truth():
    """Fresh full v6.2 forward is identity when the base equals ground truth."""
    torch.manual_seed(0)
    model = V6Model(
        V6Config(
            in_channels=9,
            scale=2,
            backbone="hat-tiny",
            canvas_capacity=16,
            token_dim=32,
            cross_attention_heads=4,
            window_size=16,
            tile_size_lr=16,
        )
    ).eval()
    lr_rgb = torch.full((1, 3, 16, 16), 0.25)
    lr = torch.cat([lr_rgb, torch.zeros(1, 6, 16, 16)], dim=1)
    ground_truth = torch.full((1, 3, 32, 32), 0.25)

    with torch.no_grad():
        out = model(lr, motion_lr=None, frame_index=0)

    torch.testing.assert_close(out, ground_truth, atol=1e-6, rtol=1e-6)
