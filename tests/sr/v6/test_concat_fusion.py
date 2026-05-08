# SPDX-License-Identifier: Apache-2.0
"""Unit tests for v6.2 ConcatFusion."""
from __future__ import annotations

import torch

from oss.sr.v6.activations import SqrSwish
from oss.sr.v6.concat_fusion import ConcatFusion


def test_sqrswish_zero_at_origin() -> None:
    """SqrSwish(0) = 0 and finite inputs produce finite outputs."""
    act = SqrSwish()
    assert torch.allclose(act(torch.tensor(0.0)), torch.tensor(0.0))

    x = torch.linspace(-3, 3, 100)
    y = act(x)
    expected = 0.5 * x * (1.0 + x / torch.sqrt(x * x + 1.0))
    assert torch.allclose(y, expected)
    assert torch.isfinite(y).all()


def test_concat_fusion_shapes() -> None:
    B, feat_dim, R, H, W = 2, 180, 4, 32, 64
    fus = ConcatFusion(feat_dim=feat_dim, latent_R=R)
    F = torch.randn(B, feat_dim, H, W)
    G = torch.randn(B, R, H, W)
    m = torch.rand(B, 1, H, W)
    I_base = torch.rand(B, 3, H, W)
    depth = torch.rand(B, 1, H, W)
    MV = torch.randn(B, 2, H, W)

    out = fus(F, G, m, I_base, depth, MV)

    assert out.shape == F.shape


def test_concat_fusion_starts_as_identity() -> None:
    """With zero-init proj_out, ConcatFusion at step 0 must be identity."""
    B, feat_dim, R, H, W = 1, 180, 4, 16, 16
    fus = ConcatFusion(feat_dim=feat_dim, latent_R=R)
    F = torch.randn(B, feat_dim, H, W)

    out = fus(
        F,
        torch.randn(B, R, H, W),
        torch.rand(B, 1, H, W),
        torch.rand(B, 3, H, W),
        torch.rand(B, 1, H, W),
        torch.randn(B, 2, H, W),
    )

    assert torch.allclose(out, F, atol=1e-6)


def test_concat_fusion_no_nan_propagation() -> None:
    """ConcatFusion must not produce NaN given finite inputs."""
    B, feat_dim, R, H, W = 1, 180, 4, 8, 8
    fus = ConcatFusion(feat_dim=feat_dim, latent_R=R)
    with torch.no_grad():
        fus.proj_out.weight.normal_(0, 0.01)
        fus.proj_out.bias.zero_()

    inputs = [
        torch.randn(B, feat_dim, H, W),
        torch.randn(B, R, H, W),
        torch.rand(B, 1, H, W),
        torch.rand(B, 3, H, W),
        torch.rand(B, 1, H, W),
        torch.randn(B, 2, H, W),
    ]
    out = fus(*inputs)

    assert torch.isfinite(out).all()
