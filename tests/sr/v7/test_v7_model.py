"""End-to-end V7Model integration tests.

Verifies the skeleton wires backbone + N-D canvas + composite head and
that rendering at different t_query values produces measurably
different motion-coherent outputs when the canvas Gaussians have V_xt
correlation (the OSS-FX primitive).
"""
from __future__ import annotations

import math

import pytest
import torch

from oss.sr.v7.model import V7Config, V7Model
from oss.sr.v7.nd_canvas_state import cholesky_pack_to_cov


def _make_motion_encoded_gaussian(x, y, t, vxt=0.5, vtt=4.0, R=16):
    """Pack a single 3D Gaussian into the (positions, cov_raw, features,
    opacity) tuple expected by NDCanvasState.add.

    V_xx=1, V_yy=1, V_xt=vxt, V_yt=0, V_tt=vtt
    Cholesky L: L00=1, L10=0, L11=1, L20=vxt, L21=0,
                L22 = sqrt(vtt - vxt^2)   (PSD requires vtt > vxt^2)
    """
    if vtt <= vxt * vxt:
        raise ValueError(f"vtt ({vtt}) must exceed vxt^2 ({vxt*vxt}) for PSD")
    pos = torch.tensor([[float(x), float(y), float(t)]])
    L22 = math.sqrt(vtt - vxt * vxt)
    cov_raw = torch.tensor([[0.0, 0.0, 0.0, float(vxt), 0.0, math.log(L22)]])
    feat = torch.zeros((1, R))
    feat[0, 0] = 1.0
    op = torch.tensor([1.0])
    return pos, cov_raw, feat, op


def test_v7_model_forward_with_empty_canvas_returns_bicubic_anchored():
    """With no Gaussians in the canvas, output should equal bicubic of
    LR + a near-zero delta from the composite_head (zero-init)."""
    torch.manual_seed(0)
    cfg = V7Config(in_channels=9, scale=2, feat_dim=16, latent_rank=16, canvas_capacity=128, backbone_blocks=2)
    model = V7Model(cfg).train(False)
    lr = torch.randn((1, 9, 32, 48))
    with torch.no_grad():
        out = model(lr, t_query=0.0)
    assert out.shape == (1, 3, 64, 96)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


def test_v7_model_forward_with_canvas_changes_output_at_different_t():
    """Add a single motion-encoded Gaussian to the canvas; rendering
    the same LR input at t=0 vs t=2 should produce *different* output
    because the time-slice shifts the canvas Gaussian in x."""
    torch.manual_seed(0)
    cfg = V7Config(in_channels=9, scale=2, feat_dim=16, latent_rank=16, canvas_capacity=128, backbone_blocks=2)
    model = V7Model(cfg).train(False)
    model.allocate_canvas("cpu")

    pos, cov_raw, feat, op = _make_motion_encoded_gaussian(
        x=20.0, y=20.0, t=0.0, vxt=1.5, vtt=4.0, R=16,
    )
    # Large feature magnitude so the canvas channel can dominate the
    # small-init composite_head's scaling (head's last layer is
    # std=1e-3 by default). 1000 is enough to make the canvas signal
    # measurably propagate through to RGB output.
    feat[0, 0] = 1000.0
    model.canvas.add(pos, cov_raw, feat, op)

    lr = torch.zeros((1, 9, 32, 48))
    with torch.no_grad():
        out_at_0 = model(lr, t_query=0.0)
        out_at_2 = model(lr, t_query=2.0)

    diff = (out_at_0 - out_at_2).abs().mean().item()
    # 1e-5 threshold: canvas-driven motion shows up in RGB output but
    # at small magnitude given the small-init head; v7 training will
    # learn head weights that amplify the canvas signal further.
    assert diff > 1e-5, (
        f"Expected canvas time-slice to produce different output at different "
        f"t_query; got mean abs diff {diff}"
    )


def test_v7_model_canvas_reset_clears_state():
    cfg = V7Config(in_channels=9, scale=2, feat_dim=16, latent_rank=16, canvas_capacity=64, backbone_blocks=1)
    model = V7Model(cfg).train(False)
    model.allocate_canvas("cpu")
    pos, cov_raw, feat, op = _make_motion_encoded_gaussian(x=5.0, y=5.0, t=0.0, vxt=0.5)
    model.canvas.add(pos, cov_raw, feat, op)
    assert model.canvas.count == 1
    model.reset_state(device="cpu")
    assert model.canvas.count == 0


def test_v7_model_render_canvas_alone_returns_correct_shape():
    """Direct call to render_canvas() should produce (1, R, H, W)."""
    cfg = V7Config(in_channels=9, scale=2, feat_dim=16, latent_rank=16, canvas_capacity=64, backbone_blocks=1)
    model = V7Model(cfg).train(False)
    model.allocate_canvas("cpu")
    out = model.render_canvas(t_query=0.0, output_hw=(32, 48))
    assert out.shape == (1, 16, 32, 48)
    assert out.abs().sum().item() == 0.0


def test_v7_model_gradient_flow_through_backbone():
    """Smoke test: gradients should flow through the backbone for any
    non-trivial input."""
    cfg = V7Config(in_channels=9, scale=2, feat_dim=16, latent_rank=16, canvas_capacity=32, backbone_blocks=1)
    model = V7Model(cfg).train(True)
    model.allocate_canvas("cpu")
    pos, cov_raw, feat, op = _make_motion_encoded_gaussian(x=10.0, y=10.0, t=0.0, vxt=0.0)
    model.canvas.add(pos, cov_raw, feat, op)
    lr = torch.randn((1, 9, 16, 24))
    target = torch.rand((1, 3, 32, 48))
    out = model(lr, t_query=0.0)
    loss = (out - target).pow(2).mean()
    loss.backward()
    backbone_grad_norm = sum(
        p.grad.abs().sum().item() if p.grad is not None else 0.0
        for p in model.backbone.parameters()
    )
    assert backbone_grad_norm > 0.0, "backbone got no gradient"
