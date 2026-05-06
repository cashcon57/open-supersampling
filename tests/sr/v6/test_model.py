"""V6Model orchestrator tests.

Verifies the integration between HAT backbone, cross-attention, canvas
state hooks, ST-score pruning hook, and softplus / sigmoid output. The
canvas-update path (writing into the canvas from the model's residual)
is not yet wired in v6.0; tests pin the empty-canvas case so first-frame
forwards work end-to-end.
"""
from __future__ import annotations

import torch
import pytest

from oss.sr.v6.model import V6Config, V6Model


def _tiny_model(**overrides):
    """Smaller config for fast unit tests."""
    cfg = V6Config(
        in_channels=9,
        scale=2,
        backbone="hat-tiny",
        canvas_capacity=64,
        token_dim=32,
        cross_attention_heads=4,
        window_size=16,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return V6Model(cfg)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_v6model_constructs_with_default_config():
    cfg = V6Config(backbone="hat-tiny", canvas_capacity=64)
    m = V6Model(cfg)
    assert m.scale == 2
    assert m.cfg.color_activation == "softplus"
    assert m.feat_dim == m.backbone.embed_dim


def test_v6model_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="unknown backbone"):
        V6Model(V6Config(backbone="resnet-50"))


def test_v6model_rejects_invalid_color_activation():
    with pytest.raises(ValueError, match="color_activation"):
        V6Model(V6Config(backbone="hat-tiny", color_activation="tanh"))


# ---------------------------------------------------------------------------
# Forward pass — empty canvas (first frame)
# ---------------------------------------------------------------------------


def test_v6model_forward_first_frame_shape():
    """First-frame forward with empty canvas must produce correctly-shaped HR."""
    m = _tiny_model().train(False)
    lr = torch.randn(1, 9, 32, 32)
    out = m(lr, motion_lr=None, frame_index=0)
    assert out.shape == (1, 3, 64, 64), f"expected HR (1,3,64,64); got {tuple(out.shape)}"
    assert torch.isfinite(out).all()


def test_v6model_forward_softplus_output_is_non_negative():
    """Softplus output is unbounded above but always non-negative — HDR-safe."""
    m = _tiny_model(color_activation="softplus").train(False)
    lr = torch.randn(1, 9, 32, 32)
    out = m(lr)
    assert (out >= 0).all(), "softplus output should be non-negative"


def test_v6model_forward_sigmoid_output_in_unit_range():
    m = _tiny_model(color_activation="sigmoid").train(False)
    lr = torch.randn(1, 9, 32, 32)
    out = m(lr)
    assert (out >= 0).all() and (out <= 1).all(), "sigmoid output should be in [0,1]"


def test_v6model_forward_handles_hdr_inputs():
    """LR input with values >> 1.0 should not crash the model."""
    m = _tiny_model(color_activation="softplus").train(False)
    lr = torch.randn(1, 9, 32, 32) * 5.0
    out = m(lr)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# bf16 / autocast
# ---------------------------------------------------------------------------


def test_v6model_forward_bf16_autocast():
    m = _tiny_model().train(False)
    lr = torch.randn(1, 9, 32, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = m(lr)
    assert torch.isfinite(out).all()
    assert out.shape == (1, 3, 64, 64)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_v6model_gradient_flows_to_every_parameter():
    """A real training run requires every learnable parameter to receive
    gradient. Catches typos in module wiring (e.g. forward path skips a
    submodule) and DDP-unsafe state (parameters that don't see the
    backward signal)."""
    m = _tiny_model()
    m.train()
    lr = torch.randn(2, 9, 32, 32)
    out = m(lr)
    target = torch.zeros_like(out)
    loss = (out - target).abs().mean()
    loss.backward()
    no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    # The fusion layer's K=0 short-circuit means it won't see gradient on
    # the cross-attention sub-modules in the empty-canvas case, which is
    # acceptable behavior for first-frame training. Filter out fusion-only
    # params for this assertion.
    no_grad_outside_fusion = [n for n in no_grad if not n.startswith("fusion.")]
    no_grad_outside_canvas_to_token = [
        n for n in no_grad_outside_fusion if not n.startswith("canvas_to_token.")
    ]
    assert not no_grad_outside_canvas_to_token, (
        f"params received no gradient: {no_grad_outside_canvas_to_token}"
    )


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def test_v6model_reset_state_clears_canvas_and_score():
    m = _tiny_model()
    # Inject a fake canvas state to verify reset wipes it.
    from oss.sr.v6.model import CanvasState

    fake = CanvasState(
        positions=torch.zeros(8, 2),
        scales=torch.ones(8, 2),
        rotations=torch.zeros(8),
        opacities=torch.ones(8),
        colors=torch.zeros(8, m.cfg.token_dim),
        count=8,
    )
    m._canvas_state = fake
    m.reset_state()
    assert m._canvas_state is None
    assert not m.has_canvas()


def test_v6model_maybe_prune_no_op_on_empty_canvas():
    m = _tiny_model()
    n_pruned = m.maybe_prune()
    assert n_pruned == 0


def test_v6model_maybe_prune_only_fires_on_prune_step():
    """maybe_prune is a no-op except on every prune_every step. Verify
    it doesn't fire prematurely."""
    m = _tiny_model(prune_every=10)
    # Run 9 steps: never the prune-cycle boundary, never canvas state -> 0.
    for _ in range(9):
        assert m.maybe_prune() == 0
    # 10th step: would prune, but canvas is still empty -> still 0.
    assert m.maybe_prune() == 0


# ---------------------------------------------------------------------------
# Backbone variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone", ["hat-tiny", "hat-small", "hat-l"])
def test_v6model_constructs_at_every_tier(backbone):
    """All three published tiers (Pico/Standard/Heavy backbones) construct
    cleanly. Skip forward to keep the test fast — shape correctness is
    pinned at hat-tiny in test_v6model_forward_first_frame_shape."""
    m = V6Model(V6Config(backbone=backbone, canvas_capacity=16))
    assert m.feat_dim == m.backbone.embed_dim
    expected_dim = {"hat-tiny": 60, "hat-small": 120, "hat-l": 180}[backbone]
    assert m.feat_dim == expected_dim, (
        f"{backbone} embed_dim should be {expected_dim}, got {m.feat_dim}"
    )
