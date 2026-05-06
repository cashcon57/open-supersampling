"""V6Model orchestrator tests.

Verifies the canonical v6 Stage 2 path: HAT features, warped persistent
canvas, active-mask fusion, HR rasterization, composite RGB head, Gaussian
write-back, ST-score state, and bicubic-residual RGB output.
"""
from __future__ import annotations

from unittest.mock import Mock

import torch
import torch.nn.functional as F
import pytest

from oss.sr.v6.model import CanvasState, V6Config, V6Model


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


def _canvas_state(count: int, token_dim: int) -> CanvasState:
    positions = torch.stack(
        [
            torch.linspace(4.0, 28.0, count),
            torch.linspace(6.0, 26.0, count),
        ],
        dim=-1,
    )
    return CanvasState(
        positions=positions,
        scales=torch.ones(count, 2),
        rotations=torch.zeros(count),
        opacities=torch.ones(count),
        colors=torch.randn(count, token_dim),
        count=count,
    )


def _motion(
    dx: float,
    dy: float,
    h: int = 32,
    w: int = 32,
    batch: int = 1,
) -> torch.Tensor:
    motion = torch.zeros(batch, 2, h, w)
    motion[:, 0].fill_(dx)
    motion[:, 1].fill_(dy)
    return motion


def _realistic_lr(
    batch: int = 1,
    h: int = 32,
    w: int = 32,
) -> torch.Tensor:
    """RGB in a plausible SDR range plus zeroed depth/motion/normal channels."""
    rgb = torch.rand(batch, 3, h, w) * 0.7 + 0.1
    gbuffers = torch.zeros(batch, 6, h, w)
    return torch.cat([rgb, gbuffers], dim=1)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_v6model_constructs_with_default_config():
    cfg = V6Config(backbone="hat-tiny", canvas_capacity=64)
    m = V6Model(cfg)
    assert m.scale == 2
    assert m.cfg.color_activation == "hdr"
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
    """First-frame empty canvas populates via write-back and renders HR."""
    m = _tiny_model().train(False)
    lr = torch.randn(1, 9, 32, 32)
    out = m(lr, motion_lr=None, frame_index=0)
    assert out.shape == (1, 3, 64, 64), f"expected HR (1,3,64,64); got {tuple(out.shape)}"
    assert torch.isfinite(out).all()
    assert m.has_canvas()
    assert m._canvas_state is not None
    assert m._canvas_state.count == 16
    assert m._st_state is not None
    assert m._st_state.spatial_accumulator.shape == (16,)


def test_v6model_empty_canvas_tokens_short_circuit():
    """Empty canvas must keep returning K=0 tokens for first-frame forwards."""
    m = _tiny_model().train(False)
    feats = torch.randn(1, m.feat_dim, 32, 32)
    tokens = m._build_canvas_tokens(feats, frame_index=0)
    assert tokens.shape == (1, 0, m.cfg.token_dim)
    assert torch.isfinite(tokens).all()


def test_v6model_forward_with_nonempty_canvas():
    m = _tiny_model()
    m.train()
    m._canvas_state = _canvas_state(count=8, token_dim=m.cfg.token_dim)
    lr = torch.randn(1, 9, 32, 32)

    out = m(lr, frame_index=0)
    assert out.shape == (1, 3, 64, 64)
    assert torch.isfinite(out).all()

    loss = out.square().mean()
    loss.backward()

    canvas_grads = [
        p.grad for p in m.canvas_to_token.parameters() if p.requires_grad
    ]
    fusion_grads = [p.grad for p in m.fusion.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in canvas_grads)
    assert all(g is not None and torch.isfinite(g).all() for g in fusion_grads)
    assert sum(float(g.abs().sum()) for g in canvas_grads) > 0.0
    assert sum(float(g.abs().sum()) for g in fusion_grads) > 0.0


def test_v6model_multiframe_canvas_warps_and_stays_bounded():
    m = _tiny_model(canvas_capacity=10, tile_size_lr=16).train(False)
    lr = torch.randn(1, 9, 32, 32)
    step_motion = _motion(2.0, 1.0)

    out = m(lr, motion_lr=None, frame_index=0)
    assert out.shape == (1, 3, 64, 64)
    assert m._canvas_state is not None
    assert m._canvas_state.count == 4

    for frame_index in range(1, 4):
        before = m._canvas_state.positions.detach().clone()
        prev_count = int(m._canvas_state.count)
        out = m(lr, motion_lr=step_motion, frame_index=frame_index)
        assert out.shape == (1, 3, 64, 64)
        assert torch.isfinite(out).all()
        assert m._canvas_state is not None
        assert m._canvas_state.count <= m.cfg.canvas_capacity

        spawned_per_frame = 4
        survivors = min(prev_count, m.cfg.canvas_capacity - spawned_per_frame)
        if survivors > 0:
            expected = before[-survivors:] + torch.tensor([2.0, 1.0])
            actual = m._canvas_state.positions[:survivors].detach()
            torch.testing.assert_close(actual, expected, atol=1.0e-4, rtol=1.0e-4)


def test_v6model_gradient_flows_through_stage2_path():
    m = _tiny_model(tile_size_lr=16)
    m.train()
    lr0 = torch.randn(1, 9, 32, 32)
    lr1 = torch.randn(1, 9, 32, 32)

    m(lr0, motion_lr=None, frame_index=0)
    out = m(lr1, motion_lr=_motion(0.5, 0.25), frame_index=1)
    loss = out.sum()
    loss.backward()

    assert m.gaussian_spawner.conv.weight.grad is not None
    assert float(m.gaussian_spawner.conv.weight.grad.abs().sum()) > 0.0
    assert m.fusion.q_proj.weight.grad is not None
    assert float(m.fusion.q_proj.weight.grad.abs().sum()) > 0.0
    assert m.canvas_to_token.weight.grad is not None
    assert float(m.canvas_to_token.weight.grad.abs().sum()) > 0.0
    assert m.composite_head[0].weight.grad is not None
    assert float(m.composite_head[0].weight.grad.abs().sum()) > 0.0

    backbone_grads = [
        p.grad for p in m.backbone.parameters() if p.requires_grad and p.grad is not None
    ]
    assert backbone_grads
    assert sum(float(g.abs().sum()) for g in backbone_grads) > 0.0


def test_v6model_init_output_has_real_signal_variance():
    torch.manual_seed(0)
    m = _tiny_model().train(False)
    lr = _realistic_lr()

    out = m(lr, motion_lr=None, frame_index=0)

    assert out.std() > 0.05, "init output should preserve bicubic image variance"


def test_v6model_init_output_close_to_bicubic():
    torch.manual_seed(1)
    m = _tiny_model().train(False)
    lr = _realistic_lr()
    bicubic = F.interpolate(
        lr[:, :3],
        size=(lr.shape[-2] * m.scale, lr.shape[-1] * m.scale),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(min=0.0)

    out = m(lr, motion_lr=None, frame_index=0)

    assert (out - bicubic).abs().mean() < 0.05


def test_v6model_forward_softplus_output_is_non_negative():
    """Deprecated softplus alias is HDR-safe: non-negative and uncapped."""
    m = _tiny_model(color_activation="softplus").train(False)
    lr = torch.randn(1, 9, 32, 32)
    out = m(lr)
    assert (out >= 0).all(), "softplus output should be non-negative"


def test_v6model_forward_sigmoid_output_in_unit_range():
    """Deprecated sigmoid alias is SDR-safe: clamped to [0, 1]."""
    m = _tiny_model(color_activation="sigmoid").train(False)
    lr = torch.randn(1, 9, 32, 32)
    out = m(lr)
    assert (out >= 0).all() and (out <= 1).all(), "sigmoid output should be in [0,1]"


def test_v6model_forward_sdr_output_in_unit_range():
    m = _tiny_model(color_activation="sdr").train(False)
    lr = torch.randn(1, 9, 32, 32)
    out = m(lr)
    assert (out >= 0).all() and (out <= 1).all(), "sdr output should be in [0,1]"


def test_v6model_forward_handles_hdr_inputs():
    """LR input with values >> 1.0 should not crash the model."""
    m = _tiny_model(color_activation="hdr").train(False)
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


def test_v6model_bf16_autocast_multiframe_forward_stays_finite():
    m = _tiny_model(tile_size_lr=16).train(False)
    lr = torch.randn(1, 9, 32, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out0 = m(lr, motion_lr=None, frame_index=0)
        out1 = m(lr, motion_lr=_motion(1.0, 0.0), frame_index=1)
        out2 = m(lr, motion_lr=_motion(1.0, 0.0), frame_index=2)
    for out in (out0, out1, out2):
        assert out.shape == (1, 3, 64, 64)
        assert torch.isfinite(out).all()


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


def test_v6model_persistent_state_is_detached_between_frames():
    """Audit finding HIGH-2/3: _canvas_state and _st_state must NOT carry
    autograd across frames. Without detach, BPTT through every previous
    frame's spawner OOMs and leaks gradient state across optimizer steps.
    Pin the contract: after one forward, every persistent tensor
    requires_grad=False.
    """
    m = _tiny_model(tile_size_lr=16)
    m.train()
    lr = torch.randn(1, 9, 32, 32, requires_grad=True)
    out = m(lr, motion_lr=None, frame_index=0)
    # The current-frame output must be live (loss flows back through it)
    assert out.requires_grad
    # Persistent state must be detached for next-frame forward.
    cs = m._canvas_state
    assert cs is not None
    for name in ("positions", "scales", "rotations", "opacities", "colors"):
        t = getattr(cs, name)
        assert not t.requires_grad, (
            f"_canvas_state.{name} should be detached between frames, "
            f"requires_grad={t.requires_grad}"
        )
        assert t.grad_fn is None, (
            f"_canvas_state.{name} should have no grad_fn; got {t.grad_fn}"
        )
    sts = m._st_state
    assert sts is not None
    assert not sts.spatial_accumulator.requires_grad
    assert sts.spatial_accumulator.grad_fn is None


def test_v6model_st_score_uses_footprint_times_opacity():
    """Audit finding HIGH-1 fix: spatial score must combine per-Gaussian
    footprint area (~ 2π·s_x·s_y) with opacity, not opacity alone. Two
    Gaussians with identical opacity but different scales must produce
    different SS values, with the larger-footprint Gaussian higher.
    """
    import math as _m
    from oss.sr.v6.model import CanvasState

    m = _tiny_model(tile_size_lr=16)
    # Two Gaussians, same opacity, scales differ by 4x in product.
    canvas = CanvasState(
        positions=torch.tensor([[16.0, 16.0], [48.0, 48.0]]),
        scales=torch.tensor([[1.0, 1.0], [2.0, 2.0]]),  # det ratio = 16
        rotations=torch.zeros(2),
        opacities=torch.full((2,), 0.5),
        colors=torch.zeros(2, m.cfg.token_dim),
        count=2,
    )
    m._canvas_state = canvas
    active = torch.ones(2, dtype=torch.bool)
    m._update_st_state(canvas, active, previous_state=None, old_count=0, new_count=2)

    spatial = m._st_state.spatial_accumulator
    assert spatial.shape == (2,)
    # SS_i = 2π · s_x · s_y · opacity
    expected_0 = 2 * _m.pi * 1.0 * 1.0 * 0.5
    expected_1 = 2 * _m.pi * 2.0 * 2.0 * 0.5
    assert torch.allclose(
        spatial,
        torch.tensor([expected_0, expected_1]),
        atol=1e-4,
    ), f"expected {[expected_0, expected_1]}, got {spatial.tolist()}"
    # Larger-footprint Gaussian must outscore the smaller, even at same opacity.
    assert spatial[1].item() > spatial[0].item() * 3.5  # 4× theoretical


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


def test_v6model_reset_between_trajectories_repopulates_from_frame_zero():
    m = _tiny_model(tile_size_lr=16).train(False)
    lr = torch.randn(1, 9, 32, 32)

    m(lr, motion_lr=None, frame_index=0)
    m(lr, motion_lr=_motion(1.0, 0.0), frame_index=1)
    assert m._canvas_state is not None
    assert m._canvas_state.count == 8

    m.reset_state()
    assert m._canvas_state is None
    assert m._st_state is None
    assert int(m._step_count.item()) == 0

    out = m(lr, motion_lr=None, frame_index=0)
    assert out.shape == (1, 3, 64, 64)
    assert m._canvas_state is not None
    assert m._canvas_state.count == 4


def test_v6model_frame_index_threads_to_keyframe_cache():
    m = _tiny_model(prune_every=100)
    m._canvas_state = _canvas_state(count=8, token_dim=m.cfg.token_dim)
    mask = torch.ones(8, dtype=torch.bool)
    m.keyframe_mask.get_mask = Mock(return_value=mask)
    lr = torch.randn(1, 9, 32, 32)

    m.maybe_prune()
    m(lr, frame_index=0)
    m.maybe_prune()
    m(lr, frame_index=10)

    frame_indices = [
        call.kwargs["frame_index"] for call in m.keyframe_mask.get_mask.call_args_list
    ]
    assert frame_indices == [0, 10]


def test_v6model_step_count_serializes_and_resets():
    m = _tiny_model()
    for _ in range(17):
        m.maybe_prune()
    assert int(m._step_count.item()) == 17

    restored = _tiny_model()
    restored.load_state_dict(m.state_dict())
    assert int(restored._step_count.item()) == 17

    restored.reset_state()
    assert int(restored._step_count.item()) == 0


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
