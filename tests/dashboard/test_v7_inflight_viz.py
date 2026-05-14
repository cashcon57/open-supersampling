"""Tests for v7 inflight-viz daemon + dashboard column wiring.

Covers:
  - viz_columns_for_run("srcnn-v7.0-pico-005") -> the 9-column v7 layout.
  - _is_v7_checkpoint detects ckpts with cfg["backbone_kind"] and rejects
    v5/v6 dict shapes that lack it.
  - _v7_predictions returns the expected {v7_alpha_1, v7_alpha_0_5} key
    set with HR-shaped (1, 3, H, W) tensors.
  - _bicubic_midpoint_triplet matches sr_eval_v7.py's symmetric average.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_viz_columns_v7_pico_005_returns_nine_column_layout():
    """The dashboard must publish the exact column order the daemon
    draws. Both must agree on naming and ordering, or the panel labels
    drift out from under the dashboard's tooltip text."""
    from scripts.build_public_dashboard import viz_columns_for_run

    cols = viz_columns_for_run("srcnn-v7.0-pico-005")
    assert cols == [
        "LR-bilinear",
        "bicubic",
        "bicubic-midpoint",
        "v6.2",
        "v7 alpha=1",
        "v7 alpha=0.5",
        "GT",
        "GT-half",
        "|err v7 alpha=0.5|",
    ]


def test_is_v7_checkpoint_detects_backbone_kind(tmp_path):
    """Mock a v7-shaped ckpt and verify the detector recognizes it.
    Also verify a v5/v6-shaped ckpt (no cfg/backbone_kind) is rejected.
    """
    torch = pytest.importorskip("torch")

    from scripts.sr_temporal_inflight_viz import _is_v7_checkpoint

    # v7-shaped: has cfg["backbone_kind"]
    v7_ckpt = tmp_path / "step-00001000.pt"
    torch.save(
        {
            "step": 1000,
            "model_state": {},
            "cfg": {"backbone_kind": "placeholder", "scale": 2, "feat_dim": 32},
            "args": {},
        },
        v7_ckpt,
    )
    assert _is_v7_checkpoint(v7_ckpt) is True

    # v5-shaped: top-level args + sr_model/temporal_model, no cfg
    v5_ckpt = tmp_path / "v5.pt"
    torch.save({"args": {"tier": "standard"}, "temporal_model": {}}, v5_ckpt)
    assert _is_v7_checkpoint(v5_ckpt) is False

    # v6-shaped: has v6_config but not cfg
    v6_ckpt = tmp_path / "v6.pt"
    torch.save(
        {"args": {"backbone": "hat-l"}, "v6_config": {"backbone": "hat-l"}, "v6_model": {}},
        v6_ckpt,
    )
    assert _is_v7_checkpoint(v6_ckpt) is False


def test_v7_predictions_returns_expected_keys_and_shapes():
    """Build a V7Model with the placeholder backbone (fast, no HAT
    dependency), run _v7_predictions, and verify both heads return
    (1, 3, H_hr, W_hr) tensors."""
    torch = pytest.importorskip("torch")

    from oss.sr.v7.model import V7Config, V7Model
    from scripts.sr_temporal_inflight_viz import _v7_predictions

    cfg = V7Config(
        in_channels=9,
        scale=2,
        feat_dim=16,
        latent_rank=8,
        canvas_capacity=128,
        backbone_blocks=1,
        backbone_kind="placeholder",
        # Disable parent-child so the model's forward doesn't depend on
        # trainer-only hooks.
        enable_parent_child=False,
    )
    model = V7Model(cfg)
    model.allocate_canvas("cpu")
    model.train(False)

    # 9-channel LR input at a small spatial size to keep the test fast.
    h_lr, w_lr = 8, 8
    h_hr, w_hr = h_lr * cfg.scale, w_lr * cfg.scale
    n_in = torch.zeros(1, cfg.in_channels, h_lr, w_lr)
    np1_in = torch.zeros(1, cfg.in_channels, h_lr, w_lr)

    preds = _v7_predictions(model, n_in, np1_in, output_hw=(h_hr, w_hr), device="cpu")

    assert set(preds.keys()) == {"v7_alpha_1", "v7_alpha_0_5"}
    for key, tensor in preds.items():
        assert tensor.shape == (1, 3, h_hr, w_hr), (
            f"{key}: expected (1, 3, {h_hr}, {w_hr}), got {tensor.shape}"
        )
        # Outputs must be in [0, 1] after the model's clamp.
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) <= 1.0


def test_bicubic_midpoint_triplet_matches_eval_helper():
    """The daemon's bicubic-midpoint baseline must equal
    sr_eval_v7.py::_bicubic_midpoint up to numerics so the +1 dB pass
    criterion compares apples to apples between inflight viz and the
    headline eval JSON."""
    torch = pytest.importorskip("torch")

    from scripts.sr_eval_v7 import _bicubic_midpoint as eval_midpoint
    from scripts.sr_temporal_inflight_viz import _bicubic_midpoint_triplet

    h_lr, w_lr = 8, 8
    h_hr, w_hr = 16, 16
    # _bicubic_midpoint (in sr_eval_v7) slices [:, :3] internally, so
    # we pass it 9-channel inputs to match its production contract.
    n_lr_9 = torch.rand(1, 9, h_lr, w_lr)
    np1_lr_9 = torch.rand(1, 9, h_lr, w_lr)
    n_lr_3 = n_lr_9[:, :3]
    np1_lr_3 = np1_lr_9[:, :3]

    eval_out = eval_midpoint(n_lr_9, np1_lr_9, (h_hr, w_hr))
    viz_out = _bicubic_midpoint_triplet(n_lr_3, np1_lr_3, (h_hr, w_hr))

    assert eval_out.shape == viz_out.shape == (1, 3, h_hr, w_hr)
    assert torch.allclose(eval_out, viz_out, atol=1e-6)
