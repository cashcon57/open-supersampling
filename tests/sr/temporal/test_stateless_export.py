"""Stateless wrapper of v5 pixel-temporal SR model for ONNX export.

These tests cover the scaffolding of ``TemporalSRModelStateless`` — a thin
``nn.Module`` that exposes ``prev_hr`` as an explicit graph input (rather than
internal engine state) so that ``torch.onnx.export`` can capture the full
forward pass for TRT / DirectML / OpenVINO consumption.

We intentionally do NOT exercise ONNX export here (requires CUDA + a real v5
checkpoint). The scaffold tests just confirm:

1. The wrapper produces an output with the same shape as the stateful model.
2. Numerical output matches ``TemporalSRModel.forward`` for the same inputs
   (the wrapper must be a pure passthrough — no math drift).
3. ``from_temporal_checkpoint`` loads weights from a ``TemporalSRModel`` ckpt.
4. ``make_first_frame_prev_hr``-initialised prev_hr produces a finite,
   correctly-shaped output (the export-time first-frame contract).
5. Wrapper returns BOTH the HR output and the disocclusion mask (debug output
   for runtime visualisation / scene-cut UX).
"""

from __future__ import annotations

from pathlib import Path

import torch

from oss.sr.temporal import TemporalSRModel, make_first_frame_prev_hr
from oss.sr.temporal.stateless_export import TemporalSRModelStateless


def _build_model(seed: int = 0) -> TemporalSRModel:
    torch.manual_seed(seed)
    return TemporalSRModel(in_channels=12, scale=2, tier="standard")


def _dummy_inputs(
    batch: int = 1, h_lr: int = 8, w_lr: int = 8, scale: int = 2
) -> dict[str, torch.Tensor]:
    h_hr, w_hr = h_lr * scale, w_lr * scale
    return dict(
        lr_inputs=torch.rand(batch, 12, h_lr, w_lr),
        prev_hr=torch.rand(batch, 3, h_hr, w_hr),
        depth_hr_curr=torch.rand(batch, 1, h_hr, w_hr),
        depth_hr_prev=torch.rand(batch, 1, h_hr, w_hr),
        motion_lr=torch.zeros(batch, 2, h_lr, w_lr),
    )


def test_stateless_wrapper_output_shape_matches_stateful() -> None:
    model = _build_model()
    wrapper = TemporalSRModelStateless(model)
    wrapper.train(False)
    model.train(False)

    inputs = _dummy_inputs()
    with torch.no_grad():
        ref = model(**inputs)
        out_hr, _ = wrapper(**inputs)

    assert out_hr.shape == ref.shape
    assert out_hr.shape == (1, 3, 16, 16)


def test_stateless_wrapper_matches_stateful_numerically() -> None:
    """The wrapper must NOT change the math — same weights -> same output."""
    model = _build_model(seed=42)
    wrapper = TemporalSRModelStateless(model)
    wrapper.train(False)
    model.train(False)

    inputs = _dummy_inputs()
    with torch.no_grad():
        ref = model(**inputs)
        out_hr, _ = wrapper(**inputs)

    # Same module references -> bitwise identical.
    assert torch.equal(ref, out_hr)


def test_stateless_wrapper_returns_disocclusion_mask() -> None:
    """The wrapper exposes the disocclusion mask as a debug output."""
    model = _build_model()
    wrapper = TemporalSRModelStateless(model)
    wrapper.train(False)

    inputs = _dummy_inputs()
    with torch.no_grad():
        out_hr, disoccl = wrapper(**inputs)

    # HR shape, single channel, in [0, 1] (sigmoid output).
    assert disoccl.shape == (1, 1, 16, 16)
    assert torch.all(disoccl >= 0.0) and torch.all(disoccl <= 1.0)
    # Output HR must be finite.
    assert torch.isfinite(out_hr).all()


def test_first_frame_init_produces_reasonable_output() -> None:
    """make_first_frame_prev_hr should produce a usable prev_hr buffer."""
    model = _build_model()
    wrapper = TemporalSRModelStateless(model)
    wrapper.train(False)

    inputs = _dummy_inputs()
    # Override prev_hr with the first-frame initialisation contract.
    inputs["prev_hr"] = make_first_frame_prev_hr(inputs["lr_inputs"][:, :3], scale=2)
    with torch.no_grad():
        out_hr, disoccl = wrapper(**inputs)

    assert out_hr.shape == (1, 3, 16, 16)
    assert torch.isfinite(out_hr).all()
    assert torch.isfinite(disoccl).all()


def test_from_temporal_checkpoint_loads_weights(tmp_path: Path) -> None:
    """Load weights from a TemporalSRModel ckpt into the stateless wrapper."""
    model = _build_model(seed=7)
    ckpt = tmp_path / "temporal.pt"
    torch.save(
        {
            "temporal_model": model.state_dict(),
            "args": {
                "tier": "standard",
                "sr_backbone": "simple",
                "in_channels": 12,
                "scale": 2,
                "backbone_kind": "simple",
            },
        },
        ckpt,
    )

    wrapper = TemporalSRModelStateless.from_temporal_checkpoint(ckpt, device="cpu")
    wrapper.train(False)
    model.train(False)

    inputs = _dummy_inputs()
    with torch.no_grad():
        ref = model(**inputs)
        out_hr, _ = wrapper(**inputs)

    assert torch.allclose(ref, out_hr, atol=1e-6)
