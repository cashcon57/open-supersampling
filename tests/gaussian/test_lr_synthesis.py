"""Tests for oss.gaussian.data.lr_synthesis — engine-aliased LR synthesis.

Test-driven development: these tests were written before the implementation.
They validate each component of the engine-aliased LR pipeline and the
EngineAliasedLRSynth orchestrator class.

Context (Decision 3, 2026-05-01 validation memo): training against
bicubic-clean LR creates a "bicubic-LR-trap" where SR networks lose to
bicubic at evaluation. These tests verify the pipeline produces realistic
engine-emitted LR with jitter, TAA blur simulation, and optional artifacts.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from oss.gaussian.data.lr_synthesis import (
    EngineAliasedLRSynth,
    area_downsample,
    apply_jitter,
    halton_jitter,
    jpeg_artifact,
    taa_blur_approx,
)
from oss.gaussian.data.base import GaussianDataset


# ---- Constants ---------------------------------------------------------------

HR_C, HR_H, HR_W = 3, 64, 96
SCALE = 2.0
LR_H, LR_W = HR_H // int(SCALE), HR_W // int(SCALE)


def _hr() -> torch.Tensor:
    """Reproducible HR tensor (3, 64, 96) in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(HR_C, HR_H, HR_W)


# ==============================================================================
# 1. halton_jitter — deterministic, range [-0.5, 0.5]
# ==============================================================================


def test_halton_jitter_deterministic() -> None:
    """Same idx must produce identical jitter regardless of call order."""
    j0a = halton_jitter(0)
    j7a = halton_jitter(7)
    j0b = halton_jitter(0)
    j7b = halton_jitter(7)
    assert j0a == j0b, "halton_jitter(0) is not deterministic"
    assert j7a == j7b, "halton_jitter(7) is not deterministic"


def test_halton_jitter_range() -> None:
    """All values must lie in [-0.5, 0.5] for the first 1000 indices."""
    for i in range(1000):
        jx, jy = halton_jitter(i)
        assert -0.5 <= jx <= 0.5, f"jx={jx} out of range at idx={i}"
        assert -0.5 <= jy <= 0.5, f"jy={jy} out of range at idx={i}"


def test_halton_jitter_returns_tuple_of_floats() -> None:
    jx, jy = halton_jitter(5)
    assert isinstance(jx, float)
    assert isinstance(jy, float)


def test_halton_jitter_different_per_idx() -> None:
    """The sequence should vary — not all identical."""
    jitters = [halton_jitter(i) for i in range(8)]
    xs = [j[0] for j in jitters]
    ys = [j[1] for j in jitters]
    assert len(set(xs)) > 1, "all x jitters are identical — Halton2 is broken"
    assert len(set(ys)) > 1, "all y jitters are identical — Halton3 is broken"


# ==============================================================================
# 2. apply_jitter — output shape matches input
# ==============================================================================


def test_apply_jitter_preserves_shape() -> None:
    hr = _hr()
    for idx in range(8):
        jitter = halton_jitter(idx)
        out = apply_jitter(hr, jitter)
        assert out.shape == hr.shape, f"shape mismatch at idx={idx}"


def test_apply_jitter_returns_float_tensor() -> None:
    hr = _hr()
    out = apply_jitter(hr, (0.0, 0.0))
    assert out.dtype == torch.float32


def test_apply_jitter_zero_offset_approx_identity() -> None:
    """Zero jitter should produce output very close to the original (within
    bilinear interpolation error on a zero-shift grid)."""
    hr = _hr()
    out = apply_jitter(hr, (0.0, 0.0))
    # Allow small float epsilon for grid_sample at exactly 0 shift.
    assert torch.allclose(out, hr, atol=1e-5), "zero jitter deviates more than expected"


def test_apply_jitter_nonzero_changes_pixels() -> None:
    """A nonzero jitter must produce a different tensor."""
    hr = _hr()
    out = apply_jitter(hr, (0.3, 0.2))
    assert not torch.equal(out, hr)


def test_apply_jitter_direction_x_axis() -> None:
    """Positive jx must shift content to the right by jx pixels.

    A sign flip in apply_jitter would silently train the network on the
    wrong jitter polarity. This test catches that.
    """
    # Build a vertical edge: left half black, right half white
    H, W = 16, 32
    img = torch.zeros(3, H, W)
    img[:, :, W // 2:] = 1.0

    # Shift content right by ~1 pixel via positive jx
    shifted = apply_jitter(img, (1.0, 0.0))

    # The edge should now be at column ~W/2 + 1
    # Check column sums: column at W/2 was the boundary; after +1px shift,
    # column at W/2 should now be brighter than original (was 0, now closer to 1)
    original_col = img[0, :, W // 2].sum().item()      # 0
    shifted_col = shifted[0, :, W // 2].sum().item()
    assert shifted_col > original_col + 1.0, (
        f"+1px jitter should shift edge right; col={W//2} sum was {original_col}, became {shifted_col}"
    )


def test_apply_jitter_direction_y_axis_independent() -> None:
    """jy=0 must not change x-direction content; row/col swap detector."""
    H, W = 16, 32
    img = torch.zeros(3, H, W)
    img[:, :, W // 2:] = 1.0  # vertical edge — no Y variation

    # Shift only in X
    shifted_x = apply_jitter(img, (1.0, 0.0))
    # Shift only in Y (jy=1.0, jx=0.0) — should NOT shift the vertical edge horizontally
    shifted_y = apply_jitter(img, (0.0, 1.0))

    # Y-only shift should leave column sums approximately unchanged on a vertical edge
    col_sums_orig = img.sum(dim=(0, 1))         # (W,)
    col_sums_y = shifted_y.sum(dim=(0, 1))
    # Allow small edge differences from border padding
    diff = (col_sums_orig - col_sums_y).abs().mean().item()
    assert diff < 0.5, f"jy=1.0 should not shift X content significantly; diff={diff}"

    # X-only shift should differ from Y-only shift (catches row/col swap)
    assert not torch.allclose(shifted_x, shifted_y, atol=1e-3), \
        "X-jitter and Y-jitter produced identical output — likely row/col swap"


# ==============================================================================
# 3. area_downsample — matches existing _box_downsample byte-for-byte
# ==============================================================================


def test_area_downsample_matches_box_downsample_regression() -> None:
    """area_downsample is a drop-in replacement for _box_downsample.

    Any deviation would be a regression in the existing dataset outputs.
    """
    hr = _hr()
    ref = GaussianDataset._box_downsample(hr, SCALE)
    got = area_downsample(hr, SCALE)
    assert torch.equal(got, ref), "area_downsample diverges from _box_downsample"


def test_area_downsample_output_shape() -> None:
    hr = _hr()
    lr = area_downsample(hr, SCALE)
    assert lr.shape == (HR_C, LR_H, LR_W)


def test_area_downsample_scale_one_is_identity() -> None:
    hr = _hr()
    out = area_downsample(hr, 1.0)
    assert torch.equal(out, hr)


# ==============================================================================
# 4. taa_blur_approx — reduces high-frequency content
# ==============================================================================


def test_taa_blur_approx_preserves_shape() -> None:
    lr = area_downsample(_hr(), SCALE)
    out = taa_blur_approx(lr)
    assert out.shape == lr.shape


def test_taa_blur_approx_reduces_high_frequency() -> None:
    """Blurring must attenuate high-frequency detail.

    Compare std of a simple high-pass filter (centre pixel minus mean of
    neighbours) before and after blur. The blurred version must have lower
    std on the high-pass residual.
    """
    lr = area_downsample(_hr(), SCALE)
    blurred = taa_blur_approx(lr, sigma=0.5)

    # Simple high-pass: centre - average of 3×3 block.
    def high_pass(t: torch.Tensor) -> torch.Tensor:
        avg = F.avg_pool2d(t.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        return t - avg

    hp_before = high_pass(lr).std()
    hp_after = high_pass(blurred).std()
    assert hp_after < hp_before, (
        f"taa_blur_approx did not reduce HF content: before={hp_before:.6f} after={hp_after:.6f}"
    )


def test_taa_blur_approx_clamps_values() -> None:
    """Output should remain in [0, 1] for inputs already in [0, 1]."""
    lr = area_downsample(_hr(), SCALE)
    out = taa_blur_approx(lr)
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6  # small float tolerance


def test_taa_blur_approx_returns_float32() -> None:
    lr = area_downsample(_hr(), SCALE)
    out = taa_blur_approx(lr)
    assert out.dtype == torch.float32


# ==============================================================================
# 5. EngineAliasedLRSynth end-to-end: HR (3, 64, 96) → LR (3, 32, 48)
# ==============================================================================


def test_engine_synth_output_shape() -> None:
    hr = _hr()
    synth = EngineAliasedLRSynth(scale=SCALE)
    lr = synth.synthesize(hr, frame_idx=0)
    assert lr.shape == (HR_C, LR_H, LR_W), f"unexpected shape {lr.shape}"


def test_engine_synth_output_dtype_float32() -> None:
    hr = _hr()
    synth = EngineAliasedLRSynth(scale=SCALE)
    lr = synth.synthesize(hr, frame_idx=0)
    assert lr.dtype == torch.float32


def test_engine_synth_output_in_unit_range() -> None:
    hr = _hr()
    synth = EngineAliasedLRSynth(scale=SCALE)
    lr = synth.synthesize(hr, frame_idx=0)
    assert lr.min() >= 0.0
    assert lr.max() <= 1.0 + 1e-5


# ==============================================================================
# 6. All toggles OFF → output matches plain area_downsample
# ==============================================================================


def test_engine_synth_all_off_matches_area_downsample() -> None:
    """With jitter, TAA blur, and JPEG all disabled, synthesize must be
    bit-exact with area_downsample (the backward-compat guarantee)."""
    hr = _hr()
    synth = EngineAliasedLRSynth(
        scale=SCALE,
        enable_jitter=False,
        enable_taa_blur=False,
        enable_jpeg=False,
    )
    lr = synth.synthesize(hr, frame_idx=0)
    ref = area_downsample(hr, SCALE)
    assert torch.equal(lr, ref), "all-off synth deviates from area_downsample"


# ==============================================================================
# 7. Jitter ON → output differs from plain area_downsample (bit-exact diff)
# ==============================================================================


def test_engine_synth_jitter_on_differs_from_area_downsample() -> None:
    """When jitter is enabled, the output MUST differ from plain area downsample.

    The jitter shifts the source HR by a sub-pixel offset before downsampling,
    so the pixel values cannot be identical to the non-jittered downsample
    (unless the HR image is perfectly flat, which it never is for real data).
    """
    hr = _hr()
    # Pick a frame index that gives a non-zero jitter.
    frame_idx = 3  # halton(3) should give a nonzero offset in base 2 and 3.
    synth = EngineAliasedLRSynth(
        scale=SCALE,
        enable_jitter=True,
        enable_taa_blur=False,
        enable_jpeg=False,
    )
    lr_jittered = synth.synthesize(hr, frame_idx=frame_idx)
    lr_plain = area_downsample(hr, SCALE)
    assert not torch.equal(lr_jittered, lr_plain), (
        "jitter-enabled synth produced bit-identical output to plain area downsample; "
        "check that the Halton jitter at frame_idx=3 is non-zero"
    )


# ==============================================================================
# 8. JPEG toggle — valid tensor in [0, 1]
# ==============================================================================


def test_jpeg_artifact_output_shape() -> None:
    lr = area_downsample(_hr(), SCALE)
    out = jpeg_artifact(lr, quality=85)
    assert out.shape == lr.shape


def test_jpeg_artifact_values_in_unit_range() -> None:
    lr = area_downsample(_hr(), SCALE)
    out = jpeg_artifact(lr, quality=85)
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-5


def test_jpeg_artifact_returns_float32() -> None:
    lr = area_downsample(_hr(), SCALE)
    out = jpeg_artifact(lr, quality=85)
    assert out.dtype == torch.float32


def test_jpeg_artifact_changes_pixel_values() -> None:
    """JPEG compression should introduce visible quantisation artefacts."""
    lr = area_downsample(_hr(), SCALE)
    out = jpeg_artifact(lr, quality=50)  # lower quality → more artefacts
    # Should differ from the input (JPEG is lossy)
    assert not torch.equal(lr, out)


def test_engine_synth_jpeg_on_valid_output() -> None:
    hr = _hr()
    synth = EngineAliasedLRSynth(
        scale=SCALE,
        enable_jitter=False,
        enable_taa_blur=False,
        enable_jpeg=True,
    )
    lr = synth.synthesize(hr, frame_idx=0)
    assert lr.shape == (HR_C, LR_H, LR_W)
    assert lr.min() >= 0.0
    assert lr.max() <= 1.0 + 1e-5


# ==============================================================================
# 9. Public API re-exported from oss.gaussian.data
# ==============================================================================


def test_public_api_importable_from_data_package() -> None:
    """Ensure EngineAliasedLRSynth and helpers are re-exported from the package."""
    from oss.gaussian.data import (  # noqa: F401
        EngineAliasedLRSynth,
        area_downsample,
        apply_jitter,
        halton_jitter,
        jpeg_artifact,
        taa_blur_approx,
    )
