"""Regression tests for the v6.1 16-pixel grid artifact fixes."""
from __future__ import annotations

import torch

from oss.sr.v6.model import CanvasState, V6Config, V6Model
from oss.sr.v6.rasterizer import V6Rasterizer


def _tiny_v6_1_model(**overrides) -> V6Model:
    cfg = V6Config(
        in_channels=9,
        scale=2,
        backbone="hat-tiny",
        canvas_capacity=64,
        token_dim=32,
        cross_attention_heads=4,
        window_size=16,
        spawn_offset_random=True,
        rasterizer_overlap=8,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return V6Model(cfg)


def _constant_lr(batch: int = 1, h: int = 32, w: int = 32) -> torch.Tensor:
    rgb = torch.full((batch, 3, h, w), 0.5)
    gbuffers = torch.zeros(batch, 6, h, w)
    return torch.cat([rgb, gbuffers], dim=1)


def _zero_motion(batch: int = 1, h: int = 32, w: int = 32) -> torch.Tensor:
    return torch.zeros(batch, 2, h, w)


def _fft_grid_amplitude(region: torch.Tensor, period: int = 16) -> float:
    """Return image-size-normalized magnitude at the 16-pixel FFT bin."""
    gray = region.detach().float().mean(dim=0)
    gray = gray - gray.mean()
    fft = torch.fft.fft2(gray)
    mag = torch.fft.fftshift(fft).abs()
    size = int(gray.shape[-1])
    center = size // 2
    bin_idx = size // period
    probe = torch.stack(
        [
            mag[center, center + bin_idx],
            mag[center, center - bin_idx],
            mag[center + bin_idx, center],
            mag[center - bin_idx, center],
        ]
    ).amax()
    return float(probe / float(size * size))


def test_constant_input_produces_smooth_output_v6_1():
    torch.manual_seed(0)
    model = _tiny_v6_1_model().train()
    lr = _constant_lr()
    motion = _zero_motion()

    with torch.no_grad():
        model.reset_state()
        out = None
        for frame_idx in range(10):
            out = model(
                lr,
                motion_lr=None if frame_idx == 0 else motion,
                frame_index=frame_idx,
            )

    assert out is not None
    inner = out[0, :, 16:48, 16:48]
    assert float(inner.std()) < 5.0e-3
    assert _fft_grid_amplitude(out[0, :, :64, :64]) < 5.0e-4


def test_v6_legacy_path_unchanged():
    torch.manual_seed(123)
    baseline = V6Model(
        V6Config(
            backbone="hat-tiny",
            canvas_capacity=64,
            token_dim=32,
            cross_attention_heads=4,
            spawn_offset_random=False,
            rasterizer_overlap=0,
        )
    ).train(False)
    candidate = V6Model(
        V6Config(
            backbone="hat-tiny",
            canvas_capacity=64,
            token_dim=32,
            cross_attention_heads=4,
        )
    ).train(False)
    candidate.load_state_dict(baseline.state_dict())
    lr = torch.randn(1, 9, 32, 32)

    with torch.no_grad():
        torch.manual_seed(999)
        out_baseline = baseline(lr, motion_lr=None, frame_index=0)
        torch.manual_seed(999)
        out_candidate = candidate(lr, motion_lr=None, frame_index=0)

    torch.testing.assert_close(out_candidate, out_baseline, atol=0.0, rtol=0.0)
    assert candidate._spawn_offset_xy is None


def test_spawner_offset_constant_within_trajectory():
    torch.manual_seed(2026)
    model = _tiny_v6_1_model(canvas_capacity=32, tile_size_lr=16).train()
    lr = _constant_lr(h=16, w=16)
    motion = _zero_motion(h=16, w=16)

    offsets = []
    with torch.no_grad():
        model.reset_state()
        for frame_idx in range(4):
            model(
                lr,
                motion_lr=None if frame_idx == 0 else motion,
                frame_index=frame_idx,
            )
            assert model._spawn_offset_xy is not None
            offsets.append(model._spawn_offset_xy.detach().clone())

        model.reset_state()
        model(lr, motion_lr=None, frame_index=0)
        assert model._spawn_offset_xy is not None
        resampled = model._spawn_offset_xy.detach().clone()

    for offset in offsets[1:]:
        torch.testing.assert_close(offset, offsets[0], atol=0.0, rtol=0.0)
    assert not torch.equal(resampled, offsets[0])


class _SyntheticSeamRenderer:
    """Renderer double that exposes full-frame 16px seams only on legacy path."""

    def __call__(self, gaussians, output_hw):
        h, w = output_hw
        feat_dim = int(gaussians.feat.shape[-1])
        out = torch.ones(feat_dim, h, w, device=gaussians.feat.device, dtype=gaussians.feat.dtype)
        if (h, w) == (64, 64):
            out[:, 15::16, :] *= 0.25
            out[:, :, 15::16] *= 0.25
        else:
            out[:, 0, :] *= 0.25
            out[:, -1, :] *= 0.25
            out[:, :, 0] *= 0.25
            out[:, :, -1] *= 0.25
        return out


def _regular_canvas(token_dim: int = 1) -> CanvasState:
    y = torch.arange(8.0, 64.0, 16.0)
    x = torch.arange(8.0, 64.0, 16.0)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    positions = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    count = int(positions.shape[0])
    return CanvasState(
        positions=positions,
        scales=torch.full((count, 2), 6.0),
        rotations=torch.zeros(count),
        opacities=torch.ones(count),
        colors=torch.ones(count, token_dim),
        count=count,
    )


def _seam_contrast(x: torch.Tensor) -> torch.Tensor:
    img = x[0, 0]
    diffs = []
    for idx in (16, 32, 48):
        diffs.append((img[:, idx] - img[:, idx - 1]).abs().mean())
        diffs.append((img[idx, :] - img[idx - 1, :]).abs().mean())
    return torch.stack(diffs).mean()


def test_rasterizer_seam_smoothness():
    canvas = _regular_canvas()
    active = torch.ones(canvas.count, dtype=torch.bool)

    legacy = V6Rasterizer(token_dim=1, overlap=0)
    legacy.renderer = _SyntheticSeamRenderer()
    blended = V6Rasterizer(token_dim=1, overlap=8)
    blended.renderer = _SyntheticSeamRenderer()

    out_legacy = legacy(canvas, active, output_hw=(64, 64))
    out_blended = blended(canvas, active, output_hw=(64, 64))
    contrast_legacy = _seam_contrast(out_legacy)
    contrast_blended = _seam_contrast(out_blended)

    assert float(contrast_legacy) > 0.1
    assert float(contrast_blended) < 0.1 * float(contrast_legacy)
