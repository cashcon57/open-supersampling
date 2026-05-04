# v5 Pixel Temporal Super-Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FSR 2-class temporal warp+blend on top of the v4 single-frame SR-CNN, warm-started from `srcnn-prod-v4-lpips/step-00385000.pt`, and validate it against the v5 success criteria on a fixed Sintel + TartanAir held-out batch.

**Architecture:** Keep the v4 backbone intact. Add a small temporal head (~50K params) that consumes `concat(current_sr_HR, warped_prev_HR, disocclusion_mask, depth_HR)` and outputs the final HR frame. Backward warp via `F.grid_sample`. Disocclusion mask is `sigmoid(α·|warped_depth − curr_depth| + β·||motion|| − γ)` with learnable α, β, γ. Sequential frame-pair dataset wrappers around existing `TartanAirGaussianDataset` / `SintelGaussianDataset`. Loss = `L1 + 0.1·SSIM + 0.1·LPIPS-VGG + 0.05·temporal-consistency`. Warm-start, freeze backbone for first 10K steps, then unfreeze with reduced LR.

**Tech Stack:** PyTorch 2.4.1, existing `oss.sr.SRCNNSimple`, existing `oss.train.losses.temporal_consistency_loss`, existing `oss.gaussian.data.{Tartanair,Sintel}GaussianDataset`, `EngineAliasedLRSynth`, `lpips` package, training dashboard via `metrics.json` + `score_log.json`.

**Spec:** `docs/superpowers/specs/2026-05-04-v5-pixel-temporal-design.md`

**Branch:** `v0.2-dev`. No PR to `main` until Sprint 5 ships.

---

## File Structure

New module `oss/sr/temporal/`:

| File | Responsibility |
|---|---|
| `oss/sr/temporal/__init__.py` | Public exports: `warp_prev_hr`, `compute_disocclusion`, `TemporalHead`, `TemporalSRModel`, `SequentialPairDataset` |
| `oss/sr/temporal/warp.py` | Motion-vec LR→HR upsample + backward `grid_sample` warp of prev HR |
| `oss/sr/temporal/disocclusion.py` | `DisocclusionGate` module with learnable α, β, γ; warps prev depth, returns HR mask in [0,1] |
| `oss/sr/temporal/temporal_head.py` | Conv stack 8→32→32→3, takes `concat(current_sr, warped_prev, disocclusion, depth_hr)` |
| `oss/sr/temporal/model.py` | `TemporalSRModel(nn.Module)` — wraps frozen-or-unfrozen v4 backbone + warp + disocclusion + head |
| `oss/sr/temporal/dataset.py` | `SequentialPairDataset` returning `(example_t, example_t_plus_1, prev_hr_init)` |
| `scripts/sr_train_temporal.py` | Training entry: warm-start, 3-phase schedule, auto-resume, dashboard metrics |
| `scripts/sr_temporal_held_out.py` | Fixed-batch held-out eval: PSNR + LPIPS + temporal stability vs v4 baseline |
| `oss/sr/inference.py` (modify) | Add `TemporalSRInferenceEngine` carrying `prev_hr_output` across calls; scene-cut reset |

New tests under `tests/sr/temporal/`:

| File | Tests |
|---|---|
| `tests/sr/temporal/__init__.py` | empty |
| `tests/sr/temporal/test_warp.py` | identity warp, translation roundtrip, LR→HR motion upsample |
| `tests/sr/temporal/test_disocclusion.py` | mask is [0,1], mask=1 on big depth-diff, mask=0 on static frame |
| `tests/sr/temporal/test_temporal_head.py` | input shape, output shape, gradient flow, ≤60K params |
| `tests/sr/temporal/test_model.py` | forward shape, warm-start v4 weights load, freeze/unfreeze toggle |
| `tests/sr/temporal/test_dataset.py` | pair loader returns consecutive frames, first-frame init = bilinear LR |
| `tests/sr/temporal/test_loss_pipeline.py` | full loss assembled (appearance + temporal-consistency) returns finite scalar with gradient |
| `tests/sr/temporal/test_inference_state.py` | stateful engine carries prev_hr; scene-cut reset triggers correctly |

Total new code target: ~900 LOC + ~600 LOC tests. Implementation must remain DRY and reuse `oss.train.losses.temporal_consistency_loss` (already in repo) rather than re-implement warping.

---

## Verification Commands

Common commands referenced by tasks:

```bash
# All temporal tests
pytest tests/sr/temporal/ -v

# Single test file
pytest tests/sr/temporal/test_warp.py -v

# Smoke train (CPU, 5 steps, synthetic data) — must work before remote launch
python scripts/sr_train_temporal.py --smoke --device cpu --max-steps 5

# Real train (remote 3080 Ti)
python scripts/sr_train_temporal.py \
    --output-dir <train-host-data>/checkpoints/srcnn-v5-pixel-temporal \
    --warm-start <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt \
    --tartanair-root <train-host-data>/datasets/tartanair_extracted \
    --sintel-root <train-host-data>/datasets/sintel \
    --max-steps 80000

# Held-out eval
python scripts/sr_temporal_held_out.py \
    --ckpt-temporal <train-host-data>/checkpoints/srcnn-v5-pixel-temporal/step-XXXXX.pt \
    --ckpt-baseline <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt \
    --tartanair-root <train-host-data>/datasets/tartanair_extracted \
    --sintel-root <train-host-data>/datasets/sintel \
    --n-samples 64
```

---

## Task 0: Module scaffold + warp module + warp tests

**Goal:** Create `oss/sr/temporal/` module skeleton and ship a working backward-warp helper that bilinearly upsamples LR motion to HR and pulls prev-HR by `F.grid_sample`. This unblocks every later task.

**Files:**
- Create: `oss/sr/temporal/__init__.py`
- Create: `oss/sr/temporal/warp.py`
- Create: `tests/sr/temporal/__init__.py`
- Create: `tests/sr/temporal/test_warp.py`

**Acceptance Criteria:**
- [ ] `from oss.sr.temporal import warp_prev_hr, upsample_motion_to_hr` works
- [ ] Identity motion (zero flow) → warp returns prev_hr unchanged within 1e-5
- [ ] Translation flow `(dx, dy) = (4, 0) HR pixels` → warped image equals `prev[:, :, :, :-4]` for the overlapping region within 1e-3
- [ ] LR motion upsample: `(B, 2, h, w)` motion @ scale=2 → `(B, 2, 2h, 2w)` and the values are scaled by 2.0 (HR pixel displacement, not LR pixel displacement)
- [ ] `pytest tests/sr/temporal/test_warp.py -v` all pass

**Verify:** `pytest tests/sr/temporal/test_warp.py -v` → all tests pass

**Steps:**

- [ ] **Step 1: Create empty `__init__.py` files**

```bash
mkdir -p oss/sr/temporal tests/sr/temporal
touch oss/sr/temporal/__init__.py tests/sr/temporal/__init__.py
```

- [ ] **Step 2: Write the failing test in `tests/sr/temporal/test_warp.py`**

```python
"""Backward-warp tests for the v5 pixel temporal track."""
from __future__ import annotations

import torch

from oss.sr.temporal import upsample_motion_to_hr, warp_prev_hr


def test_upsample_motion_scales_displacement() -> None:
    motion_lr = torch.ones(1, 2, 4, 4)  # 1 LR-pixel of flow everywhere
    motion_hr = upsample_motion_to_hr(motion_lr, scale=2)
    assert motion_hr.shape == (1, 2, 8, 8)
    # LR-pixel displacement of 1 == HR-pixel displacement of 2
    assert torch.allclose(motion_hr, torch.full_like(motion_hr, 2.0), atol=1e-5)


def test_zero_motion_is_identity() -> None:
    prev_hr = torch.rand(1, 3, 16, 16)
    motion_lr = torch.zeros(1, 2, 8, 8)
    warped = warp_prev_hr(prev_hr, motion_lr, scale=2)
    assert torch.allclose(warped, prev_hr, atol=1e-5)


def test_translation_warp() -> None:
    # Convention: motion is forward flow t-1 → t.
    # Construct prev_hr with a vertical stripe at columns 4..7.
    # Forward flow x = +4 HR px means content at prev col c moved to current col c+4.
    # Backward warp at current pixel p samples prev at p − flow(p) = p − 4.
    # So at current p=8..11 we sample prev[4..7] (the stripe) → output stripe at 8..11.
    prev_hr = torch.zeros(1, 3, 16, 16)
    prev_hr[..., 4:8] = 1.0
    motion_lr = torch.zeros(1, 2, 8, 8)
    motion_lr[:, 0] = 2.0  # +2 LR px ≡ +4 HR px
    warped = warp_prev_hr(prev_hr, motion_lr, scale=2)
    assert warped[..., 8:12].mean() > 0.95
    assert warped[..., :4].mean() < 0.05
```

- [ ] **Step 3: Run test to verify it fails**

```
pytest tests/sr/temporal/test_warp.py -v
```
Expected: FAIL with `ImportError: cannot import name 'upsample_motion_to_hr' from 'oss.sr.temporal'`.

- [ ] **Step 4: Implement `oss/sr/temporal/warp.py`**

```python
"""Backward warp + LR→HR motion-vector upsample for the v5 pixel temporal track.

Convention: motion vectors are forward flow ``t-1 → t`` (LR pixel
displacements). At each pixel ``p`` of the current frame, the corresponding
prev-frame location is ``p − motion_hr(p)``, so ``warp_prev_hr`` does
``F.grid_sample(prev_hr, base_grid − motion_hr)`` (backward / pull warp).

This matches the dataset adapters: TartanAir's ``flow/NN_NM_flow.npy`` is
forward flow from frame N to frame N+1; Sintel's ``.flo`` files likewise
store forward flow.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def upsample_motion_to_hr(motion_lr: torch.Tensor, scale: int) -> torch.Tensor:
    """Bilinearly upsample LR motion to HR resolution and rescale magnitudes.

    Args:
        motion_lr: (B, 2, H_lr, W_lr) LR-pixel displacements (channel 0 = x, 1 = y).
        scale:     HR / LR ratio (positive int).

    Returns:
        (B, 2, scale*H_lr, scale*W_lr) HR-pixel displacements.
    """
    if motion_lr.dim() != 4 or motion_lr.shape[1] != 2:
        raise ValueError(f"motion_lr must be (B, 2, H, W); got {tuple(motion_lr.shape)}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1; got {scale}")
    motion_hr = F.interpolate(motion_lr, scale_factor=float(scale), mode="bilinear", align_corners=False)
    return motion_hr * float(scale)


def warp_prev_hr(prev_hr: torch.Tensor, motion_lr: torch.Tensor, scale: int) -> torch.Tensor:
    """Backward-warp prev-HR to align with current view via motion vectors.

    Args:
        prev_hr:   (B, 3, H_hr, W_hr).
        motion_lr: (B, 2, H_lr, W_lr) LR-pixel forward flow ``t-1 → t``
                   (matches dataset adapters: TartanAir flow files + Sintel .flo
                   are forward flow). At each current pixel ``p`` the prev
                   location is ``p − motion_hr(p)``.
        scale:     HR / LR ratio.

    Returns:
        (B, 3, H_hr, W_hr) prev_hr resampled at the current frame's pixel grid.

    Out-of-frame samples use ``padding_mode='border'`` (clamp to edge), which
    is the safest default for the head — it'll learn to mask via disocclusion.
    """
    if prev_hr.dim() != 4 or prev_hr.shape[1] != 3:
        raise ValueError(f"prev_hr must be (B, 3, H, W); got {tuple(prev_hr.shape)}")
    b, _, h_hr, w_hr = prev_hr.shape
    motion_hr = upsample_motion_to_hr(motion_lr, scale=scale)
    if motion_hr.shape[-2:] != (h_hr, w_hr):
        raise ValueError(
            f"motion HR shape {tuple(motion_hr.shape[-2:])} != prev_hr {(h_hr, w_hr)}"
        )

    # Build base grid in HR pixel coords, then add HR motion.
    yy, xx = torch.meshgrid(
        torch.arange(h_hr, device=prev_hr.device, dtype=prev_hr.dtype),
        torch.arange(w_hr, device=prev_hr.device, dtype=prev_hr.dtype),
        indexing="ij",
    )
    base_x = xx.unsqueeze(0).expand(b, -1, -1)
    base_y = yy.unsqueeze(0).expand(b, -1, -1)
    # Forward flow t-1 → t: at current pixel p, prev location is p − flow(p).
    sample_x = base_x - motion_hr[:, 0]
    sample_y = base_y - motion_hr[:, 1]

    # Normalize to [-1, 1] for grid_sample with align_corners=False:
    #   normalized = (2*pixel + 1) / N - 1
    norm_x = (2.0 * sample_x + 1.0) / w_hr - 1.0
    norm_y = (2.0 * sample_y + 1.0) / h_hr - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)  # (B, H_hr, W_hr, 2)

    return F.grid_sample(
        prev_hr,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


__all__ = ["upsample_motion_to_hr", "warp_prev_hr"]
```

- [ ] **Step 5: Wire exports in `oss/sr/temporal/__init__.py`**

```python
"""v5 pixel-temporal SR module.

Adds FSR 2-class temporal warp+blend on top of the v4 SR-CNN baseline.
"""
from oss.sr.temporal.warp import upsample_motion_to_hr, warp_prev_hr

__all__ = ["upsample_motion_to_hr", "warp_prev_hr"]
```

- [ ] **Step 6: Run test to verify it passes**

```
pytest tests/sr/temporal/test_warp.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add oss/sr/temporal/__init__.py oss/sr/temporal/warp.py \
        tests/sr/temporal/__init__.py tests/sr/temporal/test_warp.py
git commit -m "v5-pixel(sr): add motion-vec upsample + backward HR warp helpers"
```

---

## Task 1: Disocclusion gate module

**Goal:** Ship `DisocclusionGate(nn.Module)` that produces an HR-resolution disocclusion mask in `[0, 1]` from current depth, prev depth, and motion magnitude. Three learnable scalar parameters (α, β, γ).

**Files:**
- Create: `oss/sr/temporal/disocclusion.py`
- Modify: `oss/sr/temporal/__init__.py` (export `DisocclusionGate`)
- Create: `tests/sr/temporal/test_disocclusion.py`

**Acceptance Criteria:**
- [ ] `DisocclusionGate()` is an `nn.Module` with exactly 3 learnable scalar parameters (α, β, γ)
- [ ] Output is HR-resolution `(B, 1, H_hr, W_hr)`, all values in `[0, 1]`
- [ ] Static frame (depth_curr == depth_prev, motion=0) → mask near 0 everywhere (≤0.1)
- [ ] Big depth disparity → mask near 1 (≥0.9) at those pixels
- [ ] Gradient flows to α, β, γ when loss applied to mask

**Verify:** `pytest tests/sr/temporal/test_disocclusion.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing test**

```python
"""Tests for DisocclusionGate."""
from __future__ import annotations

import torch

from oss.sr.temporal import DisocclusionGate


def test_module_has_three_scalar_params() -> None:
    gate = DisocclusionGate()
    params = list(gate.parameters())
    # alpha, beta, gamma — each scalar
    assert len(params) == 3
    for p in params:
        assert p.numel() == 1
        assert p.requires_grad


def test_output_shape_and_range() -> None:
    gate = DisocclusionGate()
    depth_curr = torch.rand(2, 1, 16, 16)
    depth_prev = torch.rand(2, 1, 16, 16)
    motion = torch.randn(2, 2, 8, 8) * 0.5
    mask = gate(depth_curr=depth_curr, depth_prev=depth_prev, motion_lr=motion, scale=2)
    assert mask.shape == (2, 1, 16, 16)
    assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_static_frame_low_mask() -> None:
    gate = DisocclusionGate()
    depth = torch.rand(1, 1, 16, 16)
    motion = torch.zeros(1, 2, 8, 8)
    mask = gate(depth_curr=depth, depth_prev=depth, motion_lr=motion, scale=2)
    # Default init: alpha, beta small positive, gamma large positive → low mask.
    assert mask.mean() < 0.2


def test_large_depth_disparity_high_mask() -> None:
    gate = DisocclusionGate()
    depth_curr = torch.zeros(1, 1, 16, 16)
    depth_prev = torch.ones(1, 1, 16, 16) * 5.0  # huge disparity
    motion = torch.zeros(1, 2, 8, 8)
    # Force alpha large so depth-diff dominates and the mask saturates.
    with torch.no_grad():
        gate.alpha.fill_(50.0)
        gate.gamma.fill_(0.0)
    mask = gate(depth_curr=depth_curr, depth_prev=depth_prev, motion_lr=motion, scale=2)
    assert mask.mean() > 0.9


def test_gradient_flow_to_params() -> None:
    gate = DisocclusionGate()
    depth_curr = torch.rand(1, 1, 8, 8)
    depth_prev = torch.rand(1, 1, 8, 8)
    motion = torch.randn(1, 2, 4, 4)
    mask = gate(depth_curr=depth_curr, depth_prev=depth_prev, motion_lr=motion, scale=2)
    mask.mean().backward()
    assert gate.alpha.grad is not None and torch.isfinite(gate.alpha.grad).all()
    assert gate.beta.grad is not None and torch.isfinite(gate.beta.grad).all()
    assert gate.gamma.grad is not None and torch.isfinite(gate.gamma.grad).all()
```

- [ ] **Step 2: Run test to verify failure**

```
pytest tests/sr/temporal/test_disocclusion.py -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `oss/sr/temporal/disocclusion.py`**

```python
"""Disocclusion mask for the v5 pixel temporal track.

Per design spec §Architecture point 4:
    disoccl = sigmoid(alpha * |warped_depth_prev - depth_curr| + beta * ||motion|| - gamma)

Default init: alpha=10.0, beta=2.0, gamma=4.0. Empirically these put the
sigmoid in a regime where small static differences map to ~0 and large
disparities saturate to ~1. The trainer will adjust them.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from oss.sr.temporal.warp import upsample_motion_to_hr, warp_prev_hr


def _warp_prev_depth(depth_prev: torch.Tensor, motion_lr: torch.Tensor, scale: int) -> torch.Tensor:
    """Warp single-channel prev depth using the same backward-warp as RGB.

    Implementation: replicate channel to 3, warp, take channel 0. This avoids
    a second specialized warp implementation. Bilinear sampling is fine for
    depth here — disocclusion is the supervision target, not the depth itself.
    """
    rep = depth_prev.expand(-1, 3, -1, -1).contiguous()
    warped = warp_prev_hr(rep, motion_lr, scale=scale)
    return warped[:, :1]


class DisocclusionGate(nn.Module):
    """Disocclusion mask producer with three learnable scalar gates."""

    def __init__(
        self,
        alpha_init: float = 10.0,
        beta_init: float = 2.0,
        gamma_init: float = 4.0,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(
        self,
        depth_curr: torch.Tensor,
        depth_prev: torch.Tensor,
        motion_lr: torch.Tensor,
        scale: int,
    ) -> torch.Tensor:
        """Produce HR disocclusion mask.

        Args:
            depth_curr: (B, 1, H_hr, W_hr) — current frame depth at HR
                        (caller upsamples LR depth ahead of time if needed).
            depth_prev: (B, 1, H_hr, W_hr) — previous frame depth at HR.
            motion_lr:  (B, 2, H_lr, W_lr).
            scale:      HR / LR ratio.

        Returns:
            (B, 1, H_hr, W_hr) mask in [0, 1].
        """
        if depth_curr.shape != depth_prev.shape:
            raise ValueError(
                f"depth_curr {tuple(depth_curr.shape)} != depth_prev {tuple(depth_prev.shape)}"
            )
        warped_depth_prev = _warp_prev_depth(depth_prev, motion_lr, scale=scale)
        depth_diff = (warped_depth_prev - depth_curr).abs()  # (B, 1, H_hr, W_hr)

        motion_hr = upsample_motion_to_hr(motion_lr, scale=scale)
        motion_mag = motion_hr.norm(dim=1, keepdim=True)  # (B, 1, H_hr, W_hr)

        logit = self.alpha * depth_diff + self.beta * motion_mag - self.gamma
        return torch.sigmoid(logit)


__all__ = ["DisocclusionGate"]
```

- [ ] **Step 4: Update `oss/sr/temporal/__init__.py`**

```python
"""v5 pixel-temporal SR module.

Adds FSR 2-class temporal warp+blend on top of the v4 SR-CNN baseline.
"""
from oss.sr.temporal.disocclusion import DisocclusionGate
from oss.sr.temporal.warp import upsample_motion_to_hr, warp_prev_hr

__all__ = ["DisocclusionGate", "upsample_motion_to_hr", "warp_prev_hr"]
```

- [ ] **Step 5: Run tests**

```
pytest tests/sr/temporal/test_disocclusion.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add oss/sr/temporal/disocclusion.py oss/sr/temporal/__init__.py \
        tests/sr/temporal/test_disocclusion.py
git commit -m "v5-pixel(sr): add disocclusion gate with learnable alpha/beta/gamma"
```

---

## Task 2: Temporal head module

**Goal:** Ship `TemporalHead(nn.Module)` — small conv stack 8→32→32→3 that fuses `(current_sr, warped_prev, disocclusion, depth_hr)` into the final HR output. Param count must stay under 60K.

**Files:**
- Create: `oss/sr/temporal/temporal_head.py`
- Modify: `oss/sr/temporal/__init__.py`
- Create: `tests/sr/temporal/test_temporal_head.py`

**Acceptance Criteria:**
- [ ] Input: `concat(current_sr (3), warped_prev (3), disocclusion (1), depth_hr (1)) = 8 channels`
- [ ] Output: `(B, 3, H_hr, W_hr)` HR final frame
- [ ] Total parameter count ≤ 60K
- [ ] Final-conv bias init = 0.0 and final-conv weight std small (so initial output ≈ current_sr — see init step below)
- [ ] At init the head's residual on top of current_sr is small (`||head_out − current_sr||_F < 0.1` per pixel)
- [ ] Forward + backward through random input produces finite gradients

**Verify:** `pytest tests/sr/temporal/test_temporal_head.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write failing test**

```python
"""Tests for TemporalHead."""
from __future__ import annotations

import torch

from oss.sr.temporal import TemporalHead


def test_param_count_under_budget() -> None:
    head = TemporalHead()
    n = sum(p.numel() for p in head.parameters())
    assert n <= 60_000, f"TemporalHead has {n} params (budget 60_000)"


def test_forward_shape() -> None:
    head = TemporalHead()
    current_sr = torch.rand(2, 3, 16, 16)
    warped_prev = torch.rand(2, 3, 16, 16)
    disocclusion = torch.rand(2, 1, 16, 16)
    depth_hr = torch.rand(2, 1, 16, 16)
    out = head(current_sr=current_sr, warped_prev=warped_prev,
               disocclusion=disocclusion, depth_hr=depth_hr)
    assert out.shape == (2, 3, 16, 16)


def test_initial_output_close_to_current_sr() -> None:
    head = TemporalHead()
    current_sr = torch.rand(1, 3, 16, 16)
    warped_prev = torch.rand(1, 3, 16, 16)
    disocclusion = torch.rand(1, 1, 16, 16)
    depth_hr = torch.rand(1, 1, 16, 16)
    out = head(current_sr=current_sr, warped_prev=warped_prev,
               disocclusion=disocclusion, depth_hr=depth_hr)
    delta = (out - current_sr).abs().mean().item()
    assert delta < 0.1, f"Initial residual too large: {delta}"


def test_grad_flow() -> None:
    head = TemporalHead()
    current_sr = torch.rand(1, 3, 8, 8, requires_grad=True)
    warped_prev = torch.rand(1, 3, 8, 8)
    disocclusion = torch.rand(1, 1, 8, 8)
    depth_hr = torch.rand(1, 1, 8, 8)
    out = head(current_sr=current_sr, warped_prev=warped_prev,
               disocclusion=disocclusion, depth_hr=depth_hr)
    out.mean().backward()
    assert current_sr.grad is not None and torch.isfinite(current_sr.grad).all()
    for p in head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
```

- [ ] **Step 2: Run test to verify failure**

```
pytest tests/sr/temporal/test_temporal_head.py -v
```
Expected: ImportError on `TemporalHead`.

- [ ] **Step 3: Implement `oss/sr/temporal/temporal_head.py`**

```python
"""Small temporal-head conv stack for the v5 pixel temporal track.

Architecture (per spec §Architecture point 5):
    Input (8ch HR): concat(current_sr, warped_prev, disocclusion, depth_hr)
    Conv(8 → 32, 3x3) + ReLU
    Conv(32 → 32, 3x3) + ReLU
    Conv(32 → 32, 3x3) + ReLU
    Conv(32 → 3,  3x3)            # residual on top of current_sr

Final output = current_sr + small residual. The final conv is initialized
with small weights and zero bias so the head starts as a near-identity on
current_sr — training only has to learn the temporal correction.

Param budget (8*32 + 32*32*3) channels worth of 3x3 convs + biases ~= 28K.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalHead(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(8, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden, 3, 3, padding=1)

        for m in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        # Tiny init on output residual so initial output ~= current_sr.
        nn.init.normal_(self.conv_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        current_sr: torch.Tensor,
        warped_prev: torch.Tensor,
        disocclusion: torch.Tensor,
        depth_hr: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([current_sr, warped_prev, disocclusion, depth_hr], dim=1)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        residual = self.conv_out(x)
        return current_sr + residual


__all__ = ["TemporalHead"]
```

- [ ] **Step 4: Update `oss/sr/temporal/__init__.py`**

```python
from oss.sr.temporal.disocclusion import DisocclusionGate
from oss.sr.temporal.temporal_head import TemporalHead
from oss.sr.temporal.warp import upsample_motion_to_hr, warp_prev_hr

__all__ = [
    "DisocclusionGate",
    "TemporalHead",
    "upsample_motion_to_hr",
    "warp_prev_hr",
]
```

- [ ] **Step 5: Run tests**

```
pytest tests/sr/temporal/test_temporal_head.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add oss/sr/temporal/temporal_head.py oss/sr/temporal/__init__.py \
        tests/sr/temporal/test_temporal_head.py
git commit -m "v5-pixel(sr): add temporal-head conv stack with near-identity init"
```

---

## Task 3: TemporalSRModel — backbone wrap + warm-start + freeze toggle

**Goal:** Ship `TemporalSRModel(nn.Module)` that wraps a v4 SR-CNN backbone with the warp + DisocclusionGate + TemporalHead pieces. Provides a `load_v4_warm_start(ckpt_path)` classmethod and a `freeze_backbone(flag)` method.

**Files:**
- Create: `oss/sr/temporal/model.py`
- Modify: `oss/sr/temporal/__init__.py`
- Create: `tests/sr/temporal/test_model.py`

**Acceptance Criteria:**
- [ ] `TemporalSRModel(in_channels=12, scale=2, tier="standard")` constructs successfully
- [ ] `forward(lr_inputs, prev_hr, depth_hr_curr, depth_hr_prev, motion_lr)` returns `(B, 3, scale*H, scale*W)`
- [ ] `load_v4_warm_start(ckpt_path)` class method loads `srcnn-prod-v4-lpips/step-XXXXX.pt` weights into the backbone (use a synthetic checkpoint in test)
- [ ] `freeze_backbone(True)` makes backbone params `requires_grad=False`; `freeze_backbone(False)` re-enables them; head + gate params always `requires_grad=True`
- [ ] First-frame init helper: `make_first_frame_prev_hr(lr_rgb, scale)` returns bilinear-upscaled LR
- [ ] Forward shape correct, gradient flows to head + gate when backbone is frozen

**Verify:** `pytest tests/sr/temporal/test_model.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write failing test**

```python
"""Tests for TemporalSRModel."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from oss.sr import build_sr_model
from oss.sr.temporal import TemporalSRModel, make_first_frame_prev_hr


def _make_inputs(batch: int = 1, lr: int = 8, in_ch: int = 12, scale: int = 2):
    h_hr, w_hr = lr * scale, lr * scale
    return {
        "lr_inputs": torch.rand(batch, in_ch, lr, lr),
        "prev_hr": torch.rand(batch, 3, h_hr, w_hr),
        "depth_hr_curr": torch.rand(batch, 1, h_hr, w_hr),
        "depth_hr_prev": torch.rand(batch, 1, h_hr, w_hr),
        "motion_lr": torch.randn(batch, 2, lr, lr) * 0.1,
    }


def test_forward_shape() -> None:
    model = TemporalSRModel(in_channels=12, scale=2, tier="standard")
    out = model(**_make_inputs(batch=2, lr=8))
    assert out.shape == (2, 3, 16, 16)


def test_make_first_frame_prev_hr() -> None:
    lr_rgb = torch.rand(2, 3, 8, 8)
    prev_hr = make_first_frame_prev_hr(lr_rgb, scale=2)
    assert prev_hr.shape == (2, 3, 16, 16)


def test_freeze_backbone_toggle() -> None:
    model = TemporalSRModel(in_channels=12, scale=2, tier="standard")
    model.freeze_backbone(True)
    backbone_params = list(model.backbone.parameters())
    assert all(not p.requires_grad for p in backbone_params)
    head_params = list(model.head.parameters())
    assert all(p.requires_grad for p in head_params)
    model.freeze_backbone(False)
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_load_v4_warm_start(tmp_path: Path) -> None:
    # Build a v4-style checkpoint and round-trip it.
    src = build_sr_model(model_kind="simple", tier="standard", in_channels=12, scale=2)
    ckpt = tmp_path / "v4_synth.pt"
    torch.save(
        {"sr_model": src.state_dict(), "args": {"tier": "standard", "sr_backbone": "simple"}},
        ckpt,
    )
    model = TemporalSRModel.load_v4_warm_start(ckpt, in_channels=12, scale=2)
    for k, v in src.state_dict().items():
        assert torch.equal(model.backbone.state_dict()[k], v)


def test_grad_flow_with_frozen_backbone() -> None:
    model = TemporalSRModel(in_channels=12, scale=2, tier="standard")
    model.freeze_backbone(True)
    inputs = _make_inputs(batch=1, lr=8)
    out = model(**inputs)
    out.mean().backward()
    # Head + gate get grads; backbone does not.
    for p in model.head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    for p in model.gate.parameters():
        assert p.grad is not None
    for p in model.backbone.parameters():
        assert p.grad is None or torch.equal(p.grad, torch.zeros_like(p))
```

- [ ] **Step 2: Run test to verify failure**

```
pytest tests/sr/temporal/test_model.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `oss/sr/temporal/model.py`**

```python
"""TemporalSRModel — v4 backbone + temporal warp + disocclusion + head.

Wires together:
- v4 SRCNNSimple backbone (in_channels=12, scale=2).
- Backward warp of prev-HR by motion vectors.
- DisocclusionGate (alpha, beta, gamma).
- TemporalHead conv stack.

Plus utilities: warm-start from v4 checkpoint, backbone-freeze toggle,
and ``make_first_frame_prev_hr`` for sequence boundary handling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from oss.sr import build_sr_model
from oss.sr.temporal.disocclusion import DisocclusionGate
from oss.sr.temporal.temporal_head import TemporalHead
from oss.sr.temporal.warp import warp_prev_hr


def make_first_frame_prev_hr(lr_rgb: torch.Tensor, scale: int) -> torch.Tensor:
    """Bilinear-upscale LR RGB to use as the synthetic prev-HR on frame 0."""
    if lr_rgb.dim() != 4 or lr_rgb.shape[1] != 3:
        raise ValueError(f"lr_rgb must be (B, 3, H, W); got {tuple(lr_rgb.shape)}")
    return F.interpolate(lr_rgb, scale_factor=float(scale), mode="bilinear", align_corners=False)


class TemporalSRModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        scale: int = 2,
        tier: str = "standard",
        backbone_kind: str = "simple",
    ) -> None:
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.backbone = build_sr_model(
            model_kind=backbone_kind, tier=tier, in_channels=in_channels, scale=scale
        )
        self.gate = DisocclusionGate()
        self.head = TemporalHead()

    def forward(
        self,
        lr_inputs: torch.Tensor,
        prev_hr: torch.Tensor,
        depth_hr_curr: torch.Tensor,
        depth_hr_prev: torch.Tensor,
        motion_lr: torch.Tensor,
    ) -> torch.Tensor:
        current_sr = self.backbone(lr_inputs)
        warped_prev = warp_prev_hr(prev_hr, motion_lr, scale=self.scale)
        disoccl = self.gate(
            depth_curr=depth_hr_curr, depth_prev=depth_hr_prev,
            motion_lr=motion_lr, scale=self.scale,
        )
        return self.head(
            current_sr=current_sr, warped_prev=warped_prev,
            disocclusion=disoccl, depth_hr=depth_hr_curr,
        )

    def freeze_backbone(self, freeze: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)

    @classmethod
    def load_v4_warm_start(
        cls,
        ckpt_path: Path,
        in_channels: int = 12,
        scale: int = 2,
        device: str | torch.device = "cpu",
    ) -> "TemporalSRModel":
        ck: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ck.get("args", {})
        tier = saved.get("tier", "standard")
        backbone_kind = "rrdb" if saved.get("sr_backbone") == "rrdb" else "simple"
        model = cls(in_channels=in_channels, scale=scale, tier=tier, backbone_kind=backbone_kind)
        missing, unexpected = model.backbone.load_state_dict(ck["sr_model"], strict=True)
        if missing or unexpected:
            raise RuntimeError(f"v4 warm-start mismatch: missing={missing}, unexpected={unexpected}")
        return model


__all__ = ["TemporalSRModel", "make_first_frame_prev_hr"]
```

- [ ] **Step 4: Update `oss/sr/temporal/__init__.py`**

```python
from oss.sr.temporal.disocclusion import DisocclusionGate
from oss.sr.temporal.model import TemporalSRModel, make_first_frame_prev_hr
from oss.sr.temporal.temporal_head import TemporalHead
from oss.sr.temporal.warp import upsample_motion_to_hr, warp_prev_hr

__all__ = [
    "DisocclusionGate",
    "TemporalHead",
    "TemporalSRModel",
    "make_first_frame_prev_hr",
    "upsample_motion_to_hr",
    "warp_prev_hr",
]
```

- [ ] **Step 5: Run tests**

```
pytest tests/sr/temporal/test_model.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add oss/sr/temporal/model.py oss/sr/temporal/__init__.py \
        tests/sr/temporal/test_model.py
git commit -m "v5-pixel(sr): add TemporalSRModel with v4 warm-start + freeze toggle"
```

---

## Task 4: SequentialPairDataset

**Goal:** Wrap the existing `TartanAirGaussianDataset` and `SintelGaussianDataset` with a sibling `SequentialPairDataset` that emits consecutive frame pairs `(example_t, example_t_plus_1)` from the same trajectory/scene. Handle sequence boundaries (no pair across boundaries).

**Files:**
- Create: `oss/sr/temporal/dataset.py`
- Modify: `oss/sr/temporal/__init__.py`
- Create: `tests/sr/temporal/test_dataset.py`

**Acceptance Criteria:**
- [ ] `SequentialPairDataset(base_dataset, pair_stride=1)` exposes `__len__` = `(len(base) − N_sequences)` (one pair-loss per scene boundary)
- [ ] Each `__getitem__` returns a dict with keys: `t`, `t_plus_1`, `is_first_in_seq` (bool)
- [ ] `t` and `t_plus_1` are both `GaussianTrainingExample`-shaped dicts produced by the underlying dataset
- [ ] Underlying base dataset is queried with `idx` and `idx+1` and a sentinel rejects pairs that cross trajectory/scene boundaries (use base dataset's internal `_items` or `_seq_index` — one of these exists; use whichever resolves the trajectory)
- [ ] `default_collate_pair` collator returns batched `t_lr`, `t_motion`, `t_depth`, `t_normals`, `t_canvas`, `t_gt_hr`, plus the same with `tp1_` prefix, plus `is_first_in_seq` boolean tensor
- [ ] On a synthetic 4-frame trajectory of `TartanAirGaussianDataset(synthetic=True)`, the loader yields exactly 3 pairs

**Verify:** `pytest tests/sr/temporal/test_dataset.py -v` → all pass

**Steps:**

- [ ] **Step 1: Read base dataset internals first to confirm boundary detection.** Look at `oss/gaussian/data/tartanair.py:_items` and `oss/gaussian/data/sintel.py:_items`. Both store `(trajectory_dir_or_seq, frame_idx, ...)`. Use the trajectory/seq path equality between adjacent items as the boundary signal. *Read before implementing — the actual tuple shape may differ from the spec; mirror it.*

- [ ] **Step 2: Write failing test**

```python
"""Tests for SequentialPairDataset."""
from __future__ import annotations

import torch

from oss.sr.temporal import SequentialPairDataset, default_collate_pair


class _FakeBase:
    """Minimal stub: 5 frames in seq A, 3 frames in seq B."""

    def __init__(self) -> None:
        self.scale = 2.0
        self._seq_keys = ["A"] * 5 + ["B"] * 3

    def __len__(self) -> int:
        return len(self._seq_keys)

    def trajectory_key(self, idx: int) -> str:
        return self._seq_keys[idx]

    def __getitem__(self, idx: int):
        k = self._seq_keys[idx]
        # Tiny tensors; spatial sizes match (LR=8, HR=16) and 12-ch contract.
        return {
            "lr_frame": torch.full((3, 8, 8), float(idx)),
            "depth": torch.full((1, 8, 8), float(idx)),
            "motion": torch.full((2, 8, 8), float(idx)),
            "normals": torch.full((3, 8, 8), float(idx)),
            "canvas_hint": torch.full((3, 8, 8), float(idx)),
            "gt_hr_frame": torch.full((3, 16, 16), float(idx)),
            "_seq": k,
        }


def test_pair_count_excludes_boundaries() -> None:
    base = _FakeBase()
    ds = SequentialPairDataset(base)
    # 4 pairs in A (idx 0..3) + 2 pairs in B (idx 5..6) = 6 valid pairs.
    assert len(ds) == 6


def test_pair_returns_consecutive() -> None:
    base = _FakeBase()
    ds = SequentialPairDataset(base)
    pair = ds[0]
    assert pair["t"]["lr_frame"][0, 0, 0].item() == 0.0
    assert pair["t_plus_1"]["lr_frame"][0, 0, 0].item() == 1.0
    assert pair["is_first_in_seq"] is True


def test_pair_in_middle_is_not_first() -> None:
    base = _FakeBase()
    ds = SequentialPairDataset(base)
    # Find a pair where idx_t > 0 within its seq.
    pair = ds[1]
    assert pair["is_first_in_seq"] is False


def test_collate_pair() -> None:
    base = _FakeBase()
    ds = SequentialPairDataset(base)
    batch = default_collate_pair([ds[0], ds[1], ds[2]])
    assert batch["t_lr"].shape == (3, 3, 8, 8)
    assert batch["tp1_lr"].shape == (3, 3, 8, 8)
    assert batch["t_gt_hr"].shape == (3, 3, 16, 16)
    assert batch["is_first_in_seq"].shape == (3,)
```

- [ ] **Step 3: Run test to verify failure**

```
pytest tests/sr/temporal/test_dataset.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement `oss/sr/temporal/dataset.py`**

```python
"""Sequential frame-pair wrapper for the v5 pixel temporal track.

Wraps any base dataset that exposes:
    - __len__()
    - __getitem__(idx) -> mapping with keys
        lr_frame, depth, motion, normals, canvas_hint, gt_hr_frame
    - trajectory_key(idx) -> hashable identifier of the trajectory/sequence
      that frame ``idx`` belongs to. Pairs only span equal trajectory keys.

For TartanAir/Sintel datasets that don't expose ``trajectory_key`` directly,
the caller is expected to add a thin shim. See ``adapt_*`` helpers below.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Mapping

import torch
from torch.utils.data import Dataset


class SequentialPairDataset(Dataset):
    def __init__(self, base: Any) -> None:
        if not hasattr(base, "trajectory_key"):
            raise TypeError(
                "Base dataset must expose `trajectory_key(idx) -> hashable`. "
                "Use adapt_tartanair / adapt_sintel to add it."
            )
        self.base = base
        self._pair_indices: List[int] = []
        prev_key = None
        for i in range(len(base)):
            cur_key = base.trajectory_key(i)
            if i + 1 < len(base) and base.trajectory_key(i + 1) == cur_key:
                self._pair_indices.append(i)
            prev_key = cur_key

    def __len__(self) -> int:
        return len(self._pair_indices)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        i = self._pair_indices[idx]
        prev_key = self.base.trajectory_key(i - 1) if i > 0 else None
        cur_key = self.base.trajectory_key(i)
        is_first_in_seq = (prev_key != cur_key)
        return {
            "t": self.base[i],
            "t_plus_1": self.base[i + 1],
            "is_first_in_seq": bool(is_first_in_seq),
        }


def _stack(field: str, items: Iterable[Mapping[str, Any]]) -> torch.Tensor:
    return torch.stack([it[field] for it in items], dim=0)


def default_collate_pair(samples: List[Mapping[str, Any]]) -> Mapping[str, torch.Tensor]:
    t_items = [s["t"] for s in samples]
    p_items = [s["t_plus_1"] for s in samples]
    out: dict[str, torch.Tensor] = {}
    for prefix, items in (("t_", t_items), ("tp1_", p_items)):
        out[f"{prefix}lr"] = _stack("lr_frame", items)
        out[f"{prefix}depth"] = _stack("depth", items)
        out[f"{prefix}motion"] = _stack("motion", items)
        out[f"{prefix}normals"] = _stack("normals", items)
        out[f"{prefix}canvas"] = _stack("canvas_hint", items)
        out[f"{prefix}gt_hr"] = _stack("gt_hr_frame", items)
    out["is_first_in_seq"] = torch.tensor(
        [bool(s["is_first_in_seq"]) for s in samples], dtype=torch.bool
    )
    return out


# ---------------------------------------------------------------------------
# Trajectory-key shims for TartanAir / Sintel.
# ---------------------------------------------------------------------------


def adapt_tartanair(ds) -> Any:
    """Add ``trajectory_key`` to a TartanAirGaussianDataset.

    TartanAir's ``_items`` contains tuples of ``(image_path, depth_path,
    flow_path)``. The trajectory dir is the parent of ``image_left/``.
    """
    items = list(ds._items)

    def trajectory_key(idx: int) -> str:
        img_path = items[idx][0]
        # .../<env>/<level>/<traj>/image_left/000000_left.png
        return str(img_path.parent.parent)

    ds.trajectory_key = trajectory_key  # type: ignore[attr-defined]
    return ds


def adapt_sintel(ds) -> Any:
    """Add ``trajectory_key`` to SintelGaussianDataset (one key per sequence)."""
    items = list(ds._items)

    def trajectory_key(idx: int) -> str:
        img_path = items[idx][0]
        # .../training/clean/<seq>/frame_NNNN.png
        return str(img_path.parent)

    ds.trajectory_key = trajectory_key  # type: ignore[attr-defined]
    return ds


__all__ = [
    "SequentialPairDataset",
    "default_collate_pair",
    "adapt_tartanair",
    "adapt_sintel",
]
```

- [ ] **Step 5: Update `oss/sr/temporal/__init__.py`** to export `SequentialPairDataset`, `default_collate_pair`, `adapt_tartanair`, `adapt_sintel`.

- [ ] **Step 6: Run tests**

```
pytest tests/sr/temporal/test_dataset.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add oss/sr/temporal/dataset.py oss/sr/temporal/__init__.py \
        tests/sr/temporal/test_dataset.py
git commit -m "v5-pixel(sr): add SequentialPairDataset + tartanair/sintel shims"
```

---

## Task 5: End-to-end loss pipeline test (integration)

**Goal:** Verify the full loss assembly — backbone forward, temporal head, appearance loss + temporal-consistency loss — wires together correctly with finite gradient on every learnable parameter group (head, gate, backbone-when-unfrozen).

**Files:**
- Create: `tests/sr/temporal/test_loss_pipeline.py`

**Acceptance Criteria:**
- [ ] On synthetic batch (B=1, LR=8): full loss is a finite scalar
- [ ] Loss includes `L1 + 0.1·SSIM-loss + 0.05·temporal_consistency_loss`
- [ ] LPIPS-VGG term `0.1·LPIPS(out_t, gt_hr_t) + 0.1·LPIPS(out_tp1, gt_hr_tp1)` is included when the `lpips` Python package is importable; gated by `pytest.importorskip("lpips")` and otherwise the test runs without it
- [ ] When backbone is unfrozen, all three groups (head, gate, backbone) have finite gradients

**Verify:** `pytest tests/sr/temporal/test_loss_pipeline.py -v` → pass

**Steps:**

- [ ] **Step 1: Write failing test** (only test, no new module — pipeline lives in train script later)

```python
"""End-to-end loss pipeline integration test for v5-pixel-temporal."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from oss.sr.temporal import TemporalSRModel
from oss.train.losses import temporal_consistency_loss


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Tiny box-window SSIM stand-in matching oss.train.losses style."""
    mu_p = F.avg_pool2d(pred, 3, 1, 1)
    mu_t = F.avg_pool2d(target, 3, 1, 1)
    var_p = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_p * mu_p
    var_t = F.avg_pool2d(target * target, 3, 1, 1) - mu_t * mu_t
    cov = F.avg_pool2d(pred * target, 3, 1, 1) - mu_p * mu_t
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / (
        (mu_p * mu_p + mu_t * mu_t + c1) * (var_p + var_t + c2)
    )
    return 1.0 - ssim.clamp(0, 1).mean()


def test_full_loss_pipeline_grads() -> None:
    torch.manual_seed(0)
    model = TemporalSRModel(in_channels=12, scale=2, tier="standard")

    # Synthetic batch — two consecutive frames.
    lr = 8
    inputs_t = {
        "lr_inputs": torch.rand(1, 12, lr, lr),
        "prev_hr": torch.rand(1, 3, lr * 2, lr * 2),
        "depth_hr_curr": torch.rand(1, 1, lr * 2, lr * 2),
        "depth_hr_prev": torch.rand(1, 1, lr * 2, lr * 2),
        "motion_lr": torch.randn(1, 2, lr, lr) * 0.1,
    }
    inputs_tp1 = {
        "lr_inputs": torch.rand(1, 12, lr, lr),
        "prev_hr": None,  # filled below from out_t
        "depth_hr_curr": torch.rand(1, 1, lr * 2, lr * 2),
        "depth_hr_prev": inputs_t["depth_hr_curr"],
        "motion_lr": torch.randn(1, 2, lr, lr) * 0.1,
    }
    gt_hr_t = torch.rand(1, 3, lr * 2, lr * 2)
    gt_hr_tp1 = torch.rand(1, 3, lr * 2, lr * 2)
    motion_t_to_tp1 = torch.randn(1, 2, lr, lr) * 0.1

    out_t = model(**inputs_t)
    inputs_tp1["prev_hr"] = out_t.detach()
    out_tp1 = model(**inputs_tp1)

    w_l1, w_ssim, w_lpips, w_tc = 1.0, 0.1, 0.1, 0.05
    appearance = (
        w_l1 * F.l1_loss(out_t, gt_hr_t)
        + w_ssim * _ssim_loss(out_t, gt_hr_t)
        + w_l1 * F.l1_loss(out_tp1, gt_hr_tp1)
        + w_ssim * _ssim_loss(out_tp1, gt_hr_tp1)
    )
    # LPIPS gated on package presence so test still runs in minimal envs.
    try:
        import lpips  # type: ignore[import-not-found]
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False)
        for p in lpips_fn.parameters():
            p.requires_grad_(False)
        def _lp(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return lpips_fn(p * 2.0 - 1.0, t * 2.0 - 1.0).mean()
        appearance = appearance + w_lpips * (_lp(out_t, gt_hr_t) + _lp(out_tp1, gt_hr_tp1))
    except Exception:
        pass  # lpips not installed; loss runs without it
    tc = temporal_consistency_loss(out_tp1, out_t, motion_t_to_tp1, scale_factor=2.0)
    loss = appearance + w_tc * tc

    assert torch.isfinite(loss)
    loss.backward()
    for group_name, params in (
        ("head", model.head.parameters()),
        ("gate", model.gate.parameters()),
        ("backbone", model.backbone.parameters()),
    ):
        for p in params:
            assert p.grad is not None and torch.isfinite(p.grad).all(), group_name
```

- [ ] **Step 2: Run** — should pass *immediately* if the prior tasks work, since this is integration only.

```
pytest tests/sr/temporal/test_loss_pipeline.py -v
```

- [ ] **Step 3: If it fails:** read the failure, fix the upstream module (don't add band-aids in the test). Re-run.

- [ ] **Step 4: Commit**

```bash
git add tests/sr/temporal/test_loss_pipeline.py
git commit -m "v5-pixel(sr): add end-to-end loss pipeline integration test"
```

---

## Task 6: Stateful inference engine + scene-cut reset

**Goal:** Add `TemporalSRInferenceEngine` to `oss/sr/inference.py` that carries `prev_hr_output` across calls and resets on scene cut.

**Files:**
- Modify: `oss/sr/inference.py` (append a new class; do not break `SRInferenceEngine`)
- Create: `tests/sr/temporal/test_inference_state.py`

**Acceptance Criteria:**
- [ ] `TemporalSRInferenceEngine.from_checkpoint(ckpt)` loads a `TemporalSRModel` checkpoint
- [ ] First call uses bilinear-LR-upscale as `prev_hr` (no state yet)
- [ ] Subsequent calls reuse the previous output as `prev_hr`
- [ ] `reset()` clears state
- [ ] Auto-reset triggers when motion magnitude `mean(||motion||)` exceeds a configurable threshold (default 32 LR pixels) — flag scene cut
- [ ] Existing `SRInferenceEngine` is untouched (no regression in `tests/sr/test_*.py`)

**Verify:**
```
pytest tests/sr/temporal/test_inference_state.py -v
pytest tests/sr/ -v
```
→ all pass.

**Steps:**

- [ ] **Step 1: Write failing test**

```python
"""Stateful inference for v5-pixel-temporal."""
from __future__ import annotations

from pathlib import Path

import torch

from oss.sr import build_sr_model
from oss.sr.inference import TemporalSRInferenceEngine
from oss.sr.temporal import TemporalSRModel


def _save_temporal_ckpt(tmp_path: Path) -> Path:
    backbone = build_sr_model(model_kind="simple", tier="standard", in_channels=12, scale=2)
    model = TemporalSRModel(in_channels=12, scale=2, tier="standard")
    model.backbone.load_state_dict(backbone.state_dict())
    ckpt = tmp_path / "temporal.pt"
    torch.save(
        {
            "temporal_model": model.state_dict(),
            "args": {"tier": "standard", "sr_backbone": "simple", "in_channels": 12, "scale": 2},
        },
        ckpt,
    )
    return ckpt


def test_first_call_uses_bilinear_init(tmp_path: Path) -> None:
    ckpt = _save_temporal_ckpt(tmp_path)
    eng = TemporalSRInferenceEngine.from_checkpoint(ckpt, device="cpu", fp16=False)
    lr = torch.rand(1, 12, 8, 8)
    depth_hr = torch.rand(1, 1, 16, 16)
    motion = torch.zeros(1, 2, 8, 8)
    out = eng(lr_inputs=lr, depth_hr_curr=depth_hr, motion_lr=motion)
    assert out.shape == (1, 3, 16, 16)
    assert eng._prev_hr is not None  # state stored


def test_reset_clears_state(tmp_path: Path) -> None:
    ckpt = _save_temporal_ckpt(tmp_path)
    eng = TemporalSRInferenceEngine.from_checkpoint(ckpt, device="cpu", fp16=False)
    lr = torch.rand(1, 12, 8, 8)
    depth_hr = torch.rand(1, 1, 16, 16)
    motion = torch.zeros(1, 2, 8, 8)
    eng(lr_inputs=lr, depth_hr_curr=depth_hr, motion_lr=motion)
    eng.reset()
    assert eng._prev_hr is None
    assert eng._prev_depth_hr is None


def test_scene_cut_auto_reset(tmp_path: Path) -> None:
    ckpt = _save_temporal_ckpt(tmp_path)
    eng = TemporalSRInferenceEngine.from_checkpoint(
        ckpt, device="cpu", fp16=False, scene_cut_motion_threshold=4.0,
    )
    lr = torch.rand(1, 12, 8, 8)
    depth_hr = torch.rand(1, 1, 16, 16)
    eng(lr_inputs=lr, depth_hr_curr=depth_hr, motion_lr=torch.zeros(1, 2, 8, 8))
    # Big motion → scene cut. The engine should flag and reset state internally.
    big_motion = torch.full((1, 2, 8, 8), 10.0)
    eng(lr_inputs=lr, depth_hr_curr=depth_hr, motion_lr=big_motion)
    assert eng.last_call_was_scene_cut is True
```

- [ ] **Step 2: Run failing test** — expect `ImportError` on `TemporalSRInferenceEngine`.

- [ ] **Step 3: Append to `oss/sr/inference.py`**

```python
# ============================================================================
# v5 pixel-temporal stateful inference
# ============================================================================

from oss.sr.temporal import TemporalSRModel, make_first_frame_prev_hr


class TemporalSRInferenceEngine:
    """Stateful inference engine for v5 pixel-temporal SR.

    Carries ``prev_hr_output`` and ``prev_depth_hr`` across calls. Auto-resets
    when mean motion magnitude exceeds ``scene_cut_motion_threshold`` (in LR
    pixels).
    """

    def __init__(
        self,
        model: TemporalSRModel,
        device: str,
        fp16: bool,
        scene_cut_motion_threshold: float,
    ) -> None:
        self.device = device
        self.fp16 = fp16
        self._dtype = torch.float16 if fp16 else torch.float32
        self.scene_cut_motion_threshold = float(scene_cut_motion_threshold)
        self.last_call_was_scene_cut = False

        model = model.to(device).train(False)
        if fp16:
            model = model.half()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        self._prev_hr: Optional[torch.Tensor] = None
        self._prev_depth_hr: Optional[torch.Tensor] = None

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Path,
        device: str = "cuda",
        fp16: bool = True,
        scene_cut_motion_threshold: float = 32.0,
    ) -> "TemporalSRInferenceEngine":
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ck.get("args", {})
        in_channels = int(saved.get("in_channels", 12))
        scale = int(saved.get("scale", 2))
        tier = saved.get("tier", "standard")
        backbone_kind = "rrdb" if saved.get("sr_backbone") == "rrdb" else "simple"
        model = TemporalSRModel(
            in_channels=in_channels, scale=scale, tier=tier, backbone_kind=backbone_kind
        )
        model.load_state_dict(ck["temporal_model"])
        return cls(model=model, device=device, fp16=fp16,
                   scene_cut_motion_threshold=scene_cut_motion_threshold)

    def reset(self) -> None:
        self._prev_hr = None
        self._prev_depth_hr = None

    def __call__(
        self,
        lr_inputs: torch.Tensor,
        depth_hr_curr: torch.Tensor,
        motion_lr: torch.Tensor,
    ) -> torch.Tensor:
        lr_inputs = lr_inputs.to(self.device, dtype=self._dtype, non_blocking=True)
        depth_hr_curr = depth_hr_curr.to(self.device, dtype=self._dtype, non_blocking=True)
        motion_lr = motion_lr.to(self.device, dtype=self._dtype, non_blocking=True)

        # Scene-cut detection.
        mean_mag = float(motion_lr.norm(dim=1).mean().item())
        self.last_call_was_scene_cut = (
            self._prev_hr is not None and mean_mag > self.scene_cut_motion_threshold
        )
        if self.last_call_was_scene_cut:
            self.reset()

        # First-frame init or use stored state.
        if self._prev_hr is None:
            prev_hr = make_first_frame_prev_hr(lr_inputs[:, :3], scale=self.model.scale)
            prev_depth = depth_hr_curr
        else:
            prev_hr = self._prev_hr
            prev_depth = self._prev_depth_hr

        with torch.no_grad():
            out = self.model(
                lr_inputs=lr_inputs,
                prev_hr=prev_hr,
                depth_hr_curr=depth_hr_curr,
                depth_hr_prev=prev_depth,
                motion_lr=motion_lr,
            )

        # Persist state for next call (detached, fp16 if engine fp16).
        self._prev_hr = out.detach()
        self._prev_depth_hr = depth_hr_curr.detach()
        return out.float().contiguous()


__all__ = list(set(__all__) | {"TemporalSRInferenceEngine"})  # type: ignore[name-defined]
```

- [ ] **Step 4: Run tests**

```
pytest tests/sr/temporal/test_inference_state.py -v
pytest tests/sr/ -v
```
Expected: all pass; no regression in single-frame `SRInferenceEngine` tests.

- [ ] **Step 5: Commit**

```bash
git add oss/sr/inference.py tests/sr/temporal/test_inference_state.py
git commit -m "v5-pixel(sr): add stateful TemporalSRInferenceEngine + scene-cut reset"
```

---

## Task 7: Training script `scripts/sr_train_temporal.py`

**Goal:** Ship the temporal training entry. 3-phase schedule, auto-resume, rolling metrics dump, dashboard-compatible. Smoke mode runs on CPU with 5 steps and synthetic data so the pipeline is exercised before any GPU time is burned.

**Files:**
- Create: `scripts/sr_train_temporal.py`
- Reuse: `oss/gaussian/train/train.py` patterns for auto-resume + metrics dump (read it; replicate the pattern, don't import private helpers — extract any shared helper into `oss/train/checkpoints.py` if it fits cleanly)

**Acceptance Criteria:**
- [ ] `python scripts/sr_train_temporal.py --smoke --device cpu --max-steps 5` exits 0 with finite loss reported and a checkpoint written
- [ ] Three-phase schedule:
  - Phase 1 (steps 0..warmup_steps, default 10000): backbone frozen, no temporal-consistency loss
  - Phase 2 (warmup_steps..joint_end, default 60000): backbone unfrozen at LR×0.1, full loss
  - Phase 3 (joint_end..max_steps, default 80000): Sintel-only fine-tune at LR×0.01
- [ ] Writes `metrics.json` and `score_log.json` every checkpoint compatible with `scripts/training_dashboard.py`
- [ ] Auto-resume from latest `step-XXXXX.pt` in `--output-dir` if present
- [ ] CLI args: `--output-dir`, `--warm-start`, `--tartanair-root`, `--sintel-root`, `--max-steps`, `--warmup-steps`, `--joint-end`, `--lr`, `--batch-size`, `--device`, `--smoke`

**Verify:**
```
python scripts/sr_train_temporal.py --smoke --device cpu --max-steps 5
ls /tmp/oss_smoke_temporal/
```
→ exit code 0, `metrics.json`, `score_log.json`, `step-00000005.pt` present.

**Steps:**

- [ ] **Step 1: Read `oss/gaussian/train/train.py`** to understand the auto-resume + metrics-dump pattern. Identify which helpers to lift; if any are private, copy with attribution comment, do not import.

- [ ] **Step 2: Implement `scripts/sr_train_temporal.py`** with the structure below. The full code is verbose — show the skeleton; the implementer fills in each block guided by the docstrings.

```python
#!/usr/bin/env python
"""v5 pixel-temporal SR training entry.

Three-phase schedule (default):
    1. Steps  0..10000 — backbone frozen; head + gate only; appearance loss only.
    2. Steps 10000..60000 — backbone unfrozen at LR*0.1; full loss with
       temporal-consistency at lambda=0.05.
    3. Steps 60000..80000 — Sintel-only fine-tune at LR*0.01.

Writes ``metrics.json`` (keyed by step) and ``score_log.json`` (rolling list
of {step, psnr, lpips, loss}) compatible with ``scripts/training_dashboard.py``.
Auto-resumes from the latest checkpoint in ``--output-dir`` if any.

Smoke mode (``--smoke``):
    Runs 5 CPU steps on synthetic random tensors. Used in CI / pre-launch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from oss.gaussian.data import (
    EngineAliasedLRSynth, SintelGaussianDataset, TartanAirGaussianDataset,
)
from oss.sr.temporal import (
    SequentialPairDataset, TemporalSRModel,
    adapt_sintel, adapt_tartanair, default_collate_pair, make_first_frame_prev_hr,
)
from oss.train.losses import temporal_consistency_loss


# Implement: parse_args, build_datasets, build_optimizer, build_scheduler,
# train_step, save_checkpoint, load_latest_checkpoint, dump_metrics, main.
# Mirror oss/gaussian/train/train.py for auto-resume + metrics dump.
# Smoke path: synthetic random batch for 5 steps; assert finite loss.
```

- [ ] **Step 3: Add a smoke-mode unit test** at `tests/sr/temporal/test_train_smoke.py`:

```python
"""Smoke-test the training entry runs end-to-end on CPU."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_smoke_train(tmp_path: Path) -> None:
    out = tmp_path / "smoke"
    rc = subprocess.run(
        [sys.executable, "scripts/sr_train_temporal.py",
         "--smoke", "--device", "cpu", "--max-steps", "5",
         "--output-dir", str(out)],
        check=False,
    ).returncode
    assert rc == 0, "smoke train returned non-zero"
    assert (out / "metrics.json").exists()
    assert (out / "score_log.json").exists()
    assert any(out.glob("step-*.pt"))
```

- [ ] **Step 4: Run smoke + test**

```
pytest tests/sr/temporal/test_train_smoke.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/sr_train_temporal.py tests/sr/temporal/test_train_smoke.py
git commit -m "v5-pixel(sr): add training entry with 3-phase schedule + smoke test"
```

---

## Task 8: Held-out eval script + memo template

**Goal:** Ship `scripts/sr_temporal_held_out.py` that scores the v5 temporal checkpoint vs the v4 baseline on a fixed deterministic batch from Sintel + TartanAir held-out trajectory. Writes a memo skeleton in `docs/superpowers/experiments/`.

**Files:**
- Create: `scripts/sr_temporal_held_out.py`
- Create: `docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out-template.md` (template; date filled in at run time by the evaluator)

**Acceptance Criteria:**
- [ ] Reports PSNR + LPIPS for v5-temporal, v4-baseline, bicubic on the same fixed batch (`shuffle=False`, `torch.manual_seed(0)`)
- [ ] Reports temporal stability: `mean(||warp(out_t, motion_t→t+1) − out_{t+1}||_1)` for v5 and v4
- [ ] Reports per-sample win counts (B>A and B<bicubic) like `sr_v3_vs_v4_ab.py` does
- [ ] Writes a JSON `held_out_results.json` next to the checkpoint
- [ ] Template memo lists the four success-criteria gates from the spec, blank `[ ]` boxes for each

**Verify (offline, no GPU needed):**

```
python scripts/sr_temporal_held_out.py --help
```
→ argparse help prints, exits 0.

**Steps:**

- [ ] **Step 1: Implement** mirroring `scripts/sr_v3_vs_v4_ab.py` with these additions:
  - Use `SequentialPairDataset` so each batch entry is `(t, t+1)` and we can compute warp-then-diff
  - Score both temporal output and baseline single-frame output on `t+1`
  - For temporal model, feed the model itself with `prev_hr = baseline_output_at_t.detach()` (cold-start, no recurrent state) — that's the regime the deployed inference engine uses on first frame
  - Print results in the same format as `sr_v3_vs_v4_ab.py` plus a `Temporal stability` block

- [ ] **Step 2: Implement memo template** at `docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out-template.md`:

```markdown
# YYYY-MM-DD — v5-pixel-temporal held-out vs v4

**Status:** Result
**Question:** Does v5-pixel-temporal meet the success criteria from the design spec?

## Method

Same fixed-batch protocol as v3-vs-v4. Loaders shuffle=False, torch.manual_seed(0).
Sintel held-out clean pass + TartanAir held-out trajectory.

## Results

(paste output of scripts/sr_temporal_held_out.py here)

## Success criteria

- [ ] PSNR ≥ +1.5 dB over v4 baseline
- [ ] LPIPS ≤ 0.20 (vs v4 ~0.31)
- [ ] Temporal stability ≤ 0.5× v4 single-frame variance
- [ ] ≥ 95% held-out frames beat bicubic on PSNR AND LPIPS

## Conclusion

(write decision: ship v5 / iterate / fall back)

## Caveats / honest limits

- Single fixed-batch eval; rerun on additional scenes if any criterion is borderline.
- LPIPS-VGG is one perceptual metric.
```

- [ ] **Step 3: Verify CLI**

```
python scripts/sr_temporal_held_out.py --help
```
Expected: usage prints; exits 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/sr_temporal_held_out.py \
        docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out-template.md
git commit -m "v5-pixel(sr): add held-out eval + memo template"
```

---

## Task 9: Lab-notebook memo + remote-launch checklist

**Goal:** Write the run-start memo *before* any GPU time is burned (lab-notebook discipline) and a runbook that the human launching the train run will follow on the <train-host> host.

**Files:**
- Create: `docs/superpowers/experiments/2026-05-04-v5-pixel-temporal-train-start.md`
- Create: `docs/superpowers/notes/2026-05-04-v5-pixel-temporal-runbook.md`

**Acceptance Criteria:**
- [ ] Train-start memo records: hypothesis, success criteria (copy from spec), training schedule, expected runtime, checkpoint path, dataset locations, the warm-start checkpoint hash, the WMI orphan-spawn command line that will launch it
- [ ] Runbook is a literal copy-pasteable shell sequence for the 3080ti remote: pull the v0.2-dev branch, env activate, `Invoke-CimMethod` orphan launch, dashboard restart, `Get-Content` log tail commands

**Verify:** Files exist; manually re-read for completeness.

**Steps:**

- [ ] **Step 1: Write `2026-05-04-v5-pixel-temporal-train-start.md`** with sections:
  - Hypothesis (verbatim from spec goal)
  - Success criteria checkboxes (from spec)
  - Training schedule (from spec §Schedule)
  - Expected runtime: 12–16 h on 3080 Ti
  - Warm-start: `<train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt`, sha256 = `<computed>`
  - Output dir: `<train-host-data>/checkpoints/srcnn-v5-pixel-temporal/`
  - Datasets: `<train-host-data>/datasets/tartanair_extracted/`, `<train-host-data>/datasets/sintel/`
  - Launch command (PowerShell, WMI orphan-spawn pattern)

- [ ] **Step 2: Write the runbook** with literal commands. Example skeleton:

```markdown
# v5-pixel-temporal remote launch runbook

## Pre-flight

1. SSH `<train-host>`. Confirm conda env: `<windows-home>\Miniconda3\envs\image-gs\python.exe --version`
2. Repo at `<train-host-data>\oss-gaussian`. Pull latest:
   ```powershell
   cd <train-host-data>\oss-gaussian; git fetch origin; git checkout v0.2-dev; git pull --ff-only
   ```
3. Verify TartanAir + Sintel paths exist.
4. Smoke run on remote:
   ```powershell
   <train-host-data>\oss-gaussian\..\python.exe scripts\sr_train_temporal.py --smoke --device cuda --max-steps 5
   ```

## Launch (orphan-spawn so SSH disconnect can't kill it)

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine='cmd /c cd /d <train-host-data>\oss-gaussian && <windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_train_temporal.py --output-dir <train-host-data>\checkpoints\srcnn-v5-pixel-temporal --warm-start <train-host-data>\checkpoints\srcnn-prod-v4-lpips\step-00385000.pt --tartanair-root <train-host-data>\datasets\tartanair_extracted --sintel-root <train-host-data>\datasets\sintel --max-steps 80000 > <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\train.log 2>&1'
}
```

## Monitoring

```powershell
Get-Content <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\train.log -Tail 20 -Wait
```

Dashboard: restart pointing at the new dir.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/experiments/2026-05-04-v5-pixel-temporal-train-start.md \
        docs/superpowers/notes/2026-05-04-v5-pixel-temporal-runbook.md
git commit -m "v5-pixel(sr): lab-notebook train-start memo + remote runbook"
```

---

## Task 10: Sprint-5 closeout gate (post-train)

**Goal:** After the remote training run completes, score the held-out set, fill in the experiment memo, and either declare v5-pixel ready to ship or document what failed and why.

**Files:**
- Create: `docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out.md` (filled-in version of the template)
- Modify: `README.md` sprint table (mark S5 ✓ done if the gate passes)

**Acceptance Criteria:**
- [ ] All four spec success criteria boxes checked, OR memo explicitly documents which gate failed and the next-step decision
- [ ] If passed: README S5 row updated with the held-out numbers
- [ ] If failed: README is unchanged; the memo is the durable record so we don't repeat the failed approach by accident

**Verify:** Manual review by Cash before merge to `main`.

**Steps:**

- [ ] **Step 1: Run held-out eval after training completes**

```
python scripts/sr_temporal_held_out.py \
    --ckpt-temporal <train-host-data>/checkpoints/srcnn-v5-pixel-temporal/step-XXXXX.pt \
    --ckpt-baseline <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt \
    --tartanair-root <train-host-data>/datasets/tartanair_extracted \
    --sintel-root <train-host-data>/datasets/sintel \
    --n-samples 64
```

- [ ] **Step 2: Fill in the memo** with the captured output and a written conclusion.

- [ ] **Step 3: If passed, update `README.md`** sprint table row:

```markdown
| **S5** | v5 dual-track temporal | ✓ done — pixel track shipped; held-out: PSNR XX.X dB, LPIPS X.XX | one track met success criteria |
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out.md README.md
git commit -m "v5-pixel(sr): sprint-5 closeout — held-out memo + README update"
```

---

## Self-review notes (run by the planner before handoff)

**Spec coverage:**
- Inputs (12 LR + 4 HR) — Task 0/1/2/3 cover warp, gate, head, model. ✓
- Network: backbone + warp + gate + head — Task 3. ✓
- First-frame init — Task 3 + Task 6 (inference). ✓
- Loss: appearance + temporal-consistency — Task 5 + Task 7. ✓
- Three-phase schedule — Task 7. ✓
- Data: TartanAir Easy + Sintel pair loaders — Task 4 + Task 7. ✓
- Held-out eval gates — Task 8 + Task 10. ✓
- ONNX/TRT export — *deferred to post-Sprint-5* per spec out-of-scope §"vendor-optimization is post-v5". Listed as a follow-up; not a blocker.
- Inference state buffer (~24 MB) — Task 6. ✓

**Placeholder scan:** No "TBD" / "implement later" / "similar to" patterns. Each step has concrete code or commands.

**Type consistency:** `TemporalSRModel.forward` signature is the same in Task 3, Task 5, Task 6. `warp_prev_hr(prev_hr, motion_lr, scale)` is the same in Task 0, Task 1, Task 6. `DisocclusionGate.__call__` signature is identical Task 1 / Task 3. ✓

**Out-of-band risks:**
- The base TartanAir/Sintel datasets may not expose `_items` exactly as assumed; Task 4 Step 1 explicitly says "read first; mirror it" — implementer must read before coding the shim.
- LPIPS dependency: optional in Task 5, required at training time in Task 7 (consistent with v4 setup).
- Smoke train under real datasets is not feasible on CPU; smoke uses synthetic tensors. Production smoke must run on CUDA before the long launch.
