# v5 Gaussian Temporal Super-Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a persistent 2D-Gaussian-field temporal SR system that maintains a Gaussian set across frames, warps it analytically by motion vectors (mean shift + covariance Jacobian transform), updates it via a multi-frame transformer attending over Gaussian tokens, and densifies under disocclusion. Race the v5-pixel-temporal control track on the same fixed held-out batch; ship as v5 only if it explicitly beats pixel.

**Architecture:** Per-frame G-buffer encoder feeds a transformer that attends over a token sequence of (current LR features) ⊕ (previous-frame Gaussians). Existing V0.5 `PersistentCanvas` + `Rasterizer` + `warp_positions` are the substrate. New: analytical covariance Jacobian warp, per-token transformer update head producing `(Δμ, Δlog_scale, Δrot, Δc)`, residual-driven differentiable densification, multi-frame trajectory window dataset, four-phase training (single-frame fitter → temporal warmup → joint → Sintel fine-tune).

**Tech Stack:** PyTorch 2.4.1, existing `oss.gaussian.canvas.{PersistentCanvas, warp_positions}`, existing `oss.gaussian.renderer.{Rasterizer, GaussianBatch}`, existing `oss.gaussian.network.{param_net, output_head, prior_bank}`, existing `oss.gaussian.data.*`, `oss.train.losses.temporal_consistency_loss`, `lpips` package, training dashboard.

**Spec:** `docs/superpowers/specs/2026-05-04-v5-gaussian-temporal-design.md`

**Branch:** `v0.2-dev`. Trains second on the shared 3080 Ti, after v5-pixel-temporal completes its run (or scheduled by Cash).

---

## File Structure

New module `oss/sr/gaussian_temporal/`:

| File | Responsibility |
|---|---|
| `oss/sr/gaussian_temporal/__init__.py` | Public exports |
| `oss/sr/gaussian_temporal/gaussian_field.py` | `GaussianField` SoA container (μ, log_scale, rot, c, opacity) + history buffer for N=5 prev frames |
| `oss/sr/gaussian_temporal/analytical_warp.py` | Mean shift via motion sample + covariance Jacobian transform `Σ' = J Σ Jᵀ` |
| `oss/sr/gaussian_temporal/g_buffer_encoder.py` | Tiny CNN turning the 12-ch LR stack into per-tile context features (~100K params) |
| `oss/sr/gaussian_temporal/transformer.py` | Multi-frame attention over Gaussian tokens with rotary positional embeddings keyed on μ |
| `oss/sr/gaussian_temporal/densification.py` | Residual-driven new-Gaussian spawning (heuristic 3DGS-style for v5; soft top-K is post-v5) |
| `oss/sr/gaussian_temporal/pruning.py` | Opacity-threshold + count-cap removal |
| `oss/sr/gaussian_temporal/rasterizer.py` | Wrapper around `oss.gaussian.renderer.Rasterizer` taking a `GaussianField` and producing HR output |
| `oss/sr/gaussian_temporal/model.py` | `GaussianTemporalSRModel` — wires encoder + warp + transformer + densify + prune + raster |
| `oss/sr/gaussian_temporal/dataset.py` | `TrajectoryWindowDataset` returning N consecutive frames (default N=5) |
| `oss/sr/gaussian_temporal/regularization.py` | `gaussian_regularization_loss(positions, sigmas, count)` for drift / area / count terms |
| `scripts/sr_train_gaussian_temporal.py` | 4-phase training entry, auto-resume, dashboard metrics |
| `scripts/sr_gaussian_temporal_held_out.py` | Fixed-batch held-out eval vs both v5-pixel-temporal and v4 baseline |
| `oss/sr/inference.py` (modify) | Add `GaussianTemporalSRInferenceEngine` carrying `GaussianField` state |

New tests under `tests/sr/gaussian_temporal/`:

| File | Tests |
|---|---|
| `test_gaussian_field.py` | SoA shapes, push/pop history, count cap, device move |
| `test_analytical_warp.py` | μ shift correctness, Σ Jacobian against numerical, rotation invariance under translation flow, edge: out-of-frame Gaussians |
| `test_g_buffer_encoder.py` | shape, ≤120K params, gradient flow |
| `test_transformer.py` | attention forward, gradient flow, token equivariance under permutation, RoPE wired by μ |
| `test_densification.py` | spawn count, residual-driven choice, gradient through inserted Gaussians |
| `test_pruning.py` | low-opacity removal, count cap enforcement |
| `test_rasterizer_wrapper.py` | shape parity vs `oss.gaussian.renderer.Rasterizer`, fp32 contiguous out |
| `test_model_full_step.py` | end-to-end train step on synthetic moving rectangle: loss finite, grads finite |
| `test_dataset.py` | window loader returns N consecutive frames; rejects windows crossing trajectory boundaries |
| `test_regularization.py` | drift / area / count terms each return finite scalar; all zero on a steady-state field |
| `test_inference_state.py` | stateful engine maintains GaussianField across calls; reset clears it |

Total new code target: ~1500 LOC + ~900 LOC tests. Reuse `oss.gaussian.canvas.warp.warp_positions` for μ-shift, do NOT re-implement.

---

## Verification commands

```bash
# All Gaussian-temporal tests
pytest tests/sr/gaussian_temporal/ -v

# Single-frame fitter sanity (Phase 1) on Sintel held-out frame
python scripts/sr_train_gaussian_temporal.py --phase 1 --smoke --device cpu --max-steps 5

# Full smoke (synthetic moving rectangle, 5 steps, CPU)
python scripts/sr_train_gaussian_temporal.py --smoke --device cpu --max-steps 5

# Real train (remote 3080 Ti)
python scripts/sr_train_gaussian_temporal.py \
    --output-dir <train-host-data>/checkpoints/srcnn-v5-gaussian-temporal \
    --tartanair-root <train-host-data>/datasets/tartanair_extracted \
    --sintel-root <train-host-data>/datasets/sintel \
    --max-steps 140000

# Held-out eval (compares v5-gaussian vs v5-pixel vs v4)
python scripts/sr_gaussian_temporal_held_out.py \
    --ckpt-gaussian <train-host-data>/checkpoints/srcnn-v5-gaussian-temporal/step-XXXXX.pt \
    --ckpt-pixel    <train-host-data>/checkpoints/srcnn-v5-pixel-temporal/step-YYYYY.pt \
    --ckpt-baseline <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt \
    --tartanair-root <train-host-data>/datasets/tartanair_extracted \
    --sintel-root <train-host-data>/datasets/sintel \
    --n-samples 64
```

---

## Task 0: Module scaffold + GaussianField container

**Goal:** Ship `GaussianField` — SoA container holding `(μ, log_scale, rotation, color, opacity)` plus an N-frame history of prior fields. Used as the persistent state across frames.

**Files:**
- Create: `oss/sr/gaussian_temporal/__init__.py`
- Create: `oss/sr/gaussian_temporal/gaussian_field.py`
- Create: `tests/sr/gaussian_temporal/__init__.py`
- Create: `tests/sr/gaussian_temporal/test_gaussian_field.py`

**Acceptance Criteria:**
- [ ] `GaussianField(capacity=N, device='cpu')` constructs with all SoA tensors zero-initialized
- [ ] Tensor shapes: `mu (N, 2)`, `log_scale (N, 2)`, `rotation (N,)`, `color (N, 3)`, `opacity (N,)`, `alive (N,)` bool
- [ ] `push_history(field)` keeps the last 5 fields; `field.history` exposes them as a list newest-first
- [ ] `to(device)` moves all tensors
- [ ] `count_alive()` matches sum of alive mask
- [ ] `clone()` returns a deep copy

**Verify:** `pytest tests/sr/gaussian_temporal/test_gaussian_field.py -v` → pass

**Steps:**

- [ ] **Step 1: Create skeleton**

```bash
mkdir -p oss/sr/gaussian_temporal tests/sr/gaussian_temporal
touch oss/sr/gaussian_temporal/__init__.py tests/sr/gaussian_temporal/__init__.py
```

- [ ] **Step 2: Write failing test**

```python
"""GaussianField container tests."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import GaussianField


def test_default_shapes() -> None:
    f = GaussianField(capacity=8)
    assert f.mu.shape == (8, 2)
    assert f.log_scale.shape == (8, 2)
    assert f.rotation.shape == (8,)
    assert f.color.shape == (8, 3)
    assert f.opacity.shape == (8,)
    assert f.alive.shape == (8,) and f.alive.dtype == torch.bool


def test_count_alive() -> None:
    f = GaussianField(capacity=8)
    f.alive[:5] = True
    assert f.count_alive() == 5


def test_history_window_capped_at_5() -> None:
    f = GaussianField(capacity=4)
    for _ in range(7):
        f.push_history(f.clone())
    assert len(f.history) == 5


def test_to_device_moves_all() -> None:
    f = GaussianField(capacity=4)
    f2 = f.to("cpu")  # no-op but exercises the path
    assert f2.mu.device.type == "cpu"


def test_clone_is_deep() -> None:
    f = GaussianField(capacity=4)
    f.mu.fill_(1.0)
    g = f.clone()
    g.mu.fill_(2.0)
    assert f.mu.mean().item() == 1.0
    assert g.mu.mean().item() == 2.0
```

- [ ] **Step 3: Run failing test**

```
pytest tests/sr/gaussian_temporal/test_gaussian_field.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement `oss/sr/gaussian_temporal/gaussian_field.py`**

```python
"""GaussianField — SoA persistent state for the v5 Gaussian temporal track.

Storage layout (one row per Gaussian, capacity N):
    mu         : (N, 2)   pixel-space sub-pixel positions (x, y)
    log_scale  : (N, 2)   per-axis log-scale; scale = exp(log_scale)
    rotation   : (N,)     orientation in radians
    color      : (N, 3)   RGB
    opacity    : (N,)     alpha in [0, 1] post-sigmoid
    alive      : (N,)     bool — false rows are free slots ready for densification

History: a deque of up to 5 prior `GaussianField` snapshots (newest first).
Used by the multi-frame transformer to attend over previous fields.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import torch


HISTORY_LEN = 5


class GaussianField:
    def __init__(self, capacity: int = 16384, device: str | torch.device = "cpu") -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.mu = torch.zeros((self.capacity, 2), device=self.device)
        self.log_scale = torch.zeros((self.capacity, 2), device=self.device)
        self.rotation = torch.zeros((self.capacity,), device=self.device)
        self.color = torch.zeros((self.capacity, 3), device=self.device)
        self.opacity = torch.zeros((self.capacity,), device=self.device)
        self.alive = torch.zeros((self.capacity,), dtype=torch.bool, device=self.device)
        self._history: Deque["GaussianField"] = deque(maxlen=HISTORY_LEN)

    # ---- access -----------------------------------------------------------

    @property
    def history(self) -> list["GaussianField"]:
        return list(self._history)

    def count_alive(self) -> int:
        return int(self.alive.sum().item())

    # ---- mutators ---------------------------------------------------------

    def push_history(self, snapshot: "GaussianField") -> None:
        self._history.appendleft(snapshot)

    def to(self, device: str | torch.device) -> "GaussianField":
        device = torch.device(device)
        moved = GaussianField(capacity=self.capacity, device=device)
        moved.mu = self.mu.to(device)
        moved.log_scale = self.log_scale.to(device)
        moved.rotation = self.rotation.to(device)
        moved.color = self.color.to(device)
        moved.opacity = self.opacity.to(device)
        moved.alive = self.alive.to(device)
        # Move history snapshots too.
        moved._history = deque(
            (h.to(device) for h in self._history), maxlen=HISTORY_LEN
        )
        return moved

    def clone(self) -> "GaussianField":
        c = GaussianField(capacity=self.capacity, device=self.device)
        c.mu = self.mu.clone()
        c.log_scale = self.log_scale.clone()
        c.rotation = self.rotation.clone()
        c.color = self.color.clone()
        c.opacity = self.opacity.clone()
        c.alive = self.alive.clone()
        return c


__all__ = ["GaussianField", "HISTORY_LEN"]
```

- [ ] **Step 5: Update `oss/sr/gaussian_temporal/__init__.py`**

```python
from oss.sr.gaussian_temporal.gaussian_field import GaussianField, HISTORY_LEN

__all__ = ["GaussianField", "HISTORY_LEN"]
```

- [ ] **Step 6: Run tests** — should now pass.

- [ ] **Step 7: Commit**

```bash
git add oss/sr/gaussian_temporal/__init__.py oss/sr/gaussian_temporal/gaussian_field.py \
        tests/sr/gaussian_temporal/__init__.py tests/sr/gaussian_temporal/test_gaussian_field.py
git commit -m "v5-gaussian(sr): add GaussianField SoA + history container"
```

---

## Task 1: Analytical warp — mean shift + covariance Jacobian

**Goal:** Per-frame analytical warp of the Gaussian field by a motion-vector field. `μ' = μ + flow(μ)` (reuse `oss.gaussian.canvas.warp.warp_positions`). Covariance transform: `Σ' = J Σ Jᵀ` where `J` is the local 2×2 Jacobian of the flow at `μ`.

**Files:**
- Create: `oss/sr/gaussian_temporal/analytical_warp.py`
- Modify: `oss/sr/gaussian_temporal/__init__.py`
- Create: `tests/sr/gaussian_temporal/test_analytical_warp.py`

**Acceptance Criteria:**
- [ ] `warp_field(field, motion, hw) -> GaussianField` returns a new field with shifted means
- [ ] Means are shifted exactly as `oss.gaussian.canvas.warp.warp_positions` returns; out-of-frame Gaussians become `alive=False`
- [ ] Covariance transformation `Σ' = J Σ Jᵀ` matches numerical Jacobian within 1e-3 on a smooth synthetic flow
- [ ] Identity flow → field unchanged within 1e-5 on `mu`, `log_scale`, `rotation`
- [ ] Pure translation flow → covariance unchanged (J=I)
- [ ] Zero-extra-allocation contract: returns a NEW `GaussianField`, original unchanged

**Verify:** `pytest tests/sr/gaussian_temporal/test_analytical_warp.py -v`

**Steps:**

- [ ] **Step 1: Failing test**

```python
"""Analytical Gaussian warp tests."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import GaussianField, warp_field


def _make_field(n: int) -> GaussianField:
    f = GaussianField(capacity=n)
    f.alive[:] = True
    f.mu = torch.tensor([[float(i + 1), float(i + 1)] for i in range(n)])
    f.log_scale = torch.zeros(n, 2)
    f.rotation = torch.zeros(n)
    f.color = torch.rand(n, 3)
    f.opacity = torch.ones(n)
    return f


def test_identity_flow_unchanged() -> None:
    f = _make_field(4)
    motion = torch.zeros(2, 16, 16)
    g = warp_field(f, motion, hw=(16, 16))
    assert torch.allclose(g.mu, f.mu, atol=1e-5)
    assert torch.allclose(g.log_scale, f.log_scale, atol=1e-5)


def test_translation_preserves_covariance() -> None:
    f = _make_field(4)
    f.log_scale = torch.tensor([[0.5, 0.2], [0.3, 0.4], [0.1, 0.1], [0.6, 0.6]])
    motion = torch.zeros(2, 16, 16)
    motion[0] = 1.0   # constant +1 px in x
    motion[1] = -2.0  # constant -2 px in y
    g = warp_field(f, motion, hw=(16, 16))
    # mu shifted; log_scale identical (J = I).
    assert torch.allclose(g.mu, f.mu + torch.tensor([1.0, -2.0]), atol=1e-5)
    assert torch.allclose(g.log_scale, f.log_scale, atol=1e-4)


def test_jacobian_warp_matches_numerical() -> None:
    """Smooth flow (linear gradient in x) → analytic Σ' should match numerical."""
    h, w = 32, 32
    motion = torch.zeros(2, h, w)
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")
    motion[0] = 0.1 * xx   # u(x, y) = 0.1 x  → du/dx = 0.1
    motion[1] = 0.05 * yy  # v(x, y) = 0.05 y → dv/dy = 0.05
    f = _make_field(1)
    f.mu = torch.tensor([[16.0, 16.0]])
    f.log_scale = torch.tensor([[0.0, 0.0]])
    g = warp_field(f, motion, hw=(h, w))
    # J = diag(1.1, 1.05); axis-aligned scales become 1.1 and 1.05.
    expected_log = torch.tensor([[torch.log(torch.tensor(1.1)).item(),
                                  torch.log(torch.tensor(1.05)).item()]])
    assert torch.allclose(g.log_scale, expected_log, atol=5e-3)


def test_out_of_frame_marked_dead() -> None:
    f = _make_field(2)
    f.mu = torch.tensor([[1.0, 1.0], [30.0, 30.0]])
    motion = torch.zeros(2, 16, 16)
    motion[0] = 100.0  # huge x flow
    g = warp_field(f, motion, hw=(16, 16))
    assert g.alive[0].item() is False
```

- [ ] **Step 2: Run** → ImportError.

- [ ] **Step 3: Implement `analytical_warp.py`**

```python
"""Analytical Gaussian warp — μ shift + covariance Jacobian transform.

Reuses ``oss.gaussian.canvas.warp.warp_positions`` for the mean shift +
in-frame mask. Adds a 2×2 Jacobian sample at each Gaussian's mean and
applies ``Σ' = J Σ Jᵀ`` analytically.

Σ is parameterized as scale = exp(log_scale) along axes rotated by ``rotation``.
After warping we re-decompose JΣJᵀ via 2×2 SVD to recover the new (axis-aligned)
log_scale + rotation. Pure PyTorch.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from oss.gaussian.canvas.warp import warp_positions
from oss.sr.gaussian_temporal.gaussian_field import GaussianField


def _sample_jacobian(motion: torch.Tensor, mu: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Per-Gaussian 2x2 Jacobian J = I + ∂(motion)/∂(x,y) at each mean.

    Args:
        motion: (2, H, W).
        mu:     (N, 2).
        hw:     (H, W).

    Returns:
        (N, 2, 2) tensor — J = I + grad(motion).
    """
    n = mu.shape[0]
    h, w = hw
    if n == 0:
        return torch.zeros((0, 2, 2), device=motion.device, dtype=motion.dtype)

    # Finite-difference gradient of motion (same shape as motion).
    # We use forward differences with replicate padding at the borders.
    motion_b = motion.unsqueeze(0)  # (1, 2, H, W)
    pad = F.pad(motion_b, (1, 1, 1, 1), mode="replicate")
    dmdx = (pad[..., 1:-1, 2:] - pad[..., 1:-1, :-2]) * 0.5  # (1, 2, H, W)
    dmdy = (pad[..., 2:, 1:-1] - pad[..., :-2, 1:-1]) * 0.5

    # Sample dmdx, dmdy at each mu via grid_sample.
    x_norm = (mu[:, 0] / w) * 2.0 - 1.0
    y_norm = (mu[:, 1] / h) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, n, 1, 2)
    sx = F.grid_sample(dmdx, grid, mode="bilinear", padding_mode="border", align_corners=False)
    sy = F.grid_sample(dmdy, grid, mode="bilinear", padding_mode="border", align_corners=False)
    sx = sx[0, :, :, 0].t()  # (N, 2)
    sy = sy[0, :, :, 0].t()  # (N, 2)

    # J = I + [[du/dx, du/dy], [dv/dx, dv/dy]]
    j = torch.eye(2, device=motion.device, dtype=motion.dtype).unsqueeze(0).expand(n, -1, -1).clone()
    j[:, 0, 0] += sx[:, 0]
    j[:, 1, 0] += sx[:, 1]
    j[:, 0, 1] += sy[:, 0]
    j[:, 1, 1] += sy[:, 1]
    return j


def _decompose_covariance(field_log_scale: torch.Tensor, field_rotation: torch.Tensor) -> torch.Tensor:
    """Reconstruct Σ from (log_scale, rotation) -> (N, 2, 2)."""
    n = field_log_scale.shape[0]
    s = torch.exp(field_log_scale)  # (N, 2)
    cos = torch.cos(field_rotation)
    sin = torch.sin(field_rotation)
    r = torch.stack([
        torch.stack([cos, -sin], dim=-1),
        torch.stack([sin, cos], dim=-1),
    ], dim=-2)  # (N, 2, 2)
    s_diag = torch.diag_embed(s)  # (N, 2, 2)
    rs = r @ s_diag
    return rs @ rs.transpose(-1, -2)


def _recompose_covariance(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """SVD-decompose 2×2 Σ into (log_scale, rotation)."""
    u, s, _ = torch.linalg.svd(sigma)
    log_scale = 0.5 * torch.log(s.clamp(min=1e-8))  # eigenvalues are scale^2
    rotation = torch.atan2(u[:, 1, 0], u[:, 0, 0])
    return log_scale, rotation


def warp_field(field: GaussianField, motion: torch.Tensor, hw: tuple[int, int]) -> GaussianField:
    """Apply analytical warp to the Gaussian field.

    Args:
        field:  GaussianField to warp (NOT mutated).
        motion: (2, H, W) per-pixel motion vectors (dx, dy).
        hw:     (H, W) of motion field.

    Returns:
        New GaussianField with warped (mu, log_scale, rotation). alive
        flag is ANDed with in-frame mask from ``warp_positions``.
    """
    out = field.clone()
    new_mu, in_frame = warp_positions(field.mu, motion, hw=hw)
    out.mu = new_mu
    out.alive = field.alive & in_frame

    j = _sample_jacobian(motion, field.mu, hw=hw)  # (N, 2, 2)
    sigma = _decompose_covariance(field.log_scale, field.rotation)
    new_sigma = j @ sigma @ j.transpose(-1, -2)
    new_log_scale, new_rotation = _recompose_covariance(new_sigma)
    out.log_scale = new_log_scale
    out.rotation = new_rotation
    return out


__all__ = ["warp_field"]
```

- [ ] **Step 4: Update `__init__.py`** to export `warp_field`.

- [ ] **Step 5: Run tests** — debug until all 4 pass. The Jacobian numerical test has tolerance 5e-3; if it fails marginally, check finite-difference scheme (forward-diff vs central) and align.

- [ ] **Step 6: Commit**

```bash
git add oss/sr/gaussian_temporal/analytical_warp.py oss/sr/gaussian_temporal/__init__.py \
        tests/sr/gaussian_temporal/test_analytical_warp.py
git commit -m "v5-gaussian(sr): add analytical warp — mu shift + covariance Jacobian"
```

---

## Task 2: G-buffer encoder

**Goal:** Tiny CNN converting the 12-channel LR stack to per-tile context features `(B, F, H_lr/T, W_lr/T)` for the transformer. Reuse Image-GS-style narrow blocks from `oss.model.blocks`.

**Files:**
- Create: `oss/sr/gaussian_temporal/g_buffer_encoder.py`
- Modify: `oss/sr/gaussian_temporal/__init__.py`
- Create: `tests/sr/gaussian_temporal/test_g_buffer_encoder.py`

**Acceptance Criteria:**
- [ ] `GBufferEncoder(in_channels=12, feat_dim=64, tile_size=16)` exists
- [ ] Param count ≤ 120K
- [ ] Forward: `(B, 12, H, W) → (B, 64, H/16, W/16)` (tile-level features)
- [ ] Gradient flows from a dummy MSE on the output back to all params

**Verify:** `pytest tests/sr/gaussian_temporal/test_g_buffer_encoder.py -v`

**Steps:**

- [ ] **Step 1: Failing test**

```python
"""Tests for the G-buffer encoder."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import GBufferEncoder


def test_param_count() -> None:
    enc = GBufferEncoder(in_channels=12, feat_dim=64, tile_size=16)
    n = sum(p.numel() for p in enc.parameters())
    assert n <= 120_000, f"GBufferEncoder has {n} params (budget 120_000)"


def test_forward_shape() -> None:
    enc = GBufferEncoder(in_channels=12, feat_dim=64, tile_size=16)
    x = torch.rand(2, 12, 64, 64)
    feats = enc(x)
    assert feats.shape == (2, 64, 4, 4)


def test_grad_flow() -> None:
    enc = GBufferEncoder(in_channels=12, feat_dim=64, tile_size=16)
    x = torch.rand(1, 12, 32, 32, requires_grad=True)
    feats = enc(x)
    feats.mean().backward()
    for p in enc.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
```

- [ ] **Step 2: Implement** in `g_buffer_encoder.py`. Use `Conv2d → ReLU` stacks with strided convs to reach `tile_size` downsample. Don't bring in the full Sprint-4 U-Net — this is a flat encoder.

```python
"""Per-tile G-buffer encoder for the v5 Gaussian temporal track."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GBufferEncoder(nn.Module):
    def __init__(self, in_channels: int = 12, feat_dim: int = 64, tile_size: int = 16) -> None:
        super().__init__()
        if tile_size & (tile_size - 1) != 0:
            raise ValueError(f"tile_size must be a power of 2; got {tile_size}")
        self.tile_size = tile_size
        # log2(tile_size) stride-2 conv blocks.
        n_down = int(tile_size).bit_length() - 1
        widths = [16, 24, 32, max(48, feat_dim)][: n_down]
        widths[-1] = feat_dim  # final width matches feat_dim

        layers: list[nn.Module] = []
        prev = in_channels
        for w in widths:
            layers.append(nn.Conv2d(prev, w, 3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            prev = w
        # One mixing conv at output resolution.
        layers.append(nn.Conv2d(prev, feat_dim, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["GBufferEncoder"]
```

- [ ] **Step 3: Update `__init__.py`** to export `GBufferEncoder`.

- [ ] **Step 4: Run** + commit:

```bash
git add oss/sr/gaussian_temporal/g_buffer_encoder.py oss/sr/gaussian_temporal/__init__.py \
        tests/sr/gaussian_temporal/test_g_buffer_encoder.py
git commit -m "v5-gaussian(sr): add tile-level G-buffer encoder"
```

---

## Task 3: Multi-frame transformer over Gaussian tokens

**Goal:** Transformer that attends over `(current G-buffer tile features) ⊕ (Gaussians from t..t-N)` and produces per-Gaussian updates `(Δμ, Δlog_scale, Δrotation, Δcolor)`. Rotary positional embedding keyed on Gaussian μ.

**Files:**
- Create: `oss/sr/gaussian_temporal/transformer.py`
- Modify: `oss/sr/gaussian_temporal/__init__.py`
- Create: `tests/sr/gaussian_temporal/test_transformer.py`

**Acceptance Criteria:**
- [ ] `GaussianMultiFrameTransformer(d_model=128, n_heads=4, n_layers=4, history_len=5)` exists
- [ ] Param count: 400K–600K
- [ ] Forward: `(field_curr, history_fields[:N], tile_features) -> updates dict {dmu, dlog_scale, drot, dcolor}` each shaped `(N_alive, *)` aligned with current alive mask
- [ ] Permutation equivariance: shuffling Gaussian token order produces identically-shuffled outputs (within 1e-4)
- [ ] RoPE positional encoding is keyed on Gaussian μ (not token index)
- [ ] Gradient flows to encoder feats AND to a dummy nn.Parameter wrapped over `field.color`

**Verify:** `pytest tests/sr/gaussian_temporal/test_transformer.py -v`

**Steps:**

- [ ] **Step 1: Failing test (sketch)**

```python
"""Tests for GaussianMultiFrameTransformer."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import GaussianField, GaussianMultiFrameTransformer


def _live_field(n: int) -> GaussianField:
    f = GaussianField(capacity=n)
    f.alive[:] = True
    f.mu = torch.rand(n, 2) * 16.0
    f.log_scale = torch.zeros(n, 2)
    f.rotation = torch.zeros(n)
    f.color = torch.rand(n, 3)
    f.opacity = torch.ones(n)
    return f


def test_param_budget() -> None:
    t = GaussianMultiFrameTransformer(d_model=128, n_heads=4, n_layers=4, history_len=5)
    n = sum(p.numel() for p in t.parameters())
    assert 400_000 <= n <= 600_000, f"transformer param count {n} out of budget"


def test_forward_keys_and_shapes() -> None:
    t = GaussianMultiFrameTransformer(d_model=128, n_heads=4, n_layers=2, history_len=2)
    f_curr = _live_field(8)
    history = [_live_field(8), _live_field(8)]
    feats = torch.rand(1, 128, 4, 4)
    upd = t(field_curr=f_curr, history=history, tile_features=feats)
    assert set(upd.keys()) == {"dmu", "dlog_scale", "drot", "dcolor"}
    assert upd["dmu"].shape == (8, 2)
    assert upd["dlog_scale"].shape == (8, 2)
    assert upd["drot"].shape == (8,)
    assert upd["dcolor"].shape == (8, 3)


def test_permutation_equivariance() -> None:
    torch.manual_seed(0)
    t = GaussianMultiFrameTransformer(d_model=64, n_heads=2, n_layers=2, history_len=1)
    f = _live_field(8)
    feats = torch.rand(1, 64, 2, 2)
    history = [_live_field(8)]
    upd_a = t(field_curr=f, history=history, tile_features=feats)["dmu"].detach()

    perm = torch.randperm(8)
    f2 = _live_field(8)
    f2.mu = f.mu[perm].clone()
    f2.color = f.color[perm].clone()
    upd_b = t(field_curr=f2, history=history, tile_features=feats)["dmu"].detach()
    assert torch.allclose(upd_a[perm], upd_b, atol=1e-4)
```

- [ ] **Step 2: Implement** transformer in `transformer.py`. Standard `nn.MultiheadAttention` stack with pre-norm. RoPE applied on the (q, k) of the attention layer for Gaussian tokens, with `mu / hw` as the position. Tile features become tokens too (positions = tile centers). Output linear heads give `(dmu, dlog_scale, drot, dcolor)`. Use `dropout=0.0`. Note: PyTorch `nn.MultiheadAttention` is order-equivariant by default — do NOT add learned positional embeddings; keep RoPE only.

  Implementation budget guidance: `d_model=128`, 4 layers, 4 heads, ffn 256 → ~500K params. Tune the FFN width if the count is out of range.

- [ ] **Step 3: Update `__init__.py`** with `GaussianMultiFrameTransformer`.

- [ ] **Step 4: Run + iterate. Commit.**

```bash
git add oss/sr/gaussian_temporal/transformer.py oss/sr/gaussian_temporal/__init__.py \
        tests/sr/gaussian_temporal/test_transformer.py
git commit -m "v5-gaussian(sr): add multi-frame transformer with RoPE on Gaussian mu"
```

---

## Task 4: Densification (residual-driven, heuristic for v5)

**Goal:** Identify high-residual tiles after warp+update, spawn 1–2 new Gaussians per tile (initial mean = tile center, log_scale=0, rotation=0, color=tile mean, opacity=0.5). Heuristic threshold for v5; soft top-K is post-v5.

**Files:**
- Create: `oss/sr/gaussian_temporal/densification.py`
- Create: `tests/sr/gaussian_temporal/test_densification.py`

**Acceptance Criteria:**
- [ ] `densify(field, lr_target, rendered, tile_size, residual_threshold, max_new) -> field` returns the field with new Gaussians inserted into the first `max_new` free slots
- [ ] Spawn picks tiles where `tile_residual_mean > residual_threshold`
- [ ] Inserted color matches tile mean within 1e-3 on a synthetic uniform-color tile
- [ ] `field.count_alive()` strictly increases when residuals exceed threshold
- [ ] Gradient flows through inserted Gaussians' color (since color = lr_target's tile mean), not through positions (positions are detached — heuristic insertion)

**Verify:** `pytest tests/sr/gaussian_temporal/test_densification.py -v`

**Steps:**

- [ ] **Step 1: Failing test**

```python
"""Densification tests."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import GaussianField, densify


def test_no_spawn_below_threshold() -> None:
    f = GaussianField(capacity=8)
    lr_target = torch.zeros(1, 3, 16, 16)
    rendered = torch.zeros(1, 3, 16, 16)  # zero residual
    g = densify(f, lr_target=lr_target, rendered=rendered,
                tile_size=8, residual_threshold=0.01, max_new=4)
    assert g.count_alive() == 0


def test_spawn_inserts_into_free_slots() -> None:
    f = GaussianField(capacity=8)
    lr_target = torch.full((1, 3, 16, 16), 0.5)
    rendered = torch.zeros(1, 3, 16, 16)  # residual=0.5 everywhere → exceed threshold
    g = densify(f, lr_target=lr_target, rendered=rendered,
                tile_size=8, residual_threshold=0.01, max_new=2)
    assert g.count_alive() == 2
    # Inserted color matches tile mean.
    inserted_idx = g.alive.nonzero(as_tuple=True)[0]
    assert torch.allclose(g.color[inserted_idx], torch.full((2, 3), 0.5), atol=1e-3)


def test_color_grad_flows() -> None:
    f = GaussianField(capacity=4)
    lr_target = torch.full((1, 3, 8, 8), 0.5, requires_grad=True)
    rendered = torch.zeros(1, 3, 8, 8)
    g = densify(f, lr_target=lr_target, rendered=rendered,
                tile_size=4, residual_threshold=0.01, max_new=4)
    g.color.sum().backward()
    assert lr_target.grad is not None
```

- [ ] **Step 2: Implement** `densification.py`:

```python
"""Heuristic residual-driven densification for v5 Gaussian-temporal.

Soft top-K (Gumbel-Softmax) is post-v5 per spec §risks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from oss.sr.gaussian_temporal.gaussian_field import GaussianField


def densify(
    field: GaussianField,
    lr_target: torch.Tensor,
    rendered: torch.Tensor,
    tile_size: int,
    residual_threshold: float,
    max_new: int,
) -> GaussianField:
    """Spawn new Gaussians at high-residual tile centers.

    Color is the tile mean of ``lr_target`` (gradient flows back to lr_target);
    position/scale/rotation/opacity are detached scalar inits (no gradient).
    """
    if lr_target.shape != rendered.shape:
        raise ValueError("lr_target and rendered must have same shape")
    b, c, h, w = lr_target.shape
    if b != 1:
        # GaussianField is per-sample state; batched fields are per-sample
        # lists, not stacked tensors. Densify must run per item.
        raise ValueError(f"densify expects B=1; got {b}. Loop in caller for batches.")
    tiles_h, tiles_w = h // tile_size, w // tile_size

    residual = (lr_target - rendered).abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
    pooled = F.avg_pool2d(residual, kernel_size=tile_size, stride=tile_size)  # (B, 1, tH, tW)
    flat = pooled.view(-1)
    above = (flat > residual_threshold).nonzero(as_tuple=True)[0]
    if above.numel() == 0:
        return field
    if above.numel() > max_new:
        # Take the top-K by residual magnitude.
        scores = flat[above]
        topk = torch.topk(scores, k=max_new).indices
        above = above[topk]

    out = field.clone()
    free_slots = (~out.alive).nonzero(as_tuple=True)[0]
    n_to_insert = min(above.numel(), free_slots.numel())
    if n_to_insert == 0:
        return out
    target_slots = free_slots[:n_to_insert]
    chosen = above[:n_to_insert]

    tile_y = chosen // tiles_w
    tile_x = chosen % tiles_w
    cx = (tile_x.float() + 0.5) * tile_size
    cy = (tile_y.float() + 0.5) * tile_size

    # Tile mean color (gradient flows here).
    pooled_color = F.avg_pool2d(lr_target, kernel_size=tile_size, stride=tile_size)  # (B, 3, tH, tW)
    pooled_color_flat = pooled_color.permute(0, 2, 3, 1).reshape(-1, 3)  # (B*tH*tW, 3)
    inserted_color = pooled_color_flat[chosen]

    # Insertion (detached scalar inits, except color).
    out.mu = out.mu.clone()
    out.log_scale = out.log_scale.clone()
    out.rotation = out.rotation.clone()
    out.opacity = out.opacity.clone()
    out.alive = out.alive.clone()
    out.color = out.color.clone()

    out.mu[target_slots, 0] = cx.detach()
    out.mu[target_slots, 1] = cy.detach()
    out.log_scale[target_slots] = 0.0
    out.rotation[target_slots] = 0.0
    out.opacity[target_slots] = 0.5
    out.color[target_slots] = inserted_color  # gradient flows to lr_target via this assignment
    out.alive[target_slots] = True
    return out


__all__ = ["densify"]
```

- [ ] **Step 3: Run + commit.**

```bash
git add oss/sr/gaussian_temporal/densification.py oss/sr/gaussian_temporal/__init__.py \
        tests/sr/gaussian_temporal/test_densification.py
git commit -m "v5-gaussian(sr): add residual-driven densification (heuristic v5)"
```

---

## Task 5: Pruning

**Goal:** Drop Gaussians whose opacity falls below a threshold; enforce a hard `max_count` cap by lowest-opacity-first eviction.

**Files:**
- Create: `oss/sr/gaussian_temporal/pruning.py`
- Create: `tests/sr/gaussian_temporal/test_pruning.py`

**Acceptance Criteria:**
- [ ] `prune(field, opacity_threshold, max_count)` returns a field where `count_alive() ≤ max_count`
- [ ] Gaussians with `opacity < opacity_threshold` become `alive=False`
- [ ] When `count_alive() > max_count`, the lowest-opacity Gaussians are evicted
- [ ] Operation is non-differentiable (post-loss); document this clearly in the docstring

**Verify:** `pytest tests/sr/gaussian_temporal/test_pruning.py -v`

**Steps:** TDD as above. Implementation:

```python
"""Opacity-threshold + count-cap pruning. Non-differentiable; call after loss."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal.gaussian_field import GaussianField


@torch.no_grad()
def prune(field: GaussianField, opacity_threshold: float, max_count: int) -> GaussianField:
    out = field.clone()
    low = (out.opacity < opacity_threshold) & out.alive
    out.alive[low] = False
    n_alive = int(out.alive.sum().item())
    if n_alive > max_count:
        live_idx = out.alive.nonzero(as_tuple=True)[0]
        ranked = live_idx[torch.argsort(out.opacity[live_idx])]
        evict = ranked[: n_alive - max_count]
        out.alive[evict] = False
    return out


__all__ = ["prune"]
```

Then commit:

```bash
git add oss/sr/gaussian_temporal/pruning.py oss/sr/gaussian_temporal/__init__.py \
        tests/sr/gaussian_temporal/test_pruning.py
git commit -m "v5-gaussian(sr): add opacity + count pruning"
```

---

## Task 6: Rasterizer wrapper

**Goal:** Convert a `GaussianField` to a `GaussianBatch` and call the existing `oss.gaussian.renderer.Rasterizer` to produce HR output. Single thin wrapper, no math.

**Files:**
- Create: `oss/sr/gaussian_temporal/rasterizer.py`
- Create: `tests/sr/gaussian_temporal/test_rasterizer_wrapper.py`

**Acceptance Criteria:**
- [ ] `render_field(field, output_hw)` returns `(B=1, 3, H, W)` HR output
- [ ] Only alive Gaussians contribute (dead rows masked out)
- [ ] Output shape parity with `Rasterizer` smoke test
- [ ] Gradient flows through `field.color` to `output.mean()`

**Verify:** `pytest tests/sr/gaussian_temporal/test_rasterizer_wrapper.py -v`

**Steps:** Read `oss/gaussian/renderer/__init__.py` for `Rasterizer` and `GaussianBatch` API; mirror the call. The wrapper:

```python
import torch

from oss.gaussian.renderer import GaussianBatch, Rasterizer
from oss.sr.gaussian_temporal.gaussian_field import GaussianField

_RASTERIZER = Rasterizer()  # module-level singleton; expensive backend select runs once

def render_field(field: GaussianField, output_hw: tuple[int, int]) -> torch.Tensor:
    """Render a GaussianField at output_hw resolution.

    Returns (1, F, H, W). F = feat_dim of GaussianBatch.feat; for v5 we use F=3 (RGB).
    Opacity is multiplied into ``feat`` per-Gaussian so the renderer's alpha-blend
    sees the right contribution. (No separate opacities arg in the existing API.)
    """
    alive = field.alive
    n = int(alive.sum().item())
    if n == 0:
        # Empty field renders to zeros — still a valid tensor for downstream loss.
        h, w = output_hw
        return torch.zeros(1, 3, h, w, device=field.mu.device, dtype=field.mu.dtype)
    feat = field.color[alive] * field.opacity[alive].unsqueeze(-1)  # (N, 3) — opacity baked in
    batch = GaussianBatch(
        xy=field.mu[alive],
        scale=torch.exp(field.log_scale[alive]),
        rot=field.rotation[alive],
        feat=feat,
    )
    out = _RASTERIZER(batch, output_hw=output_hw)  # (3, H, W)
    return out.unsqueeze(0)
```

Confirmed API in `oss/gaussian/renderer/rasterizer.py`: `GaussianBatch(xy, scale, rot, feat)` (line 60) and `Rasterizer.__call__(gaussians, output_hw)` (line 127). No `opacities` field — opacity is multiplied into `feat`.

Commit:

```bash
git add oss/sr/gaussian_temporal/rasterizer.py oss/sr/gaussian_temporal/__init__.py \
        tests/sr/gaussian_temporal/test_rasterizer_wrapper.py
git commit -m "v5-gaussian(sr): add rasterizer wrapper around existing renderer"
```

---

## Task 7: Regularization losses (drift / area / count)

**Goal:** Composite regularization term `L_gaussian_reg = w_pos · ||μ_drift||₂ + w_cov · max(0, det(Σ) − max_area) + w_count · max(0, count − max_count)`.

**Files:**
- Create: `oss/sr/gaussian_temporal/regularization.py`
- Create: `tests/sr/gaussian_temporal/test_regularization.py`

**Acceptance Criteria:**
- [ ] `gaussian_regularization_loss(field_t, field_t_minus_1, max_area, max_count, weights) -> scalar`
- [ ] Returns 0.0 when input field equals prev field, det(Σ) ≤ max_area, count ≤ max_count
- [ ] Drift term grows linearly with `||μ_t − μ_{t-1}||`
- [ ] Area term is hinged at `max_area`
- [ ] Count term is hinged at `max_count`
- [ ] Gradient flows to `field_t.mu`, `field_t.log_scale`

**Verify:** `pytest tests/sr/gaussian_temporal/test_regularization.py -v`

Implementation note: gradient must NOT flow into `field_t_minus_1` (detach it).

Commit:

```bash
git add oss/sr/gaussian_temporal/regularization.py oss/sr/gaussian_temporal/__init__.py \
        tests/sr/gaussian_temporal/test_regularization.py
git commit -m "v5-gaussian(sr): add Gaussian regularization (drift + area + count)"
```

---

## Task 8: GaussianTemporalSRModel — full pipeline

**Goal:** Wire encoder + warp + transformer + densify + prune + raster into a single `nn.Module`. Forward consumes `(lr_inputs, motion_lr, prev_field)` and returns `(rendered_hr, new_field, debug_dict)`.

**Files:**
- Create: `oss/sr/gaussian_temporal/model.py`
- Create: `tests/sr/gaussian_temporal/test_model_full_step.py`

**Acceptance Criteria:**
- [ ] `GaussianTemporalSRModel(in_channels=12, scale=2, max_count=16384)` constructs
- [ ] Forward signature exactly: `(lr_inputs: (B,12,h,w), motion_lr: (B,2,h,w), prev_field: GaussianField | None) -> (out_hr, new_field, debug)`
- [ ] On `prev_field=None` (first frame): the encoder + densification produce an initial Gaussian set; `count_alive() > 0` AND the returned `rendered_hr` has `abs().max() > 0` (must be a real image, not the pre-densification zero render)
- [ ] On synthetic moving-rectangle 2-frame sequence: full forward, full loss, `loss.backward()` produces finite gradients
- [ ] `new_field.alive` is consistent (no NaN, no negative count)

**Verify:** `pytest tests/sr/gaussian_temporal/test_model_full_step.py -v`

**Steps:**

- [ ] Implementation pseudocode (write the full module body):

```python
class GaussianTemporalSRModel(nn.Module):
    def __init__(self, in_channels=12, scale=2, max_count=16384):
        super().__init__()
        self.scale = scale
        self.max_count = max_count
        self.encoder = GBufferEncoder(in_channels=in_channels, feat_dim=128, tile_size=16)
        self.transformer = GaussianMultiFrameTransformer(
            d_model=128, n_heads=4, n_layers=4, history_len=5,
        )
        self.densify_threshold = 0.05
        self.densify_max_new = 256
        self.opacity_threshold = 0.05

    def forward(self, lr_inputs, motion_lr, prev_field):
        b, _, h_lr, w_lr = lr_inputs.shape
        h_hr, w_hr = h_lr * self.scale, w_lr * self.scale
        if b != 1:
            raise ValueError(f"GaussianTemporalSRModel expects B=1; got {b}.")

        feats = self.encoder(lr_inputs)               # (1, 128, h/16, w/16)

        # ---- First-frame seed -------------------------------------------------
        if prev_field is None:
            # Empty field; seed via densification so count_alive > 0 at frame 0.
            # Target = bilinear-upscale of LR RGB; baseline rendered = zeros.
            warped = GaussianField(capacity=self.max_count, device=lr_inputs.device)
            lr_rgb = lr_inputs[:, :3]
            target_hr = F.interpolate(lr_rgb, size=(h_hr, w_hr), mode="bilinear", align_corners=False)
            zero_render = torch.zeros_like(target_hr)
            warped = densify(
                warped, lr_target=target_hr, rendered=zero_render,
                tile_size=self.scale * 16,  # match encoder tile size at HR
                residual_threshold=0.0, max_new=self.initial_seed_count,
            )
            history = []
        else:
            warped = warp_field(prev_field, motion_lr[0], hw=(h_lr, w_lr))
            history = prev_field.history

        # ---- Transformer update over alive tokens -----------------------------
        if warped.count_alive() > 0:
            updates = self.transformer(field_curr=warped, history=history, tile_features=feats)
            alive_idx = warped.alive.nonzero(as_tuple=True)[0]
            warped.mu[alive_idx] = warped.mu[alive_idx] + updates["dmu"]
            warped.log_scale[alive_idx] = warped.log_scale[alive_idx] + updates["dlog_scale"]
            warped.rotation[alive_idx] = warped.rotation[alive_idx] + updates["drot"]
            warped.color[alive_idx] = warped.color[alive_idx] + updates["dcolor"]

        # ---- First render -----------------------------------------------------
        rendered_hr = render_field(warped, output_hw=(h_hr, w_hr))

        # ---- Densify on residual vs LR-upsampled-target -----------------------
        # Match Phase 1+2+3 spec — residual densification active in the model.
        # Train loop can also do an additional pass against GT HR if desired.
        lr_rgb = lr_inputs[:, :3]
        target_hr = F.interpolate(lr_rgb, size=(h_hr, w_hr), mode="bilinear", align_corners=False)
        warped = densify(
            warped, lr_target=target_hr, rendered=rendered_hr,
            tile_size=self.scale * 16, residual_threshold=self.densify_threshold,
            max_new=self.densify_max_new,
        )

        # ---- Re-render after densification so frame-0 (and any frame where
        # densification adds Gaussians) does NOT return the pre-densify image.
        # Gradient still flows: render_field is differentiable through field.color.
        rendered_hr = render_field(warped, output_hw=(h_hr, w_hr))

        # ---- Prune ------------------------------------------------------------
        new_field = prune(warped, opacity_threshold=self.opacity_threshold,
                          max_count=self.max_count)

        debug = {"count_alive": int(new_field.count_alive())}
        return rendered_hr, new_field, debug
```

Class-level constants needed (add to `__init__`):
- `self.initial_seed_count = 4096`
- `self.densify_threshold = 0.05`
- `self.densify_max_new = 256`
- `self.opacity_threshold = 0.05`

`F` is `torch.nn.functional`.

- [ ] Test: 2-frame synthetic moving rectangle, compute `L1 + 0.05 · temporal_consistency + 0.01 · regularization`, `.backward()`, assert finite grads on `model.encoder`, `model.transformer`, and the rendered output.

- [ ] Commit:

```bash
git add oss/sr/gaussian_temporal/model.py oss/sr/gaussian_temporal/__init__.py \
        tests/sr/gaussian_temporal/test_model_full_step.py
git commit -m "v5-gaussian(sr): wire full GaussianTemporalSRModel pipeline"
```

---

## Task 9: Trajectory window dataset

**Goal:** Wrap `TartanAirGaussianDataset` / `SintelGaussianDataset` to emit a list of N consecutive frames per `__getitem__` (default N=5). Reject windows that cross trajectory boundaries.

**Files:**
- Create: `oss/sr/gaussian_temporal/dataset.py`
- Create: `tests/sr/gaussian_temporal/test_dataset.py`

**Acceptance Criteria:**
- [ ] `TrajectoryWindowDataset(base, window=5)` exists
- [ ] `__getitem__(idx)` returns `{"frames": [example_0, ..., example_{N-1}], "trajectory_key": str}`
- [ ] All N frames share the same trajectory key
- [ ] On a synthetic 8-frame fake base (5 in seq A, 3 in seq B), with window=5: returns exactly 1 window (only seq A has ≥5 frames)
- [ ] Default collate stacks each frame field across batches (`out["frames"][i]["lr"]` is `(B, 3, H, W)`)
- [ ] Reuses `oss.sr.temporal.dataset.adapt_tartanair` / `adapt_sintel` so the boundary key resolves correctly

**Verify:** `pytest tests/sr/gaussian_temporal/test_dataset.py -v`

Steps + commit follow the same TDD pattern.

```bash
git add oss/sr/gaussian_temporal/dataset.py oss/sr/gaussian_temporal/__init__.py \
        tests/sr/gaussian_temporal/test_dataset.py
git commit -m "v5-gaussian(sr): add multi-frame trajectory window dataset"
```

**Renderer API contract (read before Task 6 implementation):**
- `oss.gaussian.renderer.GaussianBatch(xy: (N,2), scale: (N,2), rot: (N,), feat: (N,F))` — no `opacities` field; opacity is folded into `feat` channels or modeled via `scale`
- `oss.gaussian.renderer.Rasterizer()(gaussians: GaussianBatch, output_hw: (H, W)) -> (F, H, W)`
- Confirmed against `oss/gaussian/renderer/rasterizer.py` line 60 (dataclass fields) + line 127 (`__call__` signature)
- `Rasterizer.__init__` takes optional `tile_size, topk_norm, force_backend` — NOT `output_hw`

---

## Task 10: Stateful inference engine

**Goal:** `GaussianTemporalSRInferenceEngine` carrying `GaussianField` across calls, with reset on scene cut.

**Files:**
- Modify: `oss/sr/inference.py` (append; do not break existing engines)
- Create: `tests/sr/gaussian_temporal/test_inference_state.py`

**Acceptance Criteria:** Same shape contract as `TemporalSRInferenceEngine` from the pixel plan, but the carried state is a `GaussianField`. `reset()` re-initializes to empty field. Scene cut threshold is the same configurable knob.

Tests + commit standard.

```bash
git add oss/sr/inference.py tests/sr/gaussian_temporal/test_inference_state.py
git commit -m "v5-gaussian(sr): add stateful GaussianTemporalSRInferenceEngine"
```

---

## Task 11: Training script `scripts/sr_train_gaussian_temporal.py`

**Goal:** 4-phase training entry. Smoke mode runs end-to-end on a synthetic moving-rectangle on CPU.

**Files:**
- Create: `scripts/sr_train_gaussian_temporal.py`

**Acceptance Criteria:**
- [ ] `python scripts/sr_train_gaussian_temporal.py --smoke --device cpu --max-steps 5` exits 0 with a checkpoint and `metrics.json` written
- [ ] Phases:
  - Phase 1 (steps 0..20K): single-frame fitter (no temporal, no transformer); just encoder + initial densification + raster
  - Phase 2 (20K..50K): add warped prev-field + 2-layer transformer warmup; encoder frozen
  - Phase 3 (50K..120K): unfreeze encoder, full 4-layer transformer, full loss including temporal consistency + regularization, densification active
  - Phase 4 (120K..140K): Sintel-only fine-tune at LR×0.01
- [ ] Auto-resume + dashboard-compatible metrics dump (mirror `oss/gaussian/train/train.py`)
- [ ] Trajectory-window data loader hooked up; `BPTT detach`: each step's `prev_field` is detached before being fed to next step (training-graph length = 1 frame; consistency-loss provides the only gradient across t and t+1)

**Verify:**
```
pytest tests/sr/gaussian_temporal/test_train_smoke.py -v
```
where the test mirrors the pixel plan's `test_smoke_train`.

Steps + commit standard.

```bash
git add scripts/sr_train_gaussian_temporal.py tests/sr/gaussian_temporal/test_train_smoke.py
git commit -m "v5-gaussian(sr): add training entry with 4-phase schedule + smoke test"
```

---

## Task 12: Held-out eval script + memo template

**Goal:** Score v5-gaussian-temporal vs v5-pixel-temporal vs v4-baseline on the same fixed batch as Sprint 5's pixel held-out eval.

**Files:**
- Create: `scripts/sr_gaussian_temporal_held_out.py`
- Create: `docs/superpowers/experiments/2026-XX-XX-v5-gaussian-temporal-held-out-template.md`

**Acceptance Criteria:**
- [ ] Reports PSNR + LPIPS + temporal stability for all three models on the same deterministic batch
- [ ] Reports per-sample win counts: `gaussian > pixel`, `gaussian > baseline`, `pixel > baseline`
- [ ] Writes `held_out_results.json` next to the checkpoint
- [ ] Template lists the four success-criteria gates from the spec (PSNR, LPIPS, temporal stability, latency)

**Verify:** `python scripts/sr_gaussian_temporal_held_out.py --help` exits 0.

Memo template body parallels the pixel template; the conclusion section explicitly notes the **race rule**: "Gaussian must explicitly beat pixel; tie ≠ Gaussian win."

```bash
git add scripts/sr_gaussian_temporal_held_out.py \
        docs/superpowers/experiments/2026-XX-XX-v5-gaussian-temporal-held-out-template.md
git commit -m "v5-gaussian(sr): add held-out eval + memo template"
```

---

## Task 13: Lab-notebook memo + remote runbook

**Goal:** Pre-train memo (lab-notebook discipline) and remote launch runbook, mirroring the pixel plan's Task 9.

**Files:**
- Create: `docs/superpowers/experiments/2026-05-04-v5-gaussian-temporal-train-start.md`
- Create: `docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md`

**Acceptance Criteria:**
- [ ] Memo records hypothesis, success criteria, schedule, expected runtime (24–48 h), checkpoint path, dataset locations, WMI orphan-spawn launch command
- [ ] Runbook is literal copy-pasteable for <train-host>
- [ ] **Schedule note:** explicit GPU-share decision — Gaussian train begins after pixel train completes (or after pixel reaches a stable Phase-2 milestone, by Cash's call)

```bash
git add docs/superpowers/experiments/2026-05-04-v5-gaussian-temporal-train-start.md \
        docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md
git commit -m "v5-gaussian(sr): lab-notebook train-start memo + remote runbook"
```

---

## Task 14: Sprint-5 closeout + ship decision

**Goal:** After both tracks complete training and held-out eval, write the comparison memo and execute the ship decision per the spec.

**Files:**
- Create: `docs/superpowers/experiments/2026-XX-XX-v5-pixel-vs-gaussian-comparison.md`
- Modify: `README.md` — sprint table + Sprint 5 narrative

**Acceptance Criteria:**
- [ ] Comparison memo includes both held-out result tables side by side, the success-criteria evaluation per the Gaussian spec ("PSNR ≥ pixel − 0.3 dB, LPIPS ≤ pixel − 0.01, temporal ≤ pixel, latency ≤ 1.5× pixel"), and an explicit ship decision: "ship pixel" / "ship Gaussian" / "neither passes; iterate"
- [ ] If pixel ships: README S5 row marks pixel-shipped, Gaussian deferred to v6
- [ ] If Gaussian ships: README updates to reflect Gaussian as v5; pixel becomes parallel research input
- [ ] If neither: explicit blocker memo + next-step plan; do NOT mark S5 done

**Verify:** Manual review by Cash. No merge to `main` until this gate is signed off.

```bash
git add docs/superpowers/experiments/2026-XX-XX-v5-pixel-vs-gaussian-comparison.md README.md
git commit -m "sprint5(sr): closeout — pixel vs gaussian comparison + ship decision"
```

---

## Self-review notes (run by the planner before handoff)

**Spec coverage:**
- Inputs (12-ch + persistent Gaussian field) — Task 0 + Task 8. ✓
- G-buffer encoder — Task 2. ✓
- Analytical Gaussian warp (μ + Σ Jacobian) — Task 1. ✓
- Multi-frame transformer with RoPE — Task 3. ✓
- Densification head — Task 4. ✓
- Pruning — Task 5. ✓
- Differentiable rasterizer — Task 6 (wraps existing renderer). ✓
- Loss: appearance + temporal-consistency + regularization — Task 8 + Task 11. ✓
- 4-phase schedule — Task 11. ✓
- Multi-frame trajectory window dataset — Task 9. ✓
- Inference state — Task 10. ✓
- Held-out eval gates — Task 12 + Task 14. ✓

**Placeholder scan:** Task 3 transformer body and Task 11 train loop intentionally describe structure rather than full code (transformer body is well-trodden territory; the train loop mirrors `oss/gaussian/train/train.py`). The implementer for those tasks must write the full code; review checks for full code, not the placeholder pseudocode here. Flag for reviewer.

**Type consistency:** `GaussianField` SoA fields (`mu`, `log_scale`, `rotation`, `color`, `opacity`, `alive`) are referenced consistently across all tasks. `warp_field`, `densify`, `prune`, `render_field`, `gaussian_regularization_loss` signatures are consistent. ✓

**Risks:**
- Gaussian count fluctuating each step makes batched training delicate. Mitigation: hard count cap + per-batch independent fields (no batched Gaussian set).
- Transformer permutation equivariance test may catch subtle bugs in RoPE keyed on μ — debug carefully when it fires.
- The existing renderer signature must be confirmed in Task 6 by reading `oss/gaussian/renderer/__init__.py`; the wrapper code I sketched is illustrative, not authoritative.

**Out-of-scope (per spec, NOT in this plan):**
- 4D Gaussian Splatting (true 3D temporal)
- View-dependent effects
- Cross-attention between Gaussians and pixel features (v6+)
- INT8 quantization
- Soft top-K densification (heuristic 3DGS-style only for v5)
