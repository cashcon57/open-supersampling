# v6.2-Pico-002 Implementation Plan — File-by-File Code Changes

**Date:** 2026-05-08
**Parent spec:** `docs/architecture/2026-05-08-v62-arch-v4-spec.md`
**Companion memo:** `docs/research/2026-05-08-fsr4-architecture-observations.md`
**Goal:** Translate the architecture v4 commitment into actionable, dispatchable code-change tasks. Each task has a clear acceptance gate. Multiple tasks are **independent and can be implemented in parallel** by separate codex agents.

---

## Sequencing

The dependency graph allows broad parallelism. Tasks marked `[parallel]` are independent of each other and can run concurrently. `[depends:X]` means task must wait for X to land first.

| Task | Parallelism | Est. effort |
|---|---|---|
| T1 — concat-fusion module replacing global cross-attn | `[parallel]` | 0.5-1 day |
| T2 — disocclusion-only spawner with DGP dictionary | `[parallel]` | 1-1.5 days |
| T3 — Kalman 6-FLOP update for existing Gaussians | `[parallel]` | 0.5 day |
| T4 — R=4-8 latent splat rasterizer (forward + backward) | `[parallel]` | 1.5-2 days |
| T5 — student backbone scaffold (FasterNet + SqrSwish) | `[parallel]` | 1 day |
| T6 — V6.2 model orchestrator wiring | `[depends:T1,T2,T4]` | 0.5 day |
| T7 — training script v6.2 entry + flag set | `[depends:T6]` | 0.5 day |
| T8 — config preset for pico-002 launch | `[depends:T7]` | 0.25 day |
| T9 — launch pico-002 on 3080 Ti | `[depends:T8]` | trivial |

**Critical path:** T1 || T2 || T4 → T6 → T7 → T8 → T9 = ~3 days walltime if T1/T2/T4 truly run in parallel via codex subagents.

T3 (Kalman) and T5 (student) are non-blocking — they can land any time before pico-002.1 retraining for distillation. v6.2-pico-002 can start training without T3/T5; quality gates apply at the v6.2-pico-002.1 distillation phase.

---

## T1 — Concat-fusion module replacing global cross-attention

**File:** `oss/sr/v6/concat_fusion.py` (new) + `oss/sr/v6/cross_attention.py` (modified to keep local top-K path only)

**Per spec (`v62-arch-v4-spec.md` §3.2 step 6):**

```
F'(p) = F(p) + ψ_θ([F(p), G(p), m(p), I_base(p), depth(p), MV(p)])
ψ_θ = 1×1 → depthwise 3×3 → 1×1
```

Where:

- `F(p)`: HAT pixel features at HR (current cross-attn input)
- `G(p) = (Σ_g w_g · z_g) / (ε + Σ_g w_g)`: rasterizer canvas readout (R=4 latent)
- `m(p) = Σ_g w_g`: weight sum per pixel
- `I_base(p)`: reproject base
- `depth(p)`, `MV(p)`: G-buffer

**Concrete changes:**

1. New `ConcatFusion` module (`oss/sr/v6/concat_fusion.py`):

```python
class ConcatFusion(nn.Module):
    """Replaces global pixel↔Gaussian cross-attention.

    Inputs at HR resolution:
      F:      (B, F_pixel, H, W)  HAT pixel features
      G:      (B, R, H, W)        rasterized canvas readout (R=4 default)
      m:      (B, 1, H, W)        sum of Gaussian weights per pixel
      I_base: (B, 3, H, W)        reprojection base
      depth:  (B, 1, H, W)
      MV:     (B, 2, H, W)

    Output: F' = F + ψ_θ(concat(F, G, m, I_base, depth, MV))
    where ψ_θ = 1×1 → depthwise 3×3 → 1×1, channels-first.
    """
    def __init__(self, feat_dim: int, latent_R: int = 4, hidden: int = 64):
        super().__init__()
        in_channels = feat_dim + latent_R + 1 + 3 + 1 + 2
        self.proj_in = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = SqrSwish()
        self.proj_out = nn.Conv2d(hidden, feat_dim, kernel_size=1)
```

2. Add to `oss/sr/v6/activations.py` (new, tiny file):

```python
class SqrSwish(nn.Module):
    """0.5*v*(1 + v / sqrt(v² + 1)). Softsign-Swish. Cheaper than Swish on FP16.
    See docs/research/2026-05-08-fsr4-architecture-observations.md."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + x / torch.sqrt(x * x + 1.0))
```

3. `oss/sr/v6/cross_attention.py` — keep `PixelGaussianFusion` class but add a new mode flag. The v6.2 default skips this entirely (concat-fusion replaces it). Local top-K=16 cross-attn on disocclusion tiles ONLY is preserved for Quality+ tier, gated by a config flag (default off for pico-002 baseline).

**Acceptance gates:**

- New unit test `tests/sr/v6/test_concat_fusion.py`: shape correctness, residual connection works, no NaN propagation
- v6 test suite (`pytest tests/sr/v6/`) still passes
- ConcatFusion forward time on a 540×960 patch < 0.5ms on 3080 Ti (microbench)

---

## T2 — Disocclusion-only spawner with DGP dictionary

**Files:**

- `oss/sr/v6/dgp_dictionary.py` (new) — covariance prototype dictionary + softmax weight head
- `oss/sr/v6/disocclusion_spawner.py` (new) — disocclusion mask + hard-spawn-at-pixel-center logic
- `oss/sr/v6/gaussian_spawner.py` (modified) — add a `disocclusion_only` mode that calls into the new modules

**Per spec (`v62-arch-v4-spec.md` §3.4 + Risk #5):**

> Disocclusion-pixel-center spawn: spawn at exact disoccluded HR-pixel centers; let the canvas warp + motion-vector advection move Gaussians off-grid naturally. Decouples spawn-time grid alignment from sub-pixel positioning.
> DGP dictionary covariance: replace direct (Δscale, Δrot) regression with softmax over M=8-16 prototype Σ matrices fit from natural-image statistics. Forbids the integer-pixel-aligned local minimum by construction.

**Concrete changes:**

1. `dgp_dictionary.py`:

```python
class DGPDictionary(nn.Module):
    """Deep Gaussian Prior covariance dictionary.

    Replaces (Δscale, Δrot) regression with a softmax over M prototype
    covariance matrices. Prototypes are initialized from natural-image
    statistics (ContinuousSR DGP finding: 99% of natural-image Σ fall
    in σ²_x ∈ [0, 2.4], σ²_y ∈ [0, 2.2], ρσ_xσ_y ∈ [-0.9, 1.5]).
    """
    def __init__(self, M: int = 16, feat_dim: int = 64):
        super().__init__()
        self.M = M
        # Initialize prototypes: M points sampled in natural-image Σ range
        # Layout: [M, 3] = (a, b, d) inverse-covariance entries
        proto_init = self._sample_natural_prototypes(M)
        self.register_buffer("prototypes_abd", proto_init)  # [M, 3]
        self.weight_head = nn.Linear(feat_dim, M)
        self.scale_head = nn.Linear(feat_dim, 1)  # scalar λ multiplier

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feat: [N, feat_dim] per-Gaussian features.
        Returns (conic_abd: [N, 3], scale: [N]) — Λ_g = λ_g · Σ_m w_m · Λ_m"""
        ...
```

2. `disocclusion_spawner.py`:

```python
class DisocclusionSpawner(nn.Module):
    """Spawn at exact disoccluded pixel centers; velocity from MV.

    Disocclusion mask:
      D(p) = 1[ |Z_t(p) − Z_{t−1}(p − MV(p))| > τ_z ]

    Birth caps (configurable):
      max_births_per_frame: int = 256
      max_births_per_tile:  int = 4
    """
    def forward(self, depth_t, depth_prev, MV, lr_features, ...):
        # 1. Compute disocclusion mask
        # 2. TopK pixels by combined priority (residual + disocclusion)
        # 3. For each spawn candidate:
        #    - xy_g = (px + 0.5, py + 0.5)  ← exact pixel center, NO learned offset
        #    - velocity_g = MV(px, py)
        #    - feat_g, conic_abd_g = DGPDictionary(lr_features at (px, py))
        ...
```

3. `gaussian_spawner.py` — add `mode: str = "regress"` parameter. When `mode="disocclusion_only"`, delegate to `DisocclusionSpawner`. Default for v6.2-pico-002 is `disocclusion_only`.

**Acceptance gates:**

- Unit test: spawn at disoccluded pixel returns xy at integer + 0.5 (no learned offset)
- Unit test: DGP dictionary outputs are positive-definite covariances (a > 0, d > 0, ad - b² > 0)
- v6 smoke test passes
- **Critical FFT test:** synthetic disocclusion + spawn → render → FFT residual; λ=2px peak magnitude must be < 50,000 (vs v6.1-pico-001 baseline 177,265)

---

## T3 — Kalman 6-FLOP update for existing Gaussians

**File:** `oss/sr/v6/kalman_update.py` (new)

**Per spec (`v62-arch-v4-spec.md` §3.4):**

> Kalman 6-FLOP per Gaussian update, runs every frame on rebin step. Existing Gaussians use Kalman correction; spawner only fires for new births.

**Concrete:**

```python
class KalmanCanvasUpdate(nn.Module):
    """6-FLOP per-Gaussian xy + feat correction step.

    x̂_{t|t} = x̂_{t|t-1} + K_t · (z_t − H · x̂_{t|t-1})
    K_t = P_t / (P_t + R_t)   (diagonal, scalar)

    z_t = observed position (from MV-warped canvas)
    P_t = process variance (config; default 0.1)
    R_t = measurement variance (config; default 0.05)
    """
```

This is small. Not blocking — pico-002 trains fine without it; performance benefit comes online for v6.2-pico-002.1 (with student model) where spawner cost matters.

**Acceptance gate:** unit test for Kalman gain math; integration with `canvas_warp.py`.

---

## T4 — R=4-8 latent splat rasterizer

**Files:**

- `oss/cuda/src/rasterizer_fwd.cu` (modified)
- `oss/cuda/src/rasterizer_bwd.cu` (modified)
- `oss/sr/v6/rasterizer.py` (modified — Python wrapper for new shape)
- `oss/sr/v6/composite_head.py` (modified — decoder for R-latent → RGB)

**Per spec (`v62-arch-v4-spec.md` §3.2 step 5):**

> Sparse low-rank Gaussian raster (R=8, conic row recurrence, 2×2 register tiling). For each tile: `Z(p) = Σ_g w_g(p) · z_g`, `m(p) = Σ_g w_g(p)`.

**Concrete changes to forward kernel:**

1. The rasterizer currently accumulates F=64 channels. Add a config-driven path that accumulates only R channels (default R=4 for performance, R=8 for quality):

```cuda
// At kernel launch:
const int R = feature_rank;  // 4 or 8 (was 64)
```

2. `oss/sr/v6/rasterizer.py` — add a `latent_rank: int = 4` parameter. The output tensor shape changes from `[B, F=64, H, W]` to `[B, R, H, W]` plus an extra `[B, 1, H, W]` for the weight sum `m(p)`.

3. `oss/sr/v6/composite_head.py` — new decoder MLP:

```python
class LatentDecoder(nn.Module):
    """Decode R-latent splat to RGB.

    Input:  Z (B, R, H, W), m (B, 1, H, W), I_base (B, 3, H, W)
    Output: ΔI (B, 3, H, W)  -- residual added to reproject base

    Architecture: Conv1×1 → DepthwiseConv3×3 → SqrSwish → Conv1×1
    """
```

4. Backward: existing kernel computes per-Gaussian gradient on F=64 features; reduce to R channels. Less work, less atomic contention.

**Acceptance gates:**

- Numerical equivalence test: R=64 raster (reference) vs R=4 raster + decode (within 0.05 dB PSNR on training crops at fixed canvas density)
- CUDA microbench: R=8 forward kernel ≥4× faster than R=64 baseline on 3080 Ti
- Backward kernel: existing unit tests pass within atol=1e-5

---

## T5 — Student backbone scaffold (FasterNet + SqrSwish)

**File:** `oss/sr/v6/student/__init__.py` (new module)

**Per memo (`fsr4-architecture-observations.md`):**

```python
class StudentBackbone(nn.Module):
    """~1M-parameter student backbone for v6.2 inference path.

    Stack: Stem (3×3 conv) → 4× FasterNetBlock(48ch) → Tail (1×1 conv).
    Activation: SqrSwish throughout.
    Quantization: trainable in FP16; INT8 PTQ at TRT export time.

    Replaces HAT-Tiny in inference graph; HAT-Tiny is kept as
    distillation TEACHER for v6.2-pico-002.1 retraining.
    """
```

Not blocking pico-002 launch — v6.2-pico-002 trains with HAT-Tiny in the inference path. Student replaces HAT in v6.2-pico-002.1 distillation phase.

**Acceptance gate:** model summary shows ≤1.2M params; forward at 540×960 LR runs without error.

---

## T6 — V6.2 model orchestrator wiring

**File:** `oss/sr/v6/model.py` (modified — major)

This is the integration point. Changes:

1. Add new config fields to `V6Config`:
   - `latent_rank: int = 4` (R=4 default for performance)
   - `spawner_mode: str = "disocclusion_only"`
   - `fusion_mode: str = "concat"` (vs `"global_attn"` for legacy)
   - `enable_kalman_update: bool = False` (T3 follow-up)
   - `dgp_M: int = 16` (DGP dictionary size)

2. Replace cross-attention call site with `ConcatFusion` (T1 output)
3. Replace spawner call site to call `DisocclusionSpawner` when `spawner_mode == "disocclusion_only"`
4. Modify rasterizer wiring to pass `latent_rank` through (T4)
5. Decoder pipeline: `Z → LatentDecoder → ΔI → I = I_base + ΔI`

**Acceptance gates:** v6 test suite passes with v6.2 config; smoke training step runs on synthetic data without error.

---

## T7 — Training script v6.2 entry + flag set

**Files:** `scripts/sr_train_v6.py` (modified)

Add v6.2 flags:

```python
p.add_argument("--v62", action=argparse.BooleanOptionalAction, default=False,
               help="v6.2 architecture: R=4 latent + concat-fusion + disocclusion spawner")
p.add_argument("--latent-rank", type=int, default=None,
               help="Override R latent rank (default: 4 for v6.2, 64 for legacy)")
p.add_argument("--spawner-mode", choices=["regress", "disocclusion_only"], default=None)
p.add_argument("--fusion-mode", choices=["global_attn", "concat", "concat+local_topk"], default=None)
p.add_argument("--dgp-M", type=int, default=16)
```

When `--v62` is set, defaults flip to v6.2 architecture. Logging line includes the new config for run-tracking.

**Acceptance gate:** smoke training step works with `--v62 --max-steps 5 --output-dir /tmp/test`.

---

## T8 — Config preset for pico-002 launch

**File:** `configs/v6.2-pico-002.yaml` (new)

```yaml
# v6.2-pico-002 launch config
# Architecture: v6.2 (canvas-residual + concat-fusion + disocclusion spawner + R=4)
# Reference: docs/architecture/2026-05-08-v62-arch-v4-spec.md

run_name: srcnn-v6.2-pico-002
backbone: hat-tiny  # kept as teacher; will distill to student in v6.2-pico-002.1
v62: true
latent_rank: 4
spawner_mode: disocclusion_only
fusion_mode: concat
dgp_M: 16
canvas_capacity: 16384
batch_size: 4
patch_size: 256
trajectory_length: 8
base_lr: 2e-4
max_steps: 100000
warmup_steps: 5000
ckpt_every: 5000
first_ckpt_step: 100  # dense cold-start
```

---

## T9 — Launch pico-002 on 3080 Ti

```bash
ssh 3080ti-windows '"C:\Program Files\Git\bin\bash.exe" -lc "cd /e/oss-gaussian-server && git pull --ff-only origin main && python scripts/sr_train_v6.py --config configs/v6.2-pico-002.yaml --output-dir /e/checkpoints/srcnn-v6.2-pico-002 2>&1 | tee /tmp/v6.2-pico-002.log"'
```

WMI-orphan-spawn for long-running detachment per existing pattern (per `scripts/3080ti/launch-watcher.ps1`).

---

## Acceptance gates summary

| Gate | When | Pass condition |
|---|---|---|
| Stippling artifact (FFT) | v6.2-pico-002 step 5000 | λ=2px peak < 50,000 (vs 177,265 baseline) |
| PSNR vs v6.1 teacher | v6.2-pico-002 step 50000 | within 0.3 dB on TartanAir oldtown held-out |
| LPIPS vs v6.1 teacher | v6.2-pico-002 step 50000 | within 0.02 |
| ms/frame target | v6.2-pico-002 step 100000 + TRT export | <5ms SR+FG on 3080 Ti at 1080p output |

If stippling FFT gate FAILS at step 5000 → **stop pico-002 immediately** and root-cause. Don't repeat the pico-001 mistake of training through a structural bug.

---

## What this plan deliberately defers

- **v6.2-Pico-002.1** (student model + distillation + TRT export) — separate run after pico-002 converges
- **Cross-vendor kernels** (HIP/Metal/Vulkan/SYCL) — scaffolded; full ports queued after v6.2 ships
- **Capture tool DLL hook real D3D12 testing** — autonomous-feasible but separate work track
- **Conic row recurrence (H001)** — mathematically validated; CUDA port queued separately when we have a focused window
- **Tight ellipse AABB (H008)** — already shipped (commit 13e00b0)
- **Validity-mask reproject base pass** — initial v6.2 will use a simpler reproject; full validity mask is v6.2-pico-002.2

---

## Codex dispatch

Each of T1, T2, T4, T5 will be dispatched as a separate codex prompt to enable parallel execution under our `MAX_PARALLEL=3` queue runner. T3, T6, T7, T8, T9 are sequenced after the parallel block lands.

Prompts: see `/tmp/codex-queue/4xx_v62-*.txt` files.
