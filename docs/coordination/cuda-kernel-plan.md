# CUDA Kernel Implementation Plan — OSS Rasterizer + Cross-Attention

**Status:** draft, pending operator sign-off on Section K open questions
**Owner:** to be assigned
**Window:** ~4–6 weeks, runs parallel to v6.1-pico-001 training
**Hardware floor:** NVIDIA RTX 3080 Ti (sm_86), shop standard

## A. Scope inventory

Two pure-PyTorch modules become CUDA. Both are on the v6/v6.1 hot path; neither has a custom CUDA op today.

### A.1 Gaussian rasterizer

- **Public wrapper:** `oss/gaussian/renderer/rasterizer.py` — `Rasterizer` class, `GaussianBatch` dataclass, `TILE_SIZE = 16`, `CUDA_MAX_CHANNELS = 12`.
- **v6 caller:** `oss/sr/v6/rasterizer.py` — `V6Rasterizer.forward` calls `Rasterizer.__call__` once per frame (or 4× for the overlapped path).
- **Today's CUDA path:** vendored gsplat 1.4.0 at `oss/gaussian/renderer/vendor/image_gs/gsplat/`, built via its own `setup.py`. We are NOT replacing that today's working path; we are writing a clean OSS-native kernel that supersedes it.
- **Tensor contract** (`GaussianBatch`):
  - `xy: (N, 2)` float32 pixel-space centers, `[0, W) × [0, H)`
  - `scale: (N, 2)` float32 pixel-space, clamped `>= 1e-3`
  - `rot: (N,)` float32 radians
  - `feat: (N, F)` float32; F up to `token_dim = 64` (default), chunked to 12 in current CUDA path
  - Output: `(F, H, W)` float32, contiguous
- **Autograd boundary:** the wrapper inputs/outputs torch tensors; v6 sanitizes and `.to(float32)` inputs at `oss/sr/v6/rasterizer.py:78–82` before calling. The kernel ingests fp32 and returns fp32. The v6 wrapper casts the return back to `feat_dtype` (typically bf16) at line 108.
- **dtype expectations:** kernel runs in **fp32** end-to-end. bf16/fp16 are NOT supported in the kernel boundary; the v6 wrapper handles the cast in/out. This is deliberate — the existing reference path also forces fp32 for the exp/quadratic math.
- **Reference fallback:** `Rasterizer._render_reference` at `oss/gaussian/renderer/rasterizer.py:209–249` is the numerical truth for the equivalence test. It is naive O(N·H·W) but correct.

### A.2 Pixel↔Gaussian cross-attention

- **Module:** `oss/sr/v6/cross_attention.py` — `PixelGaussianFusion(nn.Module)`.
- **Defaults** (from `oss/sr/v6/model.py:108–110`):
  - `feat_dim = 180` (HAT-Tiny embed dim), `token_dim = 64`, `num_heads = 6`, `head_dim = 30`, `window_size = 16`, `mlp_ratio = 2.0`.
- **Tensor contract:**
  - `pixel_features: (B, feat_dim, H, W)`
  - `gaussian_tokens: (B, K, token_dim)`; K varies frame-to-frame, may be 0
  - Output: `(B, feat_dim, H, W)` matching input dtype (bf16 in training)
- **Compute graph (one forward):**
  1. LayerNorm on Q (`feat_dim`) and KV (`token_dim`)
  2. `q_proj`, `k_proj`, `v_proj` linears
  3. Window-partition Q to `(B*nW, ws*ws, C)`, broadcast K/V to `(B*nW, K, C)`
  4. 2D RoPE on Q only (`_apply_rope_2d`, lines 46–86) — Q half is split row/col, rotated, concat
  5. `F.scaled_dot_product_attention(q, k, v, scale)` — this is the kernel target
  6. `out_proj` linear, residual add, LayerNorm + 2-layer MLP residual
- **Hot inner kernel:** RoPE + SDPA fused. Steps 1–3, 6, 7 stay in PyTorch (cheap, batched matmuls). Steps 4 + 5 fuse into one CUDA kernel: `rope_then_attend(q, k, v, ws) -> attn_out`.
- **dtype expectations:** training runs **bf16**. The fused kernel must accept bf16 Q/K/V; internal accumulation in fp32; output bf16. PyTorch's SDPA already does this — match its semantics.
- **Sequence lengths in production:**
  - Q seq len = `ws * ws = 256` (fixed)
  - K seq len = `K`, the live Gaussian count, range ~0 to ~16k (canvas cap; verify against `oss/sr/v6/model.py` config — canvas size depends on `cfg`)
  - Batch (effective) = `B * nW`. For 540×960 pixel features at `ws=16`: nW = 34×60 = 2040. With B=1 that's 2040 effective batch. With B=4: 8160. The kernel needs to scale to high effective-batch, low Q-seq workloads — not the typical LLM SDPA shape.

## B. Numerical-equivalence test rig

### B.1 Layout

- New directory: `tests/cuda/` (sibling to existing `tests/gaussian/`).
- New marker: add `cuda` to `[tool.pytest.ini_options].markers` in `pyproject.toml`. Existing tests use `gpu` (gating CUDA-device presence). The new `cuda` marker is stricter: it gates **our** custom extension having compiled successfully. Tests:
  - `tests/cuda/conftest.py` — fixtures `cuda_device`, `seeded_rng(seed)`, `random_gaussian_batch(N, H, W, F, dtype)`, `random_attention_inputs(B, H, W, K, dtype)`.
  - `tests/cuda/test_rasterizer_equivalence.py`
  - `tests/cuda/test_rasterizer_backward_equivalence.py`
  - `tests/cuda/test_cross_attention_equivalence.py`
  - `tests/cuda/test_cross_attention_backward_equivalence.py`
- Run: `pytest tests/cuda/ -m cuda -v`. CI invocation: `pytest tests/cuda/ -m "cuda and not slow"`.

### B.2 Determinism

- `torch.manual_seed(0xC0DA)`, `torch.cuda.manual_seed_all(0xC0DA)` per test.
- Set `torch.use_deterministic_algorithms(True)` inside the equivalence tests (skip if it errors on SDPA — fall back to fixed seed only).
- Disable TF32: `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False` for the equivalence test only.

### B.3 Shape grid

Rasterizer:
- `N ∈ {0, 1, 16, 256, 4096}`
- `(H, W) ∈ {(32, 32), (256, 256), (540, 960), (1080, 1920)}`
- `F ∈ {1, 3, 12, 64}`
- dtype: fp32 only at the kernel boundary; v6-wrapper test casts in/out to bf16.

Cross-attention:
- `B ∈ {1, 2, 4}`, `(H, W) ∈ {(32, 32), (135, 240), (270, 480)}` (LR-feature shapes the model actually sees)
- `K ∈ {0, 1, 64, 1024, 8192}`
- `feat_dim, token_dim, num_heads, ws = (180, 64, 6, 16)` (production), plus a smaller `(60, 32, 4, 8)` smoke shape.
- dtype: `{fp32, bf16}`. fp16 is NOT in scope (training runs bf16).

### B.4 Tolerances

- **Forward fp32:** `atol=1e-5, rtol=1e-5`.
- **Forward bf16:** `atol=5e-3, rtol=5e-3` (bf16 has ~3-decimal precision; this is the SDPA-vs-naive baseline gap).
- **Backward fp32:** `atol=1e-4, rtol=1e-4`. Backward gradient checks use the reference implementation's autograd output as ground truth.
- **Backward bf16:** `atol=1e-2, rtol=1e-2`.

### B.5 Training gate

A trainer-side feature flag `OSS_USE_CUDA_KERNELS = os.environ.get("OSS_USE_CUDA_KERNELS", "0") == "1"`. Trainer reads it at startup. **Before flipping it on**, the equivalence test suite must pass on the target host. CI publishes a "kernels green" badge to the dashboard; the trainer launcher (`scripts/sr_train_v6.py`) refuses to enable kernels if the artifact's git SHA does not match `git rev-parse HEAD`.

### B.6 Fixture skeleton

```python
# tests/cuda/conftest.py
import os, pytest, torch

@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return torch.device("cuda:0")

@pytest.fixture
def kernels_built():
    try:
        from oss.cuda import rasterizer_ext, attention_ext  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Custom CUDA extension not built: {e}")
```

## C. Memory layout decisions

### C.1 Gaussian rasterizer

- **SoA**, matching canonical 3DGS. Five separate device buffers: `xy[N,2]`, `scale[N,2]`, `rot[N]`, `feat[N,F]`, plus a derived `conic[N,3]` (a, b, d) computed once per frame.
- **Tile size:** **16×16 pixels**, hard-coded. Matches the existing wrapper's `TILE_SIZE` and the v6 caller's contract. Smaller tiles waste warps; larger tiles overflow shared mem at F=64.
- **Per-tile Gaussian cap:** **256** post-cull. Empirically covers the 99th-percentile tile load at canvas sizes seen in v6.1-pico training (`max ~16k Gaussians, 540×960 frame, ~33×60 = 1980 tiles`). Overflow path: drop dimmest-by-area-weighted-α (per-frame deterministic). Log a "tile_overflow" counter to tensorboard.
- **Sort:** per-tile sort Gaussians **by depth** before alpha-blend front-to-back. We are 2D — depth is a frame-constant per-Gaussian "splat order" scalar already implicit in canvas insertion order. Use insertion order as the sort key; this matches the reference behavior. (gsplat sorts by tile-key, not depth, and uses sum compositing because `topk_norm=True`. Match the reference, not gsplat.)
- **Top-K normalization (`topk_norm=True`):** the reference does sum-compositing then divides by tile-summed weight. Our kernel does the same: one scratch `(F+1, H, W)` buffer (F channels + 1 weight), single-pass.

### C.2 Cross-attention

- **Q layout:** `(B*nW, num_heads, ws*ws, head_dim)` = `(B*nW, 6, 256, 30)`.
- **K, V layout:** `(B*nW, num_heads, K, head_dim)`. K is broadcast from `(B, K, C)` — we replicate the K/V projection across windows in PyTorch (cheap), then the kernel ingests the replicated tensors.
- **head_dim = 30** is awkward for tensor cores. Two options:
  - (a) Pad to head_dim=32 inside the kernel, ignore last 2 channels in output. Wastes 7% flops but enables fp16/bf16 mma.16x16x16.
  - (b) Use straight cuda-cores fp32 accumulation. Loses tensor-core throughput.
  - **Decision: option (a)**. The pad is internal; external API stays head_dim=30.
- **Sequence length range:** Q=256 fixed, K=[0, 16384]. For K > ~512 we want flash-attention-style blocked softmax to keep shared mem bounded. For K ≤ 256 the dense path is faster.

## D. CUDA arch target

- **Floor:** **sm_80** (A100). RTX 3080 Ti is sm_86; RTX 4090 is sm_89; H100 is sm_90.
- **nvcc flags** (target list):
  ```
  -gencode=arch=compute_80,code=sm_80
  -gencode=arch=compute_86,code=sm_86
  -gencode=arch=compute_89,code=sm_89
  -gencode=arch=compute_90,code=sm_90
  -gencode=arch=compute_90,code=compute_90
  ```
  Last entry is PTX-only forward-compat for sm_100+ (Blackwell).
- **Standard flags:** `-O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -lineinfo` (drop `--use_fast_math` for the equivalence-test build; it changes exp/sqrt behavior). Provide a `BUILD_DEBUG=1` env var that swaps in `-O0 -G`.
- Drop sm_70 (V100) and below — none of our infra uses them.

## E. PyTorch extension wiring

### E.1 Build mode

- **Prebuilt setup.py** at `oss/cuda/setup.py`, NOT JIT.
  - Reason: v6.1 training is on a Windows 3080 Ti host; JIT recompiles on every Python import on that machine, costing ~90s warmup. Prebuilt `.pyd`/`.so` lives in `oss/cuda/build/` and is `pip install -e ./oss/cuda` once.
  - JIT (`torch.utils.cpp_extension.load`) stays as the **dev path** behind a `OSS_CUDA_JIT=1` env var so iteration on the kernel doesn't require a wheel rebuild.
- Layout:
  ```
  oss/cuda/
    setup.py
    pyproject.toml
    src/
      rasterizer_fwd.cu
      rasterizer_bwd.cu
      attention_fwd.cu
      attention_bwd.cu
      bindings.cpp
      common.cuh
    oss_cuda/
      __init__.py        # imports the .so, exposes ops
      rasterizer.py      # autograd Function
      attention.py       # autograd Function
  ```
- Import surface: `from oss.cuda import rasterize_gaussians, fused_window_cross_attention`.

### E.2 Autograd Function vs custom op registration

- **Use `torch.autograd.Function`**, not `torch.library.custom_op` / `register_op`.
  - Pros: simpler, works with all torch 2.4–2.7 versions in our `pyproject.toml` pin range, no dispatcher quirks for fx/dynamo (we don't compile this anyway).
  - Cons: can't be torch.compile'd. We don't compile the rasterizer or fusion today (they live outside the HAT backbone graph), so this is fine.
  - Revisit if torch.compile lands on the rasterizer — that's post-Phase-4.

### E.3 Function skeleton

```python
# oss/cuda/oss_cuda/rasterizer.py
import torch
from torch.autograd import Function
from . import _C  # the compiled extension

class _RasterizeGaussians(Function):
    @staticmethod
    def forward(ctx, xy, scale, rot, feat, h, w, tile_size, topk_norm):
        # xy: (N,2) fp32, scale: (N,2) fp32, rot: (N,) fp32, feat: (N,F) fp32
        out, sorted_idx, tile_offsets, weight_sum = _C.rasterize_forward(
            xy, scale, rot, feat, h, w, tile_size, topk_norm,
        )
        ctx.save_for_backward(xy, scale, rot, feat, sorted_idx, tile_offsets, weight_sum)
        ctx.h, ctx.w, ctx.tile_size, ctx.topk_norm = h, w, tile_size, topk_norm
        return out  # (F, H, W) fp32

    @staticmethod
    def backward(ctx, grad_out):
        xy, scale, rot, feat, sorted_idx, tile_offsets, weight_sum = ctx.saved_tensors
        d_xy, d_scale, d_rot, d_feat = _C.rasterize_backward(
            grad_out.contiguous(), xy, scale, rot, feat,
            sorted_idx, tile_offsets, weight_sum,
            ctx.h, ctx.w, ctx.tile_size, ctx.topk_norm,
        )
        return d_xy, d_scale, d_rot, d_feat, None, None, None, None

def rasterize_gaussians(xy, scale, rot, feat, h, w, tile_size=16, topk_norm=True):
    return _RasterizeGaussians.apply(xy, scale, rot, feat, h, w, tile_size, topk_norm)
```

Same shape for `_FusedWindowCrossAttention`.

### E.4 Drop-in adapter

Add a `force_backend="oss_cuda"` to `Rasterizer.__init__` in `oss/gaussian/renderer/rasterizer.py`. When set, `_select_backend` returns `"oss_cuda"` and `__call__` dispatches to a new `_render_oss_cuda` that calls our autograd Function. The existing `"cuda"` path (vendored gsplat) stays untouched as a comparison baseline.

## F. Backward strategy

- **Rasterizer backward:** **per-pixel traverse-tile-list-back-to-front**, scatter gradients via `atomicAdd` into per-Gaussian gradient buffers.
  - This is the canonical 3DGS backward (Kerbl et al., gsplat, diff-gaussian-rasterization). It is correct, deterministic-in-practice when seeded, and the only path that handles overlapping Gaussians correctly without a full per-pixel autograd tape.
  - Atomics are unavoidable: a Gaussian can be touched by ~`(6σ)^2 / 256` ≈ 1–4 tiles, hundreds of pixels per tile, hence hundreds of grad-fragment contributions. Without atomics we'd need a per-pixel scratch + a final reduction, which costs 2× memory and is slower in practice.
  - Determinism: atomicAdd on fp32 is non-deterministic order-of-summation. Acceptable for training (drift is <1e-6 per step, washed out by optimizer noise). For the equivalence test we set `torch.use_deterministic_algorithms(True)`; if that flags our ops, we add a "deterministic backward" path that pre-sorts contributions per Gaussian — slower, gated by env var.
- **Cross-attention backward:** **standard SDPA backward with our RoPE applied to dQ before propagating.**
  - Forward saves Q (post-RoPE), K, V, log-sum-exp scaling factor. Backward computes `dQ, dK, dV` via the standard flash-attention bwd algorithm, then applies RoPE-inverse to dQ before returning.
  - Atomics required for the K/V grad accumulation across windows (every window contributes to the same global K/V). This is the same race-pattern as rasterizer-backward and the same fp32-atomicAdd resolution applies.
- **No autodiff inside CUDA.** Forward and backward are both hand-written. We do NOT do autograd-on-fwd. Reason: the per-pixel and per-window writeback patterns require domain knowledge of the math (which Gaussians touch which pixels, which windows touch which K/V) that an autograd tape would re-derive at 5–10× cost.

## G. Build / CI integration

- **Local builds:** `pip install -e ./oss/cuda` once per machine. Wheel artifact cached at `oss/cuda/build/lib.<plat>-cpython-<py>/oss_cuda/_C.<ext>`.
- **CI compile:**
  - GitHub Actions runner with the `cuda-12.1` image OR a self-hosted RTX 3080 Ti (preferred — already what the watcher uses).
  - New job `cuda-build` in `.github/workflows/`: checks out, runs `pip install -e ./oss/cuda`, then `pytest tests/cuda/ -m cuda`. Skips on PRs that don't touch `oss/cuda/**`, `oss/gaussian/renderer/**`, `oss/sr/v6/**`, or `tests/cuda/**`.
  - For non-CUDA runners (most PRs): job is skip-with-success — the `kernels_built` fixture in `conftest.py` already pytest.skip's the test. CI must NOT mark these as failures.
- **Watcher:** `scripts/3080ti-cuda-watcher.ps1` already runs on the training host. Add an entry that recompiles `oss/cuda` on `oss/cuda/src/**.cu` change and re-runs `pytest tests/cuda/ -m cuda` before resuming training. Does not block training while building (we want kernels green BEFORE flipping the flag, not before each step).
- **Artifact cache:** built `.so`/`.pyd` go in `oss/cuda/build/` and are gitignored. Wheel-publish to a private R2 bucket is **not** in scope for this 4–6 week sprint; per-host rebuild is fine.

## H. Phased rollout

- **Phase 1 — Rasterizer forward only (week 1–2)**
  - Implement `rasterize_forward` only, raises NotImplementedError on `.backward()`.
  - Equivalence test passes against `Rasterizer._render_reference`.
  - v6 trainer continues to use existing PyTorch path (no behavior change).
  - Gate: `OSS_USE_CUDA_KERNELS=0` is the default and stays so.
  - Goal: validate the build chain, the autograd-Function shape, the test rig.
- **Phase 2 — Rasterizer forward + backward (week 2–3)**
  - Backward kernel implemented, equivalence test (gradcheck-style with reference autograd) passes.
  - Run a 1k-step `v6.1-cuda-001` training: `OSS_USE_CUDA_KERNELS=rasterizer` flips the rasterizer only, fusion stays PyTorch. Compare loss curve to `v6.1-pico-001` step 0–1k.
  - Acceptance: loss matches PyTorch ±1e-3 relative for the first 1000 steps.
- **Phase 3 — Cross-attention fwd+bwd (week 3–5)**
  - Same shape: equivalence test, then 1k-step `v6.1-cuda-002` with `OSS_USE_CUDA_KERNELS=both`.
- **Phase 4 — Default-on (week 5–6)**
  - Flip default to `OSS_USE_CUDA_KERNELS=both` in `scripts/sr_train_v6.py`.
  - Benchmark wallclock + memory on RTX 3080 Ti, push to dashboard with the label `"NVIDIA RTX 3080 Ti measured with custom CUDA kernels"`.
  - The currently-running `v6.1-pico-001` finishes its 100k steps on the PyTorch path. The kernel flip applies to the next run (`v6.1-pico-002` or `v6.1-standard-*`).

## I. Risk register (top 5)

1. **Numerical drift over training.** Per-step error of 1e-4 relative compounds. **Mitigation:** Phase 2 + 3 each include a 1k-step parity training run; if loss diverges >1% relative we flag and fix before Phase 4.
2. **Race conditions / non-determinism in atomicAdd backward.** fp32 atomicAdd is non-associative. **Mitigation:** explicit `manual_seed` in the equivalence tests; deterministic-backward env var for paranoid debugging; doc the non-determinism as expected.
3. **PyTorch C++ ABI churn.** Our `pyproject.toml` pins `torch>=2.4,<2.7`. Each minor version is a different C++ ABI. **Mitigation:** rebuild the extension per torch upgrade; CI matrix tests one torch version per CUDA-CI run; document the rebuild step.
4. **bf16 mismatch from PyTorch's compiled SDPA.** PyTorch's `F.scaled_dot_product_attention` uses Flash-Attention 2 internally; bit-exact match is unrealistic. **Mitigation:** equivalence tolerance for bf16 is `5e-3`, NOT `1e-4`. We accept Flash-Attention-style numerics, not naive-attention numerics.
5. **Build flakes on non-3080ti hardware.** The compute_90 PTX path may not be exercised until first H100/cloud run. **Mitigation:** `compute_90` PTX-only entry in nvcc gencode list (already specified above); document the runtime JIT-PTX behavior; smoke-test on a short cloud run before any production run on new hardware.

## J. Wallclock / capacity model

- **Total estimated dev time: 5 weeks** (4 weeks engineering + 1 week buffer for the dual parity-training runs).
- **Phase milestones:**
  - W1: build chain green, fwd kernel boilerplate, equivalence test passing on a smoke shape.
  - W2: rasterizer fwd full shape grid green; rasterizer bwd kernel done.
  - W3: rasterizer bwd green + 1k-step parity training done. Cross-attention fwd kernel.
  - W4: cross-attention bwd green + 1k-step parity training done.
  - W5: default-on flip, dashboard benchmark numbers, docs.
- **What blocks v6.1 training:** **nothing** during Phase 1–3. The training run is on PyTorch ops; the kernel work is in a parallel branch and only touches `oss/cuda/`, `tests/cuda/`, and one feature-flag line in `oss/sr/v6/rasterizer.py` + one in `oss/sr/v6/cross_attention.py`. At Phase 4 default-on flip, the next training run starts on kernels; the in-flight `v6.1-pico-001` continues uninterrupted on PyTorch.

## K. Operator decisions (resolved 2026-05-07)

1. **License: Apache 2.0 in-repo.** Matches the rest of OSS.
2. **Bf16 default with ~5e-3 tolerance.** Matches v6.1 training; no perf regression.
3. **Determinism: within-run only (fast atomicAdd path).** Industry standard for 3DGS kernels. Drift <1e-6/step is washed out by optimizer noise.
4. **CI runner: nightly cron on the 3080 Ti.** Self-hosted, free, doesn't block PRs.
5. **Vendored gsplat: keep 1 month after Phase 4 ships, then remove in a follow-up cleanup PR.** Lets us A/B compare for a month to catch subtle regressions.

## L. Phase 1 dev workflow

- **Build/test host**: 3080 Ti during low-training-activity windows (overnight nightly cron) until the CachyOS laptop (4070M, sm_89) joins tailnet.
- **Laptop fit**: ideal Phase 1+ iteration host once tailnet'd (native Linux, fast nvcc, sm_89 forward-compat coverage). 8GB VRAM is sufficient for equivalence tests on smoke shapes.
- **MacBook fit**: drives codex/Claude conversation + edits + commits. Cannot build or run equivalence tests (no NVIDIA GPU, CUDA on macOS is moribund).
