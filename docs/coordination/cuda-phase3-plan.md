# CUDA Phase 3 Plan — Native Backward Rasterizer

**Status:** draft, awaits operator sign-off on Section J open questions
**Predecessor:** `docs/coordination/cuda-phase2-plan.md` (Phase 2 complete, commits `023700c` -> `1d131fe` -> `60814e6` -> `d94db4e`)
**Parent plan:** `docs/coordination/cuda-kernel-plan.md`
**Scope:** implement the CUDA backward of the sum-composite 2D Gaussian rasterizer to drop the `NotImplementedError` at `oss/cuda/oss_cuda/rasterizer.py`. Forward stays as Phase 2c. End state: full forward+backward CUDA path, gated by `OSS_USE_CUDA_KERNELS=rasterizer`, validated by 1k-step parity training.
**Hardware floor:** sm_80; primary verification host RTX 3080 Ti (sm_86) + RTX 4070 Laptop (sm_89). Nightly cron, NOT PR-blocking.
**Numerical bar:** fp32 boundary, `atol=1e-4 rtol=1e-4` start, fall-back `1e-3 1e-3` per-shape with documented justification.

## A. Algorithm decomposition

Forward (Phase 2c reference): `out[c, py, px] = Σ_g w_g · feat[g, c]` where `w_g = exp(-0.5 · q_g)` and `q_g = a·dx² + 2b·dx·dy + d·dy²`.

### A.1 Per-pixel local gradients (closed form)

- `dL/dw_g = Σ_c grad_out[c, py, px] · feat[g, c]`
- `dL/dfeat[g, c] += w_g · grad_out[c, py, px]`             (atomicAdd)
- `dL/dq_g = -0.5 · w_g · dL/dw_g`
- `dL/d(dx) = dL/dq · (2·a·dx + 2·b·dy)`,  `dL/d(dy) = dL/dq · (2·b·dx + 2·d·dy)`
- `dL/dxy[g] += (-dL/d(dx), -dL/d(dy))`                       (atomicAdd, sign flip from `dx = px - xy_g.x`)
- `dL/dconic[g] += (dL/dq · dx², dL/dq · 2·dx·dy, dL/dq · dy²)`  (atomicAdd)

### A.2 Tile-major launch (mirrors forward)

Same 16×16-thread tile blocks, same 256-Gaussian cooperative-load batching, same `__syncthreads()` placement. Forward kernel = pixel-major + tile-batched + sum into register `pix_out[F_chunk]`. Backward kernel = pixel-major + tile-batched + atomicAdd into per-Gaussian global outputs.

### A.3 Conic → (scale, rot) chain rule (post-pass, per-Gaussian)

```
diff = inv_sx2 - inv_sy2
d_inv_sx2 = c²·da + c·s·db + s²·dd
d_inv_sy2 = s²·da - c·s·db + c²·dd
d_sx = (sx > 1e-6) ? d_inv_sx2 · (-2 / sx³) : 0
d_sy = (sy > 1e-6) ? d_inv_sy2 · (-2 / sy³) : 0
d_rot = da · (-2·c·s·diff) + db · (diff·(c² - s²)) + dd · (2·c·s·diff)
```

CUDA owner: kernel `conic_to_scale_rot_grad` (1D launch, one thread per Gaussian).

## B. Kernel signatures

Live in NEW file `oss/cuda/src/rasterizer_bwd.cu` (split from forward).

### B.1 `rasterize_backward`

```cpp
__global__ __launch_bounds__(OSS_RASTER_BLOCK, 4)
void rasterize_backward(
    int H, int W, int num_tiles_x, int num_tiles_y,
    int F_chunk, int F_offset, int F_total,
    const int*    __restrict__ gaussian_idx_sorted,
    const int*    __restrict__ tile_offsets,
    const float2* __restrict__ xy,
    const float3* __restrict__ conic,
    const float*  __restrict__ feat,
    const float*  __restrict__ grad_out,
    float2*       __restrict__ d_xy,
    float3*       __restrict__ d_conic,
    float*        __restrict__ d_feat
);
```

Launch: `grid = (num_tiles_x, num_tiles_y)`, `block = (16, 16)`. Shared mem 4 KiB. Outer C++ loop chunks F into `F_CHUNK=16` slices. Register pressure target ≤64/thread (verify with `ptxas -v`).

### B.2 `conic_to_scale_rot_grad`

```cpp
__global__ void conic_to_scale_rot_grad(
    int N,
    const float2* __restrict__ scale,
    const float*  __restrict__ rot,
    const float3* __restrict__ d_conic,
    float2*       __restrict__ d_scale,
    float*        __restrict__ d_rot
);
```

Launch: 1D, mirrors `preprocess_gaussians`. No atomics.

### B.3 Dispatcher signatures

Forward dispatcher MUST now return a 4-tuple `(out, gaussian_idx_sorted, tile_offsets, conic)` so Phase 3 backward can `ctx.save_for_backward` them. Backward dispatcher:

```cpp
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_backward_cuda(
    torch::Tensor xy, torch::Tensor scale, torch::Tensor rot, torch::Tensor feat,
    torch::Tensor conic, torch::Tensor gaussian_idx_sorted, torch::Tensor tile_offsets,
    torch::Tensor grad_out,
    int64_t H, int64_t W, int64_t tile_size
);
// returns (d_xy, d_scale, d_rot, d_feat)
```

## C. Memory layout

| Tensor | Source | Shape | Atomic? |
|---|---|---|---|
| xy/scale/rot/feat | saved fwd | as before | no |
| conic | saved fwd | (N,3) | no |
| gaussian_idx_sorted | saved fwd | (P,) i32 | no |
| tile_offsets | saved fwd | (num_tiles+1,) i32 | no |
| grad_out | autograd | (F,H,W) | no |
| d_xy | output | (N,2) | YES |
| d_conic | scratch | (N,3) | YES |
| d_feat | output | (N,F) | YES |
| d_scale | output (post-pass) | (N,2) | no |
| d_rot | output (post-pass) | (N,) | no |

`d_xy`, `d_conic`, `d_feat` allocated as `torch.zeros` in dispatcher.

## D. Build system changes

1. NEW file `oss/cuda/src/rasterizer_bwd.cu`.
2. `oss/cuda/setup.py`: add `rasterizer_bwd.cu` to sources.
3. `oss/cuda/src/common.cuh`: add forward decls + promote `kMinScale` constant to header.
4. `oss/cuda/src/bindings.cpp`: change `rasterize_forward` return type to 4-tuple, add `rasterize_backward` pybind entry.
5. `oss/cuda/oss_cuda/rasterizer.py`: implement `_RasterizeGaussians.backward`, unpack 4-tuple in `forward`, save 7 tensors via `ctx.save_for_backward(xy, scale, rot, feat, conic, gaussian_idx_sorted, tile_offsets)`.
6. Version: `0.2.0` → `0.3.0+phase3a` → `+phase3b` → `+phase3c` → `0.3.0` final.

## E. Equivalence test changes

### E.1 Pytest backward (parametrized 100-combo grid, mirrors forward)

```python
@pytest.mark.parametrize("N", [1, 16, 256, 4096])
@pytest.mark.parametrize("H,W", [(32,32), (64,128), (256,256), (270,480), (540,960)])
@pytest.mark.parametrize("F", [1, 3, 12, 64])
def test_rasterizer_backward_equivalence(cuda_device, kernels_built, N, H, W, F):
    # Build inputs with requires_grad=True; kernel + reference forward+backward;
    # assert torch.testing.assert_close(grad_kernel, grad_reference, atol=1e-4, rtol=1e-4)
    # for each of (dxy, dscale, drot, dfeat).
```

Skip when `N·max(H,1)·max(W,1)·max(F,1) > 270M` for fast suite.

### E.2 gtest C++ unit test

`tests/cuda/cpp/test_rasterize_backward.cu`: single Gaussian, hand-derived analytic gradient, hardcoded `grad_out=ones`, verify per-Gaussian outputs.

### E.3 Compute Sanitizer

Extend `tests/cuda/sanitizer_smoke.py` to include `.backward()` call. All 4 tools must exit 0.

### E.4 Bench

Backward target: 1.5–3× forward cost. Append result to `docs/coordination/bench-baseline.json` under `backward` key.

## F. Failure modes

1. fp32 atomicAdd is non-deterministic in summation order. Cross-run bit-equivalence is NOT a goal (operator decision). Within-run determinism (same seed → same result) is what equivalence tests check.
2. Race conditions: every Gaussian-pixel touch generates atomicAdd writes; this is correct by design.
3. Empty F_chunk and N=0 must mirror forward's branches.
4. Non-finite Gaussians: skip identically to forward (`!isfinite(q) || q < 0` continue).

## G. Sub-phases

### 3a — `dfeat` only

- Extend forward dispatcher to 4-tuple return.
- `_RasterizeGaussians.forward` saves the four extra tensors.
- Kernel writes only `dfeat` atomicAdd; other grads stubbed.
- `_RasterizeGaussians.backward` returns `(None, None, None, dfeat, None, None, None, None)`.
- gtest covers `dfeat` only.
- Pytest equivalence asserts `dfeat` matches `_render_reference` autograd output.

Acceptance: pytest forward suite still 100%; pytest dfeat backward 100%; gtest backward green; sanitizer clean.

### 3b — `dxy` + `dconic`

- Add `dxy` and `dconic` writes.
- Pytest extended to assert `dxy` and `dconic_intermediate`.
- gtest extended.
- nvbench: backward time ≤ 3× forward.

### 3c — Post-pass + drop NotImplementedError

- Implement `conic_to_scale_rot_grad` kernel.
- `_RasterizeGaussians.backward` orchestrates fwd + post-pass.
- `NotImplementedError` removed.
- All 100 pytest combos pass at fp32 tolerance.
- v6 caller path: `OSS_USE_CUDA_KERNELS=rasterizer` end-to-end forward+backward smoketest passes.

### 3d — Parity training acceptance

- 100-step gate: launch v6.1-cuda-001 for 100 steps; loss must match v6.1-pico-001 first 100 steps within ±0.5%. If gate fails, halt and debug.
- Full 1k-step run: loss must match within ±1% relative.
- Document at `docs/coordination/cuda-phase3-progress.md`.

After 3d passes, operator decides on default-on flip per J.5.

## H. Time budget

- 3a: ~0.5 day engineering, ~0.5 day wallclock.
- 3b: ~0.75 day engineering, ~1 day wallclock (atomic-add tuning + equivalence debug).
- 3c: ~0.5 day.
- 3d: ~0.25 day to launch + babysit. **Wallclock dominated by 1000-step parity training: 2-4 days depending on contention with v6.1-pico-001 on the shared 3080 Ti.**

Total: **2 days engineering + 2-4 days parity-training wallclock = 4-6 calendar days** for Phase 3.

## I. Risks

1. AtomicAdd reduction-order divergence may exceed `atol=1e-4` at large shapes (540×960 + N=4096 + F=64). Mitigation: per-test relax to `atol=1e-3 rtol=1e-3` with documented justification; don't blanket-relax.
2. Compute Sanitizer racecheck false positives on global atomicAdd. If a warning fires, document as expected-and-benign; if it fires on shared-mem, that IS a real bug (sync placement).
3. 1k-step parity divergence from bf16-vs-fp32 boundary interactions. Mitigation: 100-step gate catches drift early; trainer's `_sanitize_active_gaussians` filters non-finite Gaussians.

## J. Open questions for operator

1. **Combined fwd+bwd kernel or split?** Recommendation: **split** (separate `rasterizer_bwd.cu`). Easier `__launch_bounds__` tuning, easier failure isolation. Confirm.
2. **Backward atol/rtol bar.** Recommendation: start `1e-4`, per-shape relax to `1e-3` with justification. Confirm.
3. **conic→(scale, rot) chain rule in CUDA or Python?** Recommendation: **CUDA post-pass kernel** for consistency + perf. Confirm.
4. **Drop `weight_sum` from saved-for-backward?** Phase 2 plan reserved it for forward-compat, but sum-composite backward doesn't need it. Recommendation: **drop**; reintroduce if Phase 4+ adds top-K. Confirm.
5. **Default-on flip timing.** After 1k-step parity passes, flip `OSS_USE_CUDA_KERNELS=rasterizer` for the next training run (v6.1-pico-002 / v6.2), NOT for the in-flight v6.1-pico-001. Recommendation: **wait** for next clean training start. Confirm.

## Critical files for implementation

- `/Users/cashconway/OpenSuperSampling/oss/cuda/src/rasterizer_fwd.cu` (extend dispatcher 4-tuple return)
- `/Users/cashconway/OpenSuperSampling/oss/cuda/src/rasterizer_bwd.cu` (NEW)
- `/Users/cashconway/OpenSuperSampling/oss/cuda/src/bindings.cpp`
- `/Users/cashconway/OpenSuperSampling/oss/cuda/src/common.cuh`
- `/Users/cashconway/OpenSuperSampling/oss/cuda/oss_cuda/rasterizer.py`
- `/Users/cashconway/OpenSuperSampling/oss/cuda/setup.py`
- `/Users/cashconway/OpenSuperSampling/tests/cuda/test_rasterizer_equivalence.py`
- `/Users/cashconway/OpenSuperSampling/tests/cuda/cpp/test_rasterize_backward.cu` (NEW)
- `/Users/cashconway/OpenSuperSampling/tests/cuda/sanitizer_smoke.py`
- `/Users/cashconway/OpenSuperSampling/oss/gaussian/renderer/rasterizer.py` (reference truth)
