# CUDA Phase 2 Plan — Native Forward Rasterizer

**Status:** draft, awaits operator sign-off on Section H open questions
**Predecessor:** `docs/coordination/cuda-phase1-progress.md`
**Parent plan:** `docs/coordination/cuda-kernel-plan.md`
**Scope:** replace the Phase 1 Python passthrough at `oss/cuda/src/bindings.cpp:21-25` with a real CUDA forward kernel. No backward in this phase (Phase 3).
**Hardware floor:** sm_80; primary verification host RTX 3080 Ti (sm_86) + RTX 4070 Laptop (sm_89).
**Worst-case shape:** H=1080, W=1920, F up to 64, N up to ~16k Gaussians.

## A. Algorithm decomposition

The reference at `oss/gaussian/renderer/rasterizer.py:209-249` is the numerical truth. **Critical observation:** the reference does NOT alpha-composite, does NOT depth-sort, does NOT apply top-K normalization, and **ignores the `topk_norm` flag entirely** in `_render_reference`. It computes:

> for each pixel `(x, y)`, output channel `c`:
> &nbsp;&nbsp;`out[c, y, x] = Σ_i  exp(-0.5 · Q_i(x, y)) · feat[i, c]`

where `Q_i(x, y) = a_i · dx² + 2 b_i · dx · dy + d_i · dy²` is the inverse-covariance quadratic form derived from `(scale_i, rot_i)` per lines 226-238:

```
cos = cos(rot[i]); sin = sin(rot[i])
sx = max(scale[i,0], 1e-6); sy = max(scale[i,1], 1e-6)
inv_sx2 = 1/sx²; inv_sy2 = 1/sy²
a = cos²·inv_sx2 + sin²·inv_sy2
b = cos·sin·(inv_sx2 - inv_sy2)
d = sin²·inv_sx2 + cos²·inv_sy2
```

This is the analytic form of `Σ⁻¹` where `Σ = R · diag(sx, sy)² · Rᵀ`.

Phase 2 implements **only this sum-composite path**. The `topk_norm` parameter is plumbed through but ignored, matching the reference. (We retain the parameter so the binding signature does not change.)

### Step 1 — Preprocess (per-Gaussian)
- **Inputs:** `xy:(N,2) f32`, `scale:(N,2) f32`, `rot:(N,) f32`. Frame size `H, W`, tile size `TS=16`.
- **Outputs:** `conic:(N,3) f32 = (a, b, d)`, `aabb:(N,4) i32 = (tile_x_min, tile_y_min, tile_x_max, tile_y_max)` (half-open), `tile_pair_count:(N,) i32`.
- **CUDA owner:** kernel `preprocess_gaussians` (one thread per Gaussian, 1D launch).

### Step 2 — Cumulative tile-pair count (host-side)
- **Inputs:** `tile_pair_count:(N,) i32`.
- **Outputs:** `cum_pair_count:(N+1,) i32`, `total_pairs: int`.
- **CUDA owner:** none — uses `torch.cumsum`. Mirrors gsplat `compute_cumulative_intersects` at `oss/gaussian/renderer/vendor/image_gs/gsplat/gsplat/utils.py:96-115`.

### Step 3 — Build (tile_id, gaussian_idx) pair list
- **Inputs:** `aabb:(N,4) i32`, `cum_pair_count:(N+1,) i32`.
- **Outputs:** `tile_pair_keys:(P,) i64`, `gaussian_idx_unsorted:(P,) i32`. `P = total_pairs`.
- **CUDA owner:** kernel `build_tile_pairs` (1D launch, one thread per Gaussian, mirrors gsplat `map_gaussian_to_intersects` at `forward.cu:11-43`).
- Depth field is unused — reference is sum-composite, order-independent.

### Step 4 — Sort pair keys
- **Inputs:** `tile_pair_keys:(P,) i64`.
- **Outputs:** `tile_pair_keys_sorted:(P,) i64`, `gaussian_idx_sorted:(P,) i32`.
- **CUDA owner:** none — `torch.sort` (cub::DeviceRadixSort under the hood). Matches gsplat `bin_and_sort_gaussians`.

### Step 5 — Build tile_offsets
- **Inputs:** `tile_pair_keys_sorted:(P,) i64`, `num_tiles`.
- **Outputs:** `tile_offsets:(num_tiles+1,) i32`.
- **CUDA owner:** none — `torch.searchsorted`. Bulletproof against empty tiles. (Open question H.2.)

### Step 6 — Rasterize (the big kernel)
- **Inputs:** `xy:(N,2) f32`, `feat:(N,F) f32`, `conic:(N,3) f32`, `gaussian_idx_sorted:(P,) i32`, `tile_offsets:(num_tiles+1,) i32`.
- **Outputs:** `out:(F,H,W) f32`. Optional `weight_sum:(H,W) f32` zeros (forward-compat for Phase 3 backward).
- **Pseudocode (one thread per pixel, one block per tile, 16×16 threads):**
  ```
  tile_id = blockIdx.y * num_tiles_x + blockIdx.x
  px = blockIdx.x * 16 + threadIdx.x
  py = blockIdx.y * 16 + threadIdx.y
  inside = (px < W and py < H)

  start = tile_offsets[tile_id]; end = tile_offsets[tile_id + 1]
  num   = end - start
  num_batches = (num + 256 - 1) / 256

  __shared__ int32 id_batch[256]
  __shared__ float2 xy_batch[256]
  __shared__ float3 conic_batch[256]

  float pix_out[F_CHUNK] = {0}      # F_CHUNK = 16; outer C++ loop chunks F=64 into 4 launches

  for b in 0..num_batches:
      tr = threadIdx.y * 16 + threadIdx.x   # 0..255
      bs = start + 256 * b
      if bs + tr < end:
          g = gaussian_idx_sorted[bs + tr]
          id_batch[tr] = g
          xy_batch[tr] = ((float2*)xy)[g]
          conic_batch[tr] = ((float3*)conic)[g]
      __syncthreads()

      if inside:
          k = min(256, end - bs)
          for t in 0..k:
              dx = xy_batch[t].x - px
              dy = xy_batch[t].y - py
              q  = conic_batch[t].x * dx*dx + 2*conic_batch[t].y * dx*dy + conic_batch[t].z * dy*dy
              if (!isfinite(q) || q < 0): continue
              w  = __expf(-0.5f * q)
              g  = id_batch[t]
              for c in 0..F_CHUNK:
                  pix_out[c] += w * feat[g * F_total + F_offset + c]
      __syncthreads()

  if inside:
      pix_id = py * W + px
      for c in 0..F_CHUNK:
          out[(F_offset + c) * H * W + pix_id] = pix_out[c]
  ```
- **CUDA owner:** kernel `rasterize_sum`.
- **F handling:** `F` ranges 1..64 in production. Holding `pix_out[64]` per thread = 256 bytes of registers — over budget. **Decision: chunk F.** Each launch handles up to `F_CHUNK = 16` channels. Outer C++ loops `(F + 15) // 16` times. (Open question H.3.)

## B. Kernel signatures + launch shape

All kernels in `oss/cuda/src/rasterizer_fwd.cu`, declared in `oss/cuda/src/common.cuh`.

### B.1 `preprocess_gaussians`
```cpp
__global__ void preprocess_gaussians(
    int N, int H, int W, int tile_size, int num_tiles_x, int num_tiles_y,
    const float2* __restrict__ xy,
    const float2* __restrict__ scale,
    const float*  __restrict__ rot,
    float3*       __restrict__ conic_out,
    int4*         __restrict__ aabb_out,
    int*          __restrict__ pair_count_out
);
```
- **Launch:** `dim3 grid((N + 255) / 256); dim3 block(256);`
- **Shared:** 0
- **Register target:** ≤32 regs/thread.

### B.2 `build_tile_pairs`
```cpp
__global__ void build_tile_pairs(
    int N, int num_tiles_x,
    const int4* __restrict__ aabb,
    const int*  __restrict__ cum_pair_count,
    int64_t*    __restrict__ keys_out,
    int*        __restrict__ gid_out
);
```
- **Launch:** `dim3 grid((N + 255) / 256); dim3 block(256);`
- **Shared:** 0
- **Register target:** ≤32 regs/thread.

### B.3 `rasterize_sum`
```cpp
__global__ void rasterize_sum(
    int H, int W, int num_tiles_x, int num_tiles_y, int F_chunk, int F_offset, int F_total,
    const int*    __restrict__ gaussian_idx_sorted,
    const int*    __restrict__ tile_offsets,
    const float2* __restrict__ xy,
    const float3* __restrict__ conic,
    const float*  __restrict__ feat,
    float*        __restrict__ out
);
```
- **Launch:** `dim3 grid(num_tiles_x, num_tiles_y); dim3 block(16, 16);`
- **Shared:** `id_batch (256·4B) + xy_batch (256·8B) + conic_batch (256·12B) = 6,144 B`. Well under 48 KB-per-block floor.
- **Register target:** ≤64 regs/thread @ `F_chunk=16`. Use `__launch_bounds__(256, 4)` to cap blocks/SM and force `pix_out[16]` to stay in registers.

### B.4 Host-side dispatcher (C++)
```cpp
torch::Tensor rasterize_forward_cuda(
    torch::Tensor xy, torch::Tensor scale, torch::Tensor rot, torch::Tensor feat,
    int64_t H, int64_t W, int64_t tile_size, bool topk_norm
);
```
Replaces the body of `rasterize_forward_stub` in `bindings.cpp:7-26`. Removes the `pybind11::module_::import` re-entry. Adds dtype/device/contiguity checks per Phase 1 latent issue #3:
```cpp
TORCH_CHECK(xy.is_cuda()    && xy.scalar_type()    == torch::kFloat32 && xy.is_contiguous(),    "xy must be cuda fp32 contiguous");
TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == torch::kFloat32 && scale.is_contiguous(), "scale must be cuda fp32 contiguous");
TORCH_CHECK(rot.is_cuda()   && rot.scalar_type()   == torch::kFloat32 && rot.is_contiguous(),   "rot must be cuda fp32 contiguous");
TORCH_CHECK(feat.is_cuda()  && feat.scalar_type()  == torch::kFloat32 && feat.is_contiguous(),  "feat must be cuda fp32 contiguous");
TORCH_CHECK(tile_size == 16, "tile_size must be 16 in Phase 2");
TORCH_CHECK(feat.size(1) <= 64, "F must be <= 64");
```

## C. Memory layout

| Tensor | Shape | Dtype | Source | Persisted (P3) | Scratch |
|---|---|---|---|---|---|
| `xy` | `(N, 2)` | f32 | input | yes | — |
| `scale` | `(N, 2)` | f32 | input | yes | — |
| `rot` | `(N,)` | f32 | input | yes | — |
| `feat` | `(N, F)` | f32 | input | yes | — |
| `conic` | `(N, 3)` | f32 | preprocess | yes | — |
| `aabb` | `(N, 4)` | i32 | preprocess | no | yes |
| `tile_pair_count` | `(N,)` | i32 | preprocess | no | yes |
| `cum_pair_count` | `(N+1,)` | i32 | host cumsum | no | yes |
| `tile_pair_keys` | `(P,)` | i64 | build_pairs | no | yes |
| `tile_pair_keys_sorted` | `(P,)` | i64 | sort | no | yes |
| `gaussian_idx_sorted` | `(P,)` | i32 | sort.cast | yes | — |
| `tile_offsets` | `(num_tiles+1,)` | i32 | searchsorted | yes | — |
| `weight_sum` | `(H, W)` | f32 | rasterize (zero-init in P2) | yes | — |
| `out` | `(F, H, W)` | f32 | rasterize | yes (return) | — |

`_C.rasterize_forward` returns `(out, gaussian_idx_sorted, tile_offsets, conic, weight_sum)` so Phase 3 can save them. The Python `_RasterizeGaussians.forward` unpacks the tuple, returns `out` to the caller, saves the rest via `ctx.save_for_backward` — addressing Phase 1 latent issue #2.

## D. Build system changes

### D.1 `oss/cuda/setup.py`
Switch `CppExtension` → `CUDAExtension`. Add `.cu` sources. Replicate the nvcc gencode list from parent plan D.

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NVCC_FLAGS = [
    "-O3", "-std=c++17",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "-lineinfo",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_90,code=compute_90",
]
# --use_fast_math is OMITTED. Equivalence test atol=1e-5 cannot tolerate
# fast_math's 1-ULP-loose __expf and reciprocal approximations.

setup(
    name="oss_cuda",
    version="0.2.0+phase2",
    packages=["oss_cuda"],
    ext_modules=[
        CUDAExtension(
            name="oss_cuda._C",
            sources=["src/bindings.cpp", "src/rasterizer_fwd.cu"],
            extra_compile_args={
                "cxx":  ["-O3", "-std=c++17", "-fPIC"],
                "nvcc": NVCC_FLAGS,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

### D.2 `oss/cuda/pyproject.toml`
Drop the `torch==2.4.1` build-isolation pin. Build command becomes `pip install --no-build-isolation -e ./oss/cuda` (host's torch is used).

### D.3 Build-host pre-step
Before each Phase 2 rebuild on a watcher/build host:
```
git -C <repo> pull --ff-only
pip install --no-build-isolation -e ./oss/cuda --force-reinstall
```
Without `--force-reinstall`, `setup.py build_ext` skips when source mtimes don't beat the existing `.pyd`. This addresses Phase 1 latent issue #4.

## E. Equivalence test changes

### E.1 Tolerances
Drop bit-exact. Use `torch.testing.assert_close(out_kernel, out_ref, atol=1e-5, rtol=1e-5)` per parent plan B.4.

### E.2 Shape grid (parametrized)
```python
@pytest.mark.parametrize("N", [0, 1, 16, 256, 4096])
@pytest.mark.parametrize("H,W", [(32, 32), (64, 128), (256, 256), (270, 480), (540, 960)])
@pytest.mark.parametrize("F", [1, 3, 12, 64])
def test_rasterizer_forward_equivalence(cuda_device, kernels_built, N, H, W, F):
    if N * H * W * F > 200_000_000:
        pytest.skip("too large for fast suite")
    ...
```
~100 combos, ~2 min on 3080 Ti.

### E.3 Slow large-shape tier (`@pytest.mark.slow`)
- `(8000, 540, 960, 64)`, `(16000, 540, 960, 64)`, `(4096, 1080, 1920, 64)`, `(16000, 1080, 1920, 12)`, `(16000, 1080, 1920, 64)`. Run via `pytest -m "cuda and slow"`.

### E.4 No-Python-re-entry test
```python
def test_kernel_does_not_reenter_python(cuda_device, kernels_built):
    from oss.cuda.oss_cuda import rasterizer as oss_rast
    saved = getattr(oss_rast, "_phase1_ref_forward", None)
    if saved is not None:
        oss_rast._phase1_ref_forward = None
    try:
        xy = torch.tensor([[16., 16.]], device=cuda_device, dtype=torch.float32)
        scale = torch.tensor([[3., 3.]], device=cuda_device, dtype=torch.float32)
        rot = torch.tensor([0.], device=cuda_device, dtype=torch.float32)
        feat = torch.tensor([[1.]], device=cuda_device, dtype=torch.float32)
        out = oss_rast.rasterize_gaussians(xy, scale, rot, feat, 32, 32, 16, True)
        assert out.shape == (1, 32, 32)
    finally:
        if saved is not None:
            oss_rast._phase1_ref_forward = saved
```
Addresses Phase 1 latent issue #1. After 2d, `_phase1_ref_forward` is deleted entirely; the test then becomes a `getattr` check confirming it is gone.

### E.5 Determinism + TF32 disable (autouse fixture in `conftest.py`)
```python
@pytest.fixture(autouse=True)
def _disable_tf32():
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn  = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev_matmul
    torch.backends.cudnn.allow_tf32 = prev_cudnn
```

### E.7 Google Test (gtest) C++ unit tests for kernel internals

Python pytest equivalence tests cover end-to-end correctness, but they can mask bugs in individual kernels (a wrong AABB clamp could be compensated by a downstream bug in pair construction and still produce numerically-close output). Per Gemini 2026-05-08: *"Google Test (gtest) is the industry standard for C++ unit testing; works perfectly with CUDA."*

Add `tests/cuda/cpp/` directory with:
- `tests/cuda/cpp/CMakeLists.txt` — pulls gtest via `FetchContent`, links against the same nvcc build artifacts as `oss_cuda._C`.
- `tests/cuda/cpp/test_preprocess.cu` — calls `preprocess_gaussians<<<...>>>` directly with hand-crafted small batches (N=1, 2, 4) and asserts the output `conic`, `aabb`, `pair_count` byte-for-byte.
- `tests/cuda/cpp/test_build_pairs.cu` — same shape: launches `build_tile_pairs<<<...>>>` with hand-built `aabb` arrays and asserts the output keys.
- `tests/cuda/cpp/test_rasterize.cu` — launches `rasterize_sum<<<...>>>` with a single Gaussian at frame center and asserts the per-pixel weight matches an analytic Gaussian within `atol=1e-6`.

Build + run:
```bash
cd tests/cuda/cpp && cmake -B build && cmake --build build && ./build/test_oss_cuda
```

Pass criteria for sub-phase 2a: `test_preprocess` exits 0.
Pass criteria for sub-phase 2b: `test_preprocess + test_build_pairs` both exit 0.
Pass criteria for sub-phase 2c: all three exit 0.
Pass criteria for sub-phase 2d: same — refactor must not break gtest.

Why both gtest AND pytest: gtest catches kernel-internal regressions during refactor (the fast TDD inner loop); pytest end-to-end equivalence catches integration bugs that only show on the full v6.1 production shape. Both are blocking gates.

### E.8 nvbench performance benchmark gate (REQUIRED for 2c)

Per Gemini's recommendation: *"NVIDIA nvbench: essential if you want to test for performance. It helps you verify if Claude's optimization actually made the kernel faster or slower."*

Replace the ad-hoc `bench_rasterize.py` from Section I.3 with **NVIDIA nvbench** — the standard CUDA benchmarking framework. nvbench produces stable, comparable numbers across runs; ad-hoc Python timers don't.

Add `tests/cuda/cpp/bench_rasterize.cu` registered with nvbench. It runs `rasterize_sum` on the full shape grid (5 representative shapes from E.3) and emits ms/iter + bandwidth GB/s + occupancy.

Build + run:
```bash
cd tests/cuda/cpp && cmake --build build --target bench_oss_cuda && ./build/bench_oss_cuda --json /tmp/bench-${GIT_SHA}.json
```

The bench script then diffs `/tmp/bench-${GIT_SHA}.json` against the previous baseline saved at `docs/coordination/bench-baseline.json`. Asserts:
- The CUDA kernel is at least **5× faster** than the PyTorch reference on `(N=4096, H=540, W=960, F=64)`.
- No regression more than **10%** vs the previous commit's baseline (catches "optimization that's actually a deopt" — Gemini's specific concern).

When 2c first ships, the baseline file is created from that commit's numbers. Each subsequent commit's bench output is diffed against it. If a commit makes a kernel slower, CI fails the merge.

The 50× full target lands at Phase 4 default-on (with profiling-guided refactors); the 5× floor at 2c flags catastrophic perf misses early.

### E.6 Compute Sanitizer gate (REQUIRED before declaring 2c done)

Phase 2c's `rasterize_sum` is the first kernel with shared memory + cooperative loads + arithmetic on shared buffers. Per Gemini's 2026-05-08 note: *"Moving data into Shared Memory is where the code gets incredibly brittle. One wrong offset and the whole system crashes or returns a black screen."* Compute Sanitizer (NVIDIA's `compute-sanitizer` CLI, replaces `cuda-memcheck`) catches these:

| Tool | What it catches | Required pass |
|---|---|---|
| `memcheck` | Out-of-bounds reads/writes, invalid memory access | ✅ blocking 2c |
| `racecheck` | Shared-memory race conditions | ✅ blocking 2c |
| `initcheck` | Uninitialized device memory reads | ✅ blocking 2c |
| `synccheck` | Illegal `__syncthreads` / barriers | ✅ blocking 2c |

g14 has `compute-sanitizer 2026.1.1` at `/opt/cuda/bin/compute-sanitizer`. 3080 Ti's CUDA toolkit may not — verify before assigning a sanitizer-gate run. If 3080 Ti doesn't have it, run sanitizer suite on g14 instead (sm_89 finds memory bugs just as well as sm_86).

Run pattern (added to `tests/cuda/run_sanitizer.sh`):
```bash
#!/usr/bin/env bash
set -e
PY=~/miniforge3/envs/oss-cuda/bin/python
SCRIPT=tests/cuda/sanitizer_smoke.py        # imports + runs rasterize_gaussians on a small + medium shape
for tool in memcheck racecheck initcheck synccheck; do
  echo "=== compute-sanitizer --tool=$tool ==="
  /opt/cuda/bin/compute-sanitizer --tool=$tool --error-exitcode=42 \
     --print-limit 50 --leak-check full $PY $SCRIPT
done
```

`tests/cuda/sanitizer_smoke.py` runs a minimal forward at `(N=16, H=64, W=64, F=3)` then `(N=512, H=270, W=480, F=12)` to exercise both the small-tile-list and multi-tile-batch code paths.

`--error-exitcode=42` makes the shell script fail-fast on any sanitizer error. Phase 2c push gate adds: "compute-sanitizer all four tools exit 0."

## F. Failure modes + handling

| Mode | Trigger | Handling |
|---|---|---|
| `N == 0` | Empty Gaussian batch | Return `torch::zeros({F, H, W}, ...)` immediately. |
| Degenerate scale `(1e-6, 1e-6)` | Near-zero covariance | `clamp_min(1e-6)` in preprocess. |
| AABB outside frame | Off-screen Gaussian | Clamp tile coords; `tile_pair_count[i] = 0`. |
| Tile overflow | Pathological density | No hard cap in P2; processed in 256-batches. (Backward-pass concern, P3.) |
| `rot=NaN` or `scale<0` | Sanitization gap | NaN rot → NaN conic → caught by `if (!isfinite(q))`. v6 sanitizer at `oss/sr/v6/rasterizer.py:215` is the upstream filter. |
| `total_pairs == 0` | All Gaussians culled | Skip steps 4-6, return zeros. |
| F > 64 | Out of supported range | `TORCH_CHECK(F <= 64)`. Defense in depth. |

## G. Time budget + sub-phases

Phase 2 = 4 sub-phases, each one codex dispatch + one commit + one push gate. Total ~5.5 engineering days + 1.5 buffer = **~7 calendar days** assuming prompt review.

### Sub-phase 2a — preprocess kernel + dispatcher boilerplate (1 day)
- Switch `setup.py` to `CUDAExtension`, add `rasterizer_fwd.cu` with **only** `preprocess_gaussians`.
- New C++ binding signature returns `(out, gaussian_idx_sorted, tile_offsets, conic, weight_sum)` tuple.
- Forward computes `conic` + `aabb` in CUDA, falls back to Python ref for rasterize.
- Equivalence test stays bit-exact (rasterize unchanged).
- New test `test_preprocess_conic_correctness` compares kernel `conic` vs PyTorch computation on small batch.

### Sub-phase 2b — tile-pair build + sort + tile_offsets (1 day)
- Add `build_tile_pairs` kernel. Wire `torch.cumsum` / `torch.sort` / `torch.searchsorted`.
- Forward produces real `gaussian_idx_sorted` + `tile_offsets`, rasterize still calls Python ref.
- New test `test_pair_construction_correctness` with hand-computed expected outputs.

### Sub-phase 2c — `rasterize_sum` kernel (2-3 days, the hardest)
- Add the big kernel with F-chunking.
- Forward CUDA end-to-end, with `OSS_CUDA_RASTER_DEBUG=1` env-var fallback to Python ref for A/B during bring-up.
- Switch equivalence test to `atol=1e-5, rtol=1e-5`. Run full shape grid.
- **Compute Sanitizer gate (E.6) — all 4 tools must exit 0 before merge.**
- The hard part: F-chunking inner loop vectorization without spilling. May need `__launch_bounds__(256, 4)`.

### Sub-phase 2d — drop Python re-entry (0.5 day)
- Delete `_phase1_ref_forward`. Delete `OSS_CUDA_RASTER_DEBUG` fallback.
- Kernel self-contained.
- Add `test_kernel_does_not_reenter_python`.
- Run slow-tier large-shape tests as final gate.
- Bump `setup.py` version to `0.2.0` (drop `+phase2` local segment).

## H. Open questions for operator

1. **Sort primitive: `torch.sort` vs hand-rolled `cub::DeviceRadixSort`?** Recommendation: `torch.sort`. Confirm.
2. **`tile_offsets` build: `torch.searchsorted` vs gsplat-style edge kernel?** Recommendation: `torch.searchsorted`. Confirm.
3. **`F_CHUNK` size: 8, 16, or 32?** Recommendation: 16, microbench during 2c. Confirm or override.
4. **Tile size: keep at 16 or revisit?** Recommendation: 16. Confirm.
5. **`rasterize_sum` + `normalize` (top-K): one kernel or two?** Recommendation: leave as one for P2; if top-K added P4, allocate `weight_sum` scratch + outer `normalize_top_k` kernel rather than fuse. Confirm direction (not blocking P2).

## I.0. TDD workflow per sub-phase (Gemini's recommendation, 2026-05-08)

Each sub-phase follows this loop. **Order matters** — write the test before the kernel, never the other way around.

1. **Interface first.** Define the C++ kernel signature (Section B). Lock the function name, params, return-type. Codex prompt for the sub-phase pastes the signature verbatim and forbids changing it.
2. **Golden test.** Write the test that asserts the kernel matches the PyTorch reference within tolerance. For the rasterizer, this is the parametrized shape grid (Section E.2). For per-kernel C++ unit tests of preprocess / build_pairs / tile_offsets, use Google Test (Section E.7). The test exists *before* the kernel.
3. **Kernel generation (Green).** Give codex the signature + the failing test + permission to write the minimal CUDA needed to pass it. No optimization on the first pass — the simplest correct kernel that turns red into green.
4. **Refactor loop.** Once green, ask for shared-memory cooperative loads, warp-shuffle reductions, `__launch_bounds__` tuning, etc. Each refactor is a separate commit. If it breaks the test suite, revert that commit; the test suite is the rollback safety net.
5. **Sanitizer + bench gate before merge.** `compute-sanitizer` (E.6) and `nvbench` (E.8) must both pass before the sub-phase ships.

Codex prompts for 2a/2b/2c/2d will lead with the test file the sub-phase must pass, NOT the kernel implementation. The codex's job is to make the test go green.

## I. Performance-as-bug guardrail (Gemini's note, 2026-05-08)

> "Performance is the Real Bug. In an upscaler, a kernel that works but is slow is essentially a 'failed' kernel. If Claude writes a kernel using only Global Memory, it might be 50x slower than the CPU version. Moving data into Shared Memory is where the code gets incredibly brittle. One wrong offset and the whole system crashes or returns a black screen."

Implications baked into this plan:

1. **Shared memory is mandatory, not optional.** `rasterize_sum` MUST use the cooperative-load pattern (`id_batch`, `xy_batch`, `conic_batch` shared arrays). Section A Step 6 specifies this explicitly. The codex prompt for sub-phase 2c will reject a global-memory-only implementation as "Phase 2c-fail".
2. **Compute Sanitizer is the brittleness antidote.** Section E.6 makes all 4 sanitizer tools (memcheck/racecheck/initcheck/synccheck) blocking gates for 2c. Catches the "one wrong offset" class of bug before it lands.
3. **Performance benchmark is a 2c gate, not a Phase-4-only concern.** Add `tests/cuda/bench_rasterize.py` to 2c's push gate that asserts the CUDA kernel is at least **5x faster** than the PyTorch reference on the worst-case shape `(N=4096, H=540, W=960, F=64)`. If not, sub-phase 2c is incomplete — re-tune launch bounds, F_CHUNK, or shared-mem layout. The full 50× target lands at Phase 4 default-on, but a 5× floor at 2c flags catastrophic perf regressions early.
4. **No premature optimization.** `__use_fast_math` is intentionally omitted — its 1-ULP-loose `__expf` would tank the equivalence test atol. We pay the perf cost in 2c and revisit in Phase 4 if benchmarks require.

## Critical Files for Implementation

- `/Users/cashconway/OpenSuperSampling/oss/cuda/src/bindings.cpp`
- `/Users/cashconway/OpenSuperSampling/oss/cuda/src/rasterizer_fwd.cu` (new)
- `/Users/cashconway/OpenSuperSampling/oss/cuda/src/common.cuh`
- `/Users/cashconway/OpenSuperSampling/oss/cuda/setup.py`
- `/Users/cashconway/OpenSuperSampling/oss/cuda/oss_cuda/rasterizer.py`
- `/Users/cashconway/OpenSuperSampling/tests/cuda/test_rasterizer_equivalence.py`
- `/Users/cashconway/OpenSuperSampling/tests/cuda/run_sanitizer.sh` (new)
- `/Users/cashconway/OpenSuperSampling/tests/cuda/sanitizer_smoke.py` (new)
- `/Users/cashconway/OpenSuperSampling/tests/cuda/bench_rasterize.py` (new)
- `/Users/cashconway/OpenSuperSampling/oss/cuda/pyproject.toml`
