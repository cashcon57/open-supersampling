# Intel SYCL/oneAPI Port Plan — OSS Rasterizer Kernels

**Status:** draft, awaits operator sign-off on Section G open questions
**Predecessors:** `docs/coordination/cuda-{kernel,phase2,phase3}-plan.md`, `docs/coordination/amd-hip-port-plan.md`
**Sequencing:** runs **after** NVIDIA Phase 4 default-on AND AMD/HIP Phase A4-equivalent.
**Scope:** port Phase 2/3 sum-composite rasterizer kernels to SYCL/oneAPI. Cross-attention deferred (mirrors NVIDIA + AMD precedent).
**Hardware floor:** Intel Arc A-series (DG2/Alchemist, Xe-HPG). Forward-compat target: Battlemage (Xe2). Integrated Xe-LP best-effort.
**Numerical bar:** fp32, `atol=1e-4 rtol=1e-4` matching Phase 3.

## A. Approach: SYCL/oneAPI (DPC++)

**Recommended.** Reject OpenCL (perf gap, second-class tooling on Intel). Reject Triton-Intel for v1 (too bleeding-edge for our `torch>=2.4,<2.7` floor; revisit at I7).

Rationale: Intel's first-party toolchain. Single-source C++ heterogeneous model. Direct semantic mapping from CUDA: work-group ↔ block, sub-group ↔ warp, local accessor ↔ shared mem, `atomic_ref` ↔ `atomicAdd`.

## B. Build system

`oss/sycl/` mirrors `oss/cuda/`. Use Intel `icpx -fsycl -fsycl-targets=spir64_gen` AOT for Arc Alchemist + Battlemage + PVC, with SPIR-V JIT fallback for unlisted devices.

PyTorch wiring: `torch.utils.cpp_extension.SyclExtension` for torch>=2.6+IPEX (primary path); manual `icpx` build with `torch.ops.load_library(...)` registration for older torch (fallback).

## C. Kernel adaptation

| CUDA | SYCL |
|---|---|
| `__global__ void k(...)` | `q.parallel_for(nd_range, [=](nd_item<2> it){ ... })` |
| `<<<grid, block>>>` | `nd_range<2>(global_size, local_size)` |
| `blockIdx`, `blockDim`, `threadIdx` | `it.get_group()`, `it.get_local_range()`, `it.get_local_id()` |
| `__shared__ float buf[N]` | `local_accessor<float, 1> buf({N}, h)` in command-group |
| `__syncthreads()` | `it.barrier(sycl::access::fence_space::local_space)` |
| `atomicAdd(p, v)` | `sycl::atomic_ref<...>(p).fetch_add(v)` |
| `__expf(x)` | `sycl::exp(x)` (IEEE-correct, matches CUDA) |
| `__launch_bounds__(N, M)` | `[[sycl::reqd_work_group_size(...)]]` attribute |
| Warp = 32 lanes | Sub-group, **size 16 default on Xe-HPG**, opt-in 8/32 |

The 16x16 work-groups stay (256 threads, fits Xe-HPG max 1024). LDS allocation 6 KB fits well under Xe-HPG's 64 KB SLM/work-group floor.

## D. Test rig

Mirrors `tests/cuda/`. Replace compute-sanitizer with: gtest unit tests + cross-vendor golden tensor diff against NVIDIA outputs + Intel VTune for profiling.

## E. Build/test host

Operator has no Intel discrete GPU. Options:
- Intel Developer Cloud free tier (queue-bound, 4hr sessions, Arc A770 + Max GPUs)
- Buy Arc A770 16GB ~$300 retail
- Defer until acquisition

**Recommendation:** start on Intel Developer Cloud through I3; reassess if queue blocks; buy Arc A770 if I6 ships and Intel becomes supported production target.

## F. Phased rollout

| Phase | Scope | Days |
|---|---|---|
| I1 | Hello-world dispatcher (build + 1-thread kernel) | 1 |
| I2 | preprocess_gaussians + build_tile_pairs | 1 |
| I3 | rasterize_sum (the hot kernel, 2.5d) | 2.5 |
| I4 | rasterize_backward | 1.5 |
| I5 | conic_to_scale_rot_grad + autograd Function | 0.5 |
| I6 | Bench + parity-training smoke | 1.5 |
| I7 | XMX matrix-engine optimization (open-ended) | TBD |

**Total: ~8 engineering days for I1-I6, ~10-12 calendar days with parity-training and review buffer.**

## G. Open questions for operator

1. Hardware: Intel Developer Cloud (free) OR buy Arc A770 ($300)?
2. Acceptance bar: `atol=1e-4 rtol=1e-4` per-vendor (recommended)?
3. License: Apache 2.0 in-repo (recommended)?
4. Cross-attention port: in-scope or deferred (recommended: deferred)?
5. `SyclExtension` vs manual `icpx`: support both at I1 (recommended)?

## Critical files

- `oss/sycl/src/{rasterizer_fwd.cpp, rasterizer_bwd.cpp, bindings.cpp}` (NEW)
- `oss/sycl/setup.py` (NEW)
- `oss/sycl/oss_sycl/rasterizer.py` (NEW)
- `tests/sycl/` (NEW)
