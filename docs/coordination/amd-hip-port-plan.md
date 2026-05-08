# AMD HIP Port Plan — OSS Rasterizer Kernels

**Status:** draft, awaits operator sign-off on Section G open questions
**Predecessor:** `docs/coordination/cuda-phase3-plan.md` (Phase 3 in flight)
**Parent plan:** `docs/coordination/cuda-kernel-plan.md`
**Scope:** port the four NVIDIA CUDA rasterizer kernels (`preprocess_gaussians`, `build_tile_pairs`, `rasterize_sum`, `rasterize_backward`, `conic_to_scale_rot_grad`) to AMD ROCm/HIP. Match `atol=1e-4 rtol=1e-4` against the NVIDIA path.
**Hardware floor:** gfx906 (MI50). Primary verification target: gfx942 (MI300X) or gfx1100 (RX 7900 XTX), pending operator decision (G.2).
**Trigger:** kicks off after Phase 3d's parity-training acceptance lands. Does NOT block NVIDIA Phase 4.

## A. Approach: HIPIFY-first, hand-tune the hot kernel

**Recommended:** `hipify-perl` on `oss/cuda/src/*.cu` → emit `oss/hip/src/*.hip`. Hand-tune ONLY `rasterize_sum` for wave64 + `__launch_bounds__` retuning. Skip Triton (atomic-heavy backward has known issues on AMD-Triton).

Rationale:
- Pixel-major + tile-batched + `__shared__` + `__syncthreads()` (no warp shuffles or cooperative_groups) is exactly the sweet spot for HIPIFY.
- AtomicAdd-on-fp32 has matching semantics on both vendors.
- Triton-on-AMD has rough edges for atomic-heavy backward kernels (upstream issues #4012, #4287 as of 2026-Q1).

## B. Build system

`oss/hip/` mirrors `oss/cuda/`. Use `torch.utils.cpp_extension.CUDAExtension` (auto-routes to `hipcc` on ROCm-built torch). nvcc flags become hipcc flags via the same `extra_compile_args["nvcc"]` key.

GFX targets: `gfx906;gfx908;gfx90a;gfx940;gfx942;gfx1100`.

`-munsafe-fp-atomics` for gfx9xx ONLY (RDNA1 errata excludes it from gfx10xx).

## C. Kernel adaptation per file

| Kernel | Hand-edits required |
|---|---|
| `preprocess_gaussians` | Zero — pure FMA, no atomics |
| `build_tile_pairs` | Zero — pure index arith |
| `rasterize_sum` | `__launch_bounds__` retune for CDNA wave64 (4 blocks/CU vs NVIDIA 4 blocks/SM); LDS bank-conflict review; `expf` not `__expf` for equivalence |
| `rasterize_backward` | AtomicAdd flag gating (gfx9xx only `-munsafe-fp-atomics`) |
| `conic_to_scale_rot_grad` | Zero — closed-form chain rule, no atomics |

## D. Test rig

Mirrors `tests/cuda/`. Replace compute-sanitizer with: ASAN + gtest unit tests + cross-vendor golden tensor diff.

## E. Build/test host

**Operator has no AMD GPU on hand.** Options:
- RunPod MI300X ~$2.99/hr — full sprint budget ~$200-400
- AMD Developer Cloud free tier — queue-bound, suitable for one-off bring-up

**Recommendation:** start on AMD Developer Cloud free tier through Phase A2; rent MI300X if dev iteration blocks.

## F. Phased rollout (sub-phases)

| Phase | Scope | Days |
|---|---|---|
| A1 | HIPIFY scan + dispatcher boilerplate + smoke test | 1 |
| A2 | preprocess_gaussians validated | 0.5 |
| A3 | build_tile_pairs + sort + searchsorted | 0.5 |
| A4 | rasterize_sum (the hot kernel) | 1.5 |
| A5 | rasterize_backward (full P3 backward) | 1.5 |
| A6 | Equivalence + ASAN + bench gates | 0.5 |
| A7 | AMD-specific optimization (optional) | 1.5 |

**Total: 5.5-7 engineering days, ~10-14 calendar days with rental setup + iteration buffer.**

## G. Open questions for operator

1. Hardware: provision RunPod MI300X rental ($200-400) OR wait for lab AMD acquisition?
2. Primary target: MI300X (gfx942, wave64) OR RX 7900 XTX (gfx1100, wave32)?
3. Cross-vendor acceptance bar: `atol=1e-4` per-vendor + `atol=1e-3` cross-vendor?
4. License: Apache 2.0 in `oss/hip/` (recommended)?
5. Trigger: after Phase 3d (recommended) or wait for Phase 4?

## H. Risks

1. PyTorch+ROCm wheel may lag torch+CUDA — may need from-source build (+0.5d).
2. HIPIFY edge cases on `__shared__` inside templates — review diff in A1 commit.
3. gfx10xx atomicAdd corruption errata — gate `-munsafe-fp-atomics` to gfx9xx only.
4. Cost overrun on rental — shut down idle, persist `oss/hip/build/` to network volume.
5. No compute-sanitizer equivalent — invest more in gtest unit tests.

## Critical files

- `oss/cuda/src/rasterizer_fwd.cu` (HIPIFY source)
- `oss/cuda/src/rasterizer_bwd.cu` (HIPIFY source, after Phase 3 lands)
- `oss/hip/setup.py` (NEW)
- `oss/hip/src/{rasterizer_fwd.hip, rasterizer_bwd.hip, bindings.cpp}` (NEW, hipified)
- `tests/hip/` (NEW, mirrors tests/cuda/)
