# OSS-Gaussian — Master Implementation Plan

**Spec:** `docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md`
**Branch:** `v0.2-dev` (existing branch, Gaussian track is additive)
**Total estimate:** 13–14 weeks, 7 sprints + cross-cutting code review pipeline

This plan details Sprint 1 in full. Sprints 2–7 are outlined; each is detailed at sprint kickoff with current information from the prior sprint.

---

## Cross-cutting: Code Review Pipeline

End-of-sprint gate. Two reviewer agents (correctness, spec-adherence) + one judge agent. Verdict gates next sprint start.

- Implementation: `oss/gaussian/review/` (Agent 4 dispatch in progress)
- Invocation: `python -m oss.gaussian.review.run --sprint N --commit-range <from>..<to>`
- Verdicts: APPROVE / REQUEST_CHANGES / BLOCK
- BLOCK escalates to user

---

## Sprint 1 — CUDA Gaussian Renderer Integration

**Goal:** Integrate Image-GS tile-based CUDA renderer into the OSS-Gaussian Python harness. Forward + backward differentiable. Benchmarked on RTX 3080 Ti.

**Estimate:** ~3 days (Image-GS already CUDA, no port).

**Why this first:** Every other sprint depends on the renderer. Validating it works on the 3080 Ti is also a system-readiness check.

### Files

```
oss/gaussian/renderer/
  __init__.py                 (existing)
  vendor/                     T1.1 — Image-GS source vendored
    image_gs/                 (submodule or copy)
    LICENSE.image_gs
  ext/
    setup.py                  T1.3 — CUDA extension build
    bindings.cpp              T1.3 — pybind11 wrapper
    rasterizer.cu             (from vendored Image-GS)
  rasterizer.py               T1.2 — Python wrapper class
  differentiable.py           T1.5 — torch.autograd.Function wrapper
  bench.py                    T1.6 — performance benchmark
tests/gaussian/
  test_renderer_forward.py    T1.4
  test_renderer_backward.py   T1.5
  test_renderer_bench.py      T1.6 (smoke only; full bench out-of-pytest)
```

### Tasks

#### T1.1 — Vendor Image-GS source

**Goal:** Image-GS code lives in repo at `oss/gaussian/renderer/vendor/image_gs/` with LICENSE preserved.

**Steps:**
1. `cd <home>/open-reconstruction-suite/oss/gaussian/renderer && mkdir -p vendor`
2. `cd vendor && git clone https://github.com/NYU-ICL/image-gs.git image_gs`
3. Copy `image_gs/LICENSE` → `oss/gaussian/renderer/vendor/LICENSE.image_gs`
4. Add a top-level `oss/gaussian/renderer/vendor/README.md` documenting attribution + commit hash pinned.
5. Inspect `image_gs/`'s CUDA rasterizer source — note exact filenames and entry points for T1.3.

**Verify:** `ls oss/gaussian/renderer/vendor/image_gs/` shows source. License is preserved. `git log -1 --format=%H vendor/image_gs/` (or saved to README) gives upstream commit pin.

**Acceptance:** Image-GS rasterizer source compiles standalone with their own setup.py on the 3080 Ti.

#### T1.2 — Python wrapper class

**Goal:** `oss.gaussian.renderer.Rasterizer` — clean Python class wrapping the CUDA rasterizer with typed args (Gaussian tensor, output resolution, top-K, tile size).

**API:**
```python
from oss.gaussian.renderer import Rasterizer

renderer = Rasterizer(tile_size=16, top_k=10, device="cuda")
output = renderer(
    gaussians,        # (N, 5+n) — μx, μy, θ, sx, sy, color...
    output_hw=(2160, 3840),
)
# output: (3, H, W) tensor
```

**Steps:**
1. Write `rasterizer.py` with the class above.
2. Type annotations + docstrings.
3. Defer actual CUDA call to T1.3 binding — for now, raise `NotImplementedError` with clear "needs T1.3" message OR fall back to a slow PyTorch reference renderer for validation.

**Verify:** `python -c "from oss.gaussian.renderer import Rasterizer; r = Rasterizer(); print(r)"` succeeds.

#### T1.3 — CUDA extension build

**Goal:** PyTorch CUDA extension that wraps Image-GS's rasterizer kernel and exposes it to `Rasterizer`.

**Steps:**
1. `setup.py` using `torch.utils.cpp_extension.CUDAExtension`. Include Image-GS rasterizer.cu + a thin `bindings.cpp` (pybind11) that defines `forward(gaussians, output_hw, tile_size, top_k)` and `backward(...)`.
2. Build: `cd oss/gaussian/renderer/ext && python setup.py build_ext --inplace` on 3080 Ti.
3. Verify import: `python -c "from oss.gaussian.renderer.ext import _C; print(_C)"`.
4. Wire `Rasterizer.__call__` in `rasterizer.py` to invoke `_C.forward(...)`.

**Verify:** Forward call from Python returns a non-zero tensor of shape `(3, H, W)`.

**3080 Ti dependency:** Requires CUDA toolkit + matching PyTorch CUDA build + Visual Studio Build Tools. Agent 1's report determines whether install steps are needed before T1.3.

#### T1.4 — Forward render test

**Goal:** Render a known set of Gaussians and pixel-diff against Image-GS reference output. Confirms the binding doesn't corrupt data layout.

**Steps:**
1. Vendor's `image_gs/` repo includes example/demo scripts. Render one example image at fixed seed using their script directly → save reference.
2. Render the same Gaussians via `Rasterizer` → save ours.
3. Pixel-diff: assert `mean_abs_diff < 1e-4` (FP32) or `1e-2` (FP16).
4. Pytest test in `tests/gaussian/test_renderer_forward.py`.

**Verify:** `pytest tests/gaussian/test_renderer_forward.py -v` passes.

#### T1.5 — Differentiable backward test

**Goal:** Gradients flow correctly through `Rasterizer` for training.

**Steps:**
1. `differentiable.py` — `torch.autograd.Function` subclass calling forward + backward kernels.
2. Test: small Gaussian set, loss against target image, take a gradient step, verify Gaussians move toward target.
3. `gradcheck` on a tiny example (4 Gaussians, 32×32 output) — finite-difference vs analytical gradient. Tolerance per Image-GS's own test.

**Verify:** `pytest tests/gaussian/test_renderer_backward.py -v` passes including `gradcheck`.

#### T1.6 — Performance benchmark

**Goal:** Measure render time on 3080 Ti at the configurations OSS-Gaussian will actually use.

**Configs to bench:**
- 1K, 5K, 8K, 15K Gaussians
- 1080p, 1440p, 4K output
- Tile size 16×16, top-K = 10

**Steps:**
1. `bench.py` — runs each config 100 times after 10 warm-up runs, reports mean/p50/p95/p99 in ms.
2. Output a CSV `bench_results_3080ti.csv` checked into `oss/gaussian/renderer/bench/`.
3. Reality-check: 8K Gaussians @ 1440p should be in the 1–3 ms range. If wildly different (>10ms), flag for investigation.

**Verify:** `python -m oss.gaussian.renderer.bench` produces CSV. Numbers in expected range.

#### T1.7 — Integration smoke test

**Goal:** Renderer importable + callable from existing OSS test infra. No collisions with pixel-based OSS modules.

**Steps:**
1. Add `tests/gaussian/test_renderer_integration.py`:
   - `from oss.gaussian.renderer import Rasterizer; from oss.model import OSSPico` — both work
   - Existing OSSPico tests still pass
2. CI matrix: add `gaussian` marker. Skip Gaussian tests when CUDA unavailable.

**Verify:** Full `pytest tests/ -v` passes. Existing tests untouched.

#### T1.8 — Sprint 1 code review checkpoint

**Goal:** Run code review pipeline on Sprint 1 commits before Sprint 2 starts.

**Steps:**
1. `python -m oss.gaussian.review.run --sprint 1 --commit-range main..HEAD` (or v0.2-dev base..HEAD).
2. Review artifacts saved to `oss/gaussian/review/artifacts/sprint-1/`.
3. Judge verdict APPROVE → mark sprint complete, proceed to Sprint 2.
4. REQUEST_CHANGES → iterate.
5. BLOCK → escalate to user.

**Verify:** Judge verdict file exists and is APPROVE.

---

## Sprint 2 — D3D12 Frame Interception + G-buffer Extraction

**Goal:** Hook into Cyberpunk 2077, extract color/depth/motion vectors per frame, dump for offline inspection.

**Detail:** Determined by Agent 3's research output (`docs/superpowers/d3d12-hook-design.md`). High-level tasks:

- T2.1 — Build harness: Visual Studio C++ project under `oss/gaussian/interception/`
- T2.2 — Minimal DXGI Present hook → log "hooked" to file from inside Cyberpunk
- T2.3 — DLSS DLL replacement scaffolding (NGX function stubs)
- T2.4 — G-buffer identification: which RTs are depth + motion vectors in Cyberpunk RED Engine
- T2.5 — Buffer extraction → save EXR + dump format compatible with Sprint 4 training
- T2.6 — Live capture mode: stream frames to disk during gameplay
- T2.7 — Code review checkpoint

**Estimate:** ~1.5 weeks. Risk: anti-cheat, RED Engine version drift.

---

## Sprint 3 — Tile Classifier

**Goal:** Per-frame 16×16 tile mask classifying complex (need Gaussian processing) vs simple (bilinear passthrough). Sprint 3 can run parallel with Sprint 2.

- T3.1 — CUDA kernel: gradient magnitude over LR frame
- T3.2 — Threshold selection — adaptive based on frame statistics
- T3.3 — Combine with G-buffer edge detection (depth discontinuity, motion edge)
- T3.4 — Output: bit-packed mask per tile
- T3.5 — Visualization: heatmap overlay for debugging
- T3.6 — Test: ~30% complex tiles on Cyberpunk frames; sanity check on Sintel
- T3.7 — Code review checkpoint

**Estimate:** ~1 week.

---

## Sprint 4 — Gaussian Param Network + Training

**Goal:** Lightweight CNN predicting (Δposition, Covariance Prior Bank weights, color) per complex tile. Trained on Sintel + TartanAir + Cyberpunk RenderDoc captures.

- T4.1 — Covariance Prior Bank: define 16-entry bank, freezable parameters
- T4.2 — Network architecture: small CNN, channel widths matching Image-GS style
- T4.3 — Differentiable end-to-end: network output → Rasterizer → loss
- T4.4 — Loss: composite (HDR L1 + 0.1 × (1-SSIM) + 0.05 × LPIPS) + temporal consistency
- T4.5 — Training script (Lambda H100 or local 3080 Ti)
- T4.6 — TensorRT INT8 export for inference
- T4.7 — PSNR baseline vs OSSPico on Sintel
- T4.8 — Ablation: Bank size 8 / 16 / 32
- T4.9 — Code review checkpoint

**Estimate:** ~4 weeks (training time dominates).

---

## Sprint 5 — Persistent Canvas + Motion Warp + Error Detection

**Goal:** GPU buffer holding N Gaussians across frames. Warp by motion vectors. Detect per-Gaussian reconstruction error and replace from LR.

- T5.1 — Canvas data structure: SoA Gaussian buffer in GPU mem
- T5.2 — Motion warp kernel: shift μ by motion[μ] each frame
- T5.3 — Error detection: per-tile MSE between rendered Gaussians and LR input
- T5.4 — Pruning: remove Gaussians from high-error tiles
- T5.5 — Spawning: add new Gaussians via network on prune events
- T5.6 — Cyberpunk live integration: drive canvas from Sprint 2 hook
- T5.7 — Temporal stability metric: frame-to-frame pixel delta in flat regions
- T5.8 — Comparison vs OSSPico on same Cyberpunk capture set
- T5.9 — Code review checkpoint

**Estimate:** ~3 weeks.

---

## Sprint 6 — Frame Extrapolation

**Goal:** Render canvas at t+α to produce predicted frame. Display predicted frame while GPU renders t+1. 60→120 FPS.

- T6.1 — α-conditioned warp: shift Gaussians by motion × α
- T6.2 — Latency test: render predicted frame before t+1 swapchain present
- T6.3 — Quality test: PSNR of predicted vs actual t+1 frame
- T6.4 — Comparison vs DLSS Frame Generation
- T6.5 — Code review checkpoint

**Estimate:** ~1 week.

---

## Sprint 7 — Cross-platform Ports

**Goal:** Port renderer to Metal (M3 Max) and ncnn/Vulkan (Steam Deck). Validate Gaussian-count knob covers all hardware tiers from one trained model.

- T7.1 — Metal MSL renderer (port CUDA → MSL)
- T7.2 — CoreML network export from PyTorch
- T7.3 — M3 Max integration test on Sintel
- T7.4 — ncnn network export
- T7.5 — Steam Deck Vulkan compute renderer
- T7.6 — Steam Deck integration test on Sintel at 1280×800 with 1K Gaussians
- T7.7 — Final cross-tier comparison report
- T7.8 — Code review checkpoint

**Estimate:** ~2 weeks.

---

## Graduation Decision Point

After Sprint 7, run the graduation criteria check:

1. PSNR + SSIM Gaussian vs OSSPico on Cyberpunk test set (>= 500 frames)
2. Temporal stability metric Gaussian vs OSSPico
3. User subjective approval (side-by-side video)
4. Latency: Gaussian total frame time <= 110% of OSSPico

**APPROVE all four** → archive `oss/` pixel-based to `oss/legacy/`, promote `oss/gaussian/` to `oss/`.
**Any criterion fails** → keep both tracks. Document failure mode + lessons. Iterate.
