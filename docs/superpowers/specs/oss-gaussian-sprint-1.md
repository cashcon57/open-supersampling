# Sprint 1 Spec — CUDA Gaussian Renderer Integration

**Master plan:** [`../plans/2026-05-01-gaussian-master-plan.md`](../plans/2026-05-01-gaussian-master-plan.md) (Sprint 1 section).
**Design spec:** [`2026-05-01-gaussian-temporal-canvas-design.md`](2026-05-01-gaussian-temporal-canvas-design.md).

## Goal

Integrate the Image-GS tile-based CUDA rasterizer into OSS-Gaussian as the renderer foundation. Forward + backward differentiable via PyTorch. Benchmarked on RTX 3080 Ti.

## Sprint 1 deliverables (status)

| Task | Status |
|---|---|
| T1.1 Vendor Image-GS at pinned commit | ✅ submodule `03088368d42684` |
| T1.2 Python wrapper class (`Rasterizer`, `GaussianBatch`) | ✅ `oss/gaussian/renderer/rasterizer.py` |
| T1.3 CUDA extension build on 3080 Ti | ⏳ gated on CUDA Toolkit 12.4 install |
| T1.4 Forward render test | ✅ 5 reference-backend tests pass |
| T1.5 Differentiable backward test | ✅ 2 reference-backend tests pass + grad-check |
| T1.6 Performance benchmark (CSV output, 12 configs) | ⏳ awaits CUDA build |
| T1.7 Integration smoke test | ✅ 6 tests pass (no namespace collision with pixel-based OSS) |
| T1.8 Code review checkpoint | ⏳ blocked on T1.3/T1.6 + real-API key OR dry-run accepted |

## Acceptance criteria

- All Gaussian-track tests pass on M3 Max + RTX 3080 Ti (CUDA tests permitted to skip until T1.3 lands).
- Bench (T1.6) shows ≤ 5ms per render at 8K Gaussians @ 1440p on RTX 3080 Ti — design-spec target.
- Existing pixel-based OSS tests do not regress beyond the 7 pre-existing failures documented in `docs/superpowers/integration-points.md`.
- Code review pipeline produces an APPROVE verdict (heuristic or real-API).

## Out of scope

- The full network + training (Sprint 4)
- D3D12 hook (Sprint 2)
- Persistent canvas (Sprint 5)
- Cross-platform ports (Sprint 7)
- TensorRT INT8 export (Sprint 4)

## Files in scope

```
oss/gaussian/renderer/
  rasterizer.py
  bench.py
  vendor/
    image_gs/                    (submodule)
    LICENSE.image_gs
    README.md
tests/gaussian/
  test_renderer_forward.py
  test_renderer_backward.py
  test_renderer_integration.py
```

## Risks

- gsplat extension build may need MSVC + CUDA Toolkit alignment surgery.
- Image-GS upstream API change between vendoring and current Image-GS HEAD (we pinned a commit so this is mitigated).
- 3080 Ti memory bandwidth scaling not exactly linear with A6000 reference numbers.
