# 2026-05-08 — Phase 4 Elegance M: redundant-computation audit

## Question

Where does the current v6/CUDA pipeline do dead, redundant, or strength-reducible work?

## Method

Static audit over `oss/sr/v6/`, `oss/cuda/src/`, and `oss/gaussian/renderer/`, with grep and line-number inspection.

## Inputs

- Files audited: native CUDA rasterizer forward/backward, v6 model/rasterizer/spawner/canvas-warp/keyframe/loss/cross-attention modules, renderer wrapper.
- Script: `tests/cuda/perf-math/m_audit_scan.py`

## Output

- M1: `topk_norm` is passed at `oss/gaussian/renderer/rasterizer.py:194`, accepted at `oss/cuda/src/rasterizer_fwd.cu:461`, discarded at `oss/cuda/src/rasterizer_fwd.cu:463`.
- M2: forward allocates/writes unused `aabb`/`pair_count` at `oss/cuda/src/rasterizer_fwd.cu:505-517` because full-frame pairs start at `oss/cuda/src/rasterizer_fwd.cu:522`.
- M3: full-frame `gid_sorted`/`tile_offsets` are deterministic from `N` and tile id; current buffer materialization is at `oss/cuda/src/rasterizer_fwd.cu:525-537`.
- M4: hot-path `out` and `tile_offsets` are zeroed before full overwrite at `oss/cuda/src/rasterizer_fwd.cu:502` and `oss/cuda/src/rasterizer_fwd.cu:527`.
- M5: forward weight loop recomputes row-only pixel coordinates inside the column loop at `oss/cuda/src/rasterizer_fwd.cu:208-211`.
- M6: backward recomputes `dx*dx`, `dx*dy`, `dy*dy` at `oss/cuda/src/rasterizer_bwd.cu:84` and `oss/cuda/src/rasterizer_bwd.cu:106-108`.
- M7: forward conic preprocess repeats `c*c`, `s*s`, `c*s` at `oss/cuda/src/rasterizer_fwd.cu:63-65`; backward already CSEs at `oss/cuda/src/rasterizer_bwd.cu:140`.
- M8: `d_rot` can be factored at `oss/cuda/src/rasterizer_bwd.cu:150-153`.
- M9: canvas warp filters `in_frame` at `oss/sr/v6/canvas_warp.py:221` after Jacobian/covariance work begins at `oss/sr/v6/canvas_warp.py:228`.
- M10: `_active_mask` allocates identity at `oss/sr/v6/model.py:516`; keyframe mask already treats `view_matrix is None` as identity at `oss/sr/v6/keyframe_active_mask.py:95`.
- M11: ST update allocates all-ones transmittance at `oss/sr/v6/model.py:676` for a multiply path in `oss/sr/v6/st_variation_score.py:99`.
- M12: cacheable constants are recomputed: RoPE at `oss/sr/v6/cross_attention.py:61-71`, tile centers at `oss/sr/v6/gaussian_spawner.py:171-176`, feather masks at `oss/sr/v6/rasterizer.py:177-186`, Sobel expansion at `oss/sr/v6/losses.py:114-117`.

Recommendation: first implementation batch should target M1-M4; those are pipeline-shape fixes with the largest likely payoff.

## Reproducibility

```bash
./venv-py312/bin/python tests/cuda/perf-math/m_audit_scan.py
```
