# v7 Phase 2 closeout — N-D Gaussian model wiring done

**Date:** 2026-05-12
**Status:** Phase 2 (model wiring + intermediate-frame supervision) complete.
**Next:** Phase 3 — v7-pico-005 training run on TartanAir-subsampled (100K steps, ~6 d on 3080 Ti).

## What landed in Phase 2

Sub-phases delivered (in commit order, oldest first):

| Commit | Phase | Scope |
|---|---|---|
| `e7a231c` | 1 | Python ref N-D Gaussian time-slice rasterizer + 9 tests (Schur-complement conditioning on `t`) |
| `ccbb2dc` | 2A | `NDCanvasState` (Cholesky-packed cov), `ChildState` parent-child spawner, materialize-on-loss + 20 tests |
| `262351a` | 2A | `V7Model` skeleton: backbone + N-D canvas + composite head, t-queryable forward |
| `79797ca` | 2A | `oss_fx_loss`: Charbonnier + LPIPS + FG-aux + temporal-consistency wrapper, 8 tests |
| `fd65e45` | 2A | `TartanAirIntermediateTriplets` (i, i+1, i+2) dataset adapter, 7 tests |
| `7cde5ea` | 2A | `scripts/sr_train_v7.py` training scaffold |
| `cade9a3` | 2B | `BackboneSpawner` tile-decoder + V7Model.forward(spawn_at_t=...) wiring, 12 tests |
| `ff2a5c6` | 2B | Two-frame spawn flow in trainer (`spawn_at_t=0`, then `=2`, render at `=1`) |
| `753a328` | 2C | Real HAT-Tiny backbone swap-in (`backbone_kind="hat_tiny"` chooses transformer teacher) |
| `c869820` | 2C+ | Canvas pruning policy (`prune_to_count`, `compact`) + `--backbone-kind` CLI arg |
| `221f239` | 2C closeout | End-to-end training integration test (forward → spawn → render-at-t → loss → backward → step) |

**Tests:** 61/61 v7 pass (`tests/sr/v7/`). Covers math primitives, state management, spawner mechanics, model composition, loss components, dataset adapter, and full training step.

## What the stack does end-to-end

```
LR (9-channel: rgb + depth + motion + normals)
  -> HAT-Tiny backbone (transformer, ~1.5M params)  [teacher]
  -> projection to feat_dim
  -> bilinear x2 upsample
  -> refined_hr
        |
        +-> BackboneSpawner: decode tiles -> K Gaussians (x, y, t, cov, feat, opacity)
        |     [adds to NDCanvasState at spawn_at_t]
        |
  -> canvas: render_nd_time_slice(t_query) -> canvas_hr (1, R, H, W)
        |
  -> composite_head: cat([refined_hr, canvas_hr]) -> delta_rgb (small-init)
  -> bicubic(LR_rgb) + delta_rgb -> SR_HR (1, 3, H, W)
```

Loss: `oss_fx_loss` supervises:
- Main output at t_query=2 against frame N+1 GT (SR objective).
- Intermediate output at t_query=1 against α=0.5 GT (OSS-FX objective, only meaningful once canvas has content from frames at t=0 and t=2).
- Foreground-aux + LPIPS-VGG perceptual + optional temporal consistency.

## Validation status

| Validation | Status |
|---|---|
| Math: 3D Gaussian conditional time-slice produces shifted output | ✅ `test_render_moving_gaussian_shifts_with_time_when_coupled` |
| Cholesky parameterization is PSD-by-construction | ✅ `test_cholesky_pack_to_cov_is_psd` |
| Cholesky covers all PSD covariances (round-trip) | ✅ tested at v6.3 spec time |
| Canvas state add / prune / compact / reset round-trip | ✅ 7 tests |
| Spawner materializes on opacity OR brightness threshold | ✅ `test_materialize_mask_fires_above_either_threshold` |
| `V7Model.forward` empty canvas == bicubic + tiny delta | ✅ `test_v7_model_forward_with_empty_canvas_returns_bicubic_anchored` |
| Canvas time-slice shifts output across `t_query` | ✅ `test_v7_model_forward_with_canvas_changes_output_at_different_t` |
| Backbone gradient flows | ✅ `test_v7_model_gradient_flow_through_backbone` |
| Spawner gradient flows | ✅ `test_v7_model_gradient_flow_through_spawner` |
| HAT-Tiny backbone slots in cleanly | ✅ `test_v7_model_with_hat_tiny_backbone_builds_and_forwards` |
| HAT-Tiny param count in pico-band (0.5M–4M) | ✅ `test_v7_model_hat_tiny_parameter_count_in_expected_range` |
| Full step (placeholder backbone) backward + step succeeds | ✅ `test_full_v7_training_step_runs_without_errors` |
| Full step (HAT-Tiny backbone) backward + step succeeds | ✅ `test_full_v7_training_step_with_hat_tiny_backbone` |
| Canvas count grows by spawner K each frame | ✅ `test_canvas_grows_through_two_spawns_during_step` |

## Outstanding before Phase 3 (full training run)

1. **TartanAir smoke test** on 3080 Ti — 50 steps with `--max-triplets 8` to confirm the trainer connects to the real dataset (the integration test synthesizes inputs). This is the Phase 2 spec deliverable that hasn't been exercised on remote: *"Smoke-test on TartanAir-subsampled at 1K steps."*  Should run in ~5 minutes once the remote tree is at `221f239` or later.
2. **Sync 3080ti-windows working tree** — remote is at `6b3d94d` (way behind). 472 dirty files there belong to an active capture-tool session; merging the v7 stack in requires coordination, not a fast-forward.
3. **Phase 3 plan memo** — exact ablation set, α-curriculum schedule (per spec: α=1 only for first 20K steps, then add α=0.5, then α=0.25/0.75), checkpoint cadence, eval cadence vs the OSS-FX metric.

## Notes / risks carried forward

- `BackboneSpawner` is **B=1 only** (per-rank canvas state). Trainer loops over batch items sequentially. This is fine at small batch sizes but is the obvious DDP bottleneck before Phase 4 CUDA kernels land.
- Spawner default `opacity_init_bias = -3` keeps newborn Gaussians faint (~0.05 opacity). That's deliberate (gradient signal raises opacity only on useful tiles), but rendering tests had to bump the bias manually to see canvas output. Training should be able to learn the right bias since `composite_head` consumes the canvas regardless of opacity at this stage.
- Composite head uses **small-init (std=1e-3)** on its last linear, so empty-canvas behavior is "bicubic + tiny perturbation" — bicubic-anchored start, which matches v6.x convention. Training has to learn weight magnitude to make canvas actually drive RGB output.
- The v7 stack does NOT touch v6.3 plans. v6.3-fine validation (frozen backbone + canvas-fusion + aux loss, see `2026-05-12-v63-fine-finetune-spec.md`) still runs as the bridge while pico-002 finishes and a separate v7-pico-005 host is provisioned.

## Tier matrix carried forward (verbatim from spec)

| Tier | Teacher (training) | Student (shipping) |
|---|---|---|
| Pico | HAT-Tiny (~1.5M, transformer) | OSS-Pico (≤0.4M, CNN) |
| Standard | HAT-Small (~4M, transformer) | OSS-Standard (≤1M, CNN) |
| Heavy | HAT-L (~17M, transformer) | OSS-Heavy (≤2M, CNN) |

Phase 3 = train the Pico **teacher** (HAT-Tiny) end-to-end with OSS-FX. Distillation to the CNN student happens after the teacher's metrics clear the v5/v6 bar.
