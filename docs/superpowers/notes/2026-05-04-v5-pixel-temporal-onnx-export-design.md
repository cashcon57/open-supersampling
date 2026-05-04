# v5 pixel-temporal — ONNX export design memo

**Date:** 2026-05-04
**Status:** scaffold landed, export script deferred until v5 ckpt is final
**Owner:** SR / runtime
**Related:** `oss/sr/temporal/stateless_export.py`, `oss/sr/inference.py::TemporalSRInferenceEngine`, `scripts/sr_export_tensorrt.py`

## Problem

The training-time module `oss.sr.temporal.TemporalSRModel` is a clean, stateless `nn.Module`: every prev-frame tensor flows in through the forward signature. The runtime path, `oss.sr.inference.TemporalSRInferenceEngine`, is intentionally stateful — it caches `prev_hr` and `prev_depth_hr` as Python attributes between frames so the host application only has to feed *current* frame data.

`torch.onnx.export` traces the `forward` of an `nn.Module`. It cannot capture Python attributes that mutate between calls. Therefore the engine's stateful pattern is unsuitable for ONNX/TRT/DirectML/OpenVINO consumers — every cross-frame tensor must be an explicit graph input.

## Design

### Stateless wrapper

`oss.sr.temporal.stateless_export.TemporalSRModelStateless(nn.Module)` wraps an existing `TemporalSRModel` instance and exposes a 5-input, 2-output forward:

```
forward(
    lr_inputs:     (B, in_channels, H_lr, W_lr) float
    prev_hr:       (B, 3,           H_hr, W_hr) float
    depth_hr_curr: (B, 1,           H_hr, W_hr) float
    depth_hr_prev: (B, 1,           H_hr, W_hr) float
    motion_lr:     (B, 2,           H_lr, W_lr) float
) -> (
    out_hr:        (B, 3,           H_hr, W_hr) float
    disocclusion:  (B, 1,           H_hr, W_hr) float in [0, 1]
)
```

Where `H_hr = scale * H_lr`. The wrapper is a *pure passthrough*: it calls `self.model.backbone`, `self.model.gate`, `self.model.head`, and `warp_prev_hr` in exactly the same order as the stateful `TemporalSRModel.forward`, so the math cannot drift relative to training.

The disocclusion mask is exposed as a second output so deployment dashboards can visualize the gate and so the runtime can use it for debug overlays / scene-cut UX without re-running the gate. Consumers who don't want it can simply ignore the second output.

### First-frame init contract

On the *very first* frame of a sequence (or after a scene cut) the runtime caller is responsible for synthesizing the `prev_hr` buffer using:

```
prev_hr_init = make_first_frame_prev_hr(lr_rgb, scale)
```

which is a bilinear LR→HR upsample of the current LR RGB. This matches the stateful engine's `_prev_hr is None` branch (`oss/sr/inference.py:318`), so behavior at sequence boundaries is identical between the stateful and stateless paths. `depth_hr_prev` on the first frame is the current depth (engine convention; gives `depth_diff = 0`, so the gate fires only on `motion_mag`).

### Buffer ownership at runtime

The export removes the engine's internal buffer; *something* on the consumer side must take ownership.

Two options were considered:

| Option | Pros | Cons |
|---|---|---|
| **A. Game-engine integration layer owns it** | Lets the host pick the right memory pool (D3D12 / Vulkan resource), avoids a copy back to ORT/TRT IO buffers each frame, supports scene-cut signaling that the host already knows about (camera teleport, level load) without any heuristic. | More surface for misuse; integration guide must spell out reset / first-frame contract. |
| **B. Thin C++/Python "session" wrapper around the TRT engine retains it** | Easier API for prototyping. | Defeats the point — TRT engine I/O is per-call; we'd just be re-implementing the engine state in another layer with the same statefulness problem. |

**We pick option A.** The game-integration layer owns `prev_hr`. Rationale: scene cuts are a host concern (the host knows about camera teleport / level load deterministically; motion-magnitude heuristics are a fallback, not the primary signal). The host also has zero-copy access to its own framebuffers, which is the whole reason DLSS-style pipelines pin GPU memory across frames. The `TemporalSRInferenceEngine` retains its statefulness for in-process Python users (training viz, eng dashboards, A/B harness); the stateless wrapper exists *purely* for cross-runtime export.

Documented runtime contract (will live in the integration guide once the export script lands):

1. **Frame 0 / scene cut:** allocate `prev_hr` (or zero its existing buffer) and fill via `make_first_frame_prev_hr(lr_rgb_t, scale)`. Set `depth_hr_prev = depth_hr_curr`. Run the model.
2. **Frame N (N ≥ 1):** feed previous frame's `out_hr` as `prev_hr`, previous frame's `depth_hr_curr` as `depth_hr_prev`. Run the model.
3. **Scene-cut signal:** the host calls "reset" (= treat the next frame as Frame 0). A motion-magnitude fallback may live in the integration layer mirroring `scene_cut_motion_threshold=32.0` from the engine; it is *not* baked into the ONNX graph.

### ONNX export plan (deferred)

Will live at `scripts/sr_export_temporal_onnx.py`. Requires CUDA + a real v5 ckpt; out of scope for this scaffold task.

```python
torch.onnx.export(
    wrapper,                                  # TemporalSRModelStateless, training=False
    (lr_inputs, prev_hr, depth_curr, depth_prev, motion_lr),
    str(out_path),
    input_names=["lr_inputs", "prev_hr", "depth_hr_curr", "depth_hr_prev", "motion_lr"],
    output_names=["out_hr", "disocclusion"],
    opset_version=17,
    dynamic_axes={
        "lr_inputs":     {2: "H_lr", 3: "W_lr"},
        "motion_lr":     {2: "H_lr", 3: "W_lr"},
        "prev_hr":       {2: "H_hr", 3: "W_hr"},
        "depth_hr_curr": {2: "H_hr", 3: "W_hr"},
        "depth_hr_prev": {2: "H_hr", 3: "W_hr"},
        "out_hr":        {2: "H_hr", 3: "W_hr"},
        "disocclusion":  {2: "H_hr", 3: "W_hr"},
    },
    do_constant_folding=True,
)
```

Batch is fixed at 1, matching the existing single-frame export (`scripts/sr_export_onnx.py`). H_lr / W_lr are dynamic; H_hr / W_hr are dynamic but linked at runtime by the `scale * H_lr = H_hr` invariant which is *not* expressed in the ONNX graph (the graph trusts the caller to pass shape-consistent tensors — same convention as the v3 export).

### Op-by-op exportability audit

| Op | Where | Exportable @ opset 17? |
|---|---|---|
| `F.interpolate(mode="bilinear")` | motion HR upscale, first-frame init | yes (Resize) |
| `F.grid_sample(mode="bilinear", padding_mode="border")` | `warp_prev_hr` for RGB and depth | yes (GridSample, opset 16+) |
| `torch.meshgrid` + arithmetic | `warp_prev_hr` grid construction | yes (Range + arithmetic) |
| `tensor.norm(dim=1, keepdim=True)` | motion magnitude in gate | yes (ReduceL2) |
| `torch.sigmoid` | gate output | yes |
| Conv blocks + PixelShuffle | backbone + head | yes (already exported in v3/v4) |

No custom ops, no graph-breaks.

### Known limitations (PyTorch 2.4.1 + opset 17)

- **Bicubic with antialias:** un-exportable (PyTorch doesn't emit the antialiased path through ONNX). We deliberately use plain **bilinear** in `upsample_motion_to_hr` and `make_first_frame_prev_hr`. This matched the v3/v4 decision recorded in commit `d12baea` and the README.
- **`torch.compile`:** must be disabled at export time (`do_constant_folding=True` only). The training script does not use `torch.compile` on the stateful model so this is a non-issue.
- **fp16 export:** PyTorch's preference is to export in fp32 then convert (or rely on TRT for the fp16 cast). We follow `scripts/sr_export_tensorrt.py` and let TRT handle the precision conversion via `trt_fp16_enable=True`.

### TRT FP16 plan

Reuse the ORT-TensorrtExecutionProvider pattern from `scripts/sr_export_tensorrt.py`:

- `trt_fp16_enable=True`
- `trt_engine_cache_enable=True`
- `trt_max_workspace_size=2GB`
- Profile shapes built with the same narrow LR resolution targets (Steam Deck 800x1280, 720p, 900p, 1080p) plus their HR pairs at scale=2.

The narrow profile keeps the TRT plan small (no shape inference fallback). Multi-resolution support is out of scope for the first export — the integration team can request a wider profile once the v5 numbers land and we know which res tier to ship first.

### Validation strategy (for the export script, not this scaffold)

1. Numerical: compare `wrapper(...)` (PyTorch fp32) vs ORT(CPU, fp32) on a canonical 4-frame sequence (frame 0 = first-frame init, frames 1–3 = roll prev_hr forward in Python). PSNR ≥ 50 dB on each frame.
2. Numerical: ORT(CUDA, fp16) ≥ 30 dB vs PyTorch fp32. Same gate as v3/v4.
3. Functional: scene-cut behavior — feeding a synthetic discontinuity (random `motion_lr`, prev_hr from unrelated frame) must not NaN; output should largely match a frame-0 init within a few dB.
4. Shape: dynamic axes work at all four target resolutions.
5. Disocclusion: second output has the right shape and is in [0, 1].

## Out of scope

- The actual export script (`scripts/sr_export_temporal_onnx.py`) — needs a real v5 ckpt + GPU box.
- DirectML / OpenVINO benchmarks.
- C++ / native game-integration sample code.
- INT8 PTQ for the temporal model (defer until FP16 is validated; the current pipeline calibrator in `scripts/sr_export_trt_int8.py` is single-frame and would need a temporal-aware calibration data loader).

## Risks

- `grid_sample` with `padding_mode="border"` performance on TRT FP16: historically fine but worth a microbenchmark on the target res before committing to it. Fallback: `padding_mode="zeros"` plus a learned mask, but that's a model change, not an export change.
- Disocclusion gate's three learnable scalars (`alpha`, `beta`, `gamma`) become constants at export time. If the runtime needs to retune them per-game, we'd need to expose them as graph inputs — currently not planned.
- HR depth is currently passed in by the runtime. If the host only has LR depth, we'd want to fold the LR→HR upsample inside the graph; this is an additive change to the wrapper signature and is deferred.
