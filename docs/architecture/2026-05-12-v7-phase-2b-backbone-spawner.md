# v7 Phase 2B — Backbone → Canvas Spawner

**Filed:** 2026-05-12
**Status:** spec, queued behind Phase 2A (V7Model skeleton + dataset + losses ARE landed; spawner is the next module).
**Driver:** Phase 2A trained without canvas content, so OSS-FX intermediate frames were identical to SR frames. Phase 2B adds the learnable "spawn new Gaussians from backbone features each frame" module that makes the canvas actually carry temporal scene content.

## One-line claim

A small ConvNet head consumes backbone features at frame N, emits K Gaussian parameter predictions, and adds them to the canvas at t = N. Combined with the parent-child deferred-materialization spawner (already implemented at the unit-test level in `oss/sr/v7/parent_child_spawner.py`), the canvas grows densely + adaptively without explicit splitting heuristics.

## Module interface

`oss/sr/v7/backbone_spawner.py`:

```python
class BackboneSpawner(nn.Module):
    """Decodes backbone features (B, F, H, W) at frame N into K new
    Gaussian parameters to add to the NDCanvasState at t = N."""

    def __init__(
        self,
        feat_dim: int,
        latent_rank: int,
        k_per_frame: int = 256,
        tile_size: int = 16,
    ):
        ...

    def forward(
        self,
        refined_hr: torch.Tensor,    # (B, feat_dim, H, W)
        t: float,                     # absolute time coord for the new Gaussians
    ) -> dict:
        """Returns:
            positions  (K, 3)   xy at HR pixel coords + t copied from arg
            cov_raw    (K, 6)   Cholesky params predicted by the head
            features   (K, R)   per-Gaussian feature vector
            opacity    (K,)     in (0, 1) via sigmoid
        """
```

Design:

1. **Spatial sampling.** Tile the HR image into `tile_size × tile_size` tiles (default 16×16). For each tile, predict the parameters of `k_per_tile` Gaussians anchored within that tile. Total K = (H/tile_size) × (W/tile_size) × k_per_tile.
2. **Position prediction.** Per-tile, predict an (x, y) offset within the tile (sigmoid-bounded to [0, tile_size)), then add the tile's anchor pixel coordinate. This way Gaussians are distributed spatially without clustering at the image's top-left.
3. **Time anchor.** All K Gaussians at frame N share `t = N` exactly. Their *temporal extent* is governed by the predicted L22 (Cholesky diagonal at the t dimension).
4. **Covariance prediction.** Per Gaussian, predict 6 raw Cholesky params (l00, L10, l11, L20, L21, l22). Unconstrained (per the v6.3 covariance test finding: sigmoid bound was too restrictive).
5. **Feature prediction.** A small Conv1×1 over the per-tile features produces R-channel feature vectors.
6. **Opacity prediction.** Sigmoid-bounded scalar per Gaussian. Initialized to bias=-3 so opacities start ~0.05, low enough to avoid disrupting initial training but high enough for the parent-child spawner to materialize them.

## Integration into V7Model.forward

```python
def forward(self, lr_inputs, t_query, frame_index, output_hw=None):
    refined_hr = self.backbone(lr_inputs)
    if self.training:
        # Spawn K new Gaussians from this frame's backbone features
        spawned = self.spawner(refined_hr, t=float(frame_index))
        self.canvas.add(
            positions=spawned["positions"],
            cov_raw=spawned["cov_raw"],
            features=spawned["features"],
            opacity=spawned["opacity"],
        )
    canvas_hr = self.render_canvas(t_query=t_query, output_hw=output_hw)
    ...
```

Inference-time behavior: `spawned` ALSO runs at inference (because spawner is part of the model state we need for the canvas to carry frame N's content). The canvas accumulates per-frame just like v6.x.

## Integration with parent-child spawner

Each step:

1. BackboneSpawner adds K new Gaussians from current frame features → canvas grows by K.
2. Parent-child spawner's children for ALL existing canvas Gaussians drift via gradient updates.
3. Every ~300 training steps, `materialize_to_canvas()` is called: children whose opacity/brightness crossed threshold get promoted to full Gaussians.
4. The canvas's `prune` method can be called periodically to garbage-collect low-opacity Gaussians (keeps capacity bounded).

This combination gives capacity growth at TWO time scales:

- **Per-frame**: BackboneSpawner adds ~256 Gaussians (frame-local content).
- **Per-N-steps**: ParentChildSpawner promotes children where loss demands MORE capacity (high-error regions).

## Testing plan

`tests/sr/v7/test_backbone_spawner.py`:

1. **Shape correctness**: feed random refined_hr → spawner returns positions (K, 3), cov_raw (K, 6), features (K, R), opacity (K,) with K = (H/tile_size) × (W/tile_size) × k_per_tile.
2. **Positions are within image bounds**: predicted xy ∈ [0, W) × [0, H).
3. **Positions cluster around their tile anchors**: each predicted Gaussian's xy is within its assigned tile rect.
4. **Covariance is PSD by construction**: `cholesky_pack_to_cov(spawned["cov_raw"])` produces PSD matrices for all K.
5. **Opacities are in (0, 1)** via sigmoid.
6. **Gradient flow**: backprop a loss through canvas_hr touches spawner params.

`tests/sr/v7/test_v7_model_integration.py`:

7. **End-to-end forward with spawner**: V7Model with spawner builds non-empty canvas after one forward, render at α=0.5 produces visibly different output from α=1.

## Estimated work

- BackboneSpawner module: ~150 LoC + 6 tests
- V7Model.forward wiring: ~20 LoC
- Spawner config in V7Config + V7Model.allocate_canvas changes: ~15 LoC
- Tests + ckpt round-trip: ~120 LoC
- **Total: ~300 LoC across 4 files**

One sitting; lands in tree before pico-002 finishes.

## What this still does NOT solve

- Pre-trained backbone (HAT-Tiny) is still a placeholder ConvNet in V7Model. Phase 2C swaps for the real backbone (HAT-Tiny or whatever v7-pico-005 picks).
- Canvas capacity management at runtime (pruning policy, what gets evicted on overflow). Phase 2C item.
- The CUDA / Triton rasterizer kernel. Phase 4 of the v7 spec.
- N-D LSH culling kernel. Phase 4.

## Why this is the next thing to build

Phase 2A landed all the math (rasterizer, canvas state, parent-child spawner, intermediate dataset, losses). The training scaffold (`scripts/sr_train_v7.py`) runs end-to-end but the canvas stays empty, so OSS-FX intermediate frames don't differ from SR frames. Phase 2B fills the canvas — first time the v7 stack can produce real intermediate-frame predictions on real data.

After Phase 2B lands, the next decision point is:

- **Smoke test**: run sr_train_v7 for 100 steps with the spawner on, verify canvas count grows, training loss decreases.
- **First v7 training run**: scale to 100K steps as `srcnn-v7.0-pico-005` on the 3080 Ti. Same recipe as pico-002 but with v7 architecture. ~6 days wall-clock.

The compute estimate stays at $600–$1K spot for pico-tier first cycle per the v7 spec.
