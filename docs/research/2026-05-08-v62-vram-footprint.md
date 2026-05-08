# v6.2 VRAM Footprint Probe

Date: 2026-05-08
Owner: D3

## Question

Measure total inference VRAM footprint for the v6.2 model using the
v6.2-pico-002 config, `--v62` mode, and `R=4`, with a forward pass at:

| Target output | LR input |
|---|---|
| 1080p | 540x960 |
| 1440p | 720x1280 |
| 4K | 1080x1920 |

The requested measurement is allocator-backed: `torch.cuda.memory_summary()`
plus peak allocator stats.

## Outcome

No valid VRAM values were measured in this workspace.

| Target output | LR input | Peak allocated MiB | Peak reserved MiB | Status |
|---|---:|---:|---:|---|
| 1080p | 540x960 | TBD | TBD | blocked |
| 1440p | 720x1280 | TBD | TBD | blocked |
| 4K | 1080x1920 | TBD | TBD | blocked |

## Blockers

1. The default local Python environments do not have PyTorch installed.

   Attempt:

   ```bash
   python - <<'PY'
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   PY
   ```

   Result:

   ```text
   ModuleNotFoundError: No module named 'torch'
   ```

   Same result with `.venv/bin/python` and `venv/bin/python`.
   `venv-py312/bin/python` does have PyTorch 2.11.0, but only with CPU/MPS
   support:

   ```text
   torch 2.11.0
   torch.cuda.is_available() False
   torch.version.cuda None
   ```

2. This host does not expose a CUDA measurement path. The task target is the
   3080 Ti host, but this workspace is on the local macOS checkout.

3. The v6.2 model orchestrator wiring is not present in `oss/sr/v6/model.py`.
   The current `V6Config` lacks the fields required by the v6.2-pico-002
   contract:

   ```text
   fusion_mode missing
   latent_rank missing
   spawner_mode missing
   ```

   Partial v6.2 components exist (`ConcatFusion`, `DisocclusionSpawner`,
   `LatentDecoder`, and `V6Rasterizer(latent_rank=...)`), but `V6Model.forward`
   still uses the legacy global attention and `composite_head` path:
   `self.fusion(feats, tokens)` then `self.composite_head(cat(refined_hr, canvas_hr))`.

4. The launch config and checkpoint are unavailable in this checkout:

   ```text
   configs/v6.2-pico-002.yaml: missing
   step-*.pt v6.2 checkpoints: missing
   ```

   The only local checkpoint-like files found under depth 4 were older
   ORU/ORD/Pico artifacts under `results/`.

5. `scripts/sr_inference_vram.py` is for the older SR-CNN `build_sr_model`
   path. It does not load `V6Model`, does not understand `--v62`, and cannot
   exercise the v6.2 canvas/raster/fusion pipeline.

## Helper Added

Added `scripts/sr_v62_vram_probe.py`, following the existing
`scripts/sr_inference_vram.py` style.

It is intentionally fail-fast:

- requires CUDA
- requires `V6Config` to expose `latent_rank`, `spawner_mode`, and `fusion_mode`
- loads a `step-*.pt` checkpoint from `--output-dir` or an explicit
  `--checkpoint`
- runs the three requested LR sizes
- writes `artifacts/v62-vram/v62-vram-results.json`
- writes one `torch.cuda.memory_summary()` dump per resolution

Local syntax check:

```bash
python -m py_compile scripts/sr_v62_vram_probe.py
```

Result: passed.

Local execution attempt:

```bash
.venv/bin/python scripts/sr_v62_vram_probe.py \
  --output-dir /e/checkpoints/srcnn-v6.2-pico-002 \
  --summary-dir /tmp/v62-vram
```

Result:

```text
ModuleNotFoundError: No module named 'torch'
```

## Repro Command For 3080 Ti

Once T6/T7/T8 have landed and `srcnn-v6.2-pico-002` has at least one
checkpoint:

```bash
ssh 3080ti-windows '"C:\Program Files\Git\bin\bash.exe" -lc "cd /e/oss-gaussian-server && git pull --ff-only origin main && python scripts/sr_v62_vram_probe.py --v62 --output-dir /e/checkpoints/srcnn-v6.2-pico-002 --device cuda --dtype fp16 --backbone hat-tiny --latent-rank 4 --canvas-capacity 16000 --summary-dir /e/checkpoints/srcnn-v6.2-pico-002/vram-footprint"'
```

Expected outputs:

```text
/e/checkpoints/srcnn-v6.2-pico-002/vram-footprint/v62-vram-results.json
/e/checkpoints/srcnn-v6.2-pico-002/vram-footprint/1080p-output.memory_summary.txt
/e/checkpoints/srcnn-v6.2-pico-002/vram-footprint/1440p-output.memory_summary.txt
/e/checkpoints/srcnn-v6.2-pico-002/vram-footprint/4k-output.memory_summary.txt
```

## Gate Status

| Gate | Status |
|---|---|
| CUDA environment available locally | FAIL |
| CUDA-capable PyTorch available locally | FAIL |
| v6.2 `V6Model` config surface landed | FAIL |
| `configs/v6.2-pico-002.yaml` present | FAIL |
| v6.2-pico-002 checkpoint present | FAIL |
| VRAM table measured | FAIL |
