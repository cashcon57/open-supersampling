# OSS + Gaussian Temporal Canvas: Integration Points

**Date:** 2026-05-01  
**Scope:** Mapping parallel Gaussian-based track (`oss/gaussian/`) integration with existing pixel-based OSS modules.  
**Status:** Pre-alpha (OSS v0.2.0.dev0, Gaussian track scaffolding ready)

---

## Executive Summary

The existing OSS codebase (OSS, OSSRG, OSSPico, OSSFx) follows a strict tiered-model pattern with unified training, inference, and export pipelines. The Gaussian temporal canvas should remain functionally independent while reusing infrastructure where tight ABI contracts exist: dataset loaders, training harness dispatch, inference backend dispatcher, and test scaffolding.

**Key decision:** Keep `oss/gaussian/` as a **parallel track** that mirrors OSS patterns (train, inference, export, bench) rather than merging into shared modules. This avoids breaking the frozen handoff contracts (e.g., `HANDOFF_FEATURE_CHANNELS=32`) and allows Gaussian components to evolve independently.

---

## 1. Reusable Pieces (Import/Wrap)

### 1.1 **Dataset Loaders** (oss/data/)

**Can reuse:**
- `oss.data.noisebase.NoiseBaseDataset` — Gaussian temporal models may want LR multi-frame sequences with GT + aux buffers. The current NoiseBase loader emits `{color_lr, gt_hr, motion_lr, depth_lr, normals_lr, albedo_lr}` per frame, which Gaussian can consume directly.
- `oss.data.sintel_fx.SintelFxDataset` — Real optical flow for temporal consistency training. Gaussian extrapolation benefits from the same Sintel sequences.
- `oss.data.vimeo90k_fx.Vimeo90kDataset` — Diverse real-world motion. Can feed both OSS-FX and Gaussian temporal modules.

**Recommendation:**
- `oss/gaussian/` should **import and reuse** `oss.data` loaders, not fork them.
- If Gaussian-specific preprocessing (e.g., frame stacking, multi-scale splat) is needed, wrap loaders in a new `oss/gaussian/data/` module that inherits from `oss.data` base classes.

### 1.2 **Loss Functions** (oss/train/losses.py)

**Can reuse:**
- `CompositeLoss` — relative_l2, SSIM, LPIPS, wavelet components are generic.
- `temporal_consistency_loss` — Enforces frame-to-frame coherence; directly applicable to Gaussian canvas prediction.
- `wavelet_loss` — For high-frequency detail preservation in splat/rendering outputs.

**Recommendation:**
- Import `oss.train.losses` directly in `oss/gaussian/train/`.
- If Gaussian-specific losses emerge (e.g., splat-coherence penalty, Gaussian-specific KL divergence), add them to `oss/gaussian/train/losses_gaussian.py` and keep `oss.train.losses` unchanged.

### 1.3 **Inference Backend Dispatcher** (oss/infer.py)

**Can reuse:**
- `InferenceSession` class — Auto-selects TensorRT, ORT variants, NCNN, CoreML, OpenVINO based on available hardware.
- Backend selection logic (`_select_backend`, `_BACKEND_BUILDERS` dict).
- TensorRT caching per GPU arch (`.cache/oss/engines/`).

**Recommendation:**
- **Share** `InferenceSession` verbatim. It accepts any ONNX or CoreML model; Gaussian exports are indistinguishable at the dispatcher level.
- Both OSS and Gaussian checkpoints should export to the same ONNX opset (17) and CoreML MLPackage format.
- Keep Gaussian export logic (`oss/gaussian/export/`) separate, but call the same backend builders.

### 1.4 **Model Building Blocks** (oss/model/blocks.py)

**Can reuse:**
- `ConvBlock` (Conv→GroupNorm→SiLU)
- `DownBlock`, `UpBlock` (stride-2 down, bilinear up)
- Channel assertions (multiples of 8 for cooperative-matrix tiling)

**Recommendation:**
- Gaussian network architectures should use the same blocks; avoids divergent optimization patterns across the codebase.
- If Gaussian needs Attention or MLP blocks, add them to `oss.model.blocks` with a Gaussian-specific prefix (e.g., `GaussianAttentionBlock`).

### 1.5 **Evaluation Harness** (oss/bench/)

**Can reuse:**
- `QualityRunner` — Loads a checkpoint, runs inference on LR/GT pairs, scores with PSNR/SSIM/LPIPS/timing.
- `fsr1_reference.py` — FSR1 upscaler baseline for quality comparison.

**Recommendation:**
- Extend `QualityRunner` to support both OSS and Gaussian models via a `model_type` parameter.
- Keep `oss/bench/` as the shared evaluation entry point; let Gaussian-specific harnesses live in `oss/gaussian/bench/` if needed.

---

## 2. Conflict Points

### 2.1 **Module Naming & `__init__.py` Exports**

**Current state:**
```python
# oss/model/__init__.py
from .oss_rg import OSSRG, HANDOFF_FEATURE_CHANNELS
from .oss import OSS
__all__ = ["OSSRG", "OSS", "HANDOFF_FEATURE_CHANNELS"]
```

**Conflict:** If Gaussian defines its own `GaussianCanvas` model, importing both `from oss.model import OSS` and `from oss.gaussian.model import GaussianCanvas` is unambiguous — but having separate `oss/model/` and `oss/gaussian/model/` trees can confuse users about which is the canonical location.

**Solution:**
- Keep `oss/model/` for pixel-based (OSS, OSSRG, OSSPico, OSSFx).
- Create `oss/gaussian/model/` for Gaussian-based components (GaussianCanvas, GaussianClassifier, etc.).
- In `oss/gaussian/__init__.py`, explicitly export Gaussian models:
  ```python
  from .model import GaussianCanvas
  __all__ = ["GaussianCanvas"]
  ```

### 2.2 **Training Script Naming**

**Current state:**
- `oss/train/train_sr.py` — ORU standalone
- `oss/train/train_rg.py` — ORD (OSSRG) standalone
- `oss/train/train_paired.py` — ORD + ORU two-stage
- `oss/train/train_pico.py` — OSSPico
- `oss/train/train_fx.py` — OSSFx

**Conflict:** If Gaussian has its own trainer, avoid naming collisions. Do not create `oss/train/train_canvas.py` at the OSS level.

**Solution:**
- Put Gaussian trainers in `oss/gaussian/train/` with clear names:
  - `oss/gaussian/train/train_canvas.py`
  - `oss/gaussian/train/train_temporal.py`
  - etc.
- Cloud launchers in `scripts/` can dispatch to either tree: `scripts/lambda_train_gaussian_canvas.py` vs. `scripts/lambda_train_pico.py`.

### 2.3 **Handoff Feature Contract**

**Current state:**
```python
# oss/model/oss_rg.py
HANDOFF_FEATURE_CHANNELS = 32
```
This is a frozen contract consumed by OSS (in "features" mode) and exported/cached globally.

**Conflict:** Gaussian temporal canvas might benefit from a different feature width (e.g., 64 channels for richer temporal information).

**Solution:**
- **Do not** override `HANDOFF_FEATURE_CHANNELS` in Gaussian code.
- If Gaussian needs a different feature contract, define it locally:
  ```python
  # oss/gaussian/model/canvas.py
  GAUSSIAN_TEMPORAL_FEATURE_CHANNELS = 64
  ```
- Keep OSS and Gaussian feature paths separate; do not try to pair them at export time.

### 2.4 **Tier/Scale Naming**

**Current state:**
- Tiers: `lite`, `standard`, `heavy`
- Scale factors: `1.3, 1.5, 1.7, 2.0`

**Conflict:** Gaussian canvas may not need tiering (e.g., one fixed size) or use different tier semantics (e.g., temporal window depth instead of parameter count).

**Solution:**
- Gaussian tier system is independent. Define in `oss/gaussian/model/config.py`:
  ```python
  GAUSSIAN_TIER_CONFIGS = {
      "light": {"splat_features": 128, "temporal_depth": 3, ...},
      "standard": {"splat_features": 256, "temporal_depth": 5, ...},
  }
  ```
- Do not force Gaussian to inherit OSS tier naming; let it evolve independently.

---

## 3. Recommended Integration Pattern

### 3.1 **Directory Layout**

```
oss/
├── model/                  # Pixel-based (OSS, OSSRG, OSSPico, OSSFx)
├── train/                  # Shared training primitives (data.py, losses.py, ...)
├── inference/              # Shared backend dispatcher
├── export/                 # ONNX/CoreML export (extend for Gaussian)
├── bench/                  # Shared eval harness
├── gaussian/               # Parallel Gaussian-based track
│   ├── __init__.py
│   ├── model/              # GaussianCanvas, GaussianClassifier, ...
│   ├── train/              # Gaussian-specific trainers (inherit from oss.train)
│   ├── inference/          # Gaussian-specific runners (optional, mostly uses oss.infer)
│   ├── export/             # Gaussian model export (can wrap oss.export)
│   └── bench/              # Gaussian-specific eval
```

### 3.2 **Training Pattern**

Example: Gaussian temporal canvas trainer inherits from OSS patterns.

```python
# oss/gaussian/train/train_canvas.py
from oss.data import NoiseBaseDataset
from oss.train import CompositeLoss, temporal_consistency_loss
from oss.gaussian.model import GaussianCanvas

def main():
    ds = NoiseBaseDataset(root=args.data, augment=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    
    model = GaussianCanvas(...).to(device)
    loss_fn = CompositeLoss(w_temporal=0.1)  # Use shared loss
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        for batch in dl:
            # oss.data loaders emit {color_lr, gt_hr, motion_lr, ...}
            pred_canvas = model(
                color_lr=batch["color_lr"],
                motion_lr=batch["motion_lr"],
                ...
            )
            loss = loss_fn(pred_canvas, batch["gt_hr"])
            opt.zero_grad()
            loss.backward()
            opt.step()
```

### 3.3 **Inference Pattern**

Gaussian exports ONNX (same as OSS). Inference dispatcher is shared:

```python
# User code
from oss.infer import InferenceSession

sess = InferenceSession("gaussian_canvas.onnx")  # Auto-selects backend
output = sess.run({"color_lr": arr, "motion_lr": arr2})
```

No Gaussian-specific dispatcher needed unless Gaussian models have radically different input/output contracts (e.g., multi-head or sequence inputs). Keep it simple initially.

### 3.4 **Export Pattern**

```python
# oss/gaussian/export/export_gaussian.py
from oss.gaussian.model import GaussianCanvas

def export_gaussian_canvas(model, out_path, ...):
    # Similar to oss/export/export_oss.py
    torch.onnx.export(
        model,
        dummy_inputs,
        str(out_path),
        opset_version=17,
        input_names=[...],
        output_names=[...],
        dynamic_axes={...},  # Batch + spatial dynamic
    )
    # Optionally validate parity with oss.export utilities
```

---

## 4. Test Harness Reuse

### 4.1 **Existing Test Infrastructure**

**Location:** `tests/`

**Existing markers:**
```python
@pytest.mark.gpu  # Requires CUDA GPU
@pytest.mark.mitsuba  # Requires Mitsuba 3
```

**Can reuse:**
- PyTest fixture setup (device selection, checkpoint staging)
- Smoke-test patterns (synthetic batches, quick forward passes)
- ONNX parity validation (via `test_onnx_parity.py`)
- Benchmark timing harness (via `test_safety_harness.py`)

### 4.2 **Gaussian Test Strategy**

- **Smoke tests:** `tests/test_gaussian_smoke.py` — forward pass on synthetic data, no GPU.
- **Parity tests:** `tests/test_gaussian_onnx_parity.py` — PyTorch vs ONNX outputs.
- **Temporal tests:** `tests/test_gaussian_temporal_consistency.py` — verify frame-to-frame coherence.
- **Dataset tests:** `tests/test_gaussian_data.py` — Gaussian-specific data loading (reuses base loaders).

**Recommendation:**
- Integrate Gaussian tests into the same `tests/` directory, not a separate `oss/gaussian/tests/`.
- Use the same `conftest.py` fixtures and markers.
- Add a `--gaussian` command-line filter to pytest for selective runs:
  ```
  pytest tests/ -k "not gaussian"  # Run pixel-based tests only
  pytest tests/ -k "gaussian"      # Run Gaussian tests only
  ```

---

## 5. Scripts to Share / Fork

### 5.1 **Cloud Training Launchers**

**Current:**
- `scripts/lambda_train_pico.py` — Lambda Labs launcher for OSSPico
- `scripts/runpod_train_pico.py` — RunPod launcher for OSSPico
- `scripts/lambda_stage_noisebase.py` — Staging NoiseBase to cloud storage
- `scripts/runpod_setup_check.py` — Sanity check for RunPod env

**Decision:**
- **Share:** `scripts/lambda_stage_noisebase.py` and `scripts/runpod_setup_check.py` are dataset-agnostic and can be reused verbatim.
- **Fork:** Create Gaussian-specific launchers:
  - `scripts/lambda_train_gaussian_canvas.py`
  - `scripts/runpod_train_gaussian_canvas.py`
  These inherit the structure and config patterns from OSS launchers but invoke `oss.gaussian.train.train_canvas` instead of `oss.train.train_pico`.

### 5.2 **Dataset Download Scripts**

**Current:**
- `scripts/download_sintel.sh`
- `scripts/download_vimeo90k.sh`

**Decision:**
- **Share:** Both Gaussian and OSS-FX use Sintel and Vimeo-90K. Keep these in `scripts/`.
- The NoiseBase download (not yet in `scripts/`) should support both when added.

### 5.3 **Benchmark / Evaluation Scripts**

**Current:**
- `scripts/bench_quality.py` — Generic quality runner

**Decision:**
- **Extend** `scripts/bench_quality.py` with a `--model` parameter to support both OSS and Gaussian checkpoints:
  ```bash
  python scripts/bench_quality.py --model oss_pico --ckpt results/oss_pico.pth --data data/test.exr
  python scripts/bench_quality.py --model gaussian_canvas --ckpt results/gaussian_canvas.pth --data data/test.exr
  ```

---

## 6. CI/CD and Build Integration

### 6.1 **PyProject.toml Updates**

**Current entry points:**
```toml
[project.optional-dependencies]
cuda = ["tensorrt>=10.0", "onnxruntime-gpu>=1.18"]
coreml = ["coremltools>=8.0"]
vulkan = ["ncnn>=1.0.20240102", "pnnx>=20240410"]
```

**Decision:**
- No new optional dependencies for Gaussian (initially). Gaussian models should train/infer with the same backends as OSS.
- If Gaussian gains specialized requirements (e.g., `pytorch3d` for splat rendering), add them:
  ```toml
  gaussian = ["pytorch3d>=0.7"]
  ```

### 6.2 **Test Markers**

Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "gpu: requires CUDA GPU",
    "mitsuba: requires Mitsuba 3",
    "gaussian: Gaussian temporal canvas tests",
]
```

---

## 7. Summary: Do's and Don'ts

### Do:
- ✓ Reuse `oss.data`, `oss.train.losses`, `oss.infer.InferenceSession`, `oss.model.blocks`.
- ✓ Keep Gaussian models in `oss/gaussian/model/` separate from pixel-based `oss/model/`.
- ✓ Export Gaussian to ONNX (opset 17) and CoreML for dispatch to `InferenceSession`.
- ✓ Use shared test infrastructure (`tests/`) with `@pytest.mark.gaussian`.
- ✓ Extend cloud launchers (Lambda/RunPod) with Gaussian-specific variants in `scripts/`.

### Don't:
- ✗ Modify `HANDOFF_FEATURE_CHANNELS` — keep it frozen at 32 for OSS.
- ✗ Fork dataset loaders; import `oss.data` and wrap if Gaussian-specific preprocessing is needed.
- ✗ Add Gaussian code to `oss/train/` or `oss/model/` — keep those directories pixel-centric.
- ✗ Create separate test directories (`oss/gaussian/tests/`) — consolidate in `tests/`.
- ✗ Redefine tier semantics; let Gaussian evolve independently if tier structure differs.

---

## Deliverables Checklist

- [x] Reusable pieces documented (dataset loaders, losses, backend dispatcher, blocks, eval harness)
- [x] Conflict points identified and resolved (naming, feature contracts, tier systems)
- [x] Recommended integration pattern provided (directory layout, training/inference/export examples)
- [x] Test harness reuse strategy clarified (pytest markers, fixture sharing)
- [x] Cloud scripts sharing/forking strategy outlined (shared utility scripts, Gaussian-specific launchers)
- [x] CI/CD integration guidance provided (pyproject.toml, pytest markers)

---

**End of Integration Points Report**
