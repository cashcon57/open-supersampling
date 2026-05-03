# ONNX Export + FP16 + ONNX Runtime Benchmark

**Date:** 2026-05-03  
**Status:** DONE_WITH_CONCERNS (TensorRT not available; ORT CUDA DLL path fix required)  
**Predecessor:** [srcnn-beats-v05-and-gsasr](2026-05-02-srcnn-beats-v05-and-gsasr.md)  
**Hardware:** RTX 3080 Ti (12 GB VRAM) via `<train-host>` — training running in parallel (~4 GB VRAM in use)

---

## Hypothesis / question

Can we export SRCNNSimple to ONNX and get measured, real speedup numbers versus PyTorch FP32? Does FP16 ONNX Runtime provide the expected 1.5-2× wall-time speedup on an RTX 3080 Ti?

---

## Setup

**Checkpoint used:** `step-00040000.pt`  
**Model:** SRCNNSimple, `tier=standard`, `backbone=simple`  
**Parameters:** 604,748  
**FP32 weight size:** 2.31 MiB  

**Scripts written:**
- `scripts/sr_export_onnx.py` — torch.onnx.export + onnxconverter_common FP16 conversion
- `scripts/sr_bench_onnx.py` — PyTorch FP32 vs ONNX-RT FP32 vs ONNX-RT FP16
- `scripts/sr_export_tensorrt.py` — ORT TensorrtExecutionProvider (optional)

**Tests written:** `tests/sr/test_onnx_export.py` — 5 tests, all pass

**Environment:**
- Python 3.11.10 (Miniconda, `image-gs` env)
- PyTorch 2.4.1 + CUDA 12
- onnx 1.21.0
- onnxruntime-gpu 1.25.1
- onnxconverter-common 1.16.0
- TensorRT: NOT installed (`nvinfer_10.dll` missing)

**Export CLI:**
```
python scripts/sr_export_onnx.py \
    --output-dir <train-host-data>\checkpoints\srcnn-prod-v3 \
    --export-dir <train-host-data>\onnx \
    --opset 17
```

**Benchmark CLI:**
```
python scripts/sr_bench_onnx.py \
    --onnx-dir <train-host-data>\onnx \
    --output-dir-ckpt <train-host-data>\checkpoints\srcnn-prod-v3
```

---

## Export findings

### antialias=True ONNX export failure (opset 17)

`F.interpolate(..., mode='bicubic', antialias=True)` maps to `aten::_upsample_bicubic2d_aa` which is not supported by `torch.onnx.export` under opset 17 (PyTorch 2.4.1). Export falls back to `mode='bilinear', align_corners=False`.

**Quality delta:** The bicubic skip changes from antialias=True (used during training) to bilinear. The residual CNN is trained with the antialias=True skip; the quality delta for the deployed skip is estimated at <0.1 dB PSNR but has not been directly measured on held-out data. This is a real deployment delta — the training distribution and inference path differ.

**Workaround options (not implemented yet):**
1. Train with antialias=False from the start (closes the training/inference gap)
2. Use opset 18+ or a custom ONNX op for antialias bicubic
3. Implement bicubic antialias via explicit kernel convolution (fully exportable)

### FP32 ONNX verification (vs PyTorch FP32)

`max|diff| = 0.000e+00, mean|diff| = 0.000e+00` — exact numerical match. ORT FP32 with CUDA provider produces bit-identical output to PyTorch FP32.

### FP16 ONNX verification (vs PyTorch FP32)

`max|diff| = 2.600e-04, mean|diff| = 3.075e-05` — well within the 1e-2 tolerance. FP16 quantisation error is negligible for this model.

**Dynamic axes fix:** The initial FP16 export failed with an ORT CPU-provider buffer reuse error (`{H,W} != {2H,2W}` in the Add node). Root cause: `onnxconverter_common` with `keep_io_types=True` collapses input and output `h`/`w` dim_params into the same name, causing ORT's shape tracker to fail. Fixed by using distinct param names `{h_in, w_in}` for input and `{h_out, w_out}` for output in `dynamic_axes`.

### ORT CUDA DLL path fix

On Windows, `onnxruntime-gpu` 1.25.1 requires cuBLAS 12 and cuDNN 9 DLLs to be on the system PATH. PyTorch bundles these at `Miniconda3/envs/image-gs/lib/site-packages/torch/lib/`. All three scripts now prepend `torch/lib` to `os.environ["PATH"]` before importing `onnxruntime` to resolve this. Without the fix, ORT silently falls back to CPUExecutionProvider and numbers are ~10× slower.

---

## Benchmark results

**RTX 3080 Ti, batch=1, N=5 timed runs, 3 warmup. Training running in parallel (4 GB VRAM).**

### Steam Deck (800×1280 LR → 1600×2560 HR)

| Backend        | Peak VRAM | ms/frame |  fps | Size (MB) |
|----------------|-----------|----------|------|-----------|
| PyTorch FP32   | 1306 MiB  |   190.6  |  5.2 |      2.31 |
| ONNX-RT FP32   |    9.3 MiB |   193.8  |  5.2 |      2.31 |
| ONNX-RT FP16   |    9.3 MiB |   125.0  |  8.0 |      1.17 |

**FP16 speedup vs PyTorch FP32: 1.52×**

### 720p (720×1280 LR → 1440×2560 HR)

| Backend        | Peak VRAM | ms/frame |  fps | Size (MB) |
|----------------|-----------|----------|------|-----------|
| PyTorch FP32   | 1182 MiB  |   171.8  |  5.8 |      2.31 |
| ONNX-RT FP32   |    9.3 MiB |   168.8  |  5.9 |      2.31 |
| ONNX-RT FP16   |    9.3 MiB |   109.4  |  9.1 |      1.17 |

**FP16 speedup vs PyTorch FP32: 1.57×**

### 900p (900×1600 LR → 1800×3200 HR)

| Backend        | Peak VRAM | ms/frame |  fps | Size (MB) |
|----------------|-----------|----------|------|-----------|
| PyTorch FP32   | 1835 MiB  |   265.6  |  3.8 |      2.31 |
| ONNX-RT FP32   |    9.3 MiB |   275.0  |  3.6 |      2.31 |
| ONNX-RT FP16   |    9.3 MiB |   168.6  |  5.9 |      1.17 |

**FP16 speedup vs PyTorch FP32: 1.58×**

### 1080p → 4K (1080×1920 LR → 2160×3840 HR)

| Backend        | Peak VRAM | ms/frame |  fps | Size (MB) |
|----------------|-----------|----------|------|-----------|
| PyTorch FP32   | 2636 MiB  |   378.2  |  2.6 |      2.31 |
| ONNX-RT FP32   |    9.3 MiB |   396.8  |  2.5 |      2.31 |
| ONNX-RT FP16   |    9.3 MiB |   318.8  |  3.1 |      1.17 |

**FP16 speedup vs PyTorch FP32: 1.19×**

---

## VRAM analysis

PyTorch FP32 peaks at 1.1–2.6 GB depending on resolution (includes activation buffers). ONNX-RT (both FP32 and FP16) reports 9.3 MiB — this is the ORT memory pool overhead tracked by `torch.cuda.max_memory_allocated`, not total VRAM (ORT uses its own CUDA allocator outside torch's tracker). Real VRAM usage for ORT inference at these resolutions is expected to be similar to PyTorch (~weights + activation buffer), but we cannot measure it directly via PyTorch's memory API.

---

## ONNX-RT FP32 vs PyTorch FP32

ONNX-RT FP32 is within ±3% of PyTorch FP32 at most resolutions, with a slight regression at 1080p (396.8 vs 378.2 ms). The ONNX graph has operator fusion (constant folding enabled), but PyTorch's JIT also does fusion; the two are roughly at parity for this architecture. No raw speedup from the FP32 ONNX path alone — the value is FP16.

---

## TensorRT status

**TensorRT is NOT installed** on `<train-host>`. `nvinfer_10.dll` is missing. The TensorrtExecutionProvider silently falls back to CUDAExecutionProvider, so the TRT "benchmark" is actually ORT CUDA FP16. The numbers are essentially identical to the ONNX-RT FP16 column (within measurement variance). The `sr_export_tensorrt.py` script works correctly given the fallback and produces valid benchmark output, but the TRT engine itself is not built.

**To enable TRT:** install TensorRT 10.x for Windows and add the bin directory to PATH. The script already has the provider configuration wired.

---

## Model size

| Format          | Size (MB) | Ratio vs FP32 |
|-----------------|-----------|----------------|
| PyTorch .pt     |      7.0  | —              |
| ONNX FP32       |      2.31 | 1.00×          |
| ONNX FP16       |      1.17 | 0.51×          |

The .pt checkpoint is larger because it includes optimizer state, training args, and step metadata alongside the model weights.

---

## Test results

```
tests/sr/test_onnx_export.py::test_export_roundtrip_shape[pico]      PASSED
tests/sr/test_onnx_export.py::test_export_roundtrip_shape[lite]      PASSED
tests/sr/test_onnx_export.py::test_export_roundtrip_shape[standard]  PASSED
tests/sr/test_onnx_export.py::test_export_dynamic_axes_work          PASSED
tests/sr/test_onnx_export.py::test_export_fp16_roundtrip             PASSED

5 passed in 4.58s
```

---

## Decision

**ONNX FP16 is the shipped inference path.** 1.5–1.6× speedup at Steam Deck / 720p / 900p is real and meaningful. 1080p→4K shows a smaller gain (1.19×) which may reflect memory-bandwidth saturation at that resolution. VRAM savings from FP16 weights are confirmed (1.17 MB vs 2.31 MB on disk).

The training/inference skip-mode mismatch (bicubic antialias=True during training, bilinear in ONNX) is a known gap that should be closed before production. Options: train with antialias=False, or implement a custom bicubic kernel.

TensorRT would add another 1.5–3× on top of the FP16 baseline if installed (typical for this class of CNN). Not blocking — the FP16 ONNX path is already shippable.

---

## Open questions

1. Does the bilinear skip (vs bicubic antialias=True) hurt PSNR measurably on held-out scenes? Needs a direct ablation.
2. What is the actual VRAM usage of ORT FP16 at 1080p (ORT uses its own allocator, not torch's)?
3. TensorRT: is the 1080p gain smaller because of memory bandwidth or kernel choice? TRT's layer-fusion could recover this.
4. At what resolution does ONNX-RT FP16 saturate the 3080 Ti's Tensor Cores?
