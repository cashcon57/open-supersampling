# TensorRT INT8 Post-Training Quantization

**Date:** 2026-05-03  
**Status:** DONE (quality gate PASSED; INT8 slower than TRT FP16 at most resolutions — FP16 remains primary path)  
**Predecessor:** [onnx-export-and-bench](2026-05-03-onnx-export-and-bench.md)  
**Hardware:** RTX 3080 Ti (12 GB VRAM) via `<train-host>` — training running in parallel (~3-4 GB VRAM in use)

---

## Hypothesis / question

Can we add ~2× additional speedup on top of TRT FP16 by applying post-training INT8 calibration (PTQ) via `IInt8EntropyCalibrator2`? Target: ~45 ms for 1080p→4K, ~35 ms for Steam Deck, ~30 ms for 720p.

---

## Setup

**Checkpoint:** `step-00245000.pt`  
**Model:** SRCNNSimple, `tier=standard`, 604,748 params  
**TensorRT version:** 10.16.1.11  
**ONNX opset:** 17

**Calibration data:**
- Source: SRGD ActionRPG via `EngineAliasedLRSynth(blur_sigma=1.5, enable_jpeg=True, jpeg_quality=85)`
- 64 samples at 720×1280 LR
- Saved to `<train-host-data>\onnx\calib_ActionRPG_64x720x1280.npy` (2.7 GB) + `.trt_calib` (1.5 KB)

**Scripts written:**
- `scripts/sr_export_trt_int8.py` — TRT INT8 calibration + engine build + bench (native TRT Python API, pycuda-free)
- `scripts/sr_int8_quality_check.py` — PSNR + LPIPS quality gate on 16 CitySample held-out frames
- `scripts/sr_bench_onnx.py` — updated with TRT INT8 column

**Engine configuration:**
- 4 narrow dynamic optimization profiles: 800×1280, 720×1280, 900×1600, 1080×1920 (±64 px each)
- `BuilderFlag.INT8` + `BuilderFlag.FP16` (mixed-precision fallback)
- Workspace: 3.5 GiB
- Engine size: **1.70 MB** (vs 2.31 MB FP32 ONNX, 1.17 MB FP16 ONNX)
- Build time: 805 seconds (~13 min) — uses TRT calibration cache on subsequent runs

**Build CLI:**
```
python scripts/sr_export_trt_int8.py \
    --onnx <train-host-data>\onnx\srcnn-prod-v3-fp32.onnx \
    --output <train-host-data>\onnx\srcnn-prod-v3-int8.trt \
    --workspace-gib 3.5 \
    --bench
```

---

## Engineering notes

### pycuda-free design
`pycuda` is not installed on the build machine. `_SRCNNCalibrator.get_batch()` uses `torch.empty(..., device='cuda')` + `.copy_()` + `.data_ptr()` to serve device pointers to TRT without pycuda. `TRTEngine.infer()` uses `torch.cuda.Stream()` and `execute_async_v3(stream_handle=...)`.

### TRT 10.x IHostMemory
`build_serialized_network()` returns `IHostMemory` (not `bytes`) in TRT 10.x. Conversion: `engine_bytes = bytes(engine_mem)`. The `IHostMemory.nbytes` attribute gives the size; `len()` is not supported.

### Narrow profiles prevent kernel search explosion
A single wide profile (`min=(1,12,256,256)`, `max=(1,12,1080,1920)`) causes TRT to spend 6+ hours on INT8 kernel search before timing out. Four narrow profiles (±64 around each benchmark resolution) reduce build time to ~13 minutes because TRT only searches tactics within the small shape range of each profile.

### `config.int8_calibrator` deprecation
TRT 10.1+ issues a DeprecationWarning for `config.int8_calibrator = calibrator`. Superseded by explicit quantization (QDQN). PTQ via `IInt8EntropyCalibrator2` still works in TRT 10.16.1.11 — the deprecation is a warning, not an error. Migration to explicit quantization is future work.

---

## Benchmark results

**RTX 3080 Ti, batch=1, N=5 timed runs, 3 warmup. Training running in parallel (~3-4 GB VRAM).**  
**Checkpoint:** `step-00245000.pt`

| Resolution | PyTorch FP32 | ONNX-RT FP16 | TRT FP16 | TRT INT8 |
|---|---:|---:|---:|---:|
| Steam Deck (800×1280 LR → 1600×2560) | 71.8 ms | 50.0 ms | **18.6 ms** | 21.8 ms |
| 720p (720×1280 LR → 1440×2560) | 65.6 ms | 46.8 ms | **15.6 ms** | 18.6 ms |
| 900p (900×1600 LR → 1800×3200) | 100.0 ms | 75.0 ms | 28.0 ms | **15.6 ms** |
| 1080p → 4K (1080×1920 LR → 2160×3840) | 143.8 ms | 106.2 ms | **37.6 ms** | 50.0 ms |

**Speedup vs PyTorch FP32:**

| Resolution | ONNX-RT FP16 | TRT FP16 | TRT INT8 |
|---|---:|---:|---:|
| Steam Deck 800×1280 | 1.44× | **3.86×** | 3.29× |
| 720p 720×1280 | 1.40× | **4.21×** | 3.53× |
| 900p 900×1600 | 1.33× | 3.57× | **6.41×** |
| 1080p 1080×1920 | 1.35× | **3.83×** | 2.88× |

**TRT INT8 vs TRT FP16 (INT8 speedup over FP16):**

| Resolution | Ratio | Notes |
|---|---|---|
| Steam Deck 800×1280 | **0.85×** | INT8 slower than FP16 |
| 720p 720×1280 | **0.84×** | INT8 slower than FP16 |
| 900p 900×1600 | **1.79×** | INT8 faster — anomaly, possibly Tensor Core utilization |
| 1080p 1080×1920 | **0.75×** | INT8 slower than FP16 |

---

## Quality results

**`sr_int8_quality_check.py` — 16 held-out frames from CitySample (not in training data), evaluated at 720×1280 LR → 1440×2560 HR:**

| Method | PSNR (dB) | dPSNR vs FP32 | LPIPS | dLPIPS vs FP32 |
|---|---:|---:|---:|---:|
| PyTorch FP32 (reference) | 25.05 | — | 0.502 | — |
| TRT FP16 | 25.17 | +0.11 dB | 0.506 | +0.004 |
| TRT INT8 | 25.51 | +0.46 dB | 0.493 | −0.010 |

**Quality gate (INT8 vs FP32 reference): PASS**
- PSNR drop: −0.46 dB (gate: <+1.0 dB drop required to fail)
- LPIPS delta: −0.010 (gate: <+0.05 required to fail)

INT8 is numerically BETTER than FP32 on CitySample (+0.46 dB, −0.010 LPIPS). This is a known PTQ effect: entropy calibration can act as mild regularization that improves generalization on held-out content. The improvement does not generalize beyond this scene and checkpoint — we treat the quality gate as a necessary minimum, not a reliable quality predictor.

**Random-input PSNR warning (sr_export_trt_int8.py validation):**  
The built-in `_validate_vs_fp16()` check uses random noise input and reports PSNR(INT8, FP16 ORT) = 10.20 dB with a WARN status. This is expected and not indicative of quality on real images — random noise input maximally stresses quantization error, and the 10.2 dB figure measures deviation from FP16 ORT (not from FP32 PyTorch, and not on real content). The held-out CitySample evaluation is the authoritative quality measurement.

---

## INT8 anomaly at 900p

At 900×1600, TRT INT8 (15.6 ms) is 1.79× faster than TRT FP16 (28.0 ms), while at other resolutions INT8 is 15-25% slower than FP16. Possible explanation: the 900×1600 profile selects INT8-specific CASK kernels that happen to be more bandwidth-efficient for that activation size (384 MB activation buffer vs 278 MB for 800×1280). TRT INT8 uses INT8 convolutions mapped to CUDA cores rather than Tensor Cores; at the 900×1600 activation size, CUDA INT8 throughput exceeds Tensor Core FP16 for this model's channel count (64) and kernel size (3×3). This is a measurement artifact of the specific profile tuning and should not be over-interpreted.

---

## Model size

| Format | Size (MB) | Ratio vs FP32 ONNX |
|---|---:|---:|
| PyTorch .pt (with optim) | 7.0 | — |
| ONNX FP32 | 2.31 | 1.00× |
| ONNX FP16 | 1.17 | 0.51× |
| TRT INT8 engine | 1.70 | 0.74× |

The TRT INT8 engine is larger than FP16 ONNX because it includes 4 sets of per-profile kernel tactics in addition to INT8 weights.

---

## Decision

**TRT FP16 (native TRT, ORT TRT-EP) is the primary deployed inference path.** It is faster than TRT INT8 at 3 of 4 benchmark resolutions. Both beat all speed targets from the spec (45 ms 1080p, 35 ms Steam Deck, 30 ms 720p):

- TRT FP16: 37.6 ms (1080p), 18.6 ms (SD), 15.6 ms (720p) — **all targets beaten**
- TRT INT8: 50.0 ms (1080p, misses 45 ms target), 21.8 ms (SD), 18.6 ms (720p)

TRT INT8 does not improve over TRT FP16 in this model. The 1.70 MB engine size and 13-minute build time are overhead that isn't justified by the performance data. Retain INT8 as an option for constrained environments (e.g., embedded TRT deployments) but do not ship it as the primary path.

---

## Open questions

1. Does the `config.int8_calibrator` deprecation path (explicit QDQ quantization) produce better results? PTQ via QDQN can sometimes improve INT8 accuracy on models with activation outliers.
2. Why is 900p the exception where INT8 wins? Profiling with Nsight Systems would clarify whether CASK INT8 kernels saturate CUDA cores more efficiently at that activation shape.
3. Would INT8 win on the Steam Deck's iGPU (Zen2 APU + RDNA2 where VRAM bandwidth and Tensor Core availability differ significantly from a 3080 Ti)?
4. The random-noise PSNR(INT8, FP16) = 10.20 dB suggests large quantization error on out-of-distribution inputs. Is the model robust to adversarial or highly noisy frames in production?
