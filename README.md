# OpenSuperSampling (OSS)

> Vendor-agnostic open-source real-time ray-tracing reconstruction — denoising, upscaling, and frame extrapolation.

**Status:** Pre-alpha / active research. Not yet suitable for production use.

---

## What is OSS?

OSS is a community-governed alternative to proprietary reconstruction stacks (DLSS RR, FSR Ray Regen, XeSS). It targets **+1.6 to +2 dB PSNR over DLSS 4 RR** on flagship hardware and is **the only ray-reconstruction-class option** for the ~60% of the GPU market that NVIDIA and AMD flagship stacks don't cover.

No closed weights. No SDK SLAs. No vendor lock-in.

---

## Modules

| Module | Class | Description | Params |
|--------|-------|-------------|--------|
| **OSS** | `OSS` | Core upscaler / supersampler. Three input modes: `rgb`, `rgb_aux`, `features`. | 1M – 5.5M |
| **OSS-RG** | `OSSRG` | Ray-tracing denoiser (regen). Kernel-prediction U-Net, two-branch input. Pairs with OSS via feature handoff. | ~141K |
| **OSS-Pico** | `OSSPico` | Combined denoising + SR for Steam Deck / low-end hardware. Temporal recurrent. | ~270K |
| **OSS-FX** | `OSSFx` | α-conditioned frame extrapolation. G-buffer-assisted via DLL hook. | ~480K |

---

## Hardware Tiers

### Upscaler (OSS)

| Tier | Params | Target Hardware |
|------|--------|----------------|
| Pico | ~270K | GTX 10/16, Steam Deck, integrated GPUs |
| Lite | ~1M | RTX 20+, RDNA2+ |
| Standard | ~2.55M | RTX 4080+, RX 9070 XT+, Apple M3 Max |
| Heavy | ~5.3M | RTX 4090, flagship only |

### Quality Modes

| Mode | Internal res | Notes |
|------|-------------|-------|
| Ultra Performance | 33% | |
| Performance | 50% | |
| Balanced | 59% | |
| Quality | 67% | Better-than-native threshold on Standard+ |
| Ultra Quality | 77% | |
| OSAA | 100% | OpenSuperAntiAliasing — TAA replacement mode |

### Frame Extrapolation (OSS-FX)

| Device | Resolution | Render → Display fps | Budget |
|--------|-----------|---------------------|--------|
| Steam Deck | 720p | 40→60, 60→90 | ≤16.7ms |
| M3 Max 40-core | 1080p | 60→120 | ≤8.3ms |
| RTX 4070 laptop | 1080p | 60→120 | ≤8.3ms |
| RTX 3080 Ti | 1440p | 60→120 | ≤8.3ms |

---

## Inference Backends

OSS auto-selects the most optimized available backend at runtime:

| Hardware | OS | Backend |
|----------|-----|---------|
| NVIDIA RTX (Ampere+) | Any | TensorRT (FP16, engine cached per GPU arch) |
| NVIDIA GTX | Any | ONNX Runtime CUDA EP |
| AMD RDNA3+ desktop | Linux | MIGraphX (ROCm) |
| AMD RDNA2+ | Windows | DirectML (ONNX Runtime) |
| Apple M-series | macOS | CoreML (`compute_units=ALL` → ANE + GPU) |
| Intel Arc / iGPU | Any | OpenVINO |
| Steam Deck / fallback | Any | NCNN + Vulkan |

---

## Architecture

### OSS-FX: G-Buffer-Assisted Extrapolation

A `dxgi.dll` proxy (Windows) or Vulkan layer (Linux) intercepts `Present()` and extracts `color(t)`, `depth(t)`, and `motion_vec(t-1→t)` — buffers the game already computed for its TAA/DLSS/FSR pass, at zero extra render cost.

```
DLL hook intercepts Present():
  color(t)          → History Tracker  → motion history
  depth(t)          → Geometry-aware warp → warped_est
  motion_vec(t-1→t) → Flow extrapolation → F_{t→t+α}
  warped_est + history + depth + α_embed
                    → SCN (neural net) → Frame_{t+α}
```

Three-tier fallback:
1. **Motion vectors + depth** — zero extra cost, best quality
2. **RAFT-Small estimated flow + depth** — ~+3ms
3. **Color only** — GFFE-equivalent, maximum compatibility

α-conditioning enables arbitrary target fps (60→90, 60→120, etc.) from a single model.

### OSS + OSS-RG Paired Mode

OSS-RG exports a 32-channel FP16 feature map from its penultimate layer. OSS consumes this directly via `input_mode="features"`, bypassing redundant color encoding for joint quality+performance gains.

---

## Installation

```bash
pip install oss                     # CPU / basic
pip install oss[cuda]               # NVIDIA (TensorRT + ONNX CUDA)
pip install oss[directml]           # AMD/Intel on Windows
pip install oss[rocm]               # AMD on Linux
pip install oss[coreml]             # Apple Silicon
pip install oss[openvino]           # Intel Arc / iGPU
pip install oss[vulkan]             # Steam Deck / Vulkan fallback
```

---

## Quick Start

```python
from oss.model import OSS, OSSRG

# Standalone upscaler
model = OSS(input_mode="rgb", scale_factor=2.0, tier="standard")
color_hr = model(color=color_lr, depth=depth_lr, motion=motion_lr)

# Paired denoiser + upscaler (feature handoff)
denoiser = OSSRG()
upscaler = OSS(input_mode="features", scale_factor=2.0, tier="standard")
features, color_denoised = denoiser(noisy, aux, history)
color_hr = upscaler(features=features, depth=depth_lr, motion=motion_lr)

# Runtime inference (auto-selects best backend)
from oss.infer import InferenceSession
session = InferenceSession("oss_standard.onnx")
output = session.run({"color": color_np, "depth": depth_np, "motion": motion_np})
```

---

## Training Data

- **NoiseBase** — primary for OSS-RG and OSS-FX Tier 1+2 (ground-truth depth + motion vectors)
- **Sintel** — OSS-FX fallback path training
- **Vimeo-90K** — OSS-FX real-world motion diversity

---

## License

- SDK and shaders: **Apache-2.0**
- Plugins: **MIT**
- Model weights: **CC-BY-4.0**

---

## What We Won't Use

- NRD (RTX SDK SLA)
- DLSS/FSR/XeSS decompiled binaries or leaked weights
- `tiny-cuda-nn` (CUDA-only, defeats vendor-agnosticism)
- Quixel / Megascans / CC-BY-NC-* training assets
