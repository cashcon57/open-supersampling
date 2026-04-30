# Vulkan Inference Runtime Evaluation for ORU-Pico

**Date:** 2026-04-30
**Author:** v0.2-alpha scaffold (Task 7)
**Branch:** `v0.2-dev`
**Decision:** **NCNN with Vulkan backend** (with PNNX as the PyTorch -> NCNN exporter), via the official `ncnn` Python wheel.

---

## Goal

Pick a runtime that lets us run the ORU-Pico graph (~228K params, recurrent
kernel-prediction U-Net) natively in Vulkan compute on:

- Steam Deck (RDNA 2, 8 CU iGPU at 15 W) -- the v0.2 ship target.
- Generic Linux desktop (any Vulkan 1.3 driver).
- macOS via MoltenVK (developer iteration).
- Windows via DXVK or native Vulkan (for parity testing).

Acceptance for this task is the *scaffold*: load the model, run a forward pass,
match the PyTorch reference within 1e-2 abs diff. Real Steam Deck deployment
and the gamescope plugin are deferred to v0.2-beta.

---

## Candidates

| Runtime | License | ONNX import | Vulkan backend | RDNA 2 / mobile tuning | Recurrent ops | Active? | Verdict |
|---|---|---|---|---|---|---|---|
| **NCNN** (Tencent) | BSD-3-Clause | yes (legacy `onnx2ncnn`) plus a much better PyTorch path via **PNNX** | first-class, hand-tuned | yes -- ARM Mali / Adreno are explicit targets, plus desktop AMD via Vulkan | yes -- supports the elementwise + conv ops we need; recurrent state is just a tensor input/output, not a special op | yes, frequent commits | **PICK** |
| **MNN** (Alibaba) | Apache-2.0 | yes (`MNNConvert`) | yes, mature | yes, mobile-tuned | yes | yes | strong second; bigger native footprint, conversion is a separate CLI binary, less ergonomic Python story than NCNN |
| **ONNX Runtime + Vulkan EP** | MIT | native | experimental | unclear; not the EP's focus | yes | EP is still flagged experimental as of late 2025 / early 2026 | risky for v0.2-alpha; revisit when the EP exits experimental |
| **wgpu-rs + tract** | Apache-2.0 / MIT | tract supports many ONNX ops but not all | wgpu compiles to Vulkan/Metal/DX12 | not the project's focus | partial -- recurrent state across frames is fine (just keep tensors), but op coverage gaps mean we'd patch tract upstream | yes | viable but adds a Rust toolchain to our build; better fit when we have a `cargo`-friendly environment |
| **Apache TVM (Vulkan target)** | Apache-2.0 | yes via Relax / Relay | yes | tunable per-target via AutoTVM | yes | yes | heaviest; tuning takes hours; overkill for a 228K-param graph that NCNN runs natively |
| **MIOpen / ROCm** | MIT | only via composable kernels | n/a (HIP, not Vulkan) | AMD-only | yes | yes | doesn't satisfy the cross-vendor requirement; Steam Deck APUs run a Mesa Vulkan stack, not ROCm |
| **Custom HLSL -> DXC -> SPIR-V** | n/a (we own it) | n/a | full control | full control | full control | n/a | required eventually if we want to beat NCNN's autotuning; not v0.2-alpha scope. We keep `ors/inference/vulkan/shaders/` reserved for it. |

---

## Why NCNN

1. **License.** BSD-3-Clause -- compatible with ORS's Apache-2.0; no copyleft surprises.
2. **Vulkan is the default backend on supported devices.** `ncnn::Net::opt::use_vulkan_compute = true` is a one-liner. Mesa RADV (Steam Deck), AMDVLK, NVIDIA, Intel, MoltenVK, Lavapipe -- all enumerate the same physical-device path in NCNN.
3. **PNNX is the right exporter for PyTorch graphs.** Legacy `onnx2ncnn` chokes on dynamic shapes and recent opset features; PNNX traces the live PyTorch module and emits NCNN's `.param` + `.bin` directly. We verified end-to-end conversion of `ORUPico()` (in inference mode, FP32, default optlevel=2) on the dev box -- ncnn artifacts produced and loadable.
4. **Mobile-tuned by construction.** NCNN was built for ARM phones first; the same fixed-function-shader codepath runs on RDNA 2 mobile iGPUs. This matches Pico's RDNA-2-first design.
5. **Recurrent state is trivial.** Pico's hidden state is just a tensor in / tensor out. NCNN's `Extractor` API takes named input/output blobs; we feed the prior frame's `out1` back as the next frame's `in5`. No special RNN op required.
6. **Python wheel ships with Vulkan.** `pip install ncnn` gives us a working binding on macOS/Linux today; the same wheel on a real Vulkan device (e.g. Linux desktop, Steam Deck) flips the backend on without code changes. If the loader can't find a Vulkan ICD (typical macOS-without-MoltenVK case), NCNN falls back to its CPU path -- which is itself NEON / AVX2 optimised and good enough for our parity test.

---

## Why not the others (short version)

- **MNN.** Strong contender, but: PyTorch -> MNN goes through ONNX or Torchscript, the Python API is thinner, and the Vulkan backend is less mature than its OpenCL/Metal paths on the docs as of 2026. Worth revisiting if we hit an NCNN op gap.
- **ORT + Vulkan EP.** Still experimental, and the `onnxruntime` build with the Vulkan EP is not on the default PyPI wheel -- we'd have to ship our own builds for Linux *and* macOS. Not v0.2-alpha scope.
- **wgpu + tract.** Forces a Rust toolchain into the dev path. We may want this later for the gamescope plugin's host side; not for v0.2-alpha.
- **TVM.** Tuning runs are long; tuned binaries don't generalise across drivers; the maintenance footprint is real. Overkill for a 228K-param model.
- **MIOpen/ROCm.** AMD-only. Steam Deck Mesa stack is Vulkan, not ROCm.
- **Custom SPIR-V.** Reserved -- the `ors/inference/vulkan/shaders/` dir stays in the tree for v0.2-beta. A from-scratch kernel set is thousands of LOC and we don't need it to *prove* the pipeline.

---

## Conversion path

```
PyTorch ORUPico (inference mode) -- dummy forward inputs --> PNNX trace
                                                       |
                                                       v
                                  pico.ncnn.param + pico.ncnn.bin
                                                       |
                                                       v
                              ncnn.Net().load_param/load_model
                                                       |
                                                       v
                              ex.input("in0".."in5") -> Vulkan compute -> ex.extract("out0","out1")
```

Input blob names are emitted as `in0..in5` in the order PNNX traced the
forward signature: `(color_lr, depth_lr, motion_lr, normals_lr, history_hr,
hidden_state)`. Outputs are `out0` (`rgb_hr`) and `out1` (`new_hidden_state`).
The runtime module pins this contract.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| PNNX conversion regresses on a Pico arch tweak | Re-run conversion as part of a CI step; pin PNNX wheel version in `pyproject.toml`. Fallback path: `torch.onnx.export` -> `onnx2ncnn` (kept in NCNN repo). |
| Vulkan loader missing on dev macOS | NCNN CPU fallback runs the same graph; parity test treats Vulkan as a *preference*, not a requirement. The test asserts the runtime works; whether Vulkan or CPU is selected is logged but not gated. |
| Op coverage gap in NCNN for a future Pico revision | NCNN supports custom layer registration; we can drop in a SPIR-V kernel for any one op without rewriting the whole runtime. |
| `pip install ncnn` wheel not built for some target | Source build is documented and small (CMake + Vulkan SDK). |

---

## Out of scope for v0.2-alpha

- gamescope plugin C++ host code (deferred to v0.2-beta).
- Hand-tuned SPIR-V kernels (deferred; reserved dir at `ors/inference/vulkan/shaders/`).
- FP16 weight conversion + accuracy regression tests (deferred; FP32 is fine for the parity scaffold).
- Real RDNA 2 hardware validation (blocked on Steam Deck access).

---

## Decision

**NCNN with PNNX as the PyTorch -> NCNN exporter.** Implementation lands in
`ors/inference/vulkan/runtime.py`, parity test at `tests/test_vulkan_parity.py`,
deps added to `pyproject.toml` as an optional extra (`vulkan`).
