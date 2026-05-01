# OSS-Gaussian Vulkan + ncnn Port (Steam Deck)

Sprint 7 / Track V scaffold. Ports the Sprint 1 CUDA tile rasterizer to a
GLSL compute shader (SPIR-V) and exports the Sprint 4 `GaussianParamNetwork`
to ncnn for Vulkan-backed inference on RDNA 2.

## Build prerequisites (Linux dev box; production target is Steam Deck)

1. **Vulkan SDK + glslang:**
   - Arch / CachyOS / SteamOS dev: `sudo pacman -S vulkan-devel shaderc glslang`
   - Ubuntu / Debian: `sudo apt install libvulkan-dev glslang-tools`
2. **CMake ≥ 3.20**, **gcc ≥ 11** (or clang ≥ 14).
3. **Python extras (for ncnn export only):** `pip install -e '.[vulkan]'` from repo root. This pulls `ncnn` + `pnnx` pre-built wheels.

We deliberately do **not** vendor the Vulkan SDK or ncnn source. CMake's
`find_package(Vulkan REQUIRED)` resolves against the system SDK; ncnn comes
from a pip wheel at runtime.

## Build

```sh
cd oss/gaussian/ports/vulkan_ncnn
cmake -B build
cmake --build build
```

Outputs:
- `build/rasterizer.spv` — compiled compute shader
- `build/liboss_gaussian_vk.a` — host harness static library

## Files

| File | Purpose |
| --- | --- |
| `rasterizer.comp` | GLSL compute kernel skeleton. Body is a TODO; ported in T7.V.2. |
| `rasterizer.cpp` | C++ Vulkan host harness — instance, device, pipeline, dispatch. |
| `CMakeLists.txt` | Compiles `rasterizer.comp` -> `.spv` and builds the host library. |
| `export_ncnn.py` | Converts a Sprint 4 checkpoint to ncnn `.param` + `.bin` via PNNX. |
| `__init__.py` | Python package init + `has_vulkan_toolchain()` / `has_ncnn()` host checks. |

## Run an ncnn export (dry-run, no `pnnx` required)

```sh
python -m oss.gaussian.ports.vulkan_ncnn.export_ncnn --check
```

Real export (writes `checkpoints/param_net_pico.ncnn.{param,bin}`) requires
`pnnx` on PATH or the `[vulkan]` extra installed:

```sh
python -m oss.gaussian.ports.vulkan_ncnn.export_ncnn --tier pico --ckpt path/to/sprint4.ckpt
```

## Steam Deck deployment notes

- Build on a Linux dev box (CachyOS or any Vulkan-SDK-equipped distro). Copy the
  `.spv` + `.param` + `.bin` to the Deck via Tailscale (`tailnet-ssh.md`).
- ncnn at runtime auto-detects the Deck's RDNA 2 Vulkan device. No build
  step required on the Deck itself.
- Configure ncnn's `Net` with `opt.use_vulkan_compute = true` and
  `opt.use_fp16_storage = true` — RDNA 2 has packed-FP16 math.

## Sprint 7 status

Scaffold only. Empty kernel; `cmake --build build` is expected to compile it
cleanly. Real kernel port + integration land in the Sprint 7 implementation
phase per `docs/superpowers/plans/2026-05-01-gaussian-sprint-7-plan.md`.
