# OSS-Gaussian — Sprint 7 Detailed Plan: Cross-Platform Ports

**Spec:** `docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md` § 4.2
**Master plan reference:** `docs/superpowers/plans/2026-05-01-gaussian-master-plan.md` § Sprint 7
**Predecessor sprints:** Sprints 1–6 (Windows / RTX 3080 Ti reference implementation)
**Branch:** `v0.2-dev`
**Total estimate:** ~2 weeks (12 tasks across two parallel tracks + cross-tier report)

---

## 0. Sprint 7 Outcome Definition

By end of Sprint 7 the following must be true:

1. The Sprint 1–6 reference renderer + Sprint 4 trained `GaussianParamNetwork` weights run on **two new platforms** without retraining:
   - **M3 Max MacBook Pro:** Metal compute renderer (MSL kernel) + CoreML-exported network.
   - **Steam Deck (RDNA 2 / SteamOS):** Vulkan compute renderer (GLSL → SPIR-V) + ncnn-exported network.
2. Each platform runs a Sintel benchmark (the dataset already used as the cross-tier validation harness in Sprint 4 § T4.7) at its tier's Gaussian budget:
   - M3 Max: 5K Gaussians @ 1440p (Lite tier).
   - Steam Deck: **1K Gaussians @ 1280×800** (Pico tier).
3. The Gaussian-count knob is the **only** scaling lever. Same trained checkpoint on both ports, same on RTX 3080 Ti reference, same on RTX 4090 (informational).
4. CrossOver smoke test on M3 Max: launch a CrossOver-compatible target (Skyrim SE if installed, otherwise Sintel offline render), confirm the Metal renderer + CoreML net run without crashing the Wine layer.
5. Cross-tier report committed at `docs/superpowers/gaussian-cross-tier-bench.md` covering RTX 3080 Ti / RTX 4090 (informational, if access available) / M3 Max / Steam Deck — frame time, PSNR, SSIM, power draw where measurable.
6. Sprint 7 code review pipeline run; judge verdict APPROVE.

Anything beyond these — full Cyberpunk integration on M3 Max, Steam Deck DLL-swap target, console partnerships — is **explicitly out of scope** and tracked as post-v1.

---

## 1. Architecture Recap (5-line version)

- The renderer is a tile-based top-K 2D Gaussian rasterizer. Sprint 1's CUDA kernel (vendored Image-GS) is the reference; Sprint 7 ports the same algorithm to **MSL** (Apple) and **GLSL/SPIR-V** (Vulkan).
- The network is `oss.gaussian.network.GaussianParamNetwork` (Sprint 4). Sprint 7 exports the trained checkpoint to **CoreML** (M3 Max via `coremltools`) and **ncnn** (Steam Deck via `pnnx`). Both runtimes are already declared in `pyproject.toml` extras (`coreml`, `vulkan`).
- Tile size is fixed at 16 across all backends — must match `renderer.TILE_SIZE` and `network.DEFAULT_TILE_SIZE`. Top-K and bank size are runtime parameters.
- Gaussian count is the per-tier budget knob: 1K (Pico/Deck) → 5K (Lite/M3 Max) → 8K (Standard/3080 Ti) → 15K (Ultra/4090). One trained model.
- Sprint 7 ports are **scaffolds + kernels only**; the persistent canvas + warp + frame extrapolation logic from Sprints 5–6 stays Python-driven through the renderer for now. Native canvas is a v1.1 task.

---

## 2. Reusable Insights from `2026-04-30-v0.2-deck-first-design.md`

Sprint 7 adopts a few decisions already validated for the (legacy / parallel) pixel-based Steam Deck track:

| Insight | Source § | Sprint 7 application |
| --- | --- | --- |
| Vulkan compute is the right backend for Deck, not OpenGL ES or DX12. | § 5.1 | Track V uses Vulkan compute for the rasterizer + ncnn (which already targets Vulkan as its first-class GPU backend). |
| FP16 packed math on RDNA 2 — no FP8/INT4. | § 5.2 | ncnn export defaults to FP16; we do **not** quantize below FP16 on Deck. |
| gamescope is the "free system-wide upscaler hook" but its plugin API is limited. | § 4.1 + § 12 Q1/Q2 | We do **not** wire into gamescope this sprint — Sintel offline benchmark only. Gamescope-plugin shim is post-v1. |
| Channel widths must be multiples of 8 for cooperative-matrix tiling. | § 3.3 | Already enforced in `param_net.py` (`channels must all be multiples of 8`). PNNX preserves this; verify post-export. |
| ncnn ships pre-built wheels for Linux x86_64 and macOS arm64. | § 5.1 + `pyproject.toml` `[vulkan]` | Both Track V exporter (Mac dev box) and Deck inference can pip-install without building from source. |
| OptiScaler / DLL-swap path on Deck is post-v1. | § 9 | Sprint 7 ships an offline benchmark; no DLL hooking on Deck. |

---

## 3. Tasks

Tasks split across two parallel tracks. **T7.M.\*** = Metal track, **T7.V.\*** = Vulkan/ncnn track. Final cross-tier report (T7.X.1) waits on both. Code review (T7.X.2) closes the sprint.

The two tracks have **no shared code**, only a shared trained checkpoint and a shared Sintel test set. They can run in parallel on separate machines.

---

### Track M — Metal / M3 Max (6 tasks)

#### T7.M.1 — Metal toolchain bring-up + scaffold compile

**Goal:** Confirm the M3 Max box has the Xcode + Metal compiler toolchain. The scaffold `rasterizer.metal` empty kernel compiles to `default.metallib` via `xcrun metal`. No actual rasterization logic yet.

**Steps:**
1. On M3 Max, install Xcode Command Line Tools if not present: `xcode-select --install`.
2. Confirm: `xcrun -sdk macosx metal --version` reports a Metal compiler ≥ 32.x.
3. From `oss/gaussian/ports/metal/` run `make` (Makefile in scaffold). This invokes `xcrun -sdk macosx metal -c rasterizer.metal -o rasterizer.air` then `metallib`.
4. Output: `rasterizer.metallib` exists. Empty kernel — entry point `gaussian_rasterize_tile` is declared but body is `// TODO: port from CUDA in T7.M.2`.

**Verify:** `file rasterizer.metallib` reports a Mach-O Metal library. CI (macOS runner if added) can repeat this build.

**Acceptance:** Scaffold compiles cleanly with `-Werror`. Empty kernel function symbol present in the metallib.

#### T7.M.2 — Port CUDA tile rasterizer to MSL

**Goal:** Translate the vendored Image-GS CUDA tile rasterizer (`oss/gaussian/renderer/vendor/image_gs/.../rasterizer.cu`) into MSL (`rasterizer.metal`). Tile size 16. Top-K accumulation in threadgroup memory.

**Steps:**
1. Read CUDA source line-by-line. Identify: per-tile shared-mem buffer for K Gaussians, atomic-free top-K via simdgroup ballot, weighted feature accumulation.
2. Map CUDA → MSL primitives:
   - `threadgroup` memory ↔ CUDA `__shared__`.
   - `simd_ballot` / `simd_shuffle` (Metal 2.4+) ↔ CUDA warp intrinsics.
   - `simdgroup_size = 32` on Apple GPUs (verify on M3 Max — it is **32 on M-series**, not 64).
3. Tile dispatch: one threadgroup per 16×16 output tile, 256 threads/tg. Each thread loads one pixel; the simdgroup cooperatively iterates the per-tile Gaussian list.
4. Write the kernel body, mirroring the CUDA structure as faithfully as possible. Comment the CUDA→MSL translation choices inline.
5. Sanity host harness: `rasterizer.swift` builds a 4-Gaussian fixture, dispatches the kernel, reads back the output buffer, checks one pixel value against the Python reference rasterizer (`Rasterizer(force_backend="reference")`).

**Verify:** Single-tile fixture renders a recognizable Gaussian blob. Mean abs diff vs Python reference < 1e-3 (FP32 path) or 1e-2 (FP16 path).

#### T7.M.3 — CoreML network export

**Goal:** Export the trained Sprint 4 `GaussianParamNetwork` (Lite tier — 5K target) to a `.mlpackage` runnable by CoreML on M3 Max GPU + ANE.

**Steps:**
1. From `oss/gaussian/ports/metal/export_coreml.py`: load the Sprint 4 checkpoint (`oss/gaussian/network/checkpoints/lite-best.ckpt` — produced in T4.5).
2. Use `coremltools.convert(...)` with `compute_units=ALL` (CPU + GPU + ANE), input shape `(1, 12, H_lr, W_lr)` where `H_lr × W_lr` is the Sintel benchmark resolution divided by 2× upscale ratio.
3. Force `convert_to="mlprogram"` (CoreML's modern format; the legacy NeuralNetwork format is deprecated for new exports).
4. Verify the exported model accepts a dummy input and produces a tensor of shape `(1, K * (4 + bank_size + 3), H_tile, W_tile)` matching the `param_net.output_shape(...)` contract.
5. Save to `oss/gaussian/ports/metal/checkpoints/param_net_lite.mlpackage`. Excluded from git via `.gitignore` (binary artifact); the export script is the source of truth.

**Verify:** `python -m oss.gaussian.ports.metal.export_coreml --tier lite --check` runs the conversion and a single-batch parity check against PyTorch CPU output. Per-output mean abs diff < 1e-2 (CoreML uses FP16 by default on GPU).

#### T7.M.4 — M3 Max integration: end-to-end pipeline on Sintel

**Goal:** Stitch the MSL renderer (T7.M.2) + CoreML net (T7.M.3) into an offline pipeline that consumes a Sintel input frame, predicts Gaussians, rasterizes, writes output PNG.

**Steps:**
1. Driver script: `oss/gaussian/ports/metal/run_sintel.py` (Python; the Swift host harness is for kernel testing only — production driver stays Python and calls into Swift via PyObjC or a small C shim).
2. Pipeline: load Sintel LR frame + G-buffers → CoreML inference (returns raw param tensor) → `OutputHead` decode (reuse `oss.gaussian.network.OutputHead` from CPU) → MSL rasterizer dispatch → readback → save PNG.
3. Run on 50 Sintel frames; record per-frame timing (CPU prep, CoreML inference, MSL render, readback).

**Verify:** Output PNGs exist for all 50 frames. PSNR vs Sintel HR ground truth ≥ matched frames from RTX 3080 Ti reference within ±0.3 dB (same model weights → must produce comparable quality).

#### T7.M.5 — CrossOver smoke test

**Goal:** Confirm the Metal renderer + CoreML net survive being invoked from a CrossOver / Wine bottle context. **This is a smoke test, not a real game integration** — Cyberpunk on Apple Silicon via CrossOver is post-v1.

**Steps:**
1. Use Corkscrew to launch Skyrim SE in a Wine bottle (or any CrossOver-compatible game already on the box). Confirm baseline launch works.
2. Outside the Wine layer (i.e. as a native macOS process), run `run_sintel.py` from T7.M.4 in parallel with the game.
3. Confirm: native Metal context does not collide with the Wine D3D-via-Metal translation layer. The intent is to verify Apple's GPU scheduler tolerates two Metal clients (DXMT and our renderer) — not to perform real frame interception under CrossOver yet.
4. Document any observed contention (frame time spikes, command buffer stalls).

**Verify:** Both processes complete without crash. Sintel pipeline frame time variance < 2× idle baseline.

#### T7.M.6 — M3 Max benchmark

**Goal:** Per-config frame time on M3 Max for the same configurations Sprint 1 § T1.6 measured on RTX 3080 Ti.

**Configs:**
- Gaussian count: 1K, 5K (Lite tier target), 8K (informational).
- Output resolution: 1280×800, 1440p, 4K (informational).
- Backend: MSL renderer (this sprint) vs Python reference (sanity floor) vs Sprint 1 PyTorch fallback (informational).

**Steps:**
1. `bench.py` mirrors the Sprint 1 schema: 100 runs after 10 warm-up, mean / p50 / p95 / p99.
2. Output `bench_results_m3max.csv` to `oss/gaussian/renderer/bench/` alongside the 3080 Ti CSV.
3. Bandwidth-scaling reality check: M3 Max memory bandwidth is ~400 GB/s vs 3080 Ti's ~912 GB/s; a memory-bandwidth-bound renderer should land ~2.3× slower at the same Gaussian count. If our number is wildly off (≥4×), the kernel is compute-bound or has a launch-overhead bug.

**Verify:** CSV exists, numbers match the bandwidth-scaling sanity check ±50%.

---

### Track V — Vulkan / ncnn / Steam Deck (5 tasks)

#### T7.V.1 — Vulkan SDK bring-up + scaffold compile

**Goal:** On the Sprint 7 dev box (Linux desktop, **not** Deck — Deck is integration target only), Vulkan SDK installs cleanly. The scaffold `rasterizer.comp` empty compute shader compiles to SPIR-V.

**Steps:**
1. Install Vulkan SDK on the dev box. CachyOS / Arch: `sudo pacman -S vulkan-devel shaderc`. Steam Deck: SteamOS already ships the Vulkan loader; we use a Linux dev box for shader compilation.
2. From `oss/gaussian/ports/vulkan_ncnn/` run `cmake -B build && cmake --build build`. CMake invokes `glslangValidator -V rasterizer.comp -o rasterizer.spv`.
3. Output: `rasterizer.spv` exists. Empty kernel — entry point `main` is declared but body is `// TODO: port from CUDA in T7.V.2`.

**Verify:** `spirv-dis rasterizer.spv` reports a valid module. CI (Linux runner) can repeat the build.

**Acceptance:** Scaffold compiles cleanly, no validation-layer errors when loaded via a stub Vulkan dispatch.

#### T7.V.2 — Port CUDA tile rasterizer to GLSL compute

**Goal:** Translate the same Image-GS CUDA tile rasterizer into GLSL compute (`rasterizer.comp`). Tile size 16, top-K accumulation in `shared` memory.

**Steps:**
1. Read CUDA source. Map CUDA → GLSL primitives:
   - `__shared__` ↔ GLSL `shared` qualifier on local arrays.
   - CUDA warp intrinsics ↔ `subgroupBallot`, `subgroupShuffle` (`GL_KHR_shader_subgroup_ballot`, `_shuffle`).
   - RDNA 2 wave size: **64 threads**, configurable via `local_size_x_id` specialization constant. Apple-style 32-thread waves do **not** apply on Deck.
2. Workgroup size: 256 threads, `local_size_x = 16, local_size_y = 16` to match a 16×16 tile, one thread per pixel.
3. Use `VK_KHR_shader_float16_int8` for FP16 storage of Gaussian params; accumulator stays FP32 to avoid drift on Pico's 1K Gaussian count.
4. Write the kernel body, mirroring CUDA structure. Inline-comment translation choices.
5. C++ host harness: `rasterizer.cpp` does the minimum Vulkan dance — instance, physical device, queue, descriptor set, dispatch. Pure validation harness; production driver is Python (T7.V.4).

**Verify:** Single-tile fixture parity vs Python reference renderer, same tolerance as T7.M.2.

#### T7.V.3 — ncnn network export via PNNX

**Goal:** Export the same Sprint 4 checkpoint (Pico tier — 1K target) to ncnn's `.param` + `.bin` pair via the PyTorch → PNNX → ncnn pipeline already declared in `pyproject.toml [vulkan]`.

**Steps:**
1. From `oss/gaussian/ports/vulkan_ncnn/export_ncnn.py`: load Sprint 4 Pico checkpoint.
2. Trace via `torch.jit.trace` with a representative input (`(1, 12, 360, 640)` for 1280×800 with 2× upscale and 16-tile alignment).
3. Run PNNX: `pnnx model.pt inputshape=[1,12,360,640]`. PNNX emits `.param` + `.bin` pair plus a parity-check Python script.
4. Coverage audit — produce a written analysis of which network ops PNNX→ncnn supports natively and which need fallback. Audit lives in `docs/superpowers/gaussian-port-vulkan-ncnn.md` (this sprint's design doc). Known questionable ops in our network: `GroupNorm`, `SiLU`, transposed-conv `UpBlock`. ncnn does have ops for all three; verify the exported `.param` does not have any `Custom` layers.
5. Save to `oss/gaussian/ports/vulkan_ncnn/checkpoints/param_net_pico.{param,bin}`. Binary excluded from git, export script is the source of truth.

**Verify:** `python -m oss.gaussian.ports.vulkan_ncnn.export_ncnn --tier pico --check` runs export + a single-batch parity check (ncnn CPU vs PyTorch CPU). Per-output mean abs diff < 1e-2.

#### T7.V.4 — Steam Deck integration: end-to-end pipeline on Sintel

**Goal:** Same as T7.M.4 but Deck-side. Driver script + Vulkan renderer (T7.V.2) + ncnn net (T7.V.3) running end-to-end on Sintel.

**Steps:**
1. Driver: `oss/gaussian/ports/vulkan_ncnn/run_sintel.py`. Python entry point, calls into the C++ Vulkan renderer via a small CFFI / ctypes wrapper around the host harness from T7.V.1.
2. Network inference uses `ncnn-python` bindings (from the `vulkan` extra) configured for the `ncnn::VulkanDevice`. ncnn auto-detects the Deck's RDNA 2 Vulkan device.
3. Pipeline: Sintel LR + G-buffers → ncnn Vulkan inference → OutputHead decode (CPU) → Vulkan rasterizer dispatch → readback → save PNG.
4. **Configuration:** Pico tier checkpoint, K_per_tile=3, 1K Gaussian budget, output 1280×800.
5. SSH to the Deck via Tailscale (`tailnet-ssh.md`), copy the build, run 50 Sintel frames.

**Verify:** Output PNGs exist. PSNR vs Sintel HR within ±0.5 dB of the M3 Max Lite-tier output (Pico is the lower-budget tier so absolute PSNR is lower; we're checking the model behaves consistently across tiers).

#### T7.V.5 — Steam Deck benchmark + thermal observation

**Goal:** Per-frame time on Deck at the Pico tier configuration. Capture power draw + thermal throttling behavior during a sustained 1-minute Sintel loop.

**Steps:**
1. Bench script extends T7.M.6 schema. Configs: 1K Gaussians (production), 2K (headroom probe).
2. Resolution: 1280×800 only (Deck native), informational 800×600 lower bound.
3. Sustained run: 60 seconds of continuous Sintel rendering. Sample `/sys/class/hwmon/.../temp1_input` every second. Note the frame at which APU temp first exceeds 90°C (throttle threshold) if it occurs.
4. Output `bench_results_deck.csv`. Add a `power_draw_watts` column populated from `powerstat` or `turbostat` if available; else "n/a".
5. Sanity check vs the legacy `2026-04-30-v0.2-deck-first-design.md` § 8 power table — Deck thermal budget for our Pico-tier render must leave ≥2 W headroom for the rest of the game (since the gaming integration is the Sprint 8+ work).

**Verify:** CSV exists. p99 frame time at 1K Gaussians + 1280×800 fits within the Deck's 60 fps budget (≤ 16.6 ms) **including** the future game's render time — i.e. our portion alone must be ≤ 4 ms to leave headroom.

---

### Cross-track tasks

#### T7.X.1 — Cross-tier comparison report

**Goal:** Single document comparing all four tiers using identical Sintel test frames and identical (Sprint 4) trained weights with only the Gaussian count differing.

**Steps:**
1. Aggregate: `bench_results_3080ti.csv` (Sprint 1 § T1.6), `bench_results_4090.csv` (informational, only if access available), `bench_results_m3max.csv` (T7.M.6), `bench_results_deck.csv` (T7.V.5).
2. Compute per-tier: PSNR mean/min/max, SSIM mean, p50/p95/p99 frame time, frames/second sustained, power draw (where measured).
3. Author `docs/superpowers/gaussian-cross-tier-bench.md` with a single table + a one-paragraph commentary on whether the "single trained model, Gaussian-count-only knob" claim from spec § 2 holds.
4. Generate one side-by-side comparison PNG (Deck Pico vs M3 Max Lite vs 3080 Ti Standard, same Sintel frame).

**Verify:** Document committed. Numbers reproduce by re-running the per-tier bench scripts.

#### T7.X.2 — Sprint 7 code review checkpoint

**Goal:** Run the Sprint 1+ code review pipeline on Sprint 7 commits.

**Steps:**
1. `python -m oss.gaussian.review.run --sprint 7 --commit-range <sprint-6-tip>..HEAD`.
2. Reviewer A: correctness, security (kernel-side memory safety in MSL/GLSL, ncnn descriptor management), idiomatic style.
3. Reviewer B: spec adherence — do the ports match § 4.2 of the design spec? Test coverage of the dry-run + hardware-conditional paths in `tests/gaussian/test_ports.py`?
4. Judge verdict APPROVE → Sprint 7 closes, graduation decision begins.
5. REQUEST_CHANGES → iterate. BLOCK → escalate.

**Verify:** Judge verdict file at `oss/gaussian/review/artifacts/sprint-7/judge.json` reads APPROVE.

---

## 4. Risks + mitigations

1. **MSL `simdgroup_ballot` / GLSL `subgroupBallot` semantic drift from CUDA warp intrinsics.** Mitigation: cover with the per-tile parity test in T7.M.2 / T7.V.2 against the PyTorch reference renderer. If divergence > 1e-2, fall back to a non-subgroup top-K implementation (slower but portable).
2. **CoreML drops a layer.** `coremltools` historically chokes on custom ops or unusual conv configurations. Mitigation: T7.M.3 keeps the network architecture pinned to ops in the `coremltools` op-coverage matrix (`Conv2d`, `GroupNorm`, `SiLU`, `ConvTranspose2d`). All four are supported as of `coremltools` 8.x.
3. **PNNX drops a layer.** Same risk on the ncnn side. Mitigation: T7.V.3 audits the exported `.param` for `Custom` layers; if any appear, replace the PyTorch op with a PNNX-supported equivalent before export. The fallback list is documented in the Vulkan port design doc.
4. **Deck thermal throttle hits before the bench loop completes.** Mitigation: T7.V.5 explicitly measures throttle behavior; if our Pico-tier renderer is throttle-bound on its own, we reduce Gaussian count to 512 and re-bench rather than ship a tier that can't sustain its budget.
5. **CrossOver smoke test (T7.M.5) reveals Metal context contention with DXMT.** Mitigation: this is a smoke test, not a ship blocker. If contention is observed, document it, ship native-only on M3 Max for v1, defer CrossOver-coexisting integration to v1.1.
6. **Sprint 4 weights underperform on lower Gaussian budgets.** Mitigation: if Pico-tier (1K) PSNR is unacceptable, the contingency is a single re-train with explicit per-tier dropout regularization on Gaussian count (planned in Sprint 4 § T4.10 ablation). Adds ~3 days; not on the Sprint 7 critical path because the trained checkpoint already exists.

---

## 5. Success criteria (echo of § 0)

Sprint 7 is **APPROVE-able** when:

- Both ports compile cleanly from the scaffold via their stock toolchains (Xcode metal / Vulkan SDK glslangValidator).
- Both ports run the Sintel offline benchmark end-to-end with the **same Sprint 4 checkpoint**.
- M3 Max bench numbers fit the bandwidth-scaling sanity check vs 3080 Ti.
- Deck bench numbers fit the Pico-tier thermal + frame-time budget.
- Cross-tier report committed and the single-knob claim either confirmed or explicitly disconfirmed with data.
- Code review judge verdict APPROVE.

Anything else — production game integration on either platform, a Deck DLL-swap layer, ANE-only inference path on M3 Max — is post-v1.
