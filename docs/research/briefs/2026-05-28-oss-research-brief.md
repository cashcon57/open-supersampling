# OSS Research Brief — 2026-05-28

## Project Status Summary

**Latest commits (top 10):**
```
0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc
56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)
d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume
e1ab7fe hat.py: use flash-attn 2 directly when available, fallback to SDPA
e9b4b65 trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)
26cd9da dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB
83b4f07 v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast
baf4b02 worker: apex .com → .org 301 redirect
eb0ec2e trainer: bf16 autocast around forward + loss
5c2f2e7 trainer: add throughput baselines (cudnn.benchmark, TF32, torch.compile)
```

**Open issues (12):** #14 (VRAM footprint), #13 (Nsight L2 profile), #12 (packed low-rank canvas), #10 (v6.2.1 ablation: decoupled G-buffer encoder), #9 (RepVGG/ECBSR/RepSR student backbone), #8 (cross-vendor kernel ports), #7 (distillation with FSR 4 binary signal), #6 (v6.2-pico-002 launch), #5 (No-ML shader variant), #4 (HAT-Tiny student distillation), #3 (counting-sort tile binning), #1 (R=4-8 low-rank latent rasterizer)

**Training dashboard:** v7.0 Pico OSS-FX **stalled at step 5,000/100,000** — trainer offline for ~238 hours (10 days). Dashboard data stale. Loss at 0.1195. No progress since last brief. GPU util at 27%, VRAM 13% (1.6 GiB / 12 GiB). Total run cost $0.13 (2.08 GPU-hours).

**Concern:** The trainer has been offline for 10 days with a "wipe-prep" commit preserving the step-5000 snapshot. This appears intentional — likely blocked on compute or the GPU being reclaimed. No recent commits indicate training has resumed.

---

## New Papers (arXiv)

### Gaussian Splatting / Temporal / Super-Resolution

1. **ConFi-GS: Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution** (arXiv:2605.24964, May 24 2026)
   - Jiaxiang Li, Zongtan Zhou, Zhen Tan, Yadong Liu, Dewen Hu
   - Confidence-guided HF detail injection to boost 3DGS SR. Why it matters to OSS: Directly adjacent to OSS's GS-based SR goal — their HF injection technique could inform the student distillation training recipe.

2. **Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution** (arXiv:2605.22147, May 21 2026)
   - Jiangwei Mo, Xi Lu, Hanlin Wu
   - Continuous-scale GS SR for remote sensing. Why it matters to OSS: Demonstrates continuous-scale GS SR is viable beyond ICCV papers; validates the general direction.

3. **RiGS: Rigid-aware 4D Gaussian Splatting from a Single Monocular Video** (arXiv:2605.23672, May 22 2026)
   - Chenyu Wu, Wanhua Li, Zhu-Tian Chen, Hanspeter Pfister
   - Rigid-aware 4D GS with long-term smooth transformations and high-frequency dynamics. Why it matters to OSS: 4D GS with motion modeling — the Rigid-aware decomposition could inform OSS's V_xt cross-correlation design in v7.

4. **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586, May 11 2026, ICMR 2026)
   - Nengbo Lu, Minghua Pan
   - Learning causal dynamics of 3D scenes with Gaussians. Why it matters to OSS: Physics-constrained Gaussian dynamics — potential input to the temp-consistency / extrapolation loss design.

5. **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567, May 11 2026, ICME 2026)
   - Nengbo Lu, Bin Zhao
   - Velocity field learning for Gaussians without physical priors. Why it matters to OSS: The velocity-field approach directly maps to OSS v7's N-D Gaussian primitive with native time — their velocity learning could inform OSS-FX motion extrapolation.

### Super-Resolution / Upscaling

6. **BurstGP: Enhancing Raw Burst Image Super Resolution with Generative Priors** (arXiv:2604.23508, Apr 25 2026)
   - Dong Huo et al.
   - Raw burst SR using generative priors. Why it matters to OSS: Multi-frame SR from raw inputs — relevant if OSS expands to raw-sensor game capture paths.

7. **A Scale-Adaptive Framework for Joint Spatiotemporal Super-Resolution with Diffusion Models** (arXiv:2604.21903, Apr 25 2026)
   - Max Defez et al.
   - Joint spatial+temporal SR using diffusion models. Why it matters to OSS: Demonstrates the value of joint SR+FG training (OSS's thesis) — diffusion approach is too slow for real-time, but the architecture concept validates OSS's unified approach.

8. **ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering** (arXiv:2506.13814, Jun 2025, ICML 2025)
   - Lufei Liu, Tor M. Aamodt
   - Neural network layer caching for real-time rendering inference. Why it matters to OSS: Relevant to OSS student inference optimization — layer caching could complement the <2ms budget budget via skip-connections and temporal reuse.

### Frame Generation / Extrapolation

9. **TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos** (arXiv:2508.09811, Aug 2025, ICCV 2025)
   - Jinxi Li, Ziyang Song, Bo Yang
   - 3D Gaussian physical dynamics without simulation. Why it matters to OSS: Gaussian trajectory learning — directly applicable to OSS-FX frame extrapolation via learned motion dynamics.

### Inference Optimization

10. **CUTLASS v4.5.2** — Blackwell GeForce blockscaled GEMM examples (May 27 2026)
    - NVIDIA CUTLASS team
    - Blackwell-optimized GEMM kernels with blockscaled quantization. Why it matters to OSS: If targeting Blackwell hardware, these kernels offer path to FP8 inference at <2ms — especially the blockscaled formats for the student model.

### Previous key papers (already in BIBLIOGRAPHY.md):
- **GSASR** (Hu et al., ICCV 2025) — Already cited as primary OSS prior-art baseline
- **GRAPE** (Jang & Jin, WACV 2026) — Already cited as Pico-tier student candidate
- **GS-STVSR** (Shi et al., 2026) — Already cited for covariance resampling

---

## GitHub / Open Source Monitoring

### New Projects of Interest

1. **GSASR** (`ChrisDud0257/GSASR`, ★98) — Official ICCV 2025 2D Gaussian SR code. 102 commits, active ~9-11 months ago. Custom CUDA scale-aware 2D rasterizer. **Most directly relevant open-source code to OSS's architecture.** HEAVILY overlaps with OSS's v7 2D GS approach. Worth deep-reading their CUDA rasterizer implementation.

2. **Project-FreeGen** (`ShrekDino/Project-FreeGen`, ★1, Created 2026-04-30) — Open-source Vulkan-based scaling + frame generation tool for Linux/Windows. Only 1 star, conceptually overlaps with OSS's goals. Monitor for growth.

3. **DLSS-Compositor** (`ShiinaManatsu/dlss-compositor`, ★3) — Blender Cycles rendering through DLSS RR + DLSS FG pipeline. Niche but demonstrates DLSS RR compositing pipeline.

4. **NovagPU TS1T** (`nova-studios-hw/novagpu-ts1t`, ★6, Created 2026-05-27) — Open-source GPU RTL in Verilog with HW ray tracing + frame generation. Extremely new, likely not a threat.

### Existing / Established Projects

5. **OptiScaler** (`optiscaler/OptiScaler`, ★8,535) — Maintains dominance as the DLSS/FSR/XeSS bridge tool. Supports FSR3 FG injection into non-FG titles via Nukem mod.

6. **DLSS-Enabler** (`artur-graniszewski/DLSS-Enabler`, ★1,218) — Simulates DLSS upscaler and DLSS-G FG on any DX12 GPU. Active (updated today).

### Dependency/Benchmark Projects

7. **CUTLASS v4.5.2** (May 27 2026) — Blackwell GeForce blockscaled GEMM examples added. Relevant for student inference optimization targeting Blackwell.

8. **gsplat** (nerfstudio, updated 2026-05-11) — Native CUDA kernel for MCMC perturb, CDIM padding 24→32 optimization. Active development.

9. **tiny-cuda-nn** (NVlabs, updated 2026-04-21) — Stable, low activity.

### Threat Assessment
- **GSASR repo** is the biggest threat — published 2D Gaussian SR code at ICCV 2025. OSS needs to distinguish itself via temporal/frame-extrapolation capability.
- **OptiScaler** absorbs users who might otherwise need custom SR solutions; reduces addressable market for OSS runtime.
- **No public FSR3.1 SDK** — AMD hasn't open-sourced it, leaving a gap OSS could fill for open-source FG.

---

## News

### DLSS 4
- **DLSS SDK 310.6.0** released 2026-04-16 (most recent). No new releases in past 30 days. DLSS development appears quarterly.
- Multiple game titles continue to adopt DLSS 4 Multi Frame Generation.

### Intel XeSS
- **XeSS SDK 3.0.1** (2026-04-17) — Stability fixes for XeSS-FG. Fixes crash with D3D12_RESOURCE_FLAG_USE_TIGHT_ALIGNMENT.
- **XeSS SDK 3.0.0** (2026-03-09) — Introduced 3x and 4x Multi-Frame Generation. Intel is aggressively pushing multi-FG.
- Confirms industry direction: multi-frame generation is the standard OSS-FX needs to match.

### AMD FSR
- No new FSR3.1 or FSR4 SDK announcements found.
- **FSR3.1 SDK has not been open-sourced** on GPUOpen-Effects GitHub. Still distributed through AMD developer portal only.
- AMD's silence on FSR4 is notable — either delayed or a major pivot from FSR3.

### NVIDIA TensorRT
- TensorRT 10.16.1 OSS release (2026-04-13). Continuous torch_tensorrt updates.
- CUTLASS v4.5.2 with Blackwell GEMM examples just dropped (May 27 2026).

### Industry Direction
- **Confirmed trend**: Multi-frame generation (XeSS 3.0, DLSS 4 MFG). OSS-FX's α<1 path needs to support 2x and 3x extrapolation to compete.
- **No open-source FG SDK** — OSS could fill this gap, especially for games that do not want AMD/NVIDIA vendor lock-in.
- **GSASR at ICCV 2025** sets the benchmark for 2D Gaussian SR; OSS must beat or differentiate via temporal/engine integration.

### Archive.org / Archived Papers
- **GS-STVSR** (Shi et al., arXiv:2604.18047) — Still available. Already in bibliography.
- **GSASR** — Both the arXiv paper and GitHub repo are active/available. No retraction or significant update.
- **No notable paper retractions or significant revisions** detected for key references.

---

## Action Items / Follow-ups

- [ ] **Deep read: ConFi-GS (2605.24964)** — confidence-guided HF injection technique could improve OSS student distillation quality.
- [ ] **Deep read: RiGS (2605.23672)** — Rigid-aware 4D GS motion decomposition could inform OSS v7 V_xt cross-correlation design.
- [ ] **Investigate: VeloGauss (2605.10567)** — Gaussian velocity field learning without physical priors. Maps directly to OSS-FX motion extrapolation.
- [ ] **Monitor: Project-FreeGen** — if it gains traction, could compete with OSS as an open-source FG option.
- [ ] **Read: ReFrame (2506.13814, ICML 2025)** — layer caching for real-time NN inference; potential <2ms budget optimization.
- [ ] **Compare: GSASR CUDA rasterizer** with OSS's vs gsplat. GSASR custom CUDA scale-aware 2D rasterizer may inform OSS's custom kernel roadmap.
- [ ] **Crisis flag: Training stalled 10 days at step 5K.** If GPU access is the blocker, consider:
  - Restoring from the step-5000 snapshot with a shorter run target (30K steps to get usable results)
  - Moving to a smaller Pico config that fits in a lower memory footprint
  - Documenting the v7 training loss curve thus far as a preliminary result (loss 0.1195 at 5K steps may be sufficient for a technical report)
