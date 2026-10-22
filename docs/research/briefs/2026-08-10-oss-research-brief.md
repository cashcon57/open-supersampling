# OSS Research Brief — 2026-08-10

_Collection: local Git/README, rendered public dashboard, GitHub REST, direct arXiv abstract pages (Atom rate-limited), Archive.org Advanced Search/Wayback, and primary vendor pages. “New” means newly published or newly material to OSS; re-discoveries already in `BIBLIOGRAPHY.md` are not appended again._

## Project Status Summary

- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated training controls and a dashboard panel. Adjacent work preserved the v7-pico-005 recovery checkpoint, avoids the WSL2 DataLoader deadlock, enables FlashAttention 2 when present, and adds TF32/prefetch throughput changes.
- **Open issues (10):** immediate performance chain is **#14** measure v6.2-pico-002 VRAM → **#13** Nsight L2 profile → **#12** packed low-rank canvas. Follow-on work includes G-buffer-encoder ablation (#10), reparameterized student (#9), cross-vendor kernels (#8), distillation (#7/#4), initial launch (#6), and a no-ML shader fallback (#5).
- **README snapshot:** pre-alpha v7 uses a HAT-Tiny research teacher, native `(x,y,t)` Gaussians with `V_xt` motion cross-covariance, loss-adaptive density, Mip-Splat filtering, and α=0.5 OSS-FX supervision. The ≤1M-parameter student and every non-CUDA runtime remain unbuilt.
- **Training dashboard:** headline says v7 is training at **36,650 / 100,000**, loss **0.19268**, viz **36,000**. Its health telemetry simultaneously says **Trainer — offline; stuck at step 36,650 for 27h**; watcher is healthy (published 1s ago), CF Worker is healthy (200 in 102ms), R2 is healthy, and GPU stats are unavailable. This conflicts with the README/headline “training now” wording and is **not** evidence of live progress.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **No newly published material paper cleared the gate.** The following high-value results were re-verified but are already catalogued:
  - **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** — Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha; 2026-04-20; [arXiv:2604.18047](https://arxiv.org/abs/2604.18047). 2D Gaussian continuous STVSR using covariance resampling and flow-guided motion; the closest representation-level precedent, while OSS remains differentiated by persistent `(x,y,t)` canvas state.
  - **Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution** — Du Chen, Liyi Chen, Zhengqiang Zhang, Lei Zhang; 2025-01-12; [arXiv:2501.06838](https://arxiv.org/abs/2501.06838). Feed-forward image-conditioned 2D Gaussians with scale-aware CUDA rasterization; benchmark OSS rasterizer/canvas cost against it rather than adopting its image-only design.

### Super-Resolution / Upscaling

- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** — Pham Phuong Nam Nguyen, Nam Tien Le, Thi Kim Trang Vo, Nhu Tinh Anh Nguyen; 2026-04-22; [arXiv:2604.20291](https://arxiv.org/abs/2604.20291). Deployment graph reparameterization, QAT, BN recalibration, and staged teacher guidance are directly relevant to OSS’s ≤1M-parameter student, but the paper is already in the bibliography.
- **GRAPE: Gaussian Rendering for Accelerated Pixel Enhancement Brings Fast and Lightweight Arbitrary Super-Resolution** — Jung In Jang, Kyong Hwan Jin; WACV 2026; [project](https://github.com/mulkkog/GRAPE). A 1.56M-parameter Gaussian-parameter head plus one-pass rasterizer is a strong Pico-tier quality/throughput reference, but it is single-image and already tracked.

### Frame Extrapolation / Generation

- **No credible newly published single-network real-time frame-extrapolation competitor surfaced.** **Low-latency Space-time Supersampling for Real-time Rendering** — Ruian He, Shili Zhou, Yuqi Sun, Ri Cheng, Weimin Tan, Bo Yan; 2023-12-18; [arXiv:2312.10890](https://arxiv.org/abs/2312.10890) — remains the useful unified spatial-SR/frame-extrapolation latency baseline (not Gaussian-based). It is already catalogued.

### Inference Optimization

- No new directly material TensorRT/INT8-SR or kernel-fusion paper passed the relevance gate. Treat the student deployment decision as an empirical task: close **#14 → #13 → #12** with measured bytes/frame, L2 hit rate, occupancy, DRAM traffic, tile imbalance, and image-parity gates before committing to a packed layout.

## Archive.org Findings

- Broad Archive.org searches for `gaussian splatting super-resolution`, `frame generation deep learning`, and `real-time upscaling game` were noisy (unrelated media and generic mirrors), not proof of absence.
- Targeted Wayback checks found captures for **GSASR** (arXiv, 2025-01-14 through 2026-06-07) and **STSS** (arXiv, 2023-12-19 through 2025-12-20; GitHub on 2024-04-15). No relevant Archive.org item/capture was found for GS-STVSR or GRAPE. `GSASR` and `GRAPE` keyword inventory was polluted by unrelated results.

## GitHub / Open Source

- **OptiScaler** (GPL-3.0; ~9.7k stars; pushed Aug. 10) remains the strongest compatibility signal. It bridges DLSS/XeSS/FSR inputs plus FSR-FG/XeFG and updated bundled XeSS SDK to 3.0.2 on Aug. 6. **Opportunity:** use it as a test oracle for input contracts, sync, DXGI/device spoofing, and fallback behavior—not as a dependency.
- **gsplat** (Apache-2.0; ~5.5k stars) remains CUDA-centric; its July GPU CI work covers JIT and AOT wheel paths. **Action:** make CUDA rasterization an optimization backend rather than an OSS architectural dependency; its GPU test-matrix model is worth copying.
- Frame-generation distribution is fragmenting: GPL **FrameBridge** (created July 15, pushed Aug. 7) bundles cross-vendor installation, while MIT **0ptiscaler4linux** received maintenance Aug. 4 for RDNA3/4, Arc, RTX, Proton, and Vulkan profiles. **Signal:** cross-vendor usability/packaging is a real OSS differentiation opportunity, but both are too early to depend on.
- Mature offline comparison tooling remains active: **Waifu2x-Extension-GUI** (~16.9k stars) released v3.140.01 Aug. 2 and updated Aug. 8. It is useful as an offline comparison harness, not a game-runtime architecture.

## News

- **NVIDIA / DLSS 4.5:** NVIDIA’s Aug. 4 *Gears of War: E-Day* post positions UE5/DX12 as a DLSS 4.5 reference target. **Threat:** the commercial quality bar bundles temporal SR, frame generation, ray reconstruction, and engine-side inputs. OSS comparisons must separate reconstruction, temporal stability, latency, and generated-frame artifacts rather than report FPS alone.
- **AMD / FSR:** current AMD FSR SDK documents Upscaling 4.1.1, Frame Generation 4.0.1, Ray Regeneration 1.2.0, and discrete RDNA3 RX 7000 support. **Opportunity:** its API/input contract and UE plugin are integration references; **threat:** signed prebuilt ML-effect DLLs mean it is not an open implementation to reuse.
- **Intel / XeSS:** Intel’s current announcement says upcoming partner titles will ship the XeSS 3 stack—Super Resolution, Multi-frame Generation, and Xe Low Latency. Cross-vendor-facing integration plus SR/FG/latency composition is becoming baseline market expectation.
- **Apple / Metal:** no fresh primary announcement in the window, but current Metal material describes a redesigned MetalFX temporal upscaler using Neural Engine / M5 Pro-Max neural accelerators, Metal 4 command-buffer ML encoding, and TensorOps. Prototype a compact fixed-shape Apple backend early; do not assume a CUDA-style execution model.
- **Unreal / Unity:** no primary engine upscaler release was verified in the target window. AMD documents UE coverage through UE 5.7, while NVIDIA’s UE5/DX12 example remains the current integration battleground.

## Action Items / Follow-ups

- [ ] **Critical:** restore the 3080 Ti trainer/metric emitter; require an advancing step, fresh GPU telemetry, and a new visualization before calling v7 live.
- [ ] Close **#14 → #13 → #12** with bytes/frame, L2/occupancy/DRAM/tile-imbalance measures and image-parity gates; evaluate GS-STVSR/GSASR/GRAPE only as matched baselines.
- [ ] Build a reproducible cross-vendor validation scene measuring SR quality, temporal stability, OSS-FX artifacts, motion-vector sensitivity, and latency separately against DLSS/FSR/XeSS.
- [ ] Preserve SR-only as the baseline, then make OSS-FX and reconstruction/denoise optional stages with explicit depth/motion/color contracts.
- [ ] Run feasibility probes for Vulkan/compute and Metal/ANE before cementing CUDA-only raster or packed-canvas choices.

## Sources

- https://opensupersampling.org
- https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open
- https://arxiv.org/abs/2604.18047
- https://arxiv.org/abs/2501.06838
- https://arxiv.org/abs/2604.20291
- https://arxiv.org/abs/2312.10890
- https://github.com/mulkkog/GRAPE
- https://github.com/optiscaler/OptiScaler
- https://github.com/nerfstudio-project/gsplat
- https://github.com/dreyvinixz/FrameBridge
- https://github.com/ind4skylivey/0ptiscaler4linux
- https://www.nvidia.com/en-us/geforce/news/gears-of-war-e-day-multiplayer-beta-august-6-dlss/
- https://gpuopen.com/amd-fsr-sdk/
- https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships
- https://developer.apple.com/metal/whats-new
