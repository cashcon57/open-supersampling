# OSS Research Brief — 2026-06-07

## Project Status Summary
- Latest commits:
  a97a180 feat: add tailnet-gated training controls to dashboard
  fcf8084 feat: tailnet-gated training control API + dashboard panel
  0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc
  56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)
  d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume
  e1ab7fe hat.py: use flash-attn 2 directly when available, fallback to SDPA
  e9b4b65 trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)
  26cd9da dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB
  83b4f07 v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast
  baf4b02 worker: apex .com → .org 301 redirect
- Open issues:
  - #14: D3: measure v6.2-pico-002 VRAM footprint after v6.2 wiring and checkpoint exist
  - #13: H004: run Nsight L2 hit-rate profile on 3080 Ti CUDA host
  - #12: H004: implement packed low-rank canvas layout before claiming 32B/Gaussian
  - #10: v6.2.1 ablation: decoupled G-buffer encoder before concat-fusion
  - #9: v6.2.1 ablation: RepVGG/ECBSR/RepSR reparameterization for student backbone
  - #8: v6.2: cross-vendor kernel ports (HIP / Metal / Vulkan / SYCL)
  - #7: v6.2.1: distillation pipeline with FSR 4 binary as teacher signal
  - #6: v6.2-pico-002: launch on 3080 Ti once codex T1-T5 land
  - #5: v6.2: No-ML shader variant (ships alongside v6.2)
  - #4: v6.2: HAT-Tiny student distillation (≤1M params)
- README status snapshot: v7 is current research/teacher architecture (`srcnn-v7.0-pico-005`) with N-D Gaussian canvas + native time axis; distilled ≤1M-param student remains not yet trained; cross-vendor kernels/integration path still designed/not built.
- Training dashboard: **Active run:** `srcnn-v7.0-pico-005` / **v7.0 Pico OSS-FX**; | **Trainer** | **offline** | Stuck at **step 17,700 for 18h**. Background trainer on the **3080 Ti host** emits `metrics.json` every step into the active run directory. |; - **Current loss:** **0.10913**
- Notable operational flag: dashboard reports trainer offline/stuck at step 17,700 while watcher/Worker/R2 are healthy. Treat this as urgent project health follow-up.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting** (arXiv:2603.13783v2, 2026) — Continuous-time 4DGS reconstruction with supervision designed around retiming/intermediate-time rendering. Why it matters to OSS: Direct OSS-FX/v7 prior: compare its continuous-time parameterization and temporal-aliasing mitigation against OSS native `(x,y,t)` Gaussians. Authors: Wang et al..
- **GS-NFS: Bandwidth-adaptive Streaming of Dynamic Gaussian Splats and Point Clouds** (arXiv:2606.05650v1, 2026-06-04) — Bandwidth-adaptive streaming system for dynamic 3D Gaussian splats / point-cloud video. Why it matters to OSS: Opportunity for future canvas streaming/compression; not core SR quality, but useful if OSS persists or transmits Gaussian state. Authors: Ghosh et al..
- **ZipSplat: Fewer Gaussians, Better Splats** (arXiv:2606.05102v1, 2026-06-03) — Feed-forward 3DGS work reducing Gaussian count instead of tying one Gaussian to each input pixel. Why it matters to OSS: Worth skimming for v7 canvas-count pressure and parent/child density control; lower splat counts map directly to rasterizer latency. Authors: Veicht et al..

### Super-Resolution / Upscaling
- **MaCo-GAN: Manifold-Contrastive Adversarial Learning for Single Image Super-Resolution** (arXiv:2606.05068v1, 2026-06-03) — Contrastive adversarial SR objective focused on preserving LR correspondence and reducing hallucinated artifacts. Why it matters to OSS: Candidate perceptual-stage loss if OSS adds GAN training while trying to avoid texture hallucination and temporal shimmer. Authors: Han, Hwang, Jeon.
- **Wavelet-Space Representations for Neural Super-Resolution in Rendering Pipelines** (arXiv:2508.16024v3, 2025) — Rendering-pipeline SR approach predicting wavelet coefficients aligned with auxiliary buffers/history. Why it matters to OSS: Potential head/loss ablation for texture recovery: wavelet-space supervision could complement Gaussian rasterizer AA. Authors: Poudel et al..
- **BurstGP: Enhancing Raw Burst Image Super Resolution with Generative Priors** (arXiv:2604.23508v1, 2026-04-26) — Burst SR using multiple LR frames plus generative priors. Why it matters to OSS: Adjacent temporal SR reference for how to exploit multi-frame redundancy without blindly trusting hallucinated details. Authors: Huo et al..

### Frame Extrapolation / Generation
- **G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (arXiv:2406.18551, 2024) — Real-time frame extrapolation without requiring engine G-buffers. Why it matters to OSS: Competitive foil: lower integration burden than OSS; OSS needs quality/latency evidence to justify exact engine MV/G-buffer dependence. Authors: Wu et al..
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (arXiv:2603.26835v3, 2026) — Video interpolation design using available motion-vector priors and accelerator-friendly deployment constraints. Why it matters to OSS: Supports OSS’s thesis that motion priors plus small residual networks are the right path for low-latency temporal synthesis. Authors: Liu.
- **SwiftI2V: Efficient High-Resolution Image-to-Video Generation via Conditional Segment-wise Generation** (arXiv:2605.06356v2, 2026-05-07) — Efficient high-resolution image-to-video generation via segment-wise conditioning. Why it matters to OSS: Not real-time game FG, but useful for training ideas around long-horizon temporal consistency and segment conditioning. Authors: Liu et al..

### Inference Optimization
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (arXiv:2604.20291, 2026) — INT8 SR with deployment-aware QAT, teacher guidance, and fused deploy graph training. Why it matters to OSS: Directly relevant to the ≤1M-param student and INT8/FP8 TensorRT path; deep-read for loss scheduling and calibration. Authors: Nguyen et al..
- **Spike-Aware C++ INT8 Inference for Sparse Spiking Language Models on Commodity CPUs** (arXiv:2606.03026v1, 2026-06-02) — INT8 inference for sparse spiking LMs on commodity CPUs. Why it matters to OSS: Only weakly relevant; keep as a generic sparse/INT8 runtime reference, not an OSS architecture driver. Authors: Liu.

### Game Engine Integration
- **G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (arXiv:2406.18551, 2024) — Frame extrapolation explicitly targeting low-latency rendering without G-buffer inputs. Why it matters to OSS: Threat/opportunity: easier drop-in integration than OSS, but potentially weaker quality than exact-MV/G-buffer OSS-FX. Authors: Wu et al..

## Archive.org Findings
- Archive search did not reveal a takedown-only or materially different archived copy of GS-STVSR, GSASR, GRAPE, or STSS.
- Useful archived adjacent artifact: Anime4K GitHub snapshot — historical open-source real-time upscaling baseline, but not architecture-changing for OSS. https://archive.org/details/github.com-bloc97-Anime4K_-_2019-10-11_05-29-51
- Scholar Archive hits for generic splatting/resampling were mostly old rendering/resampling PDFs, not new GS-SR papers.

## GitHub / Open Source
- **GaussianSR** — AAAI 2025 2D Gaussian splatting arbitrary-scale SR code remains the most direct public single-image GS-SR baseline. https://github.com/tljxyys/GaussianSR
- **GSASR** — generalized efficient 2DGS arbitrary-scale SR with custom CUDA scale-aware rasterizer; still the baseline to benchmark against for Gaussian SR. https://github.com/ChrisDud0257/GSASR
- **NagasaiManda/SplatSuRe** — selective SR for multi-view-consistent 3DGS; updated 2026-04-26. Opportunity: apply expensive SR only where confidence/undersampling requires it. https://github.com/NagasaiManda/SplatSuRe
- **OptiScaler** — active bridge layer for upscaling/frame-gen DLL paths, including experimental DX12 frame-generation/HUD-fix work. Threat/opportunity for OSS integration expectations. https://github.com/optiscaler/OptiScaler
- **Nukem9/dlssg-to-fsr3** — DLSS-FG-to-FSR3 compatibility shim remains relevant prior art for drop-in frame-gen routing. https://github.com/Nukem9/dlssg-to-fsr3
- **shepherdscientific/neural-spatial-upscaler** — small FPGA/ternary-CNN real-time upscaler repo; not quality-competitive, but useful as an extreme latency/area reference. https://github.com/shepherdscientific/neural-spatial-upscaler

Supplemental search hits:
- **GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale ...** — GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution · 1. Environment · 2. Running the code. https://github.com/tljxyys/GaussianSR
- **open-supersampling/docs/research/2026-05-05-existing-gaussian ...** — tljxyys/GaussianSR — High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution (AAAI 2025). URL: https://github.com/tljxyys/GaussianSR ... https://github.com/cashcon57/open-supersampling/blob/main/docs/research/2026-05-05-existing-gaussian-splatting-repos-survey.md
- **3DGS and Beyond Docs - GitHub** — GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution Jintong Hu, Bin Xia, Bin Chen, Wenming Yang, Lei Zhang arXiv ... https://github.com/yangjiheng/3DGS_and_Beyond_Docs
- **mulkkog/GRAPE - GitHub** — GRAPE is a fast and lightweight framework for arbitrary-scale super-resolution (ASSR) based on 2D Gaussian splatting. ... GaussianSR, 1.84M, 16.15 GB, 6.19 ... https://github.com/mulkkog/GRAPE
- **ChrisDud0257/GSASR: Official PyTorch Code for our ICCV25 paper ...** — This work presents GSASR. It achieve SoTA in arbitrary-scale super-resolution by representing given LR image as millions of continuous 2D Gaussians. https://github.com/chrisdud0257/gsasr

## News
- **NVIDIA DLSS 4.5 Delivers Super Resolution Upgrades and New ...** — NVIDIA introduced DLSS 4.5 at CES 2026, featuring a second-generation transformer model for Super Resolution and a 6x Multi Frame Generation ... https://developer.nvidia.com/blog/nvidia-dlss-4-5-delivers-super-resolution-upgrades-and-new-dynamic-multi-frame-generation/
- **NVIDIA DLSS 4.5 Delivers Major Upgrade With 2nd Gen ...** — Upgrade coming to a massive library of over 250 titles with DLSS Multi Frame Generation, and over 400 titles with DLSS Super Resolution. https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-gen-6x-2nd-gen-transformer-super-res/
- **DLSS 4.5 | Enhanced Super Resolution & Dynamic Multi Frame Gen** — Learn More about DLSS 4.5: https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-gen-6x-2nd-gen-transformer-super-res DLSS ... https://www.youtube.com/watch?v=u4cfzFd9HzU
- **Nvidia introduces DLSS 4.5 and Multi Frame Generation 6X at CES ...** — Nvidia introduces DLSS 4.5 and Multi Frame Generation 6X at CES 2026 — updated models can generate higher-quality upscaled frames and more of ... https://www.tomshardware.com/pc-components/cpus/nvidia-introduces-dlss-4-5-and-multi-frame-generation-6x-at-ces-2026-updated-models-can-generate-higher-quality-upscaled-frames-and-more-of-them-dynamically
- **NVIDIA DLSS 4.5 to feature 2nd Gen Transformer model ... - NeoGAF** — Dynamic Multi Frame Generation, as the name suggests, is not locked to a single multiplier. Instead, users may see 3x to 6x more frames, ... https://www.neogaf.com/threads/nvidia-dlss-4-5-to-feature-2nd-gen-transformer-model-and-dynamic-6x-frame-generation.1691936/
- **DLSS 4.5 Ray Reconstruction + 1000 RTX Games - NVIDIA** — On June 15th, a new update will add native support for DLSS 4.5 Super Resolution and Dynamic Multi Frame Generation with up to 6X Frame ... https://www.nvidia.com/en-us/geforce/news/dlss-4-5-ray-reconstruction-1000-rtx-games-apps-out-now/
- **AMD FSR 4 Confirmed for Older GPUs, July Launch Set for RDNA 3** — AMD confirms FSR Upscaling 4.1 is coming to RDNA 3 GPUs in July and RDNA 2 in 2027, finally bringing FSR 4 tech to older Radeon cards. https://hothardware.com/news/amd-bringing-fsr-41-to-rdna3-rdna2
- **AMD FSR Upscaling 4.1 for RDNA 2 and 3 | Hardware Canucks** — Per VCZ reporting, AMD is planning on releasing FSR Upscaling 4.1 for RX 6000 and RX 7000 series. RX 7000 will get support in July 2026 RX 6000 https://hardwarecanucks.com/forum/threads/amd-fsr-upscaling-4-1-for-rdna-2-and-3.90448/
- **FSR Upscaling 4.1 Coming to AMD RDNA 3 Radeon RX 7000 ...** — It's official: FSR 4.1 is making its way to AMD Radeon RX 7000-series graphics cards and is set to arrive in July. https://www.thefpsreview.com/2026/05/15/fsr-upscaling-4-1-coming-to-amd-rdna-3-radeon-rx-7000-series-gpus-in-july/

## Action Items / Follow-ups
- [ ] Investigate/restart `srcnn-v7.0-pico-005`; dashboard says trainer is stuck/offline at step 17,700 while publish infra is healthy.
- [ ] Deep-read **RetimeGS** for continuous-time Gaussian supervision and temporal anti-aliasing; compare to v7 `(x,y,t)` covariance design.
- [ ] Deep-read **INT8-SR-QAT** for student deployment training: QAT schedule, teacher confidence weighting, and fused graph constraints.
- [ ] Evaluate **GFFE** as an integration-threat baseline: what quality does OSS gain by requiring engine motion/G-buffer signals?
- [ ] Re-check OptiScaler / DLSSG-to-FSR3 integration paths when OSS DLL-shim work resumes.
