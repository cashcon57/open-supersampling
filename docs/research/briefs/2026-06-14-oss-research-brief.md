# OSS Research Brief — 2026-06-14

## Project Status Summary
- Latest commits:
  - `a97a180 feat: add tailnet-gated training controls to dashboard`
  - `fcf8084 feat: tailnet-gated training control API + dashboard panel`
  - `0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc`
  - `56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)`
  - `d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume`
  - `e1ab7fe hat.py: use flash-attn 2 directly when available, fallback to SDPA`
  - `e9b4b65 trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)`
  - `26cd9da dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB`
  - `83b4f07 v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast`
  - `baf4b02 worker: apex .com → .org 301 redirect`
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
- README snapshot: v7 remains the active architecture direction: persistent Gaussian canvas + native time axis, OSS-FX α-time extrapolation, and a future ≤1M-param distilled student; the dashboard explicitly states research/teacher models are not the shipping inference path.
- Training dashboard: rendered `https://opensupersampling.com` redirects to `.org`. Visible state: **v7.0 Pico OSS-FX** at step **30,300 / 100,000**, loss **0.18269**, viz @ step **26,000**, ETA ~1.2d, cost ~$0.76 / 12.63 GPU-hours. Service health is contradictory: page headline says `v7 is training`/`live`, but tooltip says **Trainer — offline, stuck at step 30,300 for 9h**; Watcher/CF Worker/R2 are healthy, GPU stats unavailable (`host offline or rsync failed`). Compared with 2026-06-13, run advanced +4,550 steps but now looks stuck much longer.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **MooMIns — Monocular 3D Reconstruction and Object Pose Estimation from Multiple Instances** (arXiv:2606.14389v1, 2026-06-12; Robert Langendörfer, Markus Hillemann, Markus Ulrich) — Inverts GS by reconstructing multiple object instances from one monocular image. Why it matters to OSS: low direct SR value, but useful as another feed-forward/object-instance Gaussian decomposition example.
- **SplatlessDF: Continuous Distance Field Mapping with Non-Splatting Gaussians** (arXiv:2606.13990v1, 2026-06-12; Monisha Mushtary Uttsha, Lan Wu, Teresa Vidal-Calleja) — Uses anisotropic Gaussian elements as continuous distance-field primitives and can couple with 2DGS. Why it matters to OSS: indirect, but suggests Gaussian primitives can carry non-photometric geometry fields; possible future G-buffer/depth confidence representation, not a current architecture pivot.
- **Flex4DHuman: Flexible Multi-view Video Diffusion for 4D Human Reconstruction** (arXiv:2606.13655v1, 2026-06-11; Jen-Hao Cheng, Yipeng Wang, Hao Zhang, Gengshan Yang, Jenq-Neng Hwang) — Generates dense multi-view videos from sparse/monocular video, then feeds 4DGS reconstruction. Why it matters to OSS: confirms multi-axis spatiotemporal/view positional encodings around 4D Gaussian reconstruction, but it is content-generation scale, not real-time game SR.
- **Wild3R: Feed-Forward 3D Gaussian Splatting from Unconstrained Sparse Photo Collection** (arXiv:2606.11894v2, 2026-06-10; Yuto Furutani, Takashi Otonari, Kaede Shiohara, Toshihiko Yamasaki) — Feed-forward 3DGS trained for diverse lighting/transient objects. Why it matters to OSS: supports feed-forward Gaussian prediction as a class; transient-content handling may map to disocclusion/particle/noise robustness.
- **Envision4D: Envisioning Visual Futures via Feed-forward 4D Gaussian Splatting for Autonomous Driving** (arXiv:2606.10656v1, 2026-06-09; Qi Song et al.) — Feed-forward 4DGS for future scene extrapolation with temporal attention and pose prediction. Why it matters to OSS: already added recently, still high-value for OSS-FX future-frame framing; game-engine exact motion vectors should be a stronger prior than its pose-free setting.
- **PersistGS: Differentiable Physics for Object Permanence in 4D Gaussian Splatting** (arXiv:2606.03479v1, 2026-06-02; Adrian Ramlal, John S. Zelek) — Rigid-body simulation keeps object Gaussians plausible through occlusion. Why it matters to OSS: already tracked; remains relevant for persistent canvas disocclusion and long-horizon stale-splat handling.

### Super-Resolution / Upscaling
- **EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution** (arXiv:2606.13580v1, 2026-06-11; Dachun Kai, Jiayao Lu, Yueyi Zhang, Xiaoyan Sun) — Event signals enhance high-frequency texture and temporal alignment in VSR. Why it matters to OSS: games lack event cameras, but the paper cleanly separates high-frequency texture recovery from bulk alignment; this is analogous to using engine MVs/G-buffers for low-cost alignment and a separate detail gate.
- **LiteVSR: Lightweight Adaptation of Frozen Diffusion Transformers for Video Super-Resolution** (arXiv:2606.09250v1, 2026-06-08; Yu Cao, Ziquan Liu, Zhensong Zhang, Jiankang Deng et al.) — Frozen DiT + small state-aware adapter for VSR; claims 11.25% trainable params and 12 A100-hours. Why it matters to OSS: not a runtime path, but adapter-only VSR is a useful teacher/distillation reference if HAT-Tiny teacher tuning becomes too expensive.
- **DTG-Restore: Training-Free Diffusion Refinement for Generative Video Super-Resolution** (arXiv:2605.30431v1, 2026-05-28; Hidir Yesiltepe et al.) — Decoupled-time guidance for restoring distorted low-res videos without training. Why it matters to OSS: mostly off-path due diffusion cost, but its benchmark of warped temporal artifacts is a reminder that perceptual SR metrics can hide geometry errors.
- **QuantSR+: Pushing the Limit of Quantized Image Super-Resolution Networks** (arXiv:2605.22351v1, 2026-05-21; Haotong Qin et al.) — Low-bit SR via bit redistribution, slimmable architecture, and block-aware distillation. Why it matters to OSS: already in bibliography; directly relevant to ≤1M-param INT8/FP8 student training and calibration.
- **DiffST: Spatiotemporal-Aware Diffusion for Real-World Space-Time Video Super-Resolution** (arXiv:2605.13182v1, 2026-05-13; Zheng Chen et al.) — One-step/video-level diffusion STVSR with cross-frame context aggregation. Why it matters to OSS: reinforces that single-model space-time SR/VFI is the right problem framing, but diffusion runtime remains too heavy for game budgets.

### Frame Extrapolation / Generation
- **R5DGS: Semantic-Aware 4D Gaussian Splatting with Rigid Body Constraints for Efficient Dynamic Scene Reconstruction** (arXiv:2605.25909v1, 2026-05-25; Denis Gridusov, Maxim Popov, Sergey Kolyubin) — Adds semantic identity and rigid-body inference constraints to 4DGS, with reported 11 FPS extrapolation speedup. Why it matters to OSS: already tracked; strong reference for splitting static/rigid/transient Gaussian channels and only running dynamics where needed.
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586v1, 2026-05-11; Nengbo Lu, Minghua Pan) — Learns velocity/material factors for future-frame extrapolation from multi-view video. Why it matters to OSS: useful future-loss inspiration, but less direct than OSS’s exact engine motion vectors.
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567v1, 2026-05-11; Nengbo Lu, Bin Zhao) — Per-Gaussian velocity fields with physics code and global constraints. Why it matters to OSS: already added; remains a top deep read for v7 `V_xt`, `t_query`, and velocity-loss design.
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (arXiv:2603.26835v3, 2026-03-27; Shibo Liu) — Uses codec MVs to remove learned optical flow from mobile VFI and hits 12.8 ms 1080p INT8 on Snapdragon 8 Gen 3. Why it matters to OSS: already tracked; very strong analogue for using engine MVs to delete expensive optical-flow/frame-gen components.
- **PSNR_DIV: An Efficient Quality Metric for Video Frame Interpolation Based on Motion-Field Divergence** (arXiv:2510.01361v2, 2025-10-01; Conall Daly, Darren Ramsook, Anil Kokaram) — Fast VFI quality metric weighting errors by motion-field divergence. Why it matters to OSS: candidate temporal metric/loss for OSS-FX where divergence/occlusion boundaries dominate artifacts.

### Inference Optimization
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (arXiv:2604.20291v1, 2026-04-22; Pham Phuong Nam Nguyen et al.) — Three-stage quantized SR pipeline: LR-space student, reparameterizable backbone, DCT supervision, teacher-guided QAT on deploy graph. Why it matters to OSS: already in bibliography; direct recipe for student distillation and quantization-aware export.
- **Systematic Optimization of Real-Time Diffusion Model Inference on Apple M3 Ultra** (arXiv:2605.16259v1, 2026-02-10; Yoichi Ochiai) — Reports that CUDA intuitions do not transfer cleanly to Apple Silicon; quantization/parallel inference/Neural Engine may underperform. Why it matters to OSS: useful warning for Metal backend planning: measure backend-specific bottlenecks instead of assuming NVIDIA-style TensorRT wins carry over.
- **Towards Clip-Free Quantized Super-Resolution Networks: How to Tame Representative Images** (arXiv:2308.11365v1, 2023-08-22; Alperen Kalay, Bahri Batuhan Bilecen, Mustafa Ayazoglu) — Representative-dataset augmentation removes clipped activations and improves INT8 SR runtime/quality. Why it matters to OSS: candidate calibration method if QAT is too slow for every student variant.
- **Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution** (arXiv:2308.02794v1, 2023-08-05; Yong Liu et al.) — TensorRT-friendly SR transformer with layernorm/operator-fusion substitutions. Why it matters to OSS: already tracked; relevant if any attention-like student block survives into deployment.
- **Efficient Integer-Arithmetic-Only Convolutional Neural Networks** (arXiv:2006.11735v1, 2020-06-21; Hengrui Zhao, Dong Liu, Houqiang Li) — Integer-only CNNs for SR/compression tasks. Why it matters to OSS: fallback mental model for deterministic cross-vendor INT8 if FP8 path becomes vendor-fragmented.

### Game Engine / Motion Vectors
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694v1, 2026-03-08; Yuhang Wang et al.) — Uses codec MVs/residuals/frame types for faster online VSR. Why it matters to OSS: codec metadata is a weaker analogue of engine MVs/G-buffers; supports OSS’s thesis that privileged motion/residual channels beat learned flow.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483v1, 2023-08-03; Antoine Mercier et al.) — Game-specific neural supersampling dataset with motion vectors/depth/jitter/mipmap bias. Why it matters to OSS: still a key public game-SR baseline/dataset reference.
- **FAST: A Framework to Accelerate Super-Resolution Processing on Compressed Videos** (arXiv:1603.08968v2, 2016-03-29; Zhengdong Zhang, Vivienne Sze) — Uses compressed-video motion vectors/residuals to transfer SR pixels and skip redundant work. Why it matters to OSS: old but clean precedent for motion-vector-driven adaptive compute; useful citation when arguing that MVs should reduce inference, not just improve quality.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 hits, all broad/noisy archived GitHub/star-list pages; no primary paper snapshot found.
- `"frame generation" "deep learning"`: 4 hits: FluidFrames Windows builds, DLSS Swapper Windows builds, MoCha-LD OSF registration, and a Minecraft/Radiance archive. Ecosystem breadcrumbs only.
- `"real-time upscaling" game`: 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16,568 hits but dominated by acronym/common-word noise (`GRAPE`, unrelated media/items). Broad query not useful as evidence.
- Targeted Wayback/Archive follow-up should use exact arXiv/project/GitHub URLs; broad acronym search remains too polluted.

## GitHub / Open Source
- `Sungpyokim97/SR3R` — unofficial/private-style SR3R mirror, pushed 2026-06-13, 1 star. Low signal but tracks feed-forward Gaussian SR interest. https://github.com/Sungpyokim97/SR3R
- `hvcl/3DCT_SR_DDNM_NAB-GS` — zero-shot 3D CT SR with diffusion 2D projection priors and Negative Alpha Blending Gaussian Splatting, pushed 2026-06-12, 1 star. Medical-domain; NAB/confidence blending may be conceptually useful for unreliable detail injection. https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS
- `Adam-AtlasSoftware/RDNU` — Radeon Decoupled Neural Upscaler, pushed 2026-06-09, 16 stars. **Threat/opportunity:** still the most relevant open-source competitor signal: RDNA3-focused, G-buffer-guided neural upscaling, game/virtual-data training claims. Needs seriousness/license/weights check. https://github.com/Adam-AtlasSoftware/RDNU
- `senjul07/optiscaler-client-fsr4` — OptiScaler/FSR4/DLSS/XeSS/frame-generation keyword repo, pushed 2026-06-14, 18 stars. Likely SEO/noisy wrapper; monitor only if it contains real integration code.
- `Streampogrow/lsfg-git` — Lossless Scaling Frame Generation keyword repo, pushed 2026-06-14, 0 stars. Low signal / likely SEO.
- `nova-studios-hw/novagpu-ts1t` — open GPU RTL mentioning frame generation, pushed 2026-06-14, 11 stars. Interesting hardware curiosity, not relevant to OSS software path.
- No new high-signal DLSS/FSR/XeSS analysis-tool repos found in the GitHub REST search window.

## News
- **NVIDIA / DLSS 4.5:** Google News RSS surfaced NVIDIA’s `CONTROL Resonant` launch messaging with DLSS 4.5/path tracing and UE5 plugin/Ray Reconstruction coverage. **Threat:** NVIDIA keeps raising the perceived bar around transformer/RR quality and turnkey plugin integrations.
- **AMD / FSR 4:** RSS surfaced `The Outlast Trials` FSR 4 support discovery and “no definitive decision” around FSR 4.1 support for RDNA 3.5 APUs. **Opportunity/threat:** AMD support matrix uncertainty helps vendor-neutral positioning, but FSR 4 game coverage continues to expand.
- **Intel / XeSS / Arc:** RSS showed Intel Arc G-series/handheld benchmark coverage, WHQL driver update, and low-signal XeSS-vs-DLSS-vs-FSR comparison content. **Threat:** multi-vendor upscaler benchmarking is increasingly consumer-visible; OSS needs comparable metrics/demos, not just architecture prose.
- **Apple / Metal:** RSS surfaced Cyberpunk 2077 on Mac 60 FPS coverage, Apple developer-tool updates, cross-platform Mac gaming commentary, and M5 GPU benchmark pages. **Opportunity:** Metal backend credibility matters more as Mac gaming grows; current OSS cross-vendor kernels remain designed-only.
- **Game engines:** RSS surfaced an Arm Neural Dawn / UE5 demo headline mentioning ray tracing + neural upscaling. **Signal:** neural upscaling is spreading beyond desktop GPUs; mobile/Arm paths may become relevant after CUDA proof-of-concept.

## Action Items / Follow-ups
- [ ] Ops: investigate dashboard trainer — step advanced since yesterday but is now reported offline/stuck at 30,300 for 9h while Watcher/CF/R2 remain healthy.
- [ ] Deep-read `VeloGauss`, `PersistGS`, `R5DGS`, and `Envision4D` for v7 native-time / velocity / occlusion / rigid-vs-transient canvas design.
- [ ] Evaluate `PSNR_DIV` as a cheap OSS-FX temporal metric/loss around motion-divergence/occlusion boundaries.
- [ ] Inspect `Adam-AtlasSoftware/RDNU` for runnable code, weights, license, claimed datasets, and whether it should be a formal baseline/competitor in the README.
- [ ] Re-run Archive.org with targeted exact URLs for GS-STVSR, GSASR, GRAPE, STSS, RDNU, and SR3R; broad acronym search is too noisy.
- [ ] Keep QAT/calibration work front-loaded: `QuantSR+`, `Efficient INT8 SISR`, and clip-free PTQ all support treating deployment precision as part of training, not an export afterthought.

## Bibliography Update
- No `BIBLIOGRAPHY.md` append this run. The materially relevant papers surfaced today are already present from 2026-06-10 to 2026-06-13 additions (`Envision4D`, `PersistGS`, `R5DGS`, `QuantSR+`, `Efficient INT8 SISR`, `VeloGauss`, `ANVIL`, `RDNU`) or are lower-priority indirect leads (`MooMIns`, `SplatlessDF`, `Flex4DHuman`, `LiteVSR`, `PSNR_DIV`).
