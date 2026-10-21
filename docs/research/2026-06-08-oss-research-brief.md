# OSS Research Brief — 2026-06-08

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
- README status snapshot: v7 is current research/teacher architecture: N-D Gaussian canvas with native `(x,y,t)` primitive, 3D Cholesky covariance with `V_xt` motion correlation, parent-child loss-adaptive density, Mip-Splatting AA filters, and alpha=0.5 OSS-FX supervision. Distilled shipping students are still not trained; target is ≤0.4M/≤1M/≤2M params by tier with TensorRT FP8 and cross-vendor kernels.
- Working tree note: repo already had uncommitted dashboard/API/test changes and prior research briefs before this run; this run only updates today's research brief and appends bibliography items.
- Training dashboard: active run `srcnn-v7.0-pico-005` / `v7.0 Pico OSS-FX` is **degraded/stuck at step 20,600 / 100,000 for ~5m**. Watcher, Cloudflare Worker, and R2 origin are healthy; GPU stats unavailable (`host offline or rsync failed`). No held-out score rows yet for v7. This is an improvement from the earlier 17,700-step stall, but still needs host/status follow-up if it remains stuck.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting** (arXiv:2606.07670, 2026-06-04; Mingzhao Li, Arghya Pal, Guan Yuan Tan) — Replaces the usual per-frame MLP deformation field with closed-form continuous-time liquid/CfC cells that explicitly smooth deformation as a function of time. Why it matters to OSS: direct v7-adjacent motion prior; OSS has exact engine motion vectors, but a tiny continuous-time residual field could stabilize gaps/disocclusions without a separate FG network.
- **EvoGS: Constructing Continuous-Layered Gaussian Splatting with Evolution Tree for Scalable 3D Streaming** (arXiv:2606.07179, 2026-06-05; Yuang Shi, Simone Gasparini, Géraldine Morin, Wei Tsang Ooi) — Organizes Gaussians in a wavelet-inspired parent-child evolution tree, reducing splat redundancy and VRAM/payload for progressive quality. Why it matters to OSS: overlaps strongly with v7 parent-child loss-adaptive density; worth mining for child correction/error propagation and canvas budget control.
- **SparseStreet: Sparse Gaussian Splatting for Real-Time Street Scene Simulation** (arXiv:2606.03909, 2026-06-02; Qingpo Wuwu, Xiaobao Wei, Peng Chen, Nan Huang, Zhongyu Zhao, et al.) — Uses node-based learnable pruning plus background compression to remove redundant static Gaussians while preserving dynamic objects. Why it matters to OSS: reinforces selective persistence/pruning for static vs dynamic regions in long-running game sequences.
- **Beyond Static Gaussians: An Empirical Investigation of Architectural Paradigms for Dynamic 3D Scene Reconstruction** (arXiv:2606.00452, 2026-05-30; Adrian Ramlal, John S. Zelek) — Compares structure-guided deformation/canonical methods against Gaussian-centric 4D methods and finds a quality/compactness vs rendering-speed trade-off. Why it matters to OSS: useful framing for v7's native time-axis bet; if Gaussian-centric is faster but more storage-heavy, OSS must prove packed canvas layout (#12) early.
- **LEGS: Laplacian-Enhanced Gaussian Splatting with a Nonlinear Weighted Loss** (arXiv:2606.07932, 2026-06-06; Yongfei Guo, Qizhou Huo, Xuan Sun, Yuanhao Gong) — Adds second-order Laplacian structure weighting to GS photometric loss and reports PSNR gains with no pipeline change. Why it matters to OSS: cheap loss-level ablation for edge/detail recovery in Gaussian raster training.
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047, 2026-04-20; Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, et al.) — Uses 2D-GS for continuous spatial+temporal VSR with optical-flow-guided Gaussian motion and covariance resampling alignment. Why it matters to OSS: still the closest published collision/opportunity for OSS's unified SR+FG claim; OSS differentiates by using exact engine motion vectors and a persistent game canvas.

### Super-Resolution / Upscaling
- **3DGS$^3$: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** (arXiv:2605.11489, 2026-05-12; Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu) — Unified post-rendering framework for 3DGS: gradient-aware super sampling plus lightweight temporal frame interpolation. Why it matters to OSS: direct threat/benchmark for the “super-sampling + frame interpolation together” story, though it is a post-rendering 3DGS renderer add-on rather than a vendor-neutral game upscaler canvas.
- **SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images** (arXiv:2512.20377, 2025-12-23; Linfei Li, Lin Zhang, Zhong Wang, Ying Shen) — Uses gradient/color-aware sampling and scale-adaptive initialization to compress ultra-high-res images with 2D/3D Gaussian image representations. Why it matters to OSS: candidate ideas for canvas initialization/compression and VRAM budgeting, especially if v7 Gaussian count grows.
- **Forget Superresolution, Sample Adaptively (when Path Tracing)** (arXiv:2602.08642, 2026-02-09; Martin Bálint, Corentin Salaün, Hans-Peter Seidel, Karol Myszkowski) — Argues adaptive sampling can beat uniform low-res + SR under sub-1-spp path tracing, using perceptual/tonemapping-aware training. Why it matters to OSS: not a replacement for OSS, but a reminder that future game engines may expose adaptive sample placement; OSS should treat renderer-side sample metadata as a possible input.
- **SlimDiffSR: Toward Lightweight and Efficient Remote Sensing Image Super-Resolution via Diffusion Model Distillation** (arXiv:2605.02198, 2026-05-04; Ce Wang, Zhenyu Hu, Wanjie Sun) — Distills diffusion SR using uncertainty-guided timestep assignment, structured pruning, frequency-separable convs, and MMD alignment. Why it matters to OSS: useful teacher→student compression recipe for HAT/Gaussian distillation even though the source domain differs.
- **Distillation-Supervised Convolutional Low-Rank Adaptation for Efficient Image Super-Resolution** (arXiv:2504.11271, 2025-04-15; Xinning Chai, Yao Zhang, Yuxuan Zhang, Zhengxue Cheng, et al.) — Adds ConvLoRA and spatial feature affinity distillation to efficient SR without extra inference complexity. Why it matters to OSS: directly relevant to issue #9 / student backbone ablations.

### Frame Extrapolation / Generation
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586, 2026-05-11; Nengbo Lu, Minghua Pan) — Learns velocity/material factors and differentiable physics simulation for long-term future-frame extrapolation. Why it matters to OSS: too heavy for runtime, but useful as a “Gaussian state + future frame” reference and possible synthetic-teacher prior.
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567, 2026-05-11; Nengbo Lu, Bin Zhao) — Learns per-Gaussian velocity fields with global physical constraints for novel-view interpolation and future-frame extrapolation. Why it matters to OSS: conceptual overlap with v7's time-axis and `V_xt` motion correlation; compare learned velocities against engine-MV-supervised motion.
- **R5DGS: Semantic-Aware 4D Gaussian Splatting with Rigid Body Constraints for Efficient Dynamic Scene Reconstruction** (arXiv:2605.25909, 2026-05-25; Denis Gridusov, Maxim Popov, Sergey Kolyubin) — Adds object association and rigid-body constraints to speed extrapolation by propagating centroid motion to associated Gaussians. Why it matters to OSS: possible high-level analogy for material/object-ID conditioned canvas motion where G-buffer IDs are available.
- **ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering** (arXiv:2506.13814, 2025-06-14; Lufei Liu, Tor M. Aamodt) — Caches intermediate neural features across temporally coherent frames and reports ~1.4× speedup with negligible quality loss. Why it matters to OSS: strong candidate optimization for teacher/student temporal inference if feature drift is bounded.
- **RSTT: Real-time Spatial Temporal Transformer for Space-Time Video Super-Resolution** (arXiv:2203.14186, 2022-03-27; Zhicheng Geng, Luming Liang, Tianyu Ding, Ilya Zharkov) — Older single-transformer baseline for spatial+temporal SR. Why it matters to OSS: literature framing for “one model covers SR + temporal interpolation,” but not as close as GS-STVSR/3DGS$^3$.

### Inference Optimization
- **Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution** (arXiv:2308.02794, 2023-08-05; Yong Liu, Hang Dong, Boyang Liang, Songwei Liu, et al.) — Designs SR transformer blocks to avoid TensorRT-hostile operators and includes operator substitutions/fusion. Why it matters to OSS: relevant to HAT-teacher export and any transformer-ish student variant.
- **NAWQ-SR: A Hybrid-Precision NPU Engine for Efficient On-Device Super-Resolution** (arXiv:2212.09501, 2022-12-15; Stylianos I. Venieris, Mario Almeida, Royson Lee, Nicholas D. Lane) — Hybrid precision selection beats uniform INT8 in constrained SR hardware. Why it matters to OSS: caution that uniform FP8/INT8 may need exceptions around sensitive layers/raster-composite boundaries.
- **Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers** (arXiv:2604.02344, 2026-02-09; Jędrzej Maczan) — Finds dispatch overhead/fusion critical across vendor/browser backends. Why it matters to OSS: not SR, but directly relevant to Vulkan/Metal/WebGPU backend planning: fuse tiny kernels and minimize launches.

### Game Engine / Motion Vectors
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694, 2026-03-08; Yuhang Wang, Hai Li, Shujuan Hou, Zhetao Dong, et al.) — Uses supplied motion vectors, residuals, and frame types to avoid optical flow and allocate compute adaptively. Why it matters to OSS: strong external validation of exact-engine-motion-vector compute reuse/gating.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483, 2023-08-03; Antoine Mercier, Ruan Erasmus, Yashesh Savani, Manik Dhingra, et al.) — Game-rendering neural supersampling dataset/baseline with motion vectors, depth, jitter, and mip bias. Why it matters to OSS: still one of the most directly game-specific public baselines.
- **FAST: A Framework to Accelerate Super-Resolution Processing on Compressed Videos** (arXiv:1603.08968, v2 2026; Zhengdong Zhang, Vivienne Sze) — Uses codec motion vectors/residuals to reuse SR results and skip redundant frames. Why it matters to OSS: old but reinforces the same persistent-canvas/motion-vector reuse thesis.

## Archive.org Findings
- `site:archive.org "gaussian splatting" "super-resolution"` did not surface preserved GS-STVSR/GSASR/GRAPE/STSS paper mirrors; results were mostly noisy GitHub/awesome-list or unrelated records.
- `site:archive.org "frame generation" "deep learning"` surfaced archived software/articles, not new primary research relevant to OSS.
- `site:archive.org "real-time upscaling" game` returned no useful records this run.
- Targeted `GS-STVSR OR GSASR OR GRAPE OR STSS` on Archive.org was dominated by false positives for “grape”; no reliable archived paper copies found. Continue using arXiv/GitHub/project pages as primary sources and consider manually saving PDFs to project-local references if needed.

## GitHub / Open Source
- **optiscaler/OptiScaler** — GPL-3 upscaler/frame-gen bridge for games that already expose DLSS2+/FSR2+/XeSS inputs; ~8.7k stars, `v0.9.2`/`v0.9.2a` in May 2026, latest visible commit Jun 8. It supports upscaler replacement plus experimental DX12 OptiFG, FSR3/FSR4/XeFG paths, Fakenvapi, latency hooks, ASI loading, and Nukem's DLSSG-to-FSR3 mod. Threat/opportunity: this is the strongest open-source integration precedent for OSS's DLL-shim path, but it is middleware around vendor binaries rather than a new vendor-neutral model.
- **ChrisDud0257/GSASR** — Official ICCV 2025 2D-GS arbitrary-scale SR code, Apache-2.0; HAT-L/FlashAttention/AMP “Ultra Performance” variant released. Direct benchmark for OSS's Gaussian raster quality/cost, but single-image rather than persistent temporal game canvas.
- **tljxyys/GaussianSR** — AAAI 2025 2D-GS arbitrary-scale SR code, BSD-3-Clause; ~107 stars. Good baseline for continuous-Gaussian pixel/feature fields and adaptive Gaussian kernels.
- **KeKsBoTer/upscale3dgs** — ICCV 2025 lightweight gradient-aware 3DGS image upscaling; viewer linked, training code still marked “coming soon.” Opportunity: analytical Gaussian gradients for cheap upscaling may transfer to OSS raster/composite or teacher loss.
- **lif314/SmartSplat** — AAAI 2026 MIT code for Gaussian image compression with DIV8K/DIV16K benchmarks. Opportunity: canvas compression/init prior for VRAM-heavy v7 cases.
- **Nukem9/dlssg-to-fsr3 / DLSS Swapper / DLSS tooling ecosystem** — still relevant to practical shim, swapper, and comparison harness positioning; OSS should eventually ship a repeatable bake-off harness, not only model code.

## News
- **NVIDIA DLSS 4.5 / UE plugin** (NVIDIA Technical Blog, 2026-05-27): UE plugin adds Dynamic Multi Frame Generation, 6× MFG mode, and a second-generation transformer Super Resolution model via Streamline. Threat: NVIDIA is pushing both quality and FG multiplier hard; OSS's competitive claim must be “one open architecture + cross-vendor transparency,” not merely “has FG.”
- **AMD FSR SDK 2.2 / Redstone** (GPUOpen): SDK includes FSR Upscaling 4.1.0, FSR Frame Generation 4.0.0, Ray Regeneration 1.1.0, and Radiance Caching preview, with signed DLLs and driver-updatable ML effects. Opportunity/threat: AMD's shift to ML + signed binary distribution makes open OSS more differentiated, but also raises quality expectations and complicates modding-style replacement.
- **AMD FSR 4 GPUOpen release**: FSR 4 emphasizes temporal stability, detail preservation, reduced ghosting, disocclusion handling, and particle preservation without reactive/transparency masks. Opportunity: OSS should explicitly test particles/disocclusions/missing-motion-vector cases, since AMD is using those as visible quality claims.
- **Intel XeSS / Arc**: search results show June 2026 Arc G-Series handheld positioning and XeSS game support pages; third-party TechPowerUp result about XeSS 3 on Arc B580 could not be extracted due challenge page, so treat it as unverified this run. Watch XeSS 3/MFG rollout as another vendor MFG benchmark.
- **Apple Metal 4 / MetalFX**: Apple WWDC25 material highlights MetalFX upscaling, frame interpolation, denoising, and Metal 4 quantized tensor support. Opportunity: Metal backend will matter for credibility on Mac; keep raster/composite design free of CUDA-only assumptions.
- **Unreal / Unity ecosystem**: vendor plugins remain the integration norm; Unity discussions show ongoing demand for current DLSS/FSR/FG support. OSS should prioritize motion/depth/velocity validation tooling before promising drop-in engine support.

## Action Items / Follow-ups
- [ ] If dashboard remains stuck at step 20,600, inspect trainer host / rsync / metrics publishing path; publishing infra is healthy but GPU stats are unavailable.
- [ ] Deep-read **Liquid D-3DGS** for whether continuous-time residual deformation can improve v7 t-axis interpolation/extrapolation without a second network.
- [ ] Deep-read **EvoGS** and compare its evolution tree against v7 parent-child loss-adaptive density; extract any VRAM/redundancy metrics relevant to #12.
- [ ] Deep-read **3DGS$^3$** as the closest “joint supersampling + frame interpolation” competitive paper after GS-STVSR; decide whether it deserves a dedicated OSS-vs-3DGS3 memo.
- [ ] Try a loss-only ablation inspired by **LEGS**: Laplacian/second-order edge weighting in Gaussian/raster losses.
- [ ] Add an explicit particle/disocclusion/missing-MV benchmark slice, because FSR 4 marketing now calls those out.
- [ ] Keep watching OptiScaler and swapper tooling; their compatibility lists and failure modes are practical guidance for OSS shim design.
