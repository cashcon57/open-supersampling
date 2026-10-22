# OSS Research Brief — 2026-06-03

## Project Status Summary
- Latest commits:
  - `a97a180` feat: add tailnet-gated training controls to dashboard
  - `fcf8084` feat: tailnet-gated training control API + dashboard panel
  - `0660f55` wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc
  - `56a9a65` trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)
  - `d1255c8` trainer: WindowsPath shim on Linux for cross-OS checkpoint resume
  - `e1ab7fe` hat.py: use flash-attn 2 directly when available, fallback to SDPA
  - `e9b4b65` trainer: TF32 conv + DataLoader prefetch
  - `26cd9da` dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB
  - `83b4f07` v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast
  - `baf4b02` worker: apex .com → .org 301 redirect
- Open issues: 12 returned by GitHub API. Top items: #14 VRAM footprint after v6.2 wiring/checkpoint; #13 Nsight L2 hit-rate on 3080 Ti; #12 packed low-rank canvas layout before 32B/Gaussian claim; #10 decoupled G-buffer encoder ablation; #9 RepVGG/ECBSR/RepSR student ablation; #8 cross-vendor kernel ports; #7 FSR4 binary teacher-signal distillation; #4 HAT-Tiny student distillation.
- README snapshot: still presents v7/v6 lineage as a persistent 2D Gaussian canvas, SR + OSS-FX extrapolation via one architecture, HAT-Tiny teacher → ≤1M-param student target, with end-user inference model not yet trained.
- Training dashboard: **trainer offline/stuck** at `srcnn-v7.0-pico-005` step **9,450 / 100,000** for ~3h, loss **0.11428**, SR Charbonnier **0.01515**. Watcher/CF Worker/R2 are healthy. No held-out score rows and no GPU memory samples yet. This is the main operational red flag.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **HIGH RELEVANCE — RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting** (arXiv:2603.13783, CVPR 2026; previously tracked) — Explicitly models Gaussian temporal behavior to reduce temporal aliasing/ghosting at arbitrary timestamps using optical-flow guidance and triple-rendering supervision. **Why it matters to OSS:** directly matches OSS-FX's `α<1` ghosting/future-frame rendering risk; triple-rendering supervision is worth testing in v7 training.
- **Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution / GSASR** (arXiv:2501.06838, ICCV 2025) — Feed-forward model predicts image-conditioned 2D Gaussians from LR input and renders arbitrary-scale outputs with CUDA scale-aware rasterization. **Why it matters to OSS:** validates 2D Gaussian rasterization as an ASR primitive; code exists and should be benchmarked against OSS-Pico constraints.
- **Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling / ContinuousSR** (arXiv:2503.06617, ICLR 2026 poster; previously tracked in bibliography without arXiv ID) — Builds a 2D Gaussian field in one pass and claims ~1ms rendering per scale. **Why it matters to OSS:** strongest external evidence that a Gaussian field can hit OSS's <2ms-class rendering budget after field construction.
- **Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation** (arXiv:2505.13215; previously tracked) — Converts temporally invariant regions from 4D to cheaper 3D Gaussians while retaining 4D only for dynamic content. **Why it matters to OSS:** suggests a static/dynamic split to reduce v7 N-D canvas memory and bandwidth.

### Super-Resolution / Upscaling
- **Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content** (arXiv:2602.11339) — Introduces StreamSR, benchmarks 11 real-time SR models on compressed streaming content, and proposes EfRLFN with Efficient Channel Attention + tanh activation. **Why it matters to OSS:** dataset/benchmark is not game-native, but compressed-frame robustness is relevant to noisy TAA/game capture inputs and student architecture baselines.
- **One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution** (arXiv:2506.15591, NeurIPS 2025) — Uses Dual LoRA Learning with consistency/detail branches and cross-frame retrieval to get one-step SD-based Real-VSR. **Why it matters to OSS:** not real-time enough as-is, but the separated consistency/detail training schedule maps cleanly to OSS teacher→student distillation losses.
- **DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution** (arXiv:2603.22271, CVPR 2026) — Three-stage one-step VSR distillation combining distribution matching and adversarial Real-Fake Score Feature GAN supervision. **Why it matters to OSS:** useful distillation recipe for moving from high-quality HAT/Gaussian teachers into a tiny student without losing perceptual detail.

### Frame Extrapolation / Generation
- **Taming Real-World Space-Time Video Super-Resolution with One-Step Diffusion / OSDEnhancer** (arXiv:2601.20308; v2 revised 2026-05-19) — One-step diffusion STVSR for simultaneous spatial and temporal upsampling, using temporal-coherence and texture-enrichment LoRAs plus a bidirectional deformable VAE decoder. **Why it matters to OSS:** direct STVSR competitor in task scope, but compute profile is likely far heavier; its TC/TE factorization is a training-loss design opportunity.
- **AceVFI: A Comprehensive Survey of Advances in Video Frame Interpolation** (arXiv:2506.01061, IEEE TCSVT 2026) — Survey of 250+ VFI papers with taxonomy across flow/kernel/transformer/Mamba/diffusion and CTFI vs ATFI. **Why it matters to OSS:** good reference map for evaluating OSS-FX against interpolation baselines and failure modes: large motion, occlusion, lighting variation, nonlinear motion.
- **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (arXiv:2406.18551; previously tracked) — Efficient neural + heuristic future-frame extrapolation without G-buffer inputs, with disocclusion handling and shading correction. **Why it matters to OSS:** strongest low-latency extrapolation baseline; reinforces that low-integration-cost FG is valuable, but also highlights disocclusion filling as the core problem.

### Inference Optimization
- No major new paper-level finding from this run. Search surfaced mostly general TensorRT/INT8 references and known cautions that INT8/FP16 mixed kernels can underperform if TensorRT chooses poor kernels. **Why it matters to OSS:** keep deployment-aware quantization in the student training plan; verify TensorRT engine timing rather than assuming lower precision is faster.

## Archive.org Findings
- Archive.org advanced search returned no substantive archival discoveries for `gaussian splatting super-resolution`, `frame generation deep learning`, `real-time upscaling game`, `GS-STVSR`, `GRAPE`, or `STSS`.
- `GSASR` returned an unrelated CIA Reading Room false positive. No evidence of taken-down or materially different archived copies found this run.

## GitHub / Open Source
- **ChrisDud0257/GSASR** — Official ICCV 2025 GSASR PyTorch code, ~98 stars, updated 2026-05-28. **Opportunity:** pull implementation details for scale-aware CUDA rasterization and compare with v7 rasterizer assumptions.
- **pranav-asthana/SplatSuRe** — Selective super-resolution for multi-view-consistent 3DGS, ~74 stars, updated 2026-05-30. **Relevance:** less direct for game SR, but selective SR could inspire targeted canvas refinement rather than full-frame updates.
- **OptiScaler** — GPL-3.0 middleware/mod tool, ~8.6k stars, latest release `v0.9.2a` / `v0.9.2` on 2026-05-17. Supports DLSS2+/XeSS/FSR2+ inputs, replaces native upscalers, enables FSR3/FSR4/XeFG-style paths across DX11/DX12/Vulkan depending on backend. **Threat/opportunity:** proves demand for cross-vendor upscaler swapping; OSS needs a DLL-shim/mod story early if it wants real-world adoption before native game integration.
- GitHub REST search for `frame generation open source neural upscaler`, `neural upscaler game super resolution`, and `DLSS FSR analysis upscaler` did not return useful new repositories beyond OptiScaler-style ecosystem results.

## News

### NVIDIA / DLSS
- **DLSS 4.5 Dynamic Multi Frame Generation + 6X MFG is available via NVIDIA App** (published 2026-03-31; still the dominant current story) — RTX 50 Series can dynamically adjust generated-frame multiplier to target refresh rate, with up to 5 generated frames per rendered frame. DLSS 4.5 Super Resolution uses 2nd-gen transformer controls and NVIDIA notes FP8-native support advantages on newer GPUs. **Why it matters:** NVIDIA is moving the competitive target from just quality to adaptive frame pacing/responsiveness; OSS-FX should track not just PSNR/LPIPS but latency, pacing, and generated-frame multiplier policy.

### AMD / FSR
- **AMD FSR SDK 2.2 / Redstone** — GPUOpen lists FSR Upscaling **4.1.0**, FSR Ray Regeneration **1.1.0**, FSR Frame Generation **4.0.0**, and Radiance Caching **0.9.0** technical preview. DX12 + UE5 integration, signed DLLs, driver-updatable ML tech, FSR4 requiring RX 9000/RDNA4+ with fallback to FSR 3.1.5 on unsupported hardware. **Why it matters:** AMD is converging on a vendor DLL/API update model similar to DLSS; OSS's open positioning must compete on transparency and cross-vendor support, not merely “AMD is closed”.
- Search results also suggest chatter around older RDNA support and possible open-sourcing pressure after accidental release, but primary extractable source content was weak; treat as watchlist, not confirmed architecture fact.

### Intel / XeSS
- **Intel XeSS 3 / XeSS-MFG positioning** — Intel's Arc gaming technologies page now presents XeSS-SR, XeSS-FG, XeSS-MFG, and XeLL; XeSS-MFG inserts up to 3 generated frames between rendered frames. **Why it matters:** Intel has joined NVIDIA/AMD in multi-frame-generation positioning; OSS's “one architecture for SR+FG” pitch remains differentiated, but the market now expects MFG + low-latency controls.

### Game Engines / Integration
- AMD's FSR plugin path emphasizes Unreal Engine 5 and signed DLL integration. No major new UE/Unity-native upscaling feature surfaced in this run beyond vendor plugin updates and common DLSS/FSR/XeSS integration discussions.

## Action Items / Follow-ups
- [ ] CRITICAL: investigate dashboard trainer offline/stuck at step 9,450; watcher and R2 are healthy, so this likely sits on the 3080 Ti trainer host or rsync/GPU path.
- [ ] Deep-read GSASR implementation and note how its CUDA scale-aware rasterizer differs from v7's fp32 smoothing filters and covariance path.
- [ ] Deep-read DUO-VSR and DLoRAL/OSDEnhancer for practical distillation-loss ideas: dual-stream DMD+GAN, consistency/detail LoRAs, preference refinement.
- [ ] Add frame pacing / latency / generated-frame multiplier policy to OSS-FX evaluation notes; DLSS 4.5 and XeSS-MFG are competing on smoothness policy, not just reconstruction quality.
- [ ] Benchmark or at least inspect OptiScaler integration points; OSS needs a credible shim/mod deployment path before native engine integration exists.
