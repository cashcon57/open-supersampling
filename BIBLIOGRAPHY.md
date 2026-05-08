# OpenSuperSampling Bibliography

This is a consolidated reading list of the primary references that motivate or directly inform the OSS architecture (v4 baseline, v5-pixel-temporal, v5-gaussian-temporal, and the v6 covariance-resampled online Gaussian-temporal SR design). Entries are grouped by category. Each entry is followed by a one-sentence note explaining how OSS uses or relates to it.

How to read this file:

- "Primary" means the paper directly contributes a component used in our architecture (e.g. HAT backbone, S-T variation score, covariance resampling).
- "Reference" means we use it as a motivating prior, a baseline we compare against, or methodology guidance (training discipline, loss design, dataset choice).
- For deeper math and benchmark numbers behind each Gaussian-side reference, see `docs/research/2026-05-05-gaussian-temporal-research-deep-dive.md`. For design rationale, see `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md` (sections 2-4 and 15) and the v5 design specs under `docs/superpowers/specs/`.

---

## 1. Gaussian Splatting and 4D Extensions

**[3DGS]** Kerbl et al. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. SIGGRAPH 2023. arXiv:2308.04079
> Foundational primitive and tile-based rasterizer underlying the OSS persistent Gaussian canvas; we inherit the EWA splatting formulation and the analytical 2D-projection covariance.

**[4D-GS]** Wu et al. (2024). *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*. CVPR 2024. arXiv:2310.08528
> Canonical-Gaussians-plus-deformation-field formulation that motivates the canvas warp; informs how we keep a persistent set of Gaussians while letting them move per frame.

**[4DGS-1K]** Yuan et al. (2025). *1000+ FPS 4D Gaussian Splatting for Dynamic Scene Rendering*. NeurIPS 2025. arXiv:2503.16422
> Source of the **Spatial-Temporal Variation Score** and **key-frame active mask** OSS uses for 14-34x per-frame splat reduction; this is what makes the canvas tractable at game frame rates.

**[Frosting]** Guédon and Lepetit (2024). *Gaussian Frosting: Editable Complex Radiance Fields with Real-Time Rendering*. ECCV 2024.
> Mesh-aware Gaussian shells; parked for the OSS-FX engine-integration track but informs our long-term thinking about hybrid mesh + splat assets.

**[GS-STVSR]** Shi et al. (2026). *GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting*. arXiv:2604.18047
> Direct precedent for **covariance resampling** as the SR upscaling operator on a Gaussian canvas; v6 adopts this as its core spatial upsampler in place of a pixel-shuffle head. Their optical-flow-guided motion module is the offline analog of OSS's exact-engine-MV pipeline — same math, stronger input signal in our setting.

**[3DGUT]** Wu et al. (2024). *3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting*. NVIDIA. arXiv:2412.12507
> Unscented-Transform replacement for the linearized projection; flagged as the path to wide-FOV, fisheye, and rolling-shutter robustness once OSS moves beyond pinhole game cameras.

**[GRTX]** *GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering*. HPCA 2026.
> Ray-traced splats at 3-5x prior throughput; parked under the OSS-RG track but referenced as the eventual route to reflections/refraction on a Gaussian canvas.

**[4DGC]** *4DGC: Rate-Aware 4D Gaussian Compression*. CVPR 2025. arXiv:2503.18421
> Rate-distortion compression for 4D Gaussian streams; informs canvas serialization and bandwidth budgeting for any future cloud / streaming deployment.

**[HPC]** *HPC: Hierarchical Point-based Latent Representation for Streaming Dynamic Gaussian Splatting Compression* (2026). arXiv:2602.00671
> 67% storage reduction via hierarchical latent codes; reference for canvas compression in long-running game sessions.

**[gsplat]** *gsplat library mathematical supplement* (2023). arXiv:2312.02121
> Open-source CUDA reference implementation we cross-check our custom kernels against; documents the exact gradient formulas for differentiable rasterization.

**[GRAPE]** Jang and Jin (2026). *GRAPE: Gaussian Rendering for Accelerated Pixel Enhancement Brings Fast and Lightweight Arbitrary Super-Resolution*. WACV 2026, pp. 7750-7758.
> **Concrete candidate for the OSS Pico-tier student**: a single point-wise layer predicts anisotropic Gaussian parameters (RGB + rotation + scale + offset) and a differentiable rasterizer renders the HR output in one pass. **1.56M params, ~1.10 GB VRAM, 69.33 FPS at 4× on Urban100, 315× faster than GSASR.** Single-image only — temporal extension is the OSS contribution to make.

**[DSA-SRGS]** Zhang et al. (2026). *DSA-SRGS: Super-Resolution Gaussian Splatting for Dynamic Sparse-View DSA Reconstruction*. arXiv:2603.04770
> Medical (vascular angiography) domain, but two pieces transfer to OSS: (a) **Confidence-Aware Strategy** mixing trusted-but-sparse HR signal with abundant-but-hallucinatory pseudo-labels — directly applicable to our v6.1 INSANE-mode-vs-diffusion-teacher mixing problem; (b) **Radiative Sub-Pixel Densification** — adaptive densification gradient-accumulated from HR sub-pixel sampling, candidate refinement for our densification logic.

**[SR3R]** Feng et al. (2026). *SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting*. CVPR 2026. arXiv:2602.24020
> Independent validation that **feed-forward Gaussian-field prediction across scenes** is viable — the architectural class OSS is betting on for v6. Different problem (multi-view 3D recon vs streaming temporal SR) but their plug-and-play modular pattern is a north-star for keeping OSS pixel and Gaussian modules cleanly separable.

**[Voronoi-HSI]** Zhang et al. (2026). *Voronoi-guided Bilateral 2D Gaussian Splatting for Arbitrary-Scale Hyperspectral Image Super-Resolution*. arXiv:2604.17727
> Adjacent work — hyperspectral (remote-sensing) SR via Voronoi-partitioned 2D-GS with bilateral weighting. Different problem domain than RGB game SR; cited for completeness on the 2D-GS-for-arbitrary-scale-SR literature axis.

**[upscale3dgs]** Niedermayr, Neuhauser, Westermann (2025). *Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images*. ICCV 2025. https://github.com/KeKsBoTer/upscale3dgs
> Uses analytical image-space gradients of Gaussians for **bicubic spline interpolation upscaling** with low overhead; reports 3-4× rendering speedup vs full-res 3DGS, training-time integration improves reconstruction. Explicit temporal-stability claim. **Direct study target for OSS Pico tier** alongside GRAPE.

**[GSASR]** Hu et al. (2025). *GSASR: Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution*. ICCV 2025. https://github.com/ChrisDud0257/GSASR
> Feed-forward predicts millions of image-conditioned 2D Gaussians for arbitrary scales (incl. ×6, ×12 OOD); **supports HAT-L encoders** — direct architectural overlap with v6's HAT backbone choice. Custom CUDA scale-aware 2D rasterizer. Active 2025, demo + weights released. **Primary OSS prior-art baseline for benchmarking v6 against.**

**[GaussianSR-AAAI]** Hu et al. (2025). *High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution*. AAAI 2025. https://github.com/tljxyys/GaussianSR
> Per-pixel continuous Gaussian field for arbitrary-scale SR with a classifier dynamically assigning Gaussian kernels per pixel. Distinct from the SDS-prior 3D "GaussianSR".

**[Sequence-Matters]** Lee et al. (2025). *Sequence-Matters: Harnessing Video Models in Super-Resolution*. AAAI 2025. https://github.com/Ko-Lani/Sequence-Matters
> Adaptive-Length-Sequencing orders LR multi-view images into pseudo-video subsequences, then applies off-the-shelf VSR (no fine-tuning) to drive 3DGS-SR. Avoids the LR-3DGS-render artifacts SuperGaussian struggles with. Useful auxiliary path for OSS data augmentation in v6.1.

**[S2Gaussian]** Wan et al. (2025). *S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting*. CVPR 2025. https://jeasco.github.io/S2Gaussian/
> Two-stage pipeline (LR-GS densify → "Gaussian Shuffle Split" → HR-GS optimize) handling combined sparse-view + LR. Less directly relevant to streaming SR but documents densification/upscaling trade-offs.

**[SplatSuRe]** Asthana et al. (2025). *SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3DGS*. arXiv:2512.02172
> Geometry-aware **selective** SR — applies 2D SR only in undersampled regions. Beats SRGS, GaussianSR, S2Gaussian, etc. on multi-view consistency. Selective-application principle applicable to v6 (apply expensive ops only where needed).

**[SuperGS]** Pancw et al. (2024). *SuperGS: Multi-Resolution Feature Gaussian Splatting with Latent Feature Field*. arXiv:2410.02571
> Multi-resolution Feature Gaussian Splatting + Gradient-guided Selective Splitting (GSS) for upsampling Gaussian primitives during HR optimization. Coarse-to-fine. Beihang group.

**[SuperGaussian]** Adobe Research (2024). *SuperGaussian: Repurposing Video Models for 3D Super Resolution*. ECCV 2024. https://github.com/adobe-research/SuperGaussian
> Uses pretrained video upsampling priors (RealBasicVSR, VideoGigaGAN) to upsample rendered novel views, then refits Gaussians. Modular, accepts NeRF / GS / mesh / RGB-D inputs.

**[AAA-Gaussians]** Thomas et al. (2025). *AAA-Gaussians: Adaptive 3D Smoothing + Frustum-Bounded Anti-Aliasing*. ICCV 2025. https://github.com/DerThomy/AAA-Gaussians
> Adaptive 3D smoothing filter + view-space frustum bounding; **eliminates popping artifacts**; full-3D-evaluated rasterizer (built on StopThePop). Directly applicable to OSS temporal-stability claim — popping is the artifact we most need to avoid in moving game cameras.

**[AA-2DGS]** Younes et al. (2025). *Anti-Aliased 2D Gaussian Splatting*. NeurIPS 2025. https://github.com/maeyounes/AA-2DGS
> World-space flat smoothing kernel + object-space Mip filtering for 2D-GS specifically. **OSS uses 2D Gaussians** so this is direct architectural reading.

**[Analytic-Splatting]** Zhang et al. (2024). *Analytic-Splatting: Anti-Aliased 3DGS via Analytical Pixel-Area Integral*. ECCV 2024 Oral. https://github.com/lzhnb/Analytic-Splatting
> Analytical integration of Gaussian density over pixel area for AA. Mathematical reference for OSS rasterizer-level AA.

**[Mipmap-GS]** *Mipmap-GS: scale-consistency loss for 3DGS*. https://github.com/renaissanceee/Mipmap-GS
> Plug-in scale-consistency loss; +9.25 dB zoom-in / +10.40 dB zoom-out on NeRF-synthetic. Loss recipe candidate for v6 multi-resolution training.

**[MEGA]** Xie et al. (2025). *MEGA: Memory-Efficient 4D Gaussian Splatting*. ICCV 2025. https://github.com/Xinjie-Q/MEGA
> Decomposes color into per-Gaussian DC + shared lightweight AC predictor (drops 144-coef SH); entropy-constrained deformation field. Aggressive memory reduction. Reference for OSS v6 canvas memory budget.

**[4D-Rotor-Gaussians]** Wei et al. (2024). *4D-Rotor Gaussian Splatting*. SIGGRAPH 2024. https://github.com/weify627/4D-Rotor-Gaussians
> Native 4D XYZT Gaussians with **geometric-algebra rotor** rotation representation (rotors generalize quaternions to 4D cleanly). Up to 583 FPS on RTX 4090. PKU. Reference for any v7+ extension to native-4D primitives.

**[3DGStream]** Sun et al. (2024). *3DGStream: On-the-fly 4D Streaming*. CVPR 2024 Highlight. https://github.com/SJoJoK/3DGStream
> Per-frame on-the-fly training in ~12s, 200 FPS rendering. Neural Transformation Cache (tiny-cuda-nn). Reference for online streaming case.

**[GaussianVideo-cyberiada]** Bond et al. (2024). *GaussianVideo: Neural-ODE camera trajectory + 3DGS for video*. https://cyberiada.github.io/GaussianVideo/
> Explicit **frame interpolation at arbitrary timesteps** + arbitrary spatial resampling. **Direct relevance to OSS-FX** — same α-conditioned rendering pattern. 44.21 PSNR @ 960×540, 93 FPS A40.

**[vk_gaussian_splatting]** NVIDIA (2025). *Vulkan Gaussian Splatting Testbed: 3DGRT + 3DGUT + DLSS-RR Integration*. https://github.com/nvpro-samples/vk_gaussian_splatting
> NVIDIA-published Vulkan testbed implementing **3DGRT (3D Gaussian Ray Tracing)**, **3DGUT (Unscented Transform)**, and **DLSS Ray Reconstruction integration for AA + upscaling + denoising of splats**. The closest existing reference for "DLSS-on-splats" — **OSS upper-bound benchmark target** when DLL-shim runtime exists.

**[GSCodec_Studio]** Liu et al. (2025). *GSCodec_Studio: Modular static + dynamic GS compression*. https://github.com/JasonLSC/GSCodec_Studio
> Modular framework on gsplat; integrates MPEG GSC tooling (PCC + video-codec wrappers). Reference experimental harness for v6+ canvas compression.

**[3DGStream-survey]** OSS-curated (2026). *Existing Gaussian-Splatting Repos Survey for OSS*. `docs/research/2026-05-05-existing-gaussian-splatting-repos-survey.md`
> 47 obscure-but-active GS repos categorized by relevance to OSS spatio-temporal SR, anti-aliasing, 4D temporal, engine integration, compression, PBR/RT, and mesh hybrids. Updated 2026-05-05.

---

## 2. Super-Resolution Architectures (Transformers, CNNs)

**[HAT]** Chen et al. (2023). *Activating More Pixels in Image Super-Resolution Transformer*. CVPR 2023.
> The **HAT-Base** spatial backbone adopted by v6 (replacing the v4/v5 SRCNN backbone); chosen for its hybrid window+channel attention which empirically activates more pixels per query than SwinIR-class transformers.

**[Real-ESRGAN]** Wang et al. (2021). *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*. ICCV Workshops 2021.
> Source of OSS's **GAN training discipline** for the perceptual stage: UNet discriminator, second-order degradation pipeline philosophy, and bf16 + hinge + GAN-warmup-at-step-20K recipe.

**[SRGAN]** Ledig et al. (2017). *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*. CVPR 2017.
> Original perceptual + adversarial SR formulation; the philosophical ancestor of every OSS run that uses an LPIPS + GAN term in addition to L1/SSIM.

**[SRCNN]** (v4 baseline backbone)
> The trimmed SRCNN-class CNN used as the v4 production backbone (`srcnn-prod-v4-lpips`) and as the warm-start initializer for v5-pixel-temporal; documented in `docs/superpowers/experiments/2026-05-02-srcnn-beats-v05-and-gsasr.md`.

---

## 3. Temporal Super-Resolution and Frame Extrapolation

**[v5-pixel-temporal]** OSS internal design (2026). `docs/superpowers/specs/2026-05-04-v5-pixel-temporal-design.md`
> Pixel-track temporal SR: v4 backbone + 1-frame history + motion-vector warp + disocclusion-mask blend; the proven control track that v6 must beat to ship.

**[v5-gaussian-temporal]** OSS internal design (2026). `docs/superpowers/specs/2026-05-04-v5-gaussian-temporal-design.md`
> Gaussian-track temporal SR: 2D Gaussian canvas as persistent scene memory, analytical motion-vector warp on Gaussian centers, differentiable rasterization fused with the pixel branch; the convergence baseline that v6 extends.

**[v6-canonical]** OSS internal design (2026). `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md`
> v6 architecture canonical reference: HAT backbone + cross-attention pixel-Gaussian fusion + covariance resampling + S-T variation score pruning + alpha-conditioned canvas rendering for frame extrapolation as a byproduct.

**[v6-tracks]** OSS internal design (2026). `docs/superpowers/specs/2026-05-04-v6-research-tracks-design.md`
> v6 research-track plan: Pico/Standard/Heavy tier definition and teacher-student distillation methodology.

**[OSS-frame-ex]** OSS internal design (2026). `docs/research/2026-04-30-frame-extrapolation-design.md`
> Alpha-conditioned canvas rendering at fractional time positions; how OSS produces extrapolated frames without a separate motion-prediction network.

---

## 4. Real-Time Rendering and Engine Integration

**[OSS-vulkan-eval]** OSS internal report (2026). `docs/research/2026-04-30-vulkan-runtime-eval.md`
> Vendor-runtime evaluation behind the cross-vendor kernel plan (CUDA, HIP, Metal, Level Zero, Vulkan compute).

**[OSS-ue5-data]** OSS internal report (2026). `docs/research/2026-05-01-ue5-training-data.md`
> Unreal Engine 5 capture pipeline for training data with engine motion vectors, depth, and G-buffer channels.

**[OSS-deep-research]** OSS internal report (2026). `docs/research/2026-04-30-deep-research-synthesis.md`
> Synthesis of cross-vendor real-time SR deployment options motivating the DLL-shim drop-in DLSS/FSR/XeSS-compatible integration path.

**[KHR-gsplat]** Khronos (2026). *glTF KHR_gaussian_splatting extension* (ratification Q2 2026).
> Standard interchange format adopted as the OSS canvas serialization target so trained scenes are portable across engines.

**[FSR4-SDK]** AMD (2025). *AMD FidelityFX SDK 2.0.0*. Original 2025-08-18 push at commit `01446e6a74888bf349652fcf2cbf5f642d30c2bf`. <https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK>
> AMD's first public source release of the FSR 4 ML upscaler — HLSL operator runtime (ml2code), FasterNet block, fused conv kernels, FP16 NHWC + INT8 weights, SqrSwish activation. Released under MIT, force-pushed away by AMD ~2 days later but the orphan commit remains in the official repo and the MIT grant remains irrevocable per standard MIT terms. Vendored at `oss/third_party/fidelityfx-sdk-2.0.0-mit/`; full forensic provenance in that directory's `MIT-PROVENANCE.md`. OSS uses this as: (1) measured benchmark target via the FSR 4 binary DLL bundled in the same MIT release; (2) distillation teacher for the v6.2 student model training; (3) architecture reference for designing OSS's own fused-op kernels and student backbone (independent reimplementation, not direct port).

---

## 5. Loss Functions and Training Discipline

**[LPIPS]** Zhang et al. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR 2018.
> The perceptual term in the OSS loss (`w_lpips = 0.1`) and the primary held-out evaluation metric alongside PSNR; non-negotiable because L1/PSNR alone do not correlate with the artifacts we ship to fix.

**[SSIM]** Wang et al. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity*. IEEE TIP.
> Structural-similarity term (`w_ssim = 0.1`) used alongside L1 + LPIPS in the OSS appearance loss.

**[OSS-loss-config]** OSS internal recipe (2026).
> Fixed across v4 and v5: `w_l1 = 1.0`, `w_ssim = 0.1`, `w_lpips = 0.1`, plus a temporal-consistency term in v5; documented in both v5 design specs.

---

## 6. Datasets and Benchmarks

**[TartanAir]** Wang and Liu (2020). *TartanAir: A Dataset to Push the Limits of Visual SLAM*. IROS 2020.
> **Primary OSS training corpus** (Easy split, 18 environments, ~600 GB extracted): sequential trajectories with real engine flow and depth, which is exactly the supervision signal the temporal track requires.

**[Hypersim]** Roberts et al. (2021). *Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding*. ICCV 2021.
> Photorealistic indoor synthetic dataset with full G-buffer ground truth; reference dataset for indoor-scene generalization eval.

**[Sintel]** Butler et al. (2012). *A Naturalistic Open Source Movie for Optical Flow Evaluation*. ECCV 2012.
> Held-out fixed-batch evaluation set for OSS temporal SR (cinematic motion, hard occlusions, varied lighting); manifest at `docs/superpowers/experiments/v5_held_out_manifest_sintel.json`.

**[OSS-capture]** OSS internal design (2026). `docs/superpowers/specs/2026-05-04-oss-capture-tool-design.md`
> The OSS capture installer pipeline (trickle / lite / regular / INSANE modes) for collecting in-game training data with engine motion vectors and (in INSANE mode) supersampled GT.
