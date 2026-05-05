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
