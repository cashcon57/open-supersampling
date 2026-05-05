# Real-Time Gaussian Temporal Upscaling: Technical Deep Dive (2026)

This document expands the original briefing with the actual math, algorithms, and benchmark numbers behind each technique. It assumes familiarity with vanilla 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023).

---

## 0. Math foundations: vanilla 3DGS (the baseline you're upscaling from)

Every 4D / temporal method below is a modification of this pipeline, so the formulas are worth pinning down.

**Primitive.** A single Gaussian is parameterized by mean $\mu \in \mathbb{R}^3$, covariance $\Sigma \in \mathbb{R}^{3\times3}$, opacity $o \in [0,1]$, and view-dependent color $c$ encoded as spherical harmonics. The 3D density at point $x$ is:

$$G(x) = \exp\!\left(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right)$$

**PSD reparameterization.** Because $\Sigma$ must stay symmetric positive-semidefinite during gradient descent, it's stored as a scale vector $s \in \mathbb{R}^3$ and rotation quaternion $q \in \mathbb{R}^4$:

$$\Sigma = R\, S S^\top R^\top, \quad S = \mathrm{diag}(s)$$

This gives 11 trainable scalars per Gaussian for geometry/opacity, plus 48 for SH degree 3 RGB (3 bands × 16 coeffs). **Total: ~59 floats = 236 bytes per Gaussian uncompressed.** A 1M-Gaussian scene is therefore ~236 MB before any compression.

**Projection (the EWA approximation).** The 3D covariance projects to a 2D screen-space covariance via the local affine approximation around each Gaussian's center. With world-to-camera matrix $W$ and the Jacobian of perspective projection:

$$J = \begin{bmatrix} f_x/z & 0 & -f_x x / z^2 \\ 0 & f_y/z & -f_y y / z^2 \\ 0 & 0 & 0 \end{bmatrix}, \qquad \Sigma' = J W \Sigma W^\top J^\top$$

Drop the third row/column to get the 2×2 image-plane covariance. This linearization is the source of distortion artifacts under wide-FOV / fisheye / rolling-shutter cameras — which is exactly why 3DGUT (NVIDIA, CVPR 2025) replaced it with the Unscented Transform.

**Tile-based rasterization.** The screen is partitioned into 16×16 tiles. For each Gaussian, a screen-space bounding box (typically 3σ radius from $\Sigma'$) determines tile membership. Gaussians are sorted by depth per tile, then alpha-composited front-to-back per pixel:

$$C(p) = \sum_{i \in \mathcal{N}} c_i\, \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j), \qquad \alpha_i = o_i \cdot G_i^{2D}(p)$$

Critical performance fact: **the GPU radix sort over all visible Gaussians is the dominant cost** — for 1M Gaussians on an RTX 3070 it takes 1–4 ms, with another 2–5 ms for actual rasterization. This is why most "make it faster" papers attack the count of Gaussians rather than the per-Gaussian cost.

---

## 1. The architectural fork: deformation fields vs. native 4D primitives

"4D Gaussian Splatting" hides a real architectural split. Knowing which branch a paper sits on tells you most of its tradeoffs.

### Branch A — Canonical Gaussians + deformation field (Wu et al., CVPR 2024)

Maintain *one* canonical set of 3D Gaussians. A deformation network $\mathcal{F}_\theta$ predicts per-timestamp deltas:

$$(\Delta\mu, \Delta s, \Delta q)_t = \mathcal{F}_\theta\big(\mathrm{enc}(\mu, t)\big)$$

The encoder is a HexPlane-style decomposition: instead of a dense 4D voxel grid (which would be $O(N^4)$ memory), space-time is factored into **6 orthogonal 2D planes** — $(xy, xz, yz, xt, yt, zt)$. A query at $(\mu, t)$ bilinearly samples each plane and the 6 features are combined (typically via Hadamard product + concat) into a 32–64-dim feature, then fed through a tiny 2-layer MLP (~64 hidden units) to produce the deformations.

**Why this wins on storage:** for an $N$-second sequence at 30 fps, naïve per-frame 3DGS stores $30N \times M$ Gaussians; this stores $M$ canonical Gaussians plus the planes (typically 50–150 MB regardless of sequence length).

**Reported numbers:** 82 FPS at 800×800 on RTX 3090, training in ~30 minutes for short clips. PSNR within 0.1–0.3 dB of dense per-frame baselines on D-NeRF / Plenoptic Video.

**Where it breaks:** topological change. If a Gaussian needs to *appear* (door opens, occlusion reveals new content), the deformation field has to learn an effectively discontinuous function — typically resolved by Gaussians smoothly fading opacity from 0 to 1, which produces ghosting.

### Branch B — Native 4D primitives (4D-Rotor GS, Spacetime GS)

Treat time as a fourth spatial dimension. Each primitive is a 4D Gaussian with a 4×4 covariance, factored using **rotors from geometric algebra** (4-rotor = 8 scalars vs. a quaternion's 4) which generalize quaternions to 4D rotations cleanly. At render time, condition on $t$ to "slice" each 4D Gaussian into a 3D Gaussian:

$$\Sigma_{3D}(t) = \Sigma_{xx} - \Sigma_{xt}\,\Sigma_{tt}^{-1}\,\Sigma_{tx}$$

(the conditional covariance from marginalizing $t$). Each Gaussian effectively has a temporal "lifespan" $\Sigma_{tt}$ and the contribution at frame $t$ falls off as $\exp(-(t-\mu_t)^2 / 2\Sigma_{tt})$.

**Tradeoff:** handles topology change naturally (Gaussians with short $\Sigma_{tt}$ are explicitly transient), but requires more primitives total — typically 1.5–3× more for equal quality.

---

## 2. Hitting 1000+ FPS: 4DGS-1K's pruning algorithm (NeurIPS 2025)

This is the most concrete temporal-redundancy result and it's worth understanding the algorithm in detail because the same idea generalizes.

The authors observe two redundancies in vanilla 4DGS:

- **Q1 (Short-Lifespan Gaussians):** Many Gaussians have tiny $\Sigma_{tt}$ — they exist for ~1–2 frames. These bloat storage.
- **Q2 (Inactive Gaussians):** At any frame $t$, only ~5–15% of Gaussians have non-negligible contribution, but the rasterizer processes all of them.

**Spatial-Temporal Variation Score (the pruning criterion).** For each Gaussian $i$:

$$\mathcal{S}_i = \mathrm{SS}_i \cdot \mathrm{TS}_i$$

where the **Spatial Score** $\mathrm{SS}_i$ aggregates the Gaussian's contribution to every pixel across every training image and timestamp — essentially the sum of its $\alpha_i \cdot T_i$ (alpha × transmittance) across the entire training set. The **Temporal Score** $\mathrm{TS}_i$ is proportional to $\Sigma_{tt,i}$ — directly penalizing short-lived Gaussians.

Globally rank all Gaussians by $\mathcal{S}_i$, prune the bottom X% (typically 60–80%), then fine-tune for ~30 minutes.

**Key-frame Temporal Filter (handles Q2).** Define keyframes every $K$ frames (typically $K=10$). For each keyframe, precompute a binary mask of which Gaussians have non-negligible contribution. Frames between keyframes share the mask of their nearest keyframe. This works because of the empirical observation that **active-Gaussian sets have ~85–95% overlap between adjacent frames**, so per-frame masks are wasteful.

**Numbers.** On the Plenoptic Video dataset: 1029 FPS rendering (vs. ~30 FPS for vanilla 4DGS — a ~34× speedup), storage reduced from ~2.1 GB to ~150 MB (~14× reduction), with PSNR drop of only 0.1–0.3 dB.

The general pattern — *score-based global pruning + temporal-coherent active masking* — is now standard across follow-on work (Light4GS, MEGA, Mini-Splatting variants).

---

## 3. Game-ready hybrids: the actual mesh extraction math

### Gaussian Frosting (Guédon & Lepetit, ECCV 2024)

The pipeline has three stages with distinct costs:

**Stage 1 — SuGaR mesh extraction.** Add a regularization term during training that pushes Gaussians toward becoming flat and surface-aligned:

$$\mathcal{L}_{\mathrm{reg}} = \lambda_1\, |\bar{d} - d_{\mathrm{GS}}| + \lambda_2\, (1 - |\mathbf{n} \cdot \mathbf{n}_{\mathrm{GS}}|)$$

where $\bar{d}$ is the depth from a signed distance approximation and $d_{\mathrm{GS}}$ is the rendered depth. After convergence, run Poisson surface reconstruction on the Gaussian centers to produce a triangle mesh. Typically yields 200K–800K triangles for a room-scale scene.

**Stage 2 — Frosting layer.** Define a thin shell of adaptive thickness $h(\mathbf{x})$ around the mesh. The thickness is *learned* per-vertex — thin (millimeters) over hard surfaces like floors, thick (centimeters) over fuzzy regions like hair or fabric. Gaussians are constrained to live inside this shell with barycentric coordinates $(\beta_1, \beta_2, \beta_3, \beta_4)$ where the 4th coordinate is the normal offset within $[-h, h]$.

**Stage 3 — Animation.** Skinning the mesh (standard linear blend skinning or dual quaternions) automatically transports the Gaussians: each Gaussian's pose follows its parent triangle. This is the breakthrough — **the Gaussians inherit rigging for free**.

**Numbers:** 100–300K Gaussians per object (vs. 1–3M for unconstrained), 60+ FPS at 1080p on RTX 3060. PSNR within 0.5 dB of unconstrained 3DGS for surface-heavy scenes; degrades more for highly volumetric content (smoke, foliage).

### Voxelization for collision

The simplest production recipe: voxelize the Gaussian centers at game-relevant resolution (e.g., 5 cm voxels for FPS games), keep only voxels where Σ opacity-weighted Gaussian density exceeds threshold $\tau$ (typically 0.5):

$$V(\mathbf{c}) = \mathbf{1}\!\left[\sum_{i: \mu_i \in \mathrm{cell}(\mathbf{c})} o_i \cdot |\Sigma_i|^{-1/2} > \tau\right]$$

This voxel grid is then meshed (marching cubes) or used directly as a collision shape. Decoupling visual from collision geometry like this is now the dominant production pattern.

---

## 4. Hardware: GRTX and the ray-tracing shift

The HPCA 2026 paper "GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering" addresses why ray-traced Gaussians (needed for proper reflections, refraction, secondary rays) have historically been 5–20× slower than rasterization.

**The problem.** Anisotropic Gaussians have wildly different scales along their three axes (often 10:1:1 or worse). Building a BVH over their bounding boxes produces enormous, badly-balanced acceleration structures. A typical scene's BVH for ray-traced Gaussians is 3–8× larger than the equivalent triangle BVH.

**GRTX's trick — sphere-space rasterization.** Each anisotropic Gaussian is transformed into ray-space coordinates where it becomes a unit sphere:

$$\mathbf{r}' = \Sigma^{-1/2}(\mathbf{r} - \mu)$$

Now the BVH is built over uniform-radius spheres, which packs much tighter. Ray traversal does the inverse transform per intersection test. The math shifts from "is this ray within $k\sigma$ of an arbitrary anisotropic Gaussian" to "is the transformed ray within unit distance of the origin" — a much simpler test that maps cleanly onto RT cores.

**Reported speedup:** 3–5× over previous ray-traced Gaussian methods, bringing ray-traced rendering of 1M-Gaussian scenes to 30–60 FPS on RT-core-equipped GPUs (RTX 4070 and up). Still slower than rasterization but now within striking distance.

This matters because rasterization fundamentally cannot do reflections, refractions, or shadows from secondary rays — **3DGUT and GRTX are what enable Gaussian splats to work as full PBR scene elements** rather than just background photogrammetry.

---

## 5. Relighting: the inverse rendering problem

Vanilla 3DGS bakes lighting into per-Gaussian SH coefficients, which is why a captured scene "looks wrong" when you bring a torch into it. Relightable variants decompose appearance:

$$c_i = \int_\Omega L_i(\omega) \cdot f_r(\omega, \omega_o; \rho_i, \alpha_i, \mathbf{n}_i) (\omega \cdot \mathbf{n}_i)^+\, d\omega$$

where each Gaussian now stores **albedo $\rho$, roughness $\alpha$, normal $\mathbf{n}$, and metallic $m$** instead of just SH color. The integral is approximated using Monte Carlo over a learned environment map plus per-Gaussian visibility (precomputed via ray tests against the Gaussian field itself).

**GS³ (Triple Gaussian Splatting, SIGGRAPH Asia 2024)** uses three coordinated Gaussian fields — one for geometry, one for visibility, one for appearance — and reports **90 FPS relighting on a single RTX 3090**, with 40–70 minute training. This is the first relighting approach fast enough to be game-relevant.

**GaRe (2025)** added explicit shadow ray-tracing for outdoor scenes, separating sun, sky, and ambient terms via binary clustering on residuals between full-illumination and ambient-only renders.

**OTOY's Octane Render integration (announced for 2026)** is the first commercial path tracer to support Gaussian splats as native primitives with full path tracing — combining traditional mesh rendering and splat rendering in one global illumination solution.

---

## 6. Compression and streaming: the standardization wave

Two parallel developments in 2026 are reshaping the deployment story:

### glTF KHR_gaussian_splatting (Khronos, ratification Q2 2026)

The extension defines how Gaussian attributes (position, scale, rotation, SH, opacity) are stored within glTF mesh primitives. Backed by Google, NVIDIA, Apple, and Bentley. **Once ratified, any glTF-compatible engine or viewer loads splats natively** — no plugin required. A companion SPZ extension targets progressive web streaming.

### 4DGC: rate-distortion compression for streaming (CVPR 2025)

The first end-to-end RD-optimized 4DGS codec. The framework jointly optimizes:

$$\mathcal{L} = \mathcal{L}_{\mathrm{render}} + \lambda \cdot R(\theta)$$

where $R(\theta)$ is the entropy-coded bitrate of all parameters and $\lambda$ controls the rate-distortion tradeoff (sweep $\lambda$ for an RD curve). Two key components:

- **Motion-aware compact representation:** a sparse motion grid (typically 32³ × 8-dim) plus *compensated Gaussians* that store only residuals against the motion-grid prediction. Inter-frame correlations are exploited explicitly, similar to P-frames in video codecs.
- **Differentiable quantization:** parameters are passed through a learned quantizer with straight-through estimator gradients, allowing the network to find quantization-robust representations during training.

**Numbers:** 8–15 Mbps for high-quality volumetric video at 30 fps (vs. 200+ Mbps for naïve per-frame 3DGS) — putting 4DGS streaming in the same bandwidth bracket as 4K HEVC.

### HPC (Hierarchical Point-based Compression, 2026)

Achieves **67% storage reduction** over its baseline by being the first to also compress the *neural network parameters* (the deformation MLP) using inter-frame residuals, in addition to compressing the Gaussians themselves.

---

## 7. Spatio-temporal super-resolution: GS-STVSR mechanics

The relevant insight for VR/high-refresh-rate gaming is that *spatial* upsampling and *temporal* upsampling can share computation when both operate on Gaussian primitives.

For a target view at sub-frame time $t \in [t_k, t_{k+1}]$ and sub-pixel position $p$, both interpolations reduce to a single covariance resampling:

$$\Sigma'_{\mathrm{output}} = J_t \Sigma_t J_t^\top + \Sigma_{\mathrm{recon}}$$

where $\Sigma_{\mathrm{recon}}$ is a low-pass reconstruction filter — exactly the "EWA filter" from Heckbert's original work on texture filtering. Setting $\Sigma_{\mathrm{recon}}$ to match the target output resolution prevents shimmering automatically; this is what makes covariance-resampling-based super-resolution **temporally stable** in a way that pixel-space methods (DLSS, FSR) struggle to be on splat content.

The practical impact is that **shimmering on splat surfaces in VR is much less of a problem than it was 12 months ago** — covariance resampling at the splat level means Gaussians that appear smaller in a VR eye-buffer get filtered correctly, instead of producing the high-frequency aliasing that pixel-space upscalers can't fix retroactively.

---

## 8. Engine integration: actual numbers from production plugins

### Unity — UnityGaussianSplatting (Aras Pranckevičius)

- Open-source, Unity 6 LTS, PC/Mac/mobile.
- 1M Gaussians: 180–220 MB GPU memory, 3–8 ms/frame on RTX 3070.
- GPU radix sort: 1–4 ms for 1M Gaussians.
- Coordinate fix: rotation (0,0,180) starting point — 3DGS uses OpenGL conventions (Y-up right-handed), Unity is Y-up left-handed.

### Unreal — three production options:

**SplatRenderer (Dazai Studio, 2026)** — open-source UE 5.5+, custom RenderGraph compute pipeline. **2M+ Gaussians at 100+ FPS on RTX 4080**, supports both 3DGS .ply and 4DGS .gsd files, integrates with Level Sequencer for keyframable parameters.

**NanoGS (Tim Chen, March 2026)** — "Nanite-style" splat rendering. Breaks scenes into LOD clusters, screen-space error-driven LOD selection, GPU radix sort. **>4× viewport FPS improvement on RTX 2070** in published demos. Hard `gs.MaxRenderBudget` cap (e.g. 3M splats max) gives predictable failure modes for production.

**Luma AI / XScene plugins** — commercial route; Luma is in the Marketplace, XScene from XVERSE is open-source on GitHub.

### Coordinate-system gotcha (the most common bug)

3DGS training tools output OpenGL conventions. Unity/Unreal use different ones:

| Engine | Up-axis | Handedness | Fix from raw .ply |
|---|---|---|---|
| Unity | Y | Left | Rotate (0,0,180), invert Z |
| Unreal | Z | Left | Rotate (-90,0,0), scale (1,1,-1) |

Scale is also typically off by 1–100× because 3DGS trains in normalized scene coordinates — for room-scale scenes, expect a uniform scale of 10–50.

---

## Summary table: key methods and their numbers

| Method | Year | Headline metric | Storage | Where it shines |
|---|---|---|---|---|
| 4D-GS (Wu et al.) | 2024 | 82 FPS @ 800² (RTX 3090) | 50–150 MB / clip | Smooth dynamics, no topology change |
| 4DGS-1K | 2025 | 1029 FPS, 14× smaller | ~150 MB | Production playback |
| Spacetime GS / 4D-Rotor | 2024 | 70+ FPS @ 1080p | 200–500 MB | Topology change |
| Gaussian Frosting | 2024 | 60+ FPS, riggable | 100–300K Gaussians | Game-mesh hybrid |
| GS³ (relighting) | 2024 | 90 FPS relit | – | Relightable assets |
| 3DGUT | 2025 | Pinhole speed + reflections | – | Distorted cameras, secondary rays |
| GRTX | 2026 | 3–5× faster ray tracing | – | Path-traced splats |
| 4DGC | 2025 | 8–15 Mbps streaming | Variable | Volumetric video delivery |
| HPC | 2026 | 67% storage reduction | – | Streaming dynamic scenes |

---

## Where this is heading (2026 → 2027)

Three converging trends to watch:

1. **glTF KHR_gaussian_splatting ratification** in Q2 2026 will collapse the plugin fragmentation — expect engines to ship native loaders within a year.
2. **PBR-native splats** (relighting + ray tracing combined) reach playable performance on mainstream hardware as RT-core utilization improves. The OTOY Octane integration is the first commercial proof point.
3. **Generative 4D** (DreamGaussian4D, Diffusion4D and successors) means scenes won't only be *captured* — they'll be *generated* in 4DGS form directly from text/image prompts, then dropped into engines. This is where the asset pipeline starts to look genuinely different from traditional game development.

The mathematical machinery — covariance projection, alpha compositing, deformation fields, rate-distortion compression — is now stable enough that the next year's gains will come more from systems engineering (LOD, streaming, hybrid representations) than from new rendering equations.

---

### Selected references

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.
- Wu et al., *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*, CVPR 2024.
- Yuan et al., *1000+ FPS 4D Gaussian Splatting for Dynamic Scene Rendering*, NeurIPS 2025 (arXiv:2503.16422).
- Guédon & Lepetit, *Gaussian Frosting: Editable Complex Radiance Fields with Real-Time Rendering*, ECCV 2024.
- *3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting* (NVIDIA, arXiv:2412.12507).
- *GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering*, HPCA 2026.
- *4DGC: Rate-Aware 4D Gaussian Compression*, arXiv:2503.18421.
- *HPC: Hierarchical Point-based Latent Representation for Streaming Dynamic Gaussian Splatting Compression*, arXiv:2602.00671.
- gsplat library mathematical supplement, arXiv:2312.02121.
