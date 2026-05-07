# v6 Source Extraction Notes — Architectural Specifics

Date: 2026-05-06
Method: WebFetch / WebSearch only (no clones). Source paths refer to upstream
GitHub trees on `main` unless otherwise noted. Where I could not retrieve a
specific number from the source, I flag it explicitly rather than invent.

Repos covered:

1. GSASR — `ChrisDud0257/GSASR` (ICCV 2025)
2. HAT — `XPixelGroup/HAT` (CVPR 2023)
3. GaussianVideo — Bond et al. (arXiv 2501.04782) — official code repo not located
4. AAA-Gaussians — `DerThomy/AAA-Gaussians` (arXiv 2504.12811)
5. AA-2DGS — `maeyounes/AA-2DGS` (NeurIPS 2025, arXiv 2506.11252)
6. Analytic-Splatting — `lzhnb/Analytic-Splatting` (ECCV 2024 Oral, arXiv 2403.11056)
7. vk_gaussian_splatting — `nvpro-samples/vk_gaussian_splatting`

---

## 1. GSASR — Generalized and Efficient 2D Gaussian Splatting for ASR

**Citation.** Du Chen, Liyi Chen, Zhengqiang Zhang, Lei Zhang.
*Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution*.
ICCV 2025. arXiv:2501.06838 (v5).
Repo: <https://github.com/ChrisDud0257/GSASR>

**Architectural numbers (from §4.1 Implementation Details, arXiv 2501.06838v5).**

| Knob | Value |
| --- | --- |
| Gaussians per LR pixel `m` | 16 |
| Total Gaussian queries `N` | `m * H_LR * W_LR` (e.g. for 48x48 LR patch -> 36 864) |
| Gaussian-embedding dim `d` | 180 |
| Window size `k` (window cross/self-attention) | 12 |
| Gaussian Interaction Block depth `L` | 6 |
| LR training patch | 48 x 48 |
| Scale range trained | [1.0, 4.0] |
| Batch | 64, 500k iters, lr 2e-4 |
| Position encoding | learnable Swin-style relative bias (NOT RoPE) |

Note: the task brief asked for `fea2gsropeamp_arch.py` and "RoPE details" — the
v5 paper text describes the position encoding as "inherited from Swin
Transformer" (learnable relative position bias), not RoPE. There may be a RoPE
variant in code (`fea2gsropeamp_arch.py` filename suggests "feature-to-Gaussian
RoPE amplitude") but I could not retrieve the source file via WebFetch — the
repo's deeper `TrainTestGSASR/` tree returns 404 to the GitHub HTML scraper.
**Flag: confirm RoPE vs learnable bias by inspecting the arch .py once
the repo is cloned.**

**Encoder backbones offered in the repo.** README lists EDSR, RDN, SWIN, and
HAT-L. The "Ultra Performance" preset uses HAT-L as encoder; "Paper" preset
matches the published numbers above; "Enhanced" is an intermediate preset.

**Key code paths (inferred from README — exact filenames not retrievable via
WebFetch).**
- `inference_paper.py`, `inference_enhenced.py` (sic), `inference_paper_benchmark.py`
- `setup_gscuda.py` — builds the custom 2D rasterizer CUDA extension
- `TrainTestGSASR/` — training code, model archs, options yaml

**Code snippet.** Could not retrieve the architecture file directly — WebFetch
returned 404 for blob URLs into `TrainTestGSASR/`. Flagging for follow-up.

**Adaptation notes for `oss/sr/v6/`.**
- Use `m = 16` Gaussians per LR pixel as the default budget; budget at 48x48
  LR -> ~37k Gaussians per patch. Memory: 37k * (params per Gaussian) * 4 bytes.
- Gaussian decoder: 6 interaction blocks, 180-dim embedding, window-12 attention.
- Match training schedule (500k iters, batch 64, lr 2e-4) for fair comparison
  to published numbers.
- The current OSS encoder is **OSS HAT-L-derived Heavy** (~17M target params),
  trimmed from the upstream HAT-L family (`depth=6`, `blocks_per_group=5` in
  `oss/sr/v6/hat.py`). Do not equate it with upstream HAT-L. Upstream HAT-L
  warm-start requires a separate factory that mirrors the YAML below.

---

## 2. HAT — Hybrid Attention Transformer

**Citation.** Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, Chao Dong.
*Activating More Pixels in Image Super-Resolution Transformer*.
CVPR 2023. Repo: <https://github.com/XPixelGroup/HAT>

**HAT-L hyperparameters (verbatim from
`options/test/HAT-L_SRx2_ImageNet-pretrain.yml`).**

```yaml
network_g:
  type: HAT
  upscale: 2
  in_chans: 3
  img_size: 64
  window_size: 16
  compress_ratio: 3
  squeeze_factor: 30
  conv_scale: 0.01
  overlap_ratio: 0.5
  img_range: 1.0
  depths: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]   # 12 RHAG blocks of depth 6
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'
```

For comparison, the standard HAT (non-L) config uses `depths=[6,6,6,6,6,6]`
(6 RHAGs), same `embed_dim=180`, same `window_size=16`. **HAT-L doubles the
RHAG count from 6 to 12; everything else is identical.**

**Code paths.**
- `hat/archs/hat_arch.py` — HAT class, RHAG, OCAB, HAB modules.
- `options/test/HAT-L_SRx{2,3,4}_ImageNet-pretrain.yml` — the three HAT-L
  inference configs.

**`HAT.__init__` signature (verbatim, from `hat/archs/hat_arch.py`).**

```python
def __init__(self,
             img_size=64,
             patch_size=1,
             in_chans=3,
             embed_dim=96,
             depths=(6, 6, 6, 6),
             num_heads=(6, 6, 6, 6),
             window_size=7,
             compress_ratio=3,
             squeeze_factor=30,
             conv_scale=0.01,
             overlap_ratio=0.5,
             mlp_ratio=4.,
             qkv_bias=True,
             qk_scale=None,
             drop_rate=0.,
             attn_drop_rate=0.,
             drop_path_rate=0.1,
             norm_layer=nn.LayerNorm,
             ape=False,
             patch_norm=True,
             use_checkpoint=False,
             upscale=2,
             img_range=1.,
             upsampler='',
             resi_connection='1conv',
             **kwargs):
```

The defaults shown are the **base HAT** — for HAT-L the YAML overrides
`embed_dim -> 180`, `depths -> [6]*12`, `num_heads -> [6]*12`, `window_size
-> 16`, `mlp_ratio -> 2`.

**OCAB (overlapping cross-attention block) signature (verbatim).**

```python
def __init__(self, dim,
             input_resolution,
             window_size,
             overlap_ratio,
             num_heads,
             qkv_bias=True,
             qk_scale=None,
             mlp_ratio=2,
             norm_layer=nn.LayerNorm):
```

**Adaptation notes for `oss/sr/v6/`.**
- Current code uses **OSS HAT-L-derived Heavy** (~17M target params), not the
  upstream HAT-L YAML verbatim. It is trimmed to `depth=6` and
  `blocks_per_group=5`; this is the canonical name for the current teacher.
- If upstream HAT-L warm-start is added later, implement a separate factory that
  mirrors the YAML exactly: `embed_dim=180, depths=[6]*12,
  num_heads=[6]*12, window_size=16, mlp_ratio=2, compress_ratio=3,
  squeeze_factor=30, overlap_ratio=0.5, conv_scale=0.01`.
- Param count reference: upstream HAT-L is ~40M params at upscale=4. Verify
  against the ImageNet-pretrain checkpoint published by XPixelGroup before
  claiming weight parity.
- `img_size=64` is the *training* patch size used for relative position bias
  table; at inference HAT supports tiling via `HAT_tile_example.yml`.

---

## 3. GaussianVideo — Hierarchical 3D-GS with B-spline Trajectories

**Citation.** Andrew Bond, Jui-Hsien Wang, Long Mai, Erkut Erdem, Aykut Erdem.
*GaussianVideo: Efficient Video Representation via Hierarchical Gaussian
Splatting*. arXiv:2501.04782 (Jan 2025), ICCV 2025. Project page:
<https://cyberiada.github.io/GaussianVideo/>.

**Repo status.** The brief references `cyberiada/GaussianVideo` but that GitHub
path 404s. The org `cyberiada` exists (Koç University KUIS AI Lab) but does not
publish a `GaussianVideo` repo at the time of writing. Project page hosts only
HTML+videos, no code link visible. A third-party reimplementation exists at
`jiayi1129/GaussianVideo` but is not authoritative. **Flag: official code may
not be released yet, or may live under a different name.**

**B-spline trajectory parameterization (from arXiv:2501.04782 HTML).**

Per-Gaussian trajectory:

```
mu_n(t) = sum_{i=0..N} N_{i,p}(t) * P_{n,i}
```

where

- `N_{i,p}(t)` are degree-`p` B-spline basis functions
- `P_{n,i} ∈ R^d` are control points for the n-th Gaussian
- `p = 3` (cubic B-spline)
- The spline is **clamped**: first `p+1` knots fixed at 0, last `p+1` knots
  fixed at 1, remaining knots evenly spaced.

Number of control points `N` is **not fixed** — the paper presents it as a
configurable knob, with experiments showing cubic B-splines outperform
polynomial bases (e.g. on U-shaped trajectories, see Fig. 3) by avoiding the
overfitting/instability of high-order polynomials.

**Code paths.** Could not retrieve — official code not located.

**Adaptation notes for `oss/sr/v6/`.**
- Default to **cubic clamped B-splines** for per-Gaussian temporal trajectories.
- Treat `N` (control-point count per Gaussian) as a **per-experiment knob**;
  start with N≈8 for a 16-frame window (one CP per ~2 frames) and ablate.
- The clamped-knot convention matters: it makes `mu(0) = P_0` and `mu(1) = P_N`
  exactly, which is useful if we want endpoint constraints (e.g. matching
  warm-start positions at the start of a window).
- Until upstream code is released, implement against the equation above; basis
  evaluation can use de Boor's algorithm or a cached uniform-knot LUT.

---

## 4. AAA-Gaussians — Anti-Aliased and Artifact-Free 3D-GS Rendering

**Citation.** Michael Steiner, Thomas Köhler, Lukas Radl, Felix Windisch,
Dieter Schmalstieg, Markus Steinberger. *AAA-Gaussians: Anti-Aliased and
Artifact-Free 3D Gaussian Rendering*. ICCV 2025. arXiv:2504.12811.
Repos: <https://github.com/DerThomy/AAA-Gaussians> (training) and
<https://github.com/DerThomy/AAA-Gaussians-Rasterization> (CUDA rasterizer
submodule, fork of StopThePop-Rasterization).

**Equation 10 — perpendicular-ray dilation amplitude scaling (verbatim).**

```
sqrt(|Sigma_perp| / |Sigma_hat_perp|)
  = sqrt( |Sigma|   * d^T * Sigma^-1   * d
        / (|Sigma_hat| * d^T * Sigma_hat^-1 * d) )
```

Interpretation: when the 3D covariance `Sigma` is dilated to a low-pass
filtered `Sigma_hat = Sigma + epsilon*I` (perpendicular dilation), the splat
amplitude must be rescaled by this factor — measured **only** along directions
perpendicular to the viewing ray `d`, instead of the standard 2D-determinant
rescale used by Mip-Splatting. This avoids over-darkening Gaussians whose
principal axis aligns with the ray.

**Equations 14-17 — view-space angular bounds (verbatim).**

```
(14)  theta_{1,2} = atan2( s_{1,3} ± sqrt(s_{1,3}^2 - s_{1,1} * s_{3,3}),
                          s_{3,3} )

(15)  phi_{1,2}   = atan2( s_{2,3} ± sqrt(s_{2,3}^2 - s_{2,2} * s_{3,3}),
                          s_{3,3} )

(16)  (theta_mu - pi) < theta_1 < theta_mu < theta_2 < (theta_mu + pi)

(17)  theta_1 = max(-pi/2 + epsilon, theta_1)
      theta_2 = min( pi/2 - epsilon, theta_2)
```

where `s_{i,j}` are entries of the view-space covariance matrix and
`theta_mu, phi_mu` are the angular coordinates of the Gaussian mean. These
bounds replace the standard "project center, then add a screen-space radius"
heuristic, which fails (causes popping) for Gaussians that straddle the near
plane or extend beyond the frustum. The epsilon clamp in Eq. 17 prevents
tangent-singularity at ±pi/2.

**Code paths.** The README points to:
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/` — CUDA forward
  pass (separate repo `DerThomy/AAA-Gaussians-Rasterization`, based on
  StopThePop)
- `gaussian_renderer/` — Python-side renderer wrapper.

I could not pull the verbatim CUDA implementing these equations via WebFetch.
**Flag: once cloned, the bound-projection lives in the per-tile setup phase of
the rasterizer (look for `computeAngularBounds` or a function gated by Eqs.
14-17).**

**Adaptation notes for `oss/sr/v6/`.**
- For temporal stability of the SR pipeline, the perpendicular-ray dilation
  (Eq. 10) is a drop-in improvement over the Mip-Splatting 2D-determinant
  rescale — it removes one source of view-dependent flicker.
- The angular bounds (Eqs. 14-17) matter for any pipeline rendering near the
  near plane (cockpit views, third-person close characters). They eliminate
  popping at the cost of one atan2 + sqrt per Gaussian per culling pass.
- These both go into `oss/sr/v6/rasterizer/` as forward-pass changes, keeping
  the backward in sync.

---

## 5. AA-2DGS — Anti-Aliased 2D Gaussian Splatting

**Citation.** Mae Younes, Adnane Boukhayma. *Anti-Aliased 2D Gaussian
Splatting*. NeurIPS 2025. arXiv:2506.11252.
Repo: <https://github.com/maeyounes/AA-2DGS>.

**World-space frequency clamp (Eq. 9, verbatim).**

```
V_k^eff = V_k + sigma_smooth_k^2 * I_2

with
  sigma_smooth_k^2 = s_reg / nu_hat_k^2
```

where `V_k` is the world-space 2D covariance of splat k, `nu_hat_k` is its
maximum observed sampling frequency over the training views, `s_reg` a global
hyperparameter, and `I_2` the 2x2 identity. Interpretation: regularize each
splat's covariance so its frequency content does not exceed the highest
sampling rate seen during training — eliminates aliasing under camera zoom and
varying FoV without per-frame Mip filtering of geometry.

**Object-space Mip filter (Eq. 16, verbatim).**

```
G^mip_2D(x) = G^2D_{I + sigma * J*J^T}(u)
```

where `J` is the Jacobian of the ray-splat intersection mapping
(world-to-splat-local), and `u = J(x)` is the local splat coordinate. The
Mip-filtered Gaussian in splat-local space replaces the standard 2D Gaussian
evaluation. This is computed in **object space**, so no per-view CUDA-side
recomputation of pixel-space EWA is needed.

**Implementation note (verbatim from paper §4 / §5).**
> "We implement our object-space Mip filtering with custom CUDA kernels for
> both forward and backward computation."

**Code paths.**
- `submodules/diff-surfel-rasterization/` — custom CUDA kernels (forward +
  backward) implementing both filters.
- `gaussian_renderer/` — Python pipeline wrapper.
- `scene/` — gaussian model class; would hold `nu_hat_k` per-splat statistic
  and the `s_reg` hyperparameter.

I could not pull the verbatim CUDA via WebFetch. **Flag: once cloned, search
the rasterizer's `forward.cu` for `nu_hat`, `s_reg`, or the term `sigma_smooth`
to find the exact kernel.**

**Adaptation notes for `oss/sr/v6/`.**
- AA-2DGS is **2DGS-specific** (surfels). If v6 stays 3D-GS, prefer AAA-Gaussians
  Eq. 10 + Mip-Splatting. Use AA-2DGS only if v6 swaps to 2DGS surfels.
- The world-space clamp requires per-splat tracking of `nu_hat_k` during
  training — add it to the gaussian_model state and update on each forward
  pass that registers a training view.
- The Jacobian-based Mip filter (Eq. 16) is more accurate than EWA in
  object-space and computationally cheaper than per-frame screen-space EWA.

---

## 6. Analytic-Splatting — Analytic CDF Integration for Anti-Aliased 3D-GS

**Citation.** Zhihao Liang, Qi Zhang, Wenbo Hu, Ying Feng, Lei Zhu, Kui Jia.
*Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic
Integration*. ECCV 2024 (Oral). arXiv:2403.11056.
Repo: <https://github.com/lzhnb/Analytic-Splatting>.

**Logistic-CDF approximation (Eq. 9, verbatim).**

```
S(x) = 1 / (1 + exp( -1.6*x - 0.07*x^3 ))
```

This is a **polynomial-augmented sigmoid** that approximates the standard-normal
CDF `G(x)`. The `1.6 * x` linear term matches the well-known logistic
approximation to the Gaussian CDF (`1.6 ≈ pi / sqrt(3)` is the standard
constant); the `0.07 * x^3` cubic correction tightens fit in the tails.

**Surrounding equations (verbatim).**

```
(7)  G(x)   = integral_{-inf..x} g(x) dx,    g(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
(8)  I_g    = G(x_2) - G(x_1)
(9)  S(x)   = 1 / (1 + exp(-1.6*x - 0.07*x^3))   # the CDF approximation
(10) I_g(u) = G(u + 1/2) - G(u - 1/2)             # unit-pixel window integral
(11) I_g(u) ≈ S(u + 1/2) - S(u - 1/2)             # plug in the approximation
(12) Extension to 2D: diagonalize covariance, treat as separable 1D integrals
     in the rotated splat-local frame.
```

How it replaces EWA point sampling: vanilla 3D-GS (and Mip-Splatting) evaluate
the 2D Gaussian at the **pixel center** (point sample) and weight by alpha.
Analytic-Splatting instead computes the **analytic integral of the Gaussian
density over the unit pixel square**, using the CDF approximation in Eq. 11
along each principal axis of the projected covariance. This is exact
anti-aliasing for the Gaussian primitive itself (not just a low-pass filter on
covariance).

**Code paths.**
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu` — the
  custom CUDA forward replaces the original `(1/2pi) * exp(-0.5 * d^T Σ^-1 d)`
  point evaluation with the `S(u+1/2) - S(u-1/2)` window integral.
- `gaussian_renderer/__init__.py` and `network_gui.py` — Python-side renderer.
- `train.py`, `train_ms.py` — training entry points (multi-scale variant for
  the published anti-aliasing experiments).

I could not retrieve the `forward.cu` text directly via WebFetch (the GitHub
blob URL into the submodule returned 404). **Flag: once cloned, the relevant
function will be where the per-pixel alpha is computed in the standard 3D-GS
rasterizer's `renderCUDA` kernel.**

**Adaptation notes for `oss/sr/v6/`.**
- This is a **clean drop-in** for the 3D-GS rasterizer's per-pixel alpha
  computation. Two adds and one cubic per evaluation; minor overhead.
- Compared to AAA-Gaussians Eq. 10 (perpendicular-ray dilation), Analytic-
  Splatting attacks aliasing from the opposite direction: AAA expands the
  Gaussian to band-limit, AS integrates the original Gaussian exactly. The two
  are **compatible** — use AS for primitive integration *and* AAA Eq. 10 for
  view-dependent rescaling.
- The 2D extension in Eq. 12 requires eigendecomposition of the 2D covariance.
  Splats already expose this implicitly (axis-aligned bounding box uses the
  eigenvectors); the cost is a 2x2 symmetric eigen, ~10 FMAs per splat per
  pixel covered.

---

## 7. vk_gaussian_splatting — NVIDIA Vulkan 3D-GS Sample (DLSS-RR Integration)

**Citation.** NVIDIA `nvpro-samples`. *vk_gaussian_splatting* (release 2026.1).
Repo: <https://github.com/nvpro-samples/vk_gaussian_splatting>.

**DLSS-RR input layout, currently inferred.** This is inferred from the
`vk_gaussian_splatting` sample binding layout described in
`src/dlss_denoiser.hpp` (the application-side G-buffer setup) and
`src/dlss_wrapper.hpp` (the NGX wrapper), not a verified DLSS-RR public
contract. The application allocates a G-buffer and binds these resources by
enum:

| Slot enum (G-buf) | Format | Wrapper `ResourceType` | Purpose |
| --- | --- | --- | --- |
| `eDlssInputImage` | `R32G32B32A32_SFLOAT` | `eColorIn` | Noisy / pre-DLSS color (linear HDR) |
| `eDlssAlbedo` | `R8G8B8A8_UNORM` | `eDiffuseAlbedo` | Demodulated diffuse base color |
| `eDlssSpecAlbedo` | `R16G16B16A16_SFLOAT` | `eSpecularAlbedo` | Specular albedo at primary hit |
| `eDlssNormalRoughness` | `R16G16B16A16_SFLOAT` | `eNormalRoughness` | World-space normal (xyz) packed with roughness (w) |
| `eDlssMotion` | `R16G16_SFLOAT` | `eMotionVector` | 2D motion vector |
| `eDlssDepth` | `R16_SFLOAT` | `eDepth` | Linear ViewZ (camera-space depth) |
| `eSelectImage` | `R8_UNORM` | (optional) | Selection / object-id |

**Per-frame `DenoiseInfo` (verbatim from `dlss_wrapper.hpp`).**

```cpp
struct DenoiseInfo {
  vec2 jitter;       // sub-pixel jitter, in pixels
  mat4 modelView;    // camera view matrix
  mat4 projection;   // camera projection matrix
  bool reset;        // history reset flag (camera cut, scene reload)
};
```

**Invocation (from `src/dlss_denoiser.cpp`).**

```cpp
m_dlss.cmdDenoise(cmd, m_ngx, {jitter, modelView, projection, reset});
```

**Motion-vector convention.** The header file does not encode the sign or
scale convention explicitly. Motion-vector sign/scale/jitter handling remains
unresolved until the sample shaders are inspected. The `R16G16_SFLOAT` format
and 2D layout matches the standard DLSS-SR-style binding shape, but that does
not verify direction, resolution basis, or dejitter behavior.

**Depth convention.** `R16_SFLOAT`, single channel, named `ViewZ` — this is
**linear camera-space Z** (positive in front of camera), *not* normalized
device coordinates depth. This is what DLSS-RR expects for ray reconstruction.

**Normal convention.** `R16G16B16A16_SFLOAT`, packed as `(nx, ny, nz,
roughness)` in **world space**. (The `dlss_wrapper.hpp` enum offers two
alternatives: separate `eNormalRoughness` packed channel, or split `eRoughness`
+ unpacked normals.)

**Code paths (canonical).**
- `src/dlss_wrapper.hpp` / `src/dlss_wrapper.cpp` — generic NGX wrapper. Defines
  `ResourceType` enum, `ResourceSpec` struct, `DenoiseInfo` struct, `init()` /
  `cmdDenoise()` entry points.
- `src/dlss_denoiser.hpp` / `src/dlss_denoiser.cpp` — application-side G-buffer
  manager. Owns the seven Vulkan images, hands them to the wrapper, drives
  `cmdDenoise` per frame.
- Specular-hit-distance (`eSpecularHitDistance`) is exposed by the wrapper but
  the GS sample does not currently bind it (rasterized splats do not produce a
  hit distance; the ray-traced `VK3DGRT` pipeline could).

**Adaptation notes for `oss/sr/v6/`.**
- For OSS v6 to compare against the sample's DLSS-RR path, the currently
  inferred per-frame G-buffer layout is the seven slots above. v6's Gaussian
  rasterizer likely needs to produce these, not just final color:
  - `Color` (HDR linear, R32G32B32A32_SFLOAT): pre-tonemap radiance.
  - `Albedo`: demodulated diffuse — for splat-only scenes with no PBR, this is
    the SH degree-0 color modulated by the splat's "diffuse fraction" (or
    just the SH DC term as a stand-in).
  - `SpecAlbedo`: specular component (could be 0 for a pure-Lambertian
    interpretation of splat radiance).
  - `Normal`: world-space normal — derive from the splat's shortest scale axis
    (2DGS-style) or from the local depth gradient.
  - `Roughness`: pack alongside normal in `.w`. For a non-PBR splat scene,
    treat as a constant (e.g. 1.0 — fully rough).
  - `Motion`: per-pixel screen-space motion vector. Sign, scale, resolution
    basis, and jitter handling are unresolved until shaders are inspected.
  - `Depth`: linear ViewZ at the rasterized splat surface.
- Use `DenoiseInfo.reset = true` on cuts and on scene reload to avoid history
  poisoning.
- Output color is `eColorOut` (`R32G32B32A32_SFLOAT`), allocated separately,
  written by DLSS-RR.
- `jitter` is in **pixels** at the *input* (low-res) resolution, the same
  jitter applied to the projection matrix during rasterization.

---

## Cross-cutting recommendations for v6

1. **SR backbone (image-domain SR side).** Use OSS HAT-L-derived Heavy
   (~17M target params) as the current feature encoder. Do not equate it with
   upstream HAT-L. If upstream HAT-L warm-start is pursued, add a
   separate YAML-matching factory.

2. **Gaussian decoder (image-domain GS side).** Match GSASR-paper config:
   `m=16, d=180, k=12, L=6`. Verify whether the upstream uses RoPE or learnable
   bias by inspecting `fea2gsropeamp_arch.py` once cloned.

3. **3D-GS rasterizer for v6 frame-domain SR.** Stack three orthogonal
   anti-aliasing fixes:
   - Analytic-Splatting Eq. 9 (replaces point sample with analytic window
     integral).
   - AAA-Gaussians Eq. 10 (perpendicular-ray dilation rescale, replaces the
     2D-determinant rescale used by Mip-Splatting).
   - AAA-Gaussians Eqs. 14-17 (view-space angular bounds, replaces planar
     screen-space culling — eliminates near-plane popping).
   These are independent and compose cleanly.

4. **Temporal Gaussians (v6 Gaussian-temporal track).** Use cubic clamped
   B-splines per GaussianVideo. Endpoint-clamped knots make warm-start /
   window-boundary continuity straightforward.

5. **DLSS-RR plumbing.** The seven-slot G-buffer in §7 is currently inferred
   from the `vk_gaussian_splatting` sample binding layout. Motion-vector
   sign/scale/jitter handling is unresolved until the shaders are inspected.
   Validate against the `nvpro-samples` reference behavior (cmdDenoise output
   should match their sample on a flowers.ply rotation test).

---

## Items I could NOT retrieve

- **GSASR `fea2gsropeamp_arch.py` source.** GitHub blob URLs into
  `TrainTestGSASR/` returned 404 to WebFetch. Architectural numbers above are
  from the v5 paper text, not from the code. The `RoPE` aspect specifically
  needs source-code confirmation — the paper text describes Swin-style
  learnable position bias, but the filename suggests RoPE. **Action: clone
  and inspect.**
- **GaussianVideo official code.** The `cyberiada/GaussianVideo` GitHub path
  404s; only the project page and arXiv exist. Third-party impl
  `jiayi1129/GaussianVideo` exists but is unverified. **Action: confirm with
  authors or wait for release.**
- **Verbatim CUDA snippets** for AAA-Gaussians, AA-2DGS, and Analytic-
  Splatting forward kernels. WebFetch returned 404 for the deep blob URLs into
  `submodules/`. **Action: clone the three rasterizer submodules and grep
  `forward.cu` for the equation tags listed above.**
- **vk_gaussian_splatting motion-vector sign convention.** The headers expose
  the buffer slot but not the in-shader sign. The NGX SDK convention is
  documented externally but should be confirmed by reading the GS sample's
  shaders (the `shaders/` dir). **Action: clone and inspect the relevant
  shader that writes `eDlssMotion`.**
