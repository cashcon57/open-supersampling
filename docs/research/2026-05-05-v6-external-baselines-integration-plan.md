# 2026-05-05 — v6 external-baselines integration plan

**Status:** Integration plan derived from three parallel deep-reads of published 2D-Gaussian-Splatting work, conducted 2026-05-05.

**Source memos (read these for the underlying analysis):**
- `2026-05-05-gsasr-and-upscale3dgs-deep-read.md` — v6 fusion module + backbone warm-start
- `2026-05-05-anti-aliasing-stack-deep-read.md` — rasterizer-level AA stack (AAA-Gaussians + AA-2DGS + Analytic-Splatting)
- `2026-05-05-nvidia-vk-and-gaussianvideo-deep-read.md` — NVIDIA upper-bound + OSS-FX trajectory parameterization

This plan converts those memos into concrete OSS engineering actions, sequenced and effort-estimated.

---

## TL;DR

Five external systems materially de-risk v6 implementation:

1. **GSASR** has already published a cross-attention pixel↔Gaussian fusion module related to what OSS proposed for v6. Adopt GSASR-style window cross-attention; RoPE is unconfirmed upstream and currently an OSS adaptation pending source inspection. Treat its 5-head MLP-decoder + reference-grid-offset structure as the v6 fusion-block starting point after source inspection. Warm-starting from HATL_SA1B is a separate compatibility task, not current parity.
2. **AAA-Gaussians**, **AA-2DGS**, and **Analytic-Splatting** combine into a coherent rasterizer-level AA stack: train-time world-space clamp → perpendicular-ray dilation at projection → view-space angular tile-binning → object-space Mip per splat → analytical CDF integration per pixel. Stacked, expected joint quality lift is 2–3 PSNR vs vanilla EWA, with anti-popping as a side benefit. Engineering effort is ~6–8 weeks sequential.
3. **vk_gaussian_splatting** (NVIDIA, 2026.1) is the upper-bound benchmark target: 33.88 PSNR / 846 FPS at 3DGUT on RTX 5090. The DLSS-RR input contract is undocumented — first OSS action is to clone the repo and read the integration source.
4. **cyberiada/GaussianVideo** offers a B-spline trajectory parameterization that maps cleanly onto OSS-FX's α-conditioned rendering. Adopt the B-spline; reject the Neural-ODE (wrong inference profile for real-time).

The net of this research pass: most of v6's "cross-attention pixel↔Gaussian fusion" risk has related published precedent. OSS's contributions are (a) the streaming + temporal extension on top of stateless GSASR-style fusion, (b) the four-paper combined AA stack neither system has assembled, (c) custom per-vendor kernels covering NVIDIA / AMD / Intel / Apple / Vulkan-fallback. The architectural risk on v6 dropped substantially after this research pass.

---

## Per-component integration plan

### 1. v6 cross-attention fusion module — adopt GSASR structure

**Source:** GSASR (`ChrisDud0257/GSASR`, ICCV 2025), with `fea2gsropeamp_arch.py` pending source inspection.

**What GSASR shipped that maps directly onto v6's fusion block:**

- Learnable Gaussian-seed queries cross-attend to HAT-L feature windows
- GSASR-style window cross-attention; RoPE is unconfirmed upstream and currently an OSS adaptation pending source inspection
- Scale-conditioning via multi-head attention (lets the same module handle ×2, ×4, ×6, ×12)
- Reference-grid offset μ parameterization (predicts per-Gaussian center as offset from a reference grid, not absolute coordinates)
- Five parallel MLP heads decoding (μ, Σ-rotation, Σ-scale, opacity, RGB)

**What v6 adds on top that GSASR does not:** the temporal axis. GSASR is stateless (per-frame, no canvas memory). OSS extends this fusion pattern with:
- Persistent Gaussian canvas as a queryable history of K/V tokens across frames
- Engine-motion-vector-driven analytical warp on the canvas before each forward pass
- Spatial-Temporal Variation Score pruning (4DGS-1K, NeurIPS 2025) on the canvas

**Action items:**

| # | Task | Effort | Dependencies |
|---|---|---|---|
| 1.1 | Clone GSASR locally; read `fea2gsropeamp_arch.py` end-to-end | 1 day | none |
| 1.2 | Document GSASR's exact attention dim / head count / hidden width for each block | 0.5 day | 1.1 |
| 1.3 | Implement v6 fusion module following GSASR's structure where confirmed, plus the OSS RoPE adaptation and temporal cross-attend over canvas K/V | ~1 week | 1.1, 1.2 |
| 1.4 | Verify forward-pass parity at v6-fusion-block-frozen vs GSASR-published-fusion when canvas is empty (confirms our adaptation is correct) | 1 day | 1.3 |

**Risk:** GSASR is single-image SR; OSS adds frame-to-frame temporal cross-attention. The Gaussian seed queries become persistent canvas tokens. Whether GSASR's published hyperparameters transfer to the temporal regime needs measurement.

### 2. v6 backbone warm-start — OSS HAT-L-derived Heavy compatibility

**Source:** GSASR's published HAT-L checkpoint, trained on the SA-1B-derived corpus.

**What this saves OSS:** ~1–2 weeks of HAT pretraining time. GSASR's checkpoint is already converged on diverse natural-image content; OSS warm-starts from it instead of scratch-training the HAT backbone.

**Compatibility check:** Current OSS v6 code uses an OSS HAT-L-derived trimmed Heavy backbone (~17M target params): `depth=6`, `blocks_per_group=5`, not upstream HAT-L `[6]*12`. Use the OSS Heavy name consistently and do not equate it with upstream HAT-L. Upstream HAT-L warm-start would require a separate factory that mirrors the published YAML before loading `HATL_SA1B`.

**Action items:**

| # | Task | Effort | Dependencies |
|---|---|---|---|
| 2.1 | Pull HATL_SA1B checkpoint, verify it loads into a separate upstream-HAT-L factory | 2 hours | none |
| 2.2 | Decide whether upstream HAT-L warm-start is worth adding beside OSS HAT-L-derived Heavy | meeting | 2.1 |
| 2.3 | If upstream warm-start is adopted: add the separate factory and document the distinction from OSS Heavy | 1 day | 2.2 |

**Recommendation:** Keep OSS HAT-L-derived Heavy as the current teacher naming. Add upstream HAT-L warm-start only if the separate factory is implemented and verified.

### 3. Rasterizer-level anti-aliasing stack

**Sources:** AAA-Gaussians (`DerThomy/AAA-Gaussians`, ICCV 2025), AA-2DGS (`maeyounes/AA-2DGS`, NeurIPS 2025), Analytic-Splatting (`lzhnb/Analytic-Splatting`, ECCV 2024 Oral).

**Proposed combined OSS rasterizer-level AA stack:**

```text
TRAINING TIME (per-scene optimization)
└─ AA-2DGS: world-space frequency clamp on Gaussian frequency response
                                                                    │
                                                                    ▼
INFERENCE TIME (per-frame rasterization)
├─ AAA-Gaussians (Eq. 10): adaptive Σ dilation perpendicular to viewing ray
├─ AAA-Gaussians (Eqs. 14-17): view-space angular tile-binning (replaces screen-space AABB)
├─ AA-2DGS: object-space Mip filter via affine approximation of ray-splat intersection
└─ Analytic-Splatting: replace EWA point-sample with analytical CDF integral per pixel
```

**Expected combined quality lift vs vanilla EWA gsplat:** ~2–3 PSNR on multi-scale benchmarks; eliminates popping; reduces shimmering.

**Engineering cost:** Each technique requires CUDA-kernel modifications on top of gsplat. Rough estimates:
- AAA Eq. 10 perpendicular dilation: ~1 week (modifies projection step)
- AAA view-space angular tile-binning: ~1.5 weeks (replaces the radius-based tile assignment)
- AA-2DGS world-space clamp: ~1 week (training loss term)
- AA-2DGS object-space Mip: ~1.5 weeks (per-splat filter, cached)
- Analytic-Splatting CDF integral: ~2 weeks (rasterizer inner loop change)

**Total:** ~6–8 weeks if done sequentially. The stack composes — each layer is independent and adoption can be staged.

**Outstanding question:** AAA's Eq. 10 (perpendicular-ray dilation) is derived for 3D Gaussians. AA-2DGS works on 2D-disk Gaussians (which OSS uses). Whether AAA Eq. 10 degenerates cleanly to OSS's rank-2-disk geometry, or is fully subsumed by AA-2DGS's object-space filter, needs verification before implementation. The AA-deep-read memo flags this as the top open question. **Action: read both papers' supplementary materials to settle the overlap before coding.**

**Action items:**

| # | Task | Effort | Dependencies |
|---|---|---|---|
| 3.1 | Read AAA + AA-2DGS supplements; resolve Eq.-10-vs-Mip-filter overlap | 1 day | none |
| 3.2 | Stage 1: AAA view-space angular tile-binning + perpendicular dilation | 2 weeks | 3.1 |
| 3.3 | Stage 2: AA-2DGS world-space clamp (training loss) | 1 week | none, parallel with 3.2 |
| 3.4 | Stage 3: AA-2DGS object-space Mip filter | 1.5 weeks | 3.3 |
| 3.5 | Stage 4: Analytic-Splatting CDF integral | 2 weeks | 3.2 (replaces inner loop after AAA changes) |

### 4. Upper-bound benchmark — vk_gaussian_splatting

**Source:** `nvpro-samples/vk_gaussian_splatting` 2026.1 (released at NVIDIA GTC 2026 keynote).

**Concrete numbers to beat or fall behind:**

| Scene | Backend | PSNR | FPS (RTX 5090) |
|---|---|---|---|
| NeRF Synthetic | 3DGUT | 33.88 | 846 |
| MipNeRF 360 | 3DGUT | 27.43 | 317 |
| Mip-NeRF 360 | Pure rasterization | — | 510 |

These are pre-DLSS-RR baselines from `nv-tlabs/3dgrut`. The DLSS-RR-integrated path in the published `vk_gaussian_splatting` integrates AA + upscaling + denoising on top.

**What OSS cannot replicate, by design:**
- DLSS-RR (NVIDIA-proprietary, NGX runtime, RTX-only)
- 3DGRT (RT-core-dependent ray tracing)
- 3DGUT-grade fused tensor-core kernels

**What OSS will replicate via custom per-vendor kernels:**
- 3DGUT-equivalent Unscented-Transform projection (mathematically reproducible)
- Multi-pass AA / temporal accumulation (the four-paper stack above)
- Cross-vendor: AMD HIP + WMMA, Apple Metal + ANE, Intel XMX, Vulkan compute fallback

**The undocumented gap:** The DLSS-RR input contract — what tensor format, motion-vector semantics, depth-buffer convention DLSS-RR expects — is **not documented in any public NVIDIA source**. This is the top-priority knowledge gap to close.

**Action items:**

| # | Task | Effort | Dependencies |
|---|---|---|---|
| 4.1 | Clone `nvpro-samples/vk_gaussian_splatting`; locate the DLSS-RR integration call site | 0.5 day | none |
| 4.2 | Document the exact DLSS-RR input format, motion-vector convention, depth/normal channels expected | 1 day | 4.1 |
| 4.3 | Match that input contract on the OSS DLL-shim runtime so a future benchmark substitutes OSS for DLSS-RR cleanly | spans S7 | 4.2 |
| 4.4 | Build the head-to-head benchmark harness (same scene, same LR / motion / depth, same output res, same metrics) | 2 weeks | 4.3 + S7 shim |

**Note:** Action 4.4 cannot run until S7 (`oss/runtime/dxgi-hook` + NGX shim) ships. It's the closeout benchmark for v6, not a v6-development gate.

### 5. OSS-FX trajectory parameterization

**Source:** `cyberiada/GaussianVideo` (Bond et al.).

**Specific items to port:**

- **B-spline parameterization of per-Gaussian trajectories** — supports cheap evaluation at arbitrary α, supports natural extrapolation past the last anchor frame (which is exactly what OSS-FX needs). Replaces the current direct-α-multiply-mv approach in `oss/gaussian/extrapolation/extrapolator.py`.
- **Spatiotemporal hierarchical training schedule** — coarse-to-fine over both spatial and temporal axes. Useful as a Phase-1-vs-Phase-2 schedule for OSS-FX training.

**Specific items to NOT port:**

- **Neural-ODE camera-trajectory model.** Solves a 6-DoF camera trajectory inference problem OSS-FX does not have (game engines deliver exact camera matrices each frame). Adds runtime cost incompatible with the real-time budget.
- **Fit-then-replay regime.** GaussianVideo's 93 FPS / A40 number is post-fit; not relevant to streaming online inference.

**Action items:**

| # | Task | Effort | Dependencies |
|---|---|---|---|
| 5.1 | Read GaussianVideo's B-spline impl (likely `models/trajectory.py` or similar) | 0.5 day | none |
| 5.2 | Port B-spline parameterization into `oss/gaussian/extrapolation/extrapolator.py` | 1 week | 5.1 |
| 5.3 | Add spatiotemporal hierarchical training schedule to OSS-FX trainer | 1 week | 5.2 |

---

## Sequenced integration roadmap

| Phase | What | Wall time | Blocker chain |
|---|---|---|---|
| **v6.0a (architecture port followup)** | 1.1, 1.2, 2.1, 2.2, 3.1, 4.1, 4.2, 5.1 — read source, resolve open questions, decide whether to add upstream HAT-L warm-start beside OSS Heavy | ~1 week | — |
| **v6.0b (fusion/backbone followup)** | 1.3, 1.4 (fusion module parity where confirmed), 2.3 (optional upstream HAT-L factory if adopted) | ~1.5 weeks | v6.0a done |
| **v6.0c (rasterizer AA — staged)** | 3.2, 3.3 in parallel (AAA + AA-2DGS clamp) | ~2 weeks | v6.0b |
| **v6.0d (rasterizer AA continued)** | 3.4, 3.5 (Mip + CDF integral) | ~3.5 weeks | v6.0c |
| **v6 training (teacher)** | Train OSS HAT-L-derived Heavy teacher with full AA stack | ~50–60h GPU | v6.0d |
| **OSS-FX integration** | 5.2, 5.3 — concurrent with v6 training, on local 3080 Ti or M3 Max | ~2 weeks | v6.0b |
| **v6 benchmark prep** | 4.3 — DLL-shim DLSS-RR-compatible input contract | spans S7 | v6 model exists |
| **v6 closeout benchmark** | 4.4 — head-to-head vs vk_gaussian_splatting | ~2 weeks | S7 shim, v6 trained |

**Critical observation:** every v6 component except the eventual NVIDIA head-to-head can proceed on local hardware (3080 Ti + M3 Max) without cloud spend. The only remote-compute requirement is the v6 teacher's 50–60h training run, which already fits on the 3080 Ti.

## Risks and how this research changes them

| Risk identified earlier | After this research |
|---|---|
| Cross-attention pixel↔Gaussian has no production precedent | **Reduced** — GSASR shipped exactly this module |
| Anti-aliasing approach is novel | **Reduced** — three published papers compose into a known-good stack |
| NVIDIA's DLSS-RR-on-splats is an opaque competitor | **Slightly reduced** — concrete public numbers exist; DLSS-RR input contract still opaque |
| OSS-FX α-conditioned rendering has no peer reference | **Eliminated** — GaussianVideo provides direct peer reference |

What stayed risky:

- DLSS-RR exact input contract (still undocumented)
- The streaming temporal extension to stateless GSASR (OSS's contribution)
- Custom per-vendor kernels for the AA stack (still ~6–9 months engineering)
- Whether OSS HAT-L-derived Heavy distills cleanly to smaller students (no published reference)

## References

- GSASR: ChrisDud0257/GSASR — `fea2gsropeamp_arch.py`. ICCV 2025.
- upscale3dgs: KeKsBoTer/upscale3dgs. ICCV 2025. (Limited applicability: zero-net analytical-gradient upscaling, NOT temporal-stable per the source-memo finding.)
- AAA-Gaussians: DerThomy/AAA-Gaussians. ICCV 2025. Eq. 10 perpendicular-ray dilation; Eqs. 14–17 view-space angular bounds.
- AA-2DGS: maeyounes/AA-2DGS. NeurIPS 2025. World-space clamp + object-space Mip filter via affine approximation.
- Analytic-Splatting: lzhnb/Analytic-Splatting. ECCV 2024 Oral. Logistic-CDF analytical pixel-area integral.
- vk_gaussian_splatting: nvpro-samples/vk_gaussian_splatting 2026.1. NVIDIA GTC 2026. Pre-DLSS-RR baselines via nv-tlabs/3dgrut.
- GaussianVideo: cyberiada/GaussianVideo (Bond et al.). arXiv 2501.04782.
- Underlying deep-read memos:
  - `docs/research/2026-05-05-gsasr-and-upscale3dgs-deep-read.md`
  - `docs/research/2026-05-05-anti-aliasing-stack-deep-read.md`
  - `docs/research/2026-05-05-nvidia-vk-and-gaussianvideo-deep-read.md`

## Caveat

This integration plan is based on README-level and abstract-level reading via WebFetch. Some specific architectural details (GSASR's exact attention dims, AA-2DGS's exact object-space filter math, GaussianVideo's exact B-spline knot scheme) will require cloning the repos and reading the source. Action items 1.1, 3.1, 4.1, 5.1 are the cloning gates that resolve those uncertainties before commitment.
