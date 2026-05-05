# 2026-05-05 — Anti-aliasing stack deep-read: AAA-Gaussians + AA-2DGS + Analytic-Splatting

## TL;DR

Three published anti-aliasing techniques for Gaussian Splatting, ranked by OSS relevance:

1. **AA-2DGS** (NeurIPS 2025) — directly architecturally aligned. OSS uses 2D Gaussians on a persistent canvas; AA-2DGS is the first paper to identify that 2D-GS aliasing is structurally different from 3D-GS aliasing because flat disks have no implicit volumetric low-pass and 2DGS's screen-space clamp is "ineffective." It contributes (a) a world-space flat smoothing kernel that bounds primitive frequency to the training-view sampling rate, and (b) an object-space Mip filter derived from an affine approximation of the ray–splat intersection mapping. Specific equations are not retrievable from the abstract / project landing pages, so the math must be confirmed against the PDF before integration.
2. **AAA-Gaussians** (ICCV 2025 Highlight) — directly addresses popping, which is OSS's headline temporal-stability claim. Adaptive 3D smoothing scales the Gaussian only perpendicular to the viewing ray (Eq. 10), and view-space frustum bounding solves for tangent-half-angles in view space (Eqs. 14–17), preventing the discontinuous tile-touch changes that cause pops in vanilla 3DGS. Built on StopThePop, so the perspective-correct sort assumption is baked in.
3. **Analytic-Splatting** (ECCV 2024 Oral) — the mathematical foundation. Replaces point-sampling at pixel center with an analytic pixel-area integral using a conditioned logistic CDF approximation `S(x) = 1/(1 + exp(−1.6x − 0.07x³))` (Definition 1 / Eq. 15). ~10% FPS overhead vs Mip-Splatting. Should be considered the "default" per-pixel evaluation in any v6 rasterizer rewrite.

The combined stack — AAA's perpendicular-ray world-space prefilter + AA-2DGS's 2D-disk-specific object-space Mip + Analytic-Splatting's per-pixel CDF integral — is non-redundant: each operates at a different pipeline stage (canvas-update / projection / per-pixel shading).

## The OSS temporal-stability problem these papers address

OSS's architectural moat over DLSS/FSR is that the Gaussian canvas accumulates samples in a covariance-resampling representation that lives in world space. A pixel-grid SR network fundamentally cannot anti-alias in world coordinates: it sees a downsampled raster, applies learned filters in screen space, and produces a high-frequency raster — any aliasing baked into the input low-res frame is "denoised" rather than truly resolved, and any frame-to-frame change in projection geometry produces the temporal flicker DLSS/FSR ship with motion vectors and history accumulation to mask. OSS's claim is that because the canvas itself is band-limited and resampled with proper covariance footprints, the same physical canvas can be raster-sampled at any output resolution without flicker.

That claim only holds if the rasterizer (a) prefilters primitives against sampling rate, (b) projects them without view-space discontinuities, and (c) shades each pixel as an area integral rather than a center sample. Failures of (a) cause shimmer when zooming or dollying; failures of (b) cause popping at frame boundaries (Gaussians snap into tiles as their projected AABB crosses a tile edge or as guard-band clipping toggles); failures of (c) cause sub-pixel scintillation on thin Gaussians. These are the three failure modes pixel-grid SR cannot address at the source — they all happen before the SR network sees the frame, or they are the very artifacts the SR network tries to fake-fix. The three papers reviewed here address (a), (b), and (c) respectively.

## AAA-Gaussians

### Mechanism

**Adaptive 3D smoothing filter (§3.2, Eq. 10).** Mip-Splatting's 3D filter dilates Σ isotropically by a min-pixel-size term derived from the worst-case training view. AAA-Gaussians replaces the global volume normalization with a directional one: only the component of Σ perpendicular to the viewing ray `d = (μ − cam) / ‖μ − cam‖` is normalized. The retrieved equation (10) is

```
|Σ⊥| / |Σ̂⊥|  =  √( |Σ| · dᵀΣ⁻¹d  /  |Σ̂| · dᵀΣ̂⁻¹d )
```

where Σ̂ is the post-filter covariance and Σ⊥ denotes projection onto the plane orthogonal to d. The filter "dynamically adapts to the camera's sampling frequency" while keeping the along-ray scale untouched, which avoids the over-transparency Mip-Splatting introduces under wide-FOV / near-camera viewing.

**View-space frustum bounding (§3.3, Eqs. 14–17).** Rather than computing screen-space AABBs (which are wrong for Gaussians whose centers are near or behind the near plane), AAA-Gaussians fits view-space planes by solving for tangent half-angles θ, φ:

```
θ₁,₂ = arctan( (s₁,₃ ± √(s₁,₃² − s₁,₁ s₃,₃)) / s₃,₃ )
```

with rotation/clamp constraints in Eqs. 16–17 keeping angles in `[−π/2 + ε, π/2 − ε]`. The resulting planar bounds are then promoted to 3D screen-space planes for tile-based culling and hierarchical 4×4 culling.

**What pops in vanilla 3DGS:** Gaussians whose centers leave/enter the frustum cause discontinuous AABB jumps; near-plane clipping snaps; and tile-list assignment toggles based on screen-space AABB overlap. AAA's view-space angular bound is continuous across the near plane and across frustum edges, so primitives fade rather than snap.

### Integration into OSS

OSS's v6 rasterizer is gsplat-derived; the StopThePop perspective-correct sort and per-Gaussian view-space angular bounding could be ported as a CUDA-side preprocess pass before tile binning. Recommended integration:

- **Adopt the perpendicular-ray adaptive filter** at canvas-update / projection time (i.e., when world-space Σ is projected to per-view 2D Σ′). The math is a few extra MADs per primitive — cheap.
- **Adopt view-space angular bounding** as the AABB replacement for tile-binning. Higher engineering cost (rewrites tile-list construction and bounding-box culling) but this is the specific change that buys the "no popping" claim.
- The hierarchical 4×4 culling is an optimization; defer.

OSS uses 2D Gaussians, so Eq. 10 must be re-derived for the 2D-disk case (Σ has rank-2 in world space; "perpendicular to d" means the component along the disk's normal vs along its tangent plane — see §AA-2DGS below for the 2D-specific affine mapping).

### Numbers

In-distribution PSNR (Table 1, retrieved): M360 27.84 (3DGS 27.44, Mip-Splatting 27.54, StopThePop 27.30); Deep Blend 30.49 (29.51 / 29.66 / 29.93). T&T PSNR 23.58 trails Mip-Splatting 23.82. Performance (Table 5): 7.72–10.66 ms/frame on M360, comparable to MCMC's 6.79–8.81 ms. Out-of-distribution wide-FOV results (Table 3) — claimed superiority but specific numbers not retrieved.

## AA-2DGS

### Mechanism

The retrievable description (abstract + project page) is non-mathematical. Two contributions:

**World-space flat smoothing kernel.** Constrains the frequency content of each 2D Gaussian primitive to the maximal sampling frequency observed across training views. Conceptually parallel to Mip-Splatting's 3D filter but operating on the flat-disk geometry of 2DGS rather than an ellipsoid. The exact equation for "maximal sampling frequency from training views" applied to a rank-2 disk is **not retrievable from the abstract or project page** and must be confirmed against the PDF.

**Object-space Mip filter.** Uses an affine approximation of the ray–splat intersection mapping to express the screen-space pixel footprint in the splat's local 2D coordinate frame. The Mip prefilter is then applied in that local frame. This is structurally close to EWA filtering but is computed per-splat in object space rather than per-pixel in screen space, which lets it be amortized across the splat's tile coverage. **Equations not retrieved** — flag for follow-up read of arXiv 2506.11252.

The paper identifies vanilla 2DGS's screen-space clamp (the heuristic that prevents Gaussians from collapsing below a min-pixel-size in screen space) as "ineffective" — this is the failure mode AA-2DGS specifically fixes.

### Why this is especially relevant for OSS

OSS uses 2D Gaussians on a persistent canvas. The geometric difference vs 3D-GS that matters for AA:

- A 3D ellipsoid integrated along a ray is itself a 1D Gaussian — there is implicit volumetric low-pass along the ray direction, which Mip-Splatting can exploit with a 3D filter dilation.
- A 2D disk has zero thickness along its normal. The view-ray–disk intersection is a 2D point in the disk's tangent plane. There is no implicit along-ray averaging, so the only available prefilter is in the disk's tangent plane (object-space) or in its 2D projected screen footprint (screen-space — which 2DGS's clamp does, ineffectively).

This is exactly why Mip-Splatting's 3D filter cannot be ported verbatim to OSS, and why AA-2DGS's contribution — re-deriving the prefilter in the disk's local 2D frame using an affine-approximated pixel footprint — is the right form for OSS's primitive geometry.

### Integration into OSS

Provisional recommendation pending PDF read:
- **Adopt object-space Mip filter** as the per-splat prefilter at projection time. The affine approximation of the ray–splat intersection is essentially the Jacobian of the splat-local-uv → screen-uv map, which OSS already needs to compute for its EWA projection — minimal extra cost.
- **Adopt world-space frequency clamp** during canvas optimization (training-time only, frozen at inference). This is densification-adjacent — primitives that try to encode frequencies above the worst training-view sampling rate get clamped.
- Defer integration until code is released and the equations are verified.

### Numbers

**Not retrieved.** No PSNR/SSIM/LPIPS or FPS numbers are visible on the abstract page or the GitHub README. Code release status: repository public with installation + training scripts (commit `59b23a1` "Code Release", 2025).

## Analytic-Splatting

### Mechanism

**Conditioned logistic CDF approximation (§4.1, Definition 1).** A 1D Gaussian CDF (with σ=1) is approximated by

```
S(x) = 1 / (1 + exp(−1.6 x − 0.07 x³))
```

The pixel response of a 1D Gaussian over the unit-width pixel window centered at `u` is then the difference of two CDF evaluations:

```
ℐ_g(u) ≈ S(u + ½) − S(u − ½)
```

**2D extension (§4.2, Eq. 15).** The 2D covariance Σ is diagonalized to (σ₁, σ₂) and the integration domain is rotated into the principal-axis frame. The 2D pixel-window integral factors into a product of two 1D integrals:

```
ℐ_g²ᴰ(u) = 2π σ₁ σ₂ · [S_{σ₁}(ũ_x + ½) − S_{σ₁}(ũ_x − ½)] · [S_{σ₂}(ũ_y + ½) − S_{σ₂}(ũ_y − ½)]
```

This response replaces the point-sampled `g²ᴰ(u | μ, Σ)` in the per-pixel transmittance/alpha computation:

```
C(u) = Σᵢ Tᵢ · ℐ_{gᵢ}²ᴰ(u | μᵢ, Σᵢ) · αᵢ · cᵢ
```

### Mathematical foundation

Vanilla 3DGS evaluates `g²ᴰ` at the pixel center — equivalent to nearest-sample reconstruction, which aliases under any deviation from the training sampling rate. EWA pre-filtering (Zwicker et al. 2001) convolves the projected 2D Gaussian with a screen-space reconstruction kernel, equivalent to dilating Σ′ by the kernel covariance — this is a *Gaussian-convolved-with-Gaussian* approximation of the pixel integral. Mip-Splatting extends EWA with a 2D screen-space dilation tied to a min-pixel-size, but still evaluates the result at a single point.

Analytic-Splatting differs because it is the **exact analytic integral** (under the logistic-CDF approximation, which is itself accurate to ~10⁻³ vs the true erf) of the un-prefiltered Gaussian over the pixel area. It is mathematically distinct from EWA: EWA changes the primitive's covariance; Analytic-Splatting changes the *evaluation operator* from sample-at-center to integrate-over-area, leaving the covariance untouched.

### Integration into OSS

Strong recommendation: **adopt as the default per-pixel evaluation** in v6 rasterizer. The change is local — replace the line that evaluates `exp(−½ · (p−μ)ᵀ Σ′⁻¹ (p−μ))` with the four-CDF-evaluations product. Two `S(x)` calls per dimension per Gaussian per pixel; cubic in `x` so ~6 FMAs + 1 exp + 1 reciprocal, i.e. roughly 4× the cost of the current single exp.

Reported overhead: "frame rate is only 10% lower than Mip-Splatting" (§6, Limitations). For OSS's use case where rasterizer time is small relative to the canvas-update step, a 10% rasterizer slowdown is acceptable in exchange for sub-pixel stability.

### Numbers

Mip-NeRF 360 multi-scale (Table 2): PSNR 29.51 / SSIM 0.887 / LPIPS 0.123 vs 3DGS 27.63 / 0.853 / 0.156 and Mip-Splatting 29.12 / 0.883 / 0.134. Single-scale numbers in Appendix B.1 — not retrieved.

## Synthesis: the OSS anti-aliasing stack (proposed)

The three techniques are non-overlapping in pipeline stage, so they compose. Proposed v6+ stack:

**Stage 1 — Canvas update / training-time frequency clamp (AA-2DGS world-space kernel).**
During canvas optimization, each 2D Gaussian's spatial frequency content is clamped to the maximal sampling frequency observed across training views. This is a regularizer added to the loss / a hard clamp on Σ eigenvalues. Cost: negligible at inference; modest training-time overhead. Pending PDF read for exact form.

**Stage 2 — Per-view projection (AAA-Gaussians adaptive perpendicular-ray filter, adapted to 2D disks).**
At projection time, dilate each Gaussian only along the directions perpendicular to the viewing ray, using Eq. 10 re-derived for the 2D-disk geometry. This is the dynamic, per-frame adaptation to the current camera's pixel-projected sampling rate. Cost: a few MADs per primitive per frame.

**Stage 3 — Per-view bounding & tile-binning (AAA-Gaussians view-space angular bounding, Eqs. 14–17).**
Replace screen-space AABB tile-binning with view-space angular tangent-half-angle bounds. This is the change that buys "no popping" at frustum edges and the near plane. Cost: rewrite of the tile-binning kernel; modest per-primitive arithmetic increase.

**Stage 4 — Per-splat object-space prefilter (AA-2DGS Mip filter).**
Apply the object-space Mip prefilter using the affine ray–splat-intersection Jacobian. This is the disk-local prefilter that 2DGS's screen-space clamp fails to provide. Cost: reuses the Jacobian OSS already computes for EWA projection.

**Stage 5 — Per-pixel evaluation (Analytic-Splatting CDF integral).**
Replace `exp(−½ Δᵀ Σ′⁻¹ Δ)` evaluation at pixel center with the four-CDF product (Eq. 15). Cost: ~10% rasterizer FPS hit, sub-pixel stability gain.

Order matters: Stages 1–2 prevent under-sampling artifacts (zoom-in shimmer); Stage 3 prevents projection-discontinuity pops (the headline OSS claim); Stage 4 prevents 2D-disk aliasing under oblique view angles; Stage 5 prevents sub-pixel scintillation on thin/edge-on Gaussians.

Combined quality lift (estimate, not measured): Analytic-Splatting alone reports +1.9 PSNR over 3DGS on multi-scale Mip-NeRF 360. AAA-Gaussians + Analytic-Splatting are claimed by AAA to be complementary. AA-2DGS numbers not retrieved. A reasonable upper bound on PSNR for OSS's canvas at multi-scale is 3DGS-baseline + 2.0–2.5 PSNR if all four stages compose cleanly.

Engineering effort estimate: Stage 5 is ~1 day (local kernel edit). Stage 2 is ~1 week (re-derivation + projection-kernel edit). Stages 3–4 each are ~2–3 weeks (rewrites of tile-binning and EWA-projection kernels). Stage 1 is dependent on AA-2DGS code release and another ~1 week to integrate into the canvas optimization loop.

## Outstanding questions for follow-up

1. **AA-2DGS equations.** The world-space frequency clamp and the affine ray–splat-intersection Jacobian are described abstractly in the abstract / project page. Need to read arXiv 2506.11252 PDF to extract the exact math before implementing Stages 1 and 4. Repo commit `59b23a1` is the "Code Release" — verify the implementation matches the paper.
2. **AAA-Gaussians' StopThePop dependency.** The view-space angular bounding (§3.3) is implemented inside the StopThePop fork. OSS's gsplat-derived rasterizer does not assume per-tile perspective-correct sort. Determine whether the angular bounding works without the StopThePop sort or whether porting it requires also adopting the sort.
3. **2D-disk re-derivation of AAA Eq. 10.** Eq. 10 assumes a rank-3 Σ. For OSS's rank-2 (flat-disk) Σ, the perpendicular-to-d projection collapses differently — verify that the math degenerates gracefully or whether AA-2DGS's object-space Mip already subsumes it.
4. **Combined-stack interaction.** Stages 2 (AAA perpendicular-ray dilation) and 4 (AA-2DGS object-space Mip) both prefilter against view-dependent sampling rate, in different coordinate frames. Verify by ablation whether they double-prefilter (over-blur) or compose orthogonally.

## References

- Steiner, Köhler, Radl, Windisch, Schmalstieg, Steinberger. **AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering.** ICCV 2025 (Highlight). arXiv:2504.12811. https://github.com/DerThomy/AAA-Gaussians (commit `dc5b3f2`, "Update citation for AAA-Gaussians paper"). Project page: https://derthomy.github.io/AAA-Gaussians/
- Younes, Boukhayma. **Anti-Aliased 2D Gaussian Splatting.** NeurIPS 2025. arXiv:2506.11252. https://github.com/maeyounes/AA-2DGS (commit `59b23a1`, "Code Release").
- Liang, Zhang, Hu, Zhu, Feng, Jia. **Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration.** ECCV 2024 (Oral). arXiv:2403.11056. https://github.com/lzhnb/Analytic-Splatting (commit `a905939`, "Update README.md"). Project page: https://lzhnb.github.io/project-pages/analytic-splatting/
- Zwicker, Pfister, van Baar, Gross. **EWA Volume Splatting.** IEEE Visualization 2001. (Reference for the EWA filter compared against in §Analytic-Splatting Mathematical foundation.)
- Yu, Chen, Antic, Peng, Geiger. **Mip-Splatting: Alias-free 3D Gaussian Splatting.** CVPR 2024. (Baseline for all three papers' anti-aliasing comparisons.)
- Radl, Steiner, Parger, Weinrauch, Kerbl, Steinberger. **StopThePop: Sorted Gaussian Splatting as a Ranked Renderer.** SIGGRAPH 2024. (AAA-Gaussians is built on this.)
