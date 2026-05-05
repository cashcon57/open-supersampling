# 2026-05-05 — NVIDIA vk_gaussian_splatting + cyberiada/GaussianVideo deep-read

## TL;DR

NVIDIA's `vk_gaussian_splatting` (release 2026.1, showcased at the Jensen GTC 2026 keynote) is the closest-existing reference for a "DLSS-on-splats" runtime: a Vulkan testbed that composes 3D Gaussian Splatting rasterization (3DGS), 3D Gaussian Ray Tracing (3DGRT), 3D Gaussian Unscented Transform (3DGUT), and DLSS Ray Reconstruction (DLSS-RR) into a single hybrid rendering stack. It defines the upper bound OSS must benchmark against once the OSS DLL-shim runtime exists, but its dependence on RTX hardware (RT cores), Tensor cores, and the proprietary NGX runtime puts large portions of it explicitly out of scope for OSS's cross-vendor mandate. Cyberiada/GaussianVideo (Bond et al., arXiv:2501.04782) is a peer reference for the FX subset only: it parameterizes camera trajectory through a Neural-ODE while keeping an explicit 3DGS scene, and supports arbitrary-timestep frame interpolation — structurally analogous to OSS-FX's α-conditioned canvas rendering but at video-encode quality (45 min train, 37–43 PSNR) rather than real-time game inference budget.

## vk_gaussian_splatting (NVIDIA): the upper-bound benchmark

### Architecture

The testbed is documented as "a testbed to explore and compare various approaches to real-time visualization of 3D Gaussian Splatting (3DGS)" and explicitly "is not a 3DGS reconstruction tool — it is a viewer." It loads `.ply` and `.spz` (Niantic) splat assets and exposes four render pipelines:

- **VK3DGSR** — rasterization (vertex- or mesh-shader path) with GPU radix sort or async CPU sort. The non-DLSS legacy NVIDIA blog recorded "510 FPS with 1.961 ms frame time" on Mip-NeRF 360 scenes (bicycle/kitchen/garden).
- **VK3DGRT** — ray tracing of volumetric Gaussians via Vulkan RTX, supporting distorted cameras, rolling shutter, secondary rays, reflection, refraction, ray-traced shadows.
- **VK3DGUT** — Unscented Transform front end that lets a rasterization framework handle distorted cameras and (via 3DGRT secondary-ray fallback) reflections.
- **VK3DGHR** — hybrid: primary rays via rasterization, secondary via ray tracing.

The 2026.1 release added: multi-instance splat sets with a unified global sort, GPU-built particle acceleration structures with multi-TLAS/BLAS chunking, deferred shading with front-to-back rasterization and depth consolidation, multi-light system (point/spot/directional with RT shadows), stochastic splat sorting, Monte-Carlo trace strategy for interactivity, and DLSS-RR as the AA + upscaler + denoiser for the ray-traced path.

The published material describes DLSS-RR's role as: "DLSS Ray Reconstruction then upscales and denoises the result." The exact tensor inputs (G-buffer composition, motion-vector format, noisy-radiance layout) are **not documented in the README, the project site, or the public NVIDIA dev blog post**; they are referenced only as "DLSS buffers" in the visualization-mode list. To fully resolve the input contract we would need to read the source under `vk_gaussian_splatting/src/` directly. (Flagged in Outstanding Questions.)

Reference performance on the underlying 3DGRT/3DGUT (from `nv-tlabs/3dgrut`, RTX 5090): NeRF Synthetic 33.87 PSNR / 347 FPS (3DGRT) and 33.88 PSNR / 846 FPS (3DGUT); MipNeRF360 27.43 PSNR / 317 FPS (3DGUT). These are pre-DLSS-RR composition numbers.

### Why this is OSS's upper-bound benchmark

This is what NVIDIA's vertical integration enables today: rasterized + ray-traced splats post-processed by a tensor-core ML denoiser/upscaler that was specifically trained for ray-traced inputs. OSS cannot match it on NVIDIA hardware — DLSS-RR is closed-weight, NGX-runtime-bound, and has years of NVIDIA training-data lead on ray-traced denoising. OSS's defensible position is **the open-cross-vendor alternative**: same architectural shape (splats → ML post), runs on AMD, Intel, mobile, and (notably) the consumer Vulkan path that does not have NGX licensed.

Concrete claims OSS must measure against:

- **Output quality**: pre-DLSS NVIDIA already lands ~27–34 PSNR on standard scenes. After DLSS-RR, NVIDIA implicitly claims production-grade temporal stability and AA — not separately quantified in any public source we could retrieve.
- **Latency**: VK3DGSR alone hits 510 FPS / 1.961 ms on a single tested scene. The 2026.1 RT+DLSS path's frame budget is undocumented publicly; almost certainly higher than 1.961 ms but advertised as "real-time."
- **Hardware**: explicitly RTX-class with RT cores + Tensor cores + NGX runtime. The NVIDIA blog frames this as "real-time GPU-accelerated."

OSS's honest gap: we will not beat DLSS-RR quality on NVIDIA hardware. Our wins must come from cross-vendor coverage and open-pipeline auditability.

### What OSS can NOT replicate

- **DLSS-RR weights and runtime** — closed model, NGX-only, NVIDIA-trained. No open-weight equivalent exists at production quality for ray-traced denoising.
- **Tensor core ML inference** — DLSS-RR runs on Tensor cores; matching its perf on AMD/Intel requires either WMMA/cooperative-matrix paths with substantially less training compute, or DirectML/ONNX fallbacks at lower throughput.
- **RT-core-accelerated Gaussian ray tracing** — 3DGRT's BLAS/TLAS path uses Vulkan RTX hardware ray tracing; non-RTX vendors have either software RT or weaker HW RT.
- **NGX SDK integration** — closed runtime, license-restricted.

The cross-vendor mandate explicitly excludes this stack.

### What OSS CAN learn from

- **Pipeline staging pattern**: rasterize → optional ray-traced secondary → ML post (denoise+upscale). OSS's existing pixel-temporal v5 pipeline already follows this, and the v6 Gaussian-temporal pipeline can borrow the deferred-shading + depth-consolidation pattern verbatim.
- **Multi-instance splat sets with unified global sort**: a scene-graph idea for OSS-FX once we exceed single-canvas use cases.
- **Stochastic splat sort + Monte-Carlo trace** for interactive frame budgets — relevant to OSS's frame-time budget under canvas churn.
- **Visualization-mode discipline**: "normals, depth, DLSS buffers, splat ID" as debug overlays — adopt the same in OSS for parity-of-debugging-tooling.
- **Asset-format alignment**: `.ply` + `.spz` as the canonical splat-asset I/O. OSS-FX should accept both.

### Concrete benchmark plan

Once OSS DLL-shim runtime exists:

1. **Same scene, same input frame, same target output.** Mip-NeRF 360 bicycle/kitchen/garden subset (already standard) plus one OSS-captured UE5 scene.
2. **Pin the upstream renderer.** Run OSS-shim and `vk_gaussian_splatting` against an identical reconstructed splat asset (`.ply`) at identical camera traces.
3. **Two output configs**: (a) splat-rasterization only, no ML post; (b) splat + ML post (DLSS-RR for NVIDIA, OSS-FX/SR for OSS).
4. **Metrics**: PSNR, SSIM, LPIPS, FLIP (NVIDIA's testbed has FLIP built in), per-frame ms on GPU, p99 ms, VRAM peak.
5. **Honest expected gap**: at config (a) OSS rasterizer should be within ~10% of VK3DGSR. At config (b), OSS-FX/SR likely 2–4 dB PSNR below DLSS-RR on NVIDIA hardware; the headline win is that OSS config (b) **runs on AMD/Intel/mobile** while DLSS-RR returns N/A.

## GaussianVideo (cyberiada): blueprint for OSS-FX α-conditioned rendering

### Architecture

The Bond et al. paper combines 3D Gaussian Splatting with continuous camera motion modeling via Neural ODEs. The state z evolves under the ODE `z(T) = z(0) + ∫_{t0}^{t1} f_θ(z(t), t) dt` where `f_θ` is a learned network parameterizing the dynamics. Intrinsics are held constant; extrinsics (rotation, translation) are evolved by the ODE. Gaussian positions additionally use B-spline trajectories (Eq. 2) evaluated at arbitrary intermediate times. Integration scheme is not stated in the main text — deferred to supplementary, which we did not retrieve. (Flag.)

A spatiotemporal hierarchical learning strategy trains in stages: spatially, start at the coarsest pyramid level until convergence, then descend levels and introduce more Gaussians; temporally, train on every N-th frame and progressively increase temporal resolution. The pyramid descent is applied "once, after 15K training steps, adding 100K new Gaussians."

Frame interpolation at unseen timesteps reuses the same ODE-driven pose at fractional t plus B-spline-evaluated Gaussian positions; the paper shows qualitative results (Fig. 7) but does not derive a closed-form interpolation equation.

### Mapping to OSS-FX

OSS already has `oss/gaussian/extrapolation/extrapolator.py` doing α-conditioned canvas rendering — directly conditioning the canvas at fractional α between two anchor frames. GaussianVideo differs in two ways:

1. **Where the temporal model lives.** OSS conditions the canvas (the post-rasterization Gaussian field at screen space) on α directly. GaussianVideo learns a *camera-trajectory* model, then re-renders 3D Gaussians at the new pose; the temporal model is upstream of rasterization, not at the canvas level.
2. **What the temporal model represents.** GaussianVideo's Neural-ODE is a continuous-time dynamics model — the network learns physics-of-motion-like priors. OSS's α-conditioning is a learned interpolator without an explicit dynamical-system structure.

The OSS-FX-relevant architectural lesson: a continuous-time parameterization (ODE or B-spline) gives **principled extrapolation past the last anchor frame**, not just interpolation between anchors. This matters for v6.0 frame extrapolation (rendering frame t+1 before the engine produces it). OSS's current direct-α approach is an interpolator; for true extrapolation, a dynamics-aware parameterization is at least conceptually preferable.

### Numbers

Reported in Table 1 of the paper:

- **DL3DV**: 43.21 PSNR, 0.99 SSIM, 0.013 LPIPS, ~45 min train.
- **DAVIS**: 37.38 PSNR, 0.96 SSIM, 0.021 LPIPS, ~45 min train.
- **vs baselines on DAVIS**: NeRV 26.15, HNeRV 27.82, Splatter-a-Video 28.63, GaussianImage (per-frame 3DGS) 36.25. GaussianVideo's edge over GaussianImage (+1.13 dB) is attributed to temporal consistency.
- **Project-page headline**: 44.21 PSNR @ 960×540, 93 FPS on NVIDIA A40.

The 93 FPS / A40 number is **inference-time rendering of an already-fit video**, not online frame extrapolation. Training cost is 45 minutes per video on undisclosed hardware. Memory footprint is qualitatively claimed but not quantified. **This is video-encode-quality and cost, not real-time-game-inference cost** — the 93 FPS is achievable because the scene is already fit; what we need from OSS-FX is real-time α-conditioned canvas evaluation, which is a different, smaller problem.

### Specific OSS-FX integration recommendations

- **Do not** wholesale port Neural-ODE camera trajectory: OSS-FX is α-conditioned-canvas, not 3D-camera-trajectory. The ODE prior buys principled extrapolation in 6-DoF camera space; OSS-FX operates downstream of a known camera and only needs to advance the canvas in time.
- **Do** consider porting the **B-spline trajectory parameterization for Gaussian positions** into the canvas representation: cheap per-Gaussian evaluation, naturally supports "evaluate at α ∈ [0, 1+ε]" (extrapolation), and avoids the integration-step cost of an ODE. This is the highest-leverage idea from GaussianVideo for OSS-FX.
- **Do** adopt the **spatiotemporal hierarchical training schedule** for any v6.0 training run that tries to learn an explicit temporal model from UE5 captures. The pyramid + temporal-stride curriculum is cheap to add and is reported to accelerate convergence and improve final quality.
- **Defer** the Neural-ODE direction unless v6.0 frame-extrapolation experiments show OSS's direct-α model failing on extrapolation past the last anchor — at which point the ODE prior becomes worth its compute cost.

## Synthesis: where these two systems leave OSS

Two reference systems, two different roles for OSS to play.

`vk_gaussian_splatting` defines the **competitive ceiling**. NVIDIA has shipped, on a single closed stack, the architectural composition OSS is targeting: splats + ML post for AA/upscale/denoise. The honest read is that OSS will not beat the NVIDIA stack on NVIDIA hardware on quality, and probably not on per-frame latency either. OSS's defensible value is orthogonal: **runs cross-vendor**, **open-pipeline auditability**, **no NGX runtime dependency**, **glTF-compatible asset format alignment**. The existence of NVIDIA's 2026.1 release at GTC accelerates OSS's necessity — there is now a public, named, NVIDIA-published "DLSS-on-splats" stack, which means the OSS question shifts from "is this approach viable?" to "is the open-cross-vendor version of this approach viable?". The benchmark plan above is the path to answering that empirically.

`GaussianVideo` plays a much narrower role: it is a peer reference for the FX subset only. It validates that continuous-time parameterization of Gaussian-based scenes is a productive idea, with strong PSNR (37–44 dB) when applied to fit-then-replay video. But its costs (45 min training, no online extrapolation) and its scope (camera trajectory for known video) are not OSS-FX's problem. The transferable ideas are narrower than the project page suggests: B-spline Gaussian-position trajectories and the spatiotemporal training schedule are cheap to port; Neural-ODE is overkill for OSS-FX's α-conditioned-canvas inference budget.

The unique OSS contribution stays where it was: cross-vendor real-time game super-resolution with Gaussian-temporal architecture, glTF-compatible, open-weight. Neither reference system competes with that pitch directly; vk_gaussian_splatting is single-vendor and viewer-only (no game-engine DLL shim), and GaussianVideo is video-encode-only and per-clip-trained. The opportunity surface is intact.

## Action items

1. **Read `vk_gaussian_splatting` source for DLSS-RR input contract.** Specifically `src/` files that bind DLSS-RR — what is the G-buffer composition, what motion-vector convention, what depth format. Do this without cloning by browsing files via the GitHub web UI or `gh api`. This is the single biggest unknown blocking honest gap analysis.
2. **Add `vk_gaussian_splatting` to the v6 benchmark harness target list.** Specify scenes, camera traces, metrics (PSNR/SSIM/LPIPS/FLIP/ms/VRAM), and the two configurations (rasterization-only vs ML-post). Do not run yet; pin the spec now, run when OSS DLL-shim lands.
3. **Implement B-spline Gaussian-position trajectories in `oss/gaussian/extrapolation/`** as a v6 candidate parameterization. Direct-α stays the baseline; B-spline becomes Variant B; comparison goes through the same A/B harness as v5 pixel-temporal warm-start.
4. **Adopt `.ply` + `.spz` as canonical OSS-FX splat I/O.** Aligns with NVIDIA's testbed assets and Niantic's spz, makes head-to-head comparison trivial.
5. **Do not pursue Neural-ODE camera trajectory for OSS-FX in this sprint.** Re-evaluate only if direct-α and B-spline both fail extrapolation tests in v6.0.
6. **Read the GaussianVideo supplementary material for the Neural-ODE integration scheme.** Solver choice (Euler, RK4, adaptive) is needed before any future port; flagged as outstanding.

## Outstanding questions

- What exactly does `vk_gaussian_splatting`'s DLSS-RR pass receive as input (G-buffer layout, motion-vector convention, noisy-radiance encoding, depth format)? Required reading: `src/`. Not retrievable from README/blog/project-site.
- What is the 2026.1 RT+DLSS path's per-frame cost on a defined scene (e.g. Mip-NeRF 360 bicycle) on a defined GPU (e.g. RTX 5090)? Public sources do not state it.
- What ODE solver does GaussianVideo use, and what is the per-frame cost of evaluating the ODE at inference time? Deferred to supplementary.
- Does GaussianVideo's claim of arbitrary-timestep interpolation extend to extrapolation beyond `t_max`? The paper shows interpolation within training range; extrapolation is conceptually possible but not empirically demonstrated.

## References

- NVIDIA, "Vulkan Gaussian Splatting" (`nvpro-samples/vk_gaussian_splatting`), GitHub, 2026.1 release. https://github.com/nvpro-samples/vk_gaussian_splatting
- NVIDIA, "Real-Time GPU-Accelerated Gaussian Splatting with NVIDIA DesignWorks Sample vk_gaussian_splatting," NVIDIA Technical Blog. https://developer.nvidia.com/blog/real-time-gpu-accelerated-gaussian-splatting-with-nvidia-designworks-sample-vk_gaussian_splatting/
- "NVIDIA Releases Vulkan Gaussian Splatting 2026.1," radiancefields.com. https://radiancefields.com/nvidia-releases-vulkan-gaussian-splatting-2026.1
- NVIDIA Toronto Lab, `nv-tlabs/3dgrut` (Ray tracing and hybrid rasterization of Gaussian particles). https://github.com/nv-tlabs/3dgrut
- Moënne-Loccoz et al., "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes," SIGGRAPH Asia 2024. (3DGRT)
- Wu et al., "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting," CVPR 2025. (3DGUT)
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," SIGGRAPH 2023. (foundation 3DGS)
- Bond, Wang, Mai, Erdem, Erdem, "GaussianVideo: Efficient Video Representation via Hierarchical Gaussian Splatting," arXiv:2501.04782, January 2025. https://arxiv.org/abs/2501.04782
- GaussianVideo project page: https://cyberiada.github.io/GaussianVideo/
