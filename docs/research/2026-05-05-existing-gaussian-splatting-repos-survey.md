# Obscure but Active Gaussian-Splatting Repos for Spatio-Temporal Super-Resolution and Adjacent Topics

## TL;DR
- The single most relevant new repo for your **primary** interest (spatio-temporal super-resolution with covariance resampling) is **GS-STVSR** by Mingyu Shi et al. — code is announced/being released through the *awesome-gaussians* tracker as of 2026, and it explicitly uses 2D-Gaussian covariance resampling for continuous spatio-temporal upsampling. Pair it with **KeKsBoTer/upscale3dgs** (gradient-aware temporal-stable upscaling, ICCV 2025) and **SplatSuRe / Sequence-Matters / S2Gaussian / SuperGS** for the broader 3DGS-SR family — all of these are sub-500-star, recently active, and going somewhere different from each other.
- For your **secondary** interests, the highest-signal obscure-but-active picks are: **JasonLSC/GSCodec_Studio** (modular static+dynamic GS compression), **qianghu-huber/4DGC** + **MediaX-SJTU/4DGCPro** (rate-aware streamable 4D codecs, CVPR/NeurIPS 2025), **DerThomy/AAA-Gaussians** (anti-alias + anti-pop full-3D rasterizer), **maeyounes/AA-2DGS** (2D-GS Mip-style anti-aliasing, NeurIPS 2025), **wuyize25/gsplat-unity** + **HiFi-Human/DynGsplat-unity** (modern Unity static + dynamic plugins), **JI20/unreal-splat** and **TimChen1383/NanoGaussianSplatting** (UE5 plugins from solo devs), **DazaiStudio/SplatRenderer-UEPlugin** (3D+4D UE5.5+), **PKU-VCL-Geometry/GeoSplatting** and **stopaimme/GI-GS** (PBR/inverse rendering), and **LonganWANG-cs/GSVC** + **Xinjie-Q/MEGA** (compression). For ray-traced GS the active mainline is NVIDIA's **3DGRUT / vk_gaussian_splatting** stack.
- Caveat: GitHub star counts shift quickly, the GS-STVSR public repo URL was not yet visible in our search results (only the paper page was — code "to be released"), and several of these projects (especially the UE5 and 4DGS plugin space) are very new and rough; treat the curated list below as a starting set rather than a vetted production shortlist.

---

## Key Findings

The Gaussian-Splatting research frontier in 2025–2026 has fragmented along the exact axes the user cares about. Three observations matter for prioritization:

1. **Spatio-temporal super-resolution is finally a real subfield, not a one-off paper.** GS-STVSR (covariance-resampling, 2D-GS-driven C-STVSR) is the first dedicated spatio-temporal SR method built on Gaussian primitives; alongside it, Sequence-Matters, S2Gaussian, SplatSuRe, MVGSR, 3DSR, IE-SRGS, and SuperGS form a coherent "3DGS-SR" cluster, all surveyed in the late-2025 SRGS paper (arXiv 2404.10318). Most are individual lab repos in the 30–500 star range — exactly the obscurity profile the user asked for.
2. **The "novel primitive" track and the "engine plugin" track are diverging.** Research repos (AAA-Gaussians, Analytic-Splatting, AA-2DGS, 4D-Rotor) are publishing rasterizer-level changes; meanwhile, solo-dev plugin authors (wuyize25, JI20, TimChen1383, DazaiStudio, mlslabs) are mostly racing against aras-p's Unity reference and Luma's UE asset. The plugin authors have *not* meaningfully picked up the new rasterizer research yet — this is a gap.
3. **Compression / streaming codecs have consolidated around two pipelines:** rate-aware end-to-end RD optimization (4DGC, 4DGCPro, HAC++, FCGS, GSCodec_Studio) and 2D-Gaussian video coding (GSVC). Both are 2025 work, both have public code, and almost none have crossed 500 stars.

Below, repos are organized in your requested format with super-resolution first.

---

## Details

### A. PRIMARY — Spatio-Temporal Super-Resolution & Temporal/Spatial Upscaling for Gaussians

**1. GS-STVSR — Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting**
- URL: paper at arXiv 2604.18047 (listed in `longxiang-ai/awesome-gaussians`); official GitHub link not yet surfaced in search snippets but the awesome-gaussians tracker indicates the repo is being released (date 2026).
- Authors: Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha (USTC + Huawei Noah's Ark group).
- Why it's the headline pick: First C-STVSR framework to drive 2D-GS spatio-temporal evolution via continuous motion modeling, bypassing INR dense grid queries entirely. Introduces a **Covariance Resampling Alignment** module (literally what the user named) for stable covariance prediction across time, plus optical-flow-guided motion learning with adaptive offset windows. Reports SOTA on Vid4, GoPro, and Adobe240 with >10× speedup over INR baselines.
- Category: Primary (spatio-temporal SR with covariance resampling).
- Action: Watch the awesome-gaussians tracker entry / search "GS-STVSR" weekly until the repo URL is published; this is the single highest-signal repo for your primary criterion.

**2. KeKsBoTer/upscale3dgs — Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images (ICCV 2025)**
- URL: https://github.com/KeKsBoTer/upscale3dgs
- Stars: low (sub-100, single-author repo by Simon Niedermayr, TUM); active 2025.
- Language: Python + WebGPU (Brush-based viewer demo).
- Novelty: Uses the analytical image-space gradients of Gaussians for **bicubic spline interpolation upscaling** with low overhead; reports 3×–4× rendering speedup vs full-res 3DGS, training-time integration improves reconstruction. Explicitly designed for *temporally stable* upscaling on lightweight GPUs — directly relevant to sub-pixel rendering for VR.
- Category: Primary (spatial upscaling; explicit temporal stability claim).
- Affiliation: Niedermayr / Neuhauser / Westermann, TU Munich.

**3. ChrisDud0257/GSASR — Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution (ICCV 2025)**
- URL: https://github.com/ChrisDud0257/GSASR
- Stars: a few hundred (still under 500 in our snapshot).
- Language: Python + custom CUDA scale-aware 2D rasterizer.
- Novelty: Feed-forward predicts millions of image-conditioned 2D Gaussians for *arbitrary* (incl. ×6, ×12 OOD) scaling factors; CUDA differentiable rasterization that samples discrete RGB from continuous Gaussians; supports HAT-L encoders and ROPE/Flash-Attention enhanced variant. Active (online demo released June 2025, weights for EDSR/RDN/SwinIR/HAT-L backbones May 2025).
- Category: Primary (spatial SR via GS primitives — closest mature relative to GS-STVSR).
- Affiliation: HK PolyU (Lei Zhang group).

**4. tljxyys/GaussianSR — High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution (AAAI 2025)**
- URL: https://github.com/tljxyys/GaussianSR
- Stars: low. Recent activity present.
- Novelty: Per-pixel continuous Gaussian field for ASSR with a *classifier* that dynamically assigns Gaussian kernels per pixel; long-range dependencies via mutually-stacked Gaussian fields. End-to-end (encoder, classifier, kernels, decoder).
- Category: Primary (single-image GS SR).
- Note: Different "GaussianSR" from the SDS-prior 3D one — both are listed because they take genuinely different approaches.

**5. XiangFeng66/SRGS / SR-GS — Super-Resolution 3D Gaussian Splatting**
- URL: https://github.com/XiangFeng66/SRGS (also SR-GS new branch)
- Stars: low (<200 historically).
- Novelty: First paper to focus on HRNVS in 3DGS; recently re-formalized (arXiv 2404.10318 v-update late 2025) as a *unified modular framework* for 3DGS-SR factorizing the problem into prior injection + cross-view regularization. Acts as the canonical baseline now compared against GaussianSR / Sequence-Matters / MVGSR / 3DSR / S2Gaussian / IE-SRGS / SplatSuRe.
- Category: Primary (3DGS-SR baseline + modular toolbox).

**6. Ko-Lani/Sequence-Matters — Harnessing Video Models in Super-Resolution (AAAI 2025)**
- URL: https://github.com/Ko-Lani/Sequence-Matters
- Stars: very low (single-digit org; "Striving for life" solo profile, 9 followers).
- Novelty: Adaptive-Length-Sequencing (ALS) — greedily orders raw LR multi-view images into pseudo-video subsequences via feature/pose similarity, then applies *off-the-shelf* video super-resolution (no fine-tuning) to drive 3DGS training. Achieves SOTA on NeRF-synthetic and MipNeRF-360 by avoiding the LR-3DGS-render artifacts that hurt SuperGaussian.
- Category: Primary (uses VSR temporally to drive 3DGS-SR — directly bridges super-resolution and temporal upsampling).
- Affiliation: Sungkyunkwan / Yonsei (Eunbyung Park group).

**7. jeasco / Wan et al. — S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting (CVPR 2025)**
- URL: https://jeasco.github.io/S2Gaussian/ (project page; code linked from there). Authors: Yecong Wan, Mingwen Shao, Yuanshuo Cheng, Wangmeng Zuo (China Univ. of Petroleum + HIT).
- Stars: low.
- Novelty: Two-stage (LR-GS densify → "Gaussian Shuffle Split" → HR-GS optimization) with blur-free inconsistency modeling and 3D robust optimization — handles the *combined* sparse-view + LR scenario explicitly. Establishes new SOTA on this hybrid setting.
- Category: Primary (sparse + low-res joint upscaling — a niche the user implicitly cares about).

**8. SplatSuRe — Selective Super-Resolution for Multi-view Consistent 3DGS (arXiv 2512.02172, Dec 2025)**
- URL: https://splatsure.github.io (code linked).
- Authors: Pranav Asthana, Alex Hanson, Allen Tu, Tom Goldstein, Matthias Zwicker, Amitabh Varshney (UMD).
- Novelty: Geometry-aware *selective* SR — applies 2D SR only in undersampled regions lacking high-frequency supervision. Explicitly attacks the multi-view inconsistency that uniform-SR pipelines (SRGS-style) introduce. Beats SRGS, GaussianSR, SuperGS, S2Gaussian, MVGSR, 3DSR, IE-SRGS, Sequence-Matters on Tanks & Temples / Deep Blending / Mip-NeRF 360.
- Category: Primary. Very recent — likely sub-100 stars.

**9. SuperGS (Beihang) — Latent Feature Field + Gradient-guided Splitting**
- URL: see arXiv 2410.02571; code referenced; Beihang group (Pancw).
- Novelty: Multi-resolution Feature Gaussian Splatting (MFGS) with latent feature field + Gradient-guided Selective Splitting (GSS) for upsampling Gaussian primitives during HR optimization. Coarse-to-fine.
- Category: Primary.

**10. adobe-research/SuperGaussian — Repurposing Video Models for 3D Super Resolution (ECCV 2024)**
- URL: https://github.com/adobe-research/SuperGaussian
- Stars: under a few hundred.
- Novelty: Uses pretrained *video* upsampling priors (RealBasicVSR / VideoGigaGAN / Upscale-a-Video) to upsample rendered novel views, then refits Gaussians. Modular, category-agnostic, accepts NeRF / GS / mesh / RGB-D inputs. Includes MVImgNet eval scripts.
- Category: Primary. Caveat: Adobe Research may push it into "well-known" territory soon, but the GitHub repo itself remains relatively quiet.

**11. autonomousvision/mip-splatting and forks (anti-alias / Mip-style filtering)**
- URL: https://github.com/autonomousvision/mip-splatting (well-known, but listed because…)
- More obscure variants worth tracking instead/alongside:
  - **lzhnb/Analytic-Splatting** (ECCV 2024 Oral) — analytical pixel-area integral for AA: https://github.com/lzhnb/Analytic-Splatting (low-hundred stars).
  - **DerThomy/AAA-Gaussians** (ICCV 2025) — adaptive 3D smoothing filter + view-space frustum bounding; **eliminates popping artifacts** and full-3D-evaluated rasterizer (built on StopThePop). https://github.com/DerThomy/AAA-Gaussians.
  - **maeyounes/AA-2DGS** (NeurIPS 2025) — Anti-Aliased 2D Gaussian Splatting with world-space flat smoothing kernel + object-space Mip filtering. https://github.com/maeyounes/AA-2DGS (sub-50 stars; "soon to release" code).
  - **renaissanceee/Mipmap-GS** — plug-in scale-consistency loss for any 3DGS, +9.25 dB zoom-in / +10.40 dB zoom-out PSNR on NeRF-synthetic.
- Category: Primary (anti-aliasing / temporal stability for splats).

### B. SECONDARY — 4D / Temporal Gaussian Splatting (deformation fields, native 4D, rotor-based)

**12. Xinjie-Q/MEGA — Memory-Efficient 4D Gaussian Splatting (ICCV 2025)**
- URL: https://github.com/Xinjie-Q/MEGA. Low stars.
- Novelty: Decomposes color into 3-param per-Gaussian DC + shared lightweight AC predictor (drops 144-coef SH), entropy-constrained deformation field expanding per-Gaussian action range. Aggressive memory reduction.

**13. weify627/4D-Rotor-Gaussians — 4D-Rotor Gaussian Splatting (SIGGRAPH 2024)**
- URL: https://github.com/weify627/4D-Rotor-Gaussians. Low stars.
- Novelty: Native 4D XYZT Gaussians with **geometric-algebra rotor** rotation representation (the rotor approach the user specifically asked about). Temporal slicing + CUDA splatting; up to 583 FPS on RTX 4090. PKU group.

**14. Chenwei-Liang/CoDa-4DGS — Context- and Deformation-Aware 4DGS for Autonomous Driving (ICCV 2025)**
- URL: https://github.com/Chenwei-Liang/CoDa-4DGS.
- Novelty: Deformation Compensation Network on top of vanilla 4DGS HexPlane encoding; embeds semantic + temporal deformation features per-Gaussian.

**15. hustvl/TOGS — Temporal Opacity Offset for Real-Time 4D DSA Rendering (IEEE JBHI 2025)**
- URL: https://github.com/hustvl/TOGS. Niche/medical, low stars, recent.
- Novelty: Per-Gaussian opacity offset table (interpolated over time) instead of full 4D field — a clever lightweight temporal extension. Clinical 4D DSA imaging.

**16. SJoJoK/3DGStream — On-the-fly 4D streaming (CVPR 2024 Highlight)**
- URL: https://github.com/SJoJoK/3DGStream. Stars likely a few hundred.
- Novelty: Per-frame on-the-fly training in ~12s, 200 FPS rendering, Neural Transformation Cache (tiny-cuda-nn) for translations + rotations + adaptive 3DG addition for emerging objects.

**17. yindaheng98/TrackerSplat — Point-Tracking-Driven Dynamic 3DGS (arXiv 2604.02586)**
- URL: https://github.com/yindaheng98/TrackerSplat. Very recent, low stars.
- Novelty: Pre-positions Gaussians via point tracking before gradient descent, eliminates fading/recoloring under large inter-frame displacement. Parallel multi-device throughput.

**18. cyberiada / GaussianVideo (Bond et al., Koç + Adobe + Hacettepe)**
- URL: https://cyberiada.github.io/GaussianVideo/ (code linked). Low stars.
- Novelty: Neural-ODE camera trajectory + 3DGS for video; explicit **frame interpolation at arbitrary timesteps** (relevant to your temporal upscaling ask) and arbitrary spatial resampling. 44.21 PSNR @ 960×540, 93 FPS A40.

**19. jiayi1129/GaussianVideo — alternative implementation**
- URL: https://github.com/jiayi1129/GaussianVideo. Solo dev; explicit "first step" caveat (does not yet match HNeRV / AV1 RD).

### C. SECONDARY — Engine Integration (Unity / Unreal / Bevy / Godot)

**20. wuyize25/gsplat-unity** — modern PlayCanvas-style transparent-mesh-integrated Unity package; supports BiRP/URP/HDRP; correctly blends with transparent meshes via bounding boxes. https://github.com/wuyize25/gsplat-unity. Sub-500 stars.

**21. HiFi-Human/DynGsplat-unity** — *Dynamic*-3DGS sequence player for Unity, depends on gsplat-unity. Includes a PLY-sequence compression pipeline. https://github.com/HiFi-Human/DynGsplat-unity. Companion paper RePerformer (CVPR 2025).

**22. JI20/unreal-splat** — solo-dev UE5.5 plugin via Niagara, up to 2M splats. https://github.com/JI20/unreal-splat. Few stars.

**23. TimChen1383/NanoGaussianSplatting** — Nanite-style LOD clusters + screen-space-error LOD selection + GPU radix sort for *large-scale* GS in UE 5.6/5.7. Solo-dev, novel approach. https://github.com/TimChen1383/NanoGaussianSplatting.

**24. DazaiStudio/SplatRenderer-UEPlugin** — 3D + 4D GS renderer for UE 5.5+, Sequencer keyframe support, 4DGS converter to .gsd. https://github.com/DazaiStudio/SplatRenderer-UEPlugin.

**25. mlslabs/MLSLabsGaussianSplattingRenderer-UE** — high-perf 3DGS+4DGS UE5 plugin, custom non-Niagara pipeline, "millions of Gaussians" claim. https://github.com/mlslabs/MLSLabsGaussianSplattingRenderer-UE.

**26. Italink/GaussianSplattingForUnrealEngine** — *training-side* UE plugin: builds camera arrays inside the editor, runs COLMAP + 3DGS over UE primitives. https://github.com/Italink/GaussianSplattingForUnrealEngine.

**27. YHK-UEPlugins-Public/018_UEGaussianSplatting_Public** — octree LOD + **automatic collision generation** (this is the closest existing answer to your "collision systems" ask). https://github.com/YHK-UEPlugins-Public/018_UEGaussianSplatting_Public.

**28. xverse-engine/XScene-UEPlugin** — XVERSE Tech UE5 plugin with hybrid mesh-Gaussian rendering and editing.

**29. mosure/bevy_gaussian_splatting** — Bevy plugin in Rust supporting 2DGS / 3DGS / 4DGS, gltf Gaussian extensions, planned 4DGS motion blur, deformable radial kernels, implicit MLP nodes, temporal hierarchy. v3.0.0 (Bevy 0.15) Dec 2025. ~170 stars. https://github.com/mosure/bevy_gaussian_splatting.

**30. 2Retr0/GodotGaussianSplatting** — compute-shader rasterizer for Godot via RenderingDevice. Solo dev. https://github.com/2Retr0/GodotGaussianSplatting.

**31. keijiro/SplatVFX** — Unity VFX-Graph based renderer (limited; useful for particle-effect hybrids). https://github.com/keijiro/SplatVFX.

### D. SECONDARY — Compression & Streaming Codecs

**32. JasonLSC/GSCodec_Studio** — modular static + dynamic GS compression framework built on gsplat; integrates MPEG GSC tooling (PCC + video-codec wrappers); bench scripts under `examples/benchmarks/mpeg`. https://github.com/JasonLSC/GSCodec_Studio.

**33. qianghu-huber/4DGC + zihanzheng-sjtu/4DGC** — Rate-Aware 4D Gaussian Compression (CVPR 2025). Motion grid + sparse compensated Gaussians + entropy-optimized end-to-end RD; ~16× compression over 3DGStream. https://github.com/qianghu-huber/4DGC.

**34. MediaX-SJTU/4DGCPro** — Hierarchical 4DGS Compression for *progressive volumetric video streaming* on **mobile** (NeurIPS 2025). Single-bitstream multi-bitrate. https://github.com/mediax-sjtu/4DGCPro.

**35. LonganWANG-cs/GSVC** — 2D Gaussian Splatting video codec, NOSSDAV 2025. I-frame + P-frame, 1500 FPS rendering at 1920×1080, RD competitive with HEVC/AV1. **34 stars, 1417 commits** as of fetch. https://github.com/LonganWANG-cs/GSVC.

**36. YihangChen-ee/FCGS** — Fast Feedforward 3DGS Compression (ICLR 2025), 1/10 the time of optimization-based codecs, includes CUDA arithmetic codec. https://github.com/YihangChen-ee/FCGS.

**37. YihangChen-ee/HAC-plus (HAC++)** — towards 100× 3DGS compression (TPAMI 2025). https://github.com/YihangChen-ee/HAC-plus.

**38. fraunhoferhhi/CodecGS** — feature-plane compression compatible with **standard video codecs (HEVC)** via DCT-entropy loss. Project page links to repo.

**39. H-Huang774/ADC-GS** — anchor-driven deformable + compressed GS (IJCAI 2025), 300–800% rendering speedup over per-Gaussian deformation. https://github.com/H-Huang774/ADC-GS.git.

### E. SECONDARY — Relighting / PBR / Ray-traced Gaussians

**40. stopaimme/GI-GS** — Global Illumination Decomposition on GS (ICLR 2025). Differentiable PBR + path tracing for indirect light + deferred shading. https://github.com/stopaimme/GI-GS.

**41. PKU-VCL-Geometry/GeoSplatting** — geometry-guided GS for physically-based inverse rendering (ICCV 2025). Surface-based representation + Gaussians. https://github.com/PKU-VCL-Geometry/GeoSplatting.

**42. fudan-zvg/IRGS** — Inter-Reflective GS with **fully differentiable 2D Gaussian ray tracing** (CVPR 2025). Explicit visibility/indirect-radiance via Monte Carlo sampling. (Repo linked from Fudan-ZVG project page.)

**43. nju-3dv/Relightable3DGS** — original relightable BVH-based point ray tracing for GS, BRDF + incident-light decomposition, real-time shadows.

**44. NVlabs/svraster** — Sparse Voxels Rasterization (CVPR 2025) — voxel-based but adjacent and useful when crossing into ray tracing.

**45. nvpro-samples/vk_gaussian_splatting** — NVIDIA Vulkan testbed: implements **3DGRT** (3D Gaussian Ray Tracing), **3DGUT** (Unscented Transform), DLSS Ray Reconstruction integration for **anti-aliasing + upscaling + denoising** of splats, hybrid 3DGS/3DGRT pipelines with depth of field. https://github.com/nvpro-samples/vk_gaussian_splatting. (NVIDIA-published, but the splat-DLSS integration is exactly the sub-pixel/temporal-stability angle the user wants and not yet widely cited.)

### F. SECONDARY — Mesh-Gaussian Hybrids

**46. Anttwo/Frosting** — Gaussian Frosting (ECCV 2024 Oral). Variable-thickness Gaussian "frosting" layer over a base mesh; Blender add-on for editing. https://github.com/Anttwo/Frosting. Mid-stars.
**47. Anttwo/SuGaR** — surface-aligned Gaussians + Poisson mesh extraction. https://github.com/Anttwo/SuGaR.

---

## Recommendations

**Stage 1 — Immediate (this week).** Track GS-STVSR's public release: subscribe to issues/watch on the awesome-gaussians tracker, check `Mingyu Shi` author page weekly. In parallel, clone and benchmark **upscale3dgs**, **Sequence-Matters**, and **GSASR** on a representative VR-style capture. These three sit at distinct points on the spatio-temporal SR axis (rendering-time spline upscale, dataset-side VSR-driven SR, feed-forward arbitrary-scale SR) and together define the design space.

**Stage 2 — Within a month.** If your application is VR/high-refresh sub-pixel rendering: prototype **upscale3dgs + AAA-Gaussians + AA-2DGS** as a stack — you'll have analytical-gradient bicubic upscaling, alias-free + popping-free rasterization, and 2D-GS object-space Mip filtering composed. Benchmark against vk_gaussian_splatting's DLSS-RR pipeline as a non-GS-native upper bound. If your application is volumetric video: stack **4DGCPro + DynGsplat-unity (or DazaiStudio's UE plugin)** for end-to-end mobile-decodable streamable 4DGS.

**Stage 3 — If you need novel research directions.** GS-STVSR's covariance resampling has obvious unfinished business — nobody has yet combined it with a learned deformation field (à la 4D-Rotor or MEGA) for 4D temporal upsampling beyond 2D-GS. That's a viable gap; the building blocks exist in `weify627/4D-Rotor-Gaussians` + the GS-STVSR module once it lands.

**Thresholds that would change the recommendation.**
- If GS-STVSR's covariance-resampling code does not appear publicly within 90 days, fall back to GSASR + Sequence-Matters as the primary stack.
- If `upscale3dgs` does not push commits past mid-2026, replace it with the rendering-side filter from AAA-Gaussians (which is more actively maintained).
- If the user's project requires hard real-time on consumer VR, the *only* current option that hits the latency budget without retraining is `vk_gaussian_splatting` with DLSS-RR; everything else is research code.
- Re-prioritize compression repos toward **GSCodec_Studio** if you need a single experiment harness — its modularity dominates the others. Use **4DGC / 4DGCPro / GSVC** as point comparisons rather than backbones.

---

## Caveats

1. **Star counts are point-in-time and shift fast** in this field. Several repos listed as "low stars" (GSASR, mosure/bevy_gaussian_splatting, GIFStream) may cross 500 stars within months as their associated papers get cited at ICCV/CVPR. We did not confirm star count for every entry; treat sub-500 as the working assumption only where we explicitly measured it (e.g., LonganWANG-cs/GSVC = 34 stars; mosure/bevy = ~170 stars). Verify star count in your own filter before adopting.
2. **GS-STVSR public repo URL was not directly visible** in search results — only the paper page and the awesome-gaussians tracker entry. The repo is announced ("Code:" listed by the tracker); you will need to revisit shortly to grab the URL. Until then, GSASR + upscale3dgs are the closest extant stand-ins for the covariance-resampling spatio-temporal SR niche.
3. **Activity verification was best-effort.** Repos like 2Retr0/GodotGaussianSplatting, JI20/unreal-splat, jiayi1129/GaussianVideo are clearly solo-developer projects with limited activity histories — they meet the "obscure but recent" bar but may go inactive on short notice. The compression cluster (4DGC, 4DGCPro, GSCodec_Studio, FCGS, HAC++) is more institutionally backed and lower-risk to depend on.
4. **Some entries are arXiv-stage with code "to be released"** (notably AA-2DGS, GS-STVSR, parts of SplatSuRe, IRGS' 2D ray-tracing variant). The category placements above are based on README/paper claims, not on independent reproduction; benchmark numbers are author-reported.
5. **The user's exclusions were respected** — graphdeco-inria/gaussian-splatting, aras-p/UnityGaussianSplatting, mkkellogg/GaussianSplats3D, antimatter15/splat are referenced only contextually, never as recommendations. The well-known autonomousvision/mip-splatting is mentioned only as the parent for the more obscure variants (Analytic-Splatting, AAA-Gaussians, AA-2DGS, Mipmap-GS) which *are* the actual recommendations.
6. **No single repo matches every criterion.** GS-STVSR is the only one explicitly named-and-shaped around "spatio-temporal super-resolution + covariance resampling for Gaussians"; everything else is a useful adjacent piece. If that paper's code never lands at the quality you need, you will be assembling the missing functionality from upscale3dgs (rendering-side), Sequence-Matters (data-side temporal prior), and 4D-Rotor-Gaussians (native 4D primitive) yourself.