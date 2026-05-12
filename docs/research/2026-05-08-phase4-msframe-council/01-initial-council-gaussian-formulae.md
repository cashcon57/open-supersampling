# Research Report: Novel Gaussian Formulae for 2D Canvas-Based Real-Time Upscaling \& Frame Extrapolation

## 1. Where Models Agree

| Finding | GPT-5.5 Thinking | Claude Opus 4.7 Thinking | Gemini 3.1 Pro Thinking | Evidence |
| :-- | :-- | :-- | :-- | :-- |
| GS-STVSR is your closest prior art for temporal Gaussian evolution | ✓ | ✓ | ✓ | Optical flow-guided motion module drives Gaussian position/color at arbitrary timesteps; covariance resampling prevents drift[^1][^2] |
| GSASR's scale-aware rasterization is directly relevant | ✓ | ✓ | ✓ | Feed-forward Gaussian prediction from LR features + CUDA rasterizer achieving 91ms at ×12[^3][^4] |
| ContinuousSR's Deep Gaussian Prior (DGP) can improve your spawner | ✓ | ✓ | ✓ | 99% of covariances fall in narrow ranges; pre-defined kernel dictionaries with adaptive weighting avoid local optima[^5][^6] |
| Anti-aliased 2DGS's object-space Mip filter is applicable to your rasterizer | ✓ | ✓ | ✓ | Σ'_local = I + σJJ^T maps screen-space filtering to object space via ray-splat Jacobian[^7][^8] |
| Mip-Splatting's frequency-band-limiting is critical for multi-scale rendering | ✓ | ✓ | ✓ | V_eff = V + σ²_smooth·I with opacity modulation α_smooth = α·(s_u·s_v)/√((s_u²+σ²)(s_v²+σ²))[^9][^10] |
| GaussianImage's accumulated summation (no alpha-blending sort) validates your sum-composite approach | ✓ | ✓ | ✓ | C_i = Σ_n c'_n · exp(-σ_n) — order-independent, no T_n computation, 2000 FPS[^11][^12] |

## 2. Where Models Disagree

| Topic | GPT-5.5 Thinking | Claude Opus 4.7 Thinking | Gemini 3.1 Pro Thinking | Why They Differ |
| :-- | :-- | :-- | :-- | :-- |
| Best covariance update for temporal warp | Σ' = JΣJ^T + Δt·D (your current form is optimal) | Fourier-basis temporal decomposition from Cross-Temporal 3DGS may be more stable | Covariance resampling alignment from GS-STVSR is needed to prevent drift | Different emphasis on stability vs expressivity vs interpretability |
| Whether DGP-driven covariance weighting helps at training or inference | Useful at spawner initialization time | Better as a regularization loss term constraining covariance range | Should replace direct covariance regression entirely with weighted kernel dictionary | Different assumptions about your optimization landscape |
| Relevance of Spacetime Gaussians' temporal opacity | Not relevant—your canvas is persistent | Temporal opacity σ(t) = σ^s·exp(-s^τ | t-μ^τ | ²) could gate stale Gaussians for auto-pruning |

## 3. Unique Discoveries

| Model | Unique Finding | Why It Matters |
| :-- | :-- | :-- |
| GPT-5.5 Thinking | EWA Motion Blur (Hein et al. 2010): extends EWA with 3D spatio-temporal kernels unifying spatial+temporal components for moving objects[^13] | Directly applicable to your temporal warp — instead of discrete per-frame advection, the kernel itself can be time-extended |
| Claude Opus 4.7 Thinking | Kalman filtering on Gaussian state with optical-flow velocity updates (KOFT)[^14] | Your per-Gaussian velocity field is essentially a prediction step; adding a Kalman correction step when new evidence arrives could reduce drift |
| Gemini 3.1 Pro Thinking | AA-2DGS's world-space flat smoothing projects isotropic 3D low-pass onto the splat plane: V_eff = diag(s_u² + σ²_smooth, s_v² + σ²_smooth)[^8] | Directly prevents your spawner from creating sub-pixel Gaussians that alias at HR output resolution |

## 4. Comprehensive Analysis

### High-Confidence Findings

The research council unanimously identifies **GS-STVSR** (April 2026) as your most direct competitor and source of equations. Its core contribution — driving Gaussian kernel evolution through optical flow-guided motion while keeping covariance parameters temporally stable via a resampling alignment module — maps almost exactly onto your temporal warp. The key equation you should examine is their covariance resampling: rather than allowing Σ to accumulate Jacobian-induced distortion indefinitely, they periodically realign covariance parameters to a canonical form based on the local scale of the output. This addresses your stated concern about "compounding non-determinism" over multi-frame extrapolation.[^1][^2]

All three models agree that **ContinuousSR's Deep Gaussian Prior** offers immediate value for your spawner's bias problem. The statistical finding that 99% of natural-image Gaussian covariances fall within σ²_x ∈ [0, 2.4], σ²_y ∈ [0, 2.2], ρσ_xσ_y ∈ [-0.9, 1.5] provides a principled initialization range. Their DGP-Driven Covariance Weighting replaces direct covariance regression with:[^5][^6]

$G_{\text{target}} = \sum_{i=1}^{N} w_i \cdot G_i, \quad \mathbf{W} = \text{Softmax}(\mathcal{M}_{\text{weight}}(\mathcal{F}_{\text{LR}}))$

where {G_i} are sampled from the DGP distribution. This could replace or augment your spawner's Δscale/Δrot regression, potentially eliminating the checkerboard artifact by ensuring spawned Gaussians conform to natural statistics rather than collapsing to a tight fractional bias.[^6]

The **accumulated summation rasterization** from GaussianImage validates your sum-composite EWA approach. Their equation C_i = Σ_n c'_n · exp(-σ_n) is mathematically equivalent to your out[c,py,px] = Σ_g exp(-½·q_g) · feat[g,c], confirming that sort-free order-independent splatting achieves 2000+ FPS with competitive quality. Their key ablation shows a 0.8 dB PSNR improvement over alpha-blending when depth ordering is unknown — directly applicable to your 2D canvas where no canonical depth exists.[^12]

### Areas of Divergence

The most substantive disagreement concerns **how to handle temporal covariance evolution**. GPT-5.5 Thinking endorses your current Σ' = JΣJ^T + Δt·D as theoretically sound, while Claude Opus 4.7 Thinking points to Cross-Temporal 3DGS's Fourier-basis decomposition where μ(t) and R(t) are modeled as smooth functions of time while scale/opacity remain invariant. The Fourier approach offers guaranteed smoothness but sacrifices the ability to model sudden scale changes (e.g., objects approaching camera). Gemini 3.1 Pro Thinking's recommendation of GS-STVSR's covariance resampling is a pragmatic middle ground — apply your J-based warp but periodically snap covariances back to well-conditioned forms.[^2][^15]

The **Hein et al. 2010 spatio-temporal EWA kernel** is a particularly interesting find by GPT-5.5 Thinking. Rather than treating motion as a discrete per-frame position update, they extend the 2D Gaussian kernel into a 3D spatio-temporal kernel where the temporal dimension encodes motion blur. For your frame extrapolation use case, this means a single Gaussian evaluation could produce not just the current frame but an analytically motion-blurred intermediate — potentially useful for sub-frame interpolation without separate warp passes. The formula unifies spatial reconstruction and temporal filtering into one kernel evaluation.[^13]

### Unique Insights Worth Noting

The **Kalman filtering insight** from Claude Opus 4.7 Thinking deserves serious consideration. Your per-Gaussian velocity field is a prediction model; when a new rendered frame arrives, you currently rely entirely on the spawner to correct drift. A Kalman update step would provide:[^14]

$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t(z_t - H\hat{x}_{t|t-1})$

where z_t is the observed position from the new frame's motion vectors and K_t is the Kalman gain. This adds a principled correction mechanism between the warp (prediction) and spawner (observation), potentially extending your viable extrapolation horizon beyond 1-2 frames.

The **AA-2DGS object-space Mip filter** equation Σ'_local(x) = I + σJJ^T is directly implementable in your rasterizer. The Jacobian J = ∂u/∂x of the pixel-to-Gaussian-space mapping is cheap to compute (you already have the conic parameters), and the resulting modified quadratic form prevents aliasing at output resolution without requiring explicit anti-aliasing passes. This is particularly relevant for your ×4 upscaling scenario where a single LR Gaussian may span only a fraction of an HR pixel.[^8]

### Recommendations

For immediate v6.2 integration, prioritize: (1) ContinuousSR's DGP-driven covariance weighting for the spawner to fix your checkerboard artifact with a principled statistical prior rather than just jitter; (2) GS-STVSR's covariance resampling alignment to extend extrapolation horizon; (3) AA-2DGS's object-space Mip filter (Σ'_local = I + σJJ^T) in your rasterizer to prevent aliasing at HR output. For v6.3, investigate the Kalman correction formulation for per-Gaussian state and Hein et al.'s spatio-temporal kernel for sub-frame interpolation.[^13][^2][^8][^6][^14]
<span style="display:none">[^100][^101][^102][^103][^104][^105][^106][^107][^108][^109][^110][^111][^112][^113][^114][^115][^116][^117][^118][^119][^120][^121][^122][^123][^124][^125][^126][^127][^128][^129][^130][^131][^132][^133][^134][^135][^136][^137][^138][^139][^140][^141][^142][^143][^144][^145][^146][^147][^148][^149][^150][^151][^152][^153][^154][^155][^156][^157][^158][^159][^16][^160][^161][^162][^163][^164][^165][^166][^167][^168][^169][^17][^170][^171][^172][^173][^174][^175][^176][^177][^178][^179][^18][^180][^181][^182][^183][^184][^185][^186][^187][^188][^189][^19][^190][^191][^192][^193][^194][^195][^196][^197][^198][^199][^20][^200][^201][^202][^203][^204][^205][^206][^207][^208][^209][^21][^210][^211][^212][^213][^214][^215][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75][^76][^77][^78][^79][^80][^81][^82][^83][^84][^85][^86][^87][^88][^89][^90][^91][^92][^93][^94][^95][^96][^97][^98][^99]</span>

<div align="center">⁂</div>

[^1]: https://arxiv.org/html/2604.18047v1

[^2]: https://arxiv.org/abs/2604.18047

[^3]: https://arxiv.org/abs/2501.06838

[^4]: https://arxiv.org/html/2501.06838v1

[^5]: https://arxiv.org/abs/2503.06617

[^6]: https://arxiv.org/html/2503.06617v1

[^7]: https://arxiv.org/abs/2506.11252

[^8]: https://arxiv.org/html/2506.11252v2

[^9]: https://arxiv.org/abs/2311.16493

[^10]: https://niujinshuchong.github.io/mip-splatting/

[^11]: https://github.com/Xinjie-Q/GaussianImage

[^12]: https://arxiv.org/html/2403.08551v4

[^13]: https://cgl.ethz.ch/Downloads/Publications/Papers/2010/Hein10/Hein10.pdf

[^14]: https://pasteur.hal.science/pasteur-04626732/file/Kalman_and_optical_flow_filtering-1.pdf

[^15]: https://www.emergentmind.com/topics/cross-temporal-3d-gaussian-splatting-cross-temporal-3dgs

[^16]: https://arxiv.org/html/2501.12060v1

[^17]: https://www.aimodels.fyi/papers/arxiv/gs-stvsr-ultra-efficient-continuous-spatio-temporal

[^18]: https://arxiv.org/html/2405.18133v1

[^19]: https://dl.acm.org/doi/full/10.1145/3721238.3730620

[^20]: https://www.research-collection.ethz.ch/server/api/core/bitstreams/0619041f-8de9-45f8-9860-a7b423d2f56b/content

[^21]: https://github.com/chrisdud0257/gsasr

[^22]: https://mt-cly.github.io/GSASR.github.io/

[^23]: https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Generalized_and_Efficient_2D_Gaussian_Splatting_for_Arbitrary-scale_Super-Resolution_ICCV_2025_paper.pdf

[^24]: https://www.themoonlight.io/en/review/generalized-and-efficient-2d-gaussian-splatting-for-arbitrary-scale-super-resolution

[^25]: https://neurips.cc/virtual/2025/poster/119938

[^26]: https://arxiv.org/abs/2308.04079

[^27]: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf

[^28]: https://openreview.net/forum?id=SZvhmFntRA

[^29]: https://cgl.ethz.ch/research/past_projects/surfels/ewavolumesplatting/index.html

[^30]: https://arxiv.org/abs/2407.18046

[^31]: https://openreview.net/pdf/19ea9a22fe4265812b4e511fa756c93c90696cdb.pdf

[^32]: https://www.themoonlight.io/en/review/gaussiansr-high-fidelity-2d-gaussian-splatting-for-arbitrary-scale-image-super-resolution

[^33]: https://arxiv.org/html/2605.02086v1

[^34]: https://arxiv.org/abs/2605.02086

[^35]: https://arxiv.org/pdf/2605.02086.pdf

[^36]: https://api.emergentmind.com/topics/gaussian-flow-field-representation

[^37]: https://papers.ssrn.com/sol3/Delivery.cfm/362317de-270d-4263-879a-e9a5140c0dd0-MECA.pdf?abstractid=6708185\&mirid=1\&type=2

[^38]: https://niedermayr.dev/upscale3dgs/

[^39]: https://www.cs.umd.edu/~zwicker/publications/EWASplatting-TVCG02.pdf

[^40]: https://github.com/tljxyys/GaussianSR

[^41]: https://liner.com/review/pixel-to-gaussian-ultrafast-continuous-superresolution-with-2d-gaussian-modeling

[^42]: https://www.labri.fr/perso/preuter/imageSynthesis/03-04/papers/ewavolume.pdf

[^43]: https://research.polyu.edu.hk/en/publications/gaussiansr-high-fidelity-2d-gaussian-splatting-for-arbitrary-scal/

[^44]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01421.pdf

[^45]: https://splatsure.github.io

[^46]: https://arxiv.org/html/2406.00609v3

[^47]: https://arxiv.org/abs/2404.10318

[^48]: https://www.scribd.com/document/823258804/Generalized-and-Efficient-2D-Gaussian-Splatting-for

[^49]: https://www.nature.com/articles/s40494-026-02355-4

[^50]: https://research.adobe.com/publication/supergaussian-repurposing-video-models-for-3d-super-resolution/

[^51]: https://www.cs.umd.edu/~zwicker/publications/ObjectSpaceEWASplatting-CGF02.pdf

[^52]: https://www.merl.com/publications/docs/TR2002-49.pdf

[^53]: https://dash.harvard.edu/bitstreams/7312037c-58ec-6bd4-e053-0100007fdf3b/download

[^54]: https://www.cs.nthu.edu.tw/~chunfa/CVGIP05.pdf

[^55]: https://leeyngdo.github.io/blog/computer-graphics/2024-04-09-gaussian-splatting/

[^56]: https://light.princeton.edu/publication/point-based-radiance-fields/

[^57]: https://arxiv.org/html/2501.19196v1

[^58]: https://www.emergentmind.com/topics/3d-gaussian-splat-radiance-field

[^59]: https://learnopencv.com/3d-gaussian-splatting/

[^60]: https://cgg.mff.cuni.cz/~jaroslav/papers/cgi2003/9-3_krivanek_j.pdf

[^61]: https://arxiv.org/abs/2205.14330

[^62]: https://openaccess.thecvf.com/content/CVPR2025/papers/Bulo_Hardware-Rasterized_Ray-Based_Gaussian_Splatting_CVPR_2025_paper.pdf

[^63]: https://www.cs.umd.edu/~zwicker/publications/SurfaceSplatting-SIG01.pdf

[^64]: https://www.zhqiang.org/3d-gaussian-splatting/

[^65]: https://arxiv.org/html/2503.12001v4

[^66]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07159.pdf

[^67]: https://tisl.cs.utoronto.ca/publication/EventSplat__3D_Gaussian_Splatting_from_Moving_Event_Cameras_for_Real-time_Rendering/EventSplat__3D_Gaussian_Splatting_from_Moving_Event_Cameras_for_Real-time_Rendering.pdf

[^68]: https://arxiv.org/html/2404.19706v3

[^69]: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[^70]: https://www.youtube.com/watch?v=D389imzYO04

[^71]: https://www.reddit.com/r/GaussianSplatting/comments/1iyz4si/realtime_gaussian_splatting/

[^72]: https://dl.acm.org/doi/fullHtml/10.1145/3641519.3657417

[^73]: https://papers.cool/arxiv/2604.18047

[^74]: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Spacetime_Gaussian_Feature_Splatting_for_Real-Time_Dynamic_View_Synthesis_CVPR_2024_paper.pdf

[^75]: https://openreview.net/forum?id=bLmImy7g1w

[^76]: https://arxiv.org/html/2503.14274v1

[^77]: https://www.scitepress.org/Papers/2025/133085/133085.pdf

[^78]: https://www.emergentmind.com/topics/opacity-gradient-driven-density-control

[^79]: https://neurips.cc/virtual/2025/poster/117695

[^80]: https://www.sciencedirect.com/science/article/abs/pii/S0262885625002756

[^81]: https://openreview.net/pdf/da34d30b60adda23b5b8887acc049011dd2629dd.pdf

[^82]: https://icml.cc/virtual/2025/poster/44339

[^83]: https://github.com/autonomousvision/mip-splatting/issues/19

[^84]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08041.pdf

[^85]: https://github.com/autonomousvision/mip-splatting

[^86]: https://x.com/janusch_patas/status/1858393467556401309

[^87]: https://huggingface.co/papers?q=particle-cloud+representation

[^88]: https://www.ndl.ethernet.edu.et/bitstream/123456789/1783/1/45.pdf

[^89]: https://www2.cs.kuleuven.be/publicaties/doctoraten/cw/CW2006_02.pdf

[^90]: https://ddd.uab.cat/pub/tfg/2024/tfg_8711419/TFG_Final.pdf

[^91]: http://wscg.zcu.cz/WSCG2010/Papers_2010/!_2010_Short-proceedings.pdf

[^92]: https://arxiv.org/html/2508.14682v1

[^93]: https://www.sciencedirect.com/science/article/pii/S2468502X25000531

[^94]: https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/

[^95]: https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_CoMoGaussian_Continuous_Motion-Aware_Gaussian_Splatting_from_Motion-Blurred_Images_ICCV_2025_paper.pdf

[^96]: https://arxiv.org/html/2404.11358v1

[^97]: https://cvpr.thecvf.com/virtual/2025/poster/34057

[^98]: https://arxiv.org/html/2312.16812v1

[^99]: https://www.themoonlight.io/en/review/gaussian-splatting-on-the-move-blur-and-rolling-shutter-compensation-for-natural-camera-motion

[^100]: https://arxiv.org/html/2506.07917v4

[^101]: https://oppo-us-research.github.io/SpacetimeGaussians-website/

[^102]: https://ieeexplore.ieee.org/iel8/10848542/10848533/10848695.pdf

[^103]: https://liner.com/review/spacetime-gaussian-feature-splatting-for-realtime-dynamic-view-synthesis

[^104]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12610591/

[^105]: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70280

[^106]: https://www.intel.com/content/www/us/en/developer/articles/technical/xess-sr-developer-guide.html

[^107]: https://wccftech.com/roundup/nvidia-dlss-vs-amd-fsr-vs-intel-xess-everything-you-need-to-know/

[^108]: https://en.wikipedia.org/wiki/Deep_Learning_Super_Sampling

[^109]: https://www.facebook.com/groups/pcbuilderandsetups/posts/1574581821008519/

[^110]: https://www.windowscentral.com/gaming/what-is-super-resolution-nvidia-dlss-amd-fsr-intel-xess-and-microsoft-directsr-explained

[^111]: https://arxiv.org/html/2312.10890v1

[^112]: https://www.extremetech.com/gaming/nvidias-dlss-5-uses-only-frame-data-and-motion-vectors-for-visual-overhaul

[^113]: https://arxiv.org/html/2308.06699v2

[^114]: https://games-1312234642.cos.ap-guangzhou.myqcloud.com/pdf/Games2022237XihaoFu.pdf

[^115]: https://www.tweaktown.com/news/110569/dlss-5-only-takes-2d-rendered-frames-and-motion-vectors-as-input-not-3d-game-engine-data-confirms-nvidia/index.html

[^116]: https://www.reddit.com/r/hardware/comments/1612mjv/amd_announces_fidelityfx_super_resolution_3_fsr_3/

[^117]: https://research.facebook.com/publications/neural-supersampling-for-real-time-rendering/

[^118]: https://www.reddit.com/r/nvidia/comments/swkkcw/lets_discuss_some_of_the_flaws_of_dlss_in_current/

[^119]: https://www.sciencedirect.com/org/science/article/pii/S1063801623000081

[^120]: https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Xiao20.pdf

[^121]: https://arxiv.org/html/2512.19108v2

[^122]: https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_Instant_GaussianImage_A_Generalizable_and_Self-Adaptive_Image_Representation_via_2D_ICCV_2025_paper.pdf

[^123]: https://xingtongge.github.io/GaussianImage-page/

[^124]: https://eureka.patsnap.com/report-how-dlss-5-adapted-algorithms-improve-simulation-speed

[^125]: https://arxiv.org/html/2312.16812v2

[^126]: https://www.themoonlight.io/en/review/gaussianimage-1000-fps-image-representation-and-compression-by-2d-gaussian-splatting

[^127]: https://arxiv.org/abs/2208.09127

[^128]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670261.pdf

[^129]: https://cs231n.stanford.edu/reports/2017/pdfs/714.pdf

[^130]: https://bmva-archive.org.uk/bmvc/2009/Papers/Paper260/Paper260.pdf

[^131]: https://ivi.fnwi.uva.nl/isis/publications/2002/GeusebroekECCV2002/GeusebroekECCV2002.pdf

[^132]: https://www.dcs.gla.ac.uk/~rod/publications/Gir04.pdf

[^133]: https://ui.adsabs.harvard.edu/abs/arXiv:2208.09127

[^134]: https://argmin.lis.tu-berlin.de/papers/07-willert-ICMLA

[^135]: https://en.wikipedia.org/wiki/Kalman_filter

[^136]: https://studios.disneyresearch.com/app/uploads/2023/06/Kernel-Based-Frame-Interpolation-for-Spatio-Temporally.pdf

[^137]: https://openaccess.thecvf.com/content_ICCV_2017/papers/Wannenwetsch_ProbFlow_Joint_Optical_ICCV_2017_paper.pdf

[^138]: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

[^139]: https://github.com/CMLab-Korea/Awesome-Video-Frame-Interpolation

[^140]: https://arxiv.org/html/2408.05970v1

[^141]: https://arxiv.org/html/2509.22112v1

[^142]: https://github.com/Lee-JaeWon/2025-Arxiv-Paper-List-Gaussian-Splatting

[^143]: https://mrnerf.github.io/awesome-3D-gaussian-splatting/

[^144]: https://www.xingzhang.me/blog/dec_interesting_papers

[^145]: https://arxiv.org/pdf/2509.22112.pdf

[^146]: https://www.sciencedirect.com/science/article/pii/S1524070325000189

[^147]: https://uu.diva-portal.org/smash/get/diva2:1375732/FULLTEXT01.pdf

[^148]: https://dl.acm.org/doi/full/10.1145/3768618

[^149]: https://www.scribd.com/document/1015736374/2405-18133v2

[^150]: https://arxiv.org/html/2412.01718v1

[^151]: https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_3DEnhancer_Consistent_Multi-View_Diffusion_for_3D_Enhancement_CVPR_2025_paper.pdf

[^152]: https://arxiv.org/abs/1708.01692

[^153]: https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1187\&context=compsci_fac

[^154]: https://www.reddit.com/r/MachineLearning/comments/6zhy7u/r_video_frame_interpolation_via_adaptive/

[^155]: https://github.com/sniklaus/revisiting-sepconv

[^156]: https://patents.google.com/patent/US20200012940A1/en

[^157]: https://arxiv.org/abs/1907.10244

[^158]: https://openaccess.thecvf.com/content_ICCV_2017/papers/Niklaus_Video_Frame_Interpolation_ICCV_2017_paper.pdf

[^159]: https://github.com/HyeongminLEE/AdaCoF-pytorch

[^160]: https://ieeexplore.ieee.org/iel7/8234942/8237262/08237299.pdf

[^161]: https://www.semanticscholar.org/paper/Video-Frame-Interpolation-via-Adaptive-Separable-Niklaus-Mai/ed74b9390eda908060fa3501b8f20a836ec98d63

[^162]: https://jhc.sjtu.edu.cn/~xiaohongliu/papers/2021video.pdf

[^163]: https://ui.adsabs.harvard.edu/abs/2020arXiv200608070C/abstract

[^164]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_AdaCoF_Adaptive_Collaboration_of_Flows_for_Video_Frame_Interpolation_CVPR_2020_paper.pdf

[^165]: https://github.com/sniklaus/sepconv-slomo

[^166]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04210.pdf

[^167]: https://papers.ssrn.com/sol3/Delivery.cfm/dff805a4-7924-4fc8-a46a-a2ea3cc5ca2f-MECA.pdf?abstractid=5929545\&mirid=1

[^168]: https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Bridging_Diffusion_Models_and_3D_Representations_A_3D_Consistent_Super-Resolution_ICCV_2025_paper.pdf

[^169]: https://gmd.copernicus.org/articles/14/337/2021/

[^170]: https://www.themoonlight.io/review/gs-stvsr-ultra-efficient-continuous-spatio-temporal-video-super-resolution-via-2d-gaussian-splatting

[^171]: https://papers.nips.cc/paper_files/paper/2024/file/f0b42291ddab77dcb2ef8a3488301b62-Paper-Conference.pdf

[^172]: https://learnopencv.com/2d-gaussian-splatting-2dgs/

[^173]: https://github.com/longxiang-ai/awesome-gaussians

[^174]: https://ko-lani.github.io/Sequence-Matters/

[^175]: https://huggingface.co/mutou0308/GSASR/discussions/1

[^176]: https://dl.acm.org/doi/10.1609/aaai.v39i4.32369

[^177]: https://github.com/peylnog/ContinuousSR/

[^178]: https://bytez.com/docs/arxiv/2604.18047/paper

[^179]: https://scirate.com/?date=2026-04-27\&page=131\&range=56

[^180]: https://pppoe.github.io/ArxRec/

[^181]: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/stochastic-lagrangian-dynamics-of-vorticity-part-1-general-theory-for-viscous-incompressible-fluids/CDC98F6928091EA96B5B6F21358A82A3

[^182]: https://www.intel.com/content/www/us/en/developer/articles/technical/xess-velocity-and-luminance-adaptive-rasterization.html

[^183]: https://github.com/JeremyChou28/Daily-Arxiv-Tools

[^184]: http://behindthepixels.io/assets/files/TemporalAA.pdf

[^185]: https://inria.hal.science/inria-00536064v1/document

[^186]: https://arxiv.org/html/2512.05113v1

[^187]: https://openaccess.thecvf.com/content/WACV2026/papers/Chien_Splannequin_Freezing_Monocular_Mannequin-Challenge_Footage_with_Dual-Detection_Splatting_WACV_2026_paper.pdf

[^188]: https://summergeometry.org/sgi2025/tag/gaussian-splatting/

[^189]: https://dl.acm.org/doi/10.1145/3592433

[^190]: https://openaccess.thecvf.com/content/CVPR2024/papers/Zheng_GPS-Gaussian_Generalizable_Pixel-wise_3D_Gaussian_Splatting_for_Real-time_Human_Novel_CVPR_2024_paper.pdf

[^191]: https://arxiv.org/html/2505.20270v1

[^192]: https://arxiv.org/html/2508.09811v1

[^193]: https://www.compxco.com/cql3d_manual_110218.pdf

[^194]: https://openreview.net/forum?id=0Zot73kfLB

[^195]: https://openaccess.thecvf.com/content/CVPR2025/papers/Li_FreeGave_3D_Physics_Learning_from_Dynamic_Videos_by_Gaussian_Velocity_CVPR_2025_paper.pdf

[^196]: https://arxiv.org/abs/2405.19745

[^197]: https://research.facebook.com/blog/2020/7/introducing-neural-supersampling-for-real-time-rendering/

[^198]: https://forum.orekit.org/t/covariance-frame-transformation-with-orbital-and-propagation-parameters/3223

[^199]: https://etd.lib.metu.edu.tr/upload/12622145/index.pdf

[^200]: https://www.themoonlight.io/en/review/gaussianprediction-dynamic-3d-gaussian-prediction-for-motion-extrapolation-and-free-view-synthesis

[^201]: http://bionics.seas.ucla.edu/education/MAE_263D/Robotics_04_Jacobian_03_Explicit_Method.pdf

[^202]: https://arxiv.org/html/2405.19745v1

[^203]: https://arxiv.org/abs/2312.06640

[^204]: http://graphics.cs.cmu.edu/projects/adpewa/index.html

[^205]: https://openreview.net/forum?id=s1zfBJysbI

[^206]: https://github.com/sczhou/Upscale-A-Video

[^207]: https://drexubery.github.io/EvaGaussians/

[^208]: https://x.com/Memoirs/status/2051759954369937877

[^209]: https://www.catalyzex.com/paper/gs-stvsr-ultra-efficient-continuous-spatio

[^210]: https://scirate.com/?date=2026-05-05\&page=18\&range=3

[^211]: https://www.catalyzex.com/s/Novel View Synthesis

[^212]: https://gamedev.net/news/3108/

[^213]: https://github.com/ZijunLi7/3dv-arxiv-daily

[^214]: https://gamedev.net/news/3142/

[^215]: https://www.alphaxiv.org/abs/2604.18047

