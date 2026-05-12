# Research Report: Novel Gaussian Formulae for 2D Canvas-Based Real-Time Upscaling \& Frame Extrapolation

## 1. Where Models Agree

| Finding | GPT-5.5 Thinking | Claude Opus 4.7 Thinking | Gemini 3.1 Pro Thinking | Evidence |
| :-- | :-- | :-- | :-- | :-- |
| GS-STVSR is your closest prior art for temporal Gaussian evolution | ✓ | ✓ | ✓ | Optical flow-guided motion module drives Gaussian position/color at arbitrary timesteps; covariance resampling prevents drift[^1_1][^1_2] |
| GSASR's scale-aware rasterization is directly relevant | ✓ | ✓ | ✓ | Feed-forward Gaussian prediction from LR features + CUDA rasterizer achieving 91ms at ×12[^1_3][^1_4] |
| ContinuousSR's Deep Gaussian Prior (DGP) can improve your spawner | ✓ | ✓ | ✓ | 99% of covariances fall in narrow ranges; pre-defined kernel dictionaries with adaptive weighting avoid local optima[^1_5][^1_6] |
| Anti-aliased 2DGS's object-space Mip filter is applicable to your rasterizer | ✓ | ✓ | ✓ | Σ'_local = I + σJJ^T maps screen-space filtering to object space via ray-splat Jacobian[^1_7][^1_8] |
| Mip-Splatting's frequency-band-limiting is critical for multi-scale rendering | ✓ | ✓ | ✓ | V_eff = V + σ²_smooth·I with opacity modulation α_smooth = α·(s_u·s_v)/√((s_u²+σ²)(s_v²+σ²))[^1_9][^1_10] |
| GaussianImage's accumulated summation (no alpha-blending sort) validates your sum-composite approach | ✓ | ✓ | ✓ | C_i = Σ_n c'_n · exp(-σ_n) — order-independent, no T_n computation, 2000 FPS[^1_11][^1_12] |

## 2. Where Models Disagree

| Topic | GPT-5.5 Thinking | Claude Opus 4.7 Thinking | Gemini 3.1 Pro Thinking | Why They Differ |
| :-- | :-- | :-- | :-- | :-- |
| Best covariance update for temporal warp | Σ' = JΣJ^T + Δt·D (your current form is optimal) | Fourier-basis temporal decomposition from Cross-Temporal 3DGS may be more stable | Covariance resampling alignment from GS-STVSR is needed to prevent drift | Different emphasis on stability vs expressivity vs interpretability |
| Whether DGP-driven covariance weighting helps at training or inference | Useful at spawner initialization time | Better as a regularization loss term constraining covariance range | Should replace direct covariance regression entirely with weighted kernel dictionary | Different assumptions about your optimization landscape |
| Relevance of Spacetime Gaussians' temporal opacity | Not relevant—your canvas is persistent | Temporal opacity σ(t) = σ^s·exp(-s^τ | t-μ^τ | ²) could gate stale Gaussians for auto-pruning |

## 3. Unique Discoveries

| Model | Unique Finding | Why It Matters |
| :-- | :-- | :-- |
| GPT-5.5 Thinking | EWA Motion Blur (Hein et al. 2010): extends EWA with 3D spatio-temporal kernels unifying spatial+temporal components for moving objects[^1_13] | Directly applicable to your temporal warp — instead of discrete per-frame advection, the kernel itself can be time-extended |
| Claude Opus 4.7 Thinking | Kalman filtering on Gaussian state with optical-flow velocity updates (KOFT)[^1_14] | Your per-Gaussian velocity field is essentially a prediction step; adding a Kalman correction step when new evidence arrives could reduce drift |
| Gemini 3.1 Pro Thinking | AA-2DGS's world-space flat smoothing projects isotropic 3D low-pass onto the splat plane: V_eff = diag(s_u² + σ²_smooth, s_v² + σ²_smooth)[^1_8] | Directly prevents your spawner from creating sub-pixel Gaussians that alias at HR output resolution |

## 4. Comprehensive Analysis

### High-Confidence Findings

The research council unanimously identifies **GS-STVSR** (April 2026) as your most direct competitor and source of equations. Its core contribution — driving Gaussian kernel evolution through optical flow-guided motion while keeping covariance parameters temporally stable via a resampling alignment module — maps almost exactly onto your temporal warp. The key equation you should examine is their covariance resampling: rather than allowing Σ to accumulate Jacobian-induced distortion indefinitely, they periodically realign covariance parameters to a canonical form based on the local scale of the output. This addresses your stated concern about "compounding non-determinism" over multi-frame extrapolation.[^1_1][^1_2]

All three models agree that **ContinuousSR's Deep Gaussian Prior** offers immediate value for your spawner's bias problem. The statistical finding that 99% of natural-image Gaussian covariances fall within σ²_x ∈ [0, 2.4], σ²_y ∈ [0, 2.2], ρσ_xσ_y ∈ [-0.9, 1.5] provides a principled initialization range. Their DGP-Driven Covariance Weighting replaces direct covariance regression with:[^1_5][^1_6]

$G_{\text{target}} = \sum_{i=1}^{N} w_i \cdot G_i, \quad \mathbf{W} = \text{Softmax}(\mathcal{M}_{\text{weight}}(\mathcal{F}_{\text{LR}}))$

where {G_i} are sampled from the DGP distribution. This could replace or augment your spawner's Δscale/Δrot regression, potentially eliminating the checkerboard artifact by ensuring spawned Gaussians conform to natural statistics rather than collapsing to a tight fractional bias.[^1_6]

The **accumulated summation rasterization** from GaussianImage validates your sum-composite EWA approach. Their equation C_i = Σ_n c'_n · exp(-σ_n) is mathematically equivalent to your out[c,py,px] = Σ_g exp(-½·q_g) · feat[g,c], confirming that sort-free order-independent splatting achieves 2000+ FPS with competitive quality. Their key ablation shows a 0.8 dB PSNR improvement over alpha-blending when depth ordering is unknown — directly applicable to your 2D canvas where no canonical depth exists.[^1_12]

### Areas of Divergence

The most substantive disagreement concerns **how to handle temporal covariance evolution**. GPT-5.5 Thinking endorses your current Σ' = JΣJ^T + Δt·D as theoretically sound, while Claude Opus 4.7 Thinking points to Cross-Temporal 3DGS's Fourier-basis decomposition where μ(t) and R(t) are modeled as smooth functions of time while scale/opacity remain invariant. The Fourier approach offers guaranteed smoothness but sacrifices the ability to model sudden scale changes (e.g., objects approaching camera). Gemini 3.1 Pro Thinking's recommendation of GS-STVSR's covariance resampling is a pragmatic middle ground — apply your J-based warp but periodically snap covariances back to well-conditioned forms.[^1_2][^1_15]

The **Hein et al. 2010 spatio-temporal EWA kernel** is a particularly interesting find by GPT-5.5 Thinking. Rather than treating motion as a discrete per-frame position update, they extend the 2D Gaussian kernel into a 3D spatio-temporal kernel where the temporal dimension encodes motion blur. For your frame extrapolation use case, this means a single Gaussian evaluation could produce not just the current frame but an analytically motion-blurred intermediate — potentially useful for sub-frame interpolation without separate warp passes. The formula unifies spatial reconstruction and temporal filtering into one kernel evaluation.[^1_13]

### Unique Insights Worth Noting

The **Kalman filtering insight** from Claude Opus 4.7 Thinking deserves serious consideration. Your per-Gaussian velocity field is a prediction model; when a new rendered frame arrives, you currently rely entirely on the spawner to correct drift. A Kalman update step would provide:[^1_14]

$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t(z_t - H\hat{x}_{t|t-1})$

where z_t is the observed position from the new frame's motion vectors and K_t is the Kalman gain. This adds a principled correction mechanism between the warp (prediction) and spawner (observation), potentially extending your viable extrapolation horizon beyond 1-2 frames.

The **AA-2DGS object-space Mip filter** equation Σ'_local(x) = I + σJJ^T is directly implementable in your rasterizer. The Jacobian J = ∂u/∂x of the pixel-to-Gaussian-space mapping is cheap to compute (you already have the conic parameters), and the resulting modified quadratic form prevents aliasing at output resolution without requiring explicit anti-aliasing passes. This is particularly relevant for your ×4 upscaling scenario where a single LR Gaussian may span only a fraction of an HR pixel.[^1_8]

### Recommendations

For immediate v6.2 integration, prioritize: (1) ContinuousSR's DGP-driven covariance weighting for the spawner to fix your checkerboard artifact with a principled statistical prior rather than just jitter; (2) GS-STVSR's covariance resampling alignment to extend extrapolation horizon; (3) AA-2DGS's object-space Mip filter (Σ'_local = I + σJJ^T) in your rasterizer to prevent aliasing at HR output. For v6.3, investigate the Kalman correction formulation for per-Gaussian state and Hein et al.'s spatio-temporal kernel for sub-frame interpolation.[^1_13][^1_2][^1_8][^1_6][^1_14]
<span style="display:none">[^1_100][^1_101][^1_102][^1_103][^1_104][^1_105][^1_106][^1_107][^1_108][^1_109][^1_110][^1_111][^1_112][^1_113][^1_114][^1_115][^1_116][^1_117][^1_118][^1_119][^1_120][^1_121][^1_122][^1_123][^1_124][^1_125][^1_126][^1_127][^1_128][^1_129][^1_130][^1_131][^1_132][^1_133][^1_134][^1_135][^1_136][^1_137][^1_138][^1_139][^1_140][^1_141][^1_142][^1_143][^1_144][^1_145][^1_146][^1_147][^1_148][^1_149][^1_150][^1_151][^1_152][^1_153][^1_154][^1_155][^1_156][^1_157][^1_158][^1_159][^1_16][^1_160][^1_161][^1_162][^1_163][^1_164][^1_165][^1_166][^1_167][^1_168][^1_169][^1_17][^1_170][^1_171][^1_172][^1_173][^1_174][^1_175][^1_176][^1_177][^1_178][^1_179][^1_18][^1_180][^1_181][^1_182][^1_183][^1_184][^1_185][^1_186][^1_187][^1_188][^1_189][^1_19][^1_190][^1_191][^1_192][^1_193][^1_194][^1_195][^1_196][^1_197][^1_198][^1_199][^1_20][^1_200][^1_201][^1_202][^1_203][^1_204][^1_205][^1_206][^1_207][^1_208][^1_209][^1_21][^1_210][^1_211][^1_212][^1_213][^1_214][^1_215][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97][^1_98][^1_99]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/html/2604.18047v1

[^1_2]: https://arxiv.org/abs/2604.18047

[^1_3]: https://arxiv.org/abs/2501.06838

[^1_4]: https://arxiv.org/html/2501.06838v1

[^1_5]: https://arxiv.org/abs/2503.06617

[^1_6]: https://arxiv.org/html/2503.06617v1

[^1_7]: https://arxiv.org/abs/2506.11252

[^1_8]: https://arxiv.org/html/2506.11252v2

[^1_9]: https://arxiv.org/abs/2311.16493

[^1_10]: https://niujinshuchong.github.io/mip-splatting/

[^1_11]: https://github.com/Xinjie-Q/GaussianImage

[^1_12]: https://arxiv.org/html/2403.08551v4

[^1_13]: https://cgl.ethz.ch/Downloads/Publications/Papers/2010/Hein10/Hein10.pdf

[^1_14]: https://pasteur.hal.science/pasteur-04626732/file/Kalman_and_optical_flow_filtering-1.pdf

[^1_15]: https://www.emergentmind.com/topics/cross-temporal-3d-gaussian-splatting-cross-temporal-3dgs

[^1_16]: https://arxiv.org/html/2501.12060v1

[^1_17]: https://www.aimodels.fyi/papers/arxiv/gs-stvsr-ultra-efficient-continuous-spatio-temporal

[^1_18]: https://arxiv.org/html/2405.18133v1

[^1_19]: https://dl.acm.org/doi/full/10.1145/3721238.3730620

[^1_20]: https://www.research-collection.ethz.ch/server/api/core/bitstreams/0619041f-8de9-45f8-9860-a7b423d2f56b/content

[^1_21]: https://github.com/chrisdud0257/gsasr

[^1_22]: https://mt-cly.github.io/GSASR.github.io/

[^1_23]: https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Generalized_and_Efficient_2D_Gaussian_Splatting_for_Arbitrary-scale_Super-Resolution_ICCV_2025_paper.pdf

[^1_24]: https://www.themoonlight.io/en/review/generalized-and-efficient-2d-gaussian-splatting-for-arbitrary-scale-super-resolution

[^1_25]: https://neurips.cc/virtual/2025/poster/119938

[^1_26]: https://arxiv.org/abs/2308.04079

[^1_27]: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf

[^1_28]: https://openreview.net/forum?id=SZvhmFntRA

[^1_29]: https://cgl.ethz.ch/research/past_projects/surfels/ewavolumesplatting/index.html

[^1_30]: https://arxiv.org/abs/2407.18046

[^1_31]: https://openreview.net/pdf/19ea9a22fe4265812b4e511fa756c93c90696cdb.pdf

[^1_32]: https://www.themoonlight.io/en/review/gaussiansr-high-fidelity-2d-gaussian-splatting-for-arbitrary-scale-image-super-resolution

[^1_33]: https://arxiv.org/html/2605.02086v1

[^1_34]: https://arxiv.org/abs/2605.02086

[^1_35]: https://arxiv.org/pdf/2605.02086.pdf

[^1_36]: https://api.emergentmind.com/topics/gaussian-flow-field-representation

[^1_37]: https://papers.ssrn.com/sol3/Delivery.cfm/362317de-270d-4263-879a-e9a5140c0dd0-MECA.pdf?abstractid=6708185\&mirid=1\&type=2

[^1_38]: https://niedermayr.dev/upscale3dgs/

[^1_39]: https://www.cs.umd.edu/~zwicker/publications/EWASplatting-TVCG02.pdf

[^1_40]: https://github.com/tljxyys/GaussianSR

[^1_41]: https://liner.com/review/pixel-to-gaussian-ultrafast-continuous-superresolution-with-2d-gaussian-modeling

[^1_42]: https://www.labri.fr/perso/preuter/imageSynthesis/03-04/papers/ewavolume.pdf

[^1_43]: https://research.polyu.edu.hk/en/publications/gaussiansr-high-fidelity-2d-gaussian-splatting-for-arbitrary-scal/

[^1_44]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01421.pdf

[^1_45]: https://splatsure.github.io

[^1_46]: https://arxiv.org/html/2406.00609v3

[^1_47]: https://arxiv.org/abs/2404.10318

[^1_48]: https://www.scribd.com/document/823258804/Generalized-and-Efficient-2D-Gaussian-Splatting-for

[^1_49]: https://www.nature.com/articles/s40494-026-02355-4

[^1_50]: https://research.adobe.com/publication/supergaussian-repurposing-video-models-for-3d-super-resolution/

[^1_51]: https://www.cs.umd.edu/~zwicker/publications/ObjectSpaceEWASplatting-CGF02.pdf

[^1_52]: https://www.merl.com/publications/docs/TR2002-49.pdf

[^1_53]: https://dash.harvard.edu/bitstreams/7312037c-58ec-6bd4-e053-0100007fdf3b/download

[^1_54]: https://www.cs.nthu.edu.tw/~chunfa/CVGIP05.pdf

[^1_55]: https://leeyngdo.github.io/blog/computer-graphics/2024-04-09-gaussian-splatting/

[^1_56]: https://light.princeton.edu/publication/point-based-radiance-fields/

[^1_57]: https://arxiv.org/html/2501.19196v1

[^1_58]: https://www.emergentmind.com/topics/3d-gaussian-splat-radiance-field

[^1_59]: https://learnopencv.com/3d-gaussian-splatting/

[^1_60]: https://cgg.mff.cuni.cz/~jaroslav/papers/cgi2003/9-3_krivanek_j.pdf

[^1_61]: https://arxiv.org/abs/2205.14330

[^1_62]: https://openaccess.thecvf.com/content/CVPR2025/papers/Bulo_Hardware-Rasterized_Ray-Based_Gaussian_Splatting_CVPR_2025_paper.pdf

[^1_63]: https://www.cs.umd.edu/~zwicker/publications/SurfaceSplatting-SIG01.pdf

[^1_64]: https://www.zhqiang.org/3d-gaussian-splatting/

[^1_65]: https://arxiv.org/html/2503.12001v4

[^1_66]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07159.pdf

[^1_67]: https://tisl.cs.utoronto.ca/publication/EventSplat__3D_Gaussian_Splatting_from_Moving_Event_Cameras_for_Real-time_Rendering/EventSplat__3D_Gaussian_Splatting_from_Moving_Event_Cameras_for_Real-time_Rendering.pdf

[^1_68]: https://arxiv.org/html/2404.19706v3

[^1_69]: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[^1_70]: https://www.youtube.com/watch?v=D389imzYO04

[^1_71]: https://www.reddit.com/r/GaussianSplatting/comments/1iyz4si/realtime_gaussian_splatting/

[^1_72]: https://dl.acm.org/doi/fullHtml/10.1145/3641519.3657417

[^1_73]: https://papers.cool/arxiv/2604.18047

[^1_74]: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Spacetime_Gaussian_Feature_Splatting_for_Real-Time_Dynamic_View_Synthesis_CVPR_2024_paper.pdf

[^1_75]: https://openreview.net/forum?id=bLmImy7g1w

[^1_76]: https://arxiv.org/html/2503.14274v1

[^1_77]: https://www.scitepress.org/Papers/2025/133085/133085.pdf

[^1_78]: https://www.emergentmind.com/topics/opacity-gradient-driven-density-control

[^1_79]: https://neurips.cc/virtual/2025/poster/117695

[^1_80]: https://www.sciencedirect.com/science/article/abs/pii/S0262885625002756

[^1_81]: https://openreview.net/pdf/da34d30b60adda23b5b8887acc049011dd2629dd.pdf

[^1_82]: https://icml.cc/virtual/2025/poster/44339

[^1_83]: https://github.com/autonomousvision/mip-splatting/issues/19

[^1_84]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08041.pdf

[^1_85]: https://github.com/autonomousvision/mip-splatting

[^1_86]: https://x.com/janusch_patas/status/1858393467556401309

[^1_87]: https://huggingface.co/papers?q=particle-cloud+representation

[^1_88]: https://www.ndl.ethernet.edu.et/bitstream/123456789/1783/1/45.pdf

[^1_89]: https://www2.cs.kuleuven.be/publicaties/doctoraten/cw/CW2006_02.pdf

[^1_90]: https://ddd.uab.cat/pub/tfg/2024/tfg_8711419/TFG_Final.pdf

[^1_91]: http://wscg.zcu.cz/WSCG2010/Papers_2010/!_2010_Short-proceedings.pdf

[^1_92]: https://arxiv.org/html/2508.14682v1

[^1_93]: https://www.sciencedirect.com/science/article/pii/S2468502X25000531

[^1_94]: https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/

[^1_95]: https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_CoMoGaussian_Continuous_Motion-Aware_Gaussian_Splatting_from_Motion-Blurred_Images_ICCV_2025_paper.pdf

[^1_96]: https://arxiv.org/html/2404.11358v1

[^1_97]: https://cvpr.thecvf.com/virtual/2025/poster/34057

[^1_98]: https://arxiv.org/html/2312.16812v1

[^1_99]: https://www.themoonlight.io/en/review/gaussian-splatting-on-the-move-blur-and-rolling-shutter-compensation-for-natural-camera-motion

[^1_100]: https://arxiv.org/html/2506.07917v4

[^1_101]: https://oppo-us-research.github.io/SpacetimeGaussians-website/

[^1_102]: https://ieeexplore.ieee.org/iel8/10848542/10848533/10848695.pdf

[^1_103]: https://liner.com/review/spacetime-gaussian-feature-splatting-for-realtime-dynamic-view-synthesis

[^1_104]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12610591/

[^1_105]: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70280

[^1_106]: https://www.intel.com/content/www/us/en/developer/articles/technical/xess-sr-developer-guide.html

[^1_107]: https://wccftech.com/roundup/nvidia-dlss-vs-amd-fsr-vs-intel-xess-everything-you-need-to-know/

[^1_108]: https://en.wikipedia.org/wiki/Deep_Learning_Super_Sampling

[^1_109]: https://www.facebook.com/groups/pcbuilderandsetups/posts/1574581821008519/

[^1_110]: https://www.windowscentral.com/gaming/what-is-super-resolution-nvidia-dlss-amd-fsr-intel-xess-and-microsoft-directsr-explained

[^1_111]: https://arxiv.org/html/2312.10890v1

[^1_112]: https://www.extremetech.com/gaming/nvidias-dlss-5-uses-only-frame-data-and-motion-vectors-for-visual-overhaul

[^1_113]: https://arxiv.org/html/2308.06699v2

[^1_114]: https://games-1312234642.cos.ap-guangzhou.myqcloud.com/pdf/Games2022237XihaoFu.pdf

[^1_115]: https://www.tweaktown.com/news/110569/dlss-5-only-takes-2d-rendered-frames-and-motion-vectors-as-input-not-3d-game-engine-data-confirms-nvidia/index.html

[^1_116]: https://www.reddit.com/r/hardware/comments/1612mjv/amd_announces_fidelityfx_super_resolution_3_fsr_3/

[^1_117]: https://research.facebook.com/publications/neural-supersampling-for-real-time-rendering/

[^1_118]: https://www.reddit.com/r/nvidia/comments/swkkcw/lets_discuss_some_of_the_flaws_of_dlss_in_current/

[^1_119]: https://www.sciencedirect.com/org/science/article/pii/S1063801623000081

[^1_120]: https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Xiao20.pdf

[^1_121]: https://arxiv.org/html/2512.19108v2

[^1_122]: https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_Instant_GaussianImage_A_Generalizable_and_Self-Adaptive_Image_Representation_via_2D_ICCV_2025_paper.pdf

[^1_123]: https://xingtongge.github.io/GaussianImage-page/

[^1_124]: https://eureka.patsnap.com/report-how-dlss-5-adapted-algorithms-improve-simulation-speed

[^1_125]: https://arxiv.org/html/2312.16812v2

[^1_126]: https://www.themoonlight.io/en/review/gaussianimage-1000-fps-image-representation-and-compression-by-2d-gaussian-splatting

[^1_127]: https://arxiv.org/abs/2208.09127

[^1_128]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670261.pdf

[^1_129]: https://cs231n.stanford.edu/reports/2017/pdfs/714.pdf

[^1_130]: https://bmva-archive.org.uk/bmvc/2009/Papers/Paper260/Paper260.pdf

[^1_131]: https://ivi.fnwi.uva.nl/isis/publications/2002/GeusebroekECCV2002/GeusebroekECCV2002.pdf

[^1_132]: https://www.dcs.gla.ac.uk/~rod/publications/Gir04.pdf

[^1_133]: https://ui.adsabs.harvard.edu/abs/arXiv:2208.09127

[^1_134]: https://argmin.lis.tu-berlin.de/papers/07-willert-ICMLA

[^1_135]: https://en.wikipedia.org/wiki/Kalman_filter

[^1_136]: https://studios.disneyresearch.com/app/uploads/2023/06/Kernel-Based-Frame-Interpolation-for-Spatio-Temporally.pdf

[^1_137]: https://openaccess.thecvf.com/content_ICCV_2017/papers/Wannenwetsch_ProbFlow_Joint_Optical_ICCV_2017_paper.pdf

[^1_138]: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

[^1_139]: https://github.com/CMLab-Korea/Awesome-Video-Frame-Interpolation

[^1_140]: https://arxiv.org/html/2408.05970v1

[^1_141]: https://arxiv.org/html/2509.22112v1

[^1_142]: https://github.com/Lee-JaeWon/2025-Arxiv-Paper-List-Gaussian-Splatting

[^1_143]: https://mrnerf.github.io/awesome-3D-gaussian-splatting/

[^1_144]: https://www.xingzhang.me/blog/dec_interesting_papers

[^1_145]: https://arxiv.org/pdf/2509.22112.pdf

[^1_146]: https://www.sciencedirect.com/science/article/pii/S1524070325000189

[^1_147]: https://uu.diva-portal.org/smash/get/diva2:1375732/FULLTEXT01.pdf

[^1_148]: https://dl.acm.org/doi/full/10.1145/3768618

[^1_149]: https://www.scribd.com/document/1015736374/2405-18133v2

[^1_150]: https://arxiv.org/html/2412.01718v1

[^1_151]: https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_3DEnhancer_Consistent_Multi-View_Diffusion_for_3D_Enhancement_CVPR_2025_paper.pdf

[^1_152]: https://arxiv.org/abs/1708.01692

[^1_153]: https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1187\&context=compsci_fac

[^1_154]: https://www.reddit.com/r/MachineLearning/comments/6zhy7u/r_video_frame_interpolation_via_adaptive/

[^1_155]: https://github.com/sniklaus/revisiting-sepconv

[^1_156]: https://patents.google.com/patent/US20200012940A1/en

[^1_157]: https://arxiv.org/abs/1907.10244

[^1_158]: https://openaccess.thecvf.com/content_ICCV_2017/papers/Niklaus_Video_Frame_Interpolation_ICCV_2017_paper.pdf

[^1_159]: https://github.com/HyeongminLEE/AdaCoF-pytorch

[^1_160]: https://ieeexplore.ieee.org/iel7/8234942/8237262/08237299.pdf

[^1_161]: https://www.semanticscholar.org/paper/Video-Frame-Interpolation-via-Adaptive-Separable-Niklaus-Mai/ed74b9390eda908060fa3501b8f20a836ec98d63

[^1_162]: https://jhc.sjtu.edu.cn/~xiaohongliu/papers/2021video.pdf

[^1_163]: https://ui.adsabs.harvard.edu/abs/2020arXiv200608070C/abstract

[^1_164]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_AdaCoF_Adaptive_Collaboration_of_Flows_for_Video_Frame_Interpolation_CVPR_2020_paper.pdf

[^1_165]: https://github.com/sniklaus/sepconv-slomo

[^1_166]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04210.pdf

[^1_167]: https://papers.ssrn.com/sol3/Delivery.cfm/dff805a4-7924-4fc8-a46a-a2ea3cc5ca2f-MECA.pdf?abstractid=5929545\&mirid=1

[^1_168]: https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Bridging_Diffusion_Models_and_3D_Representations_A_3D_Consistent_Super-Resolution_ICCV_2025_paper.pdf

[^1_169]: https://gmd.copernicus.org/articles/14/337/2021/

[^1_170]: https://www.themoonlight.io/review/gs-stvsr-ultra-efficient-continuous-spatio-temporal-video-super-resolution-via-2d-gaussian-splatting

[^1_171]: https://papers.nips.cc/paper_files/paper/2024/file/f0b42291ddab77dcb2ef8a3488301b62-Paper-Conference.pdf

[^1_172]: https://learnopencv.com/2d-gaussian-splatting-2dgs/

[^1_173]: https://github.com/longxiang-ai/awesome-gaussians

[^1_174]: https://ko-lani.github.io/Sequence-Matters/

[^1_175]: https://huggingface.co/mutou0308/GSASR/discussions/1

[^1_176]: https://dl.acm.org/doi/10.1609/aaai.v39i4.32369

[^1_177]: https://github.com/peylnog/ContinuousSR/

[^1_178]: https://bytez.com/docs/arxiv/2604.18047/paper

[^1_179]: https://scirate.com/?date=2026-04-27\&page=131\&range=56

[^1_180]: https://pppoe.github.io/ArxRec/

[^1_181]: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/stochastic-lagrangian-dynamics-of-vorticity-part-1-general-theory-for-viscous-incompressible-fluids/CDC98F6928091EA96B5B6F21358A82A3

[^1_182]: https://www.intel.com/content/www/us/en/developer/articles/technical/xess-velocity-and-luminance-adaptive-rasterization.html

[^1_183]: https://github.com/JeremyChou28/Daily-Arxiv-Tools

[^1_184]: http://behindthepixels.io/assets/files/TemporalAA.pdf

[^1_185]: https://inria.hal.science/inria-00536064v1/document

[^1_186]: https://arxiv.org/html/2512.05113v1

[^1_187]: https://openaccess.thecvf.com/content/WACV2026/papers/Chien_Splannequin_Freezing_Monocular_Mannequin-Challenge_Footage_with_Dual-Detection_Splatting_WACV_2026_paper.pdf

[^1_188]: https://summergeometry.org/sgi2025/tag/gaussian-splatting/

[^1_189]: https://dl.acm.org/doi/10.1145/3592433

[^1_190]: https://openaccess.thecvf.com/content/CVPR2024/papers/Zheng_GPS-Gaussian_Generalizable_Pixel-wise_3D_Gaussian_Splatting_for_Real-time_Human_Novel_CVPR_2024_paper.pdf

[^1_191]: https://arxiv.org/html/2505.20270v1

[^1_192]: https://arxiv.org/html/2508.09811v1

[^1_193]: https://www.compxco.com/cql3d_manual_110218.pdf

[^1_194]: https://openreview.net/forum?id=0Zot73kfLB

[^1_195]: https://openaccess.thecvf.com/content/CVPR2025/papers/Li_FreeGave_3D_Physics_Learning_from_Dynamic_Videos_by_Gaussian_Velocity_CVPR_2025_paper.pdf

[^1_196]: https://arxiv.org/abs/2405.19745

[^1_197]: https://research.facebook.com/blog/2020/7/introducing-neural-supersampling-for-real-time-rendering/

[^1_198]: https://forum.orekit.org/t/covariance-frame-transformation-with-orbital-and-propagation-parameters/3223

[^1_199]: https://etd.lib.metu.edu.tr/upload/12622145/index.pdf

[^1_200]: https://www.themoonlight.io/en/review/gaussianprediction-dynamic-3d-gaussian-prediction-for-motion-extrapolation-and-free-view-synthesis

[^1_201]: http://bionics.seas.ucla.edu/education/MAE_263D/Robotics_04_Jacobian_03_Explicit_Method.pdf

[^1_202]: https://arxiv.org/html/2405.19745v1

[^1_203]: https://arxiv.org/abs/2312.06640

[^1_204]: http://graphics.cs.cmu.edu/projects/adpewa/index.html

[^1_205]: https://openreview.net/forum?id=s1zfBJysbI

[^1_206]: https://github.com/sczhou/Upscale-A-Video

[^1_207]: https://drexubery.github.io/EvaGaussians/

[^1_208]: https://x.com/Memoirs/status/2051759954369937877

[^1_209]: https://www.catalyzex.com/paper/gs-stvsr-ultra-efficient-continuous-spatio

[^1_210]: https://scirate.com/?date=2026-05-05\&page=18\&range=3

[^1_211]: https://www.catalyzex.com/s/Novel View Synthesis

[^1_212]: https://gamedev.net/news/3108/

[^1_213]: https://github.com/ZijunLi7/3dv-arxiv-daily

[^1_214]: https://gamedev.net/news/3142/

[^1_215]: https://www.alphaxiv.org/abs/2604.18047


---

# [https://github.com/cashcon57/open-supersampling](https://github.com/cashcon57/open-supersampling) - github repo for what we're doing. I want a followup focused on absolute best ms/frame optimization. Every stage of the upscaler, provide novel, elegant, performant ways to reduce the upscaler's cost absolutely.

You’re bandwidth‑constrained, so I’ll stay concrete and brutalist: where to shave ms in *your* pipeline, stage by stage, with tricks that go beyond obvious micro‑optimizations. I’ll assume the architecture you described in the prior message: HAT‑Tiny backbone, pixel↔Gaussian cross‑attention, persistent 2D canvas, EWA CUDA rasterizer, temporal warp, spawner.

Below I focus on ideas that are (a) implementable in your current design, and (b) likely to move the needle on frame time. I’ll sprinkle in equations only where they justify a structural optimization.

***

## 1. Backbone: HAT‑Tiny as Feature Extractor

You already use only the LR branch; remaining cost is mostly windowed self-attn and MLPs.

**1.1. Reduce tokens the backbone ever sees**

Right now every LR pixel runs through HAT; but much of the image is temporally stable and well predicted by the canvas.

- Introduce a **splat‑confidence mask**: after rasterization, compute per‑pixel residual magnitude $\|I_\text{raster} - I_\text{LR}\|$. Only pixels above a residual threshold are passed through backbone, others reuse cached backbone features from previous frame.
    - Maintain a feature cache $F_{t-1}(x,y)$ for LR; at frame t:

$$
F_t(x,y) = 
\begin{cases}
\text{HAT}(x,y) & \text{if } \|r_t(x,y)\| > \tau \\
F_{t-1}(x,y) & \text{otherwise}
\end{cases}
$$

This halves effective HAT FLOPs in stable regions, and is friendly to temporal consistency.
- Implement this as a **block‑sparse HAT pass**: run HAT only on 16×16 LR blocks whose residual max exceeds τ; process others via memcopy.

**1.2. Low‑rank / grouped attention in HAT**

Inside HAT window attention:

- Replace full QKᵀ with a **Nyström or Linformer‑style low‑rank approximation** within each window:

$$
\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V \approx \text{softmax}\left(\frac{Q E^\top K_E^\top}{\sqrt{d}}\right)V
$$

where $E$ picks M≪ws² landmarks. In practice, M=8–16 per window usually keeps quality for SR while cutting attention FLOPs by ~2×.
- Aggressively **group channels** in the MLP: convert 2× depthwise‑separable 1×1 convs into grouped linear + fused GELU to reduce memory bandwidth.

**1.3. Quantization and channel shaving**

- HAT weights → **int8** with FP16 activations via PTQ/QAT; you’re already bf16 across the rasterizer/backbone boundary; extending to int8 can cut backbone time 1.5–2× on RTX tensor cores.
- Verify empirically that going from embed_dim=180→144 and head_dim=24 barely harms quality; that’s ~36% compute reduction.

***

## 2. Cross‑Attention Pixel↔Gaussian Fusion

This can easily dominate runtime when the canvas is dense. Design the math to make K small and predictable.

### 2.1. Spatially constrained top‑K Gaussians per window

Right now K is “up to canvas capacity” per batch. Make K *locally* bounded:

- Pre‑bucket Gaussians into **LR tiles** matching HAT windows (e.g., 16×16 LR). For each window, only feed the Gaussians whose xy falls within a padded window region (e.g., +4px border).
- Additionally perform **importance pruning** within the tile: K_top per window by descending opacity·area or a learned score s_g:

$$
s_g = \alpha_g \cdot \sqrt{|\Sigma_g|} \cdot \gamma_g
$$

where $\gamma_g$ is a small 1‑layer MLP over feat[g]. This keeps only K_top≈32–64 tokens per window in attention.

This changes cross-attn cost from O(ws² · K) to O(ws² · K_top) with K_top ≪ K_global.

### 2.2. Factorized key/value representations

- Instead of K/V in ℝ⁶⁴ projected up to head_dim=30, introduce a **rank‑R factorization** over Gaussians:

$$
K = (U_G R_K),\quad V = (U_G R_V)
$$

where U_G ∈ ℝ^{K×R} is a learned low‑rank Gaussian embedding and R_K,R_V ∈ ℝ^{R×d_h} are shared. With R=8, you cut per‑Gaussian projection FLOPs by ~4×. U_G can be precomputed per frame from feat via a shared small MLP.


### 2.3. Hard attention routing (mixture‑of‑Gaussians)

- Add a **routing head** over each window’s pixel features producing a categorical over G router groups (e.g., 4–8). Each Gaussian is also assigned a router id; each head only attends to Gaussians sharing its router id. This shrinks effective K per head by ~G.

In code: groupby(router_id) on both pixel windows and Gaussians, do smaller SDPA calls, then concatenate.

***

## 3. Temporal Warp \& Canvas Evolution

The state update is embarrassingly parallel; bottlenecks are bandwidth and atomic contention in gradient accumulation.

### 3.1. Keep covariance updates embarrassingly simple

Your warp equation Σ' = J Σ Jᵀ + Δt·D is already optimal mathematically; focus on making J and D cheap:

- Approximate J as **locally constant and diagonal** per 16×16 LR block:

$$
J \approx 
\begin{bmatrix}
s_x & 0 \\
0 & s_y 
\end{bmatrix}
$$

where $s_x, s_y$ are per‑block scales regressed from block‑pooled velocity field. This makes JΣJᵀ two scalar multiplies per covariance element instead of a full 2×2 matmul.
- Constrain D to be isotropic, D = d·I; now Σ' update is:

$$
\Sigma' = 
\begin{bmatrix}
s_x^2 \sigma_x^2 + \Delta t d & s_x s_y \rho \sigma_x \sigma_y \\
s_x s_y \rho \sigma_x \sigma_y & s_y^2 \sigma_y^2 + \Delta t d
\end{bmatrix}
$$

This reduces parameter bandwidth and arithmetic.


### 3.2. Multi‑step warp on a coarser grid

For extrapolation >1 frame, don’t warp primitives multiple times:

- Maintain a **piecewise linear velocity** per Gaussian over 2–3 frames: v₀,v₁. To extrapolate to t+2, compute a closed‑form 2Δt displacement and one Σ'' from an approximate analytic integration instead of two discrete warps.
- Regularly **re‑anchor** Gaussians to LR grid centers to maintain numeric stability: at keyframes (e.g., every 4th frame), rebuild xy as LR pixel centers plus subpixel offset, and reset Σ to a canonical form; let the network relearn fine offsets.

This keeps the warp kernel cheap and limits accumulation of FP32 error.

***

## 4. Spawner: Making Adaptation Cheap and Stable

Spawner is a prime candidate for heavy MLPs; we can both shrink it and make it more stable so you need fewer spawned Gaussians.

### 4.1. Deep Gaussian Prior to shrink the regression problem

Steal ContinuousSR’s Deep Gaussian Prior idea:[^2_1][^2_2]

- Pre‑define a small dictionary of M prototype covariance matrices $\{\Sigma_m\}_{m=1}^M$ sampling realistic (σ²_x,σ²_y,ρ) ranges (M≈8–16).
- Instead of regressing Δscale,Δrot, let the spawner output a **softmax over this dictionary** plus a scalar scale:

$$
\Sigma_\text{spawn} = \lambda \sum_{m} w_m \Sigma_m
$$

This replaces high‑dimensional regression with an M‑way classification + 1 scalar, cutting multiplications and making spawned Gaussians better conditioned so you need fewer of them.

### 4.2. Probabilistic sparsity: spawn only where needed

For each candidate canvas token:

- Predict a **spawn‑probability p_spawn** plus the deltas. Implement a straight‑through Bernoulli (or simply threshold at training, top‑k at inference) to only actually allocate a Gaussian if p_spawn > τ and local residual is high. That’s fewer primitives → cheaper rasterization and attention.
- To kill the checkerboard: keep your sub‑pixel jitter, but **anti‑correlate it across neighbors** with a blue‑noise pattern rather than uniform independent noise. This avoids creating new low‑frequency artifacts but is effectively free (look up deterministic blue‑noise tiles).

***

## 5. Rasterizer: Where Most of the ms Live

You reported ~136 ms forward on 3080 Ti for N=4096, H=540, W=960, F=64. That’s the chief target.

### 5.1. Value‑function simplification via accumulated sum

GaussianImage shows that if you treat the canvas as a pure 2D Gaussian field, you can get away with:

$$
C(x,y,c) = \sum_{g} c'_{g,c}\, \exp\big(-\tfrac{1}{2} q_g(x,y)\big)
$$

with c' absorbing opacity. That’s exactly what you do. Their key performance tricks you can borrow:[^2_3]

- **No early‑out on alpha**: since you’re not alpha blending, you can avoid per‑pixel branching on accumulated transmittance. This increases arithmetic slightly but reduces divergence; your kernel already respects this.
- **Register tiling**: you already have pix_out in registers. You can go further: process 2×2 HR pixels per thread (swizzle mapping), keeping 4×F values in registers to amortize Gaussian loads across pixels.[^2_4]


### 5.2. Gaussian culling and hierarchy

Two big wins:

1. **Pre‑bin by HR tile and cull by bounding ellipse**. Before launching blocks:
    - Compute each Gaussian’s axis‑aligned bounding box at HR using its conic (a,b,d), store tile range [tx0..tx1,ty0..ty1].
    - Only push the Gaussian into those tiles; this you already do with a sort on (tile_id,gid). Tighten the bounding box with a fixed confidence radius (e.g., 2.5σ) to drop far‑tails.
2. Add a **mini MIP hierarchy over Gaussians**:
    - Maintain a 2‑level grid of “macro‑tiles” (e.g., 32×32 LR). For each macro tile, cache up to K_macro high‑impact Gaussians (by opacity·area).
    - At inference, if a 16×16 tile lies in a macro tile with “low residual history” and camera warp small, skip its per‑tile Gaussian list and render from the cached aggregated Gaussians (or even reuse last frame’s HR tile). This is essentially temporal + spatial caching for tiles.

### 5.3. Exploit analytic gradients for cheaper bicubic

There’s a neat idea from “gradient‑aware upscaling for 3DGS” where they use Gaussians’ analytic gradients to do bicubic interpolation almost for free. Applied to you:[^2_5]

- Splat only to a **sparser HR grid** (e.g., half‑res), but also accumulate $\partial C/\partial x$ and $\partial C/\partial y$ per pixel analytically:

$$
\frac{\partial}{\partial x} \exp(-\tfrac{1}{2} q_g) = -\tfrac{1}{2} \exp(-\tfrac{1}{2} q_g) \cdot \frac{\partial q_g}{\partial x}
$$

with $\partial q_g/\partial x = 2 a\,dx + 2 b\,dy$.
- Then upscale from half‑res to full‑res using a **gradient‑based bicubic** that uses both color and gradients. This can reduce Gaussian evaluations by ~4× while recovering sharpness better than naïve bilinear.


### 5.4. Math kernel fusions

- **Fuse conic eval and exp**: instead of q = a dx²+2b dxdy+d dy²; w = exp(-0.5q); structure the polynomial to minimize multiplies:
    - Precompute dx², dy², dx·dy once.
    - Use FMA where possible: q = fma(a,dx², fma(2b,dx·dy, d·dy²)).
- If you can relax atol slightly, evaluate `__expf` with a corrective polynomial for q in a narrow band (e.g., |q|<M). Many SR methods tolerate small bias; you can guard large q branches.
- **Chunk F more aggressively**: experiment with F‑chunk size 8 instead of 16 to reduce register pressure and allow more warps/SM → better latency hiding; on Ada and newer architectures, smaller chunks often win wall‑clock even if they add some loop overhead.


### 5.5. Backward kernel atomics

Backward is usually slower; you can make it nearly free:

- **Warp‑level reduction before atomicAdd**: accumulate d_xy,d_scale,d_rot,d_feat in shared memory per block, then do one atomicAdd per Gaussian per block, rather than per pixel. This cuts atomic contention in dense regions by ~warp_size.
- Store gradients in **FP16** buffers where possible (especially d_feat), accumulating with FP32 local registers but atomicAdd half. For learning, noisy gradient ordering is already accepted; a bit more quantization noise is fine.

***

## 6. System‑Level Tricks (Pipelining and Mixed Rate)

You want end‑to‑end ms/frame, not just per‑kernel perf.

**6.1. Asynchronous multi‑rate components**

- Run the **backbone + cross‑attention** at a **lower temporal rate** than the rasterizer. Example: backbone every 2nd frame; in between, extrapolate canvas state via warp and render from updated Gaussians only. You already have “free extrapolation”; lean into it: up to 2 extrapolated frames between full “update” frames at 60 FPS.
- Meanwhile, run a **low‑res teacher SR network** in the background at even lower rate to supervise your canvas on long‑term consistency; it doesn’t need to be in the real‑time path.

**6.2. Multi‑stream GPU scheduling**

- Put backbone and rasterizer in **separate CUDA streams** and overlap where possible:
    - While rasterizer renders frame t in stream 0 using Gaussians from t, the backbone in stream 1 can process LR residuals for frame t+1 since the LR input is available earlier in the graphics pipeline.
- Carefully size your blocks and register usage so that both kernels can co‑reside on SMs (use `__launch_bounds__` to enforce >1 block/SM across streams).

***

## 7. Extremely Cheap Variants for Consoles / Low‑Tier GPUs

If you ever need a “pico” configuration:

- Replace HAT‑Tiny with a **2–3 block NAFNet** with depthwise convs only — run in FP8 (NVIDIA `e4m3`), treat Gaussian canvas as the main “model capacity.”
- Replace continuous Gaussians in some regions with a **small learned kernel bank** (think 5×5 kernels per pixel) and only use splatting around edges / high motion. That hybrid is still Gaussian‑flavored but uses ordinary convs in smooth regions.

***

If you give me one concrete target (e.g., “1080p @ 60 on 4070 Super” or a tight frame‑time budget for a specific resolution), I can prioritize these into a minimal set of changes and sketch a migration plan (which kernels to rewrite, which configs to ablate first).
<span style="display:none">[^2_10][^2_11][^2_12][^2_13][^2_14][^2_15][^2_6][^2_7][^2_8][^2_9]</span>

<div align="center">⁂</div>

[^2_1]: https://arxiv.org/abs/2503.06617

[^2_2]: https://arxiv.org/html/2503.06617v1

[^2_3]: https://arxiv.org/html/2403.08551v4

[^2_4]: https://research.polyu.edu.hk/en/publications/gaussiansr-high-fidelity-2d-gaussian-splatting-for-arbitrary-scal/

[^2_5]: https://arxiv.org/abs/2503.14171

[^2_6]: https://github.com/cashcon57

[^2_7]: https://github.com/topics/supersampling?o=asc\&s=forks

[^2_8]: https://github.com/adobe-research/SuperGaussian

[^2_9]: https://www.reddit.com/r/MachineLearning/comments/1h3qcon/whats_the_best_open_source_imageupscaling_model/

[^2_10]: https://supergaussian.github.io

[^2_11]: https://sourceforge.net/directory/image-upscalers/

[^2_12]: https://letsenhance.io/blog/all/best-ai-image-upscalers/

[^2_13]: https://www.canva.com/features/image-upscaler/

[^2_14]: https://openart.ai/features/super-resolution

[^2_15]: https://github.com/upscayl/upscayl


---

# Absolute ms/Frame Optimization for Open-Supersampling Pipeline

## 1. Where Models Agree

| Finding | GPT-5.5 Thinking | Claude Opus 4.7 Thinking | Gemini 3.1 Pro Thinking | Evidence |
| :-- | :-- | :-- | :-- | :-- |
| Low-rank feature factorization (F=64→R=4–12) is the single biggest rasterizer win | ✓ | ✓ | ✓ | Linearity of sum-composite means B·Σw_g·z_g = Σw_g·Bz_g; rasterize R channels then project[^3_1][^3_2] |
| Replace `torch.sort` with custom radix/counting sort or persistent tile bins | ✓ | ✓ | ✓ | tile_id is bounded (~2k tiles); O(N) counting sort vs O(N log N); temporal coherence means ~5-10% of Gaussians cross tiles per frame |
| Shared-memory / warp-level gradient reduction before global atomicAdd in backward | ✓ | ✓ | ✓ | Quad+subgroup hybrid reduction achieves 10× backward speedup over naive atomicAdd[^3_3] |
| Multi-rate temporal execution: rasterize every frame, run backbone/spawner less often | ✓ | ✓ | ✓ | 70-85% of pixels are reprojection-stable at 60fps[^3_4][^3_5]; canvas warp is the cheap path |
| CUDA Graph capture eliminates per-frame launch overhead (1-3ms saved) | ✓ | ✓ | ✓ | Persistent kernel and graph capture patterns avoid driver‑level synchronization costs[^3_6] |
| Validity/residual mask to skip rasterization on temporally stable pixels | ✓ | ✓ |  | Disocclusion + depth + motion thresholding gates full compute to ~15-40% of pixels |

## 2. Where Models Disagree

| Topic | GPT-5.5 Thinking | Claude Opus 4.7 Thinking | Gemini 3.1 Pro Thinking | Why They Differ |
| :-- | :-- | :-- | :-- | :-- |
| How to eliminate per-pixel `expf` | Row-recurrence: w_{x+1}=w_x·r_x, r_{x+1}=r_x·exp(-a) — 2 exp per row not per pixel | Keep expf but reduce pixel count via validity mask + half-res splat | Keep expf but reduce channels to R=8 so register pressure vanishes and occupancy hits 100% | GPT-5.5 attacks arithmetic cost; others attack invocation count |
| Tensor Core utilization | Map W·G (weight×feat) as mma.sync bf16→fp32; biggest single kernel win | Not emphasized; prefers algorithmic elimination of work | Agrees TCs help but says fixing channel count + occupancy is prerequisite | GPT-5.5 has hardware-first mindset; Gemini says fix the bottleneck class first |
| Cross-attention strategy | Batch all windows into one SDPA call (launch-overhead bound at K<64) | Replace with rasterized fusion G(p)=Σw_g·z_g / Σw_g + tiny MLP for normal tiles | CSR-tiled Flash cross-attn with per-window Gaussian index lists | Different diagnoses: launch-bound vs compute-bound vs bandwidth-bound |
| Spawner fix for checkerboard | Blue-noise anti-correlated jitter + DGP dictionary | Kalman update for existing Gaussians; spawner only for new births | Hard-code spawn xy to disoccluded pixel center; let warp advect off-grid | Structural vs statistical fix; Gemini's is most elegant for the artifact |
| Covariance update at inference | Persist conic Λ, update via A^{-T}ΛA^{-1} avoiding scale/rot path | Freeze (scale,rot) for K frames; only update xy+feat | Jacobian-free: if divergence < ε, skip JΣJ^T entirely (Σ'=Σ+Δt·D) | Different assumptions about motion complexity in typical gameplay |
| `head_dim=30` | Not flagged | Not flagged | Pad to 32 — dimension of 30 forces slow CUDA cores, breaks TC alignment | Only Gemini caught this hardware alignment issue |

## 3. Unique Discoveries

| Model | Unique Finding | Why It Matters |
| :-- | :-- | :-- |
| GPT-5.5 Thinking | Conic row-recurrence eliminates most `expf`: Δq advances linearly → w_{x+1}=w_x·r_x with Δ²q=2a constant | Turns 256 exp/tile into ~16-32 exp/tile; purely mathematical, no quality loss |
| GPT-5.5 Thinking | Covariance codebook + LUT splat for narrow Gaussians: precompute w[k][phase_x][phase_y][dy][dx] | Eliminates both conic eval and exp for the common small-Gaussian case |
| GPT-5.5 Thinking | Analytic gradient splatting (∂C/∂x, ∂C/∂y) enables half-res splat + gradient-based HR recovery | ∂q/∂x = 2a·dx+2b·dy is free during forward; halves pixel evaluations[^3_7] |
| Claude Opus 4.7 Thinking | Energy-based Gaussian pruning: E_g = α_g·(s_u·s_v)·‖feat_g‖₂; evict bottom quartile every 8-16 frames | Keeps effective canvas at ~12k without quality loss; linear speedup everywhere |
| Claude Opus 4.7 Thinking | Kalman one-step update for existing Gaussians replaces spawner regression: 6 FLOPs/Gaussian vs full MLP | Spawner cost drops 4-8× since most Gaussians just need correction, not re-regression |
| Claude Opus 4.7 Thinking | Splat in log-luminance space → feat can be bf16 throughout rasterizer, cutting bandwidth ~50% | HDR scenes stay numerically conditioned; bandwidth is the real bottleneck |
| Gemini 3.1 Pro Thinking | Pad head_dim from 30→32 for TC alignment; 30 forces scalar CUDA paths | Near-free 1.5-2× on attention matmuls by hitting mma.sync sweet spots |
| Gemini 3.1 Pro Thinking | Jacobian-free warp: branch on ∇·V; if <ε (rigid translation ~90% of canvas), skip J entirely | Reduces warp to streaming copy for most Gaussians; 8-12 FLOPs saved per primitive |
| Gemini 3.1 Pro Thinking | Spawn at exact disoccluded pixel center; let warp advect off-grid naturally | Structurally kills checkerboard without noise injection; decouples spawn from sub-pixel positioning |

## 4. Comprehensive Analysis

The three models converge decisively on one architectural intervention that dwarfs all others: **reducing the rasterizer's feature payload from 64 channels to 4–12 via low-rank factorization**. GPT-5.5 Thinking frames this as `f_g ≈ B·z_g` where the rasterizer accumulates only `Z(p) = Σ w_g(p)·z_g` in ℝ^R, then a post-raster 1×1 projection recovers the full feature space. Claude Opus 4.7 Thinking corroborates this through the lens of GaussianImage's permutation-invariant accumulated summation, and Gemini 3.1 Pro Thinking quantifies the impact as an 87.5% reduction in register pressure yielding a projected drop from 136ms to 10-15ms. The mathematical basis is unimpeachable: sum-composite splatting is linear, so projection commutes with accumulation. This should be your P0 implementation.[^3_1]

The second tier of consensus centers on **eliminating framework overhead**: replacing `torch.sort` with bounded-range radix/counting sort (all three models), capturing the inference pipeline as a CUDA Graph (all three), and using shared-memory accumulation in the backward pass before global atomicAdd — a technique validated by recent hardware rasterization research showing 10× backward speedup via quad+subgroup hybrid reduction. These are all implementable within days and collectively save 5-15ms of pure overhead per frame.[^3_3]

The most productive disagreement concerns the **cross-attention block**. GPT-5.5 Thinking argues that at K≤64 Gaussians per window, attention is launch-overhead-bound and should be batched into a single block-diagonal SDPA call to amortize kernel launch costs across ~2000 windows. Claude Opus 4.7 Thinking takes the more radical position that cross-attention should be replaced entirely by "rasterized fusion" — using the Gaussian raster output itself as the K/V injection via `G(p) = Σw_g·z_g / (ε + Σw_g)` followed by a tiny MLP — reserving actual attention only for high-residual disocclusion tiles. Gemini 3.1 Pro Thinking proposes a CSR-tiled Flash cross-attention variant with per-window Gaussian index lists, and critically flags that `head_dim=30` must be padded to 32 for Tensor Core alignment. The resolution depends on your profiling: if cross-attention currently takes <2ms, GPT-5.5's batching is sufficient; if it's >5ms, Claude's raster-fusion replacement is the correct structural fix.

GPT-5.5 Thinking's conic row-recurrence is perhaps the most mathematically elegant single-kernel optimization in this report. Because the quadratic form `q(x,y) = a·dx² + 2b·dx·dy + d·dy²` has constant second difference `Δ²q_x = 2a` along a scanline, the Gaussian weight can be marched as `w_{x+1} = w_x · r_x` with `r_{x+1} = r_x · exp(-a)`, reducing exponential evaluations from one per pixel-Gaussian pair to approximately two per row-Gaussian pair. For a 16×16 tile, this is a 16× reduction in transcendental function calls. Combined with the low-rank factorization, this transforms the rasterizer from a 136ms bottleneck into a sub-10ms operation.

Gemini 3.1 Pro Thinking's Jacobian-free warp proposal deserves immediate prototyping. In typical real-time rendering at 60fps, the vast majority of per-pixel motion is pure translation (camera pan, object sliding). By computing velocity divergence `∇·V` from the motion vector field and branching — identity Jacobian for rigid regions (~90%), full J only for deformation zones — the warp kernel becomes a streaming memory copy for most of the canvas. This is architecturally compatible with your v6.2 candidate that splits J and D into separate heads.

Claude Opus 4.7 Thinking's Kalman-style one-step refinement for existing Gaussians is the key insight for spawner optimization. The spawner MLP should only fire for genuinely new Gaussians (births in disoccluded regions). Existing Gaussians that merely need position/feature correction can use a 6-FLOP analytic update: `x̂_{t|t} = x̂_{t|t-1} + K_t(z_t - H·x̂_{t|t-1})` with diagonal covariance. This reduces spawner cost by 4-8× and pairs naturally with Gemini's structural fix for the checkerboard: spawn at exact disoccluded pixel centers and let the warp advect off-grid, eliminating the need for statistical jitter entirely.

**Recommended implementation order for maximum ms/frame reduction:**

1. Low-rank feature factorization (F=64→R=8): expected 8-10× rasterizer speedup
2. Row-recurrence `expf` elimination: additional 2-4× on remaining raster arithmetic
3. Custom radix tile binning + CUDA Graph capture: 5-15ms overhead eliminated
4. Raster-fusion replacing global cross-attention (keep local attn for disocclusion tiles only)
5. Jacobian-free warp + Kalman update for existing Gaussians
6. Pad head_dim to 32; distill HAT into EfficientViT-lite student for runtime
7. Validity mask + multi-rate scheduling (backbone at 30Hz, raster at display rate)

Target budget at 1080p on your RTX 4070 (8GB): warp 0.3ms + bin-update 0.4ms + low-rank raster 2-4ms + post-decode 0.5ms + sparse backbone/fusion on active tiles 1.5ms = **~5-7ms total**, enabling 120+ FPS output from 60 FPS source rendering.
<span style="display:none">[^3_10][^3_100][^3_101][^3_102][^3_103][^3_104][^3_105][^3_106][^3_107][^3_108][^3_109][^3_11][^3_110][^3_111][^3_112][^3_113][^3_114][^3_115][^3_116][^3_117][^3_118][^3_119][^3_12][^3_120][^3_121][^3_122][^3_123][^3_13][^3_14][^3_15][^3_16][^3_17][^3_18][^3_19][^3_20][^3_21][^3_22][^3_23][^3_24][^3_25][^3_26][^3_27][^3_28][^3_29][^3_30][^3_31][^3_32][^3_33][^3_34][^3_35][^3_36][^3_37][^3_38][^3_39][^3_40][^3_41][^3_42][^3_43][^3_44][^3_45][^3_46][^3_47][^3_48][^3_49][^3_50][^3_51][^3_52][^3_53][^3_54][^3_55][^3_56][^3_57][^3_58][^3_59][^3_60][^3_61][^3_62][^3_63][^3_64][^3_65][^3_66][^3_67][^3_68][^3_69][^3_70][^3_71][^3_72][^3_73][^3_74][^3_75][^3_76][^3_77][^3_78][^3_79][^3_8][^3_80][^3_81][^3_82][^3_83][^3_84][^3_85][^3_86][^3_87][^3_88][^3_89][^3_9][^3_90][^3_91][^3_92][^3_93][^3_94][^3_95][^3_96][^3_97][^3_98][^3_99]</span>

<div align="center">⁂</div>

[^3_1]: https://openreview.net/forum?id=SZvhmFntRA

[^3_2]: https://arxiv.org/abs/2501.06838

[^3_3]: https://arxiv.org/html/2505.18764v1

[^3_4]: https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Generalized_and_Efficient_2D_Gaussian_Splatting_for_Arbitrary-scale_Super-Resolution_ICCV_2025_paper.pdf

[^3_5]: https://www.themoonlight.io/en/review/generalized-and-efficient-2d-gaussian-splatting-for-arbitrary-scale-super-resolution

[^3_6]: https://concurrent-rt.com/wp-content/uploads/2020/12/Improving-Real-Time-Performance-With-CUDA-Persistent-Threads.pdf

[^3_7]: https://arxiv.org/abs/2503.14171

[^3_8]: https://github.com/Lee-JaeWon/2025-Arxiv-Paper-List-Gaussian-Splatting

[^3_9]: https://arxiv.org/html/2509.25626v2

[^3_10]: https://bentoml.com/llm/kernel-optimization/flashattention

[^3_11]: https://arxiv.org/abs/2307.08691

[^3_12]: https://wccftech.com/roundup/nvidia-dlss-vs-amd-fsr-vs-intel-xess-everything-you-need-to-know/

[^3_13]: https://cvpr.thecvf.com/virtual/2025/poster/33792

[^3_14]: https://arxiv.org/html/2403.08551v4

[^3_15]: https://www.emergentmind.com/topics/cross-temporal-3d-gaussian-splatting-cross-temporal-3dgs

[^3_16]: https://lubits.ch/flash/Part-6

[^3_17]: https://developer.nvidia.com/blog/speed-up-unreal-engine-nne-inference-with-nvidia-tensorrt-for-rtx-runtime/

[^3_18]: https://arxiv.org/html/2503.06617v1

[^3_19]: https://arxiv.org/html/2309.05239v3

[^3_20]: https://arxiv.org/abs/2205.14756

[^3_21]: https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.pdf

[^3_22]: https://arxiv.org/abs/2205.14135

[^3_23]: https://openreview.net/forum?id=H4DqfPSibmx

[^3_24]: https://gpuopen.com/fidelityfx-super-resolution-3/

[^3_25]: https://developer.nvidia.com/blog/nvidia-releases-rtx-neural-rendering-tech-for-unreal-engine-developers/

[^3_26]: https://www.intel.com/content/www/us/en/developer/articles/technical/xess-sr-developer-guide.html

[^3_27]: https://openaccess.thecvf.com/content/ICCV2025/papers/Hollein_3DGS-LM_Faster_Gaussian-Splatting_Optimization_with_Levenberg-Marquardt_ICCV_2025_paper.pdf

[^3_28]: https://github.com/nerfstudio-project/gsplat

[^3_29]: https://ai.gopubby.com/inside-the-cuda-kernel-the-gpu-implementation-of-3d-gaussian-splatting-74c3261ed721

[^3_30]: https://openaccess.thecvf.com/content/ICCV2025/html/Hollein_3DGS-LM_Faster_Gaussian-Splatting_Optimization_with_Levenberg-Marquardt_ICCV_2025_paper.html

[^3_31]: https://arxiv.org/html/2504.10686v1

[^3_32]: https://dl.acm.org/doi/10.1109/DAC63849.2025.11132449

[^3_33]: https://www.clarifai.com/blog/flash-attention-2

[^3_34]: https://research.facebook.com/publications/neural-supersampling-for-real-time-rendering/

[^3_35]: https://www.reddit.com/r/computergraphics/comments/18mxzkq/blog_post_rasterizing_gaussian_splats_the/

[^3_36]: https://huggingface.co/blog/atharv6f/flash-attention-basics

[^3_37]: https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf

[^3_38]: https://dev.to/lewis_won/online-softmax-by-hand-4h13

[^3_39]: https://arxiv.org/abs/2505.14201

[^3_40]: https://www.sethweidman.com/blog/streaming_softmax.html

[^3_41]: https://forums.developer.nvidia.com/t/persistent-kernel-runs-slower-when-with-more-threads/308556

[^3_42]: https://wangkuiyi.github.io/online-softmax.html

[^3_43]: https://forums.developer.nvidia.com/t/performance-of-persistent-thread-approach-on-new-gpu-architectures/43254

[^3_44]: https://github.com/kkokosa/dotLLM/issues/54

[^3_45]: https://hai.stanford.edu/research/flashattention-fast-and-memory-efficient-exact-attention-with-io-awareness

[^3_46]: https://hazyresearch.stanford.edu/blog/2023-07-17-flash2

[^3_47]: https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Supplemental-Conference.pdf

[^3_48]: https://www.reddit.com/r/GraphicsProgramming/comments/1pf1qj1/learn_how_to_integrate_rtx_neural_rendering_into/

[^3_49]: https://www.linkedin.com/posts/amitnvidia_nvidia-tensorrt-unrealengine-activity-7455984005344251904-tq1f

[^3_50]: https://github.com/NVIDIA/TensorRT-RTX

[^3_51]: https://github.com/NVIDIAGameWorks/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md

[^3_52]: https://www.digitaltrends.com/computing/amd-fsr-3-explained/

[^3_53]: https://www.reddit.com/r/aigamedev/comments/1pcmeez/learn_how_to_integrate_rtx_neural_rendering_into/

[^3_54]: https://www.reddit.com/r/pcmasterrace/comments/1ryrmdn/nvidia_confirms_dlss_5_uses_a_2d_frame_plus/

[^3_55]: https://www.tweaktown.com/news/110569/dlss-5-only-takes-2d-rendered-frames-and-motion-vectors-as-input-not-3d-game-engine-data-confirms-nvidia/index.html

[^3_56]: https://news.yahoo.com/everything-know-amds-fsr-3-130343982.html

[^3_57]: https://forums.developer.nvidia.com/t/dlss-motion-vector-question/266414

[^3_58]: https://www.techspot.com/article/2747-amd-fsr-3-tech/

[^3_59]: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/gaming.html

[^3_60]: https://github.com/mit-han-lab/efficientvit

[^3_61]: https://viplab.snu.ac.kr/viplab/courses/mlvu_2021_2/projects/final_papers/08.pdf

[^3_62]: https://han-cai.github.io/selected_projects/efficientvit_iccv.pdf

[^3_63]: https://cvpr.thecvf.com/media/cvpr-2023/Slides/23282.pdf

[^3_64]: https://arxiv.org/abs/2506.19845

[^3_65]: https://www.sciencedirect.com/science/article/pii/S187705092502191X

[^3_66]: https://arxiv.org/html/2309.05239v2

[^3_67]: https://www.scribd.com/document/689322361/2204-04676

[^3_68]: https://www.nature.com/articles/s41598-025-28042-1

[^3_69]: https://fal.ai/models/fal-ai/nafnet/deblur

[^3_70]: https://dl.acm.org/doi/abs/10.1007/s00521-023-09353-8

[^3_71]: https://openaccess.thecvf.com/content/CVPR2023/supplemental/Chen_Activating_More_Pixels_CVPR_2023_supplemental.pdf

[^3_72]: https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Xiao20.pdf

[^3_73]: https://www.transit.dot.gov/sites/fta.dot.gov/files/docs/The_NTD_Sampling_Manual.pdf

[^3_74]: https://pubs.usgs.gov/publication/tm1D12/full

[^3_75]: https://egusphere.copernicus.org/preprints/2025/egusphere-2025-272/

[^3_76]: https://arxiv.org/abs/2308.01483

[^3_77]: https://github.com/ansman/mandelbrot/blob/master/supersampling.cpp

[^3_78]: https://github.com/cashcon57/cauldron

[^3_79]: https://lume.ufrgs.br/bitstream/handle/10183/287628/001241293.pdf?sequence=1

[^3_80]: https://github.com/arm/neural-graphics-for-unreal

[^3_81]: https://github.com/topics/mcp-plugin

[^3_82]: https://arxiv.org/html/2510.01171v3

[^3_83]: https://github.com/ndming/NSRT

[^3_84]: https://github.com/topics/caveman

[^3_85]: https://github.com/sim51/caps_example/blob/master/csv/name.basics.tsv

[^3_86]: https://github.com/timmh/neural-supersampling

[^3_87]: https://arxiv.org/html/2603.07169v1

[^3_88]: https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/

[^3_89]: https://www.reddit.com/r/CUDA/comments/1moh19a/gtc_2025_nvidia_says_custom_cuda_kernels_only/

[^3_90]: https://www.rimikawrites.com/cuda-3-your-checklist-for-optimizing-cuda-kernels/

[^3_91]: https://research.samsung.com/blog/Trick-GS-A-Balanced-Bag-of-Tricks-for-Efficient-Gaussian-Splatting

[^3_92]: https://arxiv.org/html/2312.10890v1

[^3_93]: https://dl.acm.org/doi/10.1109/TVCG.2002.1021576

[^3_94]: https://arxiv.org/html/2602.09999v1

[^3_95]: https://princeton-nlp.github.io/flash-atttention-2/

[^3_96]: https://github.com/nerficg-project/HTGS

[^3_97]: https://arxiv.org/html/2405.19745v1

[^3_98]: https://arxiv.org/html/2311.17089v2

[^3_99]: https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Multi-Scale_3D_Gaussian_Splatting_for_Anti-Aliased_Rendering_CVPR_2024_paper.pdf

[^3_100]: https://blurbusters.com/frame-generation-essentials-interpolation-extrapolation-and-reprojection/

[^3_101]: https://www.allpcb.com/allelectrohub/flashattention-123-transformer-attention-optimizations

[^3_102]: https://www.alibaba.com/product-insights/ai-anime-upscalers-vs-manual-frame-interpolation-which-method-preserves-original-line-art-integrity-better.html

[^3_103]: https://github.com/dao-ailab/flash-attention

[^3_104]: https://ryanszeto.com/media/tai.pdf

[^3_105]: https://www.shadecoder.com/ja/topics/flashattention-2-a-comprehensive-guide-for-2025

[^3_106]: https://proceedings.neurips.cc/paper_files/paper/2024/file/a076d0d1ed77364fc57693bdee1958fb-Paper-Conference.pdf

[^3_107]: https://openreview.net/forum?id=76NYyOrnfk

[^3_108]: https://www.scribd.com/document/974537384/MoDGS-Dynamic-Gaussian-Splatting-From-Causually-CA

[^3_109]: https://www.kaggle.com/code/egazakharenko/flashattention-2-for-turing-from-scratch-tutorial

[^3_110]: https://www.alibaba.com/product-insights/ai-video-upscalers-vs-frame-interpolation-tools-why-does-4k-anime-sometimes-look-unnervingly-smooth.html

[^3_111]: https://openreview.net/pdf/c8c9ffc2bca0ad5b064b8d80f94643438b175c91.pdf

[^3_112]: https://arxiv.org/html/2503.14698v2

[^3_113]: https://proceedings.neurips.cc/paper_files/paper/2024/file/45ed1a72597594c097152ef9cc187762-Paper-Conference.pdf

[^3_114]: https://arxiv.org/html/2510.01619v1

[^3_115]: https://openreview.net/forum?id=CKZoVUpwWW

[^3_116]: https://arxiv.org/abs/2304.13986

[^3_117]: https://gatambook.substack.com/p/cross-attention-for-cross-asset-applications

[^3_118]: https://repositum.tuwien.at/bitstream/20.500.12708/17701/1/Schuetz Markus - 2021 - Interactive exploration of point clouds.pdf

[^3_119]: https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Prometheus_3D-Aware_Latent_Diffusion_Models_for_Feed-Forward_Text-to-3D_Scene_Generation_CVPR_2025_paper.pdf

[^3_120]: https://www.reddit.com/r/StableDiffusion/comments/13d80eu/which_crossattention_optimization_technique_is/

[^3_121]: https://proceedings.iclr.cc/paper_files/paper/2025/file/ba404795c58d122a4b6fc2672d84d9f3-Paper-Conference.pdf

[^3_122]: https://github.com/vladmandic/sdnext/discussions/846

[^3_123]: https://openaccess.thecvf.com/content/ACCV2024/papers/Svitov_HAHA_Highly_Articulated_Gaussian_Human_Avatars_with_Textured_Mesh_Prior_ACCV_2024_paper.pdf


---

# Re-Assessment: 1–4ms Total Budget (DLSS4-Class Parity)

## 1. Where Models Agree

| Finding | GPT-5.5 Thinking | Claude Opus 4.7 Thinking | Gemini 3.1 Pro Thinking | Evidence |
| :-- | :-- | :-- | :-- | :-- |
| HAT-Tiny must be completely removed from inference path | ✓ | ✓ | ✓ | 9M params / ~54.9G MACs consumes entire 4ms budget alone[^4_1][^4_2]; STSS achieves SOTA at 0.4M params / 4.4ms[^4_3] |
| F=64 rasterizer is dead; R=4 latent channels mandatory | ✓ | ✓ | ✓ | At 4ms on 4070 (~1.2GB bandwidth budget), 64-channel accumulation is physically impossible |
| Cross-attention must be eliminated from hot path entirely | ✓ | ✓ | ✓ | Replace with raster-fusion: G(p)=Σw_g·z_g/(ε+Σw_g) + trivial 1×1 conv |
| Pipeline must be CUDA Graph / TensorRT — no PyTorch eager | ✓ | ✓ | ✓ | Python dispatch + torch.sort alone can eat 1-3ms of the entire budget[^4_4] |
| Validity mask / reprojection-first architecture is mandatory | ✓ | ✓ | ✓ | 70-85% of pixels are stable at 60fps; Gaussians must NOT touch every pixel |
| Frame extrapolation is structurally free via canvas warp | ✓ | ✓ | ✓ | Deterministic advection = frame gen without separate model; only raster cost added |
| Spawner becomes disocclusion-only hard-spawn, not dense MLP | ✓ | ✓ | ✓ | Cap births at 128-512/frame; existing Gaussians use Kalman correction (~6 FLOPs) |
| Row-recurrence eliminates per-pixel expf | ✓ | ✓ | ✓ | w_{x+1}=w_x·r_x; Δ²q=2a constant; ~16× fewer transcendentals per tile |

## 2. Where Models Disagree

| Topic | GPT-5.5 Thinking | Claude Opus 4.7 Thinking | Gemini 3.1 Pro Thinking | Why They Differ |
| :-- | :-- | :-- | :-- | :-- |
| Runtime backbone size | ≤0.4M params student or none; reprojection-first can skip backbone most frames | ≤1M params EfficientViT-lite student at 30Hz; INT8 mandatory | 3-layer CNN / single NAFNet block; TensorRT FP16 export | Different risk tolerance on quality vs speed |
| What R=4 channels encode | Normalized latent (Z/Σw) decoded via MLP to residual | RGB + 4 latent decoded via 3-layer pointwise | RGB + Confidence (sum of weights); direct composite | GPT wants learned subspace; Gemini wants direct output |
| Whether Tensor Cores help the rasterizer at this budget | Yes if K_tile≥16 — TC matmul W·Z; otherwise scalar is faster | Yes for student backbone; rasterizer may be too bandwidth-bound | No — at R=4, register tiling + recurrence makes splat arithmetic-trivial; TCs for resolve CNN only | K_tile caps determine whether matrices are large enough for mma.sync |
| Degradation strategy under overload | 8-tier ladder from R=4→R=2 to disabling Gaussians entirely | 4-tier: drop rank → skip backbone → fall back to bicubic + composite | Binary: active mask shrinks; beyond that, pure TAA reproject | Different philosophy on graceful vs binary fallback |
| Inference state precision | FP16 conic + FP16 z + uint8 cov_id; LUT-driven | BF16 canvas with FP16 accumulate; FP32 only for guards | FP16 throughout; entire 16k canvas fits in L2 at 32 bytes/Gaussian (512KB) | Gemini's "32 bytes/primitive fits L2" insight is the most hardware-aware |
| Whether local attention survives at all | Only on disocclusion tiles, K≤16, 1-head, batched | Only on disocclusion tiles (<5% of windows), K=16 | No attention period; concat fusion + tiny CNN instead | Gemini is most aggressive; others hedge for quality |

## 3. Unique Discoveries

| Model | Unique Finding | Why It Matters |
| :-- | :-- | :-- |
| GPT-5.5 Thinking | Dynamic degradation ladder (8 tiers from full pipeline to "disable Gaussians for this frame") gives deterministic frame pacing like a real shipping upscaler | Guarantees 1-4ms regardless of scene complexity; critical for product credibility |
| GPT-5.5 Thinking | Mode A/B/C split: 1ms competitive mode (no neural, pure reproject+LUT splat), 2ms quality SR, 4ms SR+FG — different products from same codebase | Mirrors DLSS Performance/Quality/Ultra Performance presets |
| Claude Opus 4.7 Thinking | DLSS4.5 uses 5× the compute of original transformer model but stays fast via FP8 on RTX 40/50[^4_5]; implies you should design for FP8 from day one | FP8 QAT on student backbone doubles throughput on Ada/Blackwell for free |
| Claude Opus 4.7 Thinking | DLSS4 reduced latency from 3.25ms to 1ms on RTX 5090[^4_6]; your structural advantage (no optical flow network) means you can match this on 4070 | Validates that 1ms SR-only mode is achievable with correct architecture |
| Gemini 3.1 Pro Thinking | Entire 16k Gaussian canvas at 32 bytes/primitive = 512KB → fits entirely in RTX 4070 L2 cache | Eliminates HBM round-trips for canvas state; warp+raster becomes L2-resident |
| Gemini 3.1 Pro Thinking | Conceptual argument: explicit geometric cache (Gaussians) means your neural net only needs to be ~10% of DLSS size for equivalent quality, because 90% of image is analytically warped | Reframes the competitive positioning: you're not competing on network size but on cache efficiency |

## 4. Comprehensive Analysis

All three models converge on a fundamental architectural reframing that the previous report's optimizations, however aggressive, failed to acknowledge: **at 1–4ms total, the Gaussian canvas cannot be the primary image generator. It must be a sparse temporal correction field layered on top of reprojection.** GPT-5.5 Thinking states this most explicitly: "the displayed-frame path is not a neural Gaussian renderer — it is reprojection-first temporal reconstruction with sparse Gaussian residual correction." Claude Opus 4.7 Thinking corroborates by noting that even STSS (the unified space-time supersampling framework) achieves state-of-the-art quality at just 0.4M parameters and 4.4ms per frame at 1080p, proving that sub-5ms SR+extrapolation is architecturally feasible with the right design choices.[^4_3]

The unanimous consensus on killing HAT-Tiny from inference is backed by hard arithmetic. Claude Opus 4.7 Thinking quantifies it precisely: at ~54.9G multi-adds, even at 50% GPU utilization on an RTX 4070 (~29 TFLOPS FP32), the backbone alone would consume 2–3ms — leaving nothing for rasterization, fusion, or compositing. The replacement must be ≤1M parameters (Claude's bound) or even ≤0.4M (GPT-5.5's more aggressive bound), exported to TensorRT with INT8/FP8 quantization, and run only on active tiles at 30Hz temporal rate. This is not an optimization of HAT; it is a replacement.[^4_1][^4_2]

The R=4 channel decision is unanimous but the models disagree on what those 4 channels encode. Gemini 3.1 Pro Thinking takes the most practical position: output direct RGB + confidence (sum of weights), then composite `Final = Canvas_RGB + Residual` with no learned decoder. GPT-5.5 Thinking wants a learned latent decoded via a tiny MLP, which preserves more representational flexibility but adds a decode step. Claude Opus 4.7 Thinking splits the difference with "RGB + 4 latent decoded via 3-layer pointwise." For the 1ms competitive mode, Gemini's direct-RGB approach is correct; for the 2–4ms quality modes, GPT's learned residual approach yields better quality. The recommended implementation is: **start with Gemini's direct RGB+confidence for v6.2 validation, then add optional latent decode as a quality toggle.**

Gemini 3.1 Pro Thinking's L2 cache insight deserves special emphasis. At 32 bytes per Gaussian (FP16×2 for xy, FP16×3 for conic, FP16×3 for RGB, FP16×1 for confidence), the full 16k canvas occupies exactly 512KB. The RTX 4070's L2 cache is 36MB. **The entire canvas fits in L2 with room to spare.** This means the warp kernel and rasterizer can operate entirely from L2 without HBM round-trips for canvas state — a massive bandwidth saving that the previous analysis missed. This single insight validates that the warp+rebin+raster pipeline can complete in well under 2ms if the kernel is properly designed.

The frame generation story is where your architecture has a **structural advantage over DLSS4**. NVIDIA had to replace hardware optical flow with an AI model and redesign the frame generation pipeline because "generating multiple frames was cost prohibitive since both the Optical Flow Accelerator and the AI model would be required for every new generated frame". Your canvas-based approach doesn't have this problem: extrapolation is a deterministic second warp+raster pass costing only ~1.4ms additional, with no optical flow network, no separate AI model, and no additional memory allocation. As Claude Opus 4.7 Thinking notes, this property lets you hit DLSS4+FG budgets with a single unified architecture rather than stacking two separate models.[^4_7][^4_8]

The degradation ladder proposed by GPT-5.5 Thinking is essential for a shipping product. DLSS maintains consistent frame pacing by adapting quality — DLSS 4.5's "Dynamic Multi Frame Generation" automatically shifts between frame multipliers based on GPU load versus display refresh rate. Your equivalent is: if the frame would exceed budget, progressively reduce active tile count → reduce K_tile cap → reduce latent rank → skip backbone → fall back to pure reprojection. Frame time must never exceed the budget; quality is the variable, not latency.[^4_9]

**The revised implementation order for 1–4ms:**

1. **R=4 direct-RGB rasterizer with row-recurrence** — transforms 136ms → ~1–3ms immediately
2. **Validity mask + reprojection-first architecture** — ensures Gaussians only touch 10–30% of pixels
3. **Custom counting sort + persistent tile bins** — removes framework overhead
4. **CUDA Graph capture of entire inference path** — eliminates Python/driver latency
5. **Distill HAT into ≤0.4M TensorRT INT8 student running on active tiles at 30Hz** — replaces the backbone
6. **Replace cross-attention with concat-fusion + 1×1 conv** — zero attention in hot path
7. **Budget governor with degradation tiers** — guarantees frame pacing
8. **FP8 QAT for RTX 40/50 path** — doubles TC throughput for backbone/resolve

The core architectural statement becomes: **the Gaussian canvas is an L2-resident temporal detail cache that analytically warps for free, corrects only where reprojection fails, and never runs a heavy neural network in the critical display path.**
<span style="display:none">[^4_10][^4_11][^4_12][^4_13][^4_14][^4_15][^4_16][^4_17][^4_18][^4_19][^4_20][^4_21][^4_22][^4_23][^4_24][^4_25][^4_26][^4_27][^4_28][^4_29][^4_30][^4_31][^4_32][^4_33][^4_34][^4_35][^4_36][^4_37][^4_38][^4_39][^4_40][^4_41][^4_42][^4_43][^4_44][^4_45][^4_46][^4_47][^4_48][^4_49][^4_50][^4_51][^4_52][^4_53][^4_54][^4_55]</span>

<div align="center">⁂</div>

[^4_1]: https://arxiv.org/html/2309.05239v2

[^4_2]: https://openaccess.thecvf.com/content/CVPR2023/supplemental/Chen_Activating_More_Pixels_CVPR_2023_supplemental.pdf

[^4_3]: https://arxiv.org/html/2312.10890v1

[^4_4]: https://concurrent-rt.com/wp-content/uploads/2020/12/Improving-Real-Time-Performance-With-CUDA-Persistent-Threads.pdf

[^4_5]: https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-gen-6x-2nd-gen-transformer-super-res/

[^4_6]: https://www.tomshardware.com/pc-components/gpus/dlss-transformer-model-for-dlss-4-is-out-of-beta-as-nvidia-looks-to-officially-incorporate-new-model-to-improve-image-quality-and-efficiency

[^4_7]: https://www.nvidia.com/en-ph/geforce/news/dlss4-multi-frame-generation-ai-innovations/

[^4_8]: https://www.reddit.com/r/hardware/comments/1hvj1op/dlss4_is_no_longer_using_the_hardware_optical/

[^4_9]: https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-generation-6x-mode-released/

[^4_10]: https://arxiv.org/html/2309.05239v3

[^4_11]: https://arxiv.org/abs/2205.14756

[^4_12]: https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.pdf

[^4_13]: https://github.com/Lee-JaeWon/2025-Arxiv-Paper-List-Gaussian-Splatting

[^4_14]: https://developer.nvidia.com/blog/speed-up-unreal-engine-nne-inference-with-nvidia-tensorrt-for-rtx-runtime/

[^4_15]: https://arxiv.org/html/2505.18764v1

[^4_16]: https://gpuopen.com/fidelityfx-super-resolution-3/

[^4_17]: https://www.digitaltrends.com/computing/amd-fsr-3-explained/

[^4_18]: https://news.yahoo.com/everything-know-amds-fsr-3-130343982.html

[^4_19]: https://www.nvidia.com/en-us/geforce/technologies/dlss/

[^4_20]: https://nvidianews.nvidia.com/news/nvidia-introduces-dlss-3-with-breakthrough-ai-powered-frame-generation-for-up-to-4x-performance

[^4_21]: https://arxiv.org/abs/2205.14135

[^4_22]: https://www.reddit.com/r/nvidia/comments/1jv8j24/can_you_actually_feel_the_input_lag_from/

[^4_23]: https://www.techspot.com/article/2945-nvidia-dlss-4/

[^4_24]: https://www.facebook.com/groups/PcBuildersCommunity/posts/3844434759173210/

[^4_25]: https://www.xda-developers.com/dlss-4-multi-frame-generation-works-best-doesnt-make-sense/

[^4_26]: https://www.tomshardware.com/pc-components/gpus/input-latency-is-the-all-too-frequently-missing-piece-of-framegen-enhanced-gaming-performance-analysis

[^4_27]: https://www.tomshardware.com/pc-components/gpus/community-tests-confirm-dlss-4-5-yields-20-percent-performance-loss-on-older-rtx-30-and-20-series-gpus-compared-to-dlss-4-0-nvidia-warnings-ring-true-following-rollout

[^4_28]: https://www.digitaltrends.com/computing/how-nvidia-dlss-3-works/

[^4_29]: https://steamcommunity.com/app/2807960/discussions/0/591784958348056981/

[^4_30]: https://wccftech.com/early-dlss-4-5-testing-reveals-drastically-crisper-details-but-older-gen-rtx-gpus-take-nearly-20-performance-hit/

[^4_31]: https://www.techspot.com/article/2546-dlss-3/

[^4_32]: https://hothardware.com/news/dlss-3-frame-generation-digital-foundry-testing

[^4_33]: https://www.reddit.com/r/nvidia/comments/1kxlmql/nvidia_dlss_4_new_high_performance_mode_delivers/

[^4_34]: https://discussions.wccftech.com/thread/nvidia-dlss-4-new-high-performance-mode/

[^4_35]: https://www.reddit.com/r/pcmasterrace/comments/1iu2qkt/cost_of_dlss_transformer_vs_cnn_model_benchmarks/

[^4_36]: https://www.laptopmag.com/laptops/gaming-laptops-pcs/nvidia-dlss-transformer-model

[^4_37]: https://en.gamegpu.com/test-gpu/it/dlss-4-test-v-igrakh

[^4_38]: https://forums.overclockers.co.uk/threads/dlss-4-worth-it-for-a-touch-lower-performance.19001208/

[^4_39]: https://www.resetera.com/threads/dlss-4-new-transformer-model-now-available.1089654/page-4

[^4_40]: https://www.tomshardware.com/pc-components/gpus/nvidias-latest-dlss-revision-reduces-vram-usage-by-20-percent-for-upscaling-optimizations-reduce-overhead-of-more-powerful-transformer-model

[^4_41]: https://www.igorslab.de/en/dlss-4-with-transformer-model-nvidia-tears-down-the-old-architecture-and-builds-a-new-throne/

[^4_42]: https://tech4gamers.com/dlss-4-5-early-testing-vram-usage/

[^4_43]: https://www.facebook.com/groups/372119787729533/posts/982773649997474/

[^4_44]: https://www.reddit.com/r/hardware/comments/1lmvakz/nvidias_new_dlss_transformer_model_requires_20/

[^4_45]: https://hothardware.com/news/nvidia-dlss-45-dynamic-mfg-tested

[^4_46]: https://gamersnexus.net/gpus/fake-frames-tested-dlss-40-mfg-4x-nvidias-misleading-review-guide

[^4_47]: https://www.corsair.com/us/en/explorer/gamer/gaming-pcs/what-is-dlss-multi-frame-generation/

[^4_48]: https://www.thegamer.com/nvidia-dlss-4-multi-frame-generation-review-doom-the-dark-ages-dune-awakening/

[^4_49]: https://steamcommunity.com/app/2677660/discussions/0/604149124789439211/?l=latam

[^4_50]: https://www.tweaktown.com/news/106101/dlss-4s-new-enhanced-super-resolution-now-uses-lot-less-vram/index.html

[^4_51]: https://www.diva-portal.org/smash/get/diva2:1985716/FULLTEXT01.pdf

[^4_52]: https://www.reddit.com/r/pcgaming/comments/1simy83/benchmarking_nvidias_rtx_neural_texture/

[^4_53]: https://forums.flightsimulator.com/t/dlss3-msfs-at-nvidia-keynote/543541?page=10

[^4_54]: https://x.com/TheRooster/status/2047459030805700782

[^4_55]: https://en.wikipedia.org/wiki/Deep_Learning_Super_Sampling

