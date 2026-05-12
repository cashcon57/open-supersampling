# Absolute ms/Frame Optimization for Open-Supersampling Pipeline

## 1. Where Models Agree

| Finding | GPT-5.5 Thinking | Claude Opus 4.7 Thinking | Gemini 3.1 Pro Thinking | Evidence |
| :-- | :-- | :-- | :-- | :-- |
| Low-rank feature factorization (F=64→R=4–12) is the single biggest rasterizer win | ✓ | ✓ | ✓ | Linearity of sum-composite means B·Σw_g·z_g = Σw_g·Bz_g; rasterize R channels then project[^1][^2] |
| Replace `torch.sort` with custom radix/counting sort or persistent tile bins | ✓ | ✓ | ✓ | tile_id is bounded (~2k tiles); O(N) counting sort vs O(N log N); temporal coherence means ~5-10% of Gaussians cross tiles per frame |
| Shared-memory / warp-level gradient reduction before global atomicAdd in backward | ✓ | ✓ | ✓ | Quad+subgroup hybrid reduction achieves 10× backward speedup over naive atomicAdd[^3] |
| Multi-rate temporal execution: rasterize every frame, run backbone/spawner less often | ✓ | ✓ | ✓ | 70-85% of pixels are reprojection-stable at 60fps[^4][^5]; canvas warp is the cheap path |
| CUDA Graph capture eliminates per-frame launch overhead (1-3ms saved) | ✓ | ✓ | ✓ | Persistent kernel and graph capture patterns avoid driver‑level synchronization costs[^6] |
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
| GPT-5.5 Thinking | Analytic gradient splatting (∂C/∂x, ∂C/∂y) enables half-res splat + gradient-based HR recovery | ∂q/∂x = 2a·dx+2b·dy is free during forward; halves pixel evaluations[^7] |
| Claude Opus 4.7 Thinking | Energy-based Gaussian pruning: E_g = α_g·(s_u·s_v)·‖feat_g‖₂; evict bottom quartile every 8-16 frames | Keeps effective canvas at ~12k without quality loss; linear speedup everywhere |
| Claude Opus 4.7 Thinking | Kalman one-step update for existing Gaussians replaces spawner regression: 6 FLOPs/Gaussian vs full MLP | Spawner cost drops 4-8× since most Gaussians just need correction, not re-regression |
| Claude Opus 4.7 Thinking | Splat in log-luminance space → feat can be bf16 throughout rasterizer, cutting bandwidth ~50% | HDR scenes stay numerically conditioned; bandwidth is the real bottleneck |
| Gemini 3.1 Pro Thinking | Pad head_dim from 30→32 for TC alignment; 30 forces scalar CUDA paths | Near-free 1.5-2× on attention matmuls by hitting mma.sync sweet spots |
| Gemini 3.1 Pro Thinking | Jacobian-free warp: branch on ∇·V; if <ε (rigid translation ~90% of canvas), skip J entirely | Reduces warp to streaming copy for most Gaussians; 8-12 FLOPs saved per primitive |
| Gemini 3.1 Pro Thinking | Spawn at exact disoccluded pixel center; let warp advect off-grid naturally | Structurally kills checkerboard without noise injection; decouples spawn from sub-pixel positioning |

## 4. Comprehensive Analysis

The three models converge decisively on one architectural intervention that dwarfs all others: **reducing the rasterizer's feature payload from 64 channels to 4–12 via low-rank factorization**. GPT-5.5 Thinking frames this as `f_g ≈ B·z_g` where the rasterizer accumulates only `Z(p) = Σ w_g(p)·z_g` in ℝ^R, then a post-raster 1×1 projection recovers the full feature space. Claude Opus 4.7 Thinking corroborates this through the lens of GaussianImage's permutation-invariant accumulated summation, and Gemini 3.1 Pro Thinking quantifies the impact as an 87.5% reduction in register pressure yielding a projected drop from 136ms to 10-15ms. The mathematical basis is unimpeachable: sum-composite splatting is linear, so projection commutes with accumulation. This should be your P0 implementation.[^1]

The second tier of consensus centers on **eliminating framework overhead**: replacing `torch.sort` with bounded-range radix/counting sort (all three models), capturing the inference pipeline as a CUDA Graph (all three), and using shared-memory accumulation in the backward pass before global atomicAdd — a technique validated by recent hardware rasterization research showing 10× backward speedup via quad+subgroup hybrid reduction. These are all implementable within days and collectively save 5-15ms of pure overhead per frame.[^3]

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
<span style="display:none">[^10][^100][^101][^102][^103][^104][^105][^106][^107][^108][^109][^11][^110][^111][^112][^113][^114][^115][^116][^117][^118][^119][^12][^120][^121][^122][^123][^13][^14][^15][^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75][^76][^77][^78][^79][^8][^80][^81][^82][^83][^84][^85][^86][^87][^88][^89][^9][^90][^91][^92][^93][^94][^95][^96][^97][^98][^99]</span>

<div align="center">⁂</div>

[^1]: https://openreview.net/forum?id=SZvhmFntRA

[^2]: https://arxiv.org/abs/2501.06838

[^3]: https://arxiv.org/html/2505.18764v1

[^4]: https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Generalized_and_Efficient_2D_Gaussian_Splatting_for_Arbitrary-scale_Super-Resolution_ICCV_2025_paper.pdf

[^5]: https://www.themoonlight.io/en/review/generalized-and-efficient-2d-gaussian-splatting-for-arbitrary-scale-super-resolution

[^6]: https://concurrent-rt.com/wp-content/uploads/2020/12/Improving-Real-Time-Performance-With-CUDA-Persistent-Threads.pdf

[^7]: https://arxiv.org/abs/2503.14171

[^8]: https://github.com/Lee-JaeWon/2025-Arxiv-Paper-List-Gaussian-Splatting

[^9]: https://arxiv.org/html/2509.25626v2

[^10]: https://bentoml.com/llm/kernel-optimization/flashattention

[^11]: https://arxiv.org/abs/2307.08691

[^12]: https://wccftech.com/roundup/nvidia-dlss-vs-amd-fsr-vs-intel-xess-everything-you-need-to-know/

[^13]: https://cvpr.thecvf.com/virtual/2025/poster/33792

[^14]: https://arxiv.org/html/2403.08551v4

[^15]: https://www.emergentmind.com/topics/cross-temporal-3d-gaussian-splatting-cross-temporal-3dgs

[^16]: https://lubits.ch/flash/Part-6

[^17]: https://developer.nvidia.com/blog/speed-up-unreal-engine-nne-inference-with-nvidia-tensorrt-for-rtx-runtime/

[^18]: https://arxiv.org/html/2503.06617v1

[^19]: https://arxiv.org/html/2309.05239v3

[^20]: https://arxiv.org/abs/2205.14756

[^21]: https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.pdf

[^22]: https://arxiv.org/abs/2205.14135

[^23]: https://openreview.net/forum?id=H4DqfPSibmx

[^24]: https://gpuopen.com/fidelityfx-super-resolution-3/

[^25]: https://developer.nvidia.com/blog/nvidia-releases-rtx-neural-rendering-tech-for-unreal-engine-developers/

[^26]: https://www.intel.com/content/www/us/en/developer/articles/technical/xess-sr-developer-guide.html

[^27]: https://openaccess.thecvf.com/content/ICCV2025/papers/Hollein_3DGS-LM_Faster_Gaussian-Splatting_Optimization_with_Levenberg-Marquardt_ICCV_2025_paper.pdf

[^28]: https://github.com/nerfstudio-project/gsplat

[^29]: https://ai.gopubby.com/inside-the-cuda-kernel-the-gpu-implementation-of-3d-gaussian-splatting-74c3261ed721

[^30]: https://openaccess.thecvf.com/content/ICCV2025/html/Hollein_3DGS-LM_Faster_Gaussian-Splatting_Optimization_with_Levenberg-Marquardt_ICCV_2025_paper.html

[^31]: https://arxiv.org/html/2504.10686v1

[^32]: https://dl.acm.org/doi/10.1109/DAC63849.2025.11132449

[^33]: https://www.clarifai.com/blog/flash-attention-2

[^34]: https://research.facebook.com/publications/neural-supersampling-for-real-time-rendering/

[^35]: https://www.reddit.com/r/computergraphics/comments/18mxzkq/blog_post_rasterizing_gaussian_splats_the/

[^36]: https://huggingface.co/blog/atharv6f/flash-attention-basics

[^37]: https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf

[^38]: https://dev.to/lewis_won/online-softmax-by-hand-4h13

[^39]: https://arxiv.org/abs/2505.14201

[^40]: https://www.sethweidman.com/blog/streaming_softmax.html

[^41]: https://forums.developer.nvidia.com/t/persistent-kernel-runs-slower-when-with-more-threads/308556

[^42]: https://wangkuiyi.github.io/online-softmax.html

[^43]: https://forums.developer.nvidia.com/t/performance-of-persistent-thread-approach-on-new-gpu-architectures/43254

[^44]: https://github.com/kkokosa/dotLLM/issues/54

[^45]: https://hai.stanford.edu/research/flashattention-fast-and-memory-efficient-exact-attention-with-io-awareness

[^46]: https://hazyresearch.stanford.edu/blog/2023-07-17-flash2

[^47]: https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Supplemental-Conference.pdf

[^48]: https://www.reddit.com/r/GraphicsProgramming/comments/1pf1qj1/learn_how_to_integrate_rtx_neural_rendering_into/

[^49]: https://www.linkedin.com/posts/amitnvidia_nvidia-tensorrt-unrealengine-activity-7455984005344251904-tq1f

[^50]: https://github.com/NVIDIA/TensorRT-RTX

[^51]: https://github.com/NVIDIAGameWorks/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md

[^52]: https://www.digitaltrends.com/computing/amd-fsr-3-explained/

[^53]: https://www.reddit.com/r/aigamedev/comments/1pcmeez/learn_how_to_integrate_rtx_neural_rendering_into/

[^54]: https://www.reddit.com/r/pcmasterrace/comments/1ryrmdn/nvidia_confirms_dlss_5_uses_a_2d_frame_plus/

[^55]: https://www.tweaktown.com/news/110569/dlss-5-only-takes-2d-rendered-frames-and-motion-vectors-as-input-not-3d-game-engine-data-confirms-nvidia/index.html

[^56]: https://news.yahoo.com/everything-know-amds-fsr-3-130343982.html

[^57]: https://forums.developer.nvidia.com/t/dlss-motion-vector-question/266414

[^58]: https://www.techspot.com/article/2747-amd-fsr-3-tech/

[^59]: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/gaming.html

[^60]: https://github.com/mit-han-lab/efficientvit

[^61]: https://viplab.snu.ac.kr/viplab/courses/mlvu_2021_2/projects/final_papers/08.pdf

[^62]: https://han-cai.github.io/selected_projects/efficientvit_iccv.pdf

[^63]: https://cvpr.thecvf.com/media/cvpr-2023/Slides/23282.pdf

[^64]: https://arxiv.org/abs/2506.19845

[^65]: https://www.sciencedirect.com/science/article/pii/S187705092502191X

[^66]: https://arxiv.org/html/2309.05239v2

[^67]: https://www.scribd.com/document/689322361/2204-04676

[^68]: https://www.nature.com/articles/s41598-025-28042-1

[^69]: https://fal.ai/models/fal-ai/nafnet/deblur

[^70]: https://dl.acm.org/doi/abs/10.1007/s00521-023-09353-8

[^71]: https://openaccess.thecvf.com/content/CVPR2023/supplemental/Chen_Activating_More_Pixels_CVPR_2023_supplemental.pdf

[^72]: https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Xiao20.pdf

[^73]: https://www.transit.dot.gov/sites/fta.dot.gov/files/docs/The_NTD_Sampling_Manual.pdf

[^74]: https://pubs.usgs.gov/publication/tm1D12/full

[^75]: https://egusphere.copernicus.org/preprints/2025/egusphere-2025-272/

[^76]: https://arxiv.org/abs/2308.01483

[^77]: https://github.com/ansman/mandelbrot/blob/master/supersampling.cpp

[^78]: https://github.com/cashcon57/cauldron

[^79]: https://lume.ufrgs.br/bitstream/handle/10183/287628/001241293.pdf?sequence=1

[^80]: https://github.com/arm/neural-graphics-for-unreal

[^81]: https://github.com/topics/mcp-plugin

[^82]: https://arxiv.org/html/2510.01171v3

[^83]: https://github.com/ndming/NSRT

[^84]: https://github.com/topics/caveman

[^85]: https://github.com/sim51/caps_example/blob/master/csv/name.basics.tsv

[^86]: https://github.com/timmh/neural-supersampling

[^87]: https://arxiv.org/html/2603.07169v1

[^88]: https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/

[^89]: https://www.reddit.com/r/CUDA/comments/1moh19a/gtc_2025_nvidia_says_custom_cuda_kernels_only/

[^90]: https://www.rimikawrites.com/cuda-3-your-checklist-for-optimizing-cuda-kernels/

[^91]: https://research.samsung.com/blog/Trick-GS-A-Balanced-Bag-of-Tricks-for-Efficient-Gaussian-Splatting

[^92]: https://arxiv.org/html/2312.10890v1

[^93]: https://dl.acm.org/doi/10.1109/TVCG.2002.1021576

[^94]: https://arxiv.org/html/2602.09999v1

[^95]: https://princeton-nlp.github.io/flash-atttention-2/

[^96]: https://github.com/nerficg-project/HTGS

[^97]: https://arxiv.org/html/2405.19745v1

[^98]: https://arxiv.org/html/2311.17089v2

[^99]: https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Multi-Scale_3D_Gaussian_Splatting_for_Anti-Aliased_Rendering_CVPR_2024_paper.pdf

[^100]: https://blurbusters.com/frame-generation-essentials-interpolation-extrapolation-and-reprojection/

[^101]: https://www.allpcb.com/allelectrohub/flashattention-123-transformer-attention-optimizations

[^102]: https://www.alibaba.com/product-insights/ai-anime-upscalers-vs-manual-frame-interpolation-which-method-preserves-original-line-art-integrity-better.html

[^103]: https://github.com/dao-ailab/flash-attention

[^104]: https://ryanszeto.com/media/tai.pdf

[^105]: https://www.shadecoder.com/ja/topics/flashattention-2-a-comprehensive-guide-for-2025

[^106]: https://proceedings.neurips.cc/paper_files/paper/2024/file/a076d0d1ed77364fc57693bdee1958fb-Paper-Conference.pdf

[^107]: https://openreview.net/forum?id=76NYyOrnfk

[^108]: https://www.scribd.com/document/974537384/MoDGS-Dynamic-Gaussian-Splatting-From-Causually-CA

[^109]: https://www.kaggle.com/code/egazakharenko/flashattention-2-for-turing-from-scratch-tutorial

[^110]: https://www.alibaba.com/product-insights/ai-video-upscalers-vs-frame-interpolation-tools-why-does-4k-anime-sometimes-look-unnervingly-smooth.html

[^111]: https://openreview.net/pdf/c8c9ffc2bca0ad5b064b8d80f94643438b175c91.pdf

[^112]: https://arxiv.org/html/2503.14698v2

[^113]: https://proceedings.neurips.cc/paper_files/paper/2024/file/45ed1a72597594c097152ef9cc187762-Paper-Conference.pdf

[^114]: https://arxiv.org/html/2510.01619v1

[^115]: https://openreview.net/forum?id=CKZoVUpwWW

[^116]: https://arxiv.org/abs/2304.13986

[^117]: https://gatambook.substack.com/p/cross-attention-for-cross-asset-applications

[^118]: https://repositum.tuwien.at/bitstream/20.500.12708/17701/1/Schuetz Markus - 2021 - Interactive exploration of point clouds.pdf

[^119]: https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Prometheus_3D-Aware_Latent_Diffusion_Models_for_Feed-Forward_Text-to-3D_Scene_Generation_CVPR_2025_paper.pdf

[^120]: https://www.reddit.com/r/StableDiffusion/comments/13d80eu/which_crossattention_optimization_technique_is/

[^121]: https://proceedings.iclr.cc/paper_files/paper/2025/file/ba404795c58d122a4b6fc2672d84d9f3-Paper-Conference.pdf

[^122]: https://github.com/vladmandic/sdnext/discussions/846

[^123]: https://openaccess.thecvf.com/content/ACCV2024/papers/Svitov_HAHA_Highly_Articulated_Gaussian_Human_Avatars_with_Textured_Mesh_Prior_ACCV_2024_paper.pdf

