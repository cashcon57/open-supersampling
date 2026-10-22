# OSS Research Brief — 2026-06-05

## Project Status Summary
- Latest commits: a97a180 feat: add tailnet-gated training controls to dashboard; fcf8084 feat: tailnet-gated training control API + dashboard panel; 0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc; 56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock); d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume
- Open issues: GitHub issue parse failed: Invalid control character at: line 408 column 1223 (char 20000)
- README snapshot: current architecture is **v7**, `srcnn-v7.0-pico-005` teacher training, N-D Gaussian canvas `(x,y,t)`, 3D Cholesky covariance / V_xt cross-correlation, parent-child density, Mip-Splatting filters, and alpha=0.5 OSS-FX supervision. Student model still untrained; cross-vendor kernels still designed only.
- Training dashboard: Dashboard extract reports v6.1 Pico at 0/100K and waiting for metrics.json, conflicting with README/current commits claiming v7-pico-005 + tailnet-gated controls; public dashboard likely stale or metrics feed empty.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **Coherent room-temperature dipole synchronization in nanocavity sheets** (arXiv:2606.06490v1, 2026-06-04) — Plasmonic nanocavities enable the synchronization of spatially distant emissive dipoles through strong near-field coupling in sub-nm gaps. We report formation of a room-temperature synchronized dipole state in… Authors: Rakesh Arul, Piper Fowler-Wright, Lille Borresen, Brendon W. Lovett, Jonathan Keeling. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **Statistically and Computationally Optimal Estimation and Inference of Common Subspaces** (arXiv:2606.06483v1, 2026-06-04) — Given multiple data matrices, many problems in statistics and data science rely on estimating a common subspace that captures certain structure shared by all the data matrices. In this paper we investigate the… Authors: Joshua Agterberg. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **Two-Sample Hypothesis Testing for Subspace Equality in Network Data** (arXiv:2606.06482v1, 2026-06-04) — In many settings one is often interested in determining whether two networks share some joint structural connectivity patterns such as communities. However, while communities may be shared across networks, edge… Authors: Rajdeep Brahma, Joshua Agterberg, Yuguo Chen. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **How abundant are good interpolators?** (arXiv:2606.06469v1, 2026-06-04) — Let $S$ be the set of unit norm linear classifiers $θ\in \mathbb{R}^d$ which correctly classify every point of a labeled dataset $(X_i,y_i)_{i=1}^n$, $X_i \in \mathbb{R}^d$, $y_i \in \{-1,+1\}$, with a possibly negative… Authors: August Y. Chen, Ahmed El Alaoui. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **TempoVLA: Learning Speed-Controllable Vision-Language-Action Policies** (arXiv:2606.06491v1, 2026-06-04) — Robot manipulation alternates between low-risk transit phases that call for fast execution and high-risk contact stages that demand slow, precise motion. Yet existing Vision-Language-Action models (VLAs) only inherit a… Authors: Dong Jing, Jingchen Nie, Tianqi Zhang, Jiaqi Liu, Huaxiu Yao. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **Homeomorphic modified wave operators for the Vlasov-Poisson system** (arXiv:2606.06488v1, 2026-06-04) — We prove modified scattering for small data solutions to the Vlasov-Poisson system in a functional framework where the initial data, scattering states, and asymptotic convergence are measured in the same topology. In… Authors: Léo Bigorgne. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **Phenomenology of Inflaton-Driven Early QCD Confinement and Solution to Axion Isocurvature Problem** (arXiv:2606.06484v1, 2026-06-04) — We study the phenomenology of early QCD confinement during inflation, driven by a direct coupling between the inflaton and Standard Model gluons. This coupling dynamically raises the QCD confinement scale, making the… Authors: Evangelos I. Sfakianakis, Barmak Shams Es Haghi, Katherine Freese. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.

### Super-Resolution / Upscaling
- **Regret Minimization with Adaptive Opponents in Repeated Games** (arXiv:2606.06486v1, 2026-06-04) — In this paper, we study regret minimization in repeated games with \emph{adaptive} opponents who can respond based on histories of play. The standard metric of \emph{external regret} in online learning is known to fail… Authors: Mingyang Liu, Asuman Ozdaglar, Tiancheng Yu, Kaiqing Zhang. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **DNQ: Deep Nash Q-Network for Partially Observable n-Player Games** (arXiv:2606.06480v1, 2026-06-04) — Many real-world competitive systems require multiple decision-makers to act simultaneously under shared constraints, limited information, and repeated interaction, as in auctions, resource allocation, and security… Authors: Qintong Xie, Edward Koh, Xavier Cadet, Peter Chin. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **Pretraining Recurrent Networks without Recurrence** (arXiv:2606.06479v1, 2026-06-04) — Training recurrent neural networks (RNNs) requires assigning credit across long sequences of computations. Standard backpropagation through time (BPTT) addresses this problem poorly: it is sequential in time, limiting… Authors: Akarsh Kumar, Phillip Isola. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **Double Preconditioning (DoPr): Optimization for Test-Time Performance, not Validation Loss** (arXiv:2606.06418v1, 2026-06-04) — Many modern applications of deep learning involve training a neural network via a one-step prediction loss (e.g., $L^2$ regression, cross-entropy), but deploy the network by rolling out along its own predictions. Key… Authors: Thomas T. Zhang, Alok Shah, Yifei Zhang, Vincent Zhang, Nikolai Matni. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **PolyGraphPy: A unified Python framework for atomistic simulation and machine learning-driven polymer design** (arXiv:2606.06415v1, 2026-06-04) — Polymers are indispensable materials with applications ranging from electronics to medicine owing to their versatility, which can be tailored by adjusting their chemical composition and architecture. The design space… Authors: João G. C. S. Duarte, Shruti Venkatram, Morgan Cencer, Traian Dumitricǎ, Ketson R. M. dos Santos. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **HANDOFF: Humanoid Agentic Task-Space Whole-Body Control via Distilled Complementary Teachers** (arXiv:2606.06493v1, 2026-06-04) — For a humanoid robot to be deployed in the real world, the choice of command space (i.e., the interface between task planning and whole-body control) is crucial. Existing whole-body controllers typically demand dense… Authors: Lizhi Yang, Junheng Li, Nehar Poddar, Yiling Hou, Gio Huh. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution** (arXiv:2606.06492v1, 2026-06-04) — Code language models need repository-level context to resolve imports, APIs, and project conventions. Existing methods inject this knowledge as long inputs (retrieved through RAG or dependency analysis) or through… Authors: Liliana Hotsko, Yinxi Li, Yuntian Deng, Pengyu Nie. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **Complexity-Balanced Diffusion Splitting** (arXiv:2606.06477v1, 2026-06-04) — Standard continuous-time generative models rely on monolithic architectures that must navigate vastly different signal regimes, from isotropic noise to intricate data distributions. While scaling model capacity improves… Authors: Noam Issachar, Dani Lischinski, Raanan Fattal. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **TailLoR: Protecting Principal Components in Parameter-Efficient Continual Learning** (arXiv:2606.06494v1, 2026-06-04) — Parameter-efficient finetuning methods based on spectral decomposition have enabled progress in Continual Learning. In this paper we introduce TailLoR, which utilizes the singular bases U and V of the pre-trained… Authors: Marius Dragoi, Ioana Pintilie, Alexandra Dragomir, Antonio Barbalau, Florin Brad. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.
- **Numerical self-force calculations for scalar particles, formulated in the lab frame** (arXiv:2606.06487v1, 2026-06-04) — We derive equations of motion for scalar particles self-consistently interacting with a scalar field,including the radiation produced by the particles' acceleration. Our approach differs in three key aspects from… Authors: Stamatis Vretinaris, Erik Schnetter, Badri Krishnan. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.

### Frame Extrapolation / Generation
- **What it takes to solve the Hubble tension through Modifications of Cosmological Recombination II: in light of ACT DR6 and DESI DR2** (arXiv:2606.06495v1, 2026-06-04) — We construct data-driven solutions to the Hubble tension, in light of recent data from the Atacama Cosmology Telescope (ACT DR6) and the Dark Energy Spectroscopic Instrument (DESI DR2). We search for the minimal… Authors: Nanoom Lee, Tianji Zhou. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.

### Inference Optimization
- **Physics-Guided Deep Unfolding for Blind Cross-Sensor Spectral Super-Resolution via Learning the Spectral Transformation Function** (arXiv:2606.05759v1, 2026-06-04) — Hyperspectral imaging provides rich spectral information for quantitative remote sensing, yet hyperspectral sensors remain costly and thus unavailable in many UAV deployments. Spectral super-resolution (SSR) seeks to… Authors: Zhaolin Li, Jinsong Chen, Shanxin Guo, Tuo Zhang, Xinglong Zhang. Why it matters to OSS: Potential SR baseline/training or student-distillation idea.
- **MaCo-GAN: Manifold-Contrastive Adversarial Learning for Single Image Super-Resolution** (arXiv:2606.05068v1, 2026-06-03) — Conventional Generative Adversarial Networks (GANs) for Single Image Super-Resolution (SISR) often struggle with hallucinated artifacts, largely because standard discriminators evaluate overall image naturalness rather… Authors: Daeyoung Han, Seongmin Hwang, Moongu Jeon. Why it matters to OSS: Potential SR baseline/training or student-distillation idea.
- **A Planetary Nebula from a 5.7 $M_{\odot}$ Progenitor in a 90 Myr M31 Star Cluster** (arXiv:2606.04368v1, 2026-06-03) — Planetary nebulae (PNe) trace the late evolution of low-to-intermediate-mass stars, yet the masses of their progenitors are rarely measured directly. Here we present a PN physically associated with a young star cluster… Authors: Pinjian Chen, Bingqiu Chen, Xiaodian Chen, Shu Wang, Haibo Yuan. Why it matters to OSS: Tangential; track only if a follow-up reveals a transferable primitive or training trick.

### Game Engine / Motion Vectors
- No strong new hits.

## Archive.org Findings
- `site:archive.org "gaussian splatting" "super-resolution"` → **github.com-graphdeco-inria-gaussian-splatting_-_2023-08- ...** — # github.com-graphdeco-inria-gaussian-splatting_-_2023-08-14_08-25-42

## Item Preview

# Share or Embed This Item

# github.com-graphdeco-inria-gaussian-splatting_-_2023-08-14_08-25-42

- Publication date

- 2023-08-14

- Topics

- GitHub, code, software, git

- Item Size

- 2.3M

Original referenc... (https://archive.org/details/github.com-graphdeco-inria-gaussian-splatting_-_2023-08-14_08-25-42)
- `site:archive.org "gaussian splatting" "super-resolution"` → **github.com-graphdeco-inria-gaussian-splatting_-_2023-08-23_09-13-40 : graphdeco-inria : Free Download, Borrow, and Streaming : Internet Archive** — # github.com-graphdeco-inria-gaussian-splatting_-_2023-08-23_09-13-40

## Item Preview

# Share or Embed This Item

# github.com-graphdeco-inria-gaussian-splatting_-_2023-08-23_09-13-40

- Publication date

- 2023-08-23

- Item Size

- 2.3M

Original reference implementation of "3D Gaussian Splattin... (https://archive.org/details/github.com-graphdeco-inria-gaussian-splatting_-_2023-08-23_09-13-40)
- `site:archive.org "gaussian splatting" "super-resolution"` → **github.com-graphdeco-inria-gaussian-splatting_-_2023-08-30_11-27-54 : graphdeco-inria : Free Download, Borrow, and Streaming : Internet Archive** — # github.com-graphdeco-inria-gaussian-splatting_-_2023-08-30_11-27-54

## Item Preview

# Share or Embed This Item

# github.com-graphdeco-inria-gaussian-splatting_-_2023-08-30_11-27-54

- Publication date

- 2023-08-30

- Item Size

- 2.3M

Original reference implementation of "3D Gaussian Splattin... (https://archive.org/details/github.com-graphdeco-inria-gaussian-splatting_-_2023-08-30_11-27-54)
- `site:archive.org "frame generation" "deep learning"` → **Improving the precision and speed of mobile device object detection** — Two-stage object detection is divided into a candidate frame generation stage and a prediction stage; ... Chollet, F.: Xception: Deep learning ... (https://scholar.archive.org/work/lq6vqj22urb6tcs5rbssnhop2u/access/wayback/https:/ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/ipr2.12340)
- `site:archive.org "frame generation" "deep learning"` → **[PDF] Robust One Shot Audio to Video Generation - CVF Open Access** — They have used Pix2Pix [15] for the frame synthesis and we are using a spatially adaptive gen- erator for frame generation which is able to generate higher. (https://scholar.archive.org/work/dkw75a5mfvdpxc4lwg4yck4cla/access/wayback/https:/openaccess.thecvf.com/content_CVPRW_2020/papers/w45/Kumar_Robust_One_Shot_Audio_to_Video_Generation_CVPRW_2020_paper.pdf)
- `site:archive.org "frame generation" "deep learning"` → **[PDF] Cognitive Radio Jamming Attack Detection Using an Autoencoder ...** — ... frame generation. The simulation is repeated 10 times to realize ... using autoencoder deep learning model. The autoencoder is ... (https://scholar.archive.org/work/lhaxphfsefhd5ifn7b5k22rney/access/wayback/https:/assets.researchsquare.com/files/rs-355056/v1_stamped.pdf)
- `site:archive.org "real-time upscaling" game` → **Intel enters the PC gaming GPU battle with Arc** — Intel is branding its upcoming consumer GPUs as Intel Arc. This new Arc brand will cover both the hardware and software powering Intel’s high-end discrete GPUs, as well as multiple hardware generations. The first of those, known previously as DG2, is expected to arrive in the form of codename “Alche... (https://web.archive.org/web/20210829104356/https:/www.theverge.com/2021/8/16/22626912/intel-arc-gpu-brand-alchemist-codename-release-date)
- `site:archive.org "real-time upscaling" game` → **Full text of "Computer Gaming World Issue 167"** — # Full text of "Computer Gaming World Issue 167"
## See other formats
nnTNimiTTTTT ■$w4. Secrets HALF-LIFE vS-A UNREAL vs DAI KATANA 3SB 9 Morel Take a Bloodbath Ml Smokin' 3D Shooters! (https://archive.org/stream/Computer_Gaming_World_Issue_167/Computer_Gaming_World_Issue_167_djvu.txt)
- `site:archive.org "real-time upscaling" game` → **Nvidia’s Ada Lovelace GPU generation: $1,599 for RTX 4090, $899 and up for 4080** — After weeks of teases, Nvidia's newest computer graphics cards, the "Ada Lovelace" generation of RTX 4000 GPUs, are here. Nvidia CEO Jensen Huang debuted two new models on Tuesday: the RTX 4090, which will start at a whopping $1,599, and the RTX 4080, which will launch in two configurations. (https://web.archive.org/web/20230403183856/https:/arstechnica.com/gadgets/2022/09/nvidias-ada-lovelace-gpu-generation-1599-for-rtx-4090-899-and-up-for-4080/)
- `site:archive.org GS-STVSR OR GSASR OR GRAPE OR STSS` → **Scuppernong, North Carolina's Grape and its Wines** — Scuppernong, North Carolina's Grape and its Wines. by: Gohdes, Clarence, 1901-1997. Publication date: 1982; Topics: Muscadine grape -- North ... (https://archive.org/details/scuppernongnorth01gohd)
- `site:archive.org GS-STVSR OR GSASR OR GRAPE OR STSS` → **Compendium of Grape Diseases, Disorders, and Pests Second Edition** — Compendium of Grape Diseases, Disorders, and Pests Second Edition. viii, 232 pages : 28 cm. Includes bibliographical references and index. (https://archive.org/details/compendiumofgrap0000unse)
- `site:archive.org GS-STVSR OR GSASR OR GRAPE OR STSS` → **Proteomic analyses on the browning of shade-dried Thompson ...** — The main browning-related proteins of Thompson seedless grape were identified in this study. Their properties were tested, and their roles in the browning ... (https://scholar.archive.org/work/3rcc223h5vbmrclkimne54nbl4/access/wayback/https:/applbiolchem.springeropen.com/track/pdf/10.1186/s13765-021-00612-7.pdf)

## GitHub / Open Source
- `site:github.com "gaussian splatting" "super-resolution"` → **GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale ...** — [AAAI2025] This repo holds the code for work "GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution" ... (https://github.com/tljxyys/GaussianSR)
- `site:github.com "gaussian splatting" "super-resolution"` → **Improvement of SuGaR based Gaussian Splatting and ... - GitHub** — Improvement of SuGaR based Gaussian Splatting and Mesh reoncstruction via focus on preprocessing of images from Super Resolution and Deblurring perspective - (https://github.com/superdianuj/improved_SuGaR)
- `site:github.com "gaussian splatting" "super-resolution"` → **open-supersampling/docs/research/2026-05-05-existing-gaussian ...** — — S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting (CVPR 2025). URL: https://jeasco.github.io/S2Gaussian/ (project page; code linked from there) ... (https://github.com/cashcon57/open-supersampling/blob/main/docs/research/2026-05-05-existing-gaussian-splatting-repos-survey.md)
- `site:github.com "frame generation" "open source"` → **FrankBarretta/LSFG-Android: Lossless Scaling Frames Generator ...** — Open Source. COMMUNITY. GitHub SponsorsFund open source developers. PROGRAMS ... End-to-end frame generation works today on Adreno 7xx-class GPUs and newer. (https://github.com/FrankBarretta/LSFG-Android)
- `site:github.com "frame generation" "open source"` → **OptiScaler bridges upscaling/frame gen across GPUs. Supports ...** — Open Source. COMMUNITY. GitHub SponsorsFund open source ... 7.0+, added experimental DX12 frame generation support with possible HUDfix solution (OptiFG) ... (https://github.com/optiscaler/OptiScaler)
- `site:github.com "frame generation" "open source"` → **GitHub - Nukem9/dlssg-to-fsr3: Adds AMD FSR 3 Frame Generation ...** — Adds AMD FSR 3 Frame Generation to games by replacing Nvidia DLSS Frame Generation ( ... Open Source. COMMUNITY. GitHub SponsorsFund open source developers. (https://github.com/Nukem9/dlssg-to-fsr3)
- `site:github.com "neural upscaler"` → **OptiScaler fork with Prism neural upscaler + FGExtrap ... - GitHub** — Prism Neural Upscaler ... Custom neural network upscaler that replaces DLSS/FSR/XeSS with a GAN-trained model. Takes G-buffer inputs (color + depth + motion ... (https://github.com/NodeNestor/prism-optiscaler)
- `site:github.com "neural upscaler"` → **andreaskuster/RFSuperResolution - GitHub** — We propose RF Super Resolution, a lightweight, real-time neural upscaler for RF signals. Our approach pairs an efficient digital interpolation algorithm ... (https://github.com/andreaskuster/RFSuperResolution/)
- `site:github.com "neural upscaler"` → **DirectML Neural Upscaler (APIs available) · Blinue Magpie - GitHub** — DirectML Neural Upscaler (APIs available) #153. RyzenAdvanced started this conversation in Ideas. (https://github.com/Blinue/Magpie/discussions/153)
- `site:github.com DLSS FSR XeSS analysis tool upscaling` → **GitHub - ind4skylivey/0ptiscaler4linux: The intelligent OptiScaler ...** — TL;DR: OptiScaler for Linux automatically configures FSR4, XeSS, and DLSS upscaling for massive FPS gains in Linux games. One-command installation. (https://github.com/ind4skylivey/0ptiscaler4linux)
- `site:github.com DLSS FSR XeSS analysis tool upscaling` → **AMD FidelityFX Super Resolution v1.0.2 for mpv - Github-Gist** — Sharpening, in general, produces best results if applied AFTER enlarging/upscaling. Does FSR do 2X or fractional upscaling like ravu-zoom? I ask because if ... (https://gist.github.com/agyild/82219c545228d70c5604f865ce0b0ce5?permalink_comment_id=4071946)
- `site:github.com DLSS FSR XeSS analysis tool upscaling` → **[AMD/RADEON] Monster Hunter: Rise doesn't allow DLSS ... - GitHub** — OptiScaler bridges upscaling/frame gen across GPUs. Supports DLSS2+/XeSS/FSR2+ inputs, replaces native upscalers, enables FSR3 FG on non-FG ... (https://github.com/optiscaler/OptiScaler/issues/590)

## Vendor / Developer Blogs
- `site:developer.nvidia.com DLSS 4 neural rendering RTX 2026` → **NVIDIA RTX Neural Rendering Introduces Next Era of AI-Powered ...** — The latest breakthrough, DLSS 4, brings new Multi Frame Generation and Transformer-powered Super Resolution and Ray Reconstruction, accelerated ... (https://developer.nvidia.com/blog/nvidia-rtx-neural-rendering-introduces-next-era-of-ai-powered-graphics-innovation/)
- `site:developer.nvidia.com DLSS 4 neural rendering RTX 2026` → **Announcing the Latest NVIDIA Gaming AI and Neural Rendering ...** — DLSS 4 is a revolutionary suite of neural rendering technologies that uses AI to boost FPS, reduce latency, and improve image quality. (https://developer.nvidia.com/blog/announcing-the-latest-nvidia-gaming-ai-and-neural-rendering-technologies/)
- `site:developer.nvidia.com DLSS 4 neural rendering RTX 2026` → **NVIDIA DLSS - NVIDIA Developer** — DLSS Multi Frame Generation boosts frame rates by using AI to generate up to five frames per rendered frame powered by NVIDIA GeForce RTX 50 Series and RTX PRO ... (https://developer.nvidia.com/rtx/dlss)
- `site:gpuopen.com FSR 4 FidelityFX SDK 2026` → **AMD FidelityFX Super Resolution 4 now available on GPUOpen** — The AMD FidelityFX™ SDK v2.0 update includes our cutting-edge ML upscaler AMD FidelityFX Super Resolution 4 (FSR 4). (https://gpuopen.com/learn/amd-fsr4-gpuopen-release/)
- `site:gpuopen.com FSR 4 FidelityFX SDK 2026` → **AMD FidelityFX SDK 2.0 launches our neural rendering ...** — The AMD FidelityFX SDK 2.0 represents a new suite of features (starting with AMD FSR 4 upscaling and FSR 3.1.5 analytical frame generation) ... (https://gpuopen.com/learn/amd-fidelityfx-sdk-2-0/)
- `site:gpuopen.com FSR 4 FidelityFX SDK 2026` → **AMD FSR Upscaling** — The AMD FSR Redstone SDK enables developers to integrate neural rendering technologies, including ML-powered upscaling, frame generation, denoising, and ... (https://gpuopen.com/amd-fsr-upscaling/)
- `site:intel.com XeSS Arc upscaling 2026` → **Intel XeSS 1.3 accelerates into the next generation of AI upscaling** — Across the board we see an average 10% expected increase in FPS from a previous version of XeSS to XeSS 1.3 on the Intel Arc A750, with some ... (http://game.intel.com/us/stories/intel-xess-1-3/)
- `site:intel.com XeSS Arc upscaling 2026` → **The Excitement Grows with XeSS in Over 50 Games** — Intel's AI-powered upscaling lets you enjoy games with all the eye candy turned on, and games which started with playable framerates push into ... (https://game.intel.com/us/stories/intel-arc-graphics-xess-games/)
- `site:intel.com XeSS Arc upscaling 2026` → **Intel® XeSS on Arc™ Graphics Explained** — XeSS renders a game at a lower resolution then upscales the image using AI to a higher resolution with quality similar (or better) to a native ... (https://game.intel.com/us/stories/intel-arc-graphics-xess/)

## News
- `DLSS 4 developments past week June 2026` → **Nvidia introduces DLSS 4.5 and Multi Frame Generation 6X at CES ...** — Nvidia introduces DLSS 4.5 and Multi Frame Generation 6X at CES 2026 — updated models can generate higher-quality upscaled frames and more of ... (https://www.tomshardware.com/pc-components/cpus/nvidia-introduces-dlss-4-5-and-multi-frame-generation-6x-at-ces-2026-updated-models-can-generate-higher-quality-upscaled-frames-and-more-of-them-dynamically)
- `DLSS 4 developments past week June 2026` → **All New DLSS 4 Gameplay - CES 2026 - YouTube** — NVIDIA kicked off CES 2026 with a breakdown of their new DLSS 4 games as well as the new DLSS 4.5. We have gameplay from Resident Evil ... (https://www.youtube.com/watch?v=G_rTK_y-Ql8)
- `DLSS 4 developments past week June 2026` → **NVIDIA launches DLSS 4.5 - CG Channel** — Now out of beta: discover key new features for VFX artists and colorists in the free color grading and editing app and its $295 Studio edition. (https://www.cgchannel.com/2026/03/nvidia-launches-dlss-4-5/)
- `FSR 4 developments past week June 2026` → **AMD Caves, Extends FSR 4 to RX 7000 & 6000 GPUs ... - YouTube** — AMD finally announces FSR 4.1 coming to RX 7000 and RX 6000 series GPUs. Are these now the best used GPU in 2026? We look at FSR 4.1 Int8 ... (https://www.youtube.com/watch?v=Jq9q5HQ3azs)
- `FSR 4 developments past week June 2026` → **AMD FSR 4 Confirmed for Older GPUs, July Launch Set for RDNA 3** — AMD confirms FSR Upscaling 4.1 is coming to RDNA 3 GPUs in July and RDNA 2 in 2027, finally bringing FSR 4 tech to older Radeon cards. (https://hothardware.com/news/amd-bringing-fsr-41-to-rdna3-rdna2)
- `FSR 4 developments past week June 2026` → **Did You Really Think AMD Was Going To Leave You Hanging? FSR ...** — ... Was Going To Leave You Hanging? FSR 4 Is Coming To RDNA 3 In July. 1.1K views · 2 weeks ago ...more. FrogboyX1Gaming & Tech Reviews. 19.4K. (https://www.youtube.com/watch?v=vGWT2TsQos0)
- `Intel Arc XeSS updates June 2026` → **Intel confirms XeSS 3 multi-frame generation support for Arc B580 ...** — Intel has now confirmed that XeSS 3 with Multi-Frame Generation will arrive on the Intel Arc B580 graphics card this month. (https://www.dlcompare.com/gaming-news/intel-confirms-xess-3-multi-frame-generation-support-for-arc-b580-gpu-soon-71092)
- `Intel Arc XeSS updates June 2026` → **Intel® Arc™ Graphics Gaming Technologies** — Intel's GPU technologies deliver true-to-life graphics with superior performance, speed, efficiency, and scalability. (https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/gaming-technologies.html)
- `Intel Arc XeSS updates June 2026` → **Intel Arc G-Series Processors Set a New Standard for Handheld PC ...** — Handheld systems powered by Intel Arc G-Series processors will begin rolling out from OEM partners starting June 2026, with broader availability ... (https://newsroom.intel.com/client-computing/intel-arc-g-series-processors-set-a-new-standard-for-handheld-pc-gaming)
- `Apple Silicon Metal performance update June 2026` → **We Need a Native Apple Silicon Version of ESO (M1/M2/M3/M4/M5)** — More stable FPS and overall smoother gameplay, especially during large-scale encounters. Lower power usage and reduced heat, which is crucial ... (https://forums.elderscrollsonline.com/en/discussion/685251/we-need-a-native-apple-silicon-version-of-eso-m1-m2-m3-m4-m5)
- `Apple Silicon Metal performance update June 2026` → **Apple unleashes M5, the next big leap in AI performance for Apple ...** — M5 is Apple's next-generation system on a chip built for AI, resulting in a faster, more efficient, and more capable chip for the 14-inch ... (https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)
- `Apple Silicon Metal performance update June 2026` → **Apple Silicon's Next Big Upgrade Could Arrive in 2026** — Apple could upgrade to considerably more advanced 2-nanometer chip fabrication technology as soon as 2026, according to analyst Ming-Chi Kuo. (https://forums.macrumors.com/threads/apple-silicons-next-big-upgrade-could-arrive-in-2026.2403235/)
- `NVIDIA AMD Intel GPU announcements June 2026` → **Computex 2026: all expected announcements (AMD, Intel, NVIDIA)** — The biggest hardware trade show of the year opens its doors from June 2nd to 5th in Taipei. Here is what AMD, Intel, NVIDIA, and manufacturers ... (https://dropreference.com/en/blog/news/computex-2026-expected-announcements-amd-intel-nvidia)
- `NVIDIA AMD Intel GPU announcements June 2026` → **CES 2026: 8 Big Chip Announcements By Intel, Nvidia, AMD And ...** — CES 2026: 8 Big Chip Announcements By Intel, Nvidia, AMD And Qualcomm · Intel Core Ultra Series 3 · AMD Ryzen AI 400 Series · AMD Ryzen AI Max ... (https://www.crn.com/news/components-peripherals/2026/ces-2026-8-big-chip-announcements-by-intel-nvidia-amd-and-qualcomm)
- `NVIDIA AMD Intel GPU announcements June 2026` → **GeForce @ COMPUTEX 2026: NVIDIA RTX Spark Unveiled, DLSS ...** — NVIDIA at COMPUTEX 2026: NVIDIA RTX Spark platform for AI PCs, DLSS 4.5 Ray Reconstruction, 1000+ RTX games and apps, NVIDIA app updates, ... (https://www.nvidia.com/en-us/geforce/news/computex-2026-nvidia-geforce-rtx-announcements/)
- `Unreal Engine Unity upscaling update June 2026` → **Render Pipelines strategy for 2026 - Unity Discussions** — To our understanding this new upscaling framework is required for FSR3 and PSSR support. We already have DLSS4 in HDRP since Unity 6.3, which by ... (https://discussions.unity.com/t/render-pipelines-strategy-for-2026/1710004/print)
- `Unreal Engine Unity upscaling update June 2026` → **Render Pipelines strategy for 2026 - Unity Discussions** — It is our firm intention to remove availability of the Built-in Render Pipeline in the future, but before we decide for a final end date, we ... (https://discussions.unity.com/t/render-pipelines-strategy-for-2026/1710004)
- `Unreal Engine Unity upscaling update June 2026` → **Microsoft Game Development Tools and Services: What's New in Q1 ...** — PlayFab has released Online Subsystem (OSS) v2.3.8 with full support for Unreal Engine 5.7, so you can upgrade to the latest engine version ... (https://developer.microsoft.com/en-us/games/articles/2026/04/microsoft-game-dev-release-notes-q1-2026/)

## Action Items / Follow-ups
- [ ] Fix/check public dashboard freshness: site extraction reports v6.1/0 steps while repo README and recent commits say v7 + tailnet-gated controls.
- [ ] Deep-read RiGS / LagrangianSplats / other dynamic Gaussian hits for primitive motion decomposition ideas transferable to v7 V_xt covariance and density management.
- [ ] Track AMD FidelityFX/FSR 4 SDK source availability and developer-blog updates; public ML operator/kernel details are directly useful for OSS student + fused kernels.
- [ ] Manually inspect top GitHub hits if this run's broad search is too noisy; fresh open-source frame-gen/upscaler implementations are possible competitive threats or integration references.
