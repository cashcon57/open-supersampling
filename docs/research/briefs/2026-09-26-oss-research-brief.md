# OSS Research Brief — 2026-09-26

## Project Status Summary
- Latest local commit: `b44278a test: add local Chrome Playwright dashboard smoke` (the sole commit returned by local `git log --oneline -10`).
- Open issues (first 10): #14 VRAM footprint; #13 Nsight L2 profile; #12 packed low-rank canvas; #10 G-buffer ablation; #9 student reparameterization ablation; #8 cross-vendor ports; #7 FSR 4 teacher distillation; #6 v6.2 Pico launch; #5 no-ML shader; #4 HAT-Tiny student distillation.
- README snapshot describes pre-alpha; v6.2 baseline and v7 Pico-005 training, while the student, frame-extrapolation integration and cross-vendor kernels remain unwired/unbuilt. This is the checked-in snapshot, not live training verification.
- Dashboard: direct page request returns HTTP 403; browser rendering also failed because agent-browser binary was permission-denied. Live run/metrics/health unverified; README/dashboard comparison unavailable.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **AdaTex4D: Adaptive Texture Capacity Allocation for 4D Gaussian Splatting** (arXiv:2609.29963v1; 2026-09-24) — De Jiang, Peiqiang Wang, Kehong Yuan, Shaohua Ma. [Paper](https://arxiv.org/abs/2609.29963). Frozen 4DGS backbones allocate anisotropic per-Gaussian texture capacity using visibility-normalized gradients and deformed scales; reports 47.3% of uniform texture bytes at comparable quality. **Opportunity for #12:** prototype temporal-peak/visibility-aware feature-budget allocation for packed canvas slots; their dense texture materialization caveat means this is an allocation idea, not a demonstrated runtime win.

- **Resonant Brane Splatting for Arbitrary-Scale Super-Resolution** (arXiv:2606.29453v3; 2026-09-08) — Federico, Amato, Gennaro, Carrara, and Di Benedetto; [paper](https://arxiv.org/abs/2606.29453). Generalizes flat Gaussian splats with Gaussian-Hermite color modes to represent local contrast/texture while retaining scale-aware rasterization. **Opportunity / architecture watch:** richer primitives may raise quality per splat, but reported GPU timings and memory need a matched, independent benchmark before considering extra representation cost.
- **Adaptive Densification for High-Fidelity and Efficient Sparse Gaussian Splatting in Arbitrary-Scale Super-Resolution (QuADA-GS)** (arXiv:2606.29400v2; 2026-09-08) — Federico, Amato, Gennaro, Carrara, and Di Benedetto; [paper](https://arxiv.org/abs/2606.29400). Allocates Gaussian density adaptively from LR input and contrasts this with one-per-pixel GRAPE and dense GSASR. **Opportunity:** adaptive primitive budget could inform OSS canvas spawning/pruning and VRAM trade-offs; validate temporal stability before adopting.
- **Deformable 2D Gaussian Splatting for Efficient 4K Video Compression** (arXiv:2609.14129v1; 2026-09-12) — Zhang and Zhu; [paper](https://arxiv.org/abs/2609.14129). Coarse-to-fine 2DGS, lightweight multi-plane deformation, optical-flow guidance and QAT; claims real-time 4K decode up to 247 FPS and <1 GB VRAM. **Opportunity:** test multi-plane motion/deformation and QAT ideas for memory-bound canvas updates; compression metrics do not establish game-upscaler quality or latency.
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047; 2026-04-20) — Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha. [Paper](https://arxiv.org/abs/2604.18047). Optical-flow-guided continuous motion with covariance-resampling alignment directly parallels OSS temporal covariance stability; already tracked in bibliography.

### Super-Resolution / Upscaling
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (arXiv:2607.10495; 2026-07-11) — Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies. [Paper](https://arxiv.org/abs/2607.10495). Fully convolutional, structurally reparameterized compact VSR model; useful comparison for student size, deployment operations and temporal information budget.

### Frame Extrapolation / Generation
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** (arXiv:2608.10423; 2026-08-11) — authors not present in fetched metadata; [paper](https://arxiv.org/abs/2608.10423). Sparse layered screen-space cache with adaptive shading performs non-neural extrapolation and explicit disocclusion handling; paper claims up to 250 Hz at 4K. **Competitive threat / design check:** benchmark its cache-based approach against OSS's claimed low-marginal-cost extrapolation, especially disocclusion and large motion.
- **Continuous Space-Time Video Super-Resolution with 3D Fourier Fields** (arXiv:2509.26325v2; 2026-03-04 version) — Alexander Becker, Julius Erbach, Dominik Narnhofer, Konrad Schindler. [Paper](https://arxiv.org/abs/2509.26325). Continuous (x,y,t) field offers an alternative representation and comparison for temporal coherence; not new publication this week.

### Inference Optimization
- NanoVSR reparameterizes its training-time structure into standard convolutions at inference; directly relevant to OSS student ablation #9. No newly verified SR-specific TensorRT/INT8/kernel-fusion breakthrough surfaced in this search pass.

## Archive.org Findings
- Archive.org advanced search for `"frame generation" deep learning` returned three items, including DLSS Swapper builds and FluidFrames; these are archive inventory, not archived research papers.
- Search results for `"gaussian splatting" super-resolution` were dominated by unrelated 2023 Graphdeco repository snapshots. No verified archived/taken-down versions of GS-STVSR, GSASR, GRAPE or STSS found. Archive search coverage is noisy and not evidence of absence.

## GitHub / Open Source
- GitHub repository search (`gaussian splatting super-resolution`) surfaced [ResonantBraneSplatting](https://github.com/GiulioFede/ResonantBraneSplatting), pushed Sep 18 (1 star), and [FoundDSR](https://github.com/yanzq95/FoundDSR), pushed Sep 25 (1 star; depth SR, adjacent not direct game SR). RBS paper is the material lead; low-star repositories need code/benchmark verification.
- [OpenFrameGen](https://github.com/Nicodemo-Campos/OpenFrameGen) latest commit Sep 25; its own README says only the portable bootstrap exists and frame generation, Vulkan interception, upscaling and GUI are not yet implemented. **Threat level low today; watch integration direction** because its modular Vulkan-first cross-platform framing overlaps OSS positioning, but it is not a working competing model.
- [RDNU](https://github.com/Adam-AtlasSoftware/RDNU) (pushed Sep 24; 16 stars) describes an RDNA3-focused learned upscaler using decoupled G-buffer guidance. **Competitive watch:** direct game-upscaler positioning overlap; inspect licensing, code maturity, and reproducible quality/latency before treating as threat.
- Search also surfaced active DLSS unlocking/swapping utilities; these are compatibility tools, not open-source alternative upscaler architectures.

## News
- **NVIDIA (primary, Sep 15–22):** GeForce news page promotes DLSS 4.5, Ray Reconstruction and Dynamic Multi Frame Generation in current game updates; its Sep 22 CONTROL Resonant article is a product/game-support update, not a newly announced architecture. [NVIDIA GeForce news](https://www.nvidia.com/en-us/geforce/news/)
- **AMD (primary, current GPUOpen news listing):** FSR SDK 2.3 lists FSR Upscaling 4.1.1 support on RX 7000 and Frame Generation 4.0.1 / Ray Regeneration 1.2 improvements for RDNA 4; GPUOpen also lists its UE 5.8 FSR plugin. This reinforces the need to measure INT8 on RDNA 3 and address integration parity. [GPUOpen news](https://gpuopen.com/news/)
- **Apple (primary, Metal/WWDC26):** redesigned MetalFX temporal upscaler now uses Neural Engine + Neural Accelerators on M5 Pro/Max; API adds subrectangle processing, motion-vector input and distortion fields. **Opportunity/threat:** concrete Metal lane opportunity, but hardware-specific accelerators strengthen platform lock-in and make the cross-vendor abstraction a priority. [Metal updates](https://developer.apple.com/metal/whats-new/) · [WWDC26 games guide](https://developer.apple.com/wwdc26/guides/games/)
- No primary XeSS release-page verification completed this run; broad news-search results were noisy, so no unsupported XeSS or recent engine-upscaling claims included. NVIDIA developer blog page showed no recent SR-specific technique during this check.

## Action Items / Follow-ups
- [ ] Deep-read Amulet; compare sparse layered cache/disocclusion handling and measured 4K cost to OSS-FX.
- [ ] Deep-read QuADA-GS, AdaTex4D and Resonant Brane Splatting; test visibility/temporal-peak feature allocation against #12's packed-canvas storage and end-to-end bandwidth, not paper-only memory metrics.
- [ ] Evaluate 2609.14129 multi-plane deformation and QAT as canvas bandwidth/inference hypotheses, not direct SR evidence.
- [ ] Inspect RDNU/OpenFrameGen/dlssg-open repos for license, activity, and end-to-end integration maturity; OpenFrameGen currently remains a bootstrap.
- [ ] Retry dashboard data/status through an authorized browser/runtime path; static HTML serves but `/data.json` and `/status.json` return HTTP 403, leaving live run/metrics/health unverified.
- [ ] Track MetalFX M5 hardware acceleration and verify AMD FSR 4.1.1/FG 4.0.1 plugin compatibility against OSS's cross-vendor plans.
