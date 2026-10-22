# OSS Research Brief — 2026-08-16

## Project Status Summary
- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated training controls and dashboard UI. Recent trainer work preserves the v7 step-5K recovery snapshot, fixes cross-OS checkpoint resume, enables FlashAttention/TF32/prefetch, and keeps smoothing filters in FP32 under BF16.
- **Open issues:** 11 open. The immediately architecture-relevant ones remain ≤1M student distillation (#4), cross-vendor kernel ports (#8), reparameterized student ablation (#9), and packed-canvas / L2 / VRAM measurement work (#12–14).[11]
- **README snapshot:** v7 is the active N-D `(x,y,t)` Gaussian teacher; the shipping ≤1M CNN student and HIP/Metal/Level Zero/Vulkan kernels remain unbuilt.
- **Training dashboard:** `srcnn-v7.0-pico-005` is still displayed as “live” at **36,650 / 100,000**, loss **0.19268**, viz @36K, but Trainer is **offline, stuck for 171h** and GPU stats are unavailable. Watcher, CF Worker, and R2 report healthy. The same step/loss as the Aug. 15 brief plus a growing outage counter means this is not progress.[10]

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **EndoMD-SLAM: Endoscopic Gaussian Splatting SLAM under Optical Degradation with Memory and Static-Transient Decomposition** — Nuo Chen, Kangqi Ni, Lulin Liu, Joga Ivatury, Ying Ding, Farshid Alambeigi, Tianlong Chen, Zhiwen Fan; Aug. 9, 2026; arXiv:2608.08949. It gates unreliable observations, pauses persistent-map updates, and isolates transient contamination into a separate field. **Why it matters to OSS:** direct v7 ablation material for disocclusion/effect/transient handling—do not irreversibly fuse unreliable engine frames into the persistent canvas; retain a transient layer or update gate.[12]
- **GS²CI: Robust Gaussian Splatting For Snapshot Compressive Imaging via Large Vision Model Priors** — Yanming Yang, Chenxi Song, Ping Wang, Xin Yuan, Chi Zhang; Aug. 13, 2026; arXiv:2608.13502. Opacity-guided splitting plus explicit candidate-ratio and Gaussian-count caps stabilize ambiguous supervision. **Why it matters:** surfaced yesterday; still the cleanest direct lead for bounded parent-child spawning under weak evidence. Keep it as a density-control ablation, not a reconstruction baseline.

### Super-Resolution / Upscaling
- **Flow-Map Distillation on Relation Manifolds for Image Restoration (FoRM)** — Zihao He, Songhua Liu; Aug. 6, 2026; arXiv:2608.05769. It supervises teacher/student feature relations as a compositional flow map rather than independent static matches. **Why it matters:** a teacher-side distillation experiment for the ≤1M student, particularly where temporal canvas features need consistency across adjacent timestamps; it is not evidence of a shipping-speed architecture.[14]
- **MeanSR: Restoration Trajectory Learning for One-Step Perceptual Super-Resolution** — Axi Niu, Jiawei Kou, Kang Zhang, Qingsen Yan, Jinqiu Sun, Yanning Zhang; Aug. 10, 2026; arXiv:2608.09405. One-step trajectory learning is a potentially stronger perceptual teacher, but diffusion/trajectory machinery is outside the student latency budget. Treat it as an offline pseudo-target lead only.

### Frame Extrapolation / Generation
- **SNM-VFI: Symmetric Nonlinear Motion-Guided Generative Video Frame Interpolation** — Jisoo Jeong, Hong Cai, Jamie Menjay Lin, Hanno Ackermann, Hyeonjun Sim, Yinhao Zhu, Yunxiao Shi, Fatih Porikli; Aug. 13, 2026; arXiv:2608.13460. It combines nonlinear multi-frame flow, confidence maps, and diffusion detail synthesis. **Why it matters:** the confidence-map split is a useful OSS-FX loss/evaluation design—trust exact engine-MV canvas rendering where support is strong and explicitly gate uncertain boundaries/disocclusions; the diffusion runtime is not compatible with OSS shipping constraints.[13]
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** (arXiv:2608.10423) remains the immediate competitive baseline from Aug. 15: a layered cache with explicit future visibility and adaptive shading. No newer paper displaced it as the most relevant non-neural OSS-FX comparison.

### Inference Optimization
- No newly published TensorRT/INT8/kernel-fusion result cleared the materiality bar. NanoVSR and deployment-aware INT8 SR remain the relevant student baselines already tracked in `BIBLIOGRAPHY.md`; prioritize measured end-to-end raster + composite latency over another paper sweep.

## Archive.org Findings
- Internet Archive Advanced Search remains non-authoritative/noisy for this niche: exact Gaussian-SR finds three archived GitHub-list artifacts, “frame generation” finds consumer-tool builds plus unrelated research, real-time game upscaling and GS-STVSR return zero, and GSASR/GRAPE/STSS acronym hits are unrelated. No archived/taken-down version changes GS-STVSR, GSASR, GRAPE, or STSS provenance.[15]

## GitHub / Open Source
- **ZeroFG** remains the only current adjacent repo worth watching: public Apache-2.0 C++ experimental Vulkan/mobile frame-generation POC, one star, updated Aug. 13. It is a portability/integration signal, not a quality or architecture competitor; no benchmarks or canvas-style reconstruction evidence were found.[16]
- Current searches for Gaussian-SR, neural upscalers, and DLSS/FSR analysis otherwise returned empty, generic video tooling, or zero-star WIP repositories. No credible new open DLSS/FSR/XeSS analysis tool appeared.

## News
- **Verified primary Intel development:** Intel announced launch enablement for XeSS 3 Super Resolution, Multi-frame Generation, and Xe Low Latency in *STAR WARS: Galactic Racer* and *Tomb Raider: Legacy of Atlantis*.[17] This raises the practical bar for OSS integration: preserve a vendor-neutral input contract around MV/depth/reactive-mask-class data and benchmark against an Unreal/plugin-style deployment path.
- **Verified AMD kernel lead:** GPUOpen's current RDNA 4 WMMA material documents FP8/INT8 low-precision GEMM fusion, 128-bit load discipline, and a hipBLAS-verified path to peak memory bandwidth.[18] This is immediately relevant to the HIP student—not proof of OSS latency, but a concrete kernel-design/profiling reference for issue #8 after the packed canvas work.
- NVIDIA's Aug. 13 primary feed is continued DLSS 4.5 title enablement, not a new architecture disclosure.[19] Google News RSS otherwise shows game-level FSR/DLSS integration and an XeSS UE-plugin headline; treat unverified headlines as leads.

## Action Items / Follow-ups
- [ ] **P0 operational:** repair/restart the v7 trainer host. Dashboard is internally contradictory (“live” plus offline) and frozen at step 36,650 / loss 0.19268 for 171h.
- [ ] Prototype a persistent/transient canvas split and/or update-confidence gate inspired by EndoMD-SLAM; score stale-transient retention and disocclusion artifacts against the current canvas.
- [ ] Use SNM-VFI’s reliability split as an OSS-FX evaluation: report separate error for high-confidence exact-MV regions and uncertain/disoccluded boundaries. Keep diffusion out of the shipping path.
- [ ] Add FoRM-style relational/trajectory distillation only after issue #12/#13 establishes packed-layout and L2/VRAM budgets; benchmark it against ordinary feature/output distillation.
- [ ] Convert AMD's RDNA 4 WMMA FP8/INT8 load/fusion guidance into a small HIP student GEMM microbenchmark after #12/#13; compare against hipBLAS before designing a fused canvas+raster path.
- [ ] Deep-read and benchmark Amulet as the explicit-visibility non-neural OSS-FX baseline; it is the clearest architectural threat/opportunity.

## Sources

[10] https://opensupersampling.com — OpenSuperSampling training dashboard
[11] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open — OpenSuperSampling open issues API
[12] https://arxiv.org/abs/2608.08949 — EndoMD-SLAM arXiv
[13] https://arxiv.org/abs/2608.13460 — SNM-VFI arXiv
[14] https://arxiv.org/abs/2608.05769 — FoRM arXiv
[15] https://archive.org/advancedsearch.php?q=%22gaussian+splatting%22+super-resolution%26fl%5B%5D=identifier%26fl%5B%5D=title%26rows=10%26output=json — Internet Archive Gaussian-SR search
[16] https://github.com/hy300leosquizz-ctrl/ZeroFG — ZeroFG repository
[17] https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships — Intel Gamer Days 2026
[18] https://gpuopen.com/ — AMD GPUOpen developer portal / RDNA 4 WMMA guidance
[19] https://www.nvidia.com/en-us/geforce/news/ — NVIDIA GeForce news feed
