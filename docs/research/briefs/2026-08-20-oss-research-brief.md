# OSS Research Brief — 2026-08-20

## Project Status Summary
- **Latest commits:** local history currently exposes only `b44278a test: add local Chrome Playwright dashboard smoke`.
- **Open issues:** 12. The immediate performance dependency chain is still **#12 packed low-rank canvas → #13 Nsight L2 profile → #14 v6.2 VRAM measurement**. #1 rasterizer, #3 tile bin, #4/#7/#9 student distillation, #8 non-CUDA backends, and #10 G-buffer ablation remain open.
- **README snapshot:** v7 is the active teacher architecture: native `(x,y,t)` Gaussians, `V_xt` covariance coupling, loss-adaptive density, and time-slice OSS-FX. The ≤1M shipping student, DLL shim, and HIP/Metal/Vulkan/Level Zero kernels remain unbuilt.
- **Training dashboard:** `data.json` still labels `srcnn-v7.0-pico-005` active at **36,650/100K**, latest total loss **0.1926767** (SR Charbonnier 0.0235991; FG Charbonnier 0.0764263; 2,304 canvas primitives). `status.json` says the trainer is **offline/stuck at that step for 266h**, while watcher, CF Worker, and R2 are healthy. This is a stale-active dashboard contradiction, not evidence of training recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[Depth Anything V4: Dynamic 4D Scene Reconstruction via Riemannian Flow Matching on 4D Gaussian Splatting](https://arxiv.org/abs/2608.18388)** (arXiv:2608.18388; Jiaming Fan, Jian Lu, Jinling Jia, Chenbin Zhang; 2026-08-18) — applies flow matching directly to 4DGS parameters on their non-Euclidean scale/rotation/opacity manifolds and reports an isolated F-score gain over its matched MLP control. **Why it matters to OSS:** a teacher-side precedent for keeping time-varying Gaussian parameters valid; deep-read for stable covariance/opacity update parameterization, not as a real-time design.
- **[QuARC-GS: Quantized Anchored Residual Coding for Compact Dynamic Scene Streaming with Gaussian Splatting](https://arxiv.org/abs/2608.18285)** (arXiv:2608.18285; Vu Trung Nghia Nguyen, Yuchen Wang, Kyung Chul Lee, Kevin C. Zhou; 2026-08-18) — quantization-aware anchored residual coding for compact dynamic GS streaming. **Why it matters to OSS:** directly relevant to #12’s packed canvas and long-session bandwidth/cache budget; evaluate its anchor/residual layout against L2 locality and temporal identity rather than adopting its streaming stack wholesale.
- **[RoofGS: Roofline-Guided End-to-End Acceleration of 3D Gaussian Splatting](https://arxiv.org/abs/2608.15785)** (arXiv:2608.15785; Yang Luo, Yan Gong, Yongsheng Gao, Jie Zhao; 2026-08-16; newly surfaced) — uses roofline analysis to target front-end memory traffic and rasterizer instruction throughput with compact attributes, 32-bit depth-sort keys, approximate exponentials, culling, dual-pixel evaluation, and kernel fusion. **Why it matters to OSS:** the strongest new implementation reference for #3/#12/#13; evaluate the tactics against the OSS tile bin and packed canvas rather than copying its 3DGS pipeline.
- No new verified 2D-GS arbitrary-scale SR or GS-STVSR/GRAPE-class direct competitor appeared in the targeted fresh metadata. Broad Archive.org inventory was noisy (see below).

### Super-Resolution / Upscaling
- **[SFMformer: A Spatial-Frequency Modulation Transformer for Lightweight Image Super-Resolution](https://arxiv.org/abs/2608.17966)** (arXiv:2608.17966; Chih-Hsiang Yang, Chia-Min Lin, Ching-Yu Tsai, Yung-Che Wang, Jen-Shiun Chiang; 2026-08-18) — separates sparse-attention token selection from aggregation with pre/post spatial-frequency modulation. **Why it matters to OSS:** possible teacher/student ablation only if its published parameter/MAC/deploy graph fits the ≤1M target; transformers remain presumptively too costly until measured.
- **[Binarized High-Efficiency RAW Video Restoration and Beyond](https://arxiv.org/abs/2608.16756)** (arXiv:2608.16756; Tianyu Zhu, Ying Fu, Hesong Li, Gengchen Zhang, Xin Yuan, Yulun Zhang; 2026-08-17) — BinRVR reports about 96% compute/parameter reduction with about 4% degradation via a binarized temporal interaction module. **Why it matters to OSS:** a late-stage INT8/binary sensitivity reference for the student; RAW restoration and its operators are not a drop-in game path.

### Frame Extrapolation / Generation
- **[SPVC: Structured and Panoptic Video Fixing for Cross-Dataset Driving Scene Rendering](https://arxiv.org/abs/2608.17420)** (arXiv:2608.17420; Gen Li, Shu Han, Yun Xi Qiao, Hua Chen, Xuyang Dai, Bohan Li, Hao Zhao, Chaojian Li; 2026-08-18) — trains a two-stage controllable video-diffusion fixer on degraded/clean 3DGS render pairs to address extrapolated-trajectory blur, flicker, and foreground/background misalignment. **Why it matters to OSS:** teacher-side/benchmark reference for extrapolated-view failure modes and structured controls; not a candidate for the <2 ms shipping path.
- No newly material extrapolation paper displaced existing Amulet/3DGS³/ANVIL controls. **[SNM-VFI](https://arxiv.org/abs/2608.13460)** was re-confirmed but is already tracked; its confidence-guided flow candidates remain useful only for invalid-MV/hole controls, not OSS-FX deployment.

### Inference Optimization
- **RoofGS** is the key new lead: its roofline-guided split between memory-bound front-end work and instruction-bound rasterization gives #13 a concrete profiling hypothesis. Its reported throughput is not transferable to OSS, but its 32-bit-key, compact-attribute, fusion, and culling ablations are.
- QuARC-GS remains the companion compression lead. Prior NanoVSR remains the more direct ≤1M deploy-graph comparator; do not select quantization/fusion boundaries from papers before #12–#14 establish the actual canvas bandwidth and occupancy profile.

## Archive.org Findings
- Requested Archive.org advanced searches returned 3 `gaussian splatting`/SR inventory records (saved GitHub lists), 3 frame-generation records (two software/build items and one unrelated long-video paper), and **0** real-time-game-upscaling records. None is a preserved primary GS-STVSR, GSASR, GRAPE, or STSS artifact.
- Targeted Wayback CDX queries for known paper/repository URLs returned HTTP 400 in this run; archive coverage is incomplete, **not** evidence of no captures.

## GitHub / Open Source
- GitHub REST searches since 2026-08-06 found no material new Gaussian-SR implementation. The only fresh Gaussian-SR hit is an empty 0-star course project; exclude it.
- **[MAKO](https://github.com/eugeniosegala/MAKO)** is the new practical Vulkan/Linux frame-generation integration lead (created Aug. 17; pushed Aug. 20; 134 stars): Decky component, Vulkan renderer, per-game profiles, presentation recovery, and adaptive 2–4× generation. **Opportunity:** inspect host-owned presentation/recovery and UX boundaries only. It depends on proprietary `Lossless.dll`, has no SPDX license, and is neither a neural-quality baseline nor an OSS competitor.
- **[SR_Compare](https://github.com/Tyuwwe/SR_Compare)** is a same-day MIT Vulkan side-by-side quality/performance harness for upscalers (0 stars). **Opportunity:** early evaluation-harness lead; inspect when its source/API is reliably available. **[SelectiveSR](https://github.com/gomzee1024/SelectiveSR)** is a newly active Gaussian-SR repo with code/checkpoints but no license, stars, or temporal/game path—monitor only.
- **[ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG)** remains the only adjacent code lead: Apache-2.0, experimental Vulkan/mobile frame-generation POC, created Aug. 10 and last pushed Aug. 13 (1 star). **Opportunity:** inspect host-owned synchronization/presentation ABI only. It is neither a quality baseline nor a threat to OSS’s shared-canvas architecture.
- No material mature DLSS/FSR analysis project surfaced beyond the early SR_Compare lead.

## News
- **AMD (primary):** FSR SDK 2.3 adds FSR Upscaling 4.1 on RX 7000/RDNA 3; AMD’s page describes ML inference on RDNA 4 and RDNA 3 discrete GPUs, with analytical FSR 3 fallback for RDNA 3.5/RDNA 2. **Threat:** FSR 4’s addressable hardware is expanding; OSS must report per-vendor quality/latency rather than rely on a generic cross-vendor claim.
- **Apple (primary):** the redesigned MetalFX temporal upscaler uses Neural Engine plus M5 Pro/Max Neural Accelerators, supports dynamic-resolution subrectangles, direct motion vectors, and distortion fields. **Opportunity:** lock those ABI concepts into the future Metal backend now.
- **Intel (primary):** Intel says the full XeSS 3 suite—Super Resolution, Multi-frame Generation, and Xe Low Latency—will ship at launch in named partner titles. **Positioning:** OSS telemetry should expose SR, extrapolation, and latency separately.
- **NVIDIA (primary news feed):** DLSS 4.5 Ray Reconstruction and second-generation transformer SR/Dynamic MFG integrations are continuing. This reinforces the need to compare OSS against complete latency+quality+frame-pacing surfaces, not only image metrics.
- **AMD (primary, Aug. 20):** Adrenalin 26.8.1 adds Radeon RX 9050 4GB support. The release notes also list an RX 9000/UI-status issue where FSR Upscaling and Frame Generation can appear inactive in Adrenalin while enabled in the *Call of Duty: Modern Warfare 4* Open Beta. **Operational note:** do not infer feature inactivity from vendor control-panel state alone in future OSS validation.
- Google News also carried third-party leads for FSR 4.1 performance on RX 7800 XT and individual DLSS 4.5/FSR 4 game updates; these were not promoted to architecture claims without a vendor/game primary source.

## Action Items / Follow-ups
- [ ] **Critical:** restore/restart `srcnn-v7.0-pico-005` or correct dashboard state. It is stuck at step 36,650 for 266h while still advertised as active.
- [ ] Land #12 → profile #13 → measure #14 before locking 32B/Gaussian, quantization, or fused-kernel choices.
- [ ] Deep-read RoofGS, then make its compact-attribute/key-width/fusion/culling claims explicit Nsight ablations under #13; retain temporal identity and quality gates.
- [ ] Deep-read QuARC-GS: map its anchor/residual coding to a bounded per-tile canvas layout; benchmark L2 hit rate, reconstruction, and primitive identity under motion.
- [ ] Evaluate DAV4 only for manifold-safe covariance/opacity/time updates; reject any approach requiring test-time optimization or scene-scale training.
- [ ] Define Metal input ABI: dynamic-resolution subrect, exact MV convention, distortion/postprocess masks, and quantized-weight fallback.
- [ ] Track FSR 4.1 RDNA 3 rollout and XeSS 3’s SR/MFG/latency split as competitive positioning changes.

## Sources

1. https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
2. https://opensupersampling.com/data.json
3. https://opensupersampling.com/status.json
4. https://arxiv.org/api/query (targeted queries: `cat:cs.CV AND abs:"super-resolution"`, `cat:cs.CV AND abs:"Gaussian Splatting"`, `cat:cs.CV AND abs:"video super-resolution"`, `cat:cs.CV AND abs:"frame interpolation"`)
5. https://archive.org/advancedsearch.php
6. https://api.github.com/search/repositories
7. https://gpuopen.com/amd-fsr-upscaling/
8. https://developer.apple.com/metal/whats-new/
9. https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships
10. https://www.nvidia.com/en-us/geforce/news/
11. https://arxiv.org/abs/2608.15785
12. https://arxiv.org/abs/2608.17420
13. https://api.github.com/repos/eugeniosegala/MAKO
14. https://api.github.com/repos/Tyuwwe/SR_Compare
15. https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html
