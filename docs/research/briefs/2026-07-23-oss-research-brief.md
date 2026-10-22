# OSS Research Brief — 2026-07-23

_Collection: local repository/README, GitHub REST, rendered dashboard, arXiv Atom/direct abstract page, Archive.org Advanced Search, and official NVIDIA/AMD/Intel pages. Several broad arXiv Atom queries timed out or returned HTTP 429; results below are filtered to primary-verified, OSS-material items._

## Project Status Summary
- **Latest commits:** `a97a180` tailnet-gated dashboard training controls; `fcf8084` control API/panel; `0660f55` preserves the v7 step-5K recovery snapshot; `56a9a65` reverts the WSL2-deadlocking DataLoader; `d1255c8` adds cross-OS checkpoint-resume support; `e1ab7fe` adds a FlashAttention-2 path; `e9b4b65` enables TF32/prefetch; `26cd9da` corrects the PSNR proxy.
- **Open issues (10 shown):** #14 v6.2 VRAM measurement; #13 Nsight L2 profile; #12 packed low-rank canvas; #10 decoupled G-buffer ablation; #9 reparameterized student; #8 HIP/Metal/Vulkan/SYCL ports; #7 FSR4-teacher distillation; #6 legacy launch; #5 no-ML shader; #4 ≤1M student distillation.
- **README snapshot:** v7 is still the teacher: native `(x,y,t)` Gaussians, `V_xt` covariance motion, parent/child loss-adaptive density, Mip-Splatting, and α=0.5 OSS-FX supervision. The ≤1M TensorRT-FP8 student and non-CUDA runtimes do not exist.
- **Training dashboard:** `srcnn-v7.0-pico-005` advanced from the earlier same-day **35,200 / loss 0.10166** to **35,250 / loss 0.14658**, viz **35,077**. Hero says `live`, ETA ~1.1d; health separately says **Trainer offline, stuck at step 35,250 for 6h**. Watcher is healthy (published 1s ago), CF Worker HTTP 200/95ms, R2 origin HTTP 200; GPU stats unavailable. This is genuine interval progress followed by a new/still-current outage—not recovery. The dashboard’s “live” copy is contradictory operationally.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **CaT-GS: Efficient 3DGS Rendering for Large Scale Scenes via Inter-frame Caching and Tile Scheduling** — Tingjia Zhang, Bo Chen, Shengzhong Liu, Fan Wu, Guihai Chen; **2026-07-20**; [arXiv:2607.17842](https://arxiv.org/abs/2607.17842). Caches inter-frame preprocessing/occlusion work and uses a tile-scheduling kernel to reduce rasterizer imbalance, reporting up to 10× over its 3DGS baseline. **Why it matters:** strongest immediate runtime study target for v7’s persistent canvas—measure its cacheable stages and tile-load distribution before designing #12’s packed layout or bespoke fusion; claimed gains are abstract-level and must be reproduced on the 3080 Ti.
- **SPARE-GS** (arXiv:2607.16624; Zhang Chen, Shuai Wan, Fuzheng Yang, Jiazhi Xia, Weiyao Lin, Junhui Hou; 2026-07-18) and **SPIN-4DGS** (arXiv:2607.12362; Seung-gyeom Kim, Areum Kim, Yongjae Yoo, Sukmin Yun; 2026-07-14) remain newly material but already tracked: budgeted regional densification is the #12 control; implicit spatiotemporal attributes are the fast-motion/disocclusion stress reference for `V_xt`.

### Super-Resolution / Upscaling
- No new primary-verified paper passed the OSS relevance gate. **NanoVSR** (arXiv:2607.10495; Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11) remains the highest-value ≤1M/reparameterized-CNN control for #9/#4.

### Frame Extrapolation / Generation
- No newly verified result. **SPEED** (arXiv:2607.15585; Zihao Zhang, Haoyu Zhao, Siqian Yang, Yidi Wu, Yudong Jiang, Zuxuan Wu; 2026-07-17) remains an offline high-motion/disocclusion perceptual-teacher and benchmark lead, not a <2ms runtime architecture.

### Inference Optimization
- The remaining targeted Atom queries (Gaussian-temporal, frame generation/extrapolation, TensorRT/INT8/fusion, and game-upscaler/MV) timed out or hit HTTP 429. Do not infer no publications; re-run narrow `id_list`/fielded requests later. CaT-GS is the only newly bibliography-worthy optimization lead today.

## Archive.org Findings
- Advanced Search: `"gaussian splatting" super-resolution` returned 3 archived GitHub-star-list snapshots rather than papers; `"frame generation" deep learning` returned DLSS Swapper/FluidFrames builds plus an unrelated video-generation registration; `"real-time upscaling" game` returned 0.
- No archived replacement/takedown evidence for GS-STVSR, GSASR, GRAPE, or STSS. This noisy inventory is not evidence of absence/removal.

## GitHub / Open Source
- Search since 2026-07-09: 0 `gaussian splatting super resolution` repos and 0 DLSS/FSR/XeSS analysis tools. Most `neural upscaler` hits are zero-star GAN copies.
- **RDNU** ([Adam-AtlasSoftware/RDNU](https://github.com/Adam-AtlasSoftware/RDNU), GPL-3.0, 15 stars, updated 2026-07-24 UTC) is still the only credible open G-buffer-guided, RDNA3-targeted neural-upscaler lead. Its stated data mix (M3VIR/GameIR, vKITTI2, MPI-Sintel) makes it a baseline/data-contract comparison—not proof of game-ready quality or an OSS substitute. Already tracked in the bibliography.
- A newly indexed `Optiscaler-Client-V4-Pro-2026` has 0 stars, a 3 KB repo, no asserted SPDX license, and keyword-stuffed topics; treat as noise, not a competitor or analysis tool.

## News

### NVIDIA / DLSS
- **Competitive pressure, official developer page:** NVIDIA now documents **DLSS 4.5**: Dynamic MFG, a second-generation transformer SR model, and up to five generated frames per rendered frame on RTX 50/RTX PRO Blackwell. The July package exposes Streamline 2.11.1/NGX 310.6.0 and UE 5.5–5.8 plugins. **OSS implication:** benchmark quality/latency separately from headline generated FPS; Streamline remains the practical integration reference, while OSS’s differentiator is vendor-neutral canvas reuse and exact engine MVs.

### AMD / FSR
- **Material primary update:** AMD’s June 24 FSR SDK 2.3 release brings ML **FSR Upscaling 4.1.1** to RX 7000/RDNA3 as well as RX 9000/RDNA4; Frame Generation 4.0.1 adds motion-vector/camera-input corner-case improvements, while Ray Regeneration 1.2.0 has an ABI break. Analytical fallbacks remain for RDNA3.5/RDNA2/older parts. **OSS implication:** #7 must distinguish legal/technical teacher-signal experimentation from a shipping dependency, and any AMD positioning must include RDNA3 and fallback-path measurements rather than treating FSR4 as RX 9000-only.

### Intel / Apple / Engines
- Intel’s Arc G-Series announcement makes **XeSS 3** (SR + Multi-Frame Generation + Xe Low Latency) a concrete handheld target. This strengthens the Pico requirement: validate the student on an IGP/handheld-class latency and power envelope, not only RTX TensorRT.
- No primary Apple MetalFX API/performance change or Unreal/Unity upscaling-specific release was verified in the 14-day window. Apple’s WWDC26 catalog includes game-porting/graphics sessions, but nothing here substantiates a new OSS-relevant Metal performance claim.

## Action Items / Follow-ups
- [ ] **CRITICAL:** inspect/recover the 3080 Ti trainer. It made a 50-step interval advance but is now explicitly offline/stuck at 35,250 for 6h; watcher/CDN health only proves publication.
- [ ] Profile v7 rasterizer stages and tile occupancy, then prototype a CaT-GS-style cache/scheduler ablation under fixed canvas budget before locking #12’s packed low-rank layout.
- [ ] Make NanoVSR the reparameterized CNN control (#9/#4), measure TensorRT on the OSS RGB/depth/MV contract, then set AMD/Intel portability criteria from actual RDNA3/Arc results rather than vendor claims.
- [ ] Update competitive matrix: DLSS 4.5 dynamic MFG, FSR 4.1.1 on RDNA3, XeSS 3 handhelds. Keep SR, generated-frame quality, base FPS, latency, and power as separate columns.
- [ ] Re-run narrow arXiv `id_list`/fielded search after throttling clears; do not backfill bibliography from snippets.
