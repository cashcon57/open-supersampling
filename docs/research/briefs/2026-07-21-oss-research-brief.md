# OSS Research Brief — 2026-07-21

_Collection: local repository/README, GitHub REST, rendered dashboard, arXiv Atom + abstract pages, Archive.org Advanced Search, official NVIDIA/AMD/Apple pages, and Google News RSS. RSS-only reporting is labelled as such._

## Project Status Summary
- **Latest commits:** `a97a180` tailnet-gated training controls; `fcf8084` training-control API/panel; `0660f55` preserved v7 step-5K recovery snapshot; `56a9a65` WSL2 DataLoader rollback; `d1255c8` cross-OS checkpoint resume; `e1ab7fe` FlashAttention-2 path; `e9b4b65` TF32/prefetch; `26cd9da` corrected live PSNR proxy; `83b4b02` apex redirect.
- **Open issues (12):** #14 v6.2 VRAM measurement; #13 Nsight L2 profile; #12 packed low-rank canvas; #10 decoupled G-buffer ablation; #9 reparameterized student; #8 HIP/Metal/Vulkan/SYCL ports; #7 FSR4-teacher distillation; #6 legacy launch; #5 no-ML shader; #4 ≤1M student distillation.
- **README snapshot:** v7 is a research/teacher model with native `(x,y,t)` Gaussian primitives, `V_xt` covariance motion, loss-adaptive parent/child density, Mip-Splatting, and α=0.5 OSS-FX supervision. The ≤1M shipping student and every non-CUDA backend remain unbuilt.
- **Training dashboard:** `srcnn-v7.0-pico-005` remains **35,200 / 100,000**, loss **0.10166**, viz **35,077**. The page headline remains `live` with an ETA of ~1.1d, but health reports **Trainer offline; stuck at step 35,200 for 120h**. Watcher is healthy/published 0s ago, CF Worker is HTTP 200/91ms, R2 is HTTP 200, and GPU stats are unavailable. The unchanged step/loss/viz plus increased stale counter means continued trainer outage, not recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[SPARE-GS: Structural Parsimony and Resource Efficiency for 3D Gaussian Splatting](https://arxiv.org/abs/2607.16624)** — **Zhang Chen, Shuai Wan, Fuzheng Yang, Jiazhi Xia, Weiyao Lin, Junhui Hou**; 2026-07-18; arXiv:2607.16624. Frames Gaussian allocation as global budget-constrained optimization: capacity-normalized regional demand drives quotas, densification, pruning, and termination; abstract reports 30.38% fewer Gaussians and 23.81% less training time while improving PSNR. **Why it matters to OSS:** **HIGH RELEVANCE** direct ablation/reference for v7 parent-child loss-adaptive density, issue #12 bounded packed canvas, and quality-per-Gaussian accounting.
- **[ImprovedVBGS](https://arxiv.org/abs/2607.15542)** (previously tracked) — **Damani Mguni-Coker**; 2026-07-17; arXiv:2607.15542. Spatial truncation and reassignment target continual 3DGS bounded-memory updates. **Why it matters to OSS:** pair its locality rule with SPARE-GS’s global quota mechanism rather than adopting either alone.

### Super-Resolution / Upscaling
- No newly submitted paper cleared the direct-runtime relevance gate. **NanoVSR** (already tracked) remains the best immediate student reference: 644K parameters, structural reparameterization to plain convolutions, MIT implementation and ONNX export path.

### Frame Extrapolation / Generation
- **[SPEED: One-Step Pixel Diffusion for High-quality Video Frame Interpolation](https://arxiv.org/abs/2607.15585)** — **Zihao Zhang, Haoyu Zhao, Siqian Yang, Yidi Wu, Yudong Jiang, Zuxuan Wu**; 2026-07-17; arXiv:2607.15585 (ACM MM 2026). One-step pixel diffusion with dynamic patch scaling and noise-update-only attention; abstract reports 8.8% lower LPIPS, 63.3% faster inference, and 10.6% lower memory versus its VFI comparators on SNU-FILM. **Why it matters to OSS:** not viable for the <2ms shipping path, but a strong high-motion/disocclusion perceptual teacher and OSS-FX benchmark reference.
- **[3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering](https://arxiv.org/abs/2605.11489)** (previously tracked; newly surfaced as a direct comparison) — **Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu**; 2026-05-12; arXiv:2605.11489. Combines gradient-aware SR and a compact U-Net temporal interpolator over low-res 3DGS renders. **Why it matters to OSS:** closest published competitor to unified GS SR+FG, but it still uses separate GASS and LTFI modules; use as positioning and two-network ablation baseline.

### Inference Optimization
- No new SR-specific TensorRT/INT8/kernel-fusion paper met the relevance bar. **Opportunity:** AMD’s official RDNA4 WMMA guidance verifies that fusing FP8/INT8 WMMA operations along K to use 128-bit loads can double data supply relative to native 8-element loads; this is concrete input for the HIP student-kernel spike, not evidence of an achieved OSS speedup.

## Archive.org Findings
- Archive Advanced Search was noisy: `"gaussian splatting" super-resolution` returned three archived GitHub-star lists; `"frame generation" deep learning` returned 212 mostly unrelated items; `"real-time upscaling" game` returned 106 unrelated/miscellaneous items.
- Target names produced no usable archival recovery: GS-STVSR returned no result, GSASR’s single hit was unrelated, and `GRAPE`/`STSS` were acronym-noisy. No removed/updated primary-paper version was verified; do not infer takedown from this.

## GitHub / Open Source
- Fresh GitHub REST searches produced no credible new Gaussian-SR or game-upscaler repository. `frame generation open source` returned only an irrelevant zero-star pentest project; `neural upscaler` was dominated by zero-star copied GAN utilities.
- **[NanoVSR](https://github.com/filippawlicki/nanovsr)** is now a concrete open-source baseline: MIT, 21 stars, v1.0 pretrained release, ONNX exporter, and a July 20 update adding demo/benchmark evaluation and single-GPU training. **Opportunity:** reproduce its reparameterized CNN as issue #9’s control student before investing in custom student blocks.
- No verified new open-source DLSS/FSR/XeSS analysis tool appeared in this pass.

## News

### NVIDIA / DLSS
- **Verified:** NVIDIA’s July 2026 DLSS developer page says **DLSS 4.5** adds Dynamic Multi Frame Generation and a second-generation transformer SR model; it lists MFG up to five generated frames per rendered frame on RTX 50/RTX PRO Blackwell. **Threat:** OSS’s positioning must compare base-frame latency, artifact rate, and engine-input contract—not headline generated FPS—and retain the one-canvas/no-second-network cost argument.
- **Verified integration context:** the same NVIDIA page lists current UE and Unity plugin support (July package: Streamline 2.11.1 / NGX 310.6.0). OSS’s DLL-shim and engine-contract work remains a competitive prerequisite, not polish.
- **RSS-only lead:** reports claimed SIGGRAPH DLSS 5 details and Linux DLSS Updater/Proton-panel expansion; no primary announcement/release was verified here, so neither is treated as fact.

### AMD / FSR
- **Verified:** [FSR SDK 2.3](https://gpuopen.com/learn/amd-fsr-sdk-2-3-blog/) (2026-06-24) extends ML FSR Upscaling 4.1.1 to RX 7000/RDNA3, updates FSR Frame Generation to 4.0.1 with motion-vector/camera input improvements, and updates Ray Regeneration 1.2. **Threat/opportunity:** RDNA3 is now a nearer quality target and a practical first HIP/HLSL benchmark target; its MV/camera contract is directly comparable to OSS input requirements.
- **Verified:** AMD’s RDNA4 WMMA guide documents wide-K FP8/INT8 fusion for full 128-bit memory loads. **Opportunity:** benchmark this exact layout only after #9 establishes a convolutional student and #8 has a HIP kernel scaffold.
- **RSS-only lead:** reports of AMD 8× MFG were not confirmed by AMD; monitor but do not plan against it.

### Intel / Apple / Engines
- No primary Intel XeSS or Unreal/Unity upscaling-specific update was identified in the window.
- **Verified platform context:** Apple continues to expose MetalFX Upscaling, Frame Interpolation, and Denoising, while Metal 4 supports command-level ML/graphics integration. This keeps a Metal student+raster backend strategically relevant but provides no new performance datum.

## Action Items / Follow-ups
- [ ] **CRITICAL:** recover/inspect the 3080 Ti trainer. Dashboard state is still step 35,200/loss 0.10166/viz 35,077 after **120h**; watcher/CDN health does not establish trainer health.
- [ ] Deep-read SPARE-GS; make a v7 ablation that compares regional marginal-utility quotas against current parent-child loss-adaptive density, with fixed active-canvas memory and temporal-artifact gates.
- [ ] Use NanoVSR as issue #9’s reproducible CNN/reparameterization control; measure TensorRT and then HIP latency using the actual OSS G-buffer/MV input contract.
- [ ] Add 3DGS³ as a two-module SR+VFI baseline and SPEED as an offline high-motion/disocclusion teacher/reference; do not add diffusion to shipping OSS-FX.
- [ ] Scope RDNA3/RDNA4 as separate #8 targets: RDNA3 for FSR-quality/integration comparison, RDNA4 for FP8/INT8 WMMA fusion experiments.
