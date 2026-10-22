# OSS Research Brief — 2026-07-01

## Project Status Summary
- Latest commits: `a97a180` tailnet-gated training controls; `fcf8084` training control API/dashboard panel; `0660f55` preserve v7-pico-005 step-5000 snapshot/recovery doc; `56a9a65` DataLoader `num_workers=0` WSL2 deadlock revert; `d1255c8` WindowsPath shim for cross-OS checkpoint resume; then HAT flash-attn, TF32/DataLoader prefetch, PSNR proxy, fp32 raster smoothing, apex redirect fixes.
- Open issues: #14 VRAM footprint; #13 Nsight L2 hit-rate profile; #12 packed low-rank canvas layout; #10 G-buffer encoder ablation; #9 RepVGG/ECBSR/RepSR backbone ablation; #8 cross-vendor kernel ports; #7 FSR4 teacher distillation; #6 v6.2-pico launch; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- README status snapshot: pre-alpha; v7 `srcnn-v7.0-pico-005` is the active research/teacher architecture with N-D Gaussian canvas, native time axis, 3D Cholesky covariance with `V_xt`, parent-child loss-adaptive density, Mip-Splatting AA filters, and alpha=0.5 OSS-FX supervision. Shipping student is still not trained.
- Training dashboard: visible headline says "v7 is training" at step **30,300 / 100,000**, loss **0.18269**, viz step **26,000**, updated just now/live. Service-health tooltip contradicts that: **Trainer — offline, stuck at step 30,300 for 93h**; Watcher healthy (published 0s ago), CF Worker healthy (HTTP 200 in 1916ms), R2 origin healthy (HTTP 200). GPU stats unavailable / host offline or rsync failed. This is an ops issue, not a research signal.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Learning Video Dynamics with Predictive Differentiable Rendering** (Tang, Zhou, Lin, Tan, Hu; 2026-06-30; arXiv:2606.31050) — Uses a video-prediction model with an internal 3D Gaussian Splatting scene representation and predictive differentiable rendering to reduce over-smoothing in future-frame prediction. Why it matters to OSS: a direct external proof point for Gaussian-state video dynamics / future rendering; worth comparing against OSS-FX's exact-engine-motion, time-slice canvas story. <https://arxiv.org/abs/2606.31050v1>
- **Learning Efficient 4D Gaussian Representations from Monocular Videos with Flow Splatting** (Zhang, Li, Fei, Duan; 2026-06-29; arXiv:2606.29976) — Proposes Flow Splatting to reconstruct dynamic 3D scenes with efficient 4D Gaussian representations from monocular video. Why it matters to OSS: the flow-splatting representation is a relevant comparison for v7's native time-axis/cross-covariance motion encoding and for packed dynamic canvas design. <https://arxiv.org/abs/2606.29976v1>
- **Resonant Brane Splatting for Arbitrary-Scale Super-Resolution** (Federico, Amato, Gennaro, Carrara, Di Benedetto; 2026-06-28; arXiv:2606.29453) — Replaces smooth 2D Gaussians with higher-order Gaussian-Hermite "Brane" primitives to preserve high-frequency detail in arbitrary-scale SR. Why it matters to OSS: already added in the 2026-06-30 brief; remains a priority ablation for edge/detail recovery versus vanilla Gaussian kernels. <https://arxiv.org/abs/2606.29453v1>

### Super-Resolution / Upscaling
- **Fast and Accurate Outlier-Aware LiDAR Super-Resolution for SLAM Applications** (Anagnostopoulos, Gkillas, Piperigkos, Lalos; 2026-06-26; arXiv:2606.28607) — Deep-unrolling SR with explicit outlier removal for real-time SLAM. Why it matters to OSS: not RGB/game SR, but the robust outlier-aware update pattern is relevant to disocclusion/noisy-G-buffer gating and temporal canvas hygiene. <https://arxiv.org/abs/2606.28607v1>
- **Neural Texture Compression using Hypernetworks** (Laurent Belcour; 2026-06-25; arXiv:2606.26913) — Real-time neural texture decoder/hypernetwork approach for PBR materials. Why it matters to OSS: already tracked from 2026-06-28; useful as a tiny material/detail head reference for a student model that can exploit G-buffer/material channels. <https://arxiv.org/abs/2606.26913v1>

### Frame Extrapolation / Generation
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (Shibo Liu; updated 2026-04-01; arXiv:2603.26835) — Uses codec motion-vector priors to avoid deployment-hostile optical-flow sampling in mobile real-time frame interpolation. Why it matters to OSS: already tracked, but still one of the strongest analogues for using supplied motion vectors rather than learning motion from pixels; OSS's engine MVs are the higher-quality version of this signal. <https://arxiv.org/abs/2603.26835v3>
- **R5DGS: Semantic-Aware 4D Gaussian Splatting with Rigid Body Constraints for Efficient Dynamic Scene Reconstruction** (Gridusov, Popov, Kolyubin; 2026-05-25; arXiv:2605.25909) — Adds semantic and rigid-body constraints for dynamic 4DGS reconstruction/future extrapolation. Why it matters to OSS: already tracked; relevant to future static/rigid/transient canvas decomposition if v7 needs object-aware stability. <https://arxiv.org/abs/2605.25909v1>

### Inference Optimization
- **MLVC: Multi-platform Learned Video Codec for Real-World Deployment** (Pärnamaa et al.; 2026-06-26; arXiv:2606.28027) — Targets cross-platform learned video codec deployment with deterministic behavior across heterogeneous hardware. Why it matters to OSS: adjacent but useful for cross-vendor determinism and deployment constraints; not a direct upscaler paper. <https://arxiv.org/abs/2606.28027v1>
- **SEADA: An efficient methodology for optimizing mixed-precision DNNs on multi-precision spatial architectures** (Fiorin, Ronzani, Silvano; 2026-06-26; arXiv:2606.27884) — Mixed-precision DNN mapping for spatial architectures. Why it matters to OSS: candidate reading for FP8/INT8 student mapping methodology, though not graphics/SR-specific. <https://arxiv.org/abs/2606.27884v1>
- **SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference** (Meng et al.; 2026-06-25; arXiv:2606.26587) — Low-bit activation sparsity + FP4 quantization with fused preparation kernels. Why it matters to OSS: already tracked; relevant as a kernel-design warning for future FP8/FP4 activation paths. <https://arxiv.org/abs/2606.26587v1>

## Archive.org Findings
- Archive.org keyword search for `"gaussian splatting" "super-resolution"` returned 3 noisy inventory/star-list style hits; no useful primary paper mirror surfaced.
- Archive.org keyword search for `"frame generation" "deep learning"` returned 4 broad video-generation / temporal-chunking style items; no OSS-critical taken-down paper found.
- Archive.org keyword search for `"real-time upscaling" game` returned 0 hits.
- Targeted Wayback CDX checks for GS-STVSR, GSASR, GRAPE, and the 2026-06-30 Brane/Adaptive-Gaussian arXiv pages timed out or returned 503 during this run. Treat as Archive.org availability failure, not absence of captures.

## GitHub / Open Source
- **GiulioFede/ResonantBraneSplatting** — new/active repo for Resonant Brane Splatting, pushed 2026-07-01, Python, 0 stars, no license detected. Opportunity: code may let OSS test Brane primitives as a direct v7 raster/kernel ablation. <https://github.com/GiulioFede/ResonantBraneSplatting>
- **Azusa309/MedGSSR** — ECCV 2026 medical feed-forward Gaussian SR/reconstruction repo, MIT, 2 stars, pushed 2026-06-22. Relevant only as another hierarchical feed-forward GS-SR datapoint; domain mismatch is high. <https://github.com/Azusa309/MedGSSR>
- **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** — MIT Python repo, pushed 2026-07-01, static structural analysis of AMD FSR 4.1.0 neural upscaler with DXIL entrypoint inventory, weight extraction, blob layout, strict verification, and rebuild comparison. Threat/opportunity: useful open-source analysis handle for FSR 4.1 internals and distillation-teacher verification; also shows public reverse-engineering around FSR 4.x is accelerating. <https://github.com/Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling>
- NVIDIA developer blog currently exposes **"What's New for Game Developers in NVIDIA RTX: DLSS 4.5 for UE5 and Multilingual AI Characters"** and **"Build AI-Powered Games with NVIDIA DLSS 4.5, RTX, and Unreal Engine 5"** snippets. Competitive signal: NVIDIA messaging has moved to DLSS 4.5 / Dynamic Multi Frame Generation / MFG 6X.
- AMD GPUOpen/FidelityFX releases page exposes **FSR SDK v2.3.0**, **FSR Upscaling 4.1.1**, Ray Regeneration 1.2.0, and Redstone DLL ML tech snippets. Competitive signal: FSR 4.1+ is now a live SDK/release train, not just a leaked/orphaned artifact.
- Microsoft DirectX blog shows **Automatic Super Resolution Preview Comes to the ROG Xbox Ally X for Docked Play**. Watch for OS-level/upscaler integration pressure on handheld/console-adjacent OSS positioning.

## News
- DLSS: Google News RSS surfaced NVIDIA DLSS 4.5 coverage and Linux/NVK experimental DLSS support via imported CUDA binaries. Treat headline-level until primary posts are deep-read.
- FSR: multiple headlines report AMD **FSR 4.1** official RX 7000 support / 300+ games / Valve Steam Machine testing. This is a competitive threat: AMD appears to be widening ML upscaler availability and positioning FSR 4.1 for handheld/Steam hardware.
- Intel: Arc G3 Extreme handheld headlines appeared, but no primary XeSS technical update was confirmed in this run.
- Apple/Metal: Apple Metal page still highlights MetalFX/sample-code library; news hits were mostly broad Metal-gaming commentary and benchmark pages, not a concrete MetalFX upscaler change.
- Unreal/Unity: Unreal-related news mostly overlaps AMD FSR SDK 2.3 / FSR 4.1 headlines. Unity-specific search returned no useful fresh items.

## Action Items / Follow-ups
- [ ] Deep-read **Learning Video Dynamics with Predictive Differentiable Rendering** for any loss/state representation that can strengthen OSS-FX future-frame supervision.
- [ ] Inspect **Flow Splatting** for representation-level differences versus v7 `V_xt` cross-covariance and parent-child dynamic canvas design.
- [ ] Clone/read **GiulioFede/ResonantBraneSplatting** once license/code is clear; evaluate whether Brane primitives can be isolated as a rasterizer primitive ablation.
- [ ] Review **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** for FSR 4.1 operator/weight-layout claims and compare with existing FSR4-SDK provenance notes.
- [ ] Ops: dashboard needs reconciliation — public copy says live/training/ETA, but health tooltip says trainer offline stuck at step 30,300 for 93h and GPU stats unavailable.
