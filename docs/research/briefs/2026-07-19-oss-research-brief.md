# OSS Research Brief — 2026-07-19

_Collected from local git/README, GitHub REST, rendered dashboard, arXiv Atom plus abstract-page verification, Archive.org/Wayback, official NVIDIA/AMD pages, and Google News RSS (labelled where not primary)._ 

## Project Status Summary
- Latest commits: `a97a180` tailnet-gated dashboard controls; `fcf8084` control API/panel; `0660f55` v7-pico-005 step-5K recovery snapshot; `56a9a65` WSL2 DataLoader rollback; `d1255c8` cross-OS checkpoint shim; `e1ab7fe` FlashAttention-2 path; `e9b4b65` TF32/prefetch; `26cd9da` corrected PSNR proxy; `83b4b02` apex redirect.
- Open issues (12): #14 v6.2 VRAM measurement; #13 Nsight L2 profile; #12 packed low-rank canvas; #10 decoupled G-buffer ablation; #9 reparameterized student; #8 HIP/Metal/Vulkan/SYCL ports; #7 FSR4-teacher distillation; #6 legacy launch; #5 no-ML shader; #4 ≤1M HAT-Tiny student (plus two older open issues not in the top-ten listing).
- README snapshot: pre-alpha v7 teacher uses a native-time N-D Gaussian canvas (`V_xt` correlation, parent/child density, Mip-Splatting, α=0.5 OSS-FX supervision). No shipping student or non-CUDA backend exists yet.
- Training dashboard: `srcnn-v7.0-pico-005` is unchanged at **35,200 / 100,000**, loss **0.10166**, viz **35,077**. It displays `live`/“updated just now,” but the health tooltip says **Trainer offline, stuck at step 35,200 for 72h**. Watcher is healthy (published 0s ago), CF Worker 200/78ms, R2 origin 200; GPU stats unavailable. Step/loss remain unchanged from July 18, so this is continued outage—not recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation](https://arxiv.org/abs/2607.14203)** (arXiv:2607.14203; Jiahui Huang, Jiawei Ren, Michal Tyszkiewicz, Bjoern Haefner, Michael Shelley, Xin Kang, Seung Wook Kim, Ning Xu, Qi Wu, Janick Martinez Esturo, Shengyu Huang, Nick Schneider, Laura Leal-Taixe, Zan Gojcic, Sanja Fidler; 2026-07-15) — NVIDIA reconstructs a layered static/dynamic 3DGS world, sky cubemap, and ISP corrections from a short calibrated multi-camera driving log in one forward pass; it reports 1.5s per 10–20s scene and +2.01 dB over its strongest evaluated baseline on Waymo. **Why it matters to OSS:** a credible feed-forward static/dynamic-layering and non-pinhole (`3DGUT`) reference for v7 canvas partitioning and future multi-camera/capture-data work, though it is offline simulation rather than a game SR/FG runtime.
- No additional newly submitted material from the requested Gaussian-temporal queries cleared the relevance gate. A fresh arXiv Atom pass timed out then returned HTTP 429 after the earlier verified capture; no search-snippet-only paper was promoted. Recent direct references (SPIN-4DGS, HyperGS, Grassmannian Splatting) remain already tracked.

### Super-Resolution / Upscaling
- No new material paper. arXiv’s latest targeted results were already tracked (notably NanoVSR and AVSR-Diff); remaining hits were face/medical/remote-sensing work with no direct OSS architecture or deployment transfer.

### Frame Extrapolation / Generation
- No new material paper. Existing direct baseline **MASS** remains the relevant motion/occlusion refinement reference; broad results were generative or domain-specific and did not justify bibliography inclusion.

### Inference Optimization
- No new TensorRT/INT8/kernel-fusion SR paper with direct OSS impact. The most relevant recent student/deployment entries remain NanoVSR and deployment-aware INT8 SR, both already tracked.

## Archive.org Findings
- Inventory: `"gaussian splatting" super-resolution` returned three noisy GitHub-list archives; `"frame generation" deep learning` returned DLSS Swapper/FluidFrames binaries plus one unrelated OSF record; `"real-time upscaling" game` returned zero. No archival primary paper discovery.
- Targeted Wayback confirms preserved snapshots for GS-STVSR’s known arXiv URL (2024-07 onward), GSASR’s GitHub repository (including 2026-06), and GRAPE’s arXiv URL. The generic STSS query had no CDX rows. No evidence that GS-STVSR, GSASR, GRAPE, or STSS was removed or materially revised.

## GitHub / Open Source
- GitHub REST searches since 2026-07-05 found **no credible new Gaussian-SR repo or DLSS/FSR/XeSS analysis tool**. Results for “neural upscaler” were zero-star toy/GAN WIPs; broad “frame generation” mostly returned unrelated repositories.
- **[OpenFG](https://github.com/OpenFGG/openfg-app)** remains an ecosystem signal, not an OSS architectural competitor: Android root/Zygisk module for interpolated FG in OpenGL ES/Vulkan games, with HDR controls and emulator support. Its README explicitly says binaries-only/closed source. **Relevance:** validates demand for interception-style FG control surfaces while reinforcing anti-cheat and integration-risk constraints for OSS’s planned DLL shim.
- Official SDK repos were inspected; no new OSS-relevant source drop was identified in this window.

## News

### NVIDIA / DLSS
- **Verified primary — DLSS 4.5 UE plugin:** NVIDIA’s current developer page lists July 2026 DLSS 4.5 plugin package **v8.7.0** for UE 5.5–5.7 (Streamline 2.11.1 / NGX 310.6.0), with Dynamic Multi Frame Generation, up-to-6× MFG, and second-generation-transformer Super Resolution. **Why it matters:** OSS’s unified-SR+FX value proposition is now compared with a stronger, explicit vendor two-path integration baseline; make UE5 input-contract compatibility and artifact/latency measurements first-class.

### AMD / FSR
- **Verified primary — FSR SDK 2.3 (2026-06-24):** ML FSR Upscaling 4.1.1 is now supported on RX 7000/RDNA 3 as well as RDNA 4; the update also improves FSR Frame Generation 4.0.1 motion-vector/camera input handling and adds Ray Regeneration 1.2 options. **Why it matters:** AMD materially expanded the ML-upscaling competitive bar; RDNA3 needs to be in the OSS HIP/HLSL benchmark and backend plan, not only RDNA4.
- **RSS-only lead:** TechPowerUp reported AMD 8× MFG development on July 13; no matching primary AMD announcement was verified, so treat it as unconfirmed.

### Intel / Apple / Game Engines
- No primary XeSS, Apple MetalFX, Unreal, or Unity upscaling-specific release was found in the 14-day window. RSS/search results were generic engine/forum content and are not treated as product signals.

## Action Items / Follow-ups
- [ ] **Critical:** recover or inspect the 3080 Ti trainer. Health moved from 48h to 72h offline while step/loss/viz did not advance; watcher/CDN health is not training health.
- [ ] Deep-read Instant NuRec’s static/dynamic-layer allocation and `3DGUT` compatibility; decide whether a cheap static/rigid/transient canvas ablation is warranted before adding complexity to `V_xt`.
- [ ] Implement issue #9’s reparameterized ≤1M student baseline using NanoVSR-style standard-convolution collapse, retaining only motion/G-buffer signals that prove their cost.
- [ ] Add RDNA3 to the benchmark matrix and turn issue #8 into a concrete HIP/HLSL WMMA feasibility spike; FSR 4.1.1 widened AMD’s hardware coverage.
- [ ] Track DLSS 4.5 Dynamic MFG and AMD’s unverified 8× MFG claim as competitive targets, but compare against measured base-frame latency and artifacts rather than advertised displayed FPS.

## Bibliography Update
- Added `Instant NuRec` once as a newly surfaced, materially relevant feed-forward GS layering reference. No other new paper met the direct architecture/training/inference/positioning relevance bar.
