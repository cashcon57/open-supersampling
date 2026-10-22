# OSS Research Brief — 2026-08-31

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` is the sole entry reachable from this checkout's `git log -10`.[unverified]
- **Open issues:** **12**. The active critical path is still #12 packed low-rank canvas layout, #13 Nsight L2 profiling, #14 VRAM measurement, #3 tile binning, then #4/#7/#9 student distillation and #8 cross-vendor ports.[12]
- **README snapshot:** pre-alpha v7 remains the N-D `(x,y,t)` Gaussian-canvas **teacher**; the ≤1M shipping student and CUDA/HIP/Metal/Level Zero/Vulkan production backends remain unbuilt.[11]
- **Training dashboard:** `srcnn-v7.0-pico-005` is still **36,650 / 100,000**, loss **0.19268**. It reports **Trainer — offline; stuck 17h**, while Watcher, CF Worker, and R2 are healthy and GPU stats are unavailable.[1] This is not recovery: step/loss are unchanged from Aug. 30, while the stale-duration counter reset from 157h to 17h; the live/ETA copy remains contradictory.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[ABCD: Alpha-Composited Block Coordinate Descent: Constant-VRAM Training for Large Radiance Fields](https://arxiv.org/abs/2608.27735)** (Ka Heng Shiu, Kartic Subr; 2026-08-27; arXiv:2608.27735) — Out-of-core 3DGS training updates one spatial block while alpha-compositing frozen regions, making peak VRAM independent of total scene extent for fixed partition/image sizes.[2] **Why it matters to OSS:** a high-value v7 teacher-training ablation for canvas scale on constrained VRAM; test temporal block boundaries and disocclusions before adopting.
- **[Non-Uniform Quantisation for 3DGS Compression](https://arxiv.org/abs/2608.28272)** (Bert Van Hauwermeiren, Patrice Rondao Alface, Adrian Munteanu; 2026-08-28; arXiv:2608.28272) — Importance-weighted quantisation plus importance-weighted merging targets 3DGS storage/bitrate and MPEG 3DGS standardisation.[3] **Why it matters to OSS:** direct candidate for #12 packed canvas, but only after proving temporal primitive identity, fixed-budget quality, and L2-residency behavior.
- **[WilLaGS: Latent-Conditional 3D Appearance Fields for Robust Gaussian Splatting In-the-Wild](https://arxiv.org/abs/2608.28240)** (Yuhao Bai, Qianqiu Tan, Lilong Chen, Huanhuan Lv, Lijun Chen; 2026-08-28; arXiv:2608.28240) — an EMA teacher/student perceptual mask identifies inconsistent/transient regions under changing appearance.[4] **Why it matters to OSS:** medium-priority control for invalid engine MVs, lighting shifts, and disocclusions; its dynamic appearance field is not shipping-student compatible.

### Super-Resolution / Upscaling
- No new direct real-time/video-SR paper cleared the architecture or ≤1M deployment gate in the Aug. 31 cs.CV listing review.[13] Existing NanoVSR, GS-STVSR, GRAPE, GSASR, and reliability-gated Gaussian-SR papers remain the better controls; no duplicate bibliography entry was added.[unverified]

### Frame Extrapolation / Generation
- No new material single-network extrapolation paper was found in the Aug. 31 cs.CV listing review.[13] Amulet and SNM-VFI remain the relevant already-tracked controls for visibility refresh and invalid-motion confidence, respectively; neither displaces v7 fractional-time canvas rendering.[unverified]

### Inference Optimization
- No new material TensorRT/INT8/kernel-fusion paper changed priorities in the Aug. 31 cs.CV listing review.[13] Measure #12 packed layout, #3 bin/sort cost, and #13 L2 hit rate before selecting quantisation/fusion designs; treat the new quantisation paper as a storage-layout ablation, not proof of real-time speedup.[unverified]

## Archive.org Findings
- Internet Archive Advanced Search remains low signal: exact `"gaussian splatting" super-resolution` returned only three GitHub snapshot artifacts; `"frame generation" deep learning` returned mixed FluidFrames/registration artifacts; `"real-time upscaling" game` returned zero.[5]
- Targeted searches found zero hits for GS-STVSR and GRAPE, an unrelated CIA item for GSASR, and an unrelated optics paper for STSS.[5] This is neither a takedown nor an absence claim—Archive.org is not a reliable primary archive for these papers.[unverified]

## GitHub / Open Source
- **[OptiScaler](https://github.com/optiscaler/OptiScaler)** remains the material integration watch, not a model competitor: the GPL-3.0 C++ adapter bridges DLSS/XeSS/FSR inputs and was pushed Aug. 31 with 10,099 stars.[6] Its Aug. 31 changes correctly gate extension spoofing on config and disable Vulkan spoofing after a successful patcher run.[7] **OSS implication:** use it as an interoperability regression-test reference for DLL-shim/Vulkan capability negotiation; do not depend on GPL code.
- GitHub REST searches constrained to Aug. 30–31 produced no material new Gaussian-SR/splat repository, open DLSS/FSR analysis tool, or engine-MV upscaler competitor.[14]

## News
- **NVIDIA, Gamescom:** NVIDIA now explicitly markets DLSS 4.5 Ray Reconstruction as available with its second-generation transformer, and says the DLSS 4.5 Unreal Engine plugin exposes Dynamic MFG, 6× MFG, and the updated Super Resolution transformer.[15][16] **Threat:** the quality/integration bar is moving as a bundled engine-plugin surface; OSS needs a concrete quality/latency and input-contract benchmark rather than a generic DLL-shim claim.
- **Unity lead (not primary vendor confirmation):** a Unity Community Showcase post announced XeSS 3 Frame Generation on Aug. 28, alongside recently listed DLSS 4 and FSR 4 FG assets.[17] Treat this as an integration-distribution signal only—the post is a community asset announcement, not a Unity/Intel platform commitment.
- No fresh primary AMD FSR, Intel Arc/XeSS, Apple/MetalFX, or core Unreal/Unity engine-performance update was verified beyond those integration signals. The prior FSR SDK 2.3 / ML FSR 4.1.1 + ML FG 4.0.1 and XeSS 3 context remains the competitive baseline.[8][9]

## Action Items / Follow-ups
- [ ] **P0 ops:** recover, terminate, or explicitly document the v7 trainer. The progress metric is frozen at 36,650 even though the dashboard's offline-duration counter reset.[1]
- [ ] **#12/#13:** test importance-weighted quantisation/merge against the current packed low-rank layout: bytes/Gaussian, active Gaussians/tile, temporal-ID survival, L2 hit rate, raster time, and quality at fixed 1080p-LR budget.[unverified]
- [ ] **Teacher scale:** prototype ABCD-style spatial blocks only if the v7 run is revived; acceptance must include cross-block temporal continuity and disocclusion behavior, not just VRAM/PSNR.[unverified]
- [ ] **Robustness:** add an EMA disagreement/confidence-map ablation after exact-engine-MV warp; score bad MVs, lighting shifts, and newly revealed regions separately.[unverified]
- [ ] **Integration watch:** use OptiScaler-style Vulkan extension/capability handling as an external regression target for the eventual OSS DLL shim, without copying GPL implementation.[unverified]

## Sources

[1] https://opensupersampling.com
[2] https://arxiv.org/abs/2608.27735
[3] https://arxiv.org/abs/2608.28272
[4] https://arxiv.org/abs/2608.28240
[5] https://archive.org/advancedsearch.php
[6] https://api.github.com/repos/optiscaler/OptiScaler
[7] https://api.github.com/repos/optiscaler/OptiScaler/commits?since=2026-08-30T00%3A00%3A00Z
[8] https://gpuopen.com/amd-fsr-sdk
[9] https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/gaming-technologies.html
[10] https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-generation-6x-mode-released
[11] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[12] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[13] https://arxiv.org/list/cs.CV/new
[14] https://api.github.com/search/repositories?q=Gaussian%20splatting%20super%20resolution%20pushed%3A2026-08-30..2026-08-31&sort=updated&order=desc
[15] https://blogs.nvidia.com/blog/gamescom-rtx-spark-pc-games-technology/
[16] https://developer.nvidia.com/blog/build-on-device-ai-companions-with-the-nvidia-ace-game-agent-sdk-and-unreal-engine-5-plugins/
[17] https://discussions.unity.com/t/announcement-xess-3-frame-generation-for-unity/1735053
