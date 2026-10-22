# OSS Research Brief — 2026-08-29

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke`.
- **Open issues:** **12**, led by packed low-rank canvas layout / VRAM / L2 profiling (#12–#14), tile binning (#3), and student distillation / cross-vendor execution (#4/#7/#8/#9).[1]
- **README snapshot:** OSS remains pre-alpha. v7 is the N-D `(x,y,t)` Gaussian-canvas teacher; the ≤1M shipping student and native CUDA/HIP/Metal/Level Zero/Vulkan kernels are not built.[2]
- **Training dashboard:** `srcnn-v7.0-pico-005` is unchanged at **36,650 / 100,000**, total loss **0.19268**, SR Charbonnier **0.02360**, FG Charbonnier **0.07643**, and **2,304** canvas elements.[3] At the 2026-08-30T00:01Z dashboard refresh, the status endpoint reports the trainer **offline/stuck for 145h** while Watcher, CF Worker, and R2 are healthy.[4] “training now” UI copy is stale/conflicting; no evidence of resumed training.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **4DGS-WAM: Bridging Past and Future with an Object-Centric World Action Model based on 4D Gaussian Splatting** (arXiv:2608.25956; Yueen Ma, Zenglin Xu, Irwin King; 2026-08-26) — Uses 4D Gaussian scene state in an action-conditioned world model. **Why it matters to OSS:** only as a teacher-side ablation lead for maintaining time-aware primitive identity under motion; it is not a real-time image-space SR or engine-MV architecture, so it does not clear the bibliography gate.[12]
- **M³ISR: A Multi-Modal Multi-View Benchmark for 3D/4D Gaussian Splatting and Feedforward Compression** (arXiv:2608.22465; Xinhui Liu, Lei Liu, Zhenghao Chen, Lebin Zhou, Wei Wang, Wei Jiang; 2026-08-23) — Benchmark for dynamic/static 3D/4DGS and feed-forward compression. **Why it matters to OSS:** an external rate-distortion/control source for #12 canvas residency and update cost, but it was already added to `BIBLIOGRAPHY.md`; no duplicate entry.[13]
- The targeted temporal/GS sweep additionally returned underwater reconstruction, SLAM, physics, and generative-prior work. Those do not presently affect v7 canvas design, game training, or the shipping latency plan.[5]

### Super-Resolution / Upscaling
- No new material real-time game-SR, VSR-student, or neural-rendering-SR paper cleared the architecture/training/inference gate. The fresh broad result set was dominated by radar, satellite, RAW restoration, and generic perceptual SR; retain NanoVSR and deployment-aware INT8 SR as existing student controls rather than changing the ≤1M plan.[5]

### Frame Extrapolation / Generation
- No new material paper. **Amulet** and **SNM-VFI** appeared in the sweep but are already tracked: retain Amulet for sparse-layer visibility age / adaptive refresh and SNM-VFI for confidence-gated invalid-motion controls.[14][15]

### Inference Optimization
- No new TensorRT, INT8, or kernel-fusion result changes ordering. Profile packed canvas layout and L2 residency (#12/#13) before choosing attribute precision, sort keys, or fusion boundaries.[1][5]

## Archive.org Findings
- Archive Advanced Search remains low-signal: `"gaussian splatting" AND "super-resolution"` returned three GitHub snapshot artifacts; `"frame generation" AND "deep learning"` returned three mixed items (including FluidFrames and a registration); `"real-time upscaling" AND game` returned zero results.[6]
- Targeted Wayback CDX queries for GS-STVSR, GSASR, GRAPE, STSS, and 3DGS³ returned HTTP 400 from the endpoint during this run. This is an archive-query failure, **not** evidence that those works lack captures or were removed.[6]

## GitHub / Open Source
- **IE-SRGS**: 1 star, `NOASSERTION` license, pushed Aug 27. It is an official 3DGS-SR research implementation, but its positioning is offline high-fidelity reconstruction rather than a deployable game student. Watch, do not treat as a threat.[7]
- **OSSS**: MIT, 0 stars, pushed Aug 25. Window-level Windows frame interpolation/upscaling is a product-surface watch; its zero-dependency classical approach is not an engine-MV, Gaussian-canvas, or quality peer.[8]
- No new credible DLSS/FSR analysis tool surfaced from fresh GitHub REST queries.[7][8]

## News
- **NVIDIA:** DLSS 4.5 Ray Reconstruction became available Aug 28; its model/version rollout raises the integration UX bar, but does not alter OSS’s shared SR/OSS-FX architecture.[9]
- **AMD:** GPUOpen now lists an FSR Unreal Engine 5.8 plugin update with ML upscaling + frame generation and FSR Upscaling support extending to Radeon RX 7000-series GPUs. This is a concrete engine-integration competitive move; monitor its per-title controls and supported hardware matrix.[10]
- **Intel / Apple / GPU announcements:** no primary Intel XeSS, Apple Metal/MetalFX, or material GPU-release change was verified. Google News has an Arc B580/SteamOS report mentioning absent XeSS driver support, but that is headline-level reporting rather than a vendor claim.[11]

## Action Items / Follow-ups
- [ ] **P0 ops:** resolve or explicitly close the v7 trainer; training is unchanged at 36,650 and the service reports 145h offline.[3][4]
- [ ] **#12/#13:** use M³ISR-style streaming/rate-distortion measurements as a secondary packed-canvas control, but preserve engine-MV temporal-identity and L2-residency gates.[1][13]
- [ ] **Student:** keep reparameterizable CNN / INT8 experiments gated on fused ≤1M, 1080p-LR measurements.[2][5]
- [ ] **OSS-FX:** retain Amulet-style visibility age, refresh cadence, and frame-pacing telemetry beside PSNR/LPIPS.[14]
- [ ] **Threat watch:** capture FSR UE5.8 plugin support/quality behavior; its RX 7000 extension tightens AMD’s engine deployment story.[10]

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[2] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[3] https://opensupersampling.com/data.json
[4] https://opensupersampling.com/status.json
[5] https://export.arxiv.org/api/query?search_query=%28abs%3A%22gaussian%20splatting%22%20AND%20%28abs%3Atemporal%20OR%20abs%3Avideo%20OR%20abs%3Adynamic%20OR%20abs%3A%22super-resolution%22%29%29&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending
[6] https://archive.org/advancedsearch.php
[7] https://api.github.com/repos/ZheXues/IE-SRGS
[8] https://api.github.com/repos/Lenovive/OSSS
[9] https://www.nvidia.com/en-us/geforce/news/gamescom-2026-nvidia-geforce-rtx-dlss-4-5-announcements
[10] https://gpuopen.com/learn
[11] https://news.google.com/rss/search?q=Intel+Arc+XeSS+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen
[12] https://arxiv.org/abs/2608.25956
[13] https://arxiv.org/abs/2608.22465
[14] https://arxiv.org/abs/2608.10423
[15] https://arxiv.org/abs/2608.13460
