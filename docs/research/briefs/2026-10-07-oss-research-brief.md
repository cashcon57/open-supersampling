# OSS Research Brief — 2026-10-07

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (the sole entry returned by local `git log --oneline -10`).
- **Open issues (first 10):** #14 D3 VRAM footprint; #13 H004 Nsight L2 profile; #12 packed low-rank canvas; #10 G-buffer encoder ablation; #9 student reparameterization; #8 cross-vendor ports; #7 distillation; #6 v6.2-pico-002 launch; #5 No-ML shader; #4 HAT-Tiny student.
- **README:** describes v7 N-D Gaussian canvas/time-axis teacher `srcnn-v7.0-pico-005` in training, while shipping student is not yet trained; cross-vendor kernels and DLL shim are designed, not implemented. Report README claims separately from dashboard.
- **Dashboard (extracted Oct 7):** headline shows active `srcnn-v7.0-pico-005`, step 36,650/100,000, loss 0.19268, ETA ~1.1d. Health says **Trainer — offline**, “stuck at step 36650 for 426h”; Watcher healthy (published 0s ago), CF Worker HTTP 200/108ms, R2 origin HTTP 200. Live headline conflicts with trainer health; unchanged step/loss versus Oct 6 means ETA is not evidence of progress. Health counter rose 402h→426h.

## New Papers (arXiv)
Metadata checked on arXiv abstract pages; these are abstract-level summaries, not independent reproduction. Both relevant papers were already covered in the Oct 6 brief / bibliography; no new bibliography append warranted.

### Gaussian Splatting / Temporal
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** — Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha (2026-04-20; [arXiv:2604.18047](https://arxiv.org/abs/2604.18047)). Continuous-time Gaussian video SR combines flow-guided motion, covariance resampling and adaptive offsets; direct architectural comparator, while game motion vectors and OSS's native N-D canvas remain distinguishing inputs/designs.
- **3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** — Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu (2026-05-12; [arXiv:2605.11489](https://arxiv.org/abs/2605.11489)). Combines gradient-guided GRU supersampling and a compact U-Net temporal interpolation module over differentiable 3DGS render outputs; competitive conceptual overlap, but uses separate post-rendering modules rather than one time-query canvas. Code described as forthcoming on acceptance.

### Super-Resolution / Upscaling
- No additional newly verified, directly material paper surfaced in this scan.

### Frame Extrapolation / Generation
- No additional newly verified, directly material paper surfaced in this scan; see 3DGS³ above as interpolation (not demonstrated game-frame extrapolation).

### Inference Optimization
- No new verified inference paper surfaced.

## Archive.org Findings
- Archive.org `advancedsearch.php` keyword checks returned noisy inventories: “gaussian splatting super-resolution” (3 hits, mostly archived GitHub lists), “frame generation deep learning” (248, largely unrelated media), and “real-time upscaling game” (107, mostly unrelated). No confirmed archived GS-STVSR/GSASR/GRAPE/STSS paper copy was identified. This is inconclusive, not proof of absence.
- Wayback availability endpoint for GS-STVSR returned HTTP 404 in this environment; no archive conclusion.

## GitHub / Open Source
- GitHub REST repository search (`pushed:>=2026-09-23`, updated-sorted) surfaced **FoundDSR** ([yanzq95/FoundDSR](https://github.com/yanzq95/FoundDSR), pushed Sep 25, 1 star): generalizable 2D-Gaussian depth SR, not direct game-color upscaling or frame generation; early lead only.
- Open-source frame-gen search was heavily noisy. **DLSS Unlocked** ([ShyVortex/dlss-unlocked](https://github.com/ShyVortex/dlss-unlocked), pushed Oct 4, 757 stars) claims to unlock DLSS/MFG features on RTX GPUs; compatibility/unlock tooling rather than a new OSS SR/FG model. **Kinetik** ([WastedDesigner/Kinetik](https://github.com/WastedDesigner/Kinetik), Oct 2) is an FPS overlay, not a model. No verified fresh direct competitor emerged.

## News
- **DLSS 4.5 / games (Oct 1–6):** NVIDIA's [primary GeForce news](https://www.nvidia.com/en-us/geforce/news/) was cited in yesterday's brief; current Google News RSS surfaces The Witcher 3 remaster integration and reports around multi-frame generation. RSS/headline leads only here; not independently revalidated against vendor pages today.
- **FSR/XeSS/Metal:** RSS surfaced FSR 4.1.1 performance/compatibility claims and Intel XeSS override behavior, but no primary release material verified in this scan. Treat as unconfirmed leads. No Apple Metal performance release substantiated.
- **GPU/engine updates:** no primary-source-verified new announcement found in today's scan.

## Action Items / Follow-ups
- [ ] Verify trainer process and metrics on host; dashboard remains offline/stuck at 36,650 for 426h despite live label/ETA.
- [ ] Deep-read GS-STVSR and 3DGS³; compare temporal parameterization, intermediate-time scaling, and what their video/3DGS inputs imply for game motion-vector-conditioned extrapolation.
- [ ] Assess FoundDSR only if its generalizable Gaussian depth-SR method has a transferable representation or supervision technique; do not treat as direct competitor.
- [ ] Recheck primary NVIDIA/AMD/Intel/Apple release pages before converting RSS leads into competitive claims.

## Research Refresh — Oct 7
- **Dashboard check:** both browser rendering and direct HTTP fetch were blocked (browser executable permission denied; origin HTTP 403). The status above is the existing Oct 7 extracted snapshot, not independently refreshed in this run; do not interpret it as newly observed.
- **Additional paper lead:** *Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling* — Long Peng, Anran Wu, Wenbo Li, Peizhe Xia, Xueyuan Dai, Xinjie Zhang, Xin Di, Haoze Sun, Renjing Pei, Yang Wang, Yang Cao, Zheng-Jun Zha (2025-03-09; [arXiv:2503.06617](https://arxiv.org/abs/2503.06617)). Abstract reports one-pass pixel-to-Gaussian reconstruction and ~1 ms arbitrary-scale rendering; inspect as a rendering-cost / continuous-reconstruction baseline, but its image SR setting does not establish game temporal performance. A reproduction repo exists but states the original weights/code are unavailable and its own model is non-commercially licensed, so treat as research reference, not a shipping dependency.
- **Additional Gaussian/time lead:** *Robust, Estimator-Agnostic Dynamic 3DGS Compression* — Chenjunjie Wang, Zixi Huang, Yao Wang, Jona Ballé (2026-09-22; [arXiv:2609.25633](https://arxiv.org/abs/2609.25633)). Encodes groups of frames as indexed Gaussians and exploits temporal redundancy with a static codec; an adjacent opportunity for v7 canvas memory/bandwidth measurements, not an SR or game-frame-generation competitor.
- **Competitive threat / opportunity:** AMD GPUOpen’s current page announces FSR SDK 2.3 (June 24, 2026), FSR Upscaling 4.1.1 on RX 7000 plus Frame Generation 4.0.1, and an Unreal 5.8 plugin update (Aug 27). This strengthens the open SDK + multi-feature competitor surface; OSS differentiation must be demonstrated as one shared temporal canvas and vendor-neutral runtime, not merely “open source.” NVIDIA’s primary blog search result lists a Sep 22 DLSS 5 3D-guided neural-rendering announcement—adjacent neural-rendering competition, not evidence of equivalent temporal canvas architecture. Intel-specific fresh primary updates were not verified.
- **Archive.org:** latest direct Advanced Search query for exact Gaussian-splatting + super-resolution returned only three archived GitHub-list snapshots, none a verified archived target paper. Search snippets also surfaced archived original Graphdeco repository snapshots (Aug 2023); no confirmed paper archive among requested GS-STVSR/GSASR/GRAPE/STSS titles.
- **Coverage limits:** current web/arXiv queries were discovery searches, not exhaustive category searches; direct dashboard inaccessible. Archive broad queries and GitHub search are noisy. No claim of comprehensive absence is warranted.
