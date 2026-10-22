# OSS Research Brief — 2026-09-11

> Morning run. New since the 2026-09-10 brief: Wayback came back online and confirms **Amulet has zero captures** (abs + pdf) — it is now the top un-mirrored critical reference. NVIDIA officially committed DLSS 5 Neural Rendering to RTX 40 (after the mod wave forced it). DLSS Unlocked keeps climbing (273 → 299★). arXiv produced a quiet week for SR/frame-gen but three adjacent GS/distillation papers worth recording. Trainer outage fourth consecutive brief: 269h frozen.

## Project Status Summary

- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (shallow local checkout; single reachable commit, dated 2026-06-08). No new commits reachable from this clone.[1]
- **Open issues:** 12, unchanged set (all last updated 2026-05-08). Architecture/runtime: #1/#3/#12/#13/#14 (latent rasterizer, counting-sort tile bin, packed canvas, Nsight, VRAM). Shipping: #4/#7/#8/#9 (≤1M student, FSR 4 teacher signal, vendor kernels, reparameterized backbone). #5/#6/#10 round out v6.2.[1]
- **README snapshot:** unchanged — pre-alpha v7 research; native `(x,y,t)` Gaussian canvas with V_xt Cholesky motion; OSS-FX = same architecture at a different `t_query`; ≤1M TensorRT FP8 student and non-CUDA backends unimplemented.[2]
- **Training dashboard:** `srcnn-v7.0-pico-005` still frozen at **36,650 / 100,000** (36.7%); `sr_charbonnier` 0.0236, `fg_charbonnier` 0.0764, `total` 0.1927, `sr_lpips` 0.0927 — identical since Sep 8. `status.json` (checked 2026-09-11T00:00:45Z): trainer **offline, stuck at step 36650 for 269h**; watcher healthy (1s ago), CF Worker healthy (200 in 105ms), R2 healthy.[3][4]
- **Operational conclusion:** fourth consecutive brief with a hard trainer death — 233h → 245h → 257h → 269h — while publishing infra stays green. Nothing will move until the 3080 Ti host is attended to; the dashboard "training now" headline is still misleading.

## New Papers (arXiv)

Dedup against `BIBLIOGRAPHY.md` (61 unique IDs at run start). **API note:** the arXiv API `all:` field queries that worked in prior runs returned empty this run (index degradation), so the sweep leaned on `cat:cs.GR` (19 new), recent `cs.CV` (50) and `eess.IV` (20) listings plus `abs:` field queries, cross-checked with web search — no missed GS/SR/frame-gen papers found in the Sep 6–11 window through either path. Worth adding a second discovery feed (Hugging Face papers) next run.

### Gaussian Splatting / Temporal

- **EdMCGS: Event-Driven Markov Chain Gaussian Splatting for Extreme-Low-Frame-Rate Dynamic Scene Reconstruction** — Wang, Zhang, Zhu; 2026-09-08; arXiv:2609.08332. Models dynamic-scene evolution as a Markov chain over Gaussian states, using an event stream as inter-frame evidence, then renders **any intermediate timestamp**. **Why it matters to OSS:** nearest published "fill in missing time with splats" formulation this window — the inverse problem shape (sparse temporal evidence → continuous-time Gaussian rendering) is OSS's t_query premise in a different sensor regime. Deep-read candidate: its temporal-identity/update mechanics vs the v7 canvas update rule. Limits: 3D, event-camera input, offline reconstruction — not a latency competitor.[5]
- **View-Structured Conformal Prediction for 3D Gaussian Splatting** — Chu, Li, Huang; 2026-09-09; arXiv:2609.10307. Certifies that rendered views meet prediction-coverage guarantees (with prob 1−α, boxes cover ≥1−β of pixels) rather than just emitting uncertainty heatmaps. **Why it matters to OSS:** evaluation-methodology donor — a conformal gate over rendered canvas frames would give OSS a *certified* disocclusion/hole-coverage story, which no DLSS/FSR/XeSS competitor offers. Park until the eval harness exists.[6]
- **LinearMask-GS: Stable-Mask Importance Pruning for Compact 3D Gaussian Splatting** — Ryu, Kweon; 2026-09-09; arXiv:2609.10095. Identifies Gumbel-Sigmoid slope instability in learned-mask pruning and stabilizes it, pruning compact 3DGS without the usual quality cliff. **Why it matters to OSS:** second compaction reference this week — canvas Gaussian-count control is the same mechanism #12 needs (packed canvas presupposes a bounded Gaussian set). Compare with Compact-FFGS merging before picking a mechanism.[7]
- *(Checked and rejected: CVT-GS 2609.08730 — post-hoc Voronoi simplification, redundant with Compact-FFGS; GSComplete 2609.08449 — diffusion priors, wrong latency class; Shape-guided GS 2609.10376, X-ray/medical; LightSplat 2609.07274, SLAM; Heat Kernel Textures 2609.07557 — alternative primitive math, low relevance; PhysMAS 2609.07174 — 4D synthesis, not reconstruction.)*

### Super-Resolution / Upscaling

- No new arXiv SR papers in the window cleared the materiality bar. NanoVSR (2607.10495), MotionCraft (2608.08553), and NTIRE 2026 efficient-SR (2604.03198) remain the standing references for #4/#9. AstraMoE-SR (2609.07012, satellite jitter SR via MoE diffusion) and OracleZoom (2609.06490, recursive-SR self-distillation) were reviewed and rejected — wrong domain/latency class, though OracleZoom's on-policy self-distillation is a mild concept donor for #4.

### Frame Extrapolation / Generation

- **Nothing new.** The Sep 6–11 window produced zero single-network frame-extrapolation papers (keyword + category sweeps agree). **Amulet (2608.10423) still stands as the only direct architectural threat, with zero movement since Aug 11** — and zero Wayback captures (see Archive section). OSS-FX's "same canvas, different t_query" formulation remains unpublished-competitor-free ~5 months after GS-STVSR.

### Inference Optimization

- **EFQ-Softmax: Exp-Free Quantization for Softmax** — Han et al.; 2026-09-09; arXiv:2609.09721. Removes the exp-then-quantize softmax path for FP8/FP4 attention on tensor cores. **Why it matters to OSS:** only relevant if the ≤1M student ships transformer/attention blocks in the FP8 TensorRT path — flag for the #12→fused-TRT sequencing, not urgent.[8]
- *(Ozaki 2.5 / 2609.09095 — FP8 emulation of FP64 matmul on tensor cores; impressive but irrelevant to OSS's FP8/FP16 student path.)*

## Archive.org Findings

- **Wayback is back online, and the news is bad for Amulet: zero captures** for both `arxiv.org/abs/2608.10423` and `/pdf/2608.10423` (CDX empty; the `/web/2026/` fallback 404s). GS-STVSR's risk profile from last week — total availability of OSS's closest prior formulation resting on one copy — **now applies to Amulet, the primary competitive/systems reference for OSS-FX.** ACTION (new P0): mirror the Amulet PDF to `docs/research/papers/2608.10423-amulet.pdf` next run, same as GS-STVSR.[9]
- Spot-checks: GSASR (2501.06838) well captured since 2025-01-14 — safe. GRAPE (2604.14706) has exactly **one** capture (2026-05-02) — marginal; add to the mirror queue behind Amulet. GS-STVSR (2604.18047) still zero captures; local mirror (18.7 MB, verified last run) remains the only safety copy — commit it.

## GitHub / Open Source

- **DLSS Unlocked (ShyVortex/dlss-unlocked)** — 273 → **299 stars** (+26/24h), pushed today 13:05Z. Commit activity is real engineering (cache-poisoning fix, config regex rewrite, README/release polish), not star farming.[10]
- **OptiScaler (optiscaler/OptiScaler)** — 10,569 → **10,604 stars**, pushed today. FSR 4.1.1b support via OptiScaler now reaches RX 6000 and GeForce (pcgameshardware) — the DLL-swap community keeps expanding vendor reach faster than vendors do.[11]
- **Ecosystem breadth (new, noise-level individually, directionally interesting):** several new repos this week target the same DLL-injection/upscaler-management slot — `KillerPixelCrew/ReScaleFrame` (RE framework for adding upscaling+FG to games), `projectsaturnstudios/neural-upstream` (DLSS 5 NR *before* the game pipeline via ReShade), `jyu041/dlss-rtxsr-upscaler` and `Ishanoshada/DLSS5-VLC-UpScaler` (DLSS 5 outside games, e.g. into VLC), `LoganSMiller/NeuralSwap` (Rust+Tauri upscaler manager), `jatelop8/DynamicShaderFrameGen` (FSR 3.1 FG for Skyrim). None ≥2★; the signal is that the *deployment slot* OSS plans to occupy (per-title DLL interception) is attracting many small entrants — the integration surface is consolidating around OptiScaler-class shims exactly as OSS assumed.
- **AMD FidelityFX SDK:** repo (GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) unchanged since SDK 2.3.0 on 2026-06-24 — no new SDK drop this window; the FSR 4 source release remains promised, not shipped (see News).[12]
- GitHub search (GS+SR, frame-gen upscaler, neural upscaler, pushed>Sep 1): nothing above noise; no new DLSS/FSR analysis tooling. EE5178 course repo for 3DGS-SR is the only GS-SR-adjacent new repo (0★, coursework).

## News

### NVIDIA: DLSS 5 concedes the deterministic framing — and comes to RTX 40
- **NVIDIA officially committed DLSS 5 3D-Guided Neural Rendering to RTX 40** (NVIDIA newsroom/official statements, picked up Sep 8–10) — a direct capitulation to the modding wave covered in the last three briefs. The hardware gate on Ada is being removed by the vendor itself, weeks after modders proved it was artificial.[13]
- **Positioning read (important):** NVIDIA's own DLSS 5 page now states it "takes a frame's color and motion vectors as input to deliver lifelike lighting and materials that are **deterministic, temporally stable and anchored to the game's content**."[14] The industry's flagship neural rendering product now markets itself with OSS's exact framing (engine-buffer + MV conditioned, deterministic). Two implications: (1) the RPCS3-style "hallucination" critique from Sep 9 is being absorbed into vendor marketing — OSS loses the ability to position *against* "generative slop" as a category, because the category leader claims determinism too; (2) OSS's surviving differentiators are narrower and sharper: **OSS renders/extrapolates whole frames from a persistent canvas at arbitrary `t_query`** (frame extrapolation, not enhancement of an engine frame), and does it in **<2ms** where TweakTown's DLSS 5 analysis shows NVIDIA's approach is "effectively a 60 FPS technology" needing MFG 4X to hit 170 FPS on a 5090.[15] The README claim set should be checked against this new reality before any publish.
- DLSS 4.5 Ray Reconstruction (2nd-gen transformer RR) shipped at Gamescom — incremental, noted for completeness. DLSS 5 launched Sep 3 in NBA 2K27 on RTX 50 + GeForce NOW; NBA 2K27 is also streaming with it — first cloud-delivery of a neural-rendering pipeline.[16]

### AMD: FSR 4 source release is the pending earthquake
- New detail this window: AMD's CES roundtable transcript confirms FSR 4 source code **was accidentally published in full in August** and that AMD "intends to release the source for the FSR4 library while keeping the core technology closed" — an unusual split that will land on GitHub eventually. When it does, it is simultaneously (a) a clean-room hazard for #7 (FSR 4 as teacher signal — capture plan must complete *before* any code reading), and (b) the first vendor-grade open-source neural upscaler, a precedent for OSS's own licensing posture. Watch GPUOpen; FidelityFX SDK repo still at 2.3.0 (Jun 24).[17]
- No new FSR quality/feature news beyond the Sep 9 RX 7000 / UE 5.8 plugin coverage.

### Qualcomm / Intel / consoles
- **Qualcomm Adreno Neural Fusion:** no new technical detail since the Sep 2 announcement; Snapdragon Summit will carry specifics. Standing watch item from Sep 10 brief (mobile SR+FG on dedicated GPU matrix cores = Pico student's target regime and "one architecture" validation).[18]
- **Intel:** XeSS 3 MFG driver rollout continues; Game On driver for NBA 2K27 (32.0.101.8991) fixed XeSS MFG crashes. Nothing new on model detail.
- **Consoles/handhelds:** Stellar Blade's Switch 2 port ships with **DLSS + FSR 3.1 FG** on NVIDIA hardware (TweakTown) — frame generation is now standard across console-class fixed-hardware budgets (Switch 2, PS5 Pro/PSSR), which is the exact regime OSS's Pico student targets. Mild but consistent demand signal.[19]
- **Engines:** 007 First Light is the latest 2026 title shipping without current FSR/XeSS support (PCGamer) — vendor-parity gaps persist in AAA; the shim/interception ecosystem keeps absorbing them.[20]

## Action Items / Follow-ups

- [ ] **P0 (carried, worsening):** recover or formally stop `srcnn-v7.0-pico-005` — 269h frozen at 36,650 with healthy publishing infra. Fix the dashboard "training now" headline to read trainer state from `status.json`.
- [ ] **P0 (new):** mirror Amulet PDF (`arxiv.org/pdf/2608.10423`) to `docs/research/papers/2608.10423-amulet.pdf` — Wayback confirms zero captures; it is OSS-FX's primary competitive reference. Queue GRAPE (2604.14706, single capture) behind it. Commit all local paper mirrors.
- [ ] **Deep read:** EdMCGS (arXiv:2609.08332) — extract the Markov-chain temporal-identity/update mechanics; compare against the v7 canvas update rule for sparse-evidence regimes.
- [ ] **Eval methodology (new):** View-Structured Conformal Prediction (arXiv:2609.10307) — park as a certified-coverage gate candidate for the future eval harness; differentiator no vendor ships.
- [ ] **Reference for #12:** LinearMask-GS (arXiv:2609.10095) — stable learned-mask pruning as the Gaussian-count-control candidate alongside Compact-FFGS merging; decide mechanism before freezing packed-canvas capacity.
- [ ] **Distillation robustness (new):** RouteBridge (arXiv:2609.09606) — per-ray routed bidirectional teacher selection; concept check for #4/#7 (e.g., route HAT-Tiny vs FSR4-teacher supervision per region).
- [ ] **Competitor watch:** AMD FSR 4 source release on GitHub — complete the FSR4-teacher capture (#7) **before** touching any released code (clean-room). Also watch DLSS Unlocked's next release cadence (299★, daily commits).
- [ ] **Positioning:** NVIDIA now claims DLSS 5 is deterministic/MV-anchored — audit the README's deterministic-reconstruction claims for differentiability: canvas renders *whole extrapolated frames* at arbitrary `t_query` vs DLSS 5 enhancing the engine's frame; lean on the <2ms vs "60 FPS technology" cost gap.
- [ ] **Process:** arXiv API `all:` queries degraded this run — add a second discovery feed (Hugging Face papers daily) to next run's sweep.
- [ ] **Carried:** TruncGradGS deep read (training-dynamics ablation); HyperGS deep read (rank regularizer); GameSR engine-free control experiment; FSR4-teacher capture plan refresh incl. RDNA 3 hosts; Qualcomm Adreno Neural Fusion as Pico-student shipping target + "one architecture" evidence.

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[2] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[3] https://opensupersampling.com/data.json (generated 2026-09-10, fetched 2026-09-11T00:00Z)
[4] https://opensupersampling.com/status.json (checked_at 2026-09-11T00:00:45Z)
[5] https://arxiv.org/abs/2609.08332
[6] https://arxiv.org/abs/2609.10307
[7] https://arxiv.org/abs/2609.10095
[8] https://arxiv.org/abs/2609.09721
[9] https://web.archive.org/cdx/search/cdx?url=arxiv.org/abs/2608.10423 (empty CDX; /web/ 404)
[10] https://api.github.com/repos/ShyVortex/dlss-unlocked
[11] https://www.pcgameshardware.de/FidelityFX-Super-Resolution-Software-277617/News/Optiscaler-FSR-411b-Support-fuer-RX-6000-und-Geforce-1553229/
[12] https://api.github.com/repos/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/commits
[13] https://www.moneycontrol.com/technology/nvidia-dlss-5-will-be-available-for-rtx-40-series-gpus-soon-all-the-details-article-14025800.html
[14] https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/
[15] https://www.tweaktown.com/articles/11596/nvidia-dlss-5-3d-guided-neural-rendering-in-nba-2k27-performance-analysis-and-more/index.html
[16] https://nvidianews.nvidia.com/news
[17] https://www.tomshardware.com/pc-components/gpus/amd-hints-at-officially-open-sourcing-fsr-4-upscaling-and-frame-generation-technology-in-the-wake-of-accidental-release-accidental-release-may-have-forced-the-companys-hand
[18] https://www.qualcomm.com/news/onq/2026/09/adreno-neural-fusion-ai-rendering
[19] https://www.tweaktown.com/news/113400/stellar-blades-nintendo-switch-2-port-to-use-nvidia-dlss-and-amd-fsr-3-1-frame-generation-to-hit-60-fps/index.html
[20] https://www.pcgamer.com/hardware/graphics-cards/007-first-light-joins-an-ever-expanding-list-of-2026-games-without-up-to-date-amd-fsr-or-intel-xess-support/
