# OSS Research Brief — 2026-09-16

> Morning run (written ~12:30 UTC Sep 16 / 08:30 EDT; first brief of the day, follows the Sep-15 evening brief). Headlines: **arXiv Atom API has recovered** — after 429s all last week and Monday's SSL `CERTIFICATE_VERIFY_FAILED`, a direct Atom query succeeded this run (probe + a 7-ID batch fetch, clean), restoring primary metadata as the default path. **DecoGS (2609.17230, Wed-16 batch) is the first paper past the materiality bar since GameSR/TFE (Sep-15 midday)**: streaming Gaussian training that selectively updates only spatiotemporal regions exhibiting motion/photometric change — the training-side counterpart of the GSReuse/TFE/VS-Splat runtime tile-gating family, with a **70× temporal-flicker reduction** as headline evidence for the temporal-stability thesis. **Archive.org entered a new failure mode — the entire site serves "Temporarily Offline"** (5th consecutive run without archival progress, but now an outage, not a throttle). Trainer still dead: **401h frozen at step 36,650** (fifteenth consecutive dead-trainer brief).

## Project Status Summary

- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` — unchanged since 2026-06-08 (100 days). Uncommitted tree unchanged in kind (brief series + `BIBLIOGRAPHY.md` + `papers/` still unpushed); this run adds today's brief and one bibliography entry.
- **Open issues:** 12 open (#1–#14 minus gaps), unchanged set, all last touched 2026-05-08.
- **Training dashboard:** `status.json` (checked_at 2026-09-16T12:07:53Z): trainer **offline, "stuck at step 36650 for 401h"** — 389h → 401h, consistent ~12h drift; run dead ~16.7 days. Watcher healthy (published 1s ago), CF Worker healthy (97 ms), R2 healthy. `data.json` cross-check (generated_at 2026-09-16T12:07:53Z): `srcnn-v7.0-pico-005` still `active: true` at step 36,650 of 100,000 — **still the only active run of 8, same contradiction as every prior brief**; `history.status` still reads "active, training now (v7)" and the headline block still "training now / v7 OSS-FX inflight". Freeze-point metrics unchanged: `sr_charbonnier` 0.02360, `fg_charbonnier` 0.07643, `canvas_count` 2304, `canvas_mean_opacity` 0.4185. Cost projections (RunPod rates) still displayed for 4090/A100/H100/B200.
- **Domain migration:** `opensupersampling.com` → **301 → `.org`** (re-confirmed this run; `.org/status.json` returns 200). `.com` → `.org` audit remains open.
- **Operational conclusion:** unchanged — publishing infra green, training dead ~16.7 days. No "training now" framing in external comms until the 3080 Ti host is recovered or the run is formally stopped.

## New Papers (arXiv)

**Sweep health:** Atom API **recovered** this run (probe OK; then a clean 7-ID batch fetch). Listing fallback executed as standardized: cs.CV recent scanned in full — **641/641 entries across all five day-groups (Wed-16 123/123, Tue-15 233/233, Mon-14 90/90, Fri-11 89/89, Thu-10 106/106)**, cs.GR **46/46 (Wed-16 9/9)**. The Tue-15-and-earlier batches were already screened by prior briefs; **today's fresh delta is the Wed-16 00:00 UTC batch (123 cs.CV + 9 cs.GR), which no prior brief has seen.**

### Gaussian Splatting / Temporal

- **[DecoGS]** Sulo, Supikov, Demir, and Liu (2026). *DecoGS: Adaptive Static-Dynamic Decoupling of 3D Gaussians for Free-Viewpoint Video Streaming*. arXiv:2609.17230 (cs.CV, Wed-16 batch). **Past the bar — appended to BIBLIOGRAPHY.md.**
  > Online training of 3D Gaussians from streaming video that selectively updates only spatiotemporal regions exhibiting motion or photometric change, with gradient gating and visibility filtering; on N3DV/MeetRoom it reaches 34.55/31.60 dB PSNR, renders at 261 FPS, reports **70× lower temporal flicker** than the best prior streaming method, and needs no large-scale pretraining. **Why it matters to OSS:** the training-time analog of the runtime spend-compute-only-where-things-move tile-gating family (GSReuse, TFE, VS-Splat) — and the first of that family with a *quantified temporal-flicker* number, the exact metric dimension (flicker/drift in nominally static regions) the v7 canvas must defend when engine-MVs are bad or spawns are spurious. Direct donor for the spawner's activity-gate design note and a candidate control for #12's bounded-canvas update cost; "no large-scale pretraining" also supports the small-data teacher tier.

- **Screened and rejected (Wed-16):** PhysStream (2609.17521, cs.CV+GR) — autoregressive physics-grounded image-to-video generation with structured scene memory; generative-video lane, not the game rendering loop. Bi-FlowGS (2609.17039) — sparse-view NVS with video-diffusion view completion + flow co-refinement ("geometry cheating" framing is nice, but it's an offline reconstruction lane). PanoGS-SLAM (2609.17387) — panoramic SLAM, robotics. tcnerv (2609.16870) — implicit neural video compression; NeRV-lane codec, no splat primitive (watch-tier only if the canvas needs an INR compression baseline). 3D Field Data Reduction (2609.16024, cs.GR+CV+CE) — fixed-budget Gaussian encoding of scientific field data; interesting storage framing but simulation-data lane, not screen-space rendering. Delayed-Light Rendering (2609.16180, cs.GR) — real-time superluminal-object rendering via recorded state history; fun rendering-tech adjacent, no upscaler/canvas mechanics.

### Super-Resolution / Upscaling

- **Nothing new past the bar.** Wed-16 batch contains no SR-title entries; known set unchanged (GameSR mirrored, INT8 SISR 2604.20291, GRAPE/GSASR lineage).

### Frame Extrapolation / Generation

- **Nothing new past the bar.** Known set unchanged: TFE (SIGGRAPH Asia 2026, supplement mirrored), Amulet (2608.10423, only architectural threat), X4Rate, GFFE, PatchEX, STSS. The Wed-16 batch produced no frame-generation/extrapolation entries.

### Inference Optimization

- **Nothing new past the bar.** TensorRT 10.16 current; EFQ-Softmax (09-11) and INT8 SISR (2604.20291) remain the tracked references. Wed-16 quantization entries are edge-VLM/SSM-lane, not real-time raster-adjacent.

## Archive.org Findings

- **New failure mode: archive.org is fully offline, not throttling us.** All probes (CDX for `kyo-kuroki/SuperTimeGS` and `ZheXues/IE-SRGS`) returned the site-wide **"Internet Archive: Temporarily Offline"** maintenance page — 5th consecutive run without archival progress, but the cause changed from 429-throttling to a full outage. No Save-Page-Now possible this run either.
- **Consequence:** the `kyo-kuroki/SuperTimeGS` snapshot action (zero Wayback captures on the old `JianqiangWu/SuperTimeGS` URL, owner already moved once) remains **unexecuted and the risk is unchanged**. Local mitigation available now: `git clone https://github.com/kyo-kuroki/SuperTimeGS` into `docs/research/papers/` does not depend on archive.org at all — recommend doing that before the next run (see action items).
- GRAPE/GSASR/IE-SRGS re-checks remain deferred; both repos confirmed live via the GitHub API this run (`ZheXues/IE-SRGS` 2★, pushed 2026-08-29; `kyo-kuroki/SuperTimeGS` 1★, pushed 2026-09-13 — both unchanged since the evening brief).

## GitHub / Open Source

- **DLSS5-Autopilot (Kizzuwatnaa/DLSS5-Autopilot): 634 → 642★ (+8), pushed 2026-09-16 11:48Z** — still the fastest-moving OSS-adjacent artifact; consolidation commits continued overnight. Desktop-wide deployment press coverage continues (MakeUseOf/TechSpot "DLSS 5 on every window of your desktop," Sep 9–10).
- **Pre-upscale NR idea is forking into standalone ReShade add-ons (new this window):** `Xeakaes/dlss5-nr-pre-upscale` (1★, created/pushed 2026-09-15) and `projectsaturnstudios/neural-upstream` (0★, created 09-10, pushed 09-13) both run DLSS 5 Neural Rendering **at render resolution before the upscaler** — the Neural-Upstream ordering thesis (standing watch item) is now being productized by third parties independently of matiasLombo's original (105★, +1, no commits since 09-02). Reinforces the H006 resolution-ordering note: watch the official RTX 40 rollout for adoption of this ordering.
- **OptiScaler (optiscaler/OptiScaler): 11,055 → 11,078★ (+23), pushed 2026-09-15 13:51Z; tags re-verified earlier this series — still `v0.9.4` latest.** The #7 XeSS-MFG teacher-capture item stays gated on a new tag; Intel's discrete-B-series MFG driver rollout (reported for "next month") remains the likely first trigger.
- **dlss-unlocked (ShyVortex/dlss-unlocked): 482 → 496★ (+14), pushed 2026-09-15 22:45Z.** Consumption-side injector lane, still star-accelerating (+27 in two days).
- **nvpro-samples/vk_gaussian_splatting: 501★; `pushed_at` moved to 2026-09-16 08:13Z but latest *code* commits are 2026-09-11 (FTB pipeline fixes: storage-image visibility, render-pass attachment matching)** — this morning's push was metadata-only. Release 2026.2.9 (Sep 3) remains current; cross-vendor rasterizer reference unchanged.
- **NJUCG/tile-based-frame-extrapolation: 1★ unchanged; HECer/DLSSG-Stream-Worker: 2★ unchanged; MLX-DLSS: 73★, pushed 09-10 (unchanged); SuperTimeGS/IE-SRGS unchanged (see Archive.org).**
- **No new GS+SR or FG repos past the bar in GitHub search this run.** Search produced only low-star derivative tooling in the DLSS5 modding lane (launchers, injectors, a VLC upscaler) — the modding stack remains the fastest-moving OSS-adjacent space but nothing new is architecturally relevant.

## News

- **NVIDIA newsroom RSS + GPUOpen RSS: no new SR/FG items this window.** GPUOpen's latest is the one-step-diffusion generative illumination post (Sep 9, already known); no new FSR SDK/UE plugin movement since the Aug 27 UE 5.8 item. NVIDIA newsroom this window is AI-infra/enterprise lane (Dreamforce, Earth-2, CUDA-Q) — nothing upscaler-relevant.
- **NVFP4 DLSS 5 mod datapoint (new, note-tier):** modders got DLSS 5 running on the **NVFP4 format — performance improved by just 2% on RTX 50** (TweakTown, Sep 10). Direct H006 ammunition: even a precision-halving of the (already FP8-class) network moves frame time ~2% — the bottleneck is elsewhere (memory/composition), which supports the <2 ms student thesis and further deflates "the model is the cost" claims.
- **DLSS 5 discourse wave (dedup: covered by prior briefs):** PCWorld's positive hands-on ("I was a DLSS 5 hater…until I used it," Sep 11), Engadget's backlash roundup (Sep 12), Digital Foundry asking "will AMD deliver its own take on DLSS 5?" (Sep 11), Vávra defending DLSS 5 for KCD2 (Sep 14), Tom's Hardware testing unofficial MFG on RTX 40 (Sep 12). No new facts beyond prior coverage; sentiment signal: the neural-rendering cost/quality debate is now mainstream discourse — free positioning material for a vendor-neutral, quality-per-millisecond framing.
- **Intel:** driver 32.0.101.8991 WHQL (XeSS MFG crash fixes) already covered by the evening brief; MSI Claw 8 EX AI+ (Arc G3 Extreme) reviews (Sep 11) are consistent with the B-series/handheld MFG rollout watch — discrete-B MFG "next month" trigger unchanged.
- **Apple:** iPhone event coverage (foldable, "40% GPU jump," Monster Hunter) already covered by the Sep-13 brief; nothing new in the MetalFX lane this window.
- **AMD:** no primary FSR movement this window (BC-250 halving and RX 6000 FSR 4.1.1b already covered Sep-14/15); the FSR 4 open-sourcing watch item stays open.

## Action Items / Follow-ups

- [ ] **P0 (carried, worsening):** recover or formally stop `srcnn-v7.0-pico-005` — **401h** frozen at 36,650; fifteenth consecutive dead-trainer brief. Fix `data.json` `active: true` flag and the `history` "training now" copy to read from `status.json`.
- [ ] **P0 (carried):** commit the research brief series + `BIBLIOGRAPHY.md` + `docs/research/papers/` (~220 MB / 7+ PDFs; LFS or release attachment). Three months of provenance unpushed.
- [ ] **New (archival, quick, no archive.org dependency):** `git clone https://github.com/kyo-kuroki/SuperTimeGS` into a local mirror now — archive.org is fully offline this run, the repo has zero Wayback captures on its old URL, and a second owner move loses the analytic-time-basis comparison target for the v7 canvas. Wayback Save-Page-Now for `kyo-kuroki/SuperTimeGS` + `ZheXues/IE-SRGS` once archive.org recovers.
- [ ] **New (#12/spawner donor):** DecoGS (2609.17230) logged in BIBLIOGRAPHY.md — fold its gradient-gated selective update + temporal-flicker framing into the spawner activity-gate design note alongside GSReuse/TFE/VS-Splat (it is the *training-time* member of that family). No mirror needed (stable arXiv host).
- [ ] **New (H006 note):** add the NVFP4 +2% datapoint (TweakTown, Sep 10) to the kernel-headroom evidence list — precision-halving the network moves frame time ~2%, so the <2 ms budget is won in memory/kernel work, not model precision.
- [ ] **Carried (positioning):** retire the "MetalFX silent for four years" line — replace with the all-five-vendors in-pipeline-neural framing; open cross-vendor slot as the differentiator. Update README/positioning materials when next touched.
- [ ] **Carried (watch):** third-party pre-upscale-NR ReShade add-ons (`Xeakaes/dlss5-nr-pre-upscale`, `projectsaturnstudios/neural-upstream`) productizing the Neural-Upstream ordering — check the official RTX 40 rollout for adoption when it ships (fall 2026 window).
- [ ] **Carried:** GameSR → student-tier design review input (#4/#9) — ConvLSTM ablation (−5 dB PSNR), reparameterization vs #9, PixelUnshuffle cost profile. Deep-read queued.
- [ ] **Carried:** TFE (SIGGRAPH Asia 2026) → OSS-FX comparison row + spawner tile-gate design note. Clean-room sequencing: paper/site first.
- [ ] **Carried:** XeSS-MFG teacher capture (#7) — two triggers: OptiScaler pre-release tag (gated) or Intel discrete-B-series MFG driver rollout (likely first). Clean-room sequencing applies.
- [ ] **Carried:** FSR 4 open-sourcing watch (open *runtime*, closed *core* caveat if it lands); MLX-DLSS as alternative #7 teacher path; GSReuse → canvas-warp design note; P-GSVC deep read → #12 layered canvas; Laplacian-FH training control; VS-Splat watch; SuperSplat 3.0 + vkSplatting 2026.1 → cross-vendor rasterizer design note.
- [ ] **Infra (carried):** `.com` → `.org` 301 audit (README, docs, capture-tool installer).
- [ ] **Infra (changed):** archive.org **fully offline** this run (not a 429 throttle) — 5th consecutive run without archival progress. When it recovers: single spaced CDX probe for the two repo URLs, then Save-Page-Now POSTs. arXiv Atom API **recovered** this run (SSL failure gone) — resume Atom as the default primary-metadata path, keep the listing fallback standardized.

**BIBLIOGRAPHY.md:** one append this run — **[DecoGS]** (arXiv:2609.17230), new section `## Research brief 2026-09-16`. PhysStream/Bi-FlowGS/others screened and rejected; no other appends.
