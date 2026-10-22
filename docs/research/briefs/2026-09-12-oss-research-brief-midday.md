# OSS Research Brief — 2026-09-12 (midday)

> Morning run (08:00–08:35 EDT). Headlines: **Mob-FGSR surfaced as the closest shipped analog of OSS's thesis** — SIGGRAPH 2024 (ZJU/OPPO), joint frame-gen + SR from color/depth/MVs, no runtime NNs, 2.2–2.3 ms on Snapdragon 8 Gen 3, shipping since 2024 in OnePlus phones; it is a must-cite precedent and the natural ablation baseline for the No-ML shader variant (#5), not a threat (fixed α-grid splatting, no persistent canvas). Also newly material: **D2GV-AR** (2D-Gaussian video canvas cousin, arbitrary-scale, 250+ FPS) and **GS-NFS** (full-frame-rate GPU codec for dynamic Gaussian streams). **Infra: opensupersampling.com now 301-redirects to opensupersampling.org.** Trainer still dead: **305h frozen at 36,650** (seventh consecutive dead-trainer brief). arXiv API still flapping (2/10 lanes answered); GS/temporal + frame-extrap + TRT lanes covered by fallback discovery only.

## Project Status Summary

- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` — unchanged since 2026-06-08 (96 days). Uncommitted tree unchanged at ~35 paths (brief series, `BIBLIOGRAPHY.md`, 155MB `docs/research/papers/` mirrors).
- **Open issues:** 12, unchanged (#1–#14 minus gaps; all last touched 2026-05-08).
- **Training dashboard:** `status.json` (checked 2026-09-12T12:02Z): trainer **offline, stuck at step 36,650 for 305h** — 293 → 305h since last night, consistent drift; watcher / CF Worker / R2 healthy. `data.json` cross-check: `srcnn-v7.0-pico-005` still flagged `active: true`, `latest_step` 36,650, identical loss values (total 0.1927, sr_lpips 0.0927) — run is dead but the dashboard's `active` flag and the static-page "v7.0-pico-005 OSS-FX (training now)" copy still claim otherwise. Same contradiction as prior briefs, now observed from a new origin.
- **NEW — domain migration:** `https://opensupersampling.com/*` now returns **301 → `opensupersampling.org`** (Cloudflare; observed 12:02Z). All prior briefs/README reference `.com` URLs; external links and the capture-tool installer should be audited for the new origin. CF infra itself healthy (86 ms worker, R2 200).
- **Operational conclusion:** unchanged — publishing infra green (and re-homed), training dead 12+ days. No "training now" framing in external comms.

## New Papers (arXiv)

**Sweep health:** arXiv Atom API flapping — after the overnight outage it answered two probes, then 8/10 fielded queries returned 429/timeout (HTTPS, 3.2 s spacing, 20–60 s timeouts). Per fail-fast, the API was abandoned after the second full sweep; two lanes (`sr_realtime`, `frame_interp`) + one `id_list` attempt completed (12 candidates, 0 new after the materiality bar). Paper metadata below is from **arXiv abs-page extraction** (primary) and the Mob-FGSR project page; GS/temporal, frame-extrapolation, and TRT/int8 lanes were covered by web-search discovery only — recall for this window is again lower than a normal run.

### Gaussian Splatting / Temporal

- **D2GV-AR: Progressively Deformable 2D Gaussian Splatting for Video Representation at Arbitrary Resolutions** (Liu, Yang, Zhao, Huang, Yang, Li, Xu — arXiv:2503.05600, v2 2026-01-29, cs.CV). Canonical per-GOP 2D Gaussian set with **neural-ODE temporal evolution**, Nyquist-based scale-aware grouping for arbitrary-scale rendering, and D-optimal pruning for any-ratio progressive coding; 250+ FPS decode. Why it matters to OSS: the nearest published cousin of the v7 canvas from the video-codec side — independently validates persistent 2D Gaussian primitives with learned temporal dynamics as a video representation, and its arbitrary-resolution + progressive-coding machinery is a direct ablation donor for #12 (packed canvas, progressive capture-upload). Newly material (v2 landed Jan 2026), not newly published.
- **GS-NFS: Bandwidth-adaptive Streaming of Dynamic Gaussian Splats and Point Clouds** (Ghosh et al., USC/ISI — arXiv:2606.05650, 2026-06-04, cs.MM). GPU-parallel encode/decode of dynamic Gaussian frames at full frame rate, 1–2 orders of magnitude faster than prior dynamic-3DGS codecs at competitive compression. Why it matters to OSS: #12's serialization path has a new performance reference — if canvas state ever crosses a bus (capture upload, cross-process handoff), GS-NFS's GPU-side position/attribute parallelization is the pattern to mine before writing custom code.
- **Re-checked and rejected:** DLGStream (2606.28840 — language-embedded free-viewpoint streaming, out of scope), TD-GS/AVSplat-class sparse-view noise, event-camera VFI (2608.10479, LongE2V 2607.08770 — event-domain), test-time-adaptive VSR thesis (2608.08508 — doctoral abstract, no real-time path), 360° telepresence SR (2608.05570).

### Super-Resolution / Upscaling

- **Mob-FGSR: Frame Generation and Super Resolution for Mobile Real-Time Rendering** (Yang, Zhu, Zhuge, Qiu, Li, Yan, Yan, Yan, Jin — ZJU + OPPO + UCSB, **SIGGRAPH 2024**; no arXiv, project page + PDF + code: mob-fgsr.github.io, github.com/Mob-FGSR/MobFGSR). **Newly surfaced as material, not new.** Joint interpolation **and** extrapolation + SR from rendered color/depth/MVs only; motion is reconstructed by depth-aware **MV splatting** with atomic ops + gap filling; SR is a temporal-accumulation warp with **learned LUT resampling weights**; zero neural networks at runtime; 2.2 ms (720p FG) / 2.3 ms (1080p FG+SR) on Snapdragon 8 Gen 3; **shipped** in OnePlus Ace 3 Pro (2024-06, Genshin @120 fps) and OnePlus 13. Why it matters to OSS: (1) it is the strongest shipped validation of "one lightweight pipeline does FG + SR from engine MVs" — the exact OSS thesis — two years before the Arm/Qualcomm silicon wave, so the README positioning should cite it as precedent rather than novelty; (2) MV-splatting is a degenerate per-pixel special case of OSS's Gaussian canvas (no covariance, no persistence, no quality-per-primitive gain) — ideal ablation baseline and the natural reference point for No-ML shader variant #5; (3) its disocclusion handling (depth/brightness discrimination, simplified filling for extrapolation) is a directly comparable heuristic baseline for OSS-FX's canvas-based hole filling. Not a threat: fixed α-grid, no continuous `t_query`, no learned quality head.
- **Evaluation evidence, newly material:** **A new dataset and comparison for multi-camera frame synthesis** (Daly & Kokaram, Trinity Dublin — arXiv:2508.09068, SPIE D&D XLVIII 2025). Controlled array-camera comparison: 3DGS beats deep + classical frame interpolators by **+5 dB PSNR on synthetic scenes** (95% conf.) but **loses by up to 3.5 dB on real footage**. Why it matters to OSS: direct external evidence for the domain bet — Gaussian canvases win in synthetic/geometry-rich rendering (exactly the game-render setting with engine MVs) and degrade on noisy real captures; cite in the eval plan as the boundary-condition citation for why OSS trains on game-capture, not phone video.
- Re-checked and rejected: TRaM-VSR (2607.22231 — one-step DiT VSR, offline quality class), AVSR-Diff (2607.00987 — diffusion arbitrary-scale VSR), DTI (2606.29198 — face VSR), EvTexture++ (2606.13580 — event-camera VSR), multi-hypothesis flow VFI (2608.07120 — same control class as tracked MGMVFI, no new mechanism), CoInS-Net (2608.09391 — medical), Vertigo Vertigo (2606.29495-class art project), 2312.10890 low-latency space-time supersampling (already tracked lineage), KernelFusion ICLR 2026 blind-SR (blind-kernel estimation, not real-time game SR).

### Frame Extrapolation / Generation

- **Nothing new past the bar.** X4Rate (VISAPP 2026 routing baseline) and Amulet (2608.10423, still the only architectural threat) remain the references; GFFE/ExWarp lineage unchanged. Mob-FGSR's extrapolation model (above) is the newest *shipped* extrapolator this scan surfaced — it belongs in this lane too: single-frame backward-warp extrapolation with simplified disocclusion filling, i.e., the no-persistence control OSS-FX must beat.

### Inference Optimization

- **Nothing new past the bar.** TensorRT 10.16 (Mar 2026) remains the current release; no new INT8-SR or kernel-fusion-for-SR papers this window. Splats-to-Silicon (2609.06157) and TileGS (2609.03613) remain the standing #12/#13 methodology references. Mob-FGSR's LUT-resampling ("learned resampling weights for repeated image warping") is a tangential donor: if the Pico student's final upsample stays non-neural on some paths, LUT resampling is the shipped mobile precedent.

## Archive.org Findings

- **Wayback CDX re-verification errored this run** (timeouts on all four tracked URLs: GS-STVSR 2604.18047, Amulet 2608.10423, GRAPE 2604.14706, GSASR 2501.06838) — attempted, inconclusive, no new captures either way. Prior finding stands unchallenged: GS-STVSR and Amulet remain zero-capture (protected by local mirrors), GRAPE single-capture (digest-matched), GSASR well-captured.
- archive.org full-text search: not re-run this pass (CDX flakiness suggests API load); no new leads from web-search discovery.
- All three local PDF mirrors remain **uncommitted** (`docs/research/papers/`, 155MB) — same commit blocker as prior briefs.

## GitHub / Open Source

- **DLSS Unlocked (ShyVortex/dlss-unlocked) — 354 → 370★ (+16 in ~9h), repo push unchanged 2026-09-11 02:52Z.** Plateau continues decaying (36 → 19 → 16/day). Multi-outlet coverage confirms it as the week's story (PCMag, TechPowerUp, Wccftech all ran DLSS-5-MFG-on-old-RTX pieces Sep 7–11).
- **NEW: bmitch87/DLSS5VKLayer — 69★, created 2026-09-05, pushed 2026-09-12 (today).** Linux-"native" DLSS 5 via a Vulkan layer (Phoronix covered it Sep 8: wires DLSS 5 for native Linux games and Steam Play). First Linux-native entrant in the injection-layer slot OSS targets; a week old and growing. Worth a README-visible follow: if DLSS 5 NR becomes trivially available on Linux via a VKLayer, the "OSS is the only vendor-neutral cross-vendor layer" framing needs the Linux column of the matrix checked.
- **OptiScaler (optiscaler/OptiScaler) — 10,664 → 10,694★, pushed 2026-09-11 23:06Z.** Steady.
- Repo searches (GS+SR, frame-gen+upscaling, neural upscaler; pushed > Sep 5): nothing above noise — a 0★ course project (EE5178) and 0★ `mohmedmm/nbox` (240p→4K enhancer, no releases). Unquoted queries used; exact-phrase form returns zero.
- AMD FidelityFX SDK: unchanged (v2.3.0, 2026-06-24). FSR 4.1.1-on-RDNA-3 driver rollout remains the active story (carried).

## News

- **DLSS 5: two new deltas beyond last night's brief.** (1) **MFG modded onto RTX 30 Ampere** — TechPowerUp (Sep 9) + Wccftech (Sep 11), corroborating DLSS Unlocked's reach beyond 20/40-series; RSS-corroborated only (primary page 403'd; the mod repo itself is the verified artifact). (2) **DLSS 5 neural rendering applied to the entire Windows desktop via a driver-level path** — TweakTown headline (Sep 8, RSS-only; article URL unresolved). If confirmed, NVIDIA is pushing NR beyond games toward the OS compositor — strengthens the "neural upscaling becomes platform infrastructure" frame the README argues OSS should be part of. (3) NVIDIA 615.71.09 Linux driver ships Vulkan/Reflex improvements (Phoronix, Sep 9) — routine.
- **Arm Mali G2-Ultra NX:** no new facts since the Sep 8 announce (newsroom post unchanged); MediaTek Dimensity 9600 reveal expected at the Sep 15 Arm event — **NFRU deep-dive watch item stands**.
- **Qualcomm Adreno Neural Fusion:** coverage steady (Sep 2–4 wave + ServeTheHome follow-ups Sep 10–11); no new specs beyond matrix cores + neural fusion. Still: both major mobile IP vendors ship SR+FG as one GPU pipeline — thesis validation, differentiation bar raised (carried).
- **Intel:** nothing beyond Panther Lake B390 benchmark chatter and an MSI Claw 8 review; XeSS MFG unchanged. **Apple:** MetalFX silent (iPhone 18 preorders open today — no MetalFX news attached). **DirectSR:** still silent — the vendor-neutral FG API slot remains empty.

## Action Items / Follow-ups

- [ ] **P0 (carried, worsening):** recover or formally stop `srcnn-v7.0-pico-005` — 305h frozen at 36,650; seventh consecutive dead-trainer brief. `data.json` still flags the run `active: true` and the static page still says "(training now)" — fix both to read from `status.json`.
- [ ] **New (infra):** `.com` → `.org` 301 — audit README, docs, and the capture-tool installer for `.com` references; confirm whether `.com` canonical or `.org` canonical is intended, and update the research-brief source list going forward.
- [ ] **P0 (carried, trivial):** commit the research brief series + `BIBLIOGRAPHY.md` + 155MB `docs/research/papers/` (LFS or release attachment if over repo policy). Three months of provenance unpushed.
- [ ] **New: Mob-FGSR integration pass** — read the SIGGRAPH 2024 PDF + `Mob-FGSR/MobFGSR` code; extract (a) MV-splatting gap-filling heuristics as the #5 No-ML ablation baseline, (b) LUT resampling weights idea as a non-neural resample option, (c) their disocclusion discrimination (depth/brightness) as the heuristic baseline the canvas must beat in the OSS-FX eval. Add to positioning: cite as shipped precedent for engine-MV-driven joint FG+SR (2024), with OSS's deltas = continuous `t_query`, learned quality, vendor-neutral.
- [ ] **New: D2GV-AR deep read** — neural-ODE per-primitive temporal evolution vs v7's native time axis; Nyquist scale-grouping for arbitrary-scale rendering; D-optimal progressive coding vs #12's packed canvas. Decide: ablation donor or adjacent-alternative to cite.
- [ ] **New: GS-NFS → #12 reference** — GPU-parallel encode/decode pattern for dynamic Gaussian streams; evaluate against TSOG/QuARC-GS/STGV entries already tracked for the packed-canvas slot.
- [ ] **New: eval-plan citation** — 2508.09068 (multi-camera frame synthesis): +5 dB synthetic / −3.5 dB real result for GS-vs-VFI; use as the domain-boundary citation for game-render training data.
- [ ] **New: watch DLSS5VKLayer (bmitch87/DLSS5VKLayer)** — first Linux-native DLSS 5 injection layer; check its vendor-gating approach and whether the OSS layer matrix needs a Linux column update.
- [ ] **Carried: Arm NFRU deep-dive after Sep 15** (Arm newsroom + Neural Dawn details vs OSS-FX exact-engine-MV warping; INT8/INT16 in-shader accel vs Pico student hardware assumptions).
- [ ] **Carried: #7 clean-room sequencing** — capture FSR 4.1.1 teacher signal before reading any released model code (FSR 4.1.1 trivially runnable on RDNA 3 via Adrenalin 26.6.2).
- [ ] **Carried deep reads:** EdMCGS (2609.08332) temporal-identity mechanics; VSCP-GS (2609.10307) certified-coverage gate; LinearMask-GS (2609.10095) pruning-vs-merging; RouteBridge (2609.09606) routed teacher selection; X4Rate VISAPP PDF acquisition (SciTePress 2026/142928).
- [ ] **Infra note (carried):** arXiv Atom remains unstable (2/10 lanes; HTTPS required — plain HTTP now 301s and may be the flake source). Next run: HTTPS-only, ≤3 broad queries first, then id_list; keep abs-page extraction as the standard fallback.

## Sources

1. `status.json` — https://opensupersampling.org/status.json (checked 2026-09-12T12:02Z; via .com 301)
2. `data.json` — https://opensupersampling.org/data.json (v7 run `latest_step` 36,650, `active: true`, loss total 0.1927)
3. arXiv Atom API — export.arxiv.org (partially degraded: 8/10 fielded queries 429/timeout on HTTPS; 2 lanes + probes answered; abandoned per fail-fast)
4. arXiv abs pages — 2503.05600 (D2GV-AR), 2606.05650 (GS-NFS), 2508.09068 (multi-camera frame synthesis)
5. Mob-FGSR project page — https://mob-fgsr.github.io/ (SIGGRAPH 2024; PDF/supp/code/APK links verified present)
6. Hugging Face daily papers — 2026-09-11+12 (no GS/SR/frame-gen entries)
7. GitHub API — ShyVortex/dlss-unlocked (370★), optiscaler/OptiScaler (10,694★), bmitch87/DLSS5VKLayer (69★, created 2026-09-05), repo searches pushed > Sep 5
8. Wayback CDX — 4 tracked paper URLs (timed out this run; prior state stands)
9. News (Google News RSS, when:7–10d): NVIDIA blog/PCWorld/PC Gamer/PCMag/Wccftech (DLSS 5), TweakTown (desktop-wide NR — RSS-only), TechPowerUp/Wccftech (MFG on Ampere — RSS-only), Phoronix (DLSS5VKLayer, driver 615.71.09), newsroom.arm.com/unite.ai/videoCardz (Mali G2-Ultra NX, Sep 8), thelec/AndroidCentral/ServeTheHome (Adreno Neural Fusion)
