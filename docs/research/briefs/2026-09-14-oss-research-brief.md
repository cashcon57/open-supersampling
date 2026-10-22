# OSS Research Brief — 2026-09-14

> Morning run (written ~13:05 UTC Sep 14; follows the evening brief written ~00:15 UTC). Headlines: **Modders halve FSR 4 cost on RDNA 2** — portable FidelityFX DLL (v4.0.0-rc9) cuts FSR 4.1.1 upscaling time on the PS5-derived BC-250 by 45–53% at 1080p/1440p/4K (Tom's Hardware, Sep 14), closing the RDNA 2 FSR 4 story and adding a clean #7 teacher-capture path. **SuperSplat Editor 3.0** (PlayCanvas, ~Sep 11) proves a full GPU-compute Gaussian pipeline (projector + radix sort + indirect draw, WebGPU) at multi-million-splat scale in-browser with ~15× memory reduction — a ready-made reference architecture for the cross-vendor compute rasterizer. **VS-Splat** (arXiv:2609.12343, announced Mon 14) — watch-tier voxel-selective feed-forward GS; allocate primitives only where an object lives, a spawner-allocation analog. **No new arXiv papers past the materiality bar** (arXiv Atom API 429 for a 4th consecutive run; listing fallback covered the full Mon-14 delta, 90/90 cs.CV + 4/4 cs.GR). Trainer still dead: **353h frozen at step 36,650** (eleventh consecutive dead-trainer brief). Wayback availability API IP-throttled this run — archival re-checks deferred.

## Project Status Summary

- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` — unchanged since 2026-06-08 (98 days). Uncommitted tree unchanged in kind; this run adds the morning brief only.
- **Open issues:** 12 open (#1–#14 minus gaps), unchanged set, all last touched 2026-05-08.
- **Training dashboard:** `status.json` (checked 2026-09-14T12:00Z): trainer **offline, stuck at step 36650 for 353h** — 341h → 353h, consistent ~12h drift; counter honest, run dead. Watcher healthy (published 1s ago), CF Worker healthy (89 ms), R2 healthy. `data.json` cross-check (generated_at 12:00:11Z): `srcnn-v7.0-pico-005` still `active: true`, still the only active run — same contradiction as every prior brief. Cost projections (RunPod rates) still displayed for 4090/A100/H100/B200.
- **Domain migration:** `opensupersampling.com` → **301 → `.org`** (re-confirmed this run: `.com` returns an empty 301 body; `.org/status.json` returns 200). `.com` → `.org` audit remains open.
- **Operational conclusion:** unchanged — publishing infra green, training dead ~14.5 days. No "training now" framing in external comms until the 3080 Ti host is recovered or the run is formally stopped.

## New Papers (arXiv)

**Sweep health:** arXiv Atom API throttled a **4th consecutive run** (HTTP 429 `Rate exceeded.` on every probe, including single-id probes after 45–120 s backoff — the throttle is IP-level, not query-shaped). Fallback stack executed: **cs.CV recent listing scanned in full for the Mon-14 delta (90/90 entries; ids 2609.12343–2609.13043 plus the tail of Fri-11), cs.GR recent scanned (42 entries; Mon-14 = 4)**. All candidates below screened against the materiality bar (real-time game upscaling / frame generation / Gaussian-canvas relevance).

### Gaussian Splatting / Temporal

- **VS-Splat** (arXiv:2609.12343, submitted 2026-09-11, announced Mon 14) — *Voxel-Selective feed-forward Gaussian Splatting for end-to-end 3D object reconstruction from sparse-views.* First end-to-end feed-forward GS framework that predicts primitives **only inside selected voxels** likely to belong to objects, using a learnable 2D-rendering-driven voxel selection with no 3D structural supervision. **Why it matters to OSS (watch-tier):** the selection mechanism is the feed-forward cousin of the v7 disocclusion spawner's allocation question — *where* to spend Gaussian budget. Their "select first, splat second" gate is a candidate prior/regularizer for spawner density allocation, and another data point that the field is converging on explicit density-allocation gates rather than uniform prediction. Not a canvas/SR donor as-is (object reconstruction from sparse views, not video upscaling). Screened in, logged watch.
- **Screened and rejected (Mon-14 delta):** NOVA-GS (2609.12682 — low-light NVS denoise lane, CVPR 3D4S best paper; capture-noise domain, not canvas/SR), MGAvatar (2609.12850 — mesh-bound head avatars), VideoTok4D (2609.12874 — 4D video *tokenizer* for generative world models; adjacent to "time inside the primitive" but a compressed latent for generation, not a rasterizable real-time canvas), 3D Point Splatting for mmWave Radar (2609.11894), Hologram via Quadratic Phase GS (2609.11434 — SIGGRAPH Asia Tech Comm; display/holography), Gaussian Light Transport (2609.11430 — offline GI, already rejected Sep 13).
- **Known set unchanged:** GSReuse (deep-read complete, mirrored), ABCD (2608.27735, tracked 08-31 — 80 Level author interview ran Sep 10, press echo only), RouteBridge (2609.09606, carried deep read), LinearMask-GS / CVT-GS / Compact-FFGS (#12 three-way intact).

### Super-Resolution / Upscaling

- **Nothing new past the bar.** Guided Super-Resolution of DEMs with Diffusion generators (2609.11886) screened and rejected: conceptually adjacent (image-domain prior transfers detail into a coarse non-image field — a loose echo of G-buffer-guided canvas SR) but offline diffusion (seconds-to-minutes per tile), not real-time. No game/real-time SR work in the Mon-14 window.

### Frame Extrapolation / Generation

- **Nothing new past the bar.** Known set unchanged: Amulet (2608.10423, only architectural threat, mirrored), X4Rate, GFFE, PatchEX, STSS. No new `frame extrapolation`/`frame generation` titles in the Mon-14 window.

### Inference Optimization

- **Nothing new past the bar.** INT8 SISR deployment paper (2604.20291, CVPRW) remains the reference; TensorRT 10.16 still current; EFQ-Softmax tracked 09-11. Screened and rejected: Input-Resolution-vs-Latency detection study (2609.12920, MASCOTS — detection, not SR; the resolution-latency tradeoff framing is already covered by H006).

## Archive.org Findings

- **Wayback availability API IP-throttled this run** — 429 on every probe across ~4 minutes of spaced retries (1 request/batch). Deferred: SuperTimeGS first-capture check, IE-SRGS repo capture, GRAPE/GSASR re-check. No archival-risk state changed; all four high-risk PDFs (GS-STVSR, Amulet, GRAPE, GSReuse) already mirrored locally with verified sha256s:
  - `2604.14706-grape.pdf` (33.7 MB, sha256 2003772c…)
  - `2604.18047-gs-stvsr.pdf` (18.7 MB, sha256 f9d9dcad…)
  - `2608.10423-amulet.pdf` (110 MB, sha256 88992e4f…)
  - `GSReuse-TA-SSR-3DGS-TVDG-2026.pdf` (51.7 MB, sha256 eaed57be…)
- **SuperTimeGS repo:** still no known Wayback capture (last check 09-13 evening); repo pushed again 2026-09-13 23:48Z (see GitHub section). Re-check when the API unthrottles.
- All four local PDF mirrors (~207 MB) remain **uncommitted** — same commit blocker as prior briefs.

## GitHub / Open Source

- **OptiScaler (optiscaler/OptiScaler) — 10,903 → 10,972★ (+69 in ~12h), push unchanged 2026-09-13 20:49Z.** Still **no new tag — v0.9.4 (2026-07-18) latest** despite two days of heavy main-branch churn. The XeSS-MFG cross-vendor story (verified 09-13 evening) continues to pull stars; when a tag lands, execute the #7 XeSS-MFG teacher-capture action item.
- **DLSS Unlocked (ShyVortex/dlss-unlocked) — 402 → 432★ (+30, accelerating from +9), pushed 2026-09-14 11:25Z (this run).** The igor'sLAB 3X–6X RTX-40 MFG unlock coverage (see News) is driving a second star wave. Repo is 10 days old and already top-of-lane; its OptiScaler-fork consolidation continues.
- **SuperTimeGS (kyo-kuroki/SuperTimeGS) — 0 → 1★, pushed 2026-09-13 23:48Z** (first push since creation). Watch stands: hobbyist 4DGS temporal-Gaussian-basis project; closest hobbyist artifact to the v7 native-time-axis idea.
- **NEW: DLSSG-Stream-Worker (HECer/DLSSG-Stream-Worker) — 2★, created 2026-09-04.** "Open-source D3D12 worker for offline NVIDIA DLSS Frame Generation" — batch/offline FG rendering outside a live game loop. **Potentially useful for #7:** an offline D3D12 FG path could generate DLSS-FG teacher signals from captured frame sequences without driving a live game — same clean-room sequencing caveat applies (capture outputs before reading any released model code). Watch-tier until validated.
- **NEW (noise-tier, name-collision watch): Lenovive/OSSS — "Open Source Super Scaler" — 0★, created 2026-08-24.** Window-level frame gen + upscaling for Windows. Zero traction; tracked only because of the name adjacency to OSS.
- **NEW (noise-tier): MobileNeurAI (kolomencnvvladislav-oss) — 0★, Sep 10.** "DLSS 5 analog for mobile (iOS/Android), 540p→1080p AI upscaling."Joins the Qualcomm Adreno Neural Fusion watch as more evidence mobile neural upscaling is a 2026 H2 theme; no substance yet.
- Consumption-side DLSS 5 tooling keeps proliferating (jyu041/dlss-rtxsr-upscaler pushed 09-14; projectsaturnstudios/neural-upstream ReShade pre-pass add-on pushed 09-13; Ishanoshada/DLSS5-VLC-UpScaler). Nothing in OSS's render-side lane.
- **FidelityFX SDK: v2.3.0 (2026-06-24) confirmed unchanged.** Mob-FGSR (112★, 2025-06-11) unchanged — #5 No-ML baseline reference stands.

## News

- **FSR 4 halved on RDNA 2-class silicon — the run's strategic item.** Tom's Hardware (Sep 14): modders cut FSR 4.1.1 render time on AMD's PS5-derived **BC-250** mining APU from **11.51 → 5.92 ms at 1440p (−49%)**, 25.72 → 12.08 ms at 4K (−53%), 7.13 → 3.93 ms at 1080p (−45%), using the portable FidelityFX DLL's **v4.0.0-rc9** shader set (1706×960 Quality input at 1440p). Two OSS implications: (a) FSR 4 is now demonstrably **software-portable across RDNA generations and open to community shader optimization** — the #7 FSR-4-teacher capture path on RDNA 2 is fully proven, and vendor hardware gating is again shown to be policy, not capability; (b) shader/kernel-level work **halving** a shipped upscaler's cost is direct evidence for the kernel-headroom assumption in the <2 ms student budget (H006).
- **DLSS MFG on RTX 40 at 3X/4X/5X/6X** (igor'sLAB, Sep 14): the unofficial RTX-40 unlock now demonstrates extreme multipliers; expected artifact/lag penalties at >3X corroborate the earlier "multiplier ceiling" observations. Reinforces the DLSS 5/MFG software-unlock narrative and the #7 teacher-capture options on older cards.
- **DLSS 5 backlash is now a named story** (Engadget, Sep 12: "Why Is The Internet So Upset About NVIDIA's DLSS 5?"). Combined with the quantified NR-cost datapoints (55→30 FPS CP2077 full PT on 5080; NBA 2K27 "50% frame hit"; BioShock Remastered 5K/51 FPS on a 5090, GameGPU Sep 14), the neural-rendering-cost narrative is mainstream. This is the cost-pressure axis the OSS <2 ms student thesis is built against — positioning material, not action.
- **Digital Foundry (Sep 11): "Will AMD Deliver Its Own Take On DLSS 5?"** — the competitive-vacuum framing persists: NVIDIA has neural rendering, AMD/Intel have not answered at that layer. OSS's vendor-neutral pitch sits exactly in that gap; worth citing in any outreach deck.
- **SuperSplat Editor 3.0** (PlayCanvas, ~Sep 11): editor rebuilt **WebGPU-only** — per-frame GPU compute projection, frustum culling, compaction, **GPU radix sort**, and indirect draw for every Gaussian; chunked GPU storage; streamed exports. Peak JS heap on a 4.4M-splat scene: **1,557 → 105 MB idle (~15×)**, 1080p video render 1,673 → 142 MB. **Relevance:** a full GPU-compute Gaussian rasterization pipeline (the exact projector→sort→indirect-draw shape OSS's cross-vendor kernels need) is now proven open-source at scale, in the most hostile environment (browser). Reference architecture for the Vulkan/Metal compute rasterizer design, and a candidate OSS-WebGPU delivery path.
- **007 First Light ships with official FSR 4 support** (GameGPU, Sep 14) — FSR 4 title count growing; more potential #7 capture targets.
- **Intel:** Panther Lake/Arc handhelds landing (MSI Claw 8 EX AI+ with Arc G3 Extreme — "Intel finally gets handheld gaming right," Moneycontrol Sep 11; Minisforum M2 Pro with Arc B390). No XeSS model/MFG changes from Intel itself; XeSS-MFG movement remains community-side (OptiScaler).
- **Apple:** A20 Pro (2nm, 7-core GPU) announced Sep 9; iPhone 18 Pro launch Sep 18. **MetalFX still silent** through the entire phone cycle — the vendor-neutral temporal-upscaler slot on Apple silicon remains empty for a fourth year.
- **NVIDIA:** CUDA Toolkit 13.4 extends to **Windows on Arm** (Sep 13) — marginal, but widens the eventual TensorRT-student deployment surface to WoA devices.
- **DirectSR:** still silent across a 35+ day window — the vendor-neutral FG API slot remains empty.

## Action Items / Follow-ups

- [ ] **P0 (carried, worsening):** recover or formally stop `srcnn-v7.0-pico-005` — 353h frozen at 36,650; eleventh consecutive dead-trainer brief. Fix `data.json` `active: true` flag and the static-page "training now" copy to read from `status.json`.
- [ ] **P0 (carried, trivial):** commit the research brief series + `BIBLIOGRAPHY.md` + 207 MB `docs/research/papers/` (LFS or release attachment). Three months of provenance unpushed.
- [ ] **New: SuperSplat 3.0 → cross-vendor rasterizer design note.** Their GPU-compute pipeline (compute projector + GPU radix sort + indirect draw, chunked GPU storage) is the closest open reference implementation to the OSS cross-vendor compute rasterizer design; mine it for the Vulkan/Metal kernel design before writing code. Also note WebGPU as a candidate zero-install OSS delivery/demo path.
- [ ] **New: BC-250 FSR-4 RC9 datapoint → two notes.** (1) #7: RDNA 2 FSR 4.1.1 teacher capture is now proven portable + tunable — schedule the clean-room capture. (2) H006/<2 ms thesis: cite the 45–53% shader-level cost reduction as evidence that kernel optimization headroom (not architecture) is the student-budget lever.
- [ ] **New: DLSSG-Stream-Worker eval.** If it actually renders DLSS FG offline from frame sequences, it is a #7 DLSS-FG teacher-signal capture path that doesn't need a live game session. Verify claims before relying on it.
- [ ] **Carried: GSReuse → canvas-warp design note** — map their four reliability gates (morphological closing mask, tile-validity mask, occlusion-tested composition, drop-inpainted-pixels-before-reuse) onto the v7 disocclusion/spawner decision; cite 2.24×/2.64 ms in #13.
- [ ] **New (watch): VS-Splat (2609.12343)** — voxel-selective primitive allocation; if code releases, check whether its selection gate transfers to spawner density allocation. No deep read unless results appear.
- [ ] **Carried: OptiScaler XeSS MFG cross-vendor note** — when a pre-release tag lands, capture an XeSS-MFG teacher signal on RTX/Radeon hardware for #7 (clean-room sequencing applies); UI-composition interpolation detail → OSS-FX compositor-correctness notes.
- [ ] **Watch (1 day out): MediaTek Dimensity 9600 reveal Sep 15** — Mali G2-Ultra NX NFRU deep-dive after; compare exact-engine-MV warping claims vs OSS-FX.
- [ ] **Watch: DLSS 5 RTX 40 official rollout "later this fall"** — re-check the <2 ms-student thesis framing when it ships.
- [ ] **Watch: SuperTimeGS** — first post-creation push (Sep 13 23:48Z); if it produces rendered results, its analytic time-basis integration is directly comparable to the v7 canvas's native time axis.
- [ ] **Carried: P-GSVC deep read → #12 layered-canvas design note; #12 mechanism three-way (LinearMask-GS vs Compact-FFGS vs CVT-GS) ablation decision before freezing packed-canvas capacity; Laplacian-FH training control in the v7 ablation set; #7 clean-room sequencing; Qualcomm Adreno Neural Fusion competitive note.**
- [ ] **Infra (carried):** `.com` → `.org` 301 audit (README, docs, capture-tool installer).
- [ ] **Infra (new):** Wayback availability API 429'd this run — retry the deferred capture checks (SuperTimeGS, IE-SRGS, GRAPE/GSASR) next run, spaced ≥60 s or via a different egress.
- [ ] **Infra note (carried, worsening):** arXiv Atom throttled a **4th consecutive run** (429 on all probes, incl. sparse single-id probes after 2-min waits). The listing-page fallback remains fully effective (Mon-14 delta covered 90/90); consider dropping Atom probes entirely and standardizing on listing-page + abs-page extraction to save the retry budget.

**BIBLIOGRAPHY.md:** no new papers past the materiality bar this run — no append required. (VS-Splat logged as watch-tier in this brief only.)

## Sources

1. `status.json` — https://opensupersampling.org/status.json (checked 2026-09-14T12:00:11Z fetch)
2. `data.json` — https://opensupersampling.org/data.json (checked 2026-09-14, generated_at 12:00:11Z)
3. GitHub Issues API — repos/cashcon57/open-supersampling/issues?state=open (checked 2026-09-14; 12 open)
4. Tom's Hardware — BC-250 FSR 4 halving: https://www.tomshardware.com/pc-components/modders-halve-fsr-4-render-times-on-amds-ps5-derived-bc-250-mining-apu-portable-fidelityfx-dll-cuts-1440p-upscaling-time-from-11-51-ms-to-5-92-ms (Sep 14)
5. PlayCanvas blog — SuperSplat Editor 3.0: https://blog.playcanvas.com/new-in-supersplat-editor-3-0-rebuilt-on-webgpu/ (~Sep 11)
6. arXiv — VS-Splat abs: https://arxiv.org/abs/2609.12343 (submitted 2026-09-11); cs.CV/recent + cs.GR/recent listings (Mon-14 delta, checked 2026-09-14)
7. GitHub REST — watched repos (OptiScaler, dlss-unlocked, SuperTimeGS, DLSS5VKLayer, NVOFG, DLSSG-Stream-Worker, OSSS, MobileNeurAI) + 3 search lanes (checked 2026-09-14)
8. Google News RSS — DLSS 5 / FSR 4 / XeSS+MFG / gaussian splatting / frame generation / MetalFX / Intel Arc lanes, when:3–5d (checked 2026-09-14); igor'sLAB MFG-on-RTX-40 (Sep 14), Engadget DLSS 5 backlash (Sep 12), Digital Foundry AMD-DLSS-5-answer (Sep 11)
9. archive.org Wayback availability API — 429 throttled (6 attempts, 2026-09-14)
