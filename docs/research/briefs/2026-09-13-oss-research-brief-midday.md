# OSS Research Brief — 2026-09-13 (midday)

> Midday run (written ~12:10 UTC Sep 13; scan window Sep 11–13, following the ~00:15 UTC brief same day). Headlines: **GSReuse (IEEE TVCG 33(5), Nanjing University)** — newly surfaced, screen-space temporal reuse for 3DGS rendering via estimated per-pixel Gaussian-splatting motion vectors + warping; the pure-performance precedent for OSS's canvas-warp thesis and a #13 methodology reference. **Qualcomm Adreno Neural Fusion** (Sep 2–3) — unified "AI super resolution + frame generation in one graphics pipeline" on mobile Matrix Cores, Unity/Unreal onboard: the mobile vendor-neutral slot is now contested by silicon, and their pitch is rhetorically the OSS thesis. **NVIDIA officially confirms DLSS 5 for RTX 40 "later this fall."** Trainer still dead: **329h frozen at step 36,650** (ninth consecutive dead-trainer brief). arXiv Atom API throttled again (429 `Rate exceeded.` on probe); coverage via listing fallback (776 cs.CV + 46 cs.GR entries scanned) + HF daily-papers API + abs extraction.

## Project Status Summary

- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` — unchanged since 2026-06-08 (97 days). Uncommitted tree unchanged (brief series, `BIBLIOGRAPHY.md` +228 lines, 155MB `docs/research/papers/` mirrors).
- **Open issues:** 10 open (#1–#14 minus gaps), unchanged set, all last touched 2026-05-08. Tracker static.
- **Training dashboard:** `status.json` (checked 2026-09-13T12:01Z): trainer **offline, stuck at step 36650 for 329h** — 317h → 329h, consistent ~12h drift; counter honest, run dead. Watcher healthy (published 1s ago), CF Worker healthy (88 ms), R2 healthy. `data.json` cross-check: `srcnn-v7.0-pico-005` still `active: true`, `latest_step` 36,650, headline copy still "training now" — same contradiction as prior briefs. Cost projections (RunPod rates) still displayed for 4090/A100/H100/B200.
- **Domain migration:** `opensupersampling.com/status.json` → **301 → `.org`** (verified again this run). `.com` → `.org` audit remains open.
- **Operational conclusion:** unchanged — publishing infra green, training dead ~14 days. No "training now" framing in external comms until the 3080 Ti host is recovered or the run is formally stopped.

## New Papers (arXiv)

**Sweep health:** arXiv Atom API throttled at first probe (HTTP 429 `Rate exceeded.`, 0.2 s response — hard rate-limit, not timeout). Fallback stack executed cleanly: **full cs.CV recent listings (776 entries, Sep 3–12, +50 over the prior brief's window) + full cs.GR recent listing (46 entries — id set identical to the prior brief's, zero new cs.GR this window)**, keyword-scanned; targeted abs-page extraction for candidates; Hugging Face daily-papers API for Sep 10–12 (26 + 32 entries; Sep 12 API returns 0 — feed lag, not absence); 5-lane web-search discovery.

### Gaussian Splatting / Temporal

- **GSReuse: Temporally Adaptive Screen-Space Reuse for Accelerating 3D Gaussian Splatting** (Tao, Sun, Huang, Zhu, Guo, Zhang, Wang, Guo — Nanjing University; **IEEE TVCG vol. 33 no. 5, 2026**, IEEE Xplore 11460301; not on arXiv). Lightweight drop-in accelerator for 3DGS rendering that reuses computations across consecutive frames: from the current frame's rendered image, depth, and camera parameters it **estimates reliable Gaussian-splatting motion vectors for all pixels and warps reusable contents to the new view**. Why it matters to OSS: (1) an independent, published precedent that "per-pixel splat motion vectors + warp + reliability-gated reuse" is a winning primitive for splat-render acceleration — the same thesis as the OSS canvas warp, argued from the pure-runtime side; (2) its reliability estimation for warp reuse is directly comparable to the v7 canvas's disocclusion/spawner gating (when a warped splat may be trusted vs respawned) — the closest TVCG-side cousin of the canvas update rule; (3) #13 methodology reference alongside Splats-to-Silicon/TileGS for frame-time accounting. **Not on arXiv; IEEE paywall** — author-site PDF hunt queued (letianhuang.github.io lists it). Newness note: absent from all prior briefs' GS sets; surfaced via discovery lane, not the listing scan.
- Nothing else new in cs.GR this window — all 46 ids covered by the 09-13 brief. Re-checked and rejected from the +50 cs.CV window: Vidu S2 (2609.11638 — real-time generative video/avatar, wrong class), LightSplat (2609.07274 — IROS 2026 GS-SLAM, robotics tracking domain), Tri-DehazeGS / mmWave splatting (known rejections).

### Super-Resolution / Upscaling

- **Nothing new past the materiality bar.** FreeTransformSR (2609.05912) remains the newest tracked SR entry (#9 reparameterization set). The +50 newest cs.CV entries contain no new SR/upscaling work; OracleZoom (2609.06490) surfaced again via HF daily papers (Sep 10) — still rejected (offline quality class, not a sub-2 ms donor). MedGSSR (2609.06874, ECCV 2026) re-confirmed from its abs page; already tracked as trend evidence for the feedforward-Gaussian-SR student bet.

### Frame Extrapolation / Generation

- **Nothing new past the bar.** Discovery re-confirmed the known set: Amulet (2608.10423, only architectural threat, mirrored locally), X4Rate (VISAPP 2026), GFFE (2406.18551), PatchEX (2407.17501), STSS (2312.10890). No new `frame extrapolation` entries in listings or discovery.

### Inference Optimization

- **Nothing new past the bar.** INT8 SISR deployment paper (2604.20291, CVPRW) still the reference; TensorRT 10.16 (Mar 2026) still current; EFQ-Softmax (2609.09721) already tracked 09-11. HF Sep 10–11 dailies contributed nothing in-lane (quantization hits were LLM-KV/attention-class).

## Archive.org Findings

- **Wayback CDX re-verified (2026-09-13 ~12:00 UTC):** GS-STVSR (2604.18047) — **still zero captures** (local mirror verified, 18.7 MB; risk closed); Amulet (2608.10423) — **still zero captures** (local mirror verified, 110 MB); GRAPE (2604.14706) — unchanged single capture (2026-05-02, 200); GSASR (2501.06838) — 12 captures on the abs URL, latest successful 2026-06-07 (200). **CDX reconciliation done this run:** the pdf URL form (arxiv.org/pdf/2501.06838) has its own capture set, latest 200 on **2025-08-01** — the prior brief's "latest 2026-08-01" was a year-typo of that pdf-form capture. No takedown signal under either form; risk closed.
- All three local PDF mirrors remain **uncommitted** (`docs/research/papers/`, 155 MB) — same commit blocker as prior briefs.

## GitHub / Open Source

- **DLSS Unlocked (ShyVortex/dlss-unlocked) — 385 → 393★ (+8 in ~12h), push unchanged 2026-09-11 02:52Z.** Plateau decay continues (15 → 8/day); multi-outlet DLSS-5-on-old-RTX coverage keeps discovery alive.
- **OptiScaler (optiscaler/OptiScaler) — 10,718 → 10,799★, pushed 2026-09-13 11:45Z (this run).** Latest tagged release still v0.9.4 (2026-07-18) — main-branch churn continues without a release. OptiScaler modders are also the RDNA-2 FSR 4.1.1b DLL port authors (see News).
- **DLSS5VKLayer (bmitch87/DLSS5VKLayer) — 70 → 71★, pushed 2026-09-13 00:09Z.** Linux DLSS 5 VKLayer steady.
- **NVOFG (SimonVitzethum/NVOFG) — 1★ unchanged, pushed 2026-09-12 19:43Z.** Linux Vulkan FG middleware; watch stands.
- **FidelityFX SDK: v2.3.0 (2026-06-24) confirmed unchanged** — closes the prior brief's errored release check; no AMD SDK movement since June.
- **Mob-FGSR/MobFGSR — 112★, pushed 2025-06-11** — unchanged; #5 No-ML baseline reference stands.
- Repo searches (GS+SR, frame-gen upscaling, neural upscaler, GS+temporal; sorted by push): nothing above the known noise set. Two new 0★ noise-tier leads in the integration-management slot: **LoganSMiller/NeuralSwap** (Rust+Tauri neural-rendering upscaler manager for PC games) and **projectsaturnstudios/neural-upstream** (ReShade add-on running DLSS 5 NR *before* the game's upscaler) — conceptually interesting (NR as pre-pass) but noise-tier.

## News

- **Qualcomm Adreno Neural Fusion — the run's strategic item.** Qualcomm newsroom + Android Authority + Android Central + HotHardware + tbreak + Tech Times (Sep 2–3): dedicated **AI Matrix Cores inside the Adreno GPU**; pitch is "unifies neural processing, **AI super resolution and frame generation into a single graphics pipeline**," running models in-GPU rather than offload; worked examples: **960×540 → 1920×1080 upscaling and 30→60 fps frame generation using motion vectors and optical flow**; Unity and Unreal onboard. Xenospectrum reports the "next-gen Snapdragon Pro exclusive" claim is unconfirmed. Why it matters: a third mobile vendor (after Arm Mali G2-Ultra NX) shipping *unified* SR+FG on fixed-function silicon — rhetorically the OSS "one architecture, not two" thesis, delivered as a vendor walled garden. Watch: does Qualcomm expose per-title hooks the way DLSS/FSR/XeSS do, or is this platform-internal only? Affects the mobile column of the OSS layer matrix and the Pico-tier student hardware assumptions (INT8/INT16 in-shader accel).
- **DLSS 5 officially confirmed for RTX 40** (Wccftech, TweakTown, PCGuide, Notebookcheck, tech-insider, Sep 3): NVIDIA says RTX 50 optimization comes first, RTX 40 (Ada) support "later this fall," no date. Firmer than the prior brief's mod-only story — DLSS 5's installed base is set to widen materially this fall; the "neural rendering must get cheap to ship" narrative now has an official mid-tier rollout attached.
- **Second-GPU DLSS 5 offload story widened:** PC Gamer, Digital Foundry, PCMag ("SLI is back?"), Wccftech (Sep 7–11) — ReShade add-on shifts DLSS 5 NR to a second GPU, up to +127% FPS; Digital Foundry notes latency doubling. DLSS 5 NR cost + community workaround ecosystem is now mainstream (Tom's Hardware Sep 7 was the first outlet last brief).
- **MFG modded onto RTX 40, 30, and 20** (Wccftech Sep 11; VideoCardz, TweakTown, overclock3d Sep 3–8): multi-frame generation works on Ampere/Turing via mods despite NVIDIA app update attempts to block it. Hardware gating across both DLSS 5 and MFG is demonstrably a software boundary — third corroboration of the "integration shims are commodity; the differentiating layer is the model" positioning.
- **FSR 4.1.1b on RDNA 2 via single DLL swap** — carried from last brief (Sep 7–10, five outlets); igor'sLAB (Sep 10) corroborates ("replacing a DLL is all it takes"). Also FSR 4.1 UE 5.8 plug-in for RX 7000 (igor'sLAB, Sep 1). Teacher-signal capture on RDNA 2 remains unlocked for #7.
- **Arm / Dimensity 9600:** MediaTek confirmed the Sep 15 reveal (biggo Sep 10, YugaTech Sep 9). Mali G2-Ultra NX NFRU deep-dive watch stands for the day after.
- **Intel:** no XeSS model or MFG changes; Panther Lake Arc B390 handheld benchmark chatter continues.
- **Apple:** iPhone 18 Pro launch window (preorders Sep 12); **MetalFX still silent** — no temporal-upscaler news attached.
- **DirectSR:** still silent across a 30-day window — the vendor-neutral FG API slot remains empty.

## Action Items / Follow-ups

- [ ] **P0 (carried, worsening):** recover or formally stop `srcnn-v7.0-pico-005` — 329h frozen at 36,650; ninth consecutive dead-trainer brief. Fix `data.json` `active: true` flag and the static-page "training now" copy to read from `status.json`.
- [ ] **P0 (carried, trivial):** commit the research brief series + `BIBLIOGRAPHY.md` + 155MB `docs/research/papers/` (LFS or release attachment). Three months of provenance unpushed.
- [ ] **New: GSReuse deep read → #13 + canvas-warp design note.** Acquire the TVCG 33(5) PDF (author site letianhuang.github.io first; IEEE Xplore fallback). Extract (a) per-pixel Gaussian-motion-vector estimation method, (b) reliability gate for warp reuse vs re-render, (c) measured speedup and quality delta vs full re-rasterization. Map (b) directly onto the v7 canvas disocclusion/spawner decision — this is an independent group converging on MV-warped splat reuse, cite in the OSS-FX positioning as precedent.
- [ ] **New: Qualcomm Adreno Neural Fusion competitive note.** When full tech details land (next Snapdragon reveal), compare against OSS-FX: their unified SR+FG-in-one-pipeline pitch vs the canvas's continuous `t_query` extrapolation; check whether MVs are engine-supplied (like Mob-FGSR) or optical-flow-estimated; assess Pico-tier student assumptions against their Matrix Core INT8 claims.
- [ ] **Watch: DLSS 5 RTX 40 official rollout "later this fall"** — widens the NR-cost narrative to mid-tier hardware; re-check the <2ms-student thesis framing when it ships.
- [ ] **Carried: P-GSVC deep read → #12 layered-canvas design note** (base+enhancement split policy, cross-layer optimization mechanics, progressive quality/cost curve).
- [ ] **Carried: #12 mechanism three-way** — LinearMask-GS vs Compact-FFGS vs CVT-GS; schedule ablation decision before freezing packed-canvas capacity (temporal-identity + L2-residency gates).
- [ ] **Carried: Laplacian-FH training control** — frequency-staged growth + field archival in the v7 ablation set.
- [ ] **Carried: #7 clean-room sequencing** — FSR 4.1.1b runs on RDNA 2 via DLL swap; capture teacher signal before reading any released model code.
- [ ] **Carried: Arm NFRU deep-dive after Sep 15** (Dimensity 9600 / Mali G2-Ultra NX vs OSS-FX exact-engine-MV warping).
- [ ] **Infra (carried):** `.com` → `.org` 301 audit (README, docs, capture-tool installer).
- [ ] **Erratum fixed this run:** `BIBLIOGRAPHY.md` LinearMask-GS entry listed "Ryu and Kweon" — arXiv abs page (2609.10095) lists Donghun Ryu, Minhyeok Lee; corrected to "Ryu and Lee."
- [x] **CDX nit (resolved this run):** GSASR latest-capture dates reconciled — abs form latest 200 on 2026-06-07, pdf form latest 200 on 2025-08-01; the prior brief's "2026-08-01" was a year-typo of the pdf-form capture. No takedown signal under either form.
- [ ] **Infra note (carried):** arXiv Atom throttled again (429 on first probe). Listing-page + abs-extraction fallback remains the default; HF daily-papers JSON API added to the fallback stack this run (works via curl, JS-free).

## Sources

1. `status.json` — https://opensupersampling.org/status.json (checked 2026-09-13T12:01:04Z)
2. `data.json` — https://opensupersampling.org/data.json (v7 run `latest_step` 36,650, `active: true`)
3. `.com` redirect — https://opensupersampling.com/status.json → 301 https://opensupersampling.org/status.json (verified 2026-09-13)
4. arXiv Atom API — export.arxiv.org (throttled: 429 `Rate exceeded.` on probe)
5. arXiv listing pages — cs.CV/recent skip=0–750 (776 entries, Sep 3–12) + cs.GR/recent (46 entries, ids identical to 09-13 window)
6. arXiv abs pages — 2609.10095 (LinearMask-GS), 2609.06874 (MedGSSR), 2609.07274 (LightSplat, rejected)
7. Wayback CDX — 2604.18047 (0 captures), 2608.10423 (0), 2604.14706 (1), 2501.06838 (12, latest 200 on 2026-06-07) — checked 2026-09-13
8. GitHub API — ShyVortex/dlss-unlocked (393★), optiscaler/OptiScaler (10,799★, pushed 09-13 11:45Z; release v0.9.4), bmitch87/DLSS5VKLayer (71★), SimonVitzethum/NVOFG (1★), Mob-FGSR/MobFGSR (112★), GPUOpen FidelityFX-SDK (v2.3.0), repo searches sorted by push
9. IEEE TVCG — GSReuse, vol. 33 no. 5 (2026), csdl 11460301 + ieeexplore 11460301; author page letianhuang.github.io
10. News (Google News RSS, when:14d): Qualcomm newsroom + Android Authority/Android Central/HotHardware/Tech Times/xenospectrum (Adreno Neural Fusion), Wccftech/TweakTown/PCGuide/Notebookcheck (DLSS 5 on RTX 40), PC Gamer/Digital Foundry/PCMag/Tom's Hardware (second-GPU DLSS 5), Wccftech/VideoCardz/TweakTown (MFG on RTX 40/30/20), igor'sLAB + 5 outlets (FSR 4.1.1b on RX 6000), biggo/YugaTech (Dimensity 9600 Sep 15)
11. Hugging Face daily papers API — 2026-09-10 (32 entries), 2026-09-11 (26 entries) scanned; 2026-09-12 returned 0 (feed lag)
