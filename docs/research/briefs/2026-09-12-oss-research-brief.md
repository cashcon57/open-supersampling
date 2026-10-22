# OSS Research Brief — 2026-09-12

> Evening run (written ~00:30 UTC). Headlines: **arXiv API outage** — all complex/phrase queries (quoted phrases, ANDed terms, sortBy) hung or returned `Rate exceeded.` for the entire run; the sweep was executed via web-search + arXiv listing-page discovery instead, which surfaced **one new direct frame-extrapolation paper (X4Rate)** and confirmed the C-lane is still otherwise empty. Bigger: **Arm entered the mobile neural-upscaling race (Mali G2-Ultra NX, Sep 8)** — the second shipping "SR + FG in one GPU pipeline" vendor platform in a week, after Qualcomm. Trainer now frozen **293h** (sixth consecutive dead-trainer brief).

## Project Status Summary

- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` — unchanged since 2026-06-08. Nothing pushed to `cashcon57/open-supersampling` in 96 days. Local tree now carries **35 uncommitted paths**: the Aug 11 → Sep 12 brief series, `BIBLIOGRAPHY.md` edits, and `docs/research/papers/` (155MB, three verified PDF mirrors: GS-STVSR, Amulet, GRAPE).
- **Open issues:** 12, unchanged set (#1–#14 minus gaps; all last touched 2026-05-08). Tracker static.
- **README snapshot:** unchanged — pre-alpha v7 framing, `(x,y,t)` Gaussian canvas, OSS-FX = same canvas at a different `t_query`.
- **Training dashboard:** `srcnn-v7.0-pico-005` frozen at **36,650 / 100,000 (36.7%)**; `status.json` (checked 2026-09-12T00:00Z): trainer **offline, stuck at step 36650 for 293h** — 257 → 269 → 281 → 293h. Watcher / CF Worker / R2 all healthy. The run has now been dead ~12 days.
- **Operational conclusion:** unchanged — publishing infra green, training dead. Standing infrastructure defect; avoid all "training now" framing in external comms until the 3080 Ti host is recovered or the run is formally stopped.

## New Papers (arXiv)

**Sweep health:** the arXiv API degraded mid-run — quoted-phrase and AND queries hang server-side (HTTP 000 at 30s), single-term queries flapped between 200 and 429/`Rate exceeded.`, and nothing recovered after 90–120s backoffs. Per the fail-fast rule the API was abandoned after ~10 attempts; discovery completed via web search + arXiv abs-page extraction + Hugging Face daily papers (Sep 11 — 16 papers, zero GS/SR/frame-gen entries). Dedup was performed against `BIBLIOGRAPHY.md` (69 known IDs) by ID and by title. Caveat: without the API's category listings, recall for the Sep 5–12 window is lower than a normal run — treat this brief's "nothing new" lanes as "nothing found via search-engine discovery."

### Gaussian Splatting / Temporal

- **Nothing new past the materiality bar this window.** Re-checked and rejected: ATGS (2608.30184, already in BIBLIO — SIGGRAPH 2026, anchor-based temporal GS for volumetric video), TruncGradGS (2609.03534, training-stability), CompactAppearanceGS (2609.05255, already recorded), AVSplat/TV-SGS/RIDE/STARS-GS (sparse-view/pose/aerial — out of scope), PLSR (2609.06436 — 3D asset diffusion SR, wrong domain).

### Super-Resolution / Upscaling

- **Nothing new.** FreeTransformSR (2609.05912) remains the newest SR entry; it's already recorded for the #9 reparameterization comparison set.

### Frame Extrapolation / Generation

- **X4Rate: A Scene-Adaptive Extrapolation Framework for 4× Frame Rate Upsampling** (Dixit & Sarangi, IIT Delhi — VISAPP 2026, SciTePress proceedings 2026/142928; no arXiv ID). A meta-learner (LinUCB bandit) that **selects per-scene among multiple existing extrapolation algorithms at runtime** to hit 4× frame rate (60→240), reporting +22.13% PSNR over ExtraNet. Why it matters to OSS: it's the first direct frame-extrapolation entry in seven briefs, and its thesis is the opposite bet to OSS's — instead of one representation that extrapolates natively (canvas), it accepts that no single extrapolator generalizes and pays a router to pick one. Indirect validation that extrapolation quality is scene-dependent; the router overhead (~a classifier + N backends) is exactly the complexity the single-canvas thesis eliminates. Not a threat (no motion-vector-anchored continuous representation, no SR coupling), but a must-cite baseline in any OSS-FX eval write-up.
- Everything else in the lane checked and rejected: Amulet (2608.10423 — still the only architectural threat, mirrored), GFFE (2406.18551 — already designed against in the 2026-04-30 frame-extrapolation design doc), ExWarp (2307.12607 — RL router, older X4Rate-cousin), latency-controlled transmission extrapolation (ICIP 2022 lineage — video-codec context).

### Inference Optimization

- **Nothing new past the bar.** Splats-to-Silicon (2609.06157) remains the standing methodology reference for #13; TileGS (2609.03613, already in BIBLIO) continues to be the rasterization-perf reference (its "No-GW" depth-local ranges + Nsight attribution method is the pattern for the #12 packed-layout + #13 profiling work). INT8 SISR deployment paper (2604.20291, CVPRW) already known.

## Archive.org Findings

- **CDX re-verified all four tracked papers this run:** GS-STVSR (2604.18047) — **still zero Wayback captures** (local mirror verified, risk closed); Amulet (2608.10423) — **still zero captures** (local mirror verified); GRAPE (2604.14706) — unchanged single 2026-05-02 capture (local mirror digest-matched); GSASR (2501.06838) — well-captured since 2025-01, no regressions.
- archive.org full-text search for GS+SR: nothing new beyond GitHub-star-listing snapshots already known.
- All three local PDF mirrors remain **uncommitted** (`docs/research/papers/`, 155MB on disk) — same commit blocker as last brief.

## GitHub / Open Source

- **DLSS Unlocked (ShyVortex/dlss-unlocked) — 335 → 354★ (+19 in ~12h), pushed 2026-09-11 02:52Z.** Growth continues but the daily delta is shrinking (36 → 19), consistent with the initial-viral phase decaying into a steady plateau. Still the fastest-moving OSS-adjacent artifact; mod combines DLSS 5 NR + MFG on RTX 20/30/40.
- **OptiScaler (optiscaler/OptiScaler) — 10,646 → 10,664★, pushed 2026-09-11 23:06Z.** Steady.
- **New noise-tier entrants (0–1★, DLL-injection slot):** `KillerPixelCrew/ReScaleFrame` (framework for RE-ing games to add modern upscaling + FG — directionally interesting, empty star count), `jatelop8/DynamicShaderFrameGen` (FSR 3.1 FG + DLSS for Skyrim SE via SKSE), plus the previously tracked `interceptballast918/optiscaler-client`, `Bharath-vip/dlss5-studio`, `Ishanoshada/DLSS5-VLC-UpScaler`, `projectsaturnstudios/neural-upstream` all still being pushed. The injection-shim slot OSS targets keeps accumulating entrants — validation of the integration thesis, no technical threat.
- **AMD FidelityFX SDK:** releases unchanged (latest v2.3.0, 2026-06-24; repo push 2026-06-24). GPUOpen's FSR 4 page now advertises **FSR Upscaling 4.1.1 with RDNA 3 support inside SDK 2.3** — the "FSR 4 source release imminent" watch item should be retired and replaced by the driver-side rollout story (see News); the accidentally-published full source remains taken down with community forks circulating.
- GitHub repo search (GS+SR, frame-gen upscaling, neural upscaler; pushed > Sep 7): nothing above noise. IE-SRGS (2★, pushed Aug 29) is a course-scale 3DGS-SR repo — below the bar.

## News

- **Arm Mali G2-Ultra NX (Sep 8) — NEW VENDOR THREAT/VALIDATION, biggest item this window.** Arm's first "AI-native" mobile GPU (part of CSS for Mobile 2): dedicated neural accelerators **inside the shader cores** (INT8/INT16 + hardware optical flow), shipping three neural graphics technologies: **NSS** (neural super sampling), **NFRU** (neural frame rate upscaling — intermediate frames from motion vectors + optical flow), and **NSSD** (neural SR + denoising for ray-traced scenes). Arm claims up to 4× performance efficiency and 70% lower external memory traffic (Neural Dawn demo with Sumo Digital). Pair with Qualcomm's Adreno Neural Fusion (Sep 2): **both major mobile IP vendors now ship SR + FG as one GPU-integrated pipeline** — precisely OSS's "one architecture, two jobs" thesis, implemented in silicon. Market validation is now unambiguous; the differentiation bar for OSS's software canvas rises accordingly (OSS's differentiator vs both: vendor-neutral, engine-agnostic, continuous `t_query` extrapolation vs their fixed intermediate-frame generation). MediaTek Dimensity 9600 with G2-Ultra NX expected at the Sep 15 Arm event — watch for NFRU benchmark data.
- **AMD FSR 4.1 on RX 7000 (RDNA 3) is real and shipping:** Adrenalin 26.6.2 WHQL added official FSR Upscaling 4.1.1 for RDNA 3 discrete GPUs (driver-side, 300+ games, no dev patch needed), and the **FSR UE plugin update (Sep 1) brings FSR Upscaling 4.1.1 + FG 4.0.1 to Unreal Engine 5.8**, with RX 9000/7000 selecting the ML variant automatically and other hardware falling back to FSR 3 analytical. RDNA 2 scheduled 2027. The 3080 Ti's NVIDIA counterpart (DLSS 5 on RTX 40) remains "fall 2026, no date." For OSS: the FSR 4.1 INT8 model is now driver-distributable on the exact hardware class (#6's 3080 Ti) the Pico student targets — strengthens the case for the #7 FSR-4-teacher distillation capture (clean-room sequencing still pending: capture signal **before** reading any released code).
- **DLSS 5 / RTX 40:** no release date movement; outlets converge on "fall 2026" with no specific window. DLSS 5's launch title (NBA 2K27, Sep 3, driver 616.64) and 26 new GeForce NOW games are the only deltas.
- **Intel:** Game On driver 32.0.101.8991 (NBA 2K27 optimization) fixes XeSS MFG crashes with overlay tools (CapframeX/RivaTuner); no model changes. XeSS 3 MFG rollout continues.
- **Apple:** nothing new beyond macOS Tahoe 26's MetalFX Frame Interpolation (interpolates 1 frame per pair — same as prior briefs' coverage); WWDC26 "speedrun your game port" session documents MetalFX temporal-upscaling + frame-interpolation as separate skills — more evidence the industry norm is two mechanisms, which is the framing OSS's README already argues against.
- **DirectSR:** no movement observed (still FSR 3.1 upscaler-only preview state from May). Microsoft's abstraction layer remains SR-only — no frame-generation API surface, no threat to OSS-FX, but the slot where a vendor-neutral FG API would live is still empty.

## Action Items / Follow-ups

- [ ] **P0 (carried, worsening):** recover or formally stop `srcnn-v7.0-pico-005` — 293h frozen at 36,650; six consecutive dead-trainer briefs. Fix the dashboard headline to read trainer state from `status.json`.
- [ ] **P0 (carried, trivial):** commit the research brief series + `BIBLIOGRAPHY.md` + `docs/research/papers/` (155MB — check repo size policy; LFS or release attachment if too large). Three months of research provenance is unpushed.
- [ ] **New: X4Rate baseline citation** — obtain the VISAPP 2026 PDF (SciTePress 2026/142928) and add to BIBLIOGRAPHY as the scene-adaptive-routing counterpoint to the single-canvas thesis; cite in the OSS-FX eval plan (meta-learner routing cost vs continuous-canvas `t_query`).
- [ ] **New: Arm NFRU deep-dive after Sep 15** — Arm newsroom blog + Neural Dawn tech details; extract NFRU's motion-vector/optical-flow formulation vs OSS-FX's exact-engine-MV warping. Assess whether INT8/INT16 in-shader neural accel changes the mobile target hardware assumptions for the Pico student.
- [ ] **Watch: Arm/Qualcomm mobile SR+FG convergence** — two vendors in one week shipping "one GPU pipeline does SR + FG" validates the thesis but raises the differentiation bar; the README's differentiability claim (arbitrary `t_query`, near-zero marginal cost) should be explicitly contrasted against NFRU/Adreno's fixed intermediate-frame generation in the next positioning pass.
- [ ] **Carried: #7 clean-room sequencing** — capture FSR 4.1.1 teacher signal before reading any released model code. Note FSR 4.1.1 is now trivially runnable on RDNA 3 via Adrenalin 26.6.2, which lowers the capture cost if any AMD host is available.
- [ ] **Carried deep read:** EdMCGS (2609.08332) Markov-chain temporal mechanics vs v7 canvas update rule; VSCP-GS (2609.10307) certified-coverage eval gate; LinearMask-GS (2609.10095) pruning-vs-merging for #12; RouteBridge (2609.09606) routed teacher selection for #4/#7.
- [ ] **Carried reference:** Splats-to-Silicon (2609.06157) workload accounting before any 32B/Gaussian or <2ms claims; TileGS Nsight attribution method for #13.
- [ ] **Infra note (new):** arXiv API outage pattern (complex queries hang, 429s on recovery) — next run, if the API is still degraded, fall back immediately to web-search discovery + arXiv listing pages rather than spending 40+ minutes on retries.

## Sources

1. `status.json` — https://opensupersampling.com/status.json (checked 2026-09-12T00:00Z)
2. arXiv API — export.arxiv.org (degraded: HTTP 000 hangs on phrase/AND queries, 429 `Rate exceeded.` on singles; abandoned per fail-fast rule)
3. Web-search discovery — Google via web_search across GS/temporal, SR, frame-gen, TensorRT/int8, engine-integration queries; deduped against BIBLIOGRAPHY.md (69 known IDs)
4. arXiv abs pages — 2608.30184 (ATGS), 2609.03613 (TileGS), 2609.06436 (PLSR); SciTePress 2026/142928 (X4Rate); openreview wnJkdo5Gu9 (GameSR — known)
5. Wayback CDX — web.archive.org/cdx for 2604.18047 (empty), 2608.10423 (empty), 2501.06838 (captured), 2604.14706 (single capture)
6. GitHub API — ShyVortex/dlss-unlocked (354★), optiscaler/OptiScaler (10,664★), GPUOpen FidelityFX-SDK releases (v2.3.0, Jun 24), repo search pushed > Sep 7
7. News: newsroom.arm.com + unite.ai + techtimes (Mali G2-Ultra NX), gpuopen.com FSR 4 page + SDK 2.3 blog, windowsforum.com (Adrenalin 26.6.2, Jun 22), igorslab.de (FSR UE 5.8 plugin, Sep 1), tomshardware.com + gagadget.com (FSR 4.1 RX 7000), nvidia.com (DLSS 5 NBA 2K27, driver 616.64), neowin.net (Intel 32.0.101.8991), videocardz.com (DLSS Unlocked), engadget.com (MetalFX Frame Interpolation), devblogs.microsoft.com (DirectSR)
8. Hugging Face Daily Papers — huggingface.co/papers?date=2026-9-11 (16 papers, no GS/SR/frame-gen entries)
