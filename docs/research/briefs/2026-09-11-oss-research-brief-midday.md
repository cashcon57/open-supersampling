# OSS Research Brief — 2026-09-11 (midday update)

> Delta run over the 2026-09-11 morning brief (written ~00:20 ET). Headlines: the **P0 Amulet/GRAPE PDF mirrors are done and verified** — Amulet now has a local safety copy matching its zero-Wayback-capture status, and GRAPE's single-capture risk is closed. arXiv API recovered (187 unique hits; `abs:` field queries worked again) but produced **zero new material papers** in the Sep 5–11 window beyond the morning brief's haul — a clean negative sweep. DLSS Unlocked keeps climbing (299 → **335★**, +36 in ~12h). Trainer now frozen **281h** (fifth consecutive dead-trainer brief). No news movement on DLSS 5 / FSR 4 / Adreno beyond morning coverage.

## Project Status Summary

- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` — unchanged. Local tree carries the accumulating uncommitted research-brief series (Aug 11 → Sep 11) + BIBLIOGRAPHY.md edits; still nothing pushed since June. The brief series itself is now a liability until committed.
- **Open issues:** 12, unchanged set (#1–#14 minus gaps; all last touched 2026-05-08). Tracker is static; the live research state lives in this directory.
- **README snapshot:** unchanged — pre-alpha v7, native `(x,y,t)` Gaussian canvas, OSS-FX = same canvas at a different `t_query`.
- **Training dashboard:** `srcnn-v7.0-pico-005` frozen at **36,650 / 100,000 (36.7%)**; `status.json` (checked 12:03 UTC): trainer **offline, stuck at step 36650 for 281h** — 233h → 245h → 257h → 269h → 281h across five briefs; watcher/CF Worker/R2 all healthy. The trainer has now been dead longer than the entire month-long run of briefs that have been reporting on it.
- **Operational conclusion:** publishing infra green, training dead. This is now a standing infrastructure defect, not a transient outage; until the 3080 Ti host is recovered (or the run formally stopped), the dashboard headline remains misleading and all "training now" framing in external comms should be avoided.

## New Papers (arXiv)

**Sweep health:** the arXiv API `abs:` field queries that failed in the morning run recovered this run — 187 unique IDs across 16 queries (GS/SR/temporal/frame-gen/TensorRT/distill/upscaling), plus GitHub repo search and the Hugging Face daily papers feed (Sep 10) as the requested second discovery source. Cross-checked against the morning brief's category listings: no missed material papers. Negative sweep — dedup against 65 known IDs, nothing new crossed the materiality bar.

### Gaussian Splatting / Temporal

- **Nothing new since the morning brief.** Checked and rejected this pass: Quadratic Phase GS (2609.11434 — hologram primitive variant), Gaussian Light Transport (2609.11430 — GI via 13D Gaussians, elegant but offline), Splats-to-Silicon (2609.06157 — kept below as methodology reference), UniFusion (2609.05888 — sparse-view 4D reconstruction, wrong regime), AVSplat/RIDE/Tri-DehazeGS/TV-SGS (SLAM/pose/dehaze/sparse-view — all out of scope).

### Super-Resolution / Upscaling

- **FreeTransformSR** (arXiv:2609.05912, 2026-09-05) — lightweight image SR via free low-rank learnable transforms. Only new SR entry past the bar; it joins the #9 reparameterization comparison set (RepVGG/ECBSR/RepSR lineage) as a candidate student-backbone donor. Low urgency; record, don't deep-read.
- *(Rejected: three domain-specific SR papers — DEM diffusion SR 2609.11886, crystallography SR 2609.10898, multi-modal domain-adaptive SR 2609.11201; all wrong-domain. OracleZoom 2609.06490 re-surfaced in listings, already recorded Sep 11 morning.)*

### Frame Extrapolation / Generation

- **Nothing new.** Zero single-network frame-extrapolation papers in the window — sixth consecutive brief with an empty direct-competitor lane. **Amulet (2608.10423) remains the only direct architectural threat, now safely mirrored locally** (see Archive). OSS-FX's "same canvas, different `t_query`" formulation is still unpublished-competitor-free ~5 months after GS-STVSR.

### Inference Optimization

- **Splats to Silicon: Rethinking Computational Efficiency of 3DGS** (arXiv:2609.06157, 2026-09-05) — a workload-centric framework connecting representation/runtime/hardware optimizations to end-to-end frame time (Gaussian selection → screen-space work → data movement → stage time). **Why it matters to OSS:** exactly the accounting structure #13's Nsight L2 profiling and the future "32B/Gaussian" and <2ms claims need — adopt as the methodology reference for any canvas efficiency claims before they're published.
- *(EFQ-Softmax 2609.09721 and Ozaki 2.5 2609.09095 from the morning brief remain the only other FP8-path notes; no change.)*

## Archive.org Findings

- **ACTION COMPLETED (was P0):** mirrored and verified both papers this run:
  - `docs/research/papers/2608.10423-amulet.pdf` — **110MB, `%PDF` header verified**. Wayback re-checked this run: still **zero captures** (abs + pdf, CDX empty). This was OSS-FX's primary competitive reference resting on a single arXiv copy; risk closed.
  - `docs/research/papers/2604.14706-grape.pdf` — **33.7MB, `%PDF` header verified**, size matches Wayback's single 2026-05-02 capture (33,591,754 bytes). GRAPE's single-capture marginality resolved.
  - Local mirror set now: GS-STVSR (18.7MB, verified prior run) + Amulet + GRAPE. **All three mirrors are uncommitted** — commit `docs/research/papers/` with the brief series.
- GSASR (2501.06838): unchanged, well-captured since 2025-01. No new takedowns or availability regressions detected this window.

## GitHub / Open Source

- **DLSS Unlocked (ShyVortex/dlss-unlocked) — 299 → 335★ (+36 in ~12h), pushed 2026-09-11 02:52Z.** The morning brief's projection of continued growth is confirmed; the RTX 40 DLSS 5 unlock + daily commit cadence is compounding attention. This is now the fastest-moving OSS-adjacent artifact in the ecosystem.
- **OptiScaler (optiscaler/OptiScaler)** — 10,604 → **10,646★**, pushed 2026-09-10. Steady.
- **New noise-tier entrants (0–1★, same DLL-injection slot, directionally confirming):** `interceptballast918/optiscaler-client` (OptiScaler companion client), `Bharath-vip/dlss5-studio` ("DLSS 5 workstation" for video/image enhancement), plus the previously tracked `neural-upstream`, `jyu041/dlss-rtxsr-upscaler`, `Ishanoshada/DLSS5-VLC-UpScaler` all still being pushed. The deployment slot OSS plans to occupy keeps accumulating small entrants — no technical threat, consistent validation of the integration thesis.
- **AMD FidelityFX SDK:** unchanged — latest release still v2.3.0 (2026-06-24), repo push 2026-06-24. **The FSR 4 source release has now been "imminent" for three briefs.** FSR 4.0.2 binaries/limited source remain on GPUOpen; the accidentally-published full source remains taken down with community forks circulating.
- GitHub search (GS+SR, frame-gen upscale, neural upscaler, pushed > Sep 9): nothing above noise; no new DLSS/FSR analysis tooling.

## News

Re-swept DLSS 5 / FSR 4 / Intel / Qualcomm this run: **no material new developments since the morning brief's coverage** — the DLSS 5-on-RTX 40 confirmation, FSR 4-source-release pending story, and Qualcomm Adreno Neural Fusion items all stand as reported. Deltas only:

- **Qualcomm (positioning sharpened):** Qualcomm's own OnQ blog now states Adreno Neural Fusion is "the first mobile AI super resolution technology of its kind supported across major game engines and is **commercially available**," with Unity and Unreal integration confirmed in coverage. This upgrades the standing watch item: the Pico student's mobile regime now has a shipping vendor solution to benchmark against, and "one architecture, two jobs (SR + FG)" has its first commercial proof point.
- **DLSS 5 / RTX 40:** same story, now corroborated by more outlets (heise, notebookcheck, pcguide): official expansion to RTX 40 confirmed, **no release date** for Ada. RTX 50 optimizations "later this fall" first. Nothing new for OSS positioning beyond the morning brief's determinism-framing analysis.
- **FSR 4:** no new source-release movement; FidelityFX SDK repo and releases unchanged (v2.3.0, Jun 24). The clean-room sequencing for #7 (capture FSR 4 teacher signal **before** any released code is read) remains pending and untriggered.
- **Intel:** XeSS 3 MFG driver rollout continues; Game On driver for NBA 2K27 fixed MFG crashes. No model-detail news.

## Action Items / Follow-ups

- [x] **P0 done:** Amulet PDF mirrored + verified (`docs/research/papers/2608.10423-amulet.pdf`, 110MB); GRAPE mirrored + verified behind it. Wayback zero-capture risk closed.
- [ ] **P0 (carried, worsening):** recover or formally stop `srcnn-v7.0-pico-005` — 281h frozen at 36,650; five consecutive dead-trainer briefs. Fix the dashboard headline to read trainer state from `status.json`.
- [ ] **P0 (new, trivial):** commit the research brief series + `BIBLIOGRAPHY.md` + `docs/research/papers/` (three PDF mirrors, ~163MB total — check repo size policy before committing; consider LFS or a release attachment if too large). A month of research provenance is currently unpushed.
- [ ] **Deep read (carried):** EdMCGS (arXiv:2609.08332) — Markov-chain temporal-identity/update mechanics vs the v7 canvas update rule.
- [ ] **Reference for #13 (new):** Splats-to-Silicon (arXiv:2609.06157) — adopt its workload-centric accounting before publishing any 32B/Gaussian or <2ms claims.
- [ ] **Reference for #9:** FreeTransformSR (arXiv:2609.05912) — add to the reparameterization comparison set; low urgency.
- [ ] **Eval methodology (carried):** View-Structured Conformal Prediction (arXiv:2609.10307) — certified-coverage gate for the future eval harness.
- [ ] **Reference for #12 (carried):** LinearMask-GS (arXiv:2609.10095) — decide pruning-vs-merging mechanism before freezing packed-canvas capacity.
- [ ] **Distillation robustness (carried):** RouteBridge (arXiv:2609.09606) — per-region routed teacher selection for #4/#7.
- [ ] **Competitor watch:** AMD FSR 4 source release on GPUOpen/GitHub — "imminent" for three briefs; complete the FSR4-teacher capture (#7) **before** reading any released code (clean-room). Watch DLSS Unlocked release cadence (335★, daily commits).
- [ ] **Mobile watch (upgraded):** Qualcomm Adreno Neural Fusion is commercially available per Qualcomm — benchmark target for the Pico student; first commercial "SR + FG in one architecture" proof point.
- [ ] **Positioning (carried):** audit README deterministic-reconstruction claims vs DLSS 5's own "deterministic, temporally stable, MV-anchored" marketing — differentiability = whole-frame extrapolation at arbitrary `t_query` + <2ms vs "60 FPS technology."
- [ ] **Process (done):** Hugging Face daily papers added as second discovery feed this run — nothing relevant trending Sep 10; keep in rotation.

## Sources

1. `status.json` — https://opensupersampling.com/status.json (checked 2026-09-11T12:03Z)
2. arXiv API sweep — export.arxiv.org, 16 field/category queries, 187 unique IDs, deduped against BIBLIOGRAPHY.md (65 known → 69 after this run)
3. Wayback CDX — web.archive.org/cdx for 2608.10423 (empty) and 2604.14706 (single 2026-05-02 capture, digest-matched)
4. GitHub API — ShyVortex/dlss-unlocked (335★), optiscaler/OptiScaler (10,646★), GPUOpen-LibrariesAndSDKs/FidelityFX-SDK (v2.3.0 / pushed 2026-06-24), repo search pushed>2026-09-09
5. Hugging Face Daily Papers — huggingface.co/papers?date=2026-9-10 (no GS/SR/frame-gen entries)
6. News: windowsforum.com/notebookcheck.net/pcguide.com (DLSS 5 RTX 40), gpuopen.com/guru3d.com (FSR 4), qualcomm.com/news/onq/2026/09/adreno-neural-fusion-ai-rendering, androidheadlines.com, hothardware.com (Adreno Neural Fusion)
