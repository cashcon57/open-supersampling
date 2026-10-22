# OSS Research Brief — 2026-10-20 (evening delta)

Delta over the morning brief (`2026-10-20-oss-research-brief.md`). Only new items are listed; carried items are not repeated.

## Project Status Summary
- **Latest commits:** HEAD unchanged — `b44278a test: add local Chrome Playwright dashboard smoke`. Working tree still carries uncommitted briefs (2026-08-11 → 10-20) and modified `BIBLIOGRAPHY.md`.
- **Open issues (10, unchanged):** #4, #5, #6, #7, #8, #9, #10, #12, #13, #14.
- **Training dashboard (browser-rendered 20:00 EDT):** `srcnn-v7.0-pico-005` still at step **36,650 / 100,000**, loss 0.19268, viz @ 36,000. Health row: **"Trainer — offline, stuck at step 36650 for 116h"** (104h this morning → 116h; ~4.8 days dead). Watcher/CF Worker/R2 healthy. Header still says **"live"**, **"ETA ~1.1d"**, **"v7.0 Pico OSS-FX is training right now"** — contradicts the offline row. Still unfixed.

## New Papers (arXiv)
Tooling note: `export.arxiv.org` API returned HTTP 429 ("Rate exceeded") for all queries this run (full 17-query sweep → 0 results), so the keyword sweep could not be redone. Fallback: `rss.arxiv.org/atom/cs.CV` + `cs.GR` (today's announce list, 529 entries) filtered locally against the bibliography and earlier briefs.

### Gaussian Splatting / Temporal
- **GaussianHSI — Voronoi-guided Bilateral 2D Gaussian Splatting for Arbitrary-Scale Hyperspectral SR** (arXiv:2604.17727v2, replaced 2026-10-20) — 2D-GS arbitrary-scale SR with Voronoi-guided adaptive placement instead of plain tile rasterization. *OSS relevance:* low (hyperspectral, offline); only the Voronoi-guided Gaussian placement idea is potentially relevant to the v6.1/v7 spawner (tile-grid artifact work). Not added to bibliography.
- **GSMR3D — Arbitrary-Scale Anisotropic 3D MRI SR with Gaussian Prototypes** (arXiv:2610.24858, 2026-10-20) — learned dictionary of anisotropic 3D Gaussian prototypes, one checkpoint across scales. Off-domain (medical); noting as another point of evidence that dictionary/prototype Gaussians are an accepted ASR primitive. Not added.
- **LENS — Localization and Neural Splatting on the Edge: FPGA Scene Memory** (arXiv:2610.25068, 2026-10-20) — co-designs the two dominant kernels (scene-coordinate regression + 3DGS rendering) for an AMD KR260 FPGA/ARM. *OSS relevance:* low; a data point on how irregular splat rasterization maps to non-GPU fixed hardware (relevant if OSS ever targets NPUs, #8). Not added.
- Off-topic feed-forward 3DGS hits (FineSplat 2610.25011, ZipSplat 2606.05102v3, GeoForward, MCSplat) — scene reconstruction, no temporal/SR angle.

### Super-Resolution / Upscaling
- Nothing new beyond SCOPE-VSR (2610.24604, in morning brief).

### Frame Extrapolation / Generation
- Nothing on target. **ActionRoPE** (arXiv:2610.25861, 2026-10-20) — action-as-coordinates conditioning for planar-camera game world models; generative world-model scope, no engine-signal upscaling. Not material.

### Inference Optimization
- Nothing on target. (SpecKCast 2610.26837 is diffusion feature-forecasting; irrelevant to a ≤1M CNN.)

## Archive.org Findings
- Advanced Search (2026-10-01 → 12-31): `"gaussian splatting" super-resolution` → no relevant hits; `"frame generation"` → 7 items, all GitHub-repo snapshots (e.g. DroidDeck) — no research value.
- **Wayback availability API again HTTP 429** for GS-STVSR 2604.18047 / GSASR 2501.06838 / GRAPE 2503.06617 → takedown/version re-check not re-verified. Live `arxiv.org/abs/...` returned HTTP 200 for all three → no takedown.

## GitHub / Open Source
- **`afrizal1211/openfg-oss`** (0★, created 2026-10-20) — "Open-source interpolated frame generation for Android games". Appears to relate to the **OpenFGG/openfg-app** project (Zygisk/Magisk module hooking GL ES/Vulkan present; interpolation, closed-source binaries, v1.2 released Aug). Interpolation-based (adds latency), root-only, mobile. *Relevance:* confirms demand for hook-based FG outside vendor SDKs; not a threat to OSS-FX's extrapolation design. Worth a look only for its present-hook/frame-pacing approach (OSS integration shim).
- No new Gaussian-splatting SR / extrapolation repos (only `ResonantBraneSplatting`, already tracked). `991o2o9/…Auxiliary-Inputs…Lightweight-Neural-Upscaling` unchanged (1★, plan only).
- Carried hobby/modder tooling unchanged (OptiScaler-based DLSS 5 injection now documented — see News).

## News
- **AMD FSR Upscaling 4.1 expands to RX 6000, Ryzen AI Max/AI 9, Z2 handhelds; Xbox Series X|S "soon" (primary: AMD newsroom + Adrenalin 26.10.1 release notes, 2026-10-20; Tom's Hardware, The Verge, Digital Trends, IGN).** Key technical facts:
  - Same ML model retuned to run via **INT8 on shader hardware** (no FP8 matrix units); AMD says RDNA2 has ~½ the INT8 rate per CU and ~2.5–3× lower total INT8 throughput than RX 7000-class (diit.cz analysis).
  - **Handheld/APU model is "less than half the size" of the full INT8 model** (ASUS ROG Xbox Ally preview driver 10-23; ~400 FSR 3.1+ games enable via driver toggle with no game update).
  - AMD's quick test: **RX 6900 XT, Cyberpunk 2077, 4K FSR Balanced → FSR 4.1 costs ~13% perf vs FSR 3**.
  - AMD dev article (Redstone SDK 2.2): **FSR Upscaling Performance mode renders 4K in ~1.3 ms on RX 9070 XT**; FSR FG claims 4.7× FPS in CoD BO7/Cyberpunk at 4K.
  - Driver now auto-upgrades FSR 3.1 games → ML upscaling without title patches (the DLL-swap integration model OSS's shim path assumes).
  - Also: FSR FG frame-pacing improvements; Ray Regeneration expanding; **FSR Diamond** (Project Helix; ML upscaling + **ML multi-frame generation** + next-gen ray regen) — rumored RDNA 5-only (KeplerL2, unverified).
  - *Relevance — competitive positioning, high:* (1) A vendor now ships a ML SR **on commodity INT8 shader hardware** with ~13% frame cost at 4K Balanced and a sub-half-size handheld model. This is the direct benchmark for the Pico/Standard student tiers (#4, #7, #8) and the <2 ms target — 1.3 ms @ 4K Performance on a 9070 XT is the number to beat on a comparable class. (2) Validates INT8-first student export (not FP8-only) for cross-vendor reach (#8/#9). (3) AMD is moving FG to **ML multi-frame** (FSR Diamond) — still separate SR+FG networks, so OSS's single-canvas α<1 pitch remains differentiated, but the "two networks" claim should be phrased against DLSS/FSR generically.
- **NVIDIA — faster DLSS 5 Neural Rendering model (NBA 2K27, 2026-10-16; TechPowerUp, ExtremeTech, Tom's):** new model 24% smaller (~118M params), −20% VRAM at 4K, up to 2× faster NR pass; modders extracted DLL 310.9.16.0 (kept as Preset B, old = Preset A) and inject via OptiScaler/RenoDX; Witcher III up to 90% faster, works on RTX 20/30/40 via modified runtime. Already partly tracked; new detail is the extraction/injection path — DLL-level portability keeps being the community norm.
- **Intel — XMX XeSS under Proton:** coverage wave 10-19/20 (Phoronix, PC Gamer, TweakTown, VideoCardz). Same four PRs, all still open (MIXED Reality News 10-20). No change since the morning brief.
- **Apple / Metal:** no new MetalFX item (RSS returned unrelated Apple content).
- **Engines:** no UE5/Unity upscaling news; only PC perf guides (Hellraiser: Revival) and the CoD/Cyberpunk driver items above.

## Action Items / Follow-ups
- [ ] **Dashboard/trainer (highest priority, 116h and rising):** pico-005 stalled at 36,650. Restart or park; fix "live"/"ETA ~1.1d"/"training right now" copy while offline (carried).
- [ ] **Update competitive table / README claims with FSR 4.1 data** (INT8 on shader HW, ~13% 4K-Balanced cost, 1.3 ms 4K-Performance on 9070 XT, <½-size handheld model) → inform #4/#7/#8 targets; avoid claiming OSS "<2 ms ≈ FSR-class" without a same-hardware measurement.
- [ ] **#7 (FSR 4 as teacher signal):** FSR 4.1 now ships on far more hardware (RX 6000 INT8 path) → easier to capture teacher outputs on commodity AMD cards; check legal/ToS stance first (DLL-extraction pattern in the DLSS ecosystem is a legal foil).
- [ ] **#8/#9:** prioritize INT8 shader (non-matrix-unit) student path alongside FP8 — AMD's RDNA2 numbers show how much penalty an INT8-shader port takes (~2.5–3× lower throughput than RX 7000).
- [ ] Watch FSR Diamond (ML MFG, Helix; RDNA5-only rumor) and the ASUS ROG Xbox Ally 10-23 FSR 4.1 handheld preview driver for real handheld model frame times.
- [ ] Re-run arXiv API sweep once the 429 clears (full 17-query A–E sweep was not possible this run); re-run Wayback takedown check for GS-STVSR/GSASR/GRAPE.
- [ ] Carried: deep-read 2610.22778 (dGS/dBS); auxiliary-inputs ablation repo; Intel Proton XMX PR status; Sony FGSS claims; Frozen History (2610.15752).

## Sources
- AMD newsroom — *AMD FSR Upscaling 4.1 Boosts Frames and Image Quality on More Devices* (2026-10-20); AMD Adrenalin 26.10.1 release notes; AMD dev article *Upscale Everything: Super-Resolution Across AMD Hardware*
- [Digital Trends](https://www.digitaltrends.com/gaming/amd-drops-fsr-4-1-on-five-year-old-radeon-graphics-cards-and-xbox-is-next/) · [Inven Global](https://www.invenglobal.com/articles/27428/fsr-41-coming-to-xbox-series-x-multi-frame-gen-planned-for-project-helix) · [diit.cz](https://diit.cz/clanek/fsr-41-pro-radeony-rx-6000-apu-ryzen-ai-9-ryzen-max-prichazi) · [PC Guide — FSR Diamond](https://www.pcguide.com/news/amds-next-gen-fsr-diamond-for-project-helix-gets-multi-frame-gen-but-your-current-gpu-may-not-support-it/)
- [TechPowerUp — faster DLSS 5 model extracted](https://www.techpowerup.com/353905/nvidias-faster-dlss-5-model-gets-extracted-from-nba-2k27-modders-report-up-to-90-gains) · [ExtremeTech](https://www.extremetech.com/computing/modders-ripped-faster-dlss-5-from-nba2k27-with-impressive-results)
- [OpenFGG/openfg-app](https://github.com/OpenFGG/openfg-app)
- arXiv RSS (cs.CV, cs.GR) 2026-10-20; GitHub REST Search; archive.org advancedsearch; Google News RSS (`when:3d`); [OSS dashboard](https://opensupersampling.com) (browser-rendered 2026-10-20 20:00 EDT)
