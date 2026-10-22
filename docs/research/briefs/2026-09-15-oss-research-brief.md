# OSS Research Brief — 2026-09-15

> Evening run (written ~00:05 UTC Sep 15; follows the morning brief of Sep 14). Headlines: **The DLSS 5 cost-reduction modding stack has consolidated** — DLSS5-Autopilot (618★, pushed Sep 14 22:39Z) now auto-installs **Neural Upstream** (run Neural Rendering at internal resolution *before* DLSS SR; 63→50 FPS vs 31 FPS on RTX 4080 / AC Shadows 4K Perf, videocardz Sep 7–14 wave), and GameGPU quantified the baseline pain: **DLSS 5 NR drops RTX 5060 from 77 → 35 FPS in AC Mirage** (Sep 14). The "who pays for neural rendering" story now has a modder-written answer: **run the network at the lowest resolution available** — which is exactly the OSS canvas's input-domain thesis. **No new arXiv papers past the materiality bar** (Atom API 429 for a 5th consecutive run; listing fallback covered the full Mon-14 delta — both cs.CV pages re-scanned 90/90, cs.GR 42/42 — plus the Tue-15 batch has not been announced yet at write time). Trainer still dead: **365h frozen at step 36,650** (twelfth consecutive dead-trainer brief). Wayback availability API IP-throttled a 2nd consecutive run — archival re-checks deferred again.

## Project Status Summary

- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` — unchanged since 2026-06-08 (99 days). Uncommitted tree unchanged in kind (brief series + BIBLIOGRAPHY.md +papers/ still unpushed); this run adds the evening brief only.
- **Open issues:** 12 open (#1–#14 minus gaps), unchanged set.
- **Training dashboard:** `status.json` (checked 2026-09-15T00:00Z): trainer **offline, stuck at step 36650 for 365h** — 353h → 365h, consistent ~12h drift; counter honest, run dead. Watcher healthy (published 0s ago), CF Worker healthy (94 ms), R2 healthy. `data.json` cross-check (generated_at 2026-09-15T00:02:46Z): `srcnn-v7.0-pico-005` still `active: true` at step 36,650 — still the only active run, same contradiction as every prior brief; cost projections (RunPod rates) still displayed for 4090/A100/H100/B200.
- **Domain migration:** `opensupersampling.com` → **301 → `.org`** (re-confirmed this run). `.com` → `.org` audit remains open.
- **Operational conclusion:** unchanged — publishing infra green, training dead ~15.2 days. No "training now" framing in external comms until the 3080 Ti host is recovered or the run is formally stopped.

## New Papers (arXiv)

**Sweep health:** arXiv Atom API throttled a **5th consecutive run** (HTTP 429 `Rate exceeded.` on the first probe). Fallback stack executed: **cs.CV recent listing scanned in full for the Mon-14 delta (page 1: 50/50 entries, page 2 "continued": 40/40 — 90/90 total; the same Mon-14 announcement batch the morning brief covered, re-verified entry-by-entry), cs.GR recent scanned (42/42 entries; still the Mon-14 batch, no Tue-15 announcements at write time)**. The listing scan is now the standardized path (Atom probes dropped this run — matches the morning brief's infra recommendation).

### Gaussian Splatting / Temporal

- **Nothing new past the bar.** The Mon-14 cs.CV delta re-scan confirmed the morning brief's screen: NOVA-GS (2609.12682), MGAvatar (2609.12850), VideoTok4D (2609.12874) remain the only GS-adjacent titles and all stay rejected for the same reasons (low-light NVS denoise lane; mesh-bound head avatars; 4D video *tokenizer* for generative world models, not a rasterizable real-time canvas). VS-Splat (2609.12343) stays the one watch-tier item.
- **Known set unchanged:** GSReuse (deep-read complete, mirrored), ABCD (2608.27735), RouteBridge (2609.09606, carried deep read), LinearMask-GS / CVT-GS / Compact-FFGS (#12 three-way intact), P-GSVC (2603.10551, #12 layered-canvas donor).

### Super-Resolution / Upscaling

- **Nothing new past the bar.** No game/real-time SR work in the Mon-14 window beyond the already-screened DEM diffusion item. Known set: INT8 SISR deployment paper (2604.20291, CVPRW) remains the inference-optimization reference.

### Frame Extrapolation / Generation

- **Nothing new past the bar.** Known set unchanged: Amulet (2608.10423, only architectural threat, mirrored), X4Rate, GFFE, PatchEX, STSS. No new `frame extrapolation`/`frame generation` titles in the window.

### Inference Optimization

- **Nothing new past the bar.** TensorRT 10.16 still current; EFQ-Softmax tracked 09-11. The Input-Resolution-vs-Latency detection study (2609.12920) stays rejected (detection, not SR; framing already covered by H006) — though note it keeps gaining real-world corroboration from the Neural-Upstream news below (resolution-vs-latency is now the hottest empirical axis in shipped-game neural rendering).

## Archive.org Findings

- **Wayback availability API IP-throttled a 2nd consecutive run** — 429 on all 3 probes; a Memento TimeTravel alternate-egress probe also returned empty. Deferred again: SuperTimeGS first-capture check, IE-SRGS repo capture, GRAPE/GSASR re-check. No archival-risk state changed; all four high-risk PDFs (GS-STVSR, Amulet, GRAPE, GSReuse) already mirrored locally with verified sha256s (~207 MB, still uncommitted — same commit blocker as prior briefs).
- **SuperTimeGS repo:** still no known Wayback capture (last check 09-13 evening; API down for re-check this run). Repo unchanged (pushed 2026-09-13 23:48Z, 1★).

## GitHub / Open Source

- **DLSS5-Autopilot (Kizzuwatnaa/DLSS5-Autopilot) — the run's strategic repo.** 618★, created 2026-08-31, pushed 2026-09-14 22:39Z. One-shot configurator that scans the game library and auto-installs one of eight community DLSS 5 routes; headline integration is **Neural Upstream** (matiasLombo/neural-upstream, 103★, created 2026-08-31): run the DLSS 5 Neural Rendering network at **internal (pre-upscale) resolution, before DLSS SR**, instead of at output resolution. Videocardz' RTX 4080 / AC Shadows 4K-DLSS-Performance numbers: 63 FPS baseline → 31 FPS with NR-after-SR → **~50 FPS with Neural Upstream** (≈+60% vs the naive route; TweakTown's "modders fixed DLSS 5 before NVIDIA" framing, TechSpot: "two ways to make DLSS 5 playable, and neither one is Nvidia's"). **Why it matters to OSS:** independent, empirical confirmation that **neural networks in the render loop should run at the lowest resolution available** — the same first-principles argument behind the OSS canvas (Gaussian primitives live in a low-res domain; rendering is the upscaling step). Watch for NVIDIA adopting the ordering officially ("later this fall" DLSS 5 RTX-40 rollout); if they do, it's a public concession on the cost axis OSS is built against.
- **OptiScaler (optiscaler/OptiScaler) — 10,972 → 11,000★ (+28), pushed 2026-09-14 22:13Z.** Still **no new tag — v0.9.4 (2026-07-18) latest** (tags re-verified this run) despite heavy main-branch churn; the XeSS-MFG cross-vendor story continues to pull stars. When a tag lands, execute the #7 XeSS-MFG teacher-capture action item.
- **DLSS Unlocked (ShyVortex/dlss-unlocked) — 432 → 451★ (+19), pushed 2026-09-14 11:25Z.** igor'sLAB MFG-on-RTX-40 coverage still driving stars; repo consolidating the OptiScaler-fork lane.
- **Consumption-side DLSS 5 tooling keeps proliferating:** jyu041/dlss-rtxsr-upscaler pushed 09-14 (1★), Ishanoshada/DLSS5-VLC-UpScaler (1★), mohmedmm/nbox (0★, "240p → 2K/4K zero CPU load"), projectsaturnstudios/neural-upstream ReShade pre-pass add-on (0★, pushed 09-13). All downstream-injector lane, none in OSS's render-side lane.
- **GS + SR repos:** no new entries past the bar. IE-SRGS (2★, pushed 08-29) unchanged; SplatSuRe (94★) unchanged; course-project mirrors (EE5178) are noise. DLSSG-Stream-Worker (2★, created 09-04) unchanged — #7 offline-DLSS-FG-teacher eval item stands, still unvalidated.
- **FidelityFX SDK: v2.3.0 (2026-06-24) unchanged.** Mob-FGSR unchanged — #5 No-ML baseline reference stands.

## News

- **DLSS 5 neural-rendering cost story went from anecdote to quantified mainstream narrative.** GameGPU (Sep 14): **DLSS 5 drops RTX 5060 from 77 → 35 FPS (−55%) in AC Mirage**; NBA 2K27 "50%+ FPS hit" now confirmed by 6 outlets; BioShock Remastered 5K/51 FPS on a 5090. Engadget's Sep 12 "Why is the internet so upset about DLSS 5" plus a PS2-games-upscaling piece (Sep 13) put NR cost in consumer consciousness. **This is the cost-pressure axis the OSS <2 ms student thesis is built against — positioning material, cite the 5060 number.**
- **Modders "fixed" the cost before NVIDIA:** DLSS5-Autopilot + Neural Upstream (see GitHub section) — resolution-ordering is the lever; up to +60% by running NR pre-upscale. Also: unofficial MFG unlock now runs 3X–6X on RTX 40 (igor'sLAB Sep 14, corroborated by Tom's/Wccftech RTX 20/30 MFG mod pieces Sep 12–14).
- **KCD2's Daniel Vávra defends DLSS 5** (Sep 14): dev-side framing that NR "allows the intended aesthetics at target perf" — the pro-NR counterweight to the backlash; useful balance for any OSS positioning deck.
- **Steam Frame priced** (Sep 14): **$1,059 (1TB $1,299)**, reservation lottery live. Valve's standalone VR headset entering a market where every VR vendor runs fixed-foveation + upscaling — the vendor-neutral temporal-upscaler slot on PCVR remains open, and Steam Frame's PC-streaming mode is a natural OSS demo surface someday (no action; note-tier).
- **AMD:** no new official FSR/XeSS-class moves in the window. FSR 4 title count growing (007 First Light official FSR 4 support, GameGPU Sep 14). The AMD "Upscale Everything" dev article re-surfaced in search but dates to Jun 11 — background only. FSR Redstone SDK 2.2 (Mar 2026) remains the current shipped baseline: FSR Upscaling 4.1 ~1.3 ms at 4K-Perf on 9070 XT — the number to beat for #7 teacher comparisons.
- **Intel:** Panther Lake/Arc handheld wave continues (MSI Claw 8 EX AI+ cheaper variant confirmed, Sep 14; Minisforum M2 Pro with Arc B390 reviewed). No XeSS model/MFG changes from Intel itself; XeSS-MFG movement remains community-side (OptiScaler).
- **Apple:** MetalFX still silent through the A20 Pro cycle — vendor-neutral temporal-upscaler slot on Apple silicon empty for a fourth year.
- **Arm/Mali G2-Ultra NX (Dimensity 9600, reveal Sep 15 = today):** NFRU (Neural Frame Rate Upscaling) generates intermediate frames from MVs + scene data (30→60 FPS), alongside NSS (neural super resolution) on-GPU. The watch item from the morning brief comes due next run — after the reveal, do the exact-engine-MV-warping comparison vs OSS-FX.
- **DirectSR:** still silent across a 36+ day window — the vendor-neutral FG API slot remains empty.

## Action Items / Follow-ups

- [ ] **P0 (carried, worsening):** recover or formally stop `srcnn-v7.0-pico-005` — 365h frozen at 36,650; twelfth consecutive dead-trainer brief. Fix `data.json` `active: true` flag and the static-page "training now" copy to read from `status.json`.
- [ ] **P0 (carried, trivial):** commit the research brief series + `BIBLIOGRAPHY.md` + 207 MB `docs/research/papers/` (LFS or release attachment). Three months of provenance unpushed.
- [ ] **New: Neural-Upstream resolution-ordering note → H006 + positioning.** The modding stack's ~+60% win by running NR pre-upscale is independent empirical confirmation of the OSS input-domain thesis (networks at lowest resolution; rendering IS the upscaler). Write one paragraph into the H006/<2 ms-student evidence file; if NVIDIA adopts the ordering officially this fall, log it as a public concession datapoint.
- [ ] **New (positioning):** cite GameGPU's RTX 5060 77→35 FPS AC Mirage DLSS 5 number as the quantified cost-of-NR baseline in README/positioning materials — pairs with the BC-250 FSR-4 halving (Sep 14 morning brief) as cost-pressure bookends.
- [ ] **Carried: SuperSplat 3.0 → cross-vendor rasterizer design note** (GPU-compute projector + radix sort + indirect draw; WebGPU delivery path).
- [ ] **Carried: BC-250 FSR-4 RC9 datapoint → two notes** ((1) #7 RDNA 2 FSR 4.1.1 teacher capture proven portable + tunable; (2) H006 kernel-headroom evidence).
- [ ] **Carried: DLSSG-Stream-Worker eval** (offline D3D12 DLSS-FG teacher path for #7; verify claims before relying).
- [ ] **Carried: GSReuse → canvas-warp design note** (four reliability gates → v7 disocclusion/spawner decision; 2.24×/2.64 ms in #13).
- [ ] **Carried (watch): VS-Splat (2609.12343)** — voxel-selective primitive allocation; check selection gate → spawner density transfer if code releases.
- [ ] **Carried: OptiScaler XeSS MFG cross-vendor note** — capture an XeSS-MFG teacher signal when a pre-release tag lands (clean-room sequencing applies).
- [ ] **Due next run: Dimensity 9600 / Mali G2-Ultra NX deep-dive** (reveal day is Sep 15) — NFRU intermediate-frame generation from MVs + scene data vs OSS-FX α-sweep; NSS vs canvas SR; get exact on-GPU latency claims if published.
- [ ] **Watch: DLSS 5 RTX 40 official rollout "later this fall"** — re-check the <2 ms-student thesis framing when it ships; check whether official rollout adopts pre-upscale NR ordering.
- [ ] **Watch: SuperTimeGS** — if it produces rendered results, its analytic time-basis integration is directly comparable to the v7 canvas's native time axis.
- [ ] **Carried: P-GSVC deep read → #12 layered-canvas design note; #12 mechanism three-way ablation decision before freezing packed-canvas capacity; Laplacian-FH training control in the v7 ablation set; #7 clean-room sequencing; Qualcomm Adreno Neural Fusion competitive note.**
- [ ] **Infra (carried):** `.com` → `.org` 301 audit (README, docs, capture-tool installer).
- [ ] **Infra (carried, worsening):** Wayback availability API 429'd a **2nd consecutive run** (TimeTravel alternate egress also empty) — retry the deferred capture checks (SuperTimeGS, IE-SRGS, GRAPE/GSASR) next run, spaced ≥60 s or via a different egress.
- [ ] **Infra (resolved this run):** arXiv Atom probes dropped; listing-page extraction standardized (cs.CV 90/90 + cs.GR 42/42 covered with 3 requests). Keep this as the default sweep path.

**BIBLIOGRAPHY.md:** no new papers past the materiality bar this run — no append required. (DLSS5-Autopilot / Neural Upstream logged as watch-tier in this brief only.)

## Sources

1. `status.json` — https://opensupersampling.org/status.json (checked 2026-09-15T00:00Z fetch)
2. `data.json` — https://opensupersampling.org/data.json (checked 2026-09-15, generated_at 00:02:46Z)
3. GitHub Issues API — repos/cashcon57/open-supersampling/issues?state=open (checked 2026-09-15; 12 open)
4. Videocardz — DLSS 5 Autopilot / Neural Upstream: https://videocardz.com/newz/dlss-5-autopilot-brings-neural-rendering-to-more-games-with-lower-performance-cost (Sep 7, wave re-covered Sep 14); repos Kizzuwatnaa/DLSS5-Autopilot (618★) + matiasLombo/neural-upstream (103★)
5. GameGPU — DLSS 5 RTX 5060 AC Mirage 77→35 FPS: https://en.gamegpu.com/news/igry/dlss-5-snizil-proizvoditelnost-rtx-5060-s-77-do-35-fps-v-assassins- (Sep 14); TechSpot "two ways to make DLSS 5 playable" (Sep 14); TweakTown (Sep 14)
6. Dexerto/TheSixthAxis — Steam Frame $1,059 pricing (Sep 14)
7. Arm Mali G2-Ultra NX / NFRU coverage — gadgetsnow/techtimes (Sep 8–14); Dimensity 9600 reveal Sep 15
8. arXiv — cs.CV/recent (Mon-14 delta re-verified 90/90 via skip=50 page) + cs.GR/recent (42) listings (checked 2026-09-15); Atom API 429 (5th consecutive run)
9. GitHub REST — watched repos (OptiScaler 11,000★, dlss-unlocked 451★, SuperTimeGS, DLSSG-Stream-Worker, DLSS5-Autopilot, jyu041/dlss-rtxsr-upscaler, projectsaturnstudios/neural-upstream, mohmedmm/nbox) + 3 search lanes (checked 2026-09-15)
10. Google News RSS — DLSS 5 / FSR 4 / XeSS+Arc / MetalFX / gaussian splatting / frame generation lanes, when:3–5d (checked 2026-09-15)
11. archive.org Wayback availability API — 429 throttled (3 attempts) + TimeTravel Memento empty (1 attempt), 2026-09-15
