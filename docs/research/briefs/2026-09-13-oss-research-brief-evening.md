# OSS Research Brief — 2026-09-13 (evening)

> Evening run (written ~00:15 UTC Sep 14, scanning as the Sep 13 evening delta; follows the ~00:15 UTC and ~12:10 UTC briefs same day). Headlines: **GSReuse PDF acquired and deep-read** — full method, reliability gates, and speedup numbers extracted from the TVCG author manuscript (closes the midday's "PDF acquisition pending" item; mirrors now total 4 files / ~207 MB). **OptiScaler XeSS MFG unlocked cross-vendor via DP4a** (GameGPU, Sep 13) — Intel's multi-frame generator now runs on Radeon RX 6000/7000/9000 and GeForce RTX 20/30/40; all three vendor FG stacks are now demonstrably software-unlocked from their silicon, completing the trifecta behind the "integration shims are commodity" positioning. **SuperTimeGS** — new 0★ AGPL repo (Sep 12) expressing 4DGS attribute time-evolution as analytic temporal-Gaussian-basis superpositions; watch-tier but the closest hobbyist artifact yet to the v7 native-time-axis idea. **No new arXiv papers past the materiality bar** (arXiv Atom 429 again; listing fallback: 200 newest cs.CV entries spanning announcements Fri Sep 11 → Tue Sep 8 scanned, cs.GR 46-entry set unchanged, HF dailies lagging). Trainer still dead: **341h frozen at step 36,650** (tenth consecutive dead-trainer brief).

## Project Status Summary

- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` — unchanged since 2026-06-08 (97 days). Uncommitted tree unchanged in kind; this run adds the evening brief + the GSReuse PDF mirror (`docs/research/papers/`, now 4 PDFs / ~207 MB).
- **Open issues:** **12 open (#1–#14 minus gaps)**, unchanged set, all last touched 2026-05-08. Correction to the midday brief, which undercounted as 10 — the API returns 12 (two issues, #13/#14, were missed in its dedupe). Tracker static either way.
- **Training dashboard:** `status.json` (checked 2026-09-14T00:01Z): trainer **offline, stuck at step 36650 for 341h** — 329h → 341h, consistent ~12h drift; counter honest, run dead. Watcher healthy (published 1s ago), CF Worker healthy (62 ms), R2 healthy. `data.json` cross-check: `srcnn-v7.0-pico-005` still `active: true`, still the only active run, headline copy still "training now" — same contradiction as every prior brief. Cost projections (RunPod rates) still displayed for 4090/A100/H100/B200.
- **Domain migration:** `opensupersampling.com/status.json` → **301 → `.org`** (re-verified this run). `.com` → `.org` audit remains open.
- **Operational conclusion:** unchanged — publishing infra green, training dead ~14 days. No "training now" framing in external comms until the 3080 Ti host is recovered or the run is formally stopped.

## New Papers (arXiv)

**Sweep health:** arXiv Atom API throttled again (HTTP 429 `Rate exceeded.` on probe — same failure mode as the two prior runs). Fallback stack executed: **cs.CV recent listing pages scanned 200 entries deep (announcement dates Fri Sep 11 back through Tue Sep 8; ids 2609.05959–2609.11929)** — the post-midday delta is the 50-entry Fri-11 page, and it was scanned in full; cs.GR recent listing re-pulled (46 ids, keyword-scanned; no new GS-SR/temporal entries vs the midday id set); HF daily-papers API for Sep 12–13 returns 0 both days (feed lag, consistent with midday's finding; default endpoint returns a stale window whose only in-lane hit is OracleZoom, already rejected).

### Gaussian Splatting / Temporal

- **Nothing new on arXiv past the materiality bar.** cs.GR newcomers this window were screened and rejected: *Gaussian Light Transport* (2609.11430 — 13D Gaussian-mixture global illumination, offline rendering-equation solve; validates the N-D-Gaussian trend but is an offline GI representation, not a canvas/bandwidth/rasterization donor) and *Neural Centroidal Voronoi Tessellations* (2609.08497 — mesh-sampling solver; same CVT acronym family as CVT-GS but a different problem domain). Neither appended.
- **Non-arXiv, resolved this run: GSReuse** (IEEE TVCG 33(5), DOI 10.1109/TVCG.2026.3679889) — author PDF found on `letianhuang.github.io` and mirrored locally. Deep-read summary (replaces the midday abstract-level entry; see Action Items):
  - **Method is purely geometric — no learned network.** GS-MVs are computed per-pixel from the alpha-weighted depth map + camera poses (back-project → reproject → 2D displacement). Reusable-tile decisions come from a *morphological-closing* pixel mask (cross-shaped 4-connected kernel, deliberately conservative), a tile-level all-pixels-valid mask (16×16 tiles), max-depth hole filling, a depth-guided bilateral color fill, and an **occlusion-aware composition** that resolves multi-source conflicts by depth-tested atomicMin — this stack *is* their reliability gate for reuse-vs-re-render.
  - **Incremental rendering** (n reused frames per keyframe): speedup S ≈ (1/a + n)/(1 + n/a) with a = re-rendered fraction; max S → 1+n as a→0. Their error-control policy — **drop the previously inpainted pixels before the next reuse step** — is exactly the canvas's per-step spawner-trust question.
  - **Numbers (RTX 4090, 1600×1000):** 2.24× avg speedup on stock 3DGS (4.5 → 2.0 ms/frame), ~1.8× combined with LightGaussian/Taming-3DGS/Mini-Splatting, 1.6× on CityGaussian-v2 large scenes; generated-frame cost **2.64 ms** vs RIFE 9.89 / DMVFN 20.78 / IFRNet 27.22 ms; beats all three NN baselines on PSNR+SSIM+LPIPS in 8/9 Mip-NeRF360 scenes. Stage breakdown: compositing ~70%, sorting ~20% of baseline frame time — a ready-made frame-time accounting reference for #13.
  - **OSS mapping:** (a) their screen-space MV warp + gated reuse is the runtime-side precedent for the v7 canvas update rule — but they *replace* rasterization work, whereas OSS warps the canvas and re-rasterizes through a covariance step; (b) their depth-accuracy sensitivity (2DGS depth visibly better than 3DGS depth) supports OSS's covariance-resampled-depth design; (c) their limitations section (fast motion/parallax collapse reuse; dense Gaussian tiles resist reuse) reads as a list of the exact cases OSS's engine-MV + spawner path is designed to handle. Independent-group convergence on "MV-warped splat reuse with a trust gate" is now established from two sides.

### Super-Resolution / Upscaling

- **Nothing new past the bar.** Newest 200 cs.CV entries contain no SR/upscaling work beyond known rejections (OracleZoom re-surfaced via HF, still offline-quality-class; CEM-TUDASR is domain-adaptive medical-class SR).

### Frame Extrapolation / Generation

- **Nothing new past the bar.** Known set unchanged: Amulet (2608.10423, only architectural threat, mirrored), X4Rate, GFFE, PatchEX, STSS. No new `frame extrapolation`/`frame generation` titles in the scanned window.

### Inference Optimization

- **Nothing new past the bar.** INT8 SISR deployment paper (2604.20291, CVPRW) remains the reference; TensorRT 10.16 still current; EFQ-Softmax tracked 09-11.

## Archive.org Findings

- **GSReuse author PDF:** Wayback holds a single capture (`letianhuang.github.io/assets/pdf/GSReuse.pdf`, 2026-08-14, HTTP 200). Now mirrored locally regardless — `docs/research/papers/GSReuse-TA-SSR-3DGS-TVDG-2026.pdf`, 51,720,567 bytes, sha256 `eaed57becc2216ec50bf74381611fd2e4db45f0bac040789c700535aa5de92ee`, verified as the TVCG author manuscript (DOI 10.1109/TVCG.2026.3679889 on page 1). Risk closed.
- **SuperTimeGS repo:** zero Wayback captures (repo created 2026-09-12). No archival risk yet; re-check only if it shows material progress.
- GS-STVSR / Amulet local mirrors unchanged (risk previously closed); GRAPE / GSASR capture states unchanged from midday.
- All four local PDF mirrors (155 → **207 MB**) remain **uncommitted** — same commit blocker as prior briefs.

## GitHub / Open Source

- **OptiScaler (optiscaler/OptiScaler) — 10,799 → 10,903★ (+104 in ~12h), pushed 2026-09-13 20:49Z (this run).** Heavy main-branch churn (swapchain SRGB handling, allocation-freeze fixes, Dx11 flush removal) but **no new tag — v0.9.4 (2026-07-18) still latest**. The +104★ day and the XeSS-MFG headline below make this the hottest the modding layer has been since the DLSS-5 unlock wave.
- **OptiScaler XeSS MFG cross-vendor — the run's strategic item.** GameGPU (Sep 13, YouTube-sourced): a pre-release OptiScaler with modified XeSS libraries runs XeSS multi-frame generation on **Radeon RX 6000/7000/9000 and GeForce RTX 20/30/40** by using **DP4a instructions instead of XMX matrix units**; works in FSR 3.1/DLSS-enabled titles, borderless-window required; tested 40→115–138 FPS at 4K on RX 9070 GRE; >3× multipliers show bottom-screen artifacts (fixed by VSync) and rising input lag. Notable detail: their XeSS FG carries **UI-composition interpolation** (HUD elements stay at panel refresh while the world interpolates) — an attention-to-compositor-correctness detail OSS's DLL-shim path will also need. Every vendor's FG is now software-unlockable cross-vendor (DLSS 5 → RTX 20/30/40; MFG → Ampere/Turing; FSR 4.1.1b → RDNA 2; XeSS MFG → everything); hardware gating is repeatedly demonstrated to be a policy boundary, not a capability one.
- **DLSS Unlocked (ShyVortex/dlss-unlocked) — 393 → 402★ (+9), pushed 2026-09-13 21:00Z.** New commits: rebased onto the author's own OptiScaler fork + installer-build fixes (release-sorted YAML, `/releases/latest` query). Consolidation of the two big mod tools into one lineage continues; plateau decay continues (8 → 9★/12h ≈ steady).
- **DLSS5VKLayer (bmitch87/DLSS5VKLayer) — 71 → 80★, push unchanged 2026-09-13 00:09Z.** Linux DLSS 5 VKLayer discovery riding the news cycle.
- **NVOFG (SimonVitzethum/NVOFG) — 1★ unchanged, pushed 2026-09-12 19:43Z.** Linux Vulkan FG middleware; watch stands.
- **NEW: SuperTimeGS (kyo-kuroki/SuperTimeGS) — 0★, created 2026-09-12, AGPL-3.0.** Solo Japanese-language research project: expresses 4DGS attribute time-evolution (opacity, motion, scale, rotation; SH kept static) as a **linear superposition of K analytic temporal Gaussian bases g_k(t) = exp(−(t−μ_k)²/2σ_k²)**, with velocity- or acceleration-mode motion integrated **analytically** (closed-form trajectories, no numerical integration). Watch-tier, but notable: this is a hobbyist independently converging on *Gaussian time-basis functions* for dynamic splat representation — the same "time lives inside the primitive" design space as the v7 native-time-axis canvas (there: per-splat temporal basis mixture; OSS: canvas-level N-D Gaussians). Cite as community-adjacent precedent if it produces results.
- **FidelityFX SDK: v2.3.0 (2026-06-24) confirmed unchanged** — no AMD SDK movement since June.
- **Mob-FGSR/MobFGSR — 112★, pushed 2025-06-11** — unchanged; #5 No-ML baseline reference stands.
- Repo searches (GS+SR, frame-gen upscaling, neural upscaler, GS+temporal; unquoted queries, sorted by push): nothing above the known noise set. New noise-tier 0★/1★ leads: **mohmedmm/nbox** (MIT, "subtle neural video enhancer 240p→2K/4K, zero CPU load"), **jyu041/dlss-rtxsr-upscaler** (RTX VSR + standalone DLSS SR + experimental DLSS 5 NR for local video), **Bharath-vip/dlss5-studio** ("DLSS 5 Neural Reconstruction" video workstation), plus prior leads neural-upstream / NeuralSwap still noise-tier. Consumption-side DLSS 5 tooling is proliferating; nothing in OSS's render-side lane.

## News

- **OptiScaler XeSS MFG cross-vendor** (see GitHub section) — promoted to verified after reading the GameGPU article body, not just the headline. Completes the four-vendor unlock matrix; strengthens the #7 teacher-capture plan: an XeSS FG teacher signal is now capturable on non-Arc GPUs.
- **DLSS 5 narrative widened again:** PCWorld (Sep 11, "I was an NVIDIA DLSS 5 hater… until I used it myself") joins the PC Gamer/PCMag/TechPowerUp first-touch wave; GameGPU's Sep 13 feed shows DLSS 5 benchmarks across RTX 20 → 5090 and multiple titles, including **DLSS 5 *losing* FPS on RTX 5080 in Cyberpunk 2077 full PT (55→30)** and NBA 2K27's "50% frame hit" coverage — the NR-cost narrative is now quantified in mainstream coverage, which is the exact cost-pressure the <2 ms student thesis argues against. DLSS 5 for RTX 40 "later this fall" (carried from midday, official).
- **Qualcomm Adreno Neural Fusion** (carried): ServeTheHome ran a next-gen Adreno Neural Fusion deep-dive Sep 11; no new primary Qualcomm detail beyond the Sep 2–3 newsroom material. Watch item unchanged.
- **MediaTek Dimensity 9600 reveal Sep 15 confirmed** (carried; 2 days out). Mali G2-Ultra NX NFRU deep-dive queued for the day after.
- **Intel:** no XeSS model/MFG changes from Intel itself; the only XeSS-MFG movement is community-side (OptiScaler DP4a, above). Panther Lake Arc B390 handheld chatter continues.
- **Apple:** iPhone 18 Pro launch window (preorders Sep 12, launch Sep 18); **MetalFX still silent** — no temporal-upscaler changes announced with the phone.
- **DirectSR:** still silent across a 30+ day window — the vendor-neutral FG API slot remains empty.

## Action Items / Follow-ups

- [ ] **P0 (carried, worsening):** recover or formally stop `srcnn-v7.0-pico-005` — 341h frozen at 36,650; tenth consecutive dead-trainer brief. Fix `data.json` `active: true` flag and the static-page "training now" copy to read from `status.json`.
- [ ] **P0 (carried, trivial):** commit the research brief series + `BIBLIOGRAPHY.md` + 207 MB `docs/research/papers/` (LFS or release attachment). Three months of provenance unpushed.
- [x] **GSReuse PDF acquired + deep read (this run):** author PDF mirrored (52 MB, sha256 verified, DOI 10.1109/TVCG.2026.3679889); method, reliability gates, speedup numbers extracted (see New Papers section). **Next:** write the canvas-warp design note mapping their four reliability gates (morphological closing mask, tile-validity mask, occlusion-tested composition, drop-inpainted-pixels-before-reuse) onto the v7 disocclusion/spawner decision, and cite their 2.24×/2.64 ms numbers in #13 frame-time accounting alongside Splats-to-Silicon/TileGS. Their purely-geometric (non-learned) MV path is also a candidate ablation baseline for the canvas update rule.
- [ ] **New: OptiScaler XeSS MFG cross-vendor note.** When the pre-release lands in an OptiScaler tag, capture an XeSS-MFG teacher signal on RTX/Radeon hardware for #7 (clean-room sequencing still applies: capture teacher outputs before reading any released model code). Their UI-composition interpolation detail belongs in the OSS-FX compositor-correctness notes.
- [ ] **Carried: Qualcomm Adreno Neural Fusion competitive note** — compare against OSS-FX when full tech details land (engine-MV vs optical-flow; Matrix Core INT8 claims vs Pico-tier student assumptions).
- [ ] **Watch (2 days out): MediaTek Dimensity 9600 reveal Sep 15** — Mali G2-Ultra NX NFRU deep-dive after; compare exact-engine-MV warping claims vs OSS-FX.
- [ ] **Watch: DLSS 5 RTX 40 official rollout "later this fall"** — widens the NR-cost narrative to mid-tier hardware; re-check the <2 ms-student thesis framing when it ships.
- [ ] **Watch: SuperTimeGS** — hobbyist temporal-Gaussian-basis 4DGS project; if it produces rendered results, its analytic time-basis integration is directly comparable to the v7 canvas's native time axis.
- [ ] **Carried: P-GSVC deep read → #12 layered-canvas design note** (base+enhancement split policy, cross-layer optimization mechanics, progressive quality/cost curve).
- [ ] **Carried: #12 mechanism three-way** — LinearMask-GS vs Compact-FFGS vs CVT-GS; schedule ablation decision before freezing packed-canvas capacity (temporal-identity + L2-residency gates).
- [ ] **Carried: Laplacian-FH training control** — frequency-staged growth + field archival in the v7 ablation set.
- [ ] **Carried: #7 clean-room sequencing** — FSR 4.1.1b on RDNA 2 via DLL swap; teacher-signal capture before reading any released model code.
- [ ] **Infra (carried):** `.com` → `.org` 301 audit (README, docs, capture-tool installer).
- [x] **Correction logged (this run):** midday brief's open-issue count was 10; the API returns **12** open (#1–#14 minus gaps). No tracker change; count corrected here and in future briefs.
- [ ] **Infra note (carried):** arXiv Atom throttled a third consecutive run (429 on probe). Listing-page + abs-extraction + HF-dailies fallback remains the default; consider a slower cadence or id_list batches next run if the 429s persist.

## Sources

1. `status.json` — https://opensupersampling.org/status.json (checked 2026-09-14T00:01:34Z)
2. `data.json` — https://opensupersampling.org/data.json (checked 2026-09-14T00:05Z)
3. GitHub Issues API — repos/cashcon57/open-supersampling/issues?state=open (checked 2026-09-14T00:04Z; 12 open)
4. GSReuse author PDF — https://letianhuang.github.io/assets/pdf/GSReuse.pdf (mirrored + deep-read this run; Wayback capture 2026-08-14)
5. GameGPU — OptiScaler XeSS MFG cross-vendor: https://en.gamegpu.com/news/igry/optiscaler-dobavil-intel-xess-multi-frame-generation-dlya-amd-radeon-i-nvidia-geforce (Sep 13; YouTube-sourced, primary mod not yet released)
6. GitHub REST — watched repos + 4 search lanes (checked 2026-09-14T00:06–00:09Z)
7. Google News RSS — DLSS 5 / FSR 4 / XeSS / MetalFX / frame generation / Adreno Neural Fusion / Dimensity 9600 lanes, when:14d (checked 2026-09-14T00:08Z)
8. arXiv listings — cs.CV/recent (200 entries, Fri Sep 11 → Tue Sep 8) + cs.GR/recent (46 entries) (checked 2026-09-14T00:03Z)
9. Hugging Face daily-papers API — Sep 12/13 (0 entries both), default endpoint (checked 2026-09-14T00:07Z)
