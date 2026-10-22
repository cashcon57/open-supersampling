# OSS Research Brief — 2026-10-21 (evening)

Delta-only vs the 2026-10-21 morning brief (and 10-19 / 10-20 briefs). Run at 20:00 EDT.

## Project Status Summary
- **Latest commits:** HEAD unchanged — `b44278a test: add local Chrome Playwright dashboard smoke` (2026-06-08). `main` == `origin/main`. Working tree still carries uncommitted/untracked briefs (2026-08-11 → 10-21) and modified `BIBLIOGRAPHY.md`.
- **Open issues (10, unchanged):** #4, #5, #6, #7, #8, #9, #10, #12, #13, #14.
- **Training dashboard (data.json generated 2026-10-22T00:00:57Z):**
  - `srcnn-v7.0-pico-005` (arch v7, "v7.0 Pico OSS-FX") **still at step 36,650 / 100,000**; latest loss total 0.1927 (sr_charbonnier 0.0236, sr_lpips 0.0927, fg_charbonnier 0.0764, canvas_count 2304, mean opacity 0.418). Trainer has now been dead ~140h (128h at this morning's check + 12h) — **~5.8 days**. `gpu_status: null`, last viz still step 36,000. `cost.usd` frozen at $0.92 / 15.27 GPU-h.
  - **New: `opensupersampling.com` now 301-redirects to `https://opensupersampling.org/`** (both `/` and `/data.json`). Canonical host is `.org`. The morning brief's "BibTeX cites .org but site is .com" note is therefore backwards — the BibTeX is right; README/dashboard links and any scripts that fetch `.com/data.json` without `-L` get an empty body (my first fetch returned 0 bytes). Update README links, the cron/watcher URLs, and CNAME notes to `.org`.
  - Dashboard HTML still carries copy that contradicts the stalled state (the "Status: Trainer … live / ETA" header while the health row is offline).

## New Papers (arXiv)
Tooling: `export.arxiv.org` API still 429 / connection failure today (fourth consecutive run), so the A–E keyword sweep was done via RSS (`rss.arxiv.org/atom/{cs.CV,cs.GR,eess.IV,cs.LG}`, 727 entries keyword-filtered + deduped against `BIBLIOGRAPHY.md` and all 2026-10 briefs) plus targeted web searches. The RSS announce list is the same 2610.2xxxx batch as this morning. **Nothing new and on-target.**

### Gaussian Splatting / Temporal
- Nothing new. Off-target items from today's list (not added): CrossFit-GS (2610.28190, COLMAP-free 3DGS), RayCo4D (2610.28351, monocular 4D reconstruction), PanoGraphGS (2610.27696), SwitchSplat (2610.24812v2), Stable Signer V3 (2610.24926v2, 4D deformable primitive splatting for sign avatars).
- Reminder: KoalaGS (2610.28615), RLSplat (2610.27712), HiQGS (2610.27604) from the morning brief remain the only relevant 3DGS reads this week.

### Super-Resolution / Upscaling
- **WorldSR — Harnessing World Knowledge Search for Grounded Image SR** (arXiv:2610.28423, 2026-10-21) — generative-prior (diffusion) SR with retrieval. Not applicable to a ≤1M CNN student; skip.
- Older-but-unlisted search hits worth a bibliography check (not new, not added today): *Wavelet-Space Super-Resolution for Real-Time Rendering* (arXiv:2508.16024) — wavelet feature decomposition in neural SR for rendering; a cheap-backbone idea relevant to #9/#10 and not yet in `BIBLIOGRAPHY.md`. `Stream-DiffVSR` (2512.23709) is diffusion-scale (0.33 s @ 720p on 4090), not relevant.

### Frame Extrapolation / Generation
- Nothing new. **Matrix-Game 4.0** (2610.27999, real-time playable world models) already tracked 10-19; not a competitor to SR/FG.

### Inference Optimization
- Nothing new. (2604.20291 INT8 SISR with RepVGG-style backbone + QAT on the fused deploy graph is already in the bibliography; its three-stage recipe — spatial → Charbonnier+DCT+teacher-distill → QAT on fused graph — is a ready template for #4/#9.)

## Archive.org Findings
- Advanced Search (2026-10-01 → 12-31) for `"gaussian splatting" super-resolution`, `"frame generation" deep learning`, `"real-time upscaling" game`: still returns only generic keyword-OR hits (query is parsed as `gaussian` / `frame` / `real-time` alone) — nothing of research value.
- **No takedowns.** Live `arxiv.org/abs/` returns HTTP 200 for GS-STVSR 2604.18047, GSASR 2501.06838, GRAPE 2503.06617. Wayback CDX captures: GS-STVSR 2026-04-21; GSASR 2026-03-21 and **2026-06-07 (digest changed → page updated, likely a new version/revision; worth a quick diff against our cached copy)**; GRAPE 2025-12-31 and 2026-02-11 (identical digest).
- arXiv blog notice: scheduled DB maintenance **Wed 2026-10-28, 07:00–12:00 EDT** — avoid the research run window that morning.

## GitHub / Open Source
- GitHub search (pushed since 10-07) for `gaussian splatting super-resolution`, `neural upscaler real-time game`, `2D gaussian splatting video`, `frame extrapolation`: no new relevant repos. Only `GiulioFede/ResonantBraneSplatting` (unchanged, 1★) for the GS+SR query.
- `frame generation open source` — only interpolation hobby projects (`afrizal1211/openfg-oss` created 10-20, `menghuihuan44/fluid`, `hy300leosquizz-ctrl/ZeroFG`, `PhoenixTheSage/SE-FG`); all previously noted, none material.
- Ecosystem signal continues: GitHub topic pages show `Merserk/dlss5-visual-enhancer` (1.5k★), `Dagherbou/OptiScaler_DLSSNR`, DLSS 5 neural-rendering-on-Radeon via solo-dev HIP kernels (TechTimes 10-20, same project family as `countervolts/d4r`) — all about injecting/porting proprietary models, none are open vendor-neutral architectures.
- **Unity (new, integration-relevant):** Unity Discussions thread (2026-10-08, still active) documents a **hidden "Upscaler Framework" in URP (6.7-beta / 7.0-alpha)**: enable `ENABLE_UPSCALER_FRAMEWORK` + `ENABLE_NVIDIA` (DLSS 4) / `ENABLE_AMD` (FSR 2–4), plus a MetalFX package; URP asset then selects upscaler and supersampling with prioritisation. *Relevance:* Unity is building a first-party multi-vendor upscaler abstraction. An OSS Unity path should plug into this framework (as a fourth backend) rather than ship a separate plugin; check for a public API/interface once 6.7 LTS lands.

## News
- **NVIDIA:** NVIDIA's own 10-13 post ("DLSS 5 comes to Hogwarts Legacy & RTX Spark, powered by 2× faster model") states a **2× faster** DLSS 5 3D-guided neural rendering model; the weekly DLSS post cited this morning describes ~10× vs the March model (cumulative). Both are NVIDIA claims — don't quote "10×" without the baseline. Schedule: **RTX Remix DLSS 5 update Oct 23**, CoD MW4 launches Oct 23 with DLSS 4.5 (Switch 2 build also ships **NVIDIA DLSS** — Digital Foundry says it looks better than Filmic SMAA T2X), **Cyberpunk 2077 DLSS 4.5 update Oct 22**, Phantom Blade Zero Oct 28 (DLSS 4.5 + path tracing), Hogwarts Legacy DLSS 5 Nov 5. *Relevance:* Switch 2 DLSS is a vendor neural upscaler on a very low-power GPU (frame-time cost not published in sources I found) — a reference point for the Pico tier (Steam Deck / handheld target) and evidence that vendor-trained small models already ship on handheld-class hardware.
- **AMD:** no new items beyond this morning (FSR 4.1 INT8 expansion, Vulkan support promised, FG pacing, handheld lightweight model by end of 2026). Windows Central / Digital Foundry coverage of the Huynh "lightweight FSR 4" interview confirms scope: APUs, laptops, handhelds.
- **Intel:** XeSS 3 MFG rolled out across Arc (PCGuide test coverage); XMX-under-Proton PRs (filed 10-19) all still open. Intel Unity XeSS plugin was discontinued earlier (TechPowerUp) — relevant: a vendor retreated from Unity, leaving room for an engine-agnostic open option, but also showing plugin-maintenance cost.
- **Apple/Metal:** no new info today (MetalFX redesigned temporal upscaler on Neural Engine + Neural Accelerators, M5 Pro/Max; already covered).
- **Sony:** PS5 system update 26.0x shipped 10-21 (display names/trending UI) — no PSSR/FSR 4 news found today; the FSR 4 integration for PS5 Pro and FGSS patent items remain as carried.
- **Engines:** nothing new for Unreal 5.8 beyond the morning brief; Unity item above is the new one.

## Action Items / Follow-ups
- [ ] **Trainer stalled ~140h at step 36,650** — restart/park pico-005; set the dashboard header to "stalled/paused" when the watcher flags offline instead of "live / ETA".
- [ ] **Domain move:** `.com` → `.org` 301. Update README badges/links, `docs/` references, watcher/cron fetch URLs (use `curl -L` or hit `.org` directly), and decide whether `.com` stays as a redirect permanently.
- [ ] **GSASR (2501.06838) Wayback digest changed on 2026-06-07** — diff current abs page / versions list against our `docs/research/papers/` copy for a v2/v3 revision.
- [ ] **Unity upscaler framework (URP 6.7/7.0):** find the interface (`ENABLE_UPSCALER_FRAMEWORK`) and note whether third-party backends can register; add to #8 / integration plan.
- [ ] **Pico-tier benchmark note:** record DLSS-on-Switch-2 (MW4) as a ship-class reference for the <2 ms Pico budget; seek Digital Foundry frame-time numbers.
- [ ] Add *Wavelet-Space SR for Real-Time Rendering* (2508.16024) to `BIBLIOGRAPHY.md`/read for #9/#10 (backbone cost vs quality).
- [ ] Carried from morning: read KoalaGS + RLSplat vs H004/#12; #7/#8 `fsr4-shader-performance-overrides` + `d4r` kernel references and teacher-extraction legal stance; README "separate SR + FG stages" wording; OSS-FX multi-α + frame-pacing positioning.
- [ ] Watch: Cyberpunk DLSS 4.5 (10-22), CoD MW4 + RTX Remix DLSS 5 (10-23), ASUS ROG Xbox Ally FSR 4.1 preview (10-23), Phantom Blade Zero (10-28), arXiv API maintenance (10-28), Hogwarts Legacy DLSS 5 (11-05).
- [ ] **Tooling:** arXiv API 429 now persistent for 4+ runs — build the throttled/cached fetcher (≥3 s spacing, single connection, descriptive UA) or switch the sweep to arXiv RSS + Semantic Scholar/OpenAlex.

## Sources
- OSS dashboard data: `https://opensupersampling.org/data.json` (generated 2026-10-22T00:00:57Z); `.com` redirect verified via `curl -i`.
- [NVIDIA — DLSS 5 comes to Hogwarts Legacy & RTX Spark (2× faster model)](https://www.nvidia.com/en-us/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/) · [NVIDIA — MW4 / Cyberpunk 2077 DLSS 4.5 (2026-10-21)](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-cyberpunk-2077-planet-zoo-2-modern-warfare-4/) · [TweakTown — MW4 DLSS on Switch 2](https://www.tweaktown.com/news/114161/modern-warfare-4-brings-nvidia-dlss-to-switch-2-and-digital-foundry-says-it-looks-better-than-smaa/)
- [Unity Discussions — Unified upscalers DLSS4 & FSR4 in URP (hidden preview)](https://discussions.unity.com/t/dlss4-fsr4-in-unity-urp-hidden-preview/1738596) · [TechPowerUp — Intel drops XeSS for Unity](https://www.techpowerup.com/348014/intel-pulls-the-plug-on-xess-support-in-unity-game-engine) · [PCGuide — XeSS 3 MFG on Arc](https://www.pcguide.com/news/xess-3-multi-frame-gen-tested-on-two-generations-of-flagship-arc-gpus-after-intel-rolls-out-support/)
- [Windows Central — lightweight FSR 4](https://www.windowscentral.com/hardware/amd/amd-fsr-4-gaming-handhelds-apu) · [Digital Foundry — Huynh FSR 4 handhelds](https://www.digitalfoundry.net/news/2026/10/amds-jack-huynh-fsr-4-is-coming-to-handhelds-via-a-new-lightweight-model) · [TechTimes — DLSS 5 on Radeon](https://www.techtimes.com/articles/329228/20261020/dlss-5-neural-rendering-now-runs-three-generations-amd-radeon-gpus.htm)
- arXiv RSS (cs.CV/cs.GR/eess.IV/cs.LG) 2026-10-21; arXiv abs 2604.20291, 2610.28423, 2508.16024; web.archive.org CDX (2604.18047, 2501.06838, 2503.06617); archive.org advancedsearch; GitHub search API (pushed:>2026-10-07).
