# OSS Research Brief — 2026-10-18

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` — unchanged since 2026-06-08 (shallow clone, nothing new). Working tree still carries uncommitted research briefs (2026-08-11 → 10-17) and a modified `BIBLIOGRAPHY.md`.
- **Open issues (10, unchanged):** #14 D3 VRAM footprint, #13 Nsight L2 profile (H004), #12 packed low-rank canvas layout, #10 decoupled G-buffer encoder, #9 RepVGG/ECBSR/RepSR student backbone, #8 cross-vendor kernel ports, #7 FSR 4 binary as teacher signal, #6 pico-002 launch, #5 No-ML shader variant, #4 HAT-Tiny student distillation.
- **README:** unchanged. v5-pixel-temporal still the only measured row (25.703 dB / 0.1666 LPIPS, TartanAir `oldtown`); the ≤1M student is untrained.
- **Training dashboard (opensupersampling.com, browser-rendered 08:00 EDT):** `srcnn-v7.0-pico-005`, step 36,650 / 100,000, loss 0.19268, viz @ 36,000 — identical to every run since 10-14. Health row: **"Trainer — offline, stuck at step 36650 for 68h"** (44h at 10-17 20:00 → 56h at 10-18 08:00 → 68h at 10-18 20:00; counter is consistent, +12h per 12h, so the trainer is genuinely not advancing). Watcher healthy (published 1s ago), CF Worker HTTP 200 88 ms, R2 healthy. Header still shows **"live"**, "updated just now", and **"ETA ~1.1d (estimate)"** — contradicts the offline state (ops/UX issue, unchanged).

## New Papers (arXiv)
Tooling note: `https://export.arxiv.org` Atom API worked this run (six fielded queries, 8 s spacing, 64 unique hits; `http://` still redirects). Filtered locally against `BIBLIOGRAPHY.md` and prior briefs. Sunday: no new listing since Fri 10-16, so the scan found **no new on-target papers**.

### Gaussian Splatting / Temporal
- Nothing new on target. Already tracked and unchanged: ResStreamGS (2610.21121), TETRIS-4DGS (2610.14663), SparseView4DGS (2610.04606), GS-STVSR (2604.18047).
- Newly surfaced, **not material** (title-level only; not added to bibliography): **Make It Flow** (arXiv:2610.13860, Keune/Hahlbohm/Möller, 2026-10-09 — water animation in 3DGS scenes), **PUGS** (2610.17936, underwater dynamic 3DGS), **Edit4D** (2610.21668), **Demiurge** (2610.14685, autoregressive video rerendering for 4D worlds), **OuroWorld** (2610.12461).

### Super-Resolution / Upscaling
- Nothing new. Top reads remain **Frozen History** (2610.15752), FlashVSR+ (2610.17552, tracked, diffusion/off-budget), 2610.13179 and 2610.11746 (generative VSR, off-budget), 2610.09098 (640-KB MCU SR).

### Frame Extrapolation / Generation
- Nothing new on arXiv. See News for the Sony patent (non-arXiv prior art on input-conditioned frame prediction).

### Inference Optimization
- Nothing new. Carried skims: QuantSR+ (2605.22351) and Token-Group Activation Quantization (2610.16261) for the static-calibration low-bit recipe (#9/#4).

## Archive.org Findings
- Advanced Search: `"gaussian splatting" super-resolution` → 3 hits, all `github.com-*-starred` GitHub-mirror archives from 2025 (noise); `"frame generation" deep learning` → 3 hits (an OSF registration for an unrelated video-diffusion paper, `dlssswapperbuildswin` "DLSS Swapper builds for Windows", `fluidframeswin` FluidFrames builds — consumer-tool archives, no research value); `"real-time upscaling" game` → 0 hits.
- Wayback availability for GS-STVSR (2604.18047), GSASR (2501.06838), GRAPE (2503.06617): API returned **HTTP 429** this run (not a negative finding). All three arXiv abs pages still resolve **HTTP 200 live → no takedowns**. Save Page Now capture still recommended (carried).

## GitHub / Open Source
(REST search, unquoted queries, `pushed:>=2026-10-04`.)
- **No new Gaussian-splatting SR / VSR / frame-extrapolation repos.** Only `GiulioFede/ResonantBraneSplatting` (1★, "quantum" 2D-GS claim; ignore).
- Frame-gen / upscaler activity — all wrappers or hobby projects, none a threat: `ShyVortex/dlss-unlocked` (847★, MIT, pushed 10-17), `Echo-Storm/ls-addon-manager` (187★), `Denarmo/HelixSR-Manager` (13★; HelixSR = DLSS-3.7-class SR for Steam Deck/Linux — a community-trained SR model, worth a look as a small real-time SR reference), `hy300leosquizz-ctrl/ZeroFG` (9★, Apache-2.0, Vulkan mobile FG), `menghuihuan44/fluid` (0★, AGPL), `PhoenixTheSage/SE-FG` (Space Engineers FG), `DerAlexmann/Silberkorn` / `Brahma-Upscale` / `nimishnmn/Anime-Upscaler` (generic image/video enhancers).
- Vendor blogs: no new primary-page checks beyond those in the 10-17 brief (NVIDIA DLSS 5 faster-model post 10-13, AMD "Upscale Everything" article, Intel XeSS SDK).

## News
- **Sony / PlayStation — new (Oct 15 filing, via Patentlyze → Wccftech 10-16, ixbt 10-17):** patent for "Frame Generation Super Sampling": encoder takes a rendered frame, optional motion vectors and **controller input**, decoder emits a predicted *future* synthetic frame before the next real frame; generates multiple candidate futures (rollback-netcode analogy); non-AI "Algorithmic Frame Generation" fallback using motion vectors/previous frames. Patent-level only, no product announcement. **Relevance:** direct validation (and IP-landscape note) for the OSS-FX extrapolation path; input-conditioned prediction + a non-ML fallback is conceptually adjacent to OSS's no-ML shader variant (#5). Check claims before OSS documents input-conditioning or multi-candidate futures as novel.
- **AMD — headline-level lead (GameGPU 10-16):** "FSR 4 Lite" being *considered* for RDNA 2 / Steam Machine / handhelds: truncated net, aggressive weight quantization, filter-pass merging, **DP4a integer path**; official RDNA 2 FSR 4 support rumored early 2027. Unconfirmed by AMD. Also 10-16: "FSR 4 on handhelds by Dec 31" (tech-insider, unverified). **Relevance:** if real, converges on OSS's cross-vendor DP4a/INT8 small-student design point (#4, #8); the bar for "cheap" neural SR on non-matrix hardware moves.
- **NVIDIA:** DLSS 5 faster model (10-13, primary NVIDIA page + TechPowerUp/OC3D/Gizmodo): ~2× faster, −20% 4K VRAM. PC Gamer 10-15: NBA 2K27 quality skepticism. Modders ported DLSS Frame Gen Preset C to RTX 30 (TechPowerUp 10-16) and Universal MFG Unlocker 1.5.0 adds XeSS MFG on AMD/NVIDIA (10-13) — community continues to route around vendor gating.
- **Intel:** nothing new beyond 10-17 (XeSS 3 MFG, UE plugin 3.1).
- **Apple:** only Apple ML Research "Minions" (idle-Mac GPU cloud, 10-13) — unrelated to MetalFX.
- **Engines:** nothing new (UE 5.8 plugin coverage unchanged; Unity 6.3 LTS).
- AMD RDNA 5 reveal not expected before Computex 2027 (VideoCardz leak, 10-12).

## Action Items / Follow-ups
- [ ] **Dashboard/trainer (highest priority):** pico-005 stalled at step 36,650 for **68h** and rising. Check trainer process / GPU / rsync on the 3080 Ti host; resume, or mark the run parked and remove the "live" badge and "ETA ~1.1d" banner while offline.
- [ ] Review Sony patent claims (search Patentlyze / USPTO publication for the Oct 15 filing) vs. OSS-FX α<1 + no-ML fallback design; note under H009/issue #5 if relevant.
- [ ] Watch AMD "FSR 4 Lite" (DP4a/INT8, truncated net) — if confirmed, update competitor section and #8 portability plan; unverified for now.
- [ ] Deep-read **Frozen History** (2610.15752) for H009 (carried).
- [ ] Read **OpenDLSS-NR** (Vulkan coop-matrix/FP8) → note under #8; read AMD "Upscale Everything" (carried).
- [ ] Skim QuantSR+ and Token-Group Quant (2610.16261) for #9/#4; ResStreamGS → H004/#12; optiscaler-dlss5 `edit-cache.md` (carried).
- [ ] Look at HelixSR / `mpv-AnimeJaNai` as small real-time SR model-size/latency references (#4).
- [ ] Optional: Wayback Save Page Now for GS-STVSR / GSASR / GRAPE abs pages (Wayback availability API 429'd this run).

## Evening refresh — 2026-10-18 20:00 EDT (second run today; deltas only)
- **Dashboard (browser-rendered):** unchanged step/loss — `srcnn-v7.0-pico-005`, 36,650 / 100,000, loss 0.19268, viz @ 36,000. Health row: "Trainer — offline, stuck at step 36650 for **68h**" (counter +12h per 12h, consistent). Watcher healthy (1s), CF Worker HTTP 200 125 ms, R2 healthy. Header still "live" + "ETA ~1.1d (estimate)" — contradiction persists.
- **Repo/issues:** HEAD still `b44278a`; same 10 open issues.
- **arXiv:** Atom API heavily throttled (429/timeouts on 5 of 6 queries); the one query that succeeded (VSR + temporal) returned only already-tracked or off-budget diffusion VSR papers. `arxiv.org/list/cs.CV/new` still shows the Fri 16 Oct listing → **no new announcements since the morning scan**. No new on-target papers; no bibliography additions from arXiv.
- **Archive.org:** same results as morning (GitHub-star mirrors, DLSS Swapper/FluidFrames builds, 0 hits for "real-time upscaling" game). GS-STVSR (2604.18047), GSASR (2501.06838), GRAPE (2503.06617) abs pages all still HTTP 200 live → no takedowns.
- **NEW — DLSS-NR-on-AMD 0.8.0 (VideoCardz 10-18, primary repo checked):** [danielblnc/DLSS-NR-on-AMD](https://github.com/danielblnc/DLSS-NR-on-AMD) (2,681★, license NOASSERTION, created 2026-09-03, pushed 10-17) now runs **Ray Reconstruction (Model F) and DLSS Frame Generation / MFG up to 4×** natively via custom **HIP compute kernels + DX12**, on top of Super Resolution (Models M/L) and DLSS 5 Neural Rendering — i.e. all four main DLSS features on Radeon RX 7000/9000 (RX 6000 NR only, testing). Reported: +25% NR speed on RDNA 3; ~90–100 FPS at 1080p full HT model on RX 9070 XT; RR cost anecdotal 0–25%. Bundled **NeuralHub** manager + in-game overlay; user supplies NVIDIA DLLs; anti-cheat games may block. Related forks surfaced: `CaptSilver/DLSS-NR-on-AMD-Linux` (GPL-2.0), `mnector/ComfyUI-RH-DLSS5-AMD` (MIT). **Relevance:** (1) demand signal — community is routing around vendor gating faster than vendors ship neutral tech; (2) practical HIP kernel reference for the cross-vendor port (#8) alongside OpenDLSS-NR (Vulkan); (3) reinforces OSS's clean-room/Apache-2.0 differentiator vs. weight-redistribution-adjacent mods (license here is unstated).
- **News (RSS headline-level unless noted):** Sony frame-prediction patent now covered by TechSpot, TechPowerUp, IGN, PSX Extreme ("PS6 may get own frame generation") — mainstream pickup, still patent-only. GameGPU 10-18: DLSS Frame Gen Model C up to +25% in real-world benchmarks (on top of 10-16 RTX 30 port). VideoCardz 10-18 (primary page read): DLSS-NR-on-AMD above. NVIDIA blog 10-15: DLSS 4.5 controls on GeForce NOW. FSR 4 handheld/Lite items unchanged and still unconfirmed by AMD. Nothing new from Intel, Apple (MetalFX), or UE/Unity. RTX 5090 discontinuation rumors / RTX 5080 24GB Q1 2027 (hardware context only).
- **GitHub:** no new Gaussian-splatting SR or frame-extrapolation repos (query `pushed:>=2026-10-11`, unquoted); same wrapper/hobby set as morning (`ShyVortex/dlss-unlocked` now 851★).
- **New follow-up:** [ ] Skim DLSS-NR-on-AMD HIP kernel layout / FG implementation for #8 (HIP port) and FG-path comparison for OSS-FX; record license ambiguity.

## Sources
- [Wccftech — Sony frame-prediction patent](https://wccftech.com/playstation-solve-frame-generation-issue-patent-frame-prediction/) · [ixbt](https://ixbt.games/en/news/2026/10/17/playstation-xocet-prognozirovat-dvizeniia-igrokov-sony-rabotaet-nad-novym-reseniem.html)
- [GameGPU — FSR 4 Lite (rumor)](https://en.gamegpu.com/news/zhelezo/amd-mozhet-vypustit-fsr-4-lite-dlya-slabykh-videokart-i-ustrojstv-na-baze-rdna-2)
- [NVIDIA — DLSS 5 faster model](https://www.nvidia.com/en-us/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/) · [TechPowerUp — DLSS FG Preset C on RTX 30](https://www.techpowerup.com/)
- arXiv Atom API scan 2026-10-18 (IDs above) · [OSS dashboard](https://opensupersampling.com)
