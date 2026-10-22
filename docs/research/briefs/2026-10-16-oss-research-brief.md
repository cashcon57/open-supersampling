# OSS Research Brief — 2026-10-16

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (2026-06-08) — unchanged; local clone is shallow (single visible commit), `main...origin/main` shows no divergence. Working tree carries uncommitted research briefs (2026-08-11→10-15) + modified `BIBLIOGRAPHY.md` from earlier runs.
- **Open issues (unchanged, 10 listed):** #14 D3 VRAM footprint, #13 Nsight L2 profile (H004), #12 packed low-rank canvas layout, #10 decoupled G-buffer encoder, #9 RepVGG/ECBSR/RepSR student backbone, #8 cross-vendor kernel ports, #7 FSR 4 binary as teacher signal, #6 pico-002 launch, #5 No-ML shader variant, #4 HAT-Tiny student distillation.
- **README:** unchanged — v5-pixel-temporal is still the only measured row (25.703 dB / 0.1666 LPIPS, TartanAir `oldtown`); v7 teacher in training; ≤1M student untrained.
- **Training dashboard (opensupersampling.com):** `srcnn-v7.0-pico-005`, step 36,650 / 100,000 (36.7%), loss 0.19268 — **identical step/loss to the last two days**. Banner is now **"Trainer — offline, stuck at step 36650 for 8h"** (was 618h/630h on 10-15). The counter reset suggests something touched the run dir / metrics heartbeat ~8h ago, but no new step landed. GPU stats still "unavailable — host offline or rsync failed". Watcher / CF Worker / R2 healthy. Cost $0.92 / 15.27 GPU-h unchanged. Footer unchanged (D5–D13, H001–H005; H005 "partial"). The "ETA ~1.1d" text is still meaningless. H009 disclosure (canvas weight ≈ 0; FG demo uses ×50 canvas magnification hack) still on the page.

## New Papers (arXiv)
arXiv API (`export.arxiv.org`) returned empty responses from this host again; coverage via web search + listings. Light day — only one genuinely new relevant paper.

### Gaussian Splatting / Temporal
- **ResStreamGS: Residual-Sensitive Dual-Stream Gaussian Splatting for Mobile Volumetric Video Streaming** (arXiv:2610.21121, 2026-10-15; Haoran Liu, Di Wu, Wang Luo, Hengyuan Na, Miao Hu, Guocong Quan). Classifies Gaussians by temporal behaviour via lightweight probe adaptation: stable Gaussians are sent once per GOP, only residual-sensitive ones go in the per-frame stream; +7.51–8.35 dB post-codec PSNR vs a V³-style all-Gaussians baseline, −56.5% payload, 30.9 FPS on Snapdragon 778G. **Why it matters to OSS:** same split the persistent canvas implicitly needs — static Gaussians reused, only dynamic/disoccluded ones refreshed. Candidate policy for the spawner/retire logic and for cutting per-frame canvas update cost (H004 L2-resident scheduler: stable tiles stay resident, residual tiles get budget). Mobile-GPU evidence point for the Pico tier.
- Other 2610 GS papers seen, **not relevant** (feed-forward 3DGS multi-view / compression): Anchor3DGS 2610.21634, LoCoSplat 2610.04351, InfiniSplat v2 2610.20341, DeltaSplat 2610.09853, Feed-Forward MV Feature Coding 2610.20590. Already logged: TETRIS-4DGS 2610.14663, DynStream 2610.09720, SparseView4DGS 2610.04606, Mobile-4DGS 2610.05289.

### Super-Resolution / Upscaling
- Nothing new on game/real-time SR since **Frozen History** (arXiv:2610.15752) — still the top read (see 10-14/10-15 briefs; code github.com/CV-AC/Frozen_history).
- Off-budget (diffusion / non-SR): FlashVSR+ 2610.17552, FilterFusion video denoising 2610.21505 (efficient filter-fusion; skim only if a denoise/stabiliser head is ever added), DAGS 2610.02567 (frozen DiT generative rendering).

### Frame Extrapolation / Generation
- Nothing new in the 2610 window. Re-confirmed prior-logged **Amulet** (arXiv:2608.10423, Stuttgart; non-neural sparse layered cache, up to 250 Hz @ 4K, competitive with DLSS FG on several metrics) — still the strongest *non-neural* extrapolation comparator for OSS-FX; its predictive per-tile lifetime scheduler parallels H004's budgeted tile scheduler.
- Off-target: Speculative Sampling in Interactive Video Generation 2610.20556, Speculative Forcing 2610.22347, sprite interpolation 2610.16423.

### Inference Optimization
- Nothing new on TensorRT/INT8 SR. Prior anchors still apply: INT8 SISR 2604.20291, TAQ (static INT8 > dynamic INT8), QuantSR+ 2605.22351 (unified quantization operators/design/training for quantized SR — **not previously noted here**; worth a skim for #9/#4 QAT recipe).

## Archive.org Findings
- Advanced Search (2026-09-30→10-17): "gaussian splatting super-resolution" → 0; "frame generation deep learning" → 1 irrelevant; "real-time upscaling game" → 2 GitHub repo snapshots (bloodborne_pc, pt-pc; 10-09). 
- Wayback availability API 429'd for GS-STVSR / GSASR / GRAPE / Frozen History. Prior run recorded a Wayback snapshot of the GS-STVSR abs page (2026-04-21) and the paper is still live. **No takedowns detected** for GS-STVSR, GSASR (2501.06838), GRAPE (2503.06617; WACV 2026 supplement still hosted at CVF), STSS.

## GitHub / Open Source
- Repo search (created/pushed after 2026-10-01): **no new Gaussian-splatting super-resolution repos.**
- Already tracked and still active: **PS5 MihawkGen** (23★, pushed 10-14), **Netal** (9★, DLSS→Metal), **MoonUp**, **HelixSR-Manager** (10★, pushed 10-13), Universal MFG unlockers.
- New minor: `jinbulo/MFG-DLSS5` (Fallout 4 / Skyrim SE DLSS-G MFG mod mirror), `Roskud/Valheim-Universal-Upscaler-GUI` (FSR 3.1/4, XeSS, DLSS GUI), `Chaseriapurify12/frame-generation-2026-fps-performance-toolkit` (13★; appears to be SEO/toolkit spam — ignore). All wrappers around vendor models; none are independent single-network SR+FG. No new open DLSS/FSR analysis tools beyond Netal/HelixSR.
- **Vendor side:** AMD FSR SDK 2.3 / **FSR Upscaling 4.1.1 "Redstone" now supports RDNA 3 (RX 7000) discrete GPUs** (GPUOpen); FSR SDK 2.4 / FSR 4.1.2 reportedly delayed (community, gamegpu.com). FSR 3.1.5 / 2.3.4 bundled. UE5.3→5.8 supported.

## News
- **NVIDIA (threat/pressure):** DLSS 5 neural rendering model cut 24% to **~118M params**, ~2× faster, ~20% less VRAM at 4K; ships **today (Oct 16)** in NBA 2K27 on RTX 50 / RTX Spark laptops / GeForce NOW; Hogwarts Legacy Nov 5; Where Winds Meet late Nov; RTX Remix DLSS 5 on Oct 23. FG **Preset C performance mode** (beta in NVIDIA App; up to +35% FG perf on RTX 5060, slight quality loss; works with FG/MFG/Dynamic MFG). **RTX Spark** (NVIDIA laptop platform) launches Oct 16 with "hundreds of games", full RTX/DLSS suite, broad anti-cheat support — new hardware tier to watch for the Pico/Standard target list.
- **AMD:** FSR 4.1 on RDNA 3 (above) widens ML-FSR to prior-gen cards — erodes the "ML upscaling for older AMD GPUs" gap OSS aimed at, though RDNA 2 / 3.5 stay on analytical FSR 3. Handheld/APU FSR 4 still promised by end-2026.
- **Intel:** XeSS 3 MFG shipping on Arc; Arc G-Series handheld chips; XeSS 3 in Gamer Days titles (Star Wars: Galactic Racer, Tomb Raider: Legacy of Atlantis). D-series (Crescent Island) inference GPU is datacenter — irrelevant.
- **Apple:** nothing new; WWDC26 MetalFX neural temporal upscaler (Neural Engine + M5 Neural Accelerators) remains the reference for the Metal port (#8); Metal gained quantized tensor formats with scale factors.
- **Engines:** Unity 6.3 LTS ships DLSS 4 SR (v310.3.0); Unity 6.6 and 7.0 alpha exist (no upscaler-specific item found). UE 5.8 last UE5; UE6 EA late 2027. Nothing new.

## Action Items / Follow-ups
- [ ] **Dashboard:** trainer banner "offline, stuck at step 36650 for 8h" (reset from 630h) — check on homelab-desktop whether pico-005 was restarted/parked and whether metrics.json is actually advancing; fix rsync/GPU stats or mark the run parked.
- [ ] Deep-read **Frozen History** (2610.15752) and run the four-registration-error audit + temporal-weight ≤0.05 ablation as an H009 follow-up (carried, priority).
- [ ] Read **ResStreamGS** (2610.21121): stable-vs-residual Gaussian split as spawn/retire/refresh policy; add as note under H004 / issue #12.
- [ ] Skim **QuantSR+** (2605.22351) for the #9/#4 student quantization recipe; verify FP8-vs-INT8 conv on TensorRT (carried).
- [ ] Compare OSS Pico latency/size targets against DLSS 5's new ~118M-param / 2× faster model and FG Preset C (+35% on 5060) — raises the quality-per-ms bar.
- [ ] Update README competitor note: FSR 4.1 now on RDNA 3; Arm NFRU (MIT), MihawkGen, HelixSR (carried).
- [ ] Re-try arXiv API / Wayback availability next run (both rate-limited/empty from this host).

## Sources
- [ResStreamGS, arXiv:2610.21121](https://arxiv.org/abs/2610.21121)
- [Amulet, arXiv:2608.10423](https://arxiv.org/abs/2608.10423)
- [QuantSR+, arXiv:2605.22351](https://arxiv.org/pdf/2605.22351)
- [NVIDIA RTX Spark launch](https://www.nvidia.com/en-us/geforce/news/rtx-spark-dlss-reflex-ray-tracing-games/)
- [iXBT: DLSS 5 2× faster, Preset C](https://ixbt.games/en/news/2026/10/14/nvidia-uskorit-dlss-5-vdvoe-i-dobavit-podderzku-novyx-igr.amp.html)
- [AMD GPUOpen FSR Upscaling 4.1.1](https://gpuopen.com/amd-fsr-upscaling/)
- [Apple Metal what's new](https://developer.apple.com/metal/whats-new/)
- [OSS dashboard](https://opensupersampling.com)

---

## Evening update — 2026-10-16 20:00 EDT (second run today)

### Project status delta
- Repo, issues (#4–#14), README: unchanged since the morning run (HEAD still `b44278a`, 2026-06-08).
- **Dashboard:** `srcnn-v7.0-pico-005` still step 36,650 / loss 0.19268. Banner now **"Trainer — offline, stuck at step 36650 for 20h"** (was 8h this morning) → confirms the trainer is genuinely not advancing; the earlier counter reset was not a restart. GPU stats still "host offline or rsync failed". Watcher / CF Worker / R2 healthy. Cost $0.92 / 15.27 GPU-h unchanged. The "ETA ~1.1d" text is still misleading while the trainer is stalled.

### Papers (arXiv)
- arXiv API (`export.arxiv.org`) returned empty for all 4 query groups again; coverage via web search + `cs.CV/recent` listing (Fri 16 Oct, 165 entries).
- **No new relevant papers** beyond the morning list. Scanned the 10-16 `cs.CV` listing; only already-logged ResStreamGS (2610.21121) is on-topic. Irrelevant: Anchor3DGS, InfiniSplat v2, VisibilitySplat 3DGS relighting, RDCompGS (3DGS codec compression), VascuSR (CT diffusion SR).
- Previously logged and re-surfaced by search (no action): GS-STVSR 2604.18047 (still live on arXiv, no takedown), STGV 2604.10910, Amulet 2608.10423, GameSR (SFU, TVCG '26), Gaussian Splatting for Real-Time Scene Reconstruction survey 2610.17513, 2610.07795 GS sequence compression.

### Archive.org
- Not re-queried (Wayback API was 429-ing this morning); GS-STVSR still resolves live on arXiv, so no takedown signal.

### GitHub / open source (new today)
- **Skynizz/optiscaler-dlss5 v1.2.1** (OptiScaler fork on Dagherbou's DLSSNR branch; not previously logged) — **directly relevant to OSS's core thesis.** Its "edit cache" runs NVIDIA's DLSS 5 neural model only every N frames (N up to 8, "smart cadence": less often when the camera is still), stores the model's *per-pixel edit ratio* (not the image), **reprojects it with motion vectors, validates per pixel (depth/colour), and composites it on the fresh frame.** Numbers (TLOU-I, RTX 4070, 1440p): DLSS 5 cost 13.2 ms → 2.5 ms (Performance) → ~0.9 ms (x8), FPS 39.2 → 68.6 → 77.3. Control Resonant: 13.0 ms → 2.9 ms with flicker 0.57% → 0.47%. Adds anti-ghosting via a frame "fingerprint" that drops stale edits, and "pre-SR" placement (model at render res before SR, ~3× fewer pixels).
  - **Why it matters:** independent community evidence that *amortizing an expensive neural pass across frames via MV-reprojected cached state* works and holds quality — the same bet as the persistent canvas. It also supplies a concrete recipe for canvas refresh policy (cadence tied to camera motion, per-pixel depth/colour validity gate, anti-ghost on long intervals) and a flicker metric (frame-to-frame brightness change with camera still, excluding top 5% moving pixels) OSS could adopt for temporal stability evals. Related to H004 / H009 (canvas underutilised): the validity-gated residual is a known-good alternative to cross-attention fusion.
- **ZeroFG** (hy300leosquizz-ctrl, 9★, pushed 10-15) — open-source Vulkan frame-generation engine for mobile / emulator hosts (ARMSX2 integration). Wrapper-style FG, not single-network SR+FG; low threat, possible Vulkan kernel reference for #8.
- `SE-FG` (Space Engineers FG mod, 0★), `ShyVortex/dlss-unlocked` (839★, 10-11) — mod wrappers, no threat.
- No new Gaussian-splatting SR repos (search returned 0 for pushed >10-09).

### News (delta vs morning)
- **NVIDIA:** DLSS 5 3D-Guided Neural Rendering new model **goes live today** in NBA 2K27 Season 2 (up to 1.4× fps on RTX 50 / RTX Spark / GFN); NVIDIA claims ~10× faster than the March model, 2× faster than September, ~118M params, −20% 4K VRAM. RTX Remix gets DLSS 5 Oct 23; Hogwarts Legacy Nov 5; CoD MW4 uses DLSS 4.5 suite (early access today, launch Oct 23). Official DLSS 5 on RTX 40 promised, RTX 30/20 only via mods. Quake 1996 + DLSS 5 four-pass mod experiments (PCGH, 10-14).
- **Mod scene:** DLSSNR.dll running on Radeon RX 7000/9000 (DLSS-NR-on-AMD 0.4.3) and Arc 140V (slow); HelixSR experimental Vega support; Aeon SR ReShade add-on (auto optical-flow MVs for DLSS/FSR/XeSS in old games, GitHub BarbatosAWLS/AeonSR). Signal: vendor-neutral demand is being met by reverse-engineered wrappers — OSS's licensing/legal cleanliness is the differentiator.
- **Sony:** QSSR (PS5 base, Oct 1; Wolverine + Ghost of Yōtei first) — already logged 10-08/10-14.
- **Apple:** unchanged (WWDC26 MetalFX neural temporal upscaler on M5 Neural Accelerators).
- Nothing new from AMD / Intel since the morning brief.

### Action items (new / updated)
- [ ] **Read `OptiScaler/dlssnr/design/edit-cache.md` and `docs/comparison.md` in Skynizz/optiscaler-dlss5** — borrow cadence policy, per-pixel validity gate, and the flicker metric for OSS temporal eval; cite as prior art for MV-reprojected cached neural state.
- [ ] **Dashboard/trainer:** pico-005 has now been stalled for 20h (not a restart). Check homelab-desktop trainer process, GPU, rsync; resume or mark run parked and fix the misleading ETA (carried, now higher priority).
- [ ] Retry arXiv API + Wayback availability next run (both still failing from this host); consider a different egress or scraping `arxiv.org/list/*/recent` as the fallback, as done today.
- Carried from morning: Frozen History (2610.15752) audit, ResStreamGS → H004/#12 note, QuantSR+ skim, README competitor-note update, DLSS 5 latency bar.

### Sources (evening additions)
- [OptiScaler DLSS 5 (Skynizz)](https://github.com/Skynizz/optiscaler-dlss5)
- [ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG)
- [NVIDIA: DLSS This Week (DLSS 5 faster model)](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-crossout-remastered-modern-warfare-4-frame-gen-beta/)
- [Tom's Hardware: DLSS 5 2× faster, 118M params](https://www.tomshardware.com/pc-components/gpus/nvidia-doubles-dlss-5-performance-reduces-4k-vram-usage-by-20-percent-with-latest-model-october-update-cuts-footprint-by-24-percent-to-roughly-118-million-parameters)
- [PCGH DLSS news feed](https://www.pcgameshardware.de/Deep-Learning-Super-Sampling-Software-277618/News/)
- [AeonSR](https://github.com/BarbatosAWLS/AeonSR)
- [arXiv cs.CV recent](https://arxiv.org/list/cs.CV/recent?skip=0&show=250)
