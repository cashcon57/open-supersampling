# OSS Research Brief — 2026-10-15

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` — unchanged; checkout is shallow (single visible commit). `git pull` failed (remote access error from this env), so local HEAD may lag origin.
- **Open issues (unchanged):** #14 D3 VRAM, #13 Nsight L2 profile, #12 packed low-rank canvas, #10 decoupled G-buffer encoder, #9 RepVGG/ECBSR reparam student, #8 cross-vendor kernels, #7 FSR 4 teacher distillation, #6 pico-002 launch, #5 no-ML shader variant, #4 HAT-Tiny student distillation.
- **README:** pre-alpha; v5-pixel-temporal still the only measured row (25.703 dB / 0.1666 LPIPS); v7 is the teacher architecture; ≤1M student untrained.
- **Training dashboard (web_extract, loaded):** run `srcnn-v7.0-pico-005`, step 36,650 / 100,000 (36.7%), loss 0.19268, SR charbonnier 0.0236. **Trainer banner says "offline — stuck at step 36650 for 618h"**; GPU stats "unavailable — host offline or rsync failed"; "Waiting for score_log_v7.json" (the +1 dB OSS-FX vs bicubic gate has not surfaced). Watcher / CF Worker / R2 healthy. Cost $0.92, 15.27 GPU-h. ETA text "~1.1d" is therefore stale/misleading. **Dashboard now reflects v7 (earlier-run stale-copy concern is resolved) but the trainer-offline state is a real flag — confirm whether training is actually stopped or just the heartbeat/metrics rsync is broken.**

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- No new directly relevant papers since the 10-14 briefs (Mobile-4DGS 2610.05289, TETRIS-4DGS 2610.14663, DynStream 2610.09720, SparseView4DGS 2610.04606, GSCV 2610.07795 already logged).
- **Gaussian Density Splatting Network** (arXiv:2610.10396, v2 2026-10-08; authors not retrieved — arXiv rate-limited). Gaussian splatting for density-map prediction with count-consistency and shape-regularization losses. Why it matters to OSS: marginal; the mass-conservation/shape-regularization loss pair is a possible regularizer for canvas Gaussian covariance stability. Low priority, unverified.

### Super-Resolution / Upscaling
- **Frozen History** (arXiv:2610.15752, 2026-10-11) — still the top read; re-confirmed present in search. Temporal-loss/history-accumulation interaction in learned real-time supersampling. See 10-14 brief for action items.
- Nothing else new on game-latency SR. Diffusion VSR (FlashVSR+ 2610.17552, 2610.11746, 2610.13179) continues; not game-budget relevant.

### Frame Extrapolation / Generation
- Nothing new. (2610.16302 RAR, 2610.19212 drift-diffusion world models, 2610.16423 sprite interpolation: off-target.)

### Inference Optimization
- Nothing new on TensorRT/INT8 SR. NVIDIA docs on TensorRT-RTX note FP8 only supported for GEMM on Ada+ and INT4/INT8/FP4/FP8 types — relevant to the "TRT FP8 student" plan: a ≤1M-param conv student is conv-heavy, so verify FP8 conv actually lands on tensor cores vs INT8 before committing (check #9/#4).

## Archive.org Findings
- Advanced Search API worked. Date-limited (2026-09-15→10-15) queries for "gaussian splatting super-resolution" → 0 hits; "frame generation deep learning" → 7 hits, irrelevant; "real-time upscaling game" → 2 hits, both GitHub repo snapshots (Bloodborne PC etc.). Title searches for GS-STVSR / GSASR → 0 records; GRAPE → generic noise. Wayback availability API 429'd. **No archived/taken-down versions of GS-STVSR, GSASR, GRAPE, STSS found.**

## GitHub / Open Source
- **HelixSR** (1.9.0 on 10-13; experimental Radeon Vega support 10-14): DLSS Model E on D3D12 compute with INT8 paths, no tensor cores. Reconfirms the cross-vendor competitor/benchmark lane noted yesterday.
- **Universal MFG Unlocker 1.5.0** (10-14): adds XeSS MFG for AMD/Intel and DLSS Preset C for RTX 30/40. Demand for vendor-neutral FG keeps being met by mods around vendor SDKs, not independent models.
- **Arm Neural Graphics SDK** (MIT-licensed, open-source; NSS + new **Neural Frame Rate Upscaling (NFRU)**, blog 2026-09-23, UE 5.4/5.6 plugins, Vulkan / Vulkan ML backend, Windows via emulation layer + Android). NFRU pipeline = optical flow → frame generation → frame pacing; NFRU + NSS 2× upscaling renders 1/8 of pixels. **Most direct open-source (permissive-licensed) SR+FG competitor** — still two networks/stages, mobile-first, but the first MIT-licensed vendor stack with an engine plugin. Already referenced in earlier briefs (9/12–9/15, 10/06); worth a positioning line in README comparison.
- Awesome-gaussians (longxiang-ai) and playcanvas/supersplat remain active; no new GS-SR repos surfaced. GS-STVSR official code still unconfirmed.

## News
- **NVIDIA:** DLSS 5 2× faster model ships to Hogwarts Legacy (Nov 5); DLSS FG Preset C (lighter second-model replacement) — both covered 10-14. No new announcements today.
- **AMD:** FSR 4 for handhelds/laptops by end-2026 (The Verge, PC Guide); FSR "Redstone" SDK 2.2 (FSR Upscaling 4.1, FG 4.0.0, Ray Regeneration 1.1.0) per AMD developer article; community sentiment on Redstone mixed (Tech4Gamers). Call of Duty MW4 PC ships DLSS 4.5 + FSR Redstone + XeSS 3.
- **Intel:** XeSS FG override only engages when game's own FG is on (Oct 6); XeSS MFG on Arc via driver; SDK 3.0.2 latest on GitHub (Jul). Intel announced D-series AI inference GPU (1.3 PFLOPS FP8) 10-12 — datacenter, not gaming.
- **Apple:** WWDC26 Metal: redesigned temporal MetalFX upscaler using Neural Engine + Neural Accelerators on M5 Pro/Max (no new news today).
- **Engines:** Unreal Engine 5.8 released (June); Epic announced UE6 with early access targeted late 2027, and 5.8 is the last UE5 release. No new upscaler-specific announcement. UE6 plugin architecture is worth watching for DLL-shim/plugin integration path.

## Action Items / Follow-ups
- [ ] **Dashboard:** trainer "offline, stuck at step 36650 for 618h" — verify whether pico-005 is actually running on the 3080 Ti host; fix heartbeat/rsync or restart. Also pico-005 has no score_log_v7.json yet, so no OSS-FX gate result.
- [ ] Deep-read **Frozen History** (2610.15752) and map its four registration-error modes to OSS (carried).
- [ ] Add Arm NFRU/NSS (MIT, open-source) to README competitor comparison as the closest permissive-licensed SR+FG stack.
- [ ] Benchmark target: HelixSR INT8 on RDNA vs OSS Pico student (carried).
- [ ] Verify FP8 conv vs INT8 path on TensorRT for the nano-CNN student (#9/#4).
- [ ] Retry `git pull` / confirm local clone is current; remote access failed this run.

## Sources
- [Frozen History, arXiv:2610.15752](https://arxiv.org/abs/2610.15752)
- [Gaussian Density Splatting Network, arXiv:2610.10396](https://arxiv.org/abs/2610.10396)
- [Arm Neural Frame Rate Upscaling blog](https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/introducing-neural-frame-rate-upscaling-in-arm-neural-graphics-sdk-for-game-engines-and-ue-plugins)
- [AMD FSR Redstone SDK 2.2 article](https://www.amd.com/en/developer/resources/technical-articles/2026/upscale-everything-super-resolution-across-amd-hardware-.html)
- [AMD FSR 4 handhelds (The Verge)](https://www.theverge.com/games/1008353/amd-will-bring-fsr-4-to-handhelds-by-the-end-of-2026)
- [Intel XeSS FG override](https://mixed-news.com/en/intel-xess-frame-generation-override-game-setting-confirmed/)
- [Apple Metal what's new](https://developer.apple.com/metal/whats-new/)
- [UE6 roadmap (Inven)](https://www.invenglobal.com/articles/22900/unreal-engine-6-to-feature-fundamental-overhaul-targeting-early-access-release-by-late-2027)
- [OSS dashboard](https://opensupersampling.com)

---

## Evening update — 2026-10-15 (second run)

### Project status
- **Commits:** still `b44278a` (shallow clone, HEAD dated 2026-06-08); no new local commits. Open issues unchanged (12 open, #4–#14).
- **Dashboard:** `srcnn-v7.0-pico-005` still step 36,650 / 100,000, loss 0.19268. Banner now says **"stuck at step 36650 for 630h"** (was 618h this morning → heartbeat is advancing 1:1 with wall clock, i.e. the trainer is genuinely not running, not just a stale label). GPU stats "host offline or rsync failed". Watcher / CF Worker / R2 healthy. Cost still $0.92 / 15.27 GPU-h. The "ETA ~1.1d" text is meaningless while the trainer is down. Dashboard footer still lists D5–D13 and H001–H005 with all H-series "untested" (H005 "partial").

### New papers (arXiv)
arXiv API rate-limited again (HTTP "Rate exceeded"); coverage via web search + cs.CV listing. **No new directly relevant papers since the morning run.**
- **Frozen History** (arXiv:2610.15752; Hao Yu, Guoying Zhao; 2026-10-11) — abstract read in full this run. Key numbers: with correct registration, temporal-loss weight 0.25 costs 0.7–0.9 dB on static frames; a plain registered net + small temporal weight + normalised sample accumulator beats best published QRISP PSNR by 0.34–0.62 dB at ×2/×3/×4 to 1080p; static-pixel std at ×2 is ¼ of DLSS 2; ×4 model = **2.03 ms / 1080p frame on H200**; beats classical TAAU by ≥4.4 dB on Arm's jittered Bistro capture. Four registration errors: jitter-carrying MVs, one motion field for both LR/HR lattices, jittered depth, unregistered skip path. Code: github.com/CV-AC/Frozen_history. **Why it matters to OSS:** direct audit checklist for the canvas warp (we warp a persistent canvas by engine MVs — the same failure modes apply, and OSS's H009 "composite_head weights canvas ≈ 0" looks like the same freeze/copy pathology via a different route). Candidate explanation for H009 beyond fusion-magnitude imbalance → check temporal-loss weight in v7 config. Also gives a ready cross-benchmark (QRISP, Arm Bistro) for OSS's first non-TartanAir eval.
- **ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering** (arXiv:2506.13814; Lufei Liu, Tor Aamodt; ICML 2025) — *not previously logged.* Feature caching across frames for encoder-decoder nets in denoising / supersampling / frame extrapolation; 1.4× avg speedup, negligible quality loss. **Why it matters:** cheap inference-budget lever for the ≤1M student (cache deep-encoder features on static regions); complements the canvas's own temporal reuse. Code on project page.
- **Gaussian Splatting for Real-Time Scene Reconstruction: Methods, Systems, and Trade-offs** (arXiv:2610.17513; 2026-10-13) — survey; not logged. Low priority; skim for systems/kernel references relevant to issue #8/H004.
- Re-seen, already logged: GS-STVSR (2604.18047), TAQ (ECCV 2026 #11260; TensorRT INT8 3.56× on Orin Nano, dynamic INT8 *slower* than FP32 — supports static-quant plan), INT8 SISR (2604.20291), Deformable 2DGS 4K video (2609.14129), Mobile-4DGS, TETRIS-4DGS, FlashVSR+ (off-budget).

### Archive.org
- Date-limited queries (2026-09-15→10-16): GS-SR → 0; "frame generation deep learning" → 7 irrelevant; "real-time upscaling game" → 2 GitHub snapshots (Bloodborne PC, pt-pc). Title searches GS-STVSR / GSASR / GS video SR → 0 records.
- **Wayback has a snapshot of GS-STVSR's arXiv abs page** (`web.archive.org/web/20260421072344/https://arxiv.org/abs/2604.18047`), and the paper is still live on arXiv (no takedown). Frozen History has no snapshot yet (too new). No takedowns of GS-STVSR / GSASR / GRAPE / STSS detected.

### GitHub / open source (new in last ~14 days — none previously logged)
- **PS5 MihawkGen** (`mihawk-99/PS5_MihawkGen`, created 2026-10-14, GPL-3.0+MIT per README / NOASSERTION in API) — **0.43M-param** neural frame generation, ~**2.2 ms at 4K on a base PS5** (compute shaders, 4.4 GFLOPs @288×512), 1.7 MB weights; **interpolation** from two colour frames, no MVs needed; claims to beat FSR 3.1 on unseen frames and, on 33 never-trained 2D clip sets, 30.08 dB PSNR vs 30.44 for a 20M-param GIMM-VFI-R. Optional `mvsplat` path uses depth+MVs. **Relevance:** existence proof that a sub-0.5M FG net hits ~2 ms-class budgets on console-class GPU — validates the OSS Pico (≤0.4M) sizing and the cost argument for OSS-FX (which should be cheaper still). It is interpolation (adds a frame of latency); OSS-FX's extrapolation pitch still differentiates. Good PSNR/LPIPS comparator for the α<1 path once it's trained.
- **Netal** (`nschlaepfer/Netal`, MIT, 2026-10-10) — reconstructs NVIDIA DLSS SR + neural FG runtime on Apple Metal (BYO NVIDIA DLLs/models), experimental DLSS 5 neural rendering on still frames. Shows reverse-engineered DLSS-on-non-NVIDIA is now live (MLX-DLSS by iamwavecut earlier). Legal/positioning angle: OSS's clean-room/open-teacher story gets more valuable as these derivatives proliferate.
- **MoonUp** (`twayst0/MoonUp`, PolyForm Noncommercial, 2026-10-06) — Windows any-GPU window-level upscaler + FG (optical flow) + "Neural Render" modelled on DLSS 5. Not a quality competitor; shows market appetite for vendor-neutral, any-GPU solutions.
- **HelixSR-Manager** (Denarmo, 2026-10-11) — GUI/model compiler for HelixSR (DLSS Model E on D3D12 compute). Confirms HelixSR ecosystem growth (carried benchmark target).
- GitHub `frame-generation` topic: 122 repos, dominated by DLSS-5/MFG unlockers & OptiScaler plugins (e.g., dlss5-visual-enhancer 1.3k★, DLSS5-Autopilot 890★) — all wrappers around vendor models, none are independent single-network SR+FG.
- No new Gaussian-splatting-SR repos in the window (GitHub search `created:>2026-09-30` returned nothing).

### News (past 7–14 days)
- **NVIDIA:** (10-13) "DLSS This Week": **new faster DLSS 5 model (≈2× faster, ~20% less VRAM per 3DNews) debuts in NBA 2K27 Season 2 on Oct 16** (tomorrow); new faster **Performance Mode for Frame Generation enters beta**; DLSS 4.5 Dynamic MFG + SR in CoD: MW4 early access Oct 16; DLSS 4.5 controls on GeForce NOW. DLSS 5 = "3D-guided neural rendering" (dev blog 2026-09-22; engine frame as ground truth + neural lighting/material layer) — a direction beyond SR/FG; RTX Kit 2026.3 + Mega Geometry 2.0 released. RTX 5090 GB202 allocation reportedly halted; RTX 50 Super stuck on GDDR7 pricing (rumour: 2027) — hardware refresh slipping helps the "older GPU / cross-vendor" positioning.
- **AMD:** (10-07) FSR 4 confirmed for APUs/handhelds by end-2026 with lighter models; Redstone SDK 2.2 (FSR Upscaling 4.1, FG 4.0). Lighter-FSR4 handheld models are the direct Pico-tier competitor (Steam Deck-class, <2 ms @720p→1080p).
- **Intel:** XeSS 3 MFG available on Arc + XeSS 2 games; XeSS 3 plugin 3.1 supports UE 5.8 (07-30); Arc G-series handheld chips announced.
- **Apple:** WWDC26 redesigned MetalFX temporal upscaler on Neural Engine + M5 Neural Accelerators — nothing new this week.
- **Engines:** nothing upscaler-specific new; UE 5.8 last UE5, UE6 EA late 2027 (carried). AMD FSR plugin for UE 5.3–5.8 available.

### Action items / follow-ups (additions)
- [ ] **Training host:** pico-005 has been dead ≥630h per dashboard; either restart on 3080 Ti (homelab-desktop) or mark run parked on the dashboard so the ETA/“live” banner isn't misleading. (Carried from morning; escalating.)
- [ ] **Frozen History audit (priority):** check v7 temporal-loss weight, whether canvas warp uses jitter-free MVs, separate MV fields for LR vs HR lattice, jitter in depth, and canvas skip path registration. Test lowering temporal weight to ≤0.05 as an H009 follow-up ablation. Clone CV-AC/Frozen_history and run on Arm Bistro / QRISP for a first cross-dataset number.
- [ ] Read **ReFrame** (2506.13814): evaluate feature caching in the student for static tiles; add as issue under #4/#9.
- [ ] Add **MihawkGen** (0.43M, 2.2 ms @4K PS5) as a size/latency comparator in student-budget docs and as an interpolation-baseline for OSS-FX once α<1 is trained.
- [ ] Watch NVIDIA's faster DLSS 5 / FG Performance Mode (Oct 16) — expect more quality-per-ms pressure on the <2 ms budget.
- [ ] Positioning: add Arm NFRU (MIT) + MihawkGen + HelixSR to the README "adjacent open-source" comparison (carried/extended).

### Sources (evening)
- [Frozen History abstract](https://arxiv.org/abs/2610.15752) · [code](https://github.com/CV-AC/Frozen_history)
- [ReFrame, arXiv:2506.13814](https://arxiv.org/abs/2506.13814)
- [GS real-time survey, arXiv:2610.17513](https://arxiv.org/abs/2610.17513)
- [GS-STVSR Wayback snapshot](http://web.archive.org/web/20260421072344/https://arxiv.org/abs/2604.18047)
- [PS5 MihawkGen](https://github.com/mihawk-99/PS5_MihawkGen) · [Netal](https://github.com/nschlaepfer/Netal) · [MoonUp](https://github.com/twayst0/MoonUp)
- [NVIDIA DLSS This Week (10-13)](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-crossout-remastered-modern-warfare-4-frame-gen-beta/)
- [TechPowerUp: FSR 4 on APUs](https://www.techpowerup.com/353477/amd-confirms-fsr-4-is-coming-to-apus-by-the-end-of-2026)
