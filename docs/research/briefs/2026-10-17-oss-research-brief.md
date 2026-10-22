# OSS Research Brief — 2026-10-17

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (2026-06-08) — unchanged; clone is shallow, `main...origin/main` clean. Working tree still carries uncommitted research briefs (2026-08-11 → 10-16) and a modified `BIBLIOGRAPHY.md` from earlier runs.
- **Open issues (unchanged, 10 listed):** #14 D3 VRAM footprint, #13 Nsight L2 profile (H004), #12 packed low-rank canvas layout, #10 decoupled G-buffer encoder, #9 RepVGG/ECBSR/RepSR student backbone, #8 cross-vendor kernel ports, #7 FSR 4 binary as teacher signal, #6 pico-002 launch, #5 No-ML shader variant, #4 HAT-Tiny student distillation.
- **README:** unchanged. v5-pixel-temporal remains the only measured row (25.703 dB / 0.1666 LPIPS, TartanAir `oldtown`); ≤1M student untrained.
- **Training dashboard (opensupersampling.com):** `srcnn-v7.0-pico-005`, step 36,650 / 100,000, loss 0.19268 — identical to every run since 10-14. Banner: **"Trainer — offline, stuck at step 36650 for 44h"** at the 20:00 refresh (8h → 20h → 32h → 44h across the last four runs: trainer is genuinely not advancing; counter is consistent, +12h per 12h). Watcher / CF Worker / R2 healthy. The "ETA ~1.1d" text is still shown and is misleading.

## New Papers (arXiv)
Tooling note: `export.arxiv.org` API times out from this host again (curl 000); `arxiv.org/list/*/recent` returns a placeholder page. Coverage is via web search + title scans. Today is Saturday — no new arXiv listing since Fri 10-16, so little is new.

### Gaussian Splatting / Temporal
- No new on-topic papers since ResStreamGS (2610.21121, logged 10-16).
- Newly surfaced, low relevance (skim titles only): **PCAsplat** (2610.11011, local PCA regularization for 3DGS), **GaussianBench** (2610.10554, physics-fidelity evaluation of Gaussian scene representations — may offer ideas for canvas fidelity metrics), **pose-graph failure analysis for GS** (2610.21664). Already logged and unchanged: GS-STVSR 2604.18047, Deformable 2D GS 4K video compression 2609.14129, TETRIS-4DGS, DynStream, Mobile-4DGS, SparseView4DGS, GS real-time reconstruction survey 2610.17513.

### Super-Resolution / Upscaling
- **Frozen History** (arXiv:2610.15752) — still the top read for OSS (learned real-time supersampling, temporal-loss accumulation, 2.03 ms/1080p ×4 on H200). Carried action item.
- **High-Throughput Generative VSR on a Single Consumer GPU** (arXiv:2610.13179) and **Streaming-Aware Diffusion for Real-Time VSR** (2610.11746): diffusion-based, zero-shot/one-step. Off-budget for the <2 ms target; only relevant as the "quality ceiling" comparison class. Previously noted.
- **On-Device SR for SPAD array on 640-KB-SRAM MCU** (2610.09098): extreme-size SR; of marginal interest as an existence proof for tiny-student design.

### Frame Extrapolation / Generation
- Nothing new. Amulet (2608.10423) remains the strongest non-neural extrapolation comparator; MihawkGen (0.43M params, ~2.2 ms 4K on PS5) the size/latency comparator.

### Inference Optimization
- **Memory-Efficient Integer-Only Training of DNNs** (arXiv:2610.16298, 2026-10-12) — integer-only training algorithm. Not previously logged. Why it matters: probably not for the student's QAT (which wants FP8/INT8 inference, not integer training), but worth one skim if training VRAM on the 3080 Ti becomes the bottleneck.
- QuantSR+ (2605.22351) skim remains open for #9/#4.

## Archive.org Findings
- Advanced Search (2026-09-30 → 10-18): `"gaussian splatting" super-resolution`, `"frame generation" deep learning`, `"real-time upscaling" game` → **0 hits each.**
- Wayback availability API returns empty `archived_snapshots` for arxiv.org/abs/2604.18047 (GS-STVSR), 2501.06838 (GSASR), 2503.06617 (GRAPE) — **but all three still resolve live (HTTP 200)**. **No takedowns.** Recommend requesting a Save Page Now capture of these three abs pages so a snapshot exists if a takedown ever happens (no action taken by this run).

## GitHub / Open Source
- **OpenDLSS-NR** (maanHimself/OpenDLSS-NR, MIT; mirror aloshdenny/open-dlss) — *not previously logged in briefs.* Vulkan reimplementation of NVIDIA's DLSS 5 Neural Rendering network (71-block Swin/ViT U-Net, FP8 E4M3 on tensor cores via `VK_KHR_cooperative_matrix`), claimed **bit-exact** against NVIDIA's build 310.8.0; ~809★ by Oct 4; WebGPU port also. User must supply NVIDIA's 141 MiB weights (not in repo). Reported 39–50% frame-rate cost on RTX 40 (FP8 fallback, no NVFP4). Relevance to OSS: (1) validates there is demand for an open, vendor-neutral path — but it is a reverse-engineered, weights-dependent wrapper, so OSS's clean-room/Apache-2.0 + own-weights story is a legal differentiator; (2) a **reference for Vulkan cooperative-matrix + FP8 kernels** relevant to issue #8 (cross-vendor ports) and the FP8 TensorRT plan; (3) the 39–50% FP8 fallback cost is a cautionary data point for "FP8 on non-Blackwell".
- Wrappers/mods (no threat; vendor models underneath): `ShyVortex/dlss-unlocked` (841★, pushed today), `Echo-Storm/ls-addon-manager` (187★, Lossless Scaling add-on manager for DLSS 5 NR/XeSS/FSR), `HelixSR-Manager` (11★), `ZeroFG` (9★ Vulkan mobile FG), `SE-FG`, `menghuihuan44/fluid` (0★ generic FG/enhancement project).
- No new Gaussian-splatting super-resolution repos (only a quirky `ResonantBraneSplatting` — 1★, 2D-GS "quantum" claim, ignore).
- Previously logged and still active: MihawkGen, Netal, optiscaler-dlss5 (edit-cache), AeonSR, DLSS-NR-on-AMD/Intel mods.

## News
- **NVIDIA:** DLSS 5 neural-rendering model cut to ~118M params, ~2× faster than September, −20% 4K VRAM; NBA 2K27 live 10-16, RTX Remix Oct 23, Hogwarts Legacy Nov 5, CoD MW4 DLSS 4.5 launch Oct 23. UE 5.8 DLSS 5 NR plugin access is being requested on NVIDIA forums; NvRTX-5.8 prerelease posted Oct 8. RTX 40 rollout of DLSS 5 officially confirmed (per Turkish/Tom's coverage). Bar to beat on quality-per-ms keeps rising (already in action items).
- **AMD:** FSR Redstone ~10 months in — FSR 4/4.1 supports 415 titles; FSR 4.1.1 reaches RDNA 3; GPUOpen UE plugin covers 5.3–5.8 with ML upscale + ML FG; AMD published "Upscale Everything: Super-Resolution Across AMD Hardware" technical article (worth reading for the cross-hardware ML path, relevant to #8/#7). DLSS-3.7-on-Steam-Deck/AMD mod reported 10-11.
- **Intel:** XeSS 3 MFG on Arc; **XeSS 3 plugin 3.1 adds Unreal Engine support** (PCGH 10-15); Arc G-Series handheld; Gamer Days titles with XeSS 3. Nothing architecture-shifting.
- **Apple:** unchanged — WWDC26 redesigned MetalFX temporal upscaler on Neural Engine + M5 Neural Accelerators remains the reference for the Metal port.
- **Engines:** UE 5.8 (last UE5) plugin support from NVIDIA/AMD/Intel; Unity 6.3 LTS ships DLSS 4 SR. Nothing new.

## Action Items / Follow-ups
- [ ] **Dashboard/trainer (highest priority):** pico-005 stalled at step 36,650 for 44h and rising (20:00 refresh). Check trainer process / GPU / rsync on the 3080 Ti host; resume, or mark the run parked and remove the misleading "ETA ~1.1d" banner. Also: GPU stats "host offline or rsync failed" earlier this week.
- [ ] Deep-read **Frozen History** (2610.15752): four-registration-error audit + temporal-weight ≤0.05 ablation for H009 (carried).
- [ ] Read **OpenDLSS-NR** repo (`maanHimself/OpenDLSS-NR`): Vulkan cooperative-matrix/FP8 kernel structure → note under issue #8; check its license/clean-room statements to sharpen OSS's legal-differentiation line in the README competitor section.
- [ ] Read AMD **"Upscale Everything"** technical article for portability patterns (issue #8 / #7).
- [ ] ResStreamGS → H004/#12 note on stable-vs-residual Gaussian refresh policy; QuantSR+ skim for #9/#4; optiscaler-dlss5 `edit-cache.md` read (all carried).
- [ ] Optional: submit Wayback Save Page Now for GS-STVSR / GSASR / GRAPE abs pages.
- [ ] Fix research tooling: arXiv API unreachable from this host for >1 week; consider fetching via a different egress or the `arxiv.org/a/…` Atom feeds by category as fallback.

## Evening refresh — 2026-10-17 20:00 EDT (deltas only)
- **Dashboard:** `srcnn-v7.0-pico-005` step 36,650 / 100,000, loss 0.19268, viz @ 36,000 — unchanged. Banner now **"Trainer — offline, stuck at step 36650 for 44h"**; Watcher healthy (published 0s ago), CF Worker HTTP 200 119 ms, R2 healthy. Header still says "live" and "ETA ~1.1d" — contradictory with offline state (ops follow-up unchanged).
- **Repo:** no new commits (`b44278a`), 10 open issues unchanged (#14 → #4 listed above; #11 absent).
- **arXiv (Atom API reachable via `https://export.arxiv.org`, `http://` 301s; heavily rate-limited — one query per ~20s):** no new on-topic GS/VSR/VFI papers since this morning. Newly surfaced, not previously logged:
  - **Hardware-Friendly Token-Group Activation Quantization for Low-Bit Mamba SR** (arXiv:2610.16261, Chung & Kim, 2026-10-12) — static, calibration-table-based activation PTQ with a lightweight group predictor, no runtime stats. Why it matters to OSS: Mamba backbone, so not directly applicable to the CNN/pico student, but the "offline-calibrated clipping tables, zero runtime statistics" recipe is the right shape for an FP8/INT8 student that must fit <2 ms. Skim alongside QuantSR+ for #9/#4. Not added to BIBLIOGRAPHY (low direct relevance).
  - Already tracked / unchanged: FlashVSR+ 2610.17552, 2610.13179, 2610.11746, ReCaVSR 2609.37831, RelayVSR 2609.37850, Amulet 2608.10423, Bezier-control-point VFI 2609.23408, ResStreamGS, TETRIS-4DGS. `2610.16423` (sprite keyframe interpolation) and `2610.12127` LVS (local view synthesis reusing previous views) seen; off-target / low relevance.
- **Archive.org:** the three Advanced Search queries again return 0 hits (window 2026-09-30 → 10-18). Wayback availability: GS-STVSR (2604.18047) and GRAPE (2503.06617) still no snapshots; GSASR query hit "Internet Archive: Temporarily Offline". All three abs pages still HTTP 200 live → no takedowns. Save Page Now request still advised.
- **GitHub (REST, unquoted queries):** no new GS-SR repos (only `ResonantBraneSplatting`, 1★, ignore). Active leads, all wrappers over vendor models, not threats: `ShyVortex/dlss-unlocked` (843★, pushed today), `Likely7/Veyra-NRVideo` (758★, DLSS 5 video-player/capture enhancement, pushed today — consumer demand signal for neural enhancement of arbitrary video), `Echo-Storm/ls-addon-manager`, `HelixSR-Manager`, OpenImagingLab `FlashVSR` (1.9k★, CVPR 2026, diffusion VSR; off-budget comparator), `mpv-AnimeJaNai` (764★, lightweight real-time SR models in mpv — reference for small-model real-time SR).
- **News (Google News RSS, headline-level unless noted):**
  - **VideoCardz 10-17:** NVIDIA's faster DLSS 5 model already runs on RTX 40 within hours of release (RTX 40 support also in morning brief); NVIDIA primary page (10-13) confirms 2× faster model, ~118M params, −20% 4K VRAM.
  - **Universal MFG Unlocker 1.5.0 (10-13):** adds faster DLSS Preset C for RTX 40/30 and **XeSS Multi-Frame-Generation on AMD & NVIDIA GPUs** — headline-level; indicates XeSS MFG is being used cross-vendor in the wild (DP4a-style portability is working for FG).
  - **FSR:** FSR 4 handheld support by Dec 31 / Steam Deck waiting (tech-insider, unverified lead); HelixSR brings DLSS 3.7-class SR to Steam Deck (10-13, SteamDeckHQ); "DLSS on AMD GPU in Cyberpunk" (PC Gamer 10-14). Community is routing around vendor lock-in faster than vendors ship — supports OSS's vendor-neutral positioning.
  - **Apple/MetalFX, Intel Arc, UE/Unity:** nothing new beyond the morning brief (MetalFX only via GPTK 4 beta test coverage; The Matrix Awakens on PS5 with DLSS 5 headline from GameGPU is unverified).

### Evening action-item deltas
- [ ] Dashboard/trainer stall now **44h** — still highest priority; if pico-005 is parked, update banner/ETA.
- [ ] Skim Token-Group Quantization (2610.16261) with QuantSR+ for the static-calibration PTQ recipe (#9/#4).
- [ ] Look at `mpv-AnimeJaNai` model-size/latency choices as a real-time small-SR reference (student design, #4).

## Sources
- [OpenDLSS-NR (GitHub)](https://github.com/maanHimself/OpenDLSS-NR) · [RunAIHome write-up](https://runaihome.com/blog/opendlss-nr-rtx-40-series-home-lab-neural-rendering-2026/) · [Gigazine](https://gigazine.net/gsc_news/en/20261002-opendlss-nr/)
- [arXiv 2610.16298 Integer-Only Training](https://arxiv.org/html/2610.16298v1) · [2610.13179](https://arxiv.org/abs/2610.13179) · [2610.11746](https://arxiv.org/abs/2610.11746) · [2610.09098](https://arxiv.org/html/2610.09098v1) · [Frozen History 2610.15752](https://arxiv.org/html/2610.15752v1)
- [AMD: Upscale Everything](https://www.amd.com/en/developer/resources/technical-articles/2026/upscale-everything-super-resolution-across-amd-hardware-.html) · [GPUOpen UE plugin](https://gpuopen.com/unreal-engine/) · [FSR Redstone after 10 months](https://framesgames.com/amd-answer-to-nvidia-redstone/)
- [NVIDIA: DLSS 5 faster model](https://www.nvidia.com/en-us/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/) · [Intel XeSS SDK](https://github.com/intel/xess) · [PCGH Arc news](https://www.pcgameshardware.de/Intel-Arc-Grafikkarte-267650/News/) · [Apple Metal what's new](https://developer.apple.com/metal/whats-new/)
- [OSS dashboard](https://opensupersampling.com)
