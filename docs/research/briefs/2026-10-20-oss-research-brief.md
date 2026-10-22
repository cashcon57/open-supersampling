# OSS Research Brief — 2026-10-20

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (2026-06-08) — HEAD unchanged. Working tree still carries uncommitted research briefs (2026-08-11 → 10-19) and modified `BIBLIOGRAPHY.md`.
- **Open issues (10, unchanged):** #14 D3 VRAM footprint, #13 Nsight L2 profile (H004), #12 packed low-rank canvas layout, #10 decoupled G-buffer encoder, #9 RepVGG/ECBSR/RepSR student backbone, #8 cross-vendor kernel ports, #7 FSR 4 binary as teacher signal, #6 pico-002 launch, #5 No-ML shader variant, #4 HAT-Tiny student distillation.
- **README:** unchanged (v5-pixel-temporal still the only measured row; ≤1M student untrained).
- **Training dashboard (opensupersampling.com, browser-rendered):** `srcnn-v7.0-pico-005` still at step **36,650 / 100,000**, loss 0.19268, viz @ 36,000 — identical to 10-19. Health row: **"Trainer — offline, stuck at step 36650 for 104h"** (92h at 10-19 20:00 → 104h now: counter is consistent, ~4.3 days dead). Watcher healthy (0s), CF Worker HTTP 200 96 ms, R2 healthy. Header still says **"live"**, "updated just now", **"ETA ~1.1d"**, and "v7.0 Pico OSS-FX is training right now" — all contradict the offline state. Still unfixed.

## New Papers (arXiv)
Tooling: `curl` against the arXiv Atom API (python-urllib SSL fails in the kernel). The API rate-limited ("Rate exceeded"/empty) on roughly half of the first-pass fielded queries; failed queries were re-run unfielded (`all:`) with 5–12 s spacing. Final coverage: ~95 unique papers, all A–E themes hit at least once. Filtered against `BIBLIOGRAPHY.md` and the 10-19 brief.

### Gaussian Splatting / Temporal
- Nothing new on target since 10-16. Newest hits are already tracked: ResStreamGS (2610.21121), dGS/N-D conditional parameterization (2610.22778), TETRIS-4DGS (2610.14663), SparseView4DGS (2610.04606), GS-STVSR (2604.18047), Resonant Brane Splatting for arbitrary-scale SR (2606.29453), FoundDSR (2609.32323, depth SR with 2D GS).
- Surfaced, not material: **Drive2Gauss** (arXiv:2610.26585, Tong Zhao et al., 2026-10-19) — feed-forward 4D Gaussians from generated RGB+depth+flow driving video; Gaussian query decoder uses flow to associate queries across time. Autonomous-driving world-model scope, no SR/real-time budget; only conceptual overlap with flow-associated temporal Gaussians. Also PUGS (2610.17936, underwater dynamic GS), KFS⁴LAM, DriveWorld3D, Edit4D — off-topic.

### Super-Resolution / Upscaling
- **SCOPE-VSR: A Broad-Coverage Training Dataset for Video Super-Resolution** (arXiv:2610.24604; Gengyu Liao, Yechun Yong, Zheng Chen, Jianze Li, Ziqing Zhang, Linghe Kong, Yulun Zhang; 2026-10-17) — ~27k-video, semantically rebalanced open-domain VSR training set plus SCOPE-test, and GraCo-VSR, a one-step reliability-aware video restoration model (code: github.com/gyliao2768/SCOPE-VSR). *Why it matters to OSS:* low direct impact — it's real-world video (no engine motion vectors/depth) and a one-step diffusion-style restorer, off the <2 ms budget. Relevant only as a possible data source for cross-domain generalization eval (OSS currently has TartanAir-only held-out numbers). Not added to the bibliography (below relevance gate).
- Already tracked/off-budget diffusion VSR: FlashVSR+ (2610.17552), 2610.13179, Streaming-Aware Diffusion VSR (2610.11746), RelayVSR (2609.37850).

### Frame Extrapolation / Generation
- Nothing new on arXiv. Latest interpolation hits (Bézier-control-point motion estimation 2609.23408, Following Motion for VFI 2608.22861, Sprite keyframe interpolation 2610.16423) are generic VFI, not engine-driven extrapolation. Non-arXiv prior art remains the Sony FGSS patent (see 10-18/10-19 briefs).

### Inference Optimization
- Nothing new on target in this run's quantization/TensorRT/kernel-fusion queries (carry: QuantSR+ 2605.22351, Token-Group Activation Quantization 2610.16261).

## Archive.org Findings
- Advanced Search (date 2026-10-01 → 12-31): `"gaussian splatting" super-resolution` → 0; `"frame generation" deep learning` → 1 (unrelated media item); `"real-time upscaling" game` → 2 (GitHub-repo snapshot items for unrelated game ports). No research value.
- **Wayback availability API returned HTTP 429 on every call this run** (3 URLs, retried once) → takedown re-check of GS-STVSR 2604.18047 / GSASR 2501.06838 / GRAPE 2503.06617 **not re-verified today**. Carry-forward from 10-19: snapshots exist for all three (2026-04-21 / 2026-06-07 / 2026-02-11). Live `arxiv.org/abs/...` for all three returned HTTP 200 today → no takedown.

## GitHub / Open Source
(REST search, `pushed:>=2026-10-13`, unquoted queries.)
- **No new Gaussian-splatting SR / frame-extrapolation repos.** Only `GiulioFede/ResonantBraneSplatting` (1★, ignore).
- **New, topical — watch:** [`991o2o9/An-Ablation-Study-of-Auxiliary-Inputs-for-Lightweight-Neural-Upscaling-in-Real-Time-Rendering`](https://github.com/991o2o9/An-Ablation-Study-of-Auxiliary-Inputs-for-Lightweight-Neural-Upscaling-in-Real-Time-Rendering) (1★, created 2026-10-17, no license, two authors — Zhanbolot Kydyraliev & Amin Soltobekov; course-project workflow, related-work phase). Plans to ablate depth / motion / normals as auxiliary inputs to lightweight neural upscalers. README/weekly plan only — no code, data or results; paper PDF gated behind GitHub login. *Relevance:* same question as OSS #10 (decoupled G-buffer encoder) and the "which engine signals pay for themselves" design axis; if it produces results they're a free citable ablation. Not a threat. Not added to bibliography (no content yet).
- Wrapper / hobby / modder activity, no architectural threat: `ShyVortex/dlss-unlocked` (859★), `Echo-Storm/ls-addon-manager` (192★), `Likely7/Veyra-NRVideo` (790★, "DLSS 5" video-player enhancement), `Denarmo/HelixSR-Manager` (13★), `hy300leosquizz-ctrl/ZeroFG` (9★, Vulkan mobile FG), `menghuihuan44/fluid` (0★), `PhoenixTheSage/SE-FG`, `alex89213/espcn-mpv` (0★, train small ESPCN on own footage → compiled mpv shader), `the-database/mpv-AnimeJaNai` (765★), `xocialize/anime4k-metal` (0★, Anime4K as Metal compute shaders — small Metal reference).
- FlashVSR repo (OpenImagingLab, 1.9k★) still active — diffusion, off-budget.

## News
(Google News RSS headline-level unless marked primary. Primary pages opened: GamingOnLinux.)
- **Intel — XMX-accelerated XeSS under Proton, now with PRs (primary: GamingOnLinux 10-20; Phoronix/PC Gamer 10-19/20):** four PRs (Valve `wine` #369 adds a builtin `libxess.dll` forwarding D3D12 calls to Linux `libxess.so` via vkd3d-proton interop; vkd3d-proton #3379 enables needed Vulkan extensions; dxvk #5992 stops hiding Intel GPUs; fourth PR not read). All still open/unmerged. *Relevance:* Linux-side native XMX upscaling raises the bar for OSS's Linux/Level Zero/Vulkan story (#8) and shows the DLL-forwarding shim pattern OSS's integration path (DLL shim) would use under Proton.
- **NVIDIA:** nothing new since 10-16 (DLSS 5 faster model −24% params, FG Preset C beta, RTX Remix 10-23, Hogwarts Legacy DLSS 5 11-05). DLSS 4.5 controls on GeForce NOW (10-15). Community mods keep extending DLSS 5 features to other GPUs (DLSS-NR-on-AMD; headline-level, HotHardware/Tom's 10-18/19).
- **AMD:** no new primary item. Headline-level: FSR 4 handheld lightweight model by Dec 31 (already tracked); HelixSR ("DLSS 3.7-class" community SR on Steam Deck) coverage on Steam Deck HQ 10-13; GameGPU FSR 4 vs FSR 3 clarity in Wuchang. Steam Deck itself not confirmed.
- **Apple / Metal:** no new MetalFX item (RSS returned only unrelated Apple/CrossOver content). Carry: WWDC26 neural MetalFX temporal upscaler + `MTLFXFrameInterpolator`.
- **Sony:** PS5 Pro "QSSR" coverage (NoobFeed 10-15, headline-level) — Sony SR/FG tech is a growing console-side competitor; unverified details. Sony FGSS patent unchanged.
- **Engines:** nothing new for UE 5.8 / Unity upscaling. DLSS Swapper bulk-swap for DLSS/FSR/XeSS (10-18/19) — ecosystem tooling, confirms DLL-swap is the community integration norm (supports OSS DLL-shim path).
- **Hardware context:** Call of Duty MW4 reportedly blocks DLSS/FSR/XeSS when ray tracing is on (GameGPU, headline-level) — illustrates vendor-SDK integration friction OSS's engine-agnostic canvas could sidestep; unverified.

## Action Items / Follow-ups
- [ ] **Dashboard/trainer (highest priority, 104h and rising):** pico-005 stalled at 36,650 since ~10-16. Check trainer process / GPU / rsync on the 3080 Ti host; resume or mark parked. Drop the "live" badge, "ETA ~1.1d" and "training right now" copy while offline; fix stale "Currently training: srcnn-v6.2-pico-002" tier-table line.
- [ ] Deep-read arXiv:2610.22778 (direct conditional parameterization for N-D splatting) → v7 time-axis slice cost, H002/H004 (carried).
- [ ] Watch the **auxiliary-inputs ablation repo** (991o2o9/…) for code/results → feeds #10; re-check weekly.
- [ ] Watch Intel Proton XMX-XeSS PR merge status (wine #369, vkd3d-proton #3379, dxvk #5992) → Linux upscaler baseline + DLL-forwarding pattern for #8.
- [ ] Watch AMD lightweight FSR 4 (due by 12-31): capture frame times/INT8 details → #4/#7/#8 baseline (carried).
- [ ] Re-run Wayback takedown check for GS-STVSR/GSASR/GRAPE when archive.org stops 429-ing (not verified this run).
- [ ] Optional: evaluate SCOPE-VSR (2610.24604) as a cross-domain eval source for H009 generalization checks; low priority.
- [ ] Carried: Sony FGSS claims vs OSS-FX; FG Preset C as reference for no-ML variant (#5); Frozen History (2610.15752) for H009; OpenDLSS-NR / DLSS-NR-on-AMD HIP layout for #8; QuantSR+ / Token-Group Quant for #9/#4.

## Sources
- arXiv Atom API (curl) — IDs above; [2610.24604](https://arxiv.org/abs/2610.24604), [2610.26585](https://arxiv.org/abs/2610.26585)
- [GamingOnLinux — Intel XMX XeSS Proton PRs](https://www.gamingonlinux.com/2026/10/intel-submits-code-to-proton-to-improve-xess-to-use-the-fast-upscaler-xmx/) · [Valve wine PR #369](https://github.com/ValveSoftware/wine/pull/369)
- [Ablation-study repo](https://github.com/991o2o9/An-Ablation-Study-of-Auxiliary-Inputs-for-Lightweight-Neural-Upscaling-in-Real-Time-Rendering)
- GitHub REST Search; archive.org advancedsearch + wayback availability (429); Google News RSS (`when:7d`)
- [OSS dashboard](https://opensupersampling.com) (browser-rendered 2026-10-20)
