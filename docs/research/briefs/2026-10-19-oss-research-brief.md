# OSS Research Brief — 2026-10-19

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` — HEAD unchanged since 2026-06-08 (shallow clone). Working tree still carries uncommitted research briefs (2026-08-11 → 10-18) and modified `BIBLIOGRAPHY.md`.
- **Open issues (10, unchanged):** #14 D3 VRAM footprint, #13 Nsight L2 profile (H004), #12 packed low-rank canvas layout, #10 decoupled G-buffer encoder, #9 RepVGG/ECBSR/RepSR student backbone, #8 cross-vendor kernel ports, #7 FSR 4 binary as teacher signal, #6 pico-002 launch, #5 No-ML shader variant, #4 HAT-Tiny student distillation.
- **README:** unchanged (v5-pixel-temporal still the only measured row; ≤1M student untrained).
- **Training dashboard (opensupersampling.com, browser-rendered):** `srcnn-v7.0-pico-005` still at step **36,650 / 100,000**, loss 0.19268, viz @ 36,000. Health row: **"Trainer — offline, stuck at step 36650 for 80h"** (68h at 10-18 20:00 → 80h now; the trainer is genuinely not advancing, ~4 days now). Watcher healthy (0s), CF Worker HTTP 200 113 ms. Header still says **"live"**, "updated just now", **"ETA ~1.1d"** — contradicts the offline state. Static fetch also still shows the stale tier-table line "Currently training: srcnn-v6.2-pico-002" (docs drift vs. v7.0-pico-005).

## New Papers (arXiv)
Tooling: `curl https://export.arxiv.org/api/query` works (python-in-kernel urllib fails SSL verify — use curl). 18 fielded queries (A–E), plus a keyword scan of today's (Mon 10-19) `cs.CV/new` + `cs.GR/new` listings (126 + 4 entries). Filtered against `BIBLIOGRAPHY.md` and prior briefs.

### Gaussian Splatting / Temporal
- **Direct Conditional Parameterization for N-Dimensional Splatting** (arXiv:2610.22778, Zhongpai Gao, Benjamin Planche, Van Nguyen Nguyen, Meng Zheng, Anwesa Choudhuri, Arun Innanje; 2026-10-16; NeurIPS 2026) — N-D splatting (N-DGS / Universal Beta Splatting) lifts each primitive into a joint Gaussian over position + conditioning vars (view dir, **time**) and *conditionally slices* it per render query, which needs per-primitive matrix inversions every forward. They replace the covariance-derived readout with explicit per-effect params (Cholesky precision factor for opacity + factorized query→position regression operator with learnable per-dim coupling): up to +1.26 dB, **slicing 6.9–7.7× faster**, full-render 1.25–1.91× faster. **Why it matters to OSS:** this is the closest published relative of the v7 "N-D Gaussian canvas with native time axis" — the α∈(0,1) extrapolation path *is* a conditional slice along t. Their direct (non-inverted) conditional readout is a concrete way to make the time-slice cheap for the rasterizer and fits the H002 low-rank/H004 budgeted-scheduler line; also prior art to cite for the N-D canvas. **Top deep-read this run.**
- **Frame-Consistent Feed-Forward 3DGS from Sparse Unposed Images** (arXiv:2610.23039, Vinasirajan & Rodrigo; 2026-10-16) — fixes coordinate-frame/translation-scale mismatches in a frozen VGGT→TokenGS pipeline. Not material (static NVS, no temporal/SR).
- Already tracked, unchanged: ResStreamGS (2610.21121), TETRIS-4DGS (2610.14663), SparseView4DGS (2610.04606), GS-STVSR (2604.18047). Surfaced, not material: Edit4D (2610.21668), DriveWorld3D (2610.22450), KFS⁴LAM (2610.22505, endoscopic GS-SLAM), X-Splat (2610.23237), LonGaussian (2610.23250), TriFold (2610.23050).

### Super-Resolution / Upscaling
- Nothing new on target. Title scan of today's cs.CV listing found no SR/VSR/upscaler papers. Carried reads: Frozen History (2610.15752), FlashVSR+ (2610.17552, diffusion/off-budget), 2610.09098 (MCU SR).

### Frame Extrapolation / Generation
- Nothing new on arXiv (only unrelated TGuide, 2610.14657, autoregressive video diffusion speedup). Non-arXiv prior art: Sony patent (see News).

### Inference Optimization
- Nothing new on target (SpikeQ 2610.23046 is spiking-NN PTQ — off-topic; FlashControl 2610.22537 one-step distillation for controllable generation — off-topic). Carried skims: QuantSR+ (2605.22351), Token-Group Activation Quantization (2610.16261).

## Archive.org Findings
- Advanced Search (2026-dated): `"gaussian splatting" super-resolution` → 0; `"frame generation" deep learning` → 1 (OSF registration for an unrelated long-video diffusion paper, MoCha-LD); `"real-time upscaling" game` → 0. No research value.
- Wayback availability API worked this run (no 429): snapshots exist for GS-STVSR 2604.18047 (2026-04-21), GSASR 2501.06838 (2026-06-07), GRAPE 2503.06617 (2026-02-11). All three arXiv abs pages still HTTP 200 live → **no takedowns**. Archived copies are already present, so Save Page Now is no longer needed for these three.

## GitHub / Open Source
(REST search, `pushed:>=2026-10-12`.)
- **No new Gaussian-splatting SR / VSR / frame-extrapolation repos.** Only `GiulioFede/ResonantBraneSplatting` (1★, ignore).
- Frame-gen / upscaler activity, all wrappers/hobby: `ShyVortex/dlss-unlocked` (852★), `Echo-Storm/ls-addon-manager` (190★), `hy300leosquizz-ctrl/ZeroFG` (9★, Vulkan mobile FG), `menghuihuan44/fluid` (0★), `Denarmo/HelixSR-Manager` (13★; HelixSR community Steam Deck SR), `PhoenixTheSage/SE-FG`. No threat.
- Modder activity: Universal MFG Unlocker v1.5.0 brings DLSS FG Preset C to RTX 30 (DX12 only, validated offline on Ampere) and adds XeSS MFG 2–6× via OptiScaler plugin on any vendor (TechPowerUp 10-19). Carried: DLSS-NR-on-AMD 0.8.0, OpenDLSS-NR.

## News
- **AMD — now CONFIRMED (upgrade from last week's "rumor"):** Jack Huynh told TheElec on **2026-10-08** (reported by Digital Foundry, The Verge, TechPowerUp, Windows Central, HWBusters) that FSR 4 reaches APUs, gaming laptops and handhelds **by end of 2026** via a **"newly developed dedicated lightweight model"** for existing APUs lacking ML hardware, while new APUs (likely RDNA 4-class) run the full model. No frame times, no word on FG/Ray Regen. Context: FSR 4.1.1 INT8 path for RDNA 3 shipped in FSR SDK 2.3 / UE 5.8 plugin (UE 5.3–5.8, released Aug 2026); RDNA 2 official support 2027. **Relevance:** a vendor-shipped small, non-FP8, low-latency SR net is exactly OSS's ≤0.4M Pico/DP4a target tier; the bar for "cheap neural SR on non-matrix hardware" is about to be set publicly (benchmark target for #4/#7/#8; first independent frame-time numbers will be a baseline for the Pico tier).
- **NVIDIA:** DLSS 5 faster model (−24% params → ~118M, −20% 4K VRAM, 2× faster; NBA 2K27 +1.4×; shipped 10-16 on RTX 50 / RTX Spark / GeForce NOW). **RTX Spark laptops (Grace+Blackwell, up to 128GB unified) available 10-16 from $2,599.** DLSS FG **Preset C** (beta, 10-13): replaces FG's *second* AI model (optical flow for UI/RT effects) with a lighter non-NN algorithm, up to 35% faster on RTX 5060 at a small quality cost — it ships in **RTX Remix on 10-23** alongside DLSS 5; Hogwarts Legacy DLSS 5 on 11-05. DLSS 4.5 controls on GeForce NOW; Broadcast 2.3 adds 4K video upscaling + 4× FG for prerecorded video. **Relevance:** NVIDIA is itself cutting FG cost by swapping a NN for a cheap algorithm — supports OSS's no-ML/near-zero-marginal-cost extrapolation thesis (#5, OSS-FX) but narrows the "FG is expensive" argument.
- **Sony (carried, details sharpened):** patent application US 2026/0320740 A1 "Frame Generation Super Sampling" published 10-15; inventor Denis Dupeyron; encoder network → compact embedding → decoder predicts next frame; optional motion vectors + controller input; **self-supervised on the video stream** (compare prediction vs. real next frame). No latency/IQ numbers. Self-supervised next-frame training is worth noting for the OSS-FX training recipe; claims still unread.
- **Intel:** nothing new (XeSS 3 MFG; plugin 3.1 for UE 5.8; XeSS FG override bug acknowledged 10-06).
- **Apple:** no new item this week. Background (WWDC26): redesigned MetalFX temporal upscaler running on Neural Engine + Neural Accelerators of M5 Pro/Max, motion-vector passthrough, `MTLFXFrameInterpolator` — Metal port (#8) will face a native neural temporal upscaler.
- **Engines:** UE 5.8 (Aug 2026) now has AMD FSR plugin 4.1.1 (earlier forum reports of breakage predate the plugin); Intel XeSS 3.1 UE 5.8 plugin. Nothing new on Unity.
- RTX 5090 gone from NVIDIA Marketplace (hardware context only).

## Action Items / Follow-ups
- [ ] **Dashboard/trainer (highest priority, 80h and rising):** pico-005 stalled at 36,650. Check trainer process / GPU / rsync on 3080 Ti host, resume or mark parked; drop the "live" badge and "ETA ~1.1d" while offline; fix stale "Currently training: srcnn-v6.2-pico-002" tier-table line.
- [ ] **Deep-read 2610.22778 (direct conditional parameterization for N-D splatting)** → note against v7 time-axis slice cost, H002/H004; consider cite in README architecture section; check project page https://gaozhongpai.github.io/dgs/ for code.
- [ ] Watch AMD lightweight FSR 4 for APUs (due by 12-31): capture frame times/INT8-DP4a details once reviewers get it → #4/#7/#8 baseline.
- [ ] Read Sony US 2026/0320740 A1 claims vs OSS-FX / no-ML fallback; note self-supervised next-frame training idea under H005/H009.
- [ ] Evaluate NVIDIA FG Preset C approach (lighter algorithmic flow in place of 2nd NN) as a reference design for the no-ML variant (#5).
- [ ] Carried: Frozen History (2610.15752) for H009; OpenDLSS-NR / DLSS-NR-on-AMD HIP kernel layout for #8; QuantSR+ / Token-Group Quant for #9/#4; ResStreamGS → H004/#12; HelixSR as small SR latency reference.

## Sources
- arXiv Atom API + `cs.CV/new` 2026-10-19 (IDs above); [2610.22778](https://arxiv.org/abs/2610.22778)
- [Digital Foundry — AMD lightweight FSR 4](https://www.digitalfoundry.net/news/2026/10/amds-jack-huynh-fsr-4-is-coming-to-handhelds-via-a-new-lightweight-model) · [The Verge](https://www.theverge.com/games/1008353/amd-will-bring-fsr-4-to-handhelds-by-the-end-of-2026) · [TechPowerUp](https://www.techpowerup.com/353477/amd-confirms-fsr-4-is-coming-to-apus-by-the-end-of-2026)
- [NVIDIA — DLSS 5 faster model / RTX Remix](https://www.nvidia.com/en-us/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/) · [VideoCardz — FG Preset C](https://videocardz.com/newz/nvidia-introduces-up-to-35-faster-dlss-frame-generation-mode-for-rtx-40-and-rtx-50-gpus) · [TechPowerUp front page 10-19 (Sony patent, MFG Unlocker)](https://www.techpowerup.com/)
- [GPUOpen — UE plugin](https://gpuopen.com/unreal-engine/) · [Apple WWDC26 Metal](https://developer.apple.com/wwdc26/guides/metal/) · [OSS dashboard](https://opensupersampling.com)

---

## Evening update — 2026-10-19 20:00 EDT

**Delta since the 08:00 brief: no new target-relevant papers; one dashboard escalation; two minor news items.**

### Project status
- HEAD still `b44278a`; 10 open issues unchanged (#14 #13 #12 #10 #9 #8 #7 #6 #5 #4); README unchanged.
- **Dashboard:** `srcnn-v7.0-pico-005` still step **36,650 / 100,000**, loss 0.19268, viz @ 36,000. Trainer row now **"offline — stuck at step 36650 for 92h"** (80h this morning → ~3.8 days). Watcher/CF Worker (118 ms)/R2 all healthy. Header still shows **"live"**, "updated just now", **"ETA ~1.1d"**, and "v7.0 Pico OSS-FX is training right now" — all misleading while the trainer is dead. Still unfixed.

### arXiv
- 17 fielded A–E queries via `curl` (python urllib still fails SSL verify) + latest cs.CV/cs.GR listing check. Nothing new post-10-16 on GS+temporal, SR, frame extrap/gen, or inference-opt. Newest hits are all already tracked (2610.22778 dGS, 2610.21121 ResStreamGS, 2610.17552 FlashVSR+, 2610.13179, 2610.11746, 2609.37850, 2609.37831 — diffusion/VSR, off-budget).
- Surfaced, not material: Holi-Spatial 360 (2610.19502, sparse-panorama generative recon), DoReMe (2610.19120, MRI + GS). No BIBLIOGRAPHY.md addition this run.

### Archive.org
- Advanced Search (2026-10-01→12-31) for all three queries: 0 hits. Wayback still has snapshots of GS-STVSR 2604.18047 (2026-04-21), GSASR 2501.06838 (2026-06-07), GRAPE 2503.06617 (2026-02-11); live arXiv abs pages all HTTP 200 — no takedowns.

### GitHub
- Nothing new of note (`pushed:>=2026-10-12`): no GS-SR/VSR/extrapolation repos. Same wrapper/hobby set (dlss-unlocked 855★, HelixSR-Manager, ZeroFG, SE-FG). Modder ecosystem: `jatelop8/DynamicShaderFrameGen` 0.48 (DLSS5 variant) and `janblade/OptiScaler-F5-DLSSNR-Multipass` v0.1.32 (per-pass model presets, "faster optical F5Low frame generation") — community FG-over-DLSS5 tooling iterating weekly; no architectural threat.

### News (new vs. morning)
- **Intel:** working on **XMX-accelerated XeSS in Valve's Proton** (TechPowerUp 353884, 10-19) → native XMX XeSS on Linux; raises the bar for OSS's Linux/Level Zero story (#8). PCGH also reports DLSS 5 neural rendering running on Intel Arc via community port.
- **Kunlun Matrix-Game 4.0** (open-sourced 10-15): video-foundation-model "game" at 480p/60 FPS on one RTX 5090, <150 ms latency, game engine keeps world state while the video model renders. Different problem (generative world model), but another data point that generative-frame approaches are far off the <2 ms budget — OSS's engine-MV-driven canvas remains the cheap path.
- **Unity** 6000.3.26f1 (10-08): routine. **NVIDIA:** RTX 5090 discontinuation report denied by NVIDIA (10-16). Sony FGSS patent, AMD lightweight FSR 4, DLSS 5 faster model, FG Preset C: unchanged from morning.

## Action Items (changes)
- [ ] **Dashboard (escalated, 92h):** restart or park pico-005 trainer; set status to offline/paused and remove "live"/"ETA ~1.1d"/"training right now" while stuck; fix stale tier-table line. Header-vs-health-row contradiction is a credibility issue for reviewers.
- [ ] Watch Intel Proton XMX-XeSS merge (Linux upscaler baseline for #8).
- [ ] All prior items (2610.22778 deep-read, AMD lightweight FSR 4 frame times, Sony claims, FG Preset C → #5) carry over.
