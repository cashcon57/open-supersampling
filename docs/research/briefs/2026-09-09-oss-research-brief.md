# OSS Research Brief — 2026-09-09 (evening update)

> Supersedes the 12:05Z run of this date. New since then: DLSS 5 launched (Sept 1–3), AMD FSR 4.1 shipped for RDNA 3, GameSR surfaced at ICLR 2026, HyperGS feedforward gaussian video found, PSSR 2.0 talk details, XeSS 3 MFG driver. Trainer outage continues.

## Project Status Summary

- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke` (shallow local checkout; single reachable commit).
- **Open issues:** 12. Architecture/runtime: #1/#3/#12/#13/#14 (latent rasterizer, counting-sort tile bin, packed canvas, Nsight, VRAM). Shipping: #4/#7/#8/#9 (≤1M student, FSR 4 teacher signal, vendor kernels, reparameterized backbone).[1]
- **README snapshot:** pre-alpha v7 research — native `(x,y,t)` Gaussian canvas, fractional-time OSS-FX teacher; ≤1M TensorRT student and non-CUDA backends unimplemented.[2]
- **Training dashboard:** `srcnn-v7.0-pico-005` unchanged at **36,650 / 100,000**, last `sr_charbonnier` 0.0236 / `total` ~0.17 (loss data unchanged since Sep 8). `status.json` (checked 2026-09-10T00:01Z): trainer **offline, stuck 245h** — outage counter rose 233h → 245h today; watcher/Worker/R2 all healthy.[3][4]
- **Operational conclusion:** still a hard trainer outage on the 3080 Ti host, not recovery. Publishing infra is fine; nothing is consuming GPU. The "training now" headline on the dashboard is stale and now misleads visitors.

## New Papers (arXiv)

Dedup pass against `BIBLIOGRAPHY.md` (57 unique arXiv IDs) done for every candidate. Items already known (GS-STVSR, NanoVSR, Amulet, TileGS, GSASR line, M³ISR) are not repeated below except where there is a status change.

### Gaussian Splatting / Temporal

- **HyperGS: Fast and Generalizable Gaussian Video Representation** — Fatimah Zohra, Chen Zhao, Shuming Liu, Yahya Al Malallah, Bernard Ghanem (KAUST); 2026-07-13 (v2); arXiv:2607.11500. Feedforward, optimization-free Transformer that predicts per-frame 8-parameter Gaussians from any video in one pass — no per-video optimization — with a rank-based geometric regularizer fixing a "needle-like degeneration" of naive Gaussian prediction. Claims 10⁴–10⁵× faster encoding than per-video optimization and out-of-distribution generalization at higher resolutions. **Why it matters to OSS:** the per-frame 8-parameter Gaussian head + feedforward amortization is structurally the same move as OSS's canvas-from-features student, but in the video-representation domain. Their degeneration regularizer is directly transplantable to v7 canvas training if opacity/shape collapse appears at scale; and their "feedforward beats optimize-per-input" result is evidence for the student-side (not per-scene) approach OSS already bets on. **Recommended: add to bibliography; deep-read the rank regularizer.**[5]

### Super-Resolution / Upscaling

- **GameSR: Real-Time Super-Resolution for Interactive Gaming** — Haseeb Ur Rehman, Shervin Shirmohammadi, Mohamed Hefeeda; ICLR 2026 submission (OpenReview, modified 2026-02-11; no arXiv ID found). Lightweight engine-independent SR operating directly on *encoded* game frames: reparameterized conv blocks + PixelUnshuffle + lightweight ConvLSTM, up to 240 FPS client-side, 30–60% cloud-gaming bandwidth reduction, near-parity with DLSS/FSR on CS2/OW2/TF2 **without engine data access**. **Why it matters:** same reparameterization recipe as #9, but validated specifically on game content (where OSS's SRGD-style training may not transfer); it is also a direct "no-engine-integration" competitor to the OSS deployment story — its existence weakens the claim that engine MVs/G-buffers are *required* for acceptable quality, so OSS's engine-MV premise needs an explicit quality-gap measurement against a decode-only baseline like this. **Recommended: add to bibliography (OpenReview link) as a competitor control.**[6]
- *(No new arXiv SR papers inside the 14-day window cleared the materiality bar; NTIRE 2026 efficient-SR report (arXiv:2604.03198, DWMamba 0.674M progressive KD) was re-checked and remains a useful #4 reference but is already covered by prior briefs' student-baseline work.)*

### Frame Extrapolation / Generation

- No new frame-extrapolation papers beyond **Amulet** (arXiv:2608.10423, already in bib and still the architectural threat to watch; nothing new published since Aug 11). cs.GR "frame generation" queries returned no new 2609 submissions.

### Inference Optimization

- No new TensorRT/INT8/kernel-fusion papers of consequence. `2609.07012` (AstraMoE-SR) is satellite-domain; rejected. The packed-canvas → fused-TensorRT sequencing from #12/#4 remains the right order.

## Archive.org Findings

- Wayback CDX coverage of tracked papers: **GSASR** 6 captures (latest 2025-12-18), **GRAPE/GaussianSR** (2407.18046) 6 captures (latest 2025-12-03), **TileGS** 1 capture (2026-09-05, current), **GS-STVSR** (2604.18047) **zero indexed captures**, **Amulet** zero captures. No evidence of takedowns — this is Wayback lag/coverage, same conclusion as prior briefs. GS-STVSR remains the only one of the four with no safety copy anywhere; if it matters, mirror the PDF locally now.
- Archive.org full-text `"gaussian splatting" super-resolution`: 3 hits, all unrelated GitHub starred-list mirrors. `"frame generation" deep learning` / `"real-time upscaling" game`: nothing usable. No archived alternate versions of any tracked paper found this run.

## GitHub / Open Source

- **OptiScaler** — 10,552 stars, pushed today (2026-09-09). OptiFG (their own DX12 FG, not just vendor FG passthrough) continues to mature; the ecosystem is converging on "one shim, all vendors," which is exactly OSS's long-term integration slot. Watch their DX12 hook surface for clean-room reference, not code.[7]
- **ShyVortex/dlss-unlocked** — grew 212 → **244 stars** since this morning's brief, still pushed today. Same assessment: demand signal for interception layers; do not import behavior/claims.[8]
- **arm/accuracy-super-resolution (+ UE plugin)** — Arm ASR ships as UE 5.3–5.5 plugin + library + upcoming Unity plugin; mobile temporal upscaling. Not new this window, but confirmed as another "engine-plugin-first" distribution model OSS should match at integration time.[9]
- GitHub search (GS+SR, frame-gen, neural upscaler, sorted by recency): nothing new above noise. `NodeNestor/prism-inference` (1 star, claims GAN-based DLSS/FSR/XeSS replacement) is unvalidated vaporware-grade; `IE-SRGS` (1 star) rejected again as 3D-view-synthesis. No credible new open-source DLSS/FSR analysis tool appeared.

## News

### The big one: DLSS 5 is out
- **NVIDIA DLSS 5 / 3D-Guided Neural Rendering launched Sept 1–3, 2026, first in NBA 2K27** (RTX 50-series only, also GeForce NOW). It is *not* an upscaler change: it takes the engine's rendered frame + motion vectors as an "unyielding foundation" and *generates lighting/material detail* (SSS, hair/foliage transmission, contact shadows + GI) as the final pipeline stage, deterministically, one-frame-in/one-frame-out, on Tensor Cores, integrated via Streamline like the rest of DLSS. NVIDIA frames it as "23 of every 24 pixels AI-drawn" now spanning the DLSS 4.5 suite.[10][11]
- **Performance cost is severe:** NVIDIA itself confirmed 50–60% frame-rate cuts; PC Gamer measured up to −73% on RTX 5080 (183 → 49 fps in 2K27); Digital Foundry ~40 vs ~95 fps; Hardwareluxx −32.6%. Reception fractured between "photorealistic game-changer" and "AI slop," but hands-on tests (Tom's Hardware, Beebom, Tom's Guide) mostly conclude the quality is real and the slop narrative overblown — at a huge perf price, 1080p-viable even on a 5060.[12][13][14]
- **Positioning read for OSS:** DLSS 5 *concedes* the deterministic, engine-buffer-conditioned premise OSS was built on ("engine frame defines what must remain") and then spends its entire budget on generative material/lighting enrichment at 2× frame cost. That validates OSS's MV-anchored temporal SR as the correct family, and hands OSS a clean differentiation axis: **OSS targets reconstruction (ms-scale, no frame-rate cliff); DLSS 5 targets enrichment (tens-of-ms, 50%+ cost).** These are adjacent but different products. Risk: if "neural rendering" becomes the expectation at 4K, plain SR+FG may read as stale in marketing terms even when it is the right engineering. Keep OSS's public claim scoped exactly as the Sep 5 brief set it: deterministic, engine-buffer-conditioned SR + extrapolation.

### AMD
- **FSR 4.1 ML upscaling is officially open to Radeon RX 7000 (RDNA 3)** via the new FSR UE plugin (FSR SDK 2.3, UE 5.8, released ~Sept 1–3; RX 9000/7000 auto-select ML path, older HW falls back to FSR 3 analytical). One plugin now bundles ML upscaling + analytical upscalers + FG + Ray Regeneration 1.2.0.[15] Impact on OSS: (a) widens the pool of games shipping FSR 4.1 output that issue #7 wants as a distillation teacher signal — the RDNA-3 route makes teacher data acquirable on far more machines; (b) the SDK bundling pattern (SR+FG+RR in one integration) is the competitive baseline OSS's "one architecture instead of two" claim must beat on integration ergonomics.
- Standing lead unchanged: AMD continues to hint at open-sourcing FSR 4 sources after the August accidental release (Tom's Hardware). If that lands, OSS's #7 teacher-signal story gets license-clean; watch GPUOpen for the drop.[16]

### Intel / Sony / Apple / engines
- **Intel:** driver 32.0.101.8991 (Aug 25) brings **XeSS 3 Multi-Frame Generation to all Arc A/B and Core Ultra iGPUs**, plus CapFrameX/RTSS overlay integration; XeSS 3 MFG is now shipping at the driver level across the stack. Confirms Intel is fully in the FG arms race; no new SR-quality model detail.[17]
- **Sony PSSR 2.0 (SIGGRAPH 2026 talk, Aug):** the most architecturally interesting console news of the month — PSSR moved from a *color-predicting* CNN to a **kernel-predicting network (KPN)**: neural net only finds edges/detail-vs-noise, conventional GPU math does HDR blending and reconstruction. Network runs at ≤540p, kernels defined at input scale with elliptical filtering, and **training dropped from 4 GPU-months to under 1 GPU-week** via quantization changes. No transformer involved.[18] **Why it matters:** this is the same "let the network predict structure, let fixed math do color" split that OSS's Gaussian canvas makes even more explicitly (Gaussians *are* spatially-structured kernels with analytic composition). PSSR 2's input-scale elliptical kernels and 540p neural stage are independent validation of v7's coarse-neural/analytic-compose split and a citable precedent for the v7.1 training-cost narrative (OSS's 100k-step pico runs vs Sony's week-long training).
- **Apple:** no change this window; Metal 4/MetalFX Frame Interpolation + MV-accepting temporal upscaler (WWDC26 material, covered Sep 5) remains the Metal backend design target. **Unreal/Unity:** the FSR UE 5.8 plugin (above) is the only engine-upscaling change verified; Unity's Arm ASR plugin is "coming soon."

## Action Items / Follow-ups

- [ ] **P0 (unchanged, worsening):** recover or formally stop `srcnn-v7.0-pico-005` — now 245h frozen at 36,650 with healthy infra. Also fix the dashboard "training now" headline to read from `status.json` trainer state, since it currently misleads visitors.
- [ ] **Deep read (new):** HyperGS (arXiv:2607.11500) — extract the rank-based Gaussian-degeneration regularizer and test it on v7 canvas training; their 8-param feedforward Gaussian head is the nearest published relative of the OSS student.
- [ ] **Competitor test (new):** GameSR (ICLR 2026 / OpenReview wnJkdo5Gu9) — build the engine-free decode-only control: reparam-CNN + ConvLSTM on encoded frames vs v7 engine-MV canvas, measured on the same game clips. Quantify what engine MVs actually buy before leaning on them in the README claim.
- [ ] **#7 enabler:** RDNA 3 FSR 4.1 widens teacher-signal availability — refresh the FSR4-teacher capture plan to include RDNA 3 hosts, and track the potential FSR 4 open-sourcing for a license-clean teacher.
- [ ] **Positioning:** adopt the "reconstruction vs enrichment" split in OSS docs — DLSS 5 = ms-costly generative material enrichment (RTX 50 only, −50–73% fps), OSS = deterministic SR+extrapolation at <2ms. Do not compete on "neural rendering" language.
- [ ] **Citable precedent:** add PSSR 2.0 KPN slide deck (advances.realtimerendering.com/s2026/#pssr) to the v7.1 memo as external validation of neural-structure/analytic-color split and week-scale training claims.
- [ ] **Archive:** mirror GS-STVSR PDF locally — it has zero Wayback captures and is OSS's closest prior formulation.

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[2] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[3] https://opensupersampling.com/data.json
[4] https://opensupersampling.com/status.json
[5] https://arxiv.org/abs/2607.11500
[6] https://openreview.net/forum?id=wnJkdo5Gu9
[7] https://api.github.com/repos/OptiScaler/OptiScaler
[8] https://api.github.com/repos/ShyVortex/dlss-unlocked
[9] https://developer.arm.com/mobile-graphics-and-gaming/arm-accuracy-super-resolution
[10] https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/
[11] https://www.gadgets360.com/games/news/nvidia-dlss-5-graphics-upscaler-launch-announced-september-3-nba-2k27-visual-concepts-11991022
[12] https://www.pcgamer.com/hardware/graphics-cards/dlss-5-can-slash-performance-by-up-to-73-percent-on-an-rtx-5080-gaming-pc-in-nba-2k27-and-im-left-wondering-what-kind-of-mega-rig-youd-need-to-justify-the-hit/
[13] https://www.tomshardware.com/video-games/pc-gaming/we-tested-dlss-5-in-nba-2k27-with-every-rtx-50-series-gpu-first-official-release-comes-with-a-big-performance-hit-but-almost-every-blackwell-card-can-run-it-at-1080p
[14] https://www.resetera.com/threads/digital-foundry-dlss-5-tested-nba-2k27-image-quality-benchmarks-mods-and-more.1625434/
[15] https://www.igorslab.de/en/amd-fsr-4-1-radeon-rx-7000-plugin-unreal-engine-5-8/
[16] https://www.tomshardware.com/pc-components/gpus/amd-hints-at-officially-open-sourcing-fsr-4-upscaling-and-frame-generation-technology-in-the-wake-of-accidental-release-accidental-release-may-have-forced-the-companys-hand
[17] https://www.windowscentral.com/hardware/intel/intel-xess-3-multi-frame-gen-driver
[18] https://www.digitalfoundry.net/news/2026/08/sony-engineer-reveals-pssr-2-0-secrets-in-siggraph-2026-talk
