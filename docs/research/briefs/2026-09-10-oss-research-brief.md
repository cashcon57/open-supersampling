# OSS Research Brief — 2026-09-10

> Morning run. New since the 2026-09-09 evening brief: GS-STVSR PDF finally mirrored locally (closing the standing archive P0), DLSS Unlocked mod ecosystem exploded (Tom's Hardware coverage, NVIDIA failed to block it), Qualcomm Adreno Neural Fusion announced (mobile SR+FG on dedicated GPU matrix cores), AMD "Upscale Everything" dev article, three new 2609 arXiv entries. Trainer outage worsened again: 257h frozen.

## Project Status Summary

- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (shallow local checkout; single reachable commit, dated 2026-06-08). No new commits reachable from this clone.[1]
- **Open issues:** 12 (unchanged set, all last updated 2026-05-08). Architecture/runtime: #1/#3/#12/#13/#14 (latent rasterizer, counting-sort tile bin, packed canvas, Nsight, VRAM). Shipping: #4/#7/#8/#9 (≤1M student, FSR 4 teacher signal, vendor kernels, reparameterized backbone). #5/#6/#10 round out the v6.2 line.[1]
- **README snapshot:** pre-alpha v7 research — native `(x,y,t)` Gaussian canvas with V_xt Cholesky motion, OSS-FX = same architecture at a different `t_query`; ≤1M TensorRT FP8 student and non-CUDA backends still unimplemented. Positioning unchanged.[2]
- **Training dashboard:** `srcnn-v7.0-pico-005` frozen at **36,650 / 100,000** (36.7%). Last metrics: `sr_charbonnier` 0.0236, `fg_charbonnier` 0.0764, `total` 0.1927, `sr_lpips` 0.0927 — unchanged since Sep 8. `status.json` (checked 2026-09-10T12:00:55Z): trainer **offline, stuck at step 36650 for 257h**; watcher healthy (published 1s ago), CF Worker healthy (200 in 88ms), R2 healthy.[3][4]
- **Operational conclusion:** outage counter keeps climbing on schedule (233h → 245h → 257h across three briefs) while publishing infra stays green — this is a hard trainer death on the 3080 Ti host, not a stall. Third consecutive day. The dashboard "training now" headline remains misleading.

## New Papers (arXiv)

Dedup pass against `BIBLIOGRAPHY.md` (58 unique arXiv IDs at run start). Queries: GS+SR, GS+video (Aug 10–Sep 11 window), cs.CV+cs.GR recent, frame-gen/interp, VSR/efficient-SR, INT8/FP8/TRT+SR, GS rasterization/tiling. Items already in bib (GS-STVSR, HyperGS, TileGS, MotionCraft, MGMVFI, NanoVSR, Amulet) not repeated.

### Gaussian Splatting / Temporal

- **TruncGradGS: Improved 3D Gaussian Splatting via Truncated Gradient Updates** — Morales, Le-Pham, Atkins, Hua; 2026-09-03; arXiv:2609.03534. Truncating gradient updates during GS optimization to improve convergence stability and final quality. **Why it matters to OSS:** v7's diagnosed failure modes (v6.1 stippling, v6.1 plateau loss imbalance, D12/D13) are training-dynamics problems; truncated/clipped gradient schedules are a cheap ablation candidate for canvas training if opacity/shape instability reappears at longer horizons. Low-cost to test, plausibly synergistic with the HyperGS rank regularizer already queued for deep read.[5]
- **Compact Neural Appearance Models for Efficient Gaussian Splatting** — Hahlbohm, Condor, Franke, Eisemann, Magnor; 2026-09-04; arXiv:2609.05255. Replaces per-Gaussian SH appearance with compact neural appearance models for memory/render efficiency. **Why it matters to OSS:** per-Gaussian attribute compression is exactly issue #12 (packed low-rank canvas layout, 32B/Gaussian claim). Their appearance-compaction tricks are a direct reference for shrinking canvas attribute storage before the packed layout is finalized.[6]
- **MedGSSR: Generalizable Medical Image Super-Resolution 3D Reconstruction via Hierarchical Feed-forward Gaussian Splatting** — Wang et al.; 2026-09-06; arXiv:2609.06874. Hierarchical feed-forward (optimization-free) Gaussian prediction for SR reconstruction, medical domain. **Why it matters to OSS:** weak signal — medical 3D, not real-time 2D — but it is a third independent group (after HyperGS, SR3R) converging on feedforward generalizable Gaussian-SR, reinforcing the student-side bet OSS already made. Cite as trend evidence only.[7]

*(Checked and rejected: SelfLift 2609.02036 — few-step diffusion acceleration, wrong latency class for <2ms budgets; Reparam-3DGS palette editing 2609.03897 — color editing, not SR; RoGe/InceptionGS/GaussVid/NemoSplat — 3D/4D view synthesis, not frame SR.)*

### Super-Resolution / Upscaling

- No new arXiv SR papers inside the window cleared the materiality bar. NanoVSR (2607.10495) and MotionCraft (2608.08553) remain the standing references; NTIRE 2026 efficient-SR (2604.03198) stays the #4/#9 student-baseline anchor.

### Frame Extrapolation / Generation

- Nothing new beyond **Amulet** (2608.10423, still the architectural threat to watch; zero movement since Aug 11). Notably, the 2609 cs.GR/cs.CV window produced **no** single-network frame-extrapolation work — OSS-FX's "same canvas, different t_query" formulation still has no published direct competitor 5 months after GS-STVSR.

### Inference Optimization

- No new TensorRT/FP8/INT8/kernel-fusion papers of consequence this window. The #12 packed-canvas → fused-TensorRT sequencing remains the right order; Compact Neural Appearance (above) is the nearest new input to that work.

## Archive.org Findings

- **ACTION CLOSED: GS-STVSR PDF mirrored locally.** The 18.7 MB PDF (arXiv:2604.18047) is now at `docs/research/papers/2604.18047-gs-stvsr.pdf` (verified `%PDF-1.7`). Wayback still shows **zero captures** for both the abs and pdf URLs, so this local mirror is the only safety copy of OSS's closest prior formulation. Recommend committing it (or an extract) so it survives workspace cache eviction.[8]
- Wayback CDX spot-checks: HyperGS (2607.11500) 1 capture (2026-07-17); TileGS (2609.03613) 1 capture (2026-09-05); Amulet (2608.10423) check failed — **web.archive.org returned "Temporarily Offline"** during this run's CDX queries (the same outage that has made Amulet's capture status un-verifiable for several briefs). Retry Amulet next run.
- Archive.org full-text search: `"gaussian splatting" AND "super-resolution"` returns only unrelated GitHub starred-list mirrors; `"frame generation" deep learning` / `"real-time upscaling" game` — nothing usable. No archived alternate versions of tracked papers found.

## GitHub / Open Source

- **DLSS Unlocked (ShyVortex/dlss-unlocked)** — 244 → **273 stars** in ~24h, pushed today 11:12Z. Now covered by Tom's Hardware (Sep 8): all-in-one mod combining OptiScaler_DLSSNR + DLSS Enabler brings **DLSS 5 neural rendering + multi-frame gen to RTX 20/30/40**, routing FG through a modified FSR 3.1 path on older cards; claims up to 6× frame rates. Follow-up coverage: NVIDIA app update **failed to block** the MFG mod on RTX 40 (modders restored support within hours), a second mod runs **DLSS 5 on a second GPU** (+127% neural-rendered FPS, igor's lab), and a third got DLSS 5 onto **RDNA 4** (RX 9070 XT, 30 fps at 1080p today). Also a 600W/16-pin-melt report attributed to DLSS 5 power draw.[9][10][11]
  - **Read for OSS:** this is the community building OSS's exact deployment slot (per-title DLL interception, vendor-neutral, hardware-gate removal) faster than any vendor ships it. It is also a demand signal: people want DLSS-class quality on non-RTX-50 / non-NVIDIA hardware and will duct-tape three tools to get it. OSS's DLL-shim design (Sprint 7) should treat this ecosystem as the integration surface to join, not compete with — clean-room reference only, no code/claims import.
- **OptiScaler (optiscaler/OptiScaler)** — 10,552 → **10,569 stars**, pushed today (FSR 3.1 Dx11 library updates, nvngx replacement tooltips). Still the "one shim, all vendors" reference implementation; its DX12 hook surface remains the clean-room reference for OSS's shim.[12]
- GitHub search (GS+SR, frame-gen upscaler, neural upscaler, sorted by updated): nothing new above noise. `pranav-asthana/SplatSuRe` (CVPR 2026, 94★) confirmed as the official SplatSuRe repo — 3D multi-view, not a threat. No credible new open-source DLSS/FSR analysis or FG tooling appeared.

## News

### DLSS 5 modding wave + backlash (the week's story)
- The DLSS Unlocked ecosystem (above) means DLSS 5 + MFG now effectively runs on Turing/Ampere/Ada and even RDNA 4 via mods, despite NVIDIA's hardware gating — with NVIDIA visibly unable to stop it.[9][10]
- **RPCS3 devs publicly slammed DLSS 5 as an "AI-slop generator,"** saying the industry is pushing "more upscalers and frame generation to hallucinate games and hide their lack of optimisation," and that the emulator won't support it.[13]
- **Positioning read:** two tailwinds for OSS's Sep-5 framing. (1) The interception-layer ecosystem proves cross-vendor DLL shims are viable and demanded — OSS's planned integration path. (2) The deterministic-vs-generative split now has a loud public constituency: OSS's canvas is engine-buffer-conditioned reconstruction (deterministic, MV-anchored), which is precisely the alternative to "hallucinate the missing pixels" that RPCS3 is demanding. Consider citing the backlash (not the emulator) when OSS publishes its deterministic-reconstruction claims.

### AMD
- New dev article: **"Upscale Everything: Super-Resolution Across AMD Hardware"** — positions FSR Redstone SDK 2.2 (4 neural rendering features: Upscaling 4.1, Frame Generation 4, Ray Regeneration 1.x, Radiance Caching) as the SR story across RDNA 4/3/handhelds, plus a separate piece on SR acceleration on Ryzen AI NPUs (zero-copy tiling approach). No new quality numbers; it's a consolidation/positioning piece. Confirms AMD is marketing SR+FG as one bundled pipeline — the exact integration-ergonomics baseline OSS's "one architecture" claim must beat.[14][15]
- FSR 4.1 on RX 7000 via UE 5.8 plugin (covered Sep 9) unchanged; FSR 4 open-sourcing still promised, not shipped — watch GPUOpen.

### Qualcomm (new vendor move this window)
- **Adreno Neural Fusion announced (Sep 2, Snapdragon 8 Elite Gen 6):** dedicated AI matrix cores **inside the mobile GPU** + 18MB Adreno HPM, unifying neural processing, AI super-resolution, **and frame generation in one graphics pipeline**; FG uses motion vectors/optical flow; native Unity and Unreal support; per-game opt-in required. Coverage frames it as "mobile DLSS."[16][17]
- **Why it matters to OSS:** (a) third major vendor after NVIDIA/AMD (fourth counting Sony PSSR) to ship SR+FG as one pipeline — OSS's "one architecture instead of two" thesis is now the industry's stated direction, not a contrarian bet; (b) it validates MV-driven FG on tiny power budgets, the regime OSS's Pico student (≤0.4M, <2ms, Steam Deck/iGPU/mobile tier) targets; (c) per-game opt-in = same integration shape as OSS's shim. Watch as both a competitor (mobile SR) and a potential target backend for the student's non-CUDA kernel matrix (issue #8).

### Intel / Apple / engines
- **Intel:** Game On driver 32.0.101.8991 (NBA 2K27) fixes XeSS MFG crash bugs; XeSS 3 MFG driver rollout (covered Sep 9) unchanged. No new SR model detail.[18]
- **Apple:** A20 Pro announced (Sep 9) — first 2nm smartphone chip; eventual Metal backend target hardware, no MetalFX changes this window.[19]
- **Engines:** FSR UE 5.8 plugin (RX 7000 ML upscaling) remains the only verified engine-upscaling change; Qualcomm claims native Unity/UE support for Neural Fusion at launch. DirectSR: no movement observed this window (last known state: preview supporting FSR 3.1 upscaler-only; not re-verified this run).

## Action Items / Follow-ups

- [x] **DONE this run:** GS-STVSR PDF mirrored to `docs/research/papers/2604.18047-gs-stvsr.pdf` — the standing archive P0 from the last four briefs is closed. Commit it.
- [ ] **P0 (unchanged, worsening):** recover or formally stop `srcnn-v7.0-pico-005` — 257h frozen at 36,650 with healthy publishing infra. Fix the dashboard "training now" headline to read trainer state from `status.json`.
- [ ] **Deep read (new):** TruncGradGS (arXiv:2609.03534) — extract the truncated-gradient schedule; queue as a v7 training-dynamics ablation alongside the HyperGS rank regularizer.
- [ ] **Reference for #12 (new):** Compact Neural Appearance Models (arXiv:2609.05255) — mine their attribute-compaction for the packed canvas layout before freezing the 32B/Gaussian claim.
- [ ] **Vendor watch (new):** Qualcomm Adreno Neural Fusion — assess as mobile shipping target for the Pico student (matrix cores + HPM ≈ the <2ms budget story) and as evidence in the "one architecture" README claim; note per-game opt-in mirrors OSS's shim integration.
- [ ] **Positioning:** the RPCS3 backlash is free ammunition for the deterministic-reconstruction claim — when OSS publishes, contrast canvas reconstruction (engine-buffer-conditioned) against generative neural rendering without naming/emulating the "slop" framing.
- [ ] **Retry next run:** Wayback CDX for Amulet (2608.10423) — archive.org was temporarily offline during this run.
- [ ] **Carried:** HyperGS deep read (rank regularizer → v7 ablation); GameSR engine-free control experiment; FSR4-teacher capture plan refresh incl. RDNA 3 hosts.

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[2] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[3] https://opensupersampling.com/data.json (generated_at 2026-09-10T12:00:54Z)
[4] https://opensupersampling.com/status.json (checked_at 2026-09-10T12:00:55Z)
[5] https://arxiv.org/abs/2609.03534
[6] https://arxiv.org/abs/2609.05255
[7] https://arxiv.org/abs/2609.06874
[8] https://arxiv.org/pdf/2604.18047 (mirrored locally 2026-09-10)
[9] https://www.tomshardware.com/pc-components/gpus/all-in-one-dlss-unlocked-mod-brings-dlss-5-and-multi-frame-gen-to-rtx-20-30-and-40-series-hybrid-tool-taps-amd-fsr-3-1-to-boost-frame-rates-up-to-6x
[10] https://www.tomshardware.com/video-games/pc-gaming/nvidia-app-update-fails-to-block-unofficial-dlss-multi-frame-generation-on-rtx-40-series-modders-restore-support-across-multiple-games-within-hours
[11] https://www.tomshardware.com/pc-components/gpus/astonishing-mod-runs-nvidia-dlss-5-on-a-second-gpu-using-a-reshade-add-on-to-reduce-performance-impact-boosts-neural-rendered-fps-up-to-127-percent-game-renders-on-one-card-neural-post-processing-runs-on-the-other-much-like-dedicated-physx-gpus
[12] https://api.github.com/repos/OptiScaler/OptiScaler
[13] https://www.tomshardware.com/video-games/pc-gaming/rpcs3-emulator-devs-slam-nvidia-dlss-5-as-ai-slop-generator-says-industry-pushing-more-upscalers-and-frame-generation-to-hallucinate-games-and-hide-their-lack-of-optimisation
[14] https://www.amd.com/en/developer/resources/technical-articles/2026/upscale-everything-super-resolution-across-amd-hardware-.html
[15] https://www.amd.com/en/developer/resources/technical-articles/2026/super-res-acceleration-on-ryzen-ai-npu.html
[16] https://www.qualcomm.com/news/onq/2026/09/adreno-neural-fusion-ai-rendering
[17] https://www.androidheadlines.com/2026/09/qualcomm-unveils-adreno-gpu-dedicated-ai-cores.html
[18] https://www.neowin.net/news/intel-releases-new-arc-graphics-driver-with-nba-2k27-optimization-and-bug-fixes/
[19] https://www.tomshardware.com/pc-components/cpus/apple-a20-pro-powers-iphone-18-pro-the-companys-first-2-nanometer-smartphone-chip
