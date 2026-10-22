# OSS Research Brief — 2026-08-11

## Project Status Summary
- **Latest commits:** `a97a180` adds tailnet-gated dashboard training controls; `fcf8084` adds their API/panel; the preceding work preserves the v7-pico-005 step-5K recovery snapshot and fixes Windows/Linux resume, FlashAttention, TF32/prefetch, PSNR display, and BF16 raster filtering.
- **Open issues:** 11, led by P0 low-rank rasterizer (#1), custom counting-sort tile bin (#3), ≤1M HAT-Tiny student distillation (#4), no-ML shader (#5), and packed canvas/L2/VRAM investigation (#12–14).
- **README snapshot:** v7 N-D `(x,y,t)` Gaussian teacher is the active research architecture; shipping remains an untrained ≤1M CNN student with a <2 ms target. Native HIP/Metal/Level Zero/Vulkan and DLL-shim runtime remain designed-only.
- **Training dashboard:** rendered state says `srcnn-v7.0-pico-005`, step **36,650 / 100,000**, loss **0.19268**, viz step 36K. It is internally contradictory: headline says “live”/ETA ~1.1d, but Trainer is **offline, stuck at step 36,650 for 39h** and GPU stats are unavailable; Watcher (published 1s ago), Cloudflare Worker (200/94ms), and R2 origin are healthy. Treat as a trainer-host outage, not training progress.

## New Papers (arXiv)

No newly submitted (since the Aug. 9 brief) paper materially changes OSS architecture, training, or deployment. The targeted query did revalidate the following already-bibliographed direct references; no duplicate bibliography entries were added.

### Gaussian Splatting / Temporal
- **ACA-GS: Adaptive-Capacity Anchored Gaussian Splatting for Compact Dynamic Radiance Fields** (arXiv:2608.04581, Aug. 5, 2026; Seunghyeon Song, Joo Chan Lee, Chanung Park, Jun Young Jeong, Minseo Lee, Eunbyung Park, Jong Hwan Ko) — dynamically allocates Gaussian count and feature channels by local spatiotemporal complexity. **Why it matters:** strongest current ablation reference for v7 parent-child density, packed-canvas budgets, and motion/detail-aware capacity allocation; already in `BIBLIOGRAPHY.md`.
- **CLEAR: Conflict-aware Learning via Evidence-guided Adaptive Routing for Unified Sparse-View 3D Gaussian Super-Resolution** (arXiv:2608.02206, Aug. 3, 2026; Hantang Li, Qiang Zhu, Xiandong Meng, Debin Zhao, Xiaopeng Fan) — routes high-frequency gradients/densification using evidence confidence while reconciling LR observations with HR priors. **Why it matters:** a direct training ablation for preventing perceptual-teacher hallucinations from contaminating an engine-evidence canvas; already tracked.

### Super-Resolution / Upscaling
- **Resonant Brane Splatting for Arbitrary-Scale Super-Resolution** (arXiv:2606.29453v2, Jun. 28, 2026; Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto) — replaces flat 2D Gaussians with Gaussian-Hermite “Branes” that encode local contrast/high frequencies inside one footprint, with culling-aware rasterization. **Why it matters:** candidate primitive/rasterizer ablation for edge recovery with fewer overlaps; likely expensive versus the Pico budget. Already tracked.
- **Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution (QuADA-GS)** (arXiv:2606.29400, Jun. 28, 2026; Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto) — predicts feed-forward complexity-based Gaussian densification under a global budget, using O(1) pointer-convolution neighbourhood lookup. **Why it matters:** close architectural comparator for v7 loss-adaptive parent-child density and issue #12’s packed layout. Already tracked.
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (arXiv:2607.10495, Jul. 11, 2026; Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies) — a 644K fully convolutional VSR model structurally reparameterized into standard deployable convolutions. **Why it matters:** direct baseline for issue #9 and the ≤1M student; its “no explicit flow” design is a useful lower-complexity comparator to exact engine MV. Already tracked.

### Frame Extrapolation / Generation
- No new material paper. Broad `frame extrapolation` results were largely physics/video-generation work, not a real-time game extrapolator; the existing OSS-FX baselines (MASS, SPEED, ExtraGS) remain the relevant comparison set.

### Inference Optimization
- No new material paper. Existing student/quantization leads remain NanoVSR and deployment-aware INT8 SR; the urgent work is empirical: profile #13 and implement/test the issue #12 packed layout before choosing an FP8/INT8 kernel strategy.

## Archive.org Findings
- Archive.org advanced search found no preserved GS-STVSR, GRAPE, or STSS record and no relevant item for “real-time upscaling game.”
- Keyword hits for Gaussian SR and frame generation were an unrelated GitHub-stars archive, an OSF latent-diffusion registration, and consumer software; the `GSASR` hit was an unrelated CIA document. None are citable OSS sources.
- No archived/taken-down paper version worth adding was found. Primary arXiv records for the two direct ASR papers remain available.

## GitHub / Open Source
- GitHub REST search (pushed since Jul. 28) found no credible new open-source DLSS/FSR/XeSS analysis or OSS-like frame-generation implementation.
- `julesklord/bananascaler` is a MIT WIP “GPU-accelerated neural video upscaler,” pushed Aug. 4, with zero stars; it is not a competitive signal.
- `hvcl/3DCT_SR_DDNM_NAB-GS` is Apache-2.0 but medical/CT-specific; no useful game-runtime transfer established.
- **Positioning:** no open project surfaced that joins engine motion vectors, persistent Gaussian SR, and zero-extra-network extrapolation. OSS’s immediate risk is execution/validation, not a new open competitor.

## News
- **Verified primary Intel signal (Aug. 10):** Intel announced Gamer Days partnerships for *STAR WARS: Galactic Racer* and *Tomb Raider: Legacy of Atlantis*, with **XeSS 3 support at launch**. This reinforces the need for an Unreal/plugin integration path and a vendor-neutral game-input contract. Source: [Intel Newsroom](https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships).
- **Headline-level leads only:** Google News RSS reports an XeSS 3.1 Unreal Engine 5.8 plugin, FSR 4.1 Steam Machine/Proton support, and new game integrations for DLSS 4.5/FSR 4. These were not corroborated by a direct AMD/NVIDIA/Intel SDK release page in this run, so do not treat them as confirmed API/quality claims.
- No primary Apple/MetalFX upscaling change or GPU announcement with a direct OSS architectural consequence surfaced in the last 14 days.

## Action Items / Follow-ups
- [ ] **P0 operational:** inspect/restart the v7 trainer host; step/loss have been frozen for 39h despite the dashboard’s “live” label.
- [ ] Implement a bounded-capacity ablation: parent-child density vs ACA-GS-style per-anchor count/channel allocation, measured against #12 packed-canvas memory and #13 L2 profile.
- [ ] Deep-read QuADA-GS’s global-budget routing and pointer-convolution layout before designing the packed low-rank canvas.
- [ ] Use NanoVSR’s reparameterized-conv baseline in issue #9; compare exact-MV student, implicit-temporal student, TensorRT FP8/INT8 latency, and artifact metrics.
- [ ] Watch official XeSS 3 / Unreal support and FSR 4.1 Steam/Proton integration; an open vendor-neutral DLL/plugin remains an opportunity, but only after OSS demonstrates stable v7 SR/FX use of its canvas.
