# OSS Research Brief — 2026-10-08

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (only entry returned by local `git log --oneline -10`; clone currently exposes one commit).
- **Open issues:** 12. Leading architectural/perf items: #14 VRAM footprint, #13 Nsight L2 profile, #12 packed low-rank canvas, #8 cross-vendor ports; #4 student distillation and #7 FSR 4 teacher signal remain open.
- **README snapshot:** describes v7 N-D Gaussian canvas with native time axis, v7 Pico-005 teacher training and an untrained ≤1M student targeting <2 ms. Project-authored claims; distinguish from live telemetry.
- **Training dashboard:** not freshly verifiable. Browser renderer fails because agent-browser binary is permission denied; direct HTTP returns 403. Previous Oct 7 snapshot reported trainer offline/stuck at step 36,650 for 426h, while dashboard headline/README said v7 Pico-005. Watcher, Worker and R2 were then healthy. Treat as stale prior evidence, not today's observation; reconcile run identity/health.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **No newly verified papers.** Three Atom API queries (targeted Gaussian-temporal, VSR/frame interpolation, inference optimization) encountered HTTP 429/timeouts. Do not interpret as no new literature. Existing directly relevant references already in the bibliography include GS-STVSR (arXiv:2604.18047) and the continuous pixel-to-Gaussian SR baseline (arXiv:2503.06617).

### Super-Resolution / Upscaling
- No new arXiv metadata verified this pass. The Atom API was rate-limited/unresponsive; no bibliography additions made.

### Frame Extrapolation / Generation
- No new arXiv metadata verified this pass; same API limitation.

### Inference Optimization
- No new arXiv metadata verified this pass; same API limitation.

## Archive.org Findings
- Broad Advanced Search: “gaussian splatting super-resolution” returned 3 hits, mostly archived GitHub-list snapshots; “frame generation deep learning” returned 3 noisy records including unrelated media/software; “real-time upscaling game” returned 0. No confirmed archived version of GS-STVSR, GSASR, GRAPE or STSS found. These broad queries are inconclusive, not evidence of absence.

## GitHub / Open Source
- **Opportunity / adjacent threat: [ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG)** — newly active, open-source platform-agnostic Vulkan frame-generation engine aimed at mobile; repo API reports 5 stars, Apache-2.0, pushed Oct 8. Described as experimental V1 POC, not a demonstrated competitor in quality/latency. Track its motion-vector/input contract, Vulkan portability, and benchmarks against OSS’s shared SR/FG canvas premise.
- **[DLSS Unlocked](https://github.com/ShyVortex/dlss-unlocked)** — 773 stars, MIT, pushed Oct 8; claims to unlock DLSS/MFG/neural-rendering features on supported RTX/DX12 games. Compatibility tooling, not an open model or direct architectural competitor; monitor for implications to deployment expectations and proprietary feature access.
- **[FoundDSR](https://github.com/yanzq95/FoundDSR)** — 1 star, pushed Sep 25; guided 2D Gaussian depth super-resolution, not game-color SR or frame generation. Adjacent technique only; no license listed by GitHub API.
- Search queries used: quoted “gaussian splatting” + “super-resolution”, quoted “frame generation” + “open source”, and “neural upscaler”; results are noisy. No independently verified new direct open-source game-upscaler model surfaced. Direct primary vendor landing pages (GPUOpen, NVIDIA GeForce News, Intel newsroom, Apple Developer News) were reachable, but no release claims were established from page content in this sweep.

## News
- Google News RSS, 14-day query (headline-level leads, not independently verified against primary announcements): Oct 8 reporting says AMD expects FSR 4 support on APUs, gaming laptops and handhelds by end-2026. If confirmed, this broadens FSR reach beyond discrete GPUs and raises the cross-device compatibility bar; verify on AMD/GPUOpen before treating as a committed milestone.
- RSS also surfaced DLSS 4.5 running on AMD Radeon via community mods (Sep 29–Oct 1) and Sony QSSR/PS5 AI upscaling reports (Oct 1–7). These are secondary reporting leads, not evidence of vendor-supported portability or a new DLSS/FSR/XeSS release.
- No primary-source-verified new Intel XeSS, Apple MetalFX, Unreal/Unity upscaling, or GPU announcement established. RSS is incomplete/noisy; absence here is not proof no update occurred.

## Action Items / Follow-ups
- [ ] **HIGH:** Restore dashboard observability (browser binary permission / 403 path); verify live trainer process, run identity, step/loss and health counter against the README.
- [ ] Track ZeroFG’s implementation and benchmark maturity; compare Vulkan integration, motion inputs, latency/quality, and whether it remains FG-only versus OSS’s shared canvas.
- [ ] Confirm AMD’s reported FSR 4 APU/handheld timeline using a primary AMD/GPUOpen announcement; assess implications for OSS’s deployment targets.
- [ ] Retry targeted arXiv Atom discovery after rate-limit cooldown; record paper IDs/authors/date from primary metadata before adding citations.
- [ ] Continue #13 Nsight L2 profiling and #12 packed-layout/VRAM work; require measured end-to-end latency and footprint before adopting techniques.
