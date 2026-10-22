# OSS Research Brief — 2026-09-30

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (the requested `git log --oneline -10` returned only this commit).
- **Open issues:** 12. Top items: #14 VRAM footprint, #13 Nsight L2 hit-rate, #12 packed low-rank canvas, #10 encoder ablation, #9 student reparameterization, #8 cross-vendor ports, #7 FSR 4 teacher signal, #6 pico launch, #5 no-ML shader and #4 HAT-Tiny distillation.
- **README:** pre-alpha snapshot still frames v6/v6.2 as current and says HAT-Tiny research teacher is far outside real-time budget; shipping ≤1M student remains untrained. This conflicts with newer v7 material in the existing bibliography; README likely lags the current architecture.
- **Training dashboard (checked 2026-09-30 local / 2026-10-01 UTC):** Direct `curl` later succeeded (HTTP 200) for the home page and `/status.json`; the JSON was timestamped `2026-10-01T00:01:50Z`. Trainer is explicitly `offline`, reported `stuck at step 36650 for 270h`; watcher `healthy` (published 1s ago), CF Worker `healthy` (HTTP 200 in 75ms), R2 origin `healthy` (HTTP 200). `/data.json` briefly returned a valid document with the same UTC generation timestamp, but a subsequent fetch was empty, so active-run/step/loss could not be corroborated. Browser rendering still fails locally with agent-browser permission denied. This is an operational trainer outage with healthy publishing/origin, not an overall dashboard outage; its 270h counter may be stale/reset-prone and should be checked against run metrics.

## New Papers (arXiv)
Queries used the arXiv Atom API (fielded searches for Gaussian splatting + SR, neural frame extrapolation, temporal VSR, kernel fusion, plus an ID lookup). Summaries below are abstract-level unless noted.

### Gaussian Splatting / Temporal
- No newly surfaced paper directly applicable to real-time game RGB SR or the v7 2D temporal canvas. Recent broad results were medical/depth/3D reconstruction and excluded from bibliography as out of scope. Existing GS-STVSR, GSASR and other relevant references remain in `BIBLIOGRAPHY.md`.

### Super-Resolution / Upscaling
- **[ReCaVSR: One-Step Streaming Diffusion Video Super-Resolution with Recycled Latents and Learned Cache Routing](https://arxiv.org/abs/2609.37831)** (Xijun Wang, Xin Li, Suhang Yao, Zirui Lang, Bingchen Li, Zhibo Chen; 2026-09-29; arXiv:2609.37831) — Proposes one-step streaming diffusion VSR with recycled output latents and learned per-layer temporal KV-cache scopes. **Why it matters:** the cache-routing/static-schedule idea is a possible teacher-side or offline-training reference, but the diffusion-scale model is not a plausible <2 ms student; do not infer game-real-time feasibility from the title.
- **[LoCoVSR: Local Context Diffusion Posterior Sampling for Video Super-Resolution](https://arxiv.org/abs/2609.32742)** (Matan Ben Chorin, Michael Elad; 2026-09-26; arXiv:2609.32742) — Uses shared-noise local temporal windows for diffusion posterior sampling, avoiding explicit flow and unbounded recursive history. **Why it matters:** potential comparison for temporal-consistency/evaluation strategies, but posterior-sampling diffusion is not aligned with OSS's latency target.
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11; [arXiv:2607.10495](https://arxiv.org/abs/2607.10495)) — Compact fully convolutional VSR with structural reparameterization, positioned for TensorRT-friendly deployment. **Why it matters:** strongest architecture/training comparator for the compact student and #9 reparameterization ablation; already present in `BIBLIOGRAPHY.md` (not a new append).

### Frame Extrapolation / Generation
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** (Sebastian Künzel et al.; 2026-08-11; [arXiv:2608.10423](https://arxiv.org/abs/2608.10423)) remains the relevant recent scene-cache comparator; it was already described in the Sep 29 brief and bibliography. Its explicit layered disocclusion cache/adaptive refresh is worth comparing with OSS-FX, but it is not a newly discovered paper this run.

### Inference Optimization
- No new verified paper directly applicable to TensorRT/INT8/kernel fusion for OSS's real-time compact student surfaced in the queried results. Continue prioritizing measured profiling (#13) and packed-canvas footprint (#12) over generic optimization claims.

## Archive.org Findings
- Searches for “gaussian splatting” + super-resolution, “frame generation” + deep learning, “real-time upscaling” + game and known names GS-STVSR, GSASR, GRAPE, STSS returned noisy/irrelevant records or zero matches. **No archived copies of the named papers were confirmed.** Archive full-text search is not evidence that papers were removed or unavailable.

## GitHub / Open Source
- **Threat/watch — [OpenFrameGen](https://github.com/Nicodemo-Campos/OpenFrameGen):** GitHub Search API reports MPL-2.0, 0 stars, pushed 2026-09-30; description claims cross-platform Linux/Windows frame generation, interpolation and upscaling with Vulkan/modular backends. Adjacent positioning, but no maturity, image-quality or latency validation. Audit code/benchmarks before classifying as a direct competitor.
- **Relevant student comparator — NanoVSR** remains in the bibliography; not a newly surfaced repo this sweep.
- **Adjacent implementation to watch — [RDNU](https://github.com/Adam-AtlasSoftware/RDNU):** repository search reports an RDNA3-focused decoupled neural upscaler trained across game/rendering datasets, GPL-3.0, 15 stars, pushed 2026-09-24. Potentially relevant competitive/teacher signal, but not audited and not equivalent to OSS's temporal Gaussian canvas; verify code, weights, latency and license compatibility before reuse/comparison.
- GitHub repository searches were updated-since-2026-09-16 for Gaussian SR, open-source frame generation and neural upscaler; per-repo metadata was checked for the highlighted candidate. Search results are leads, not implementation/benchmark audits.

## News
- Google News RSS was polled for DLSS/NVIDIA, FSR 4, XeSS/Intel, Apple MetalFX, and Unreal/Unity upscaling over the last 14 days. It returned mostly general or low-quality headlines; no new claim below is treated as vendor-confirmed.
- **DLSS 5:** coverage of remastered games and naming/feature commentary appeared in RSS, but no new primary NVIDIA developer announcement was verified in this run. Existing Sep 29 brief's primary-page DLSS 5 report remains unchanged.
- **FSR 4 / XeSS / MetalFX / engines:** no material new primary-source update confirmed in the fetched RSS window. One Apple/M6/MetalFX headline is third-party reporting only; not elevated to fact.
- No verified new GPU announcement or engine-upscaling release found in this pass. This is a monitoring limitation, not evidence of no announcements.

## Action Items / Follow-ups
- [ ] Audit OpenFrameGen's actual Vulkan path, licensing boundaries, quality/latency benchmarks and engine integration; compare with OSS's proposed API surface.
- [ ] Deep-read ReCaVSR only for cache-scheduling/training ideas; keep diffusion inference outside the <2 ms student path absent strong evidence.
- [ ] Continue #12 packed-canvas and #13 Nsight/VRAM measurements; use NanoVSR as the compact convolutional/reparameterization comparison for #9.
- [ ] Recheck `/data.json` and trainer-host metrics: dashboard status says trainer offline/stuck at step 36650 for 270h despite healthy watcher, worker and R2; data snapshot was intermittent.
- [ ] Refresh README architecture/status snapshot to reconcile its v6/v6.2 wording with newer v7 bibliography/research state.

## Sources / Method Notes
- Repository: `git log --oneline -10`, GitHub open issues API, README, `BIBLIOGRAPHY.md`, prior dated brief `docs/research/2026-09-29-oss-research-brief.md`.
- arXiv Atom API: fielded Gaussian-SR, frame-extrapolation, temporal-VSR and kernel-fusion queries; ID lookup for ReCaVSR, LoCoVSR and NanoVSR. IDs/authors/dates checked against Atom metadata.
- GitHub REST Search API: updated-since-2026-09-16 query terms listed above; OpenFrameGen license/stars/pushed timestamp from repository metadata.
- Archive.org Advanced Search: broad requested phrases plus exact GS-STVSR, GSASR, GRAPE and STSS probes.
- News: Google News RSS, 14-day window; RSS headline-level only unless independently verified through primary vendor pages. Dashboard home and `/status.json` returned HTTP 200 via curl; `/data.json` was intermittently readable; rendered browser attempt hit local executable permission denial.
- **Bibliography:** no additions. New VSR diffusion papers are useful leads but currently not sufficiently aligned with OSS's architecture/latency constraints; NanoVSR is already tracked.
