# OSS Research Brief — 2026-10-10

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (`git log --oneline -10` yielded one commit).
- **Open issues:** GitHub API returned at least 10; newest ten are #14 VRAM footprint, #13 Nsight L2 profile, #12 packed low-rank canvas, #10 G-buffer encoder ablation, #9 student reparameterization, #8 cross-vendor ports, #7 FSR 4 teacher distillation, #6 pico launch, #5 shader variant, #4 HAT-Tiny distillation.
- **README/status:** prior local brief reports v7 N-D Gaussian/time-axis as research direction while shipping student remains untrained; older README run-state text conflicts with newer dashboard. Current clone has extensive untracked daily research reports and modified `BIBLIOGRAPHY.md`; preserved existing same-day brief rather than replacing it.
- **Dashboard (`https://opensupersampling.org/`, rendered Oct 10):** Trainer tooltip says **offline, stuck at step 36,650 for 510h**; Watcher published 0s ago, CF Worker HTTP 200/109ms, R2 HTTP 200. Dashboard headline simultaneously shows v6.1 Pico 0/100,000, waiting for metrics; previous Oct 10 brief recorded v7.0-pico-005 step 36,650/loss 0.19268 and “live”/ETA. This is a serious stale/conflicting run-state display, not evidence of progress. Published page says teachers are live research and shipping students “have not been trained yet”; tier table describes HAT-Tiny teacher and ≤0.4M/≤1M CNN students.

## New Papers (arXiv)
Search discovery used web search (arXiv Atom request failed due local TLS certificate verification); filtered against bibliography and recent briefs. No new directly relevant primary paper surfaced that warrants bibliography addition this run. Search surfaced potentially adjacent papers, but broad results included diffusion/3DGS papers outside OSS's real-time budget. Existing high-value leads remain GS-STVSR (2604.18047), Amulet (2608.10423), 3DGS³ (2605.11489), NanoVSR (2607.10495), and SPEED (2607.15585), already recorded in project bibliography/reports.

### Gaussian Splatting / Temporal
- **No newly verified material paper.** GS-STVSR remains the closest continuous 2D-Gaussian spatiotemporal-SR comparator; see [arXiv:2604.18047](https://arxiv.org/abs/2604.18047), already tracked.

### Super-Resolution / Upscaling
- No newly surfaced real-time game-SR or ≤1M student paper verified. Generic latent/diffusion SR search results were far outside OSS's inference budget.

### Frame Extrapolation / Generation
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** (2026-08-11, [arXiv:2608.10423](https://arxiv.org/abs/2608.10423), already tracked). Non-neural sparse tiled/layered cache explicitly handles disocclusion; reported 250 Hz at 4K. Relevant architecture comparator, not a new discovery this run.

### Inference Optimization
- No newly verified game-SR inference result. Existing [NanoVSR (arXiv:2607.10495)](https://arxiv.org/abs/2607.10495) remains a relevant small reparameterized VSR student reference (644k parameters per the existing bibliography).

## Archive.org Findings
- Archive.org `advancedsearch.php` queries returned noisy inventories: 3 results for Gaussian-splatting/SR (GitHub list snapshots), 249 for frame-generation/deep-learning, and 109 for real-time-upscaling/game; leading hits were irrelevant repository snapshots or media. No confirmed archived/taken-down versions of GS-STVSR, GSASR, GRAPE, or STSS identified. Broad keyword results are not evidence of absence; targeted Wayback CDX checks remain outstanding.

## GitHub / Open Source
- **Opportunity / early competitor signal — [ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG):** repository metadata confirms a newly pushed (Oct 9) Apache-2.0, platform-agnostic Vulkan frame-generation engine focused on mobile, described as experimental V1 POC; 7 stars. Inspect implementation and quality/latency before judging competitiveness. This overlaps OSS's cross-vendor/mobile positioning, but is not yet a verified comparable full SR+extrapolation system.
- GitHub search found no fresh Gaussian-SR repository in the top results. Other “neural upscaler” results were mostly wrapper/mod tooling (including DLSS unlock/add-on projects), not independent vendor-neutral reconstruction models.

## News
- **AMD FSR 4 expansion (RSS/headline-level):** Google News RSS dated Oct 7–10 reports AMD plans to bring FSR 4 to APUs, laptops and handhelds before 2027. Could expand competitive reach beyond discrete GPUs; verify with AMD primary announcement before treating as confirmed.
- **DLSS compatibility mods (RSS/headline-level):** Oct 10 reporting describes mods bringing DLSS 3.7 to Radeon/Steam Deck. Experimental wrappers, not official NVIDIA support and not a direct threat to OSS model quality; notable user demand for runtime cross-vendor access.
- No primary vendor page was verified in this pass for new Intel XeSS, Apple Metal, or game-engine upscaling updates. News sources were RSS headlines, not primary-source confirmation.

## Action Items / Follow-ups
- [ ] Reconcile dashboard's contradictory current headline/run identity with its 510h trainer-offline indicator; verify host process and metrics source.
- [ ] Review ZeroFG code/license maturity and benchmark relevance for OSS's Vulkan/mobile lane.
- [ ] Confirm FSR 4 device-expansion claims against AMD primary communications; track actual SDK/hardware availability.
- [ ] Run targeted Wayback CDX checks for GS-STVSR, GSASR, GRAPE and STSS paper/project URLs.
- [ ] Retry primary arXiv API via a working certificate/network path; this run's Atom request failed TLS verification, so paper-search completeness is limited.
