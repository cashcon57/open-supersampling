# OSS Research Brief — 2026-09-29

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke` (only commit returned by the requested `git log --oneline -10`).
- **Open issues:** 12 returned by GitHub API; top items include #14 VRAM measurement, #13 Nsight L2 hit-rate, #12 packed low-rank canvas, #10 encoder ablation, #9 student reparameterization, #8 cross-vendor ports, #7 FSR 4 teacher distillation, #6 pico launch, #5 shader variant, #4 HAT-Tiny distillation.
- **README snapshot:** marked pre-alpha; describes v6/v6.2 workflows and HAT-Tiny teacher, measured inference well above budget, ≤1M shipping student not trained. README architecture/status and bibliography currently have newer v7/v7.0-pico-005 material; treat README overview as lagging where it conflicts with these newer docs.
- **Training dashboard:** direct fetch of https://opensupersampling.com returned HTTP 403; dashboard state remains unverified. Prior dated brief reported the same access limitation; no inference about training activity.

## New Papers (arXiv)
- No new papers verified in this run. arXiv Atom API requests returned HTTP 429 across four broad searches; do not treat this as evidence that no papers exist. Previously tracked GS-STVSR, GSASR and Amulet remain relevant (see prior daily briefs/BIBLIOGRAPHY.md).
### Gaussian Splatting / Temporal
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha; 2026-04-20; [arXiv:2604.18047](https://arxiv.org/abs/2604.18047)) — Continuous-time Gaussian motion, optical-flow guidance and covariance resampling; reports near-constant inference across x2–x8 and >3× speedup at x32. Direct math/ablation precedent, but already tracked in `BIBLIOGRAPHY.md`; no duplicate.
- **Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution (GSASR)** (Du Chen, Liyi Chen, Zhengqiang Zhang, Lei Zhang; 2025-01-12, revised 2025-07-30; [arXiv:2501.06838](https://arxiv.org/abs/2501.06838)) — Feed-forward image-conditioned Gaussians plus scale-aware CUDA rasterization support unseen scales. Existing bibliography and repo references; no duplicate.

### Super-Resolution / Upscaling
- No newly verified, materially relevant paper surfaced beyond already-tracked GSASR. Search was web-index based, not a complete arXiv Atom harvest.

### Frame Extrapolation / Generation
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** (Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11; [arXiv:2608.10423](https://arxiv.org/abs/2608.10423)) — Non-neural sparse layered image-space cache predictively stores visible geometry/shading and schedules tile refresh; reports up to 250 Hz at 4K. Tracked already in bibliography; useful OSS-FX comparator for disocclusion and refresh policy, not a new addition.

### Inference Optimization
- No newly verified SR-specific TensorRT/INT8/kernel-fusion paper with direct implementation impact found in this sweep. Prioritize measured Nsight L2 and packed-layout work over generic quantization claims.

## Archive.org Findings
- Broad exact-title probes for “GS-STVSR,” “GSASR” and “GRAPE” returned zero Archive.org records; an “STSS”/Gaussian probe found an unrelated 2003 arXiv physics record. No archived paper versions confirmed; index results are not evidence papers were removed.
- `GS-STVSR` and `STSS Gaussian splatting super-resolution` exact broad searches returned 0; `GSASR` returned an unrelated CIA item; `GRAPE` returned 17,020 unrelated/media results. These are noisy keyword-search outcomes, not evidence of missing/removed papers. No archived paper version confirmed.
- Search result lead for arbitrary-scale Gaussian SR points to already known GSASR (arXiv:2501.06838); Archive.org search itself did not establish an archived copy.

## GitHub / Open Source
- **Threat/watch — [OpenFrameGen](https://github.com/Nicodemo-Campos/OpenFrameGen):** repo search describes a cross-platform open-source frame-generation/interpolation/upscaling framework for Windows/Linux with Vulkan modular backends; last pushed Sep 27, 2026, 0 stars, MPL-2.0. Fresh and adjacent competitor; inspect code and benchmarks before assessing maturity.
- **Adjacent — [DLSSG-Open](https://github.com/dniminenn/dlssg-open):** NVIDIA DLSS Frame Generation on RTX 30 under Proton; updated Sep 23, 2026, MIT, 1 star. Compatibility/interop rather than vendor-neutral alternative.
- **Adjacent — [FoundDSR](https://github.com/yanzq95/FoundDSR):** Gaussian-based depth SR project, updated Sep 25, 2026; not game RGB SR.
- GitHub search results are leads only; no new repository code/benchmark audit.
- **Threat / watch — [OpenFrameGen](https://github.com/Nicodemo-Campos/OpenFrameGen):** search result describes a cross-platform open-source interpolation/upscaling framework for Linux/Windows with Vulkan/modular backends; repo updated Sep 27. Potential adjacent OSS positioning; inspect implementation, license, quality and latency before treating as a direct competitor.
- **[afmf-linux](https://github.com/serialexperimentslainnn/afmf-linux):** search result describes open-source Linux Vulkan frame generation; updated Sep 22. Worth tracking as an integration/competition signal, not a shared Gaussian architecture.
- **[FoundDSR](https://github.com/yanzq95/FoundDSR):** Gaussian-splatting depth-SR repo updated Sep 25, but depth task rather than game RGB SR; low direct relevance.
- **[GSASR](https://github.com/ChrisDud0257/GSASR):** existing official code corresponding to the known paper, not a new discovery.
- GitHub REST search was exploratory; search snippets only, not per-repository license/code validation. No credible new DLSS/FSR analysis tool verified.

## News
- Google News RSS for the past 14 days surfaced NVIDIA’s DLSS 5/3D-Guided Neural Rendering coverage, including the NVIDIA developer blog headline. This is headline-level monitoring; no additional primary-page checks completed in this run. RSS also returned unofficial modder claims and rumor-like reports; excluded from verified claims. No primary vendor announcements independently verified here for FSR 4, XeSS, Apple Metal, or engine updates.
- **NVIDIA (primary source):** [DLSS 5 launch](https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/) now available in NBA 2K27 on RTX 50 and GeForce NOW, as 3D-guided neural rendering for lighting/material enhancement. Competitive threat is adjacent neural-rendering positioning, not direct SR architecture; monitor how motion/color inputs and latency interact with OSS.
- **Apple (primary source):** [Metal What's New](https://developer.apple.com/metal/whats-new/) describes redesigned MetalFX temporal upscaler using Neural Engine/Neural Accelerators on M5 Pro/Max, with motion-vector and subrectangle support. Opportunity/pressure for OSS Metal backend; assess portability and model/runtime constraints.
- **AMD / Intel:** search surfaced FSR/XeSS version and support claims, but no current primary vendor announcement was verified in this run. Treat those as unconfirmed leads.
- **Window:** current-date search did not verify further primary-source changes in the past 7–14 days for UE5/Unity or GPU announcements.

## Action Items / Follow-ups
- [ ] Retry arXiv Atom queries after HTTP 429 cooldown; use narrow ID batch and add only verified material papers.
- [ ] Audit OpenFrameGen implementation, license details, and latency/quality benchmarks.
- [ ] Recheck dashboard from an accessible rendered session; current direct request was blocked (403).
- [ ] Inspect OpenFrameGen code/license/benchmark and compare integration model against OSS target.
- [ ] Deep-read Amulet’s layer-cache disocclusion and adaptive refresh evaluation against OSS-FX fractional-time rendering.
- [ ] Continue #12/#13 packed canvas + measured L2/VRAM profiling; avoid assuming sub-2ms until shipping student is measured.
- [ ] Retry dashboard from an accessible network/rendered session; this run's 403 is an access limitation, not a status report.
- [ ] Track MetalFX temporal changes for potential Metal kernel/API design implications.

## Sources / Method Notes
- arXiv Atom API: four broad fielded searches; all returned HTTP 429.
- GitHub Search REST: gaussian splatting super-resolution; frame generation open source; neural upscaler. Metadata retrieved for reported repos.
- Google News RSS query: DLSS OR FSR OR XeSS, when:14d; headline-level only.
- Archive.org Advanced Search: exact probes for GS-STVSR, GSASR, GRAPE, STSS.
- GitHub issues API: https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
- Repository README and `BIBLIOGRAPHY.md` read locally; latest prior dated brief: `docs/research/2026-08-29-oss-research-brief.md`.
- Dashboard URLs attempted: https://opensupersampling.com, https://opensupersampling.com/data.json, https://opensupersampling.com/status.json (HTTP 403).
- Archive.org Advanced Search: https://archive.org/advancedsearch.php (exact keyword probes noted above).
- GitHub REST repository Search API; queries: gaussian-splatting super-resolution, frame-generation open-source, neural-upscaler. Search result snippets, not repo audits.
- Vendor pages: NVIDIA DLSS 5 page and Apple Metal What's New page linked above. Other vendor claims are not elevated beyond leads.