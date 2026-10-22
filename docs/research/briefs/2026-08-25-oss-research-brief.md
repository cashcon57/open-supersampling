# OSS Research Brief — 2026-08-25

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke`; no new model, training, or kernel commit is present in the local history.
- **Open issues:** **12** visible open issues. The first ten cover canvas/raster measurement and layout (#12–#14), student/distillation (#4/#7/#9), cross-vendor ports (#8), shader fallback (#5), launch (#6), and the concat-fusion ablation (#10).[9]
- **README snapshot:** v7 `srcnn-v7.0-pico-005` remains the active HAT-Tiny teacher design; the ≤1M shipping student and CUDA/HIP/Metal/Level Zero/Vulkan deployment kernels remain target work.[10]
- **Training dashboard:** at `2026-08-26T00:00:51Z`, v7 remains at step **36,650** with 2,304 canvas elements, total loss **0.19268**, SR Charbonnier **0.02360**, and FG Charbonnier **0.07643**.[12] The health endpoint calls the trainer **offline**, “stuck at step 36650 for 49h,” while watcher, Worker, and R2 are healthy.[11] The historical panel still says “training now”; that is stale/conflicting dashboard copy, not evidence of progress.

## New Papers (arXiv)

- The follow-up arXiv Atom queries for all five requested topic groups were rate-limited/timed out (HTTP 429/30s); the verified primary metadata below is preserved from the successful earlier scan rather than replaced by search snippets. No additional paper was added to `BIBLIOGRAPHY.md`: both material titles already have single entries.[19]

### Gaussian Splatting / Temporal
- **Towards Alias-Free 4D Gaussian Representations with Motion-Aware Filtering** (Ankit Dhiman, Kunal A. Kathare, Pranav Vignesh, Lokesh R. Boregowda, Venkatesh Babu Radhakrishnan; 2026-08-22; arXiv:2608.21828) — proposes a local-motion-conditioned smoothing filter for 4D Gaussian representations rather than a static 3D/Mip-Splat filter.[1] **Why it matters to OSS:** this is a direct v7 anti-aliasing ablation: condition the existing Mip-Splat-style covariance/filter choice on engine motion magnitude and time-depth behavior, then test shimmer versus blur under fast camera/object motion. It is a teacher/rasterizer research lead, not shipping-student evidence.

### Super-Resolution / Upscaling
- No newly published, architecture-changing real-time game-SR/student paper passed the relevance gate. NanoVSR remains the existing ≤1M structural-reparameterization comparator; do not substitute generic diffusion or sparse-view SR results for exact-engine-MV game deployment evidence.[10]

### Frame Extrapolation / Generation
- **Following Motion for Sequential Modeling in Video Frame Interpolation** (Jaehyun Park, Nam Ik Cho; 2026-08-24; arXiv:2608.22861) — Motion-Guided Mamba serializes features along optical-flow trajectories and adds contextual synthesis when flow is unreliable.[3] **Why it matters to OSS:** a useful control for the residual/refinement path after exact engine-MV warp: test motion-aligned state ordering or, more cheaply, use its “unreliable-flow → contextual synthesis” idea as a confidence-gated local fallback. It does not supersede the canvas/time-query architecture.
- **Amulet** remains the strongest already-tracked OSS-FX systems control: sparse layered caching, predictive shading, and tile lifetimes explicitly address disocclusions and refresh scheduling.[2] It remains more relevant to product behavior than a conventional VFI quality claim.

### Inference Optimization
- No new TensorRT/INT8/kernel-fusion paper materially changes the current plan. Profile #12/#13 first; only then select packed attributes, sort keys, fusion boundaries, and post-fusion FP8/INT8 for the student.

## Archive.org Findings
- Archive Advanced Search remained noisy: `gaussian splatting super-resolution` returned three repository/listing artifacts; `frame generation deep learning` (226 hits) and `real-time upscaling game` (107 hits) were dominated by unrelated media/software items.[14][15][16]
- Targeted strings GS-STVSR, GSASR, GRAPE, and STSS did not reveal a verified archival paper copy; GRAPE/STSS are especially acronym-polluted.[6] This is archive-coverage status only—not evidence a paper was withdrawn.
- Targeted Wayback checks found one historical arXiv capture each for **QuADA-GS** and **RBS** (2026-07-02), but none for GS-STVSR or Motion-Aware 4DGS at query time.[18] This confirms limited archival coverage only; arXiv remains the metadata source of record.

## GitHub / Open Source
- **IE-SRGS** (`ZheXues/IE-SRGS`) is a new AAAI-2026 3DGS-SR code lead, created/pushed 2026-08-24, with 0 stars and no asserted license.[4] **Watchlist only:** inspect code, paper provenance, and licensing before treating its internal/external fusion as a v7 competitor or importing ideas.
- **OSSS** (`Lenovive/OSSS`) is a new MIT Windows window-level FG/upscaler project (0 stars), while **ZeroFG** remains an Apache-2.0 experimental Vulkan/mobile FG POC (1 star).[5][13] Both are integration/plumbing reconnaissance, not demonstrated quality or latency threats.
- No verified new open-source DLSS/FSR analysis tool surfaced in the structured GitHub search.

## News
- The generic search provider was credit-exhausted; structured Google News RSS was used as the fallback. It yields only headline-level leads, not primary-source confirmation.[7][17]
- Current leads are DLSS 4.5/Ray Reconstruction reports in game/beta coverage and Alan Wake 2 FSR 4/FSR FG coverage. No direct NVIDIA, AMD, Intel, Apple, Unreal, or Unity announcement was verified in this run. Treat this as competitive monitoring only; frame-pacing/present-interval telemetry still belongs in OSS-FX evaluation regardless of these reports.[17]

## Action Items / Follow-ups
- [ ] **P0 ops:** investigate/restart or explicitly close the trainer state at step 36,650; healthy publication infrastructure is not training progress.[11][12]
- [ ] **Deep read / ablation:** implement motion-conditioned covariance filtering as a v7 anti-aliasing control; score temporal shimmer, blur, and disocclusions against the current static filter.[1]
- [ ] **Technique:** add an engine-MV confidence/disocclusion gate with cheap contextual refinement; use Motion-Guided Mamba as the high-cost reference, not a default deployment architecture.[3]
- [ ] **Competitor watch:** triage IE-SRGS code/license/measurements; maintain Amulet as the explicit-visibility OSS-FX control.[2][4]
- [ ] **Product gate:** add present-interval/frame-pacing telemetry beside FG image metrics before any FG quality claim.[17]

## Sources

[1] https://arxiv.org/abs/2608.21828
[2] https://arxiv.org/abs/2608.10423
[3] https://arxiv.org/abs/2608.22861
[4] https://api.github.com/repos/ZheXues/IE-SRGS
[5] https://api.github.com/repos/Lenovive/OSSS
[6] https://archive.org/advancedsearch.php
[7] https://news.google.com/rss/search
[8] https://opensupersampling.com
[9] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[10] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[11] https://opensupersampling.com/status.json
[12] https://opensupersampling.com/data.json
[13] https://api.github.com/repos/hy300leosquizz-ctrl/ZeroFG
[14] https://archive.org/advancedsearch.php?q=gaussian%20splatting%20super-resolution%26fl%5B%5D=identifier%26fl%5B%5D=title%26fl%5B%5D=date%26rows=10%26output=json
[15] https://archive.org/advancedsearch.php?q=frame%20generation%20deep%20learning%26fl%5B%5D=identifier%26fl%5B%5D=title%26fl%5B%5D=date%26rows=10%26output=json
[16] https://archive.org/advancedsearch.php?q=real-time%20upscaling%20game%26fl%5B%5D=identifier%26fl%5B%5D=title%26fl%5B%5D=date%26rows=10%26output=json
[17] https://news.google.com/rss/search?q=DLSS%204%20OR%20FSR%204%20OR%20XeSS%20when%3A14d%26hl=en-US%26gl=US%26ceid=US%3Aen
[18] https://web.archive.org/cdx/search/cdx
[19] https://export.arxiv.org/api/query
