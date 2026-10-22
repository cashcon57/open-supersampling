# OSS Research Brief — 2026-09-25

## Project Status Summary
- Latest local commit: `b44278a test: add local Chrome Playwright dashboard smoke` (only commit returned by local `git log --oneline -10`).
- Open issues (first 10 returned by GitHub API): #14 D3 VRAM footprint; #13 Nsight L2 hit-rate; #12 packed low-rank canvas; #10 decoupled G-buffer ablation; #9 RepVGG/ECBSR/RepSR student ablation; #8 cross-vendor ports; #7 FSR 4 teacher distillation; #6 launch v6.2-pico-002; #5 no-ML shader; #4 HAT-Tiny student distillation.
- README status snapshot says v6.2 is baseline for v7 and v7 Pico-005 is training. **README/dashboard disagreement:** dashboard webpage was reachable, but its client-side data endpoints returned HTTP 403 via direct fetch. Could not verify current step, loss, health, or whether metrics are advancing. Previous brief's reported stale step was not revalidated.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** — Mingyu Shi et al.; 2026-04-20; [arXiv:2604.18047](https://arxiv.org/abs/2604.18047). Continuous Gaussian motion, optical-flow guidance, and covariance resampling alignment; highly relevant reference for OSS's canvas temporal stability and anti-aliasing. Already in bibliography.
- **3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** — Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu; 2026-05-12; [arXiv:2605.11489](https://arxiv.org/abs/2605.11489). Unifies post-render super-sampling and frame interpolation, deploying with TensorRT; competitive adjacent approach, but interpolation is not future-frame extrapolation and it is 3DGS-rendering specific.

### Super-Resolution / Upscaling
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** — Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11; [arXiv:2607.10495](https://arxiv.org/abs/2607.10495). Reparameterized fully convolutional 644k-param model reports 27.2 FPS on Jetson Orin NX; direct architecture/distillation candidate against OSS's ≤1M student target. Strong opportunity: study training recipe and deployment constraints; compare quality/latency on matched game data.

### Frame Extrapolation / Generation
- No novel verified paper found in this pass. 3DGS³ is frame interpolation; do not label it extrapolation.

### Inference Optimization
- NanoVSR's structural reparameterization into standard convolutions is a directly actionable accelerator-friendly approach; assess for student backbone/ablation (issue #9).

## Archive.org Findings
- IA Advanced Search API query for `"gaussian splatting" super-resolution` returned three GitHub snapshot identifiers (no paper snapshots). `"frame generation" deep learning` returned three items including DLSS Swapper builds and FluidFrames; `"real-time upscaling" game` returned zero. No evidence from these queries of archived/taken-down GS-STVSR, GSASR, GRAPE, or STSS versions.

## GitHub / Open Source
- Search surfaced no independently verified new release/update of an OSS competitor project this run. A prior project survey already records GSASR and related Gaussian-SR candidates; this run's searches did not establish new commit activity.
- **Competitive watch:** 3DGS³ is a research implementation direction combining SS and interpolation with TensorRT; closer conceptual overlap than conventional frame-by-frame SR, but still 3DGS-specific and not evidence of an engine-integrated game upscaler.
- Focused repository/activity audit remains incomplete; broad search results were mostly existing survey material and topic pages.

## News
- NVIDIA developer search listing: Sep 22 blog entry announced DLSS 5 / 3D-Guided Neural Rendering and related developer capabilities. NVIDIA's developer DLSS page states DLSS 4.5 and MFG support; distinguish neural rendering from traditional SR/FG when comparing OSS.
- NVIDIA DLSS page also shows updated Unreal plugin availability (UE 5.7/5.8) and September 2026 update. Potential integration and competition signal; DLSS remains vendor/hardware tied.
- No primary AMD or Intel news source corroborating a meaningful development in the past 7–14 days was found. Secondary reports were excluded from firm claims. No verified new Apple Silicon/Metal performance announcement found.

## Action Items / Follow-ups
- [ ] Deep-read NanoVSR; reproduce parameter count, latency, and quality trade-offs on game-rendered sequences; test reparameterization in issue #9 ablation.
- [ ] Deep-read 3DGS³ and compare its TensorRT/post-render pipeline and temporal alignment against OSS's single-canvas thesis.
- [ ] Reconcile README's v6/v7 snapshot with dashboard state; retrieve dashboard client API data through browser/authorized endpoint and verify freshness.
- [ ] Track DLSS 5 developer integration and clarify its operational distinction from SR/FG competitors.
- [ ] Continue direct GitHub repository activity audit for GS-STVSR/GSASR/GRAPE/STSS and open-source DLSS/FSR analysis tools.
