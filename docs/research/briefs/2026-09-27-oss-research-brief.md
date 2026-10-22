# OSS Research Brief — 2026-09-27

## Project Status Summary
- Latest commits: `b44278a test: add local Chrome Playwright dashboard smoke` (only commit returned by local `git log --oneline -10`; local clone history is shallow/limited).
- Open issues: 12. Highest relevance: #14 v6.2 VRAM footprint, #13 Nsight L2 hit-rate, #12 packed low-rank canvas layout, #8 cross-vendor kernel ports, #4 HAT-Tiny student distillation.
- README snapshot: README still presents v6.1/v6.2 as active; teacher is research-only and the ≤1M student remains untrained; best reported held-out quality is v5 on TartanAir (PSNR 25.703, LPIPS 0.1666). This differs from the prompt's v7/N-D architecture summary—confirm which architecture is authoritative before using this memo to steer implementation.
- Training dashboard: not verifiable this run. Browser rendering failed (`agent-browser-linux-x64` permission denied); direct homepage and `/data.json` fetch both returned HTTP 403. Dashboard status left unknown, not assumed unchanged.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- No new, untracked directly applicable paper verified. **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** — Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha (2026-04-20; [arXiv:2604.18047](https://arxiv.org/abs/2604.18047); already bibliographed). Abstract reports covariance-resampling alignment and near-constant inference across temporal scales; directly relevant to temporal covariance stability and canvas resampling.

### Super-Resolution / Upscaling
- No new directly applicable SR paper identified in the limited search. **3DGS³** below is relevant to joint rendering quality tasks, but is separately placed in frame generation.

### Frame Extrapolation / Generation
- **3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** — Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu (2026-05-12; [arXiv:2605.11489](https://arxiv.org/abs/2605.11489)). Combines gradient-aware super-sampling with a compact recurrent interpolation module. **Why it matters:** joint task baseline, but it uses distinct modules and therefore does not validate OSS's unified-representation/near-zero-marginal-cost claim.
- **SPEED: One-Step Pixel Diffusion for High-quality Video Frame Interpolation** — Zihao Zhang, Haoyu Zhao, Siqian Yang, Yidi Wu, Yudong Jiang, Zuxuan Wu (2026-07-17; [arXiv:2607.15585](https://arxiv.org/abs/2607.15585)). One-step pixel diffusion with staged dynamic patches and Noise-Update-Only Attention; abstract reports 63.3% faster inference and 10.6% lower memory than compared method, plus LPIPS gains. **Why it matters:** a quality-oriented VFI ceiling/comparator, but diffusion's runtime and memory claims need validation against OSS's sub-2ms game budget; likely teacher/benchmark signal rather than student design.

### Inference Optimization
- No new directly relevant TensorRT/quantization/kernel-fusion paper verified in this pass. Prior tracked **ANVIL** remains a motion-vector/operator-elimination lead; not newly discovered here.

## Archive.org Findings
- Advanced Search (three requested keyword families) returned 3, 243, and 107 results respectively, but top results were GitHub archive bundles and unrelated/noisy records. No relevant archived paper version established for GS-STVSR, GSASR, GRAPE, or STSS; this is inconclusive, not evidence of removal. Targeted Wayback/CDX checks were not completed.

## GitHub / Open Source
- Search results surfaced [RDNU](https://github.com/Adam-AtlasSoftware/RDNU) (RDNA3-focused game neural upscaler, 16 stars, GPL-3.0, last pushed Sep 24) as an adjacent open implementation worth tracking. License is incompatible with direct Apache-2.0 code reuse absent separate permission; useful for comparative methods/benchmarking only.
- [dlss-unlocked](https://github.com/ShyVortex/dlss-unlocked) advertises unlocking DLSS/MFG and Neural Rendering on RTX GPUs; 682 stars, MIT, pushed Sep 26. This is an integration/modding signal, not a competing independent upscaler. Claims should be treated as repo-description claims pending technical audit.
- Search surfaced NPU placement benchmark [npu-upscaling-placement](https://github.com/hamzakazih1/npu-upscaling-placement), created Sep 26 (MIT); potential cross-device runtime/latency evidence, not a new game-upscaler implementation.
- No new verified Gaussian-SR repo identified. GitHub search was noisy; result metadata inspected selectively, not exhaustive.

## News
- Google News RSS (past 14 days) surfaced reports that modders halved FSR 4 render time on AMD's BC-250 APU (~11.51→5.92 ms at 1440p); secondary-source lead, not independently verified against AMD primary sources. If accurate, demonstrates deployment/runtime optimization can materially change competitive baselines.
- RSS reported DLSS 5 experimentation and unlock tooling; primary announcement/feature updates were not verified in this run. Treat as headline-level leads only.
- RSS searches for Intel Arc/XeSS, Apple Metal, and engine upscaling returned mostly unrelated or low-quality items; no primary-source announcement verified in the requested window.

## Action Items / Follow-ups
- [ ] Resolve dashboard 403/access and inspect live trainer step/loss/health; this run has no trustworthy dashboard status.
- [ ] Confirm architecture version mismatch: task context says v7 N-D canvas, local README says v6.1/v6.2 active.
- [ ] Deep-read SPEED only if extrapolation quality/teacher strategy is in scope; measure actual latency before considering deployment relevance.
- [ ] Compare RDNU's game inputs, runtime, and benchmark protocol; do not import GPL-3.0 code into OSS.
- [ ] Re-run targeted Wayback/CDX lookups for GS-STVSR, GSASR, GRAPE, and STSS.
- [ ] Verify BC-250 FSR timing claim from primary/technical source before drawing competitive conclusions.

Sources: local README/git/issues; arXiv primary abstract pages; GitHub repository search metadata; Archive.org Advanced Search; Google News RSS (headline-level). Dashboard requests failed (HTTP 403 / browser binary permission denied). Paper summaries are abstract-level.
