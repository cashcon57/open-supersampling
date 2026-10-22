# OSS Research Brief — 2026-10-14

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (only commit returned by local `git log --oneline -10`; repository history appears shallow or truncated in this checkout).
- **Open issues:** #14 D3 VRAM measurement; #13 Nsight L2 hit-rate profile; #12 packed low-rank canvas layout; #10 concat-fusion ablation; #9 student backbone reparameterization; #8 cross-vendor kernels; #7 FSR 4 teacher-signal distillation; #6 v6.2-pico-002 launch dependency; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- **README snapshot:** pre-alpha; README describes v7 as the current research/teacher architecture and the ≤1M-parameter student as not yet trained. Current repo README says v7 Pico-005 training, while dashboard discovery results surfaced older v6-era data; do not treat search snippets as live dashboard state.
- **Training dashboard:** direct request to `https://opensupersampling.com` returned HTTP 403 from this environment. Could not verify current run/metrics; README calls it the live dashboard. ([dashboard](https://opensupersampling.com))

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047; Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha; 2026-04-20). Continuous 2D-Gaussian motion modeling, covariance resampling, and adaptive offsets target arbitrary spatial/temporal scales, with reported nearly constant runtime for ×2–×8 and >3× speedup at ×32. **Why it matters:** direct method-level comparator and possible source for covariance-resampling and large-motion handling; compare its optical-flow-driven motion to OSS's engine motion vectors/native time axis. ([arXiv](https://arxiv.org/abs/2604.18047))

### Super-Resolution / Upscaling
- **Frozen History: When Temporal Losses Stop Accumulation in Learned Real-Time Supersampling** (arXiv:2610.15752; Hao Yu, Guoying Zhao; 2026-10-11). Reports that temporal consistency losses can suppress history accumulation under accurate registration, identifies four registration errors, and reports up to 0.34–0.62 dB gains over QRISP with a normalized sample accumulator. **Why it matters:** unusually actionable training/evaluation warning for temporal canvases; audit jitter handling, resolution-lattice motion vectors, depth registration, and skip-path alignment before using temporal losses as a proxy for stability. ([arXiv](https://arxiv.org/abs/2610.15752))

### Frame Extrapolation / Generation
- **3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** (arXiv:2605.11489; Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu; 2026-05-12). Combines gradient-aware spatial supersampling with a compact recurrent frame-interpolation module for 3DGS rendering. **Why it matters:** related unified output goal, but it retains distinct spatial and temporal branches—useful contrast for evaluating OSS's single time-conditioned canvas claim. ([arXiv](https://arxiv.org/abs/2605.11489))

### Inference Optimization
- **FlashVSR+: Efficient Video Super-Resolution via Hybrid Attention and Reward-Yield Distribution Matching** (arXiv:2610.17552; 2026-10-13). Search-indexed paper describes hybrid attention and distribution matching, reporting up to 2× DiT acceleration and nearly half peak memory versus its baseline. **Why it matters:** not a game-latency competitor, but a recent example of attention sparsification and memory reduction worth screening for transferable student-teacher training ideas. Author list was not exposed in the available result; verify from paper before adding a formal citation. ([arXiv HTML](https://arxiv.org/html/2610.17552v1))

## Archive.org Findings
- Archive.org Advanced Search was queried for Gaussian-splatting SR, frame-generation, real-time game upscaling, GS-STVSR, GSASR, GRAPE, and STSS. Returned records were not reliably parseable in this run, and the exact-paper query for GS-STVSR returned no records. **No verified archived paper/version found**; absence from these results is not proof that no capture exists. ([Internet Archive Advanced Search](https://archive.org/advancedsearch.php))

## GitHub / Open Source
- **GSASR** has a public official implementation (PyTorch plus custom CUDA scale-aware rasterizer), with arbitrary-scale Gaussian reconstruction. It is an image-SR baseline, not a low-latency temporal game upscaler; useful for rasterization and arbitrary-scale evaluation ideas. ([repository](https://github.com/ChrisDud0257/GSASR))
- No directly verified new OSS repository for GS-STVSR surfaced; prior repository survey in this repo also reported the official code link unclear. Treat code-release status as unconfirmed, not unavailable.
- **OptiScaler** remains a relevant integration/competition signal: a community shim that swaps supported DLSS/FSR/XeSS paths and has experimental frame-generation support. This validates demand for vendor-neutral compatibility, but it is a compatibility layer, not an independent reconstruction model. ([repository](https://github.com/Dagherbou/OptiScaler_DLSSNR))
- Search also surfaced XeSS frame-generation manager projects, but no evidence in results of a newly released research-grade open model equivalent to OSS. ([XeSS Frame Generation](https://github.com/XeSS-Frame-Generation))

## News
- **NVIDIA:** official developer material describes DLSS 4.5's second-generation SR transformer, Dynamic and 6× Multi Frame Generation, plus a DLSS Frame Generation Performance Mode beta. Competitive threat: rapidly expanding bundled performance and a turnkey Unreal/Unity-facing SDK raises the quality and integration bar for OSS. ([DLSS developer page](https://developer.nvidia.com/rtx/dlss), [NVIDIA update](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-crossout-remastered-modern-warfare-4-frame-gen-beta/))
- **AMD:** October reporting says a lighter ML-based FSR 4 variant is planned for more APUs, laptops, and handhelds by end-2026. Treat rollout and device coverage as announced/reporting, not shipped/measured. **Threat:** expansion beyond discrete GPUs narrows OSS's platform-access differentiator. ([report](https://www.windowscentral.com/hardware/amd/amd-fsr-4-gaming-handhelds-apu))
- **Apple / Metal:** current Apple developer documentation highlights an updated temporal MetalFX upscaler using Neural Engine/Neural Accelerators on newer M-series hardware, with motion-vector and dynamic-resolution integration. **Threat/opportunity:** native temporal SR is becoming a platform baseline; Metal's quantized tensor support and motion-vector interface are useful portability targets. ([Metal what's new](https://developer.apple.com/metal/whats-new/))
- **Intel / engines:** searches did not surface a confirmed new Intel XeSS or Unreal/Unity engine announcement from the last 7–14 days. No specific update recorded rather than extrapolated from older search hits.

## Action Items / Follow-ups
- [ ] Deep-read **Frozen History** and map its four registration-error modes onto OSS's motion vectors, canvas warp, HR/LR lattice, depth, and temporal loss tests.
- [ ] Compare **GS-STVSR** covariance resampling and adaptive offset window against v7 covariance/time-axis implementation; locate official code if released.
- [ ] Re-check dashboard from a permitted browser/network path and reconcile dashboard run with README's v7 Pico-005 status.
- [ ] Track AMD's promised FSR 4 handheld/APU deployment and benchmark device coverage when available.
- [ ] No BIBLIOGRAPHY.md update this run: it already has user/uncommitted modifications in the working tree, so left it untouched to avoid mixing research changes with that work.

## Sources
- [GS-STVSR, arXiv:2604.18047](https://arxiv.org/abs/2604.18047)
- [Frozen History, arXiv:2610.15752](https://arxiv.org/abs/2610.15752)
- [3DGS³, arXiv:2605.11489](https://arxiv.org/abs/2605.11489)
- [FlashVSR+, arXiv:2610.17552](https://arxiv.org/html/2610.17552v1)
- [GSASR repository](https://github.com/ChrisDud0257/GSASR)
- [OptiScaler repository](https://github.com/Dagherbou/OptiScaler_DLSSNR)
- [NVIDIA DLSS developer page](https://developer.nvidia.com/rtx/dlss)
- [NVIDIA October update](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-crossout-remastered-modern-warfare-4-frame-gen-beta/)
- [AMD handheld FSR 4 coverage](https://www.windowscentral.com/hardware/amd/amd-fsr-4-gaming-handhelds-apu)
- [Apple Metal what's new](https://developer.apple.com/metal/whats-new/)
- [Internet Archive Advanced Search](https://archive.org/advancedsearch.php)

---

## Second Pass — 2026-10-14 (evening run)

### Project Status
- Git: still `b44278a test: add local Chrome Playwright dashboard smoke` (single visible commit; no new commits). Open issues unchanged: #14, #13, #12, #10, #9, #8, #7, #6, #5, #4.
- Dashboard (fetched via web_extract this run, loaded): shows **v6.1 Pico 0/100,000 "waiting for metrics.json"** and the v6.2-pico-002 step-70K OSS-FX clip. Static page / JS-loaded data did not populate in extraction, so live run state is unverified. Dashboard text still says v6.2-pico-002 and H009 (canvas weight ≈ 0, ×50 magnification hack), while README snapshot references v7. **Dashboard copy is stale vs README — worth a refresh pass.**
- arXiv API (`export.arxiv.org`) rate-limited (HTTP "Rate exceeded") and Python SSL failed; paper discovery done via web search + arxiv abs pages.

### New Papers (arXiv)
**Gaussian Splatting / Temporal**
- **Mobile-4DGS: Unified Static-Dynamic Real-time Mobile Gaussian Splatting** (arXiv:2610.05289; Xiaobiao Du, Zhen Fang, Tianqing Zhu, Richard Hartley, Xin Yu; v1 2026-10-04, v2 10-10). Compact explicit 4D representation using second-order Gaussian motion for real-time mobile rendering. Why it matters: second-order (acceleration) motion per Gaussian is a candidate parameterization for OSS-FX α<1 extrapolation beyond linear `α·motion`; mobile-rendering budget data point for Pico tier.
- **TETRIS-4DGS: Efficient Post-Training Compression of 4D Gaussian Splatting** (arXiv:2610.14663; Wu, Liu, Li, Yu, Liao; 2026-10-10). Importance-based adaptive quantization + motion/importance-aware packing into codec-friendly 2D maps; >40× compression in seconds. Why it matters: supports issue #12 (packed canvas layout) — importance-based per-Gaussian bit allocation as a precedent for D10 quantized state; companion to GSCV (2610.07795).
- **DynStream: Online Streaming 4D Gaussian Reconstruction of Dynamic Worlds from Unposed Video** (arXiv:2610.09720; Xian, Zhou, Xiong, Wang, Ming-Hsuan Yang; 2026-10-07). Online/streaming feed-forward 4D Gaussian update. Why it matters: closest conceptual cousin to a persistent, incrementally-updated canvas; check how it handles Gaussian spawn/retire per frame (not game-rate; pose-free).
- **STGV: Spatio-Temporal Hash Encoding for Gaussian-based Video Representation** (arXiv:2604.10910; 2026-04-14, surfaced this run). 2DGS video representation with spatio-temporal hash encoding. Why it matters: hash-grid time conditioning is an alternative to OSS's native time axis; compare cost.

**Super-Resolution / Upscaling**
- **Streaming-Aware Diffusion for Real-Time VSR via Cross-Step Attention** (arXiv:2610.11746; Partaourides, Chatzis; 2026-10-08) and **High-Throughput Generative VSR on a Single Consumer GPU** (arXiv:2610.13179; Panaretou, Chatzis; 2026-10-09). Diffusion VSR with feature reuse across frames/steps; zero-shot, built from pretrained parts. Why it matters: "real-time" here is still orders of magnitude over a <2 ms game budget — not a competitor, but confirms generative VSR is moving to consumer GPUs; useful as perceptual-quality upper bounds / possible teacher signals only.
- (Carried from earlier today) **Frozen History** (arXiv:2610.15752) remains the highest-priority read.

**Frame Extrapolation / Generation**
- No new frame-extrapolation papers beyond Amulet (2608.10423) and 3DGS³ (2605.11489). 2610.16302 (RAR, rate-adaptive world-model rollout) and 2610.16423 (sprite keyframe interpolation) surfaced but are off-target.

**Inference Optimization**
- Nothing new on TensorRT/INT8 SR directly relevant this run (2610.09098 on-device SR on 640-KB MCU is tiny-model deployment trivia). See HelixSR below for the practical INT8 evidence.

### Archive.org
- Advanced Search (JSON API worked this run): "gaussian splatting super-resolution" → 3 hits, all GitHub-starred-list snapshots (latest 2025-10-03); "frame generation deep learning" → 251 hits, none relevant; "real-time upscaling game" → 109 hits, only GitHub repo snapshots (e.g. 2026-10-09 Bloodborne PC repo) — nothing relevant. No archived GS-STVSR/GSASR/GRAPE/STSS versions found; no evidence of takedowns.

### GitHub / Open Source
- **HelixSR 1.9.0** (lonewolf0622/HelixSR, released 2026-10-13): runs NVIDIA's DLSS **Model E** (DLSS 3.7-era) as plain D3D12 compute shaders on AMD (RDNA1+, experimental Vega), no Tensor Cores/CUDA/ROCm; INT8 path for large parts of the net; Linux/Proton + Windows; slots in as an FSR 3.1 replacement or via OptiScaler. Source not fully open (builds network locally from an official DLSS DLL). **Relevance: strongest evidence yet that a DLSS-class CNN runs cross-vendor with INT8 on generic compute (supports issue #8 and the ≤1M-param cross-vendor student plan) — but also that the community can already ship "good-enough" DLSS quality everywhere without OSS.** Also a benchmark target: compare OSS student ms/frame vs HelixSR on RDNA.
- **ZeroFG** (hy300leosquizz-ctrl/ZeroFG, 9★, 2026-10-09): open-source platform-agnostic Vulkan frame-gen aimed at mobile. Small; check if it uses motion-vector extrapolation.
- **ShyVortex/dlss-unlocked** (825★) and Universal MFG Unlocker: unlockers for DLSS MFG/features on older cards — more evidence of demand for vendor-neutral FG; not research.
- Aeon SR (ReShade mod bridging DLSS/FSR/XeSS into old games, 2026-10-02), HelixSR-Manager (GUI for HelixSR).
- Gaussian-SR GitHub (older but still cited): ContinuousSR (ICLR 2026, 2D Gaussian field + ~1 ms render, github.com/peylnog/ContinuousSR), 3DGS³ code (github.com/Eabo20001/3DGS3), WaveletSR, SplatSuRe (CVPR 2026). ContinuousSR's "construct once, render many scales" ~1 ms render is directly comparable to OSS's rasterizer budget.

### News (past ~2 weeks)
- **NVIDIA DLSS 5** (3D-Guided Neural Rendering): Oct update claims 2× faster model, ~118M params (−24%), −20% 4K VRAM, up to 1.4× FPS in NBA 2K27 (Oct 16, RTX 50 / RTX Spark / GeForce NOW); Hogwarts Legacy Nov 5; coming to RTX 40 per ComputerBase. 
- **DLSS FG "Preset C"** (NVIDIA app beta, RTX 40/50, RTX Spark): replaces FG's second AI model (UI/RT-flow model) with a lighter algorithm — up to 35% faster on RTX 5060, minor IQ cost; also in RTX Remix from Oct 23. **Signal for OSS: NVIDIA itself is shedding FG network cost — validates the "one canvas, no second network" pitch but narrows the cost gap.** Modders already unlocked Preset C/MFG on RTX 30 and non-NVIDIA (Universal MFG Unlocker, 2026-10-14).
- **DLSS 4.5 on Radeon RX 9000** via FP8 (d4r, Oct 1) and DLSS 5 mod claims on RX 7000 (+207%) — NVIDIA models running on competitor hardware via community runtimes.
- **ComputerBase blind test** (>1000 gamers, 6 games): majority preferred DLSS 4.5 over FSR 4 and native TAA; gap small. Useful public perceptual baseline for the dashboard comparison table.
- **Microsoft Auto SR** extended to Intel Core Ultra 300 (Panther Lake) NPUs — OS-level, engine-agnostic NPU upscaling for thousands of games; a vendor-neutral competitor in the "works without game integration" lane.
- **Sony QSSR** (Quick Spectral Super Resolution): PSSR-like ML upscaler coming to base PS5, offered to all PlayStation developers. Console ML SR becoming baseline.
- **Intel XeSS 3** SDK published; XeSS 3 MFG official on discrete Arc (driver 8509) — earlier in Oct/Sept; Capcom's Dragon's Dogma 2 update (Oct 9) adds FSR 3 FG and decouples DLSS FG from SR.
- Nothing new found on Unreal/Unity upscaling beyond Gears of War: E-Day shipping DLSS 4.5, FSR, XeSS, TSR (Oct 1/6).

### Action Items / Follow-ups (second pass)
- [ ] **Threat/benchmark:** profile HelixSR (INT8 DLSS Model E in D3D12 compute) on RDNA as the cross-vendor latency reference; compare against OSS student target (<2 ms Pico).
- [ ] Update dashboard copy/data (still v6.1/v6.2-pico-002 framing vs README v7 Pico-005); fix "waiting for metrics.json" state if live.
- [ ] Read Mobile-4DGS second-order motion; test quadratic motion in OSS-FX α-path vs linear.
- [ ] Read TETRIS-4DGS/GSCV together for importance-based quantization to inform #12/D10.
- [ ] Read DynStream's spawn/retire policy vs OSS spawner.
- [ ] Pitch language: NVIDIA's Preset C removing FG's second model — decide whether to cite it in README "why one architecture" section.
- [ ] Add ComputerBase blind-test result to dashboard comparison caveats (non-trained-on perceptual metric).
