# OSS Research Brief — 2026-07-16

## Project Status Summary
- Latest commits:
  - `a97a180 feat: add tailnet-gated training controls to dashboard`
  - `fcf8084 feat: tailnet-gated training control API + dashboard panel`
  - `0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc`
  - `56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)`
  - `d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume`
  - `e1ab7fe hat.py: use flash-attn 2 directly when available, fallback to SDPA`
  - `e9b4b65 trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)`
  - `26cd9da dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB`
  - `83b4f07 v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast`
  - `baf4b02 worker: apex .com → .org 301 redirect`
- Open issues: #14 D3: measure v6.2-pico-002 VRAM footprint after v6.2 wiring and checkpoint exist; #13 H004: run Nsight L2 hit-rate profile on 3080 Ti CUDA host; #12 H004: implement packed low-rank canvas layout before claiming 32B/Gaussian; #10 v6.2.1 ablation: decoupled G-buffer encoder before concat-fusion; #9 v6.2.1 ablation: RepVGG/ECBSR/RepSR reparameterization for student backbone; #8 v6.2: cross-vendor kernel ports (HIP / Metal / Vulkan / SYCL); #7 v6.2.1: distillation pipeline with FSR 4 binary as teacher signal; #6 v6.2-pico-002: launch on 3080 Ti once codex T1-T5 land; #5 v6.2: No-ML shader variant (ships alongside v6.2); #4 v6.2: HAT-Tiny student distillation (≤1M params).
- README snapshot: pre-alpha; current architecture is v7 `srcnn-v7.0-pico-005`, an N-D Gaussian canvas with native time axis, HAT-Tiny teacher, parent-child loss-adaptive density, Mip-Splatting filters, and alpha=0.5 OSS-FX supervision. Shipping ≤1M-param student is still not trained.
- Training dashboard: rendered page shows `srcnn-v7.0-pico-005` at step 35,200 / 100,000, loss 0.10166, viz @ step 35,077, headline says `updated just now` / `live`. Service health says **Trainer — offline, stuck at step 35,200 for 19m**; Watcher healthy published 1s ago; CF Worker healthy HTTP 200 in 82ms; R2 origin healthy HTTP 200; GPU stats unavailable / host offline or rsync failed. Compared with the earlier 2026-07-16 draft and 2026-07-15 brief, the stuck counter reset again (10h → 19m, after yesterday's 46h) while step/loss/viz stayed effectively unchanged, so treat this as a dashboard health-counter/staleness inconsistency, not recovery.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **Implicit 4D Gaussian Splatting for Fast Motion with Large Inter-Frame Displacements** (arXiv:2607.12362v1, 2026-07-14; Seung-gyeom Kim, Areum Kim, Yongjae Yoo, Sukmin Yun) — Introduces SPIN-4DGS, predicting Gaussian attributes from explicit spatiotemporal positions to preserve fast-moving objects under large inter-frame displacement. Why it matters to OSS: already in bibliography, but still the top fast-motion read for v7 native-time attributes, `V_xt` covariance, and OSS-FX large-displacement failures.
- **ExtraGS: Enhancing Endoscopic View Extrapolation via Diffusion-Guided 3D Gaussian Splatting** (arXiv:2607.12785v2, 2026-07-14; Cheng-Tai Hsieh, Jiwei Shan, Han Fang, Jianshu Hu et al.) — Uses diffusion-guided 3DGS to hallucinate/extrapolate anatomy outside a narrow endoscopic field of view. Why it matters to OSS: newly surfaced off-domain view-extrapolation paper; diffusion-guided 3DGS may be useful only as a disocclusion/unknown-region prior, not as a shipping real-time path.
- **GeoFovea-GS: Geometry-Aware Cross-Layer Gaussian Splatting for Wireless Aerial VR** (arXiv:2607.12641v1, 2026-07-14; Zeyi Ren, Wencheng Yan, Jiawen Zhang, Jintao Yan et al.) — Targets bandwidth/latency-constrained aerial VR using geometry-aware foveated Gaussian rendering. Why it matters to OSS: newly surfaced foveated/wireless-VR Gaussian rendering lead; watch for latency/bandwidth-aware canvas allocation ideas, though it is not game SR.
- **NoDrift3R: Raymap-Guided Coupling for Drift-Robust Unposed Feed-Forward 3D Reconstruction** (arXiv:2607.07168v1, 2026-07-08; Xiangyu Sun, Liu Liu, Seungkwon Yang, Jingbing Han et al.) — Uses raymap-guided coupling to reduce drift in long unposed feed-forward 3DGS reconstructions. Why it matters to OSS: long-sequence drift robustness is less central with exact engine motion vectors, but the coupling idea may inform canvas consistency diagnostics.

### Super-Resolution / Upscaling
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (arXiv:2607.10495v1, 2026-07-11; Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies) — Fully convolutional VSR with structural reparameterization into standard inference convolutions, avoiding explicit flow/custom ops while retaining temporal alignment through progressive training. Why it matters to OSS: already added to bibliography; highest-value student-model lead for structural reparameterization and TensorRT-friendly ≤1M-param VSR.
- **Wavelet-Space Representations for Neural Super-Resolution in Rendering Pipelines** (arXiv:2508.16024v3, 2025-08-22; Prateek Poudel, Prashant Aryal, Kirtan Kunwar, Navin Nepal et al.) — Predicts stationary wavelet coefficients rather than RGB directly, using rendering-pipeline auxiliary buffers/history. Why it matters to OSS: already tracked; wavelet/coefficient-space prediction remains a strong high-frequency-detail/shimmer-control ablation for teacher or student heads.
- **Neural Super-Resolution for Real-time Rendering with Radiance Demodulation** (arXiv:2308.06699v2, 2023-08-13; Jia Li, Ziling Chen, Xiaolong Wu, Lu Wang et al.) — Demodulates radiance/material components to preserve textures and reduce ghosting in real-time rendering SR. Why it matters to OSS: already tracked/known rendering-SR baseline; radiance demodulation is relevant to G-buffer/material separation and texture preservation.

### Frame Extrapolation / Generation
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586v1, 2026-05-11; Nengbo Lu, Minghua Pan) — Models causal physical dynamics of Gaussian-represented scenes for future trajectory prediction. Why it matters to OSS: off-domain physical-causality Gaussian dynamics; useful only if OSS-FX needs learned residual object motion beyond engine MVs.
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567v1, 2026-05-11; Nengbo Lu, Bin Zhao) — Learns Gaussian velocity fields from dynamic multi-view video with physical-consistency constraints. Why it matters to OSS: off-domain velocity-field 3DGS; conceptually adjacent to v7 native time/covariance but likely too reconstruction-focused for the immediate shipping path.
- **ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering** (arXiv:2506.13814v1, 2025-06-14; Lufei Liu, Tor M. Aamodt) — Exploits temporal coherence in rendering networks through layer/feature caching. Why it matters to OSS: already tracked; layer/feature caching remains an inference optimization candidate for temporally coherent render workloads.
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (arXiv:2603.26835v3, 2026-03-27; Shibo Liu) — Uses codec motion-vector priors to make mobile-NPU video interpolation more accelerator-friendly. Why it matters to OSS: already tracked; codec-motion-vector VFI is a useful analogue for engine-MV-driven OSS-FX deployment constraints.

### Inference Optimization
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (arXiv:2607.10495v1) — Structural reparameterization into ordinary convolutions is directly actionable for issue #9 and the student TensorRT path.
- Older quantized/mobile SR results re-surfaced (`NAWQ-SR`, integer-only CNNs, Mobile AI challenge reports). No new INT8/TensorRT paper beat NanoVSR as the actionable lead today.

### Game Engine Integration
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694v1, 2026-03-08; Yuhang Wang, Hai Li, Shujuan Hou, Zhetao Dong et al.) — Uses codec-side motion vectors, residual maps, and frame types for more efficient online VSR. Why it matters to OSS: already tracked; compressed-domain motion vectors/residual maps are the closest non-game analogue to OSS exact-engine-buffer reuse.
- `Efficient neural supersampling on a novel gaming dataset` re-surfaced as the cleanest public game-neural-supersampling baseline, but it is already in the bibliography.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 item hits, still broad GitHub/star-list archives rather than primary paper artifacts.
- `"frame generation" "deep learning"`: 3 hits: DLSS Swapper builds, FluidFrames builds, and an OSF MoCha-LD record; useful ecosystem preservation leads, not OSS architecture inputs.
- `"real-time upscaling" game`: 0 useful item hits returned.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16639 hits but acronym pollution dominates (`GRAPE`, `STSS`); use targeted URL CDX checks instead of broad inventory.
- Targeted Wayback CDX: GSASR GitHub has 2025 HTTP-200 captures; 3DGS3 arXiv has a 2026-06-07 HTTP-200 capture; GS-STVSR arXiv CDX timed out; Rolaand-Jayz FSR 4.1.0 repo CDX returned 503; ResonantBraneSplatting GitHub returned no rows in this check.

## GitHub / Open Source
- `gaussian splatting super-resolution`: no recently pushed relevant repositories returned by GitHub REST search.
- `frame generation open source`: 1 result, `OULDRABAH95/AllPen-framework-`, unrelated pentest/agent framework; no OSS action.
- `neural upscaler` (`pushed:>=2026-07-02`): 5 results, mostly toy/site repos. Notable already-tracked lead: `Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling` (4★, MIT, pushed 2026-07-02) — still worth inspecting before citing as competitor evidence.
- `DLSS FSR analysis`: no recently pushed relevant repositories returned by GitHub REST search.
- `FSR 4 upscaler`: same `Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling` result; no new independent FSR reverse-engineering repo surfaced today.
- `XeSS upscaler open source`: no recently pushed relevant repositories returned by GitHub REST search.
- Vendor/direct probes: NVIDIA GeForce News HTTP 200 hits=DLSS; NVIDIA Developer Blog DLSS search HTTP 200 hits=DLSS,upscal,super resolution,frame generation,Unreal Engine 5; AMD GPUOpen news HTTP 200 hits=FSR,FidelityFX,upscal,super resolution,frame generation,Unreal Engine 5; AMD FidelityFX SDK releases HTTP 200 hits=FSR,FidelityFX; Intel XeSS page HTTP None hits=none; Unreal Engine blog HTTP 200 hits=Unreal Engine 5; Unity blog HTTP 200 hits=none. Intel XeSS page fetch failed from this runtime; Google News/RSS used as fallback for XeSS headlines.

## News
- **NVIDIA / DLSS 4**: Google News RSS shows Linux DLSS Updater/Proton upscaler tooling, Gears of War: E-Day DLSS 4.5 / dynamic multi-frame-generation headlines, and NVIDIA GeForce/news pages still keyword-hit DLSS. Non-NVIDIA headlines remain leads, not primary-source confirmation.
- **AMD / FSR 4**: RSS repeatedly surfaces AMD FSR multi-frame-generation / 8× mode references and FSR 4.1 coverage. Threat: AMD is moving the public narrative toward multi-frame generation; OSS should keep its “single architecture SR+FG” positioning sharp.
- **Intel Arc / XeSS**: RSS was mostly noisy benchmarks and hardware/process articles; no strong primary XeSS SDK/runtime update verified. Direct Intel page fetch failed.
- **Apple Silicon / Metal**: RSS surfaced M5/Metal gaming and frame-generation-on-Mac headlines; no primary Metal upscaler/runtime API change verified. Keep Metal backend watch, but no architecture change today.
- **Unreal / Unity**: Unreal feed fetched but only generic UE5 keyword hits; Unity blog fetched with no relevant upscaling hits. No engine-integration action this run.

## Action Items / Follow-ups
- [ ] Ops: dashboard/trainer health still conflicts — step 35,200 unchanged, trainer offline, stuck counter reset again (46h yesterday → 10h earlier today → 19m now); verify trainer process and metrics counter semantics on the 3080 Ti host.
- [ ] Deep-read **ExtraGS** only for disocclusion/view-extrapolation priors; explicitly reject diffusion-heavy components for the real-time path unless they can be distilled.
- [ ] Deep-read **GeoFovea-GS** if/when OSS adds foveated rendering or bandwidth-aware streaming; otherwise keep as watchlist.
- [ ] Continue **NanoVSR** → RepVGG/ECBSR/RepSR student-backbone ablation mapping (#9).
- [ ] Inspect `Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling`; if real, extract concrete FSR 4.1 operator/weight-layout facts for competitive positioning.
- [ ] Watch AMD FSR multi-frame-generation / 8× rumors for primary confirmation; if confirmed, update OSS positioning around one-architecture SR+FG versus vendor multi-network stacks.
