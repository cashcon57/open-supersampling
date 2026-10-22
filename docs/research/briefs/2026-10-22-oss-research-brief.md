# OSS Research Brief — 2026-10-22

Delta-only vs the 2026-10-19 → 10-21 briefs (incl. 10-21 evening). Run on 2026-10-22 morning EDT.

## Project Status Summary
- **Latest commits:** HEAD unchanged — `b44278a test: add local Chrome Playwright dashboard smoke`. `main` == `origin/main`. Working tree still carries uncommitted/untracked briefs and a modified `BIBLIOGRAPHY.md` (not touched by this run beyond its own brief).
- **Open issues (10, unchanged):** #4, #5, #6, #7, #8, #9, #10, #12, #13, #14.
- **Training dashboard (`https://opensupersampling.org/data.json`, generated 2026-10-22T12:06:17Z):** `srcnn-v7.0-pico-005` **still at step 36,650 / 100,000** (loss total 0.1927; sr_charbonnier 0.0236, sr_lpips 0.0927, fg_charbonnier 0.0764; canvas_count 2304, mean opacity 0.418). `gpu_status: null`, cost frozen at $0.92 / 15.27 GPU-h. The data.json timestamp advances (publisher alive) but the step does not — **trainer has been stalled ~6 days**. Dashboard `history.status` still says "active, training now (v7)" — contradicts reality. `.com` → `.org` 301 (from last evening) still in effect; use `.org` or `curl -L`.

## New Papers (arXiv)
Tooling: arXiv API again unusable (first query returned results, the next 16 came back empty — rate-limited; 5th+ consecutive degraded run). Sweep done via `rss.arxiv.org` (cs.CV/cs.GR/eess.IV/cs.LG, 564 entries) keyword-filtered and deduped against `BIBLIOGRAPHY.md` + all 2026-10 briefs, plus targeted web search. **Nothing new and on-target.**

### Gaussian Splatting / Temporal
- Nothing new on-target. API hit list (2026-10-01 → 10-15) for "gaussian splatting temporal video": ResStreamGS (2610.21121, mobile volumetric video streaming), Affine-Aligned Atlas for Canonical Gaussian Construction in Video Representation (2610.01114, Takabe et al. — video-as-Gaussians; tangentially relevant to a persistent canvas but offline/representation-only), DriveWorld3D, Edit4D, Make It Flow, Sparse-View 4D GS — all off-target.
- Off-target RSS items seen today (not added): AOV-GS (2610.29374), ClunoGS (2610.30013), *Arbitrary-Scale Anisotropic 3D MRI SR with Gaussian Prototypes* (2610.24858v2 — medical, Gaussian-prototype ASR; idea-level echo of GSASR only).
- Standing reads unchanged: KoalaGS (2610.28615), RLSplat (2610.27712), Adaptive Densification for Sparse GS in Arbitrary-Scale SR (2606.29400, already in bibliography).

### Super-Resolution / Upscaling
- Nothing new. LiveAct (2610.29408, real-time multi-agent world models) and RiO-DETRv2 are not SR-relevant.

### Frame Extrapolation / Generation
- Nothing new. (World-model "real-time game" papers keep appearing — Matrix-Game 4.0, LiveAct — still not SR/FG competitors.)

### Inference Optimization
- Nothing new on-target (today's quantization hits are all LLM-focused: 2:4 sparsity vs quantization 2610.28947, InfoQuant, etc. — no transfer to a ≤1M conv student).

## Archive.org Findings
- **GSASR (2501.06838) follow-up from last evening: resolved.** arXiv abs page lists v1 (2025-01-12) → v5 (**2025-07-30**); no revision after that. The Wayback digest change on 2026-06-07 is therefore page chrome, not a paper update. No diff needed against `docs/research/papers/` unless our cached copy predates v5 (July 2025) — worth a one-line check of which version we cached.
- No takedowns observed (abs pages for GS-STVSR / GSASR / GRAPE still live as of last run; GSASR re-verified live today).
- Advanced-search for the three target queries remains keyword-OR noise (nothing of research value) — not re-run in depth.

## GitHub / Open Source
- Search (pushed since 10-15) `gaussian splatting super-resolution`: only `GiulioFede/ResonantBraneSplatting` (1★, "quantum physics" 2DGS — speculative, ignore) and the previously-tracked `991o2o9/An-Ablation-Study-of-Auxiliary-Inputs-for-Lightweight-Neural-Upscaling-in-Real-Time-Rendering` (1★, workspace; still no code/results — keep watching for #10).
- Frame-gen topic: `MONZikWasTaken/Framegen` (MIT, 184★, created 2026-07-01) — real-time neural frame *interpolation* in the browser, hand-written WGSL on raw WebGPU, 2.9 MB model, 3–4 ms/frame. Not new to us (already in prior briefs), but a datapoint for a tiny cross-vendor (WebGPU) hand-written-kernel runtime — relevant to #8 (no-vendor-SDK portable kernels).
- Ecosystem continues to be injection/porting of proprietary models: `Merserk/dlss5-visual-enhancer`, `Kizzuwatnaa/DLSS5-Autopilot` (936★), `perseval-BLR/NeuralScreen`, OptiScaler topic (62 repos). OptiScaler now also injects FSR 4 into **Vulkan** games and supports FSR 4.1 + XeSS 3.0 (PCGH). None are open vendor-neutral architectures.
- No new open-source DLSS/FSR analysis tools beyond the above.

## News
- **AMD (10-20, confirmed again today):** FSR 4.1 is live on RX 6000 via Adrenalin 26.10.1 (months ahead of the early-2027 target); coming to Ryzen AI Max/AI 9/Z2, Xbox Series X|S and ROG Xbox Ally; **FSR 4 on Vulkan promised in FSR SDK 2.4** (no date, "coming weeks"). **UE 5.8 FSR plugin** now out (FSR "Redstone" SDK 2.3 → FSR 4.1.1 + Ray Regeneration 1.2.0 + ML FG; plugin supports UE 5.3–5.8; GPUOpen). *Relevance:* AMD is now closing the handheld/older-GPU and Vulkan gaps OSS's vendor-neutral pitch partly rested on — the differentiator shifts to **one-network SR+FG**, openness, and Metal/Intel coverage. Vulkan SDK 2.4 is the thing to read when it lands (interface surface for an OSS backend, #8).
- **NVIDIA:** DLSS 5 (3D-guided neural rendering) gets a "2× faster" model (NVIDIA 10-13; NBA 2K27 up to 1.4× fps on 10-16), Hogwarts Legacy debut 11-05, toned down vs. March per The Verge. **Cyberpunk 2077 update 2.5 (10-22)** adds DLSS 4.5 + *Dynamic Multi Frame Generation*; CoD MW4 launches 10-22/23 with DLSS 4.5; driver **621.01** (MW4 / Phantom Blade Zero / Cyberpunk). GTC Berlin (10-21) was local-AI/RTX Spark hardware — no new upscaler tech. *Relevance:* "Dynamic MFG" = variable N generated frames per rendered frame — OSS-FX should plan for multi-α (α∈(0,1)) schedules per displayed frame and frame pacing, which a canvas-render approach handles cheaply; benchmark claim to test.
- **Switch 2 / MW4:** DLSS and Filmic SMAA T2X both selectable; Digital Foundry (via Nintendo Life) prefers DLSS on stills, runtime cost not published; campaign locked 30 fps. Pico-tier ship-class reference remains unquantified.
- **Intel:** Proton/XMX-XeSS requests (filed 10-19) all still open; XeSS 3 MFG on Arc covered by PCGuide; Intel dropped Unity XeSS plugin (prior brief).
- **Apple:** nothing new (MetalFX redesigned temporal upscaler on Neural Engine + Neural Accelerators, M5 Pro/Max only — WWDC26 session "Build real-time neural rendering pipelines with Metal" is a useful transcript for the Metal kernel port, #8).
- **Engines — Unity:** Unity's "path to a single render pipeline in Unity 7" post states **URP in 6.7 LTS supports DLSS (incl. DLSS 4.5 SR) and FSR alongside STP** (forum summary of the post; first-party, no longer just the hidden preview). Confirms the Upscaler Framework direction noted 10-21; integration target for OSS = register as an additional backend once the interface is documented.
- **Engines — Unreal:** UE 5.8 has official DLSS, XeSS and (now) FSR plugins; TSR remains the in-box fallback. No third-party upscaler registration API news.

## Action Items / Follow-ups
- [ ] **Trainer stalled ~6 days at step 36,650** (pico-005) — restart or formally park; fix the dashboard status string (`history.status` "active, training now") to reflect stalled/paused when `gpu_status` is null.
- [ ] Confirm which GSASR version `docs/research/papers/` cached (latest is v5, 2025-07-30); refresh only if older.
- [ ] Read AMD FSR SDK 2.4 (Vulkan) release notes when published — compare FSR 4 Vulkan integration interface to our #8 Vulkan kernel/shim plan.
- [ ] Unity: pull the URP 6.7 LTS upscaler framework docs (`ENABLE_UPSCALER_FRAMEWORK`) and check whether a custom backend can register; add to integration plan.
- [ ] Positioning: DLSS 4.5 Dynamic MFG (Cyberpunk 2.5) — add multi-α OSS-FX + frame-pacing section to README/OSS-FX plan; update README wording "separate SR + FG stages" for AMD's FSR 4.1 + ML FG pairing.
- [ ] Carried: #7/#8 teacher-extraction legal stance (`fsr4-shader-performance-overrides`, `d4r`); KoalaGS/RLSplat vs H004/#12; Wavelet-Space SR (2508.16024) read for #9/#10.
- [ ] Watch: Phantom Blade Zero 10-28, arXiv DB maintenance **Wed 10-28 07:00–12:00 EDT** (skip research run window), Hogwarts Legacy DLSS 5 11-05, ASUS ROG Xbox Ally FSR 4.1 preview.
- [ ] **Tooling:** arXiv API is persistently rate-limited / returns empty after the first query. Build the throttled+cached fetcher (≥3 s spacing, UA header) or standardise on RSS + OpenAlex/Semantic Scholar; Python `urllib` also fails SSL verification on this host — use `curl`.

## Sources
- OSS dashboard: `https://opensupersampling.org/data.json` (generated 2026-10-22T12:06:17Z).
- [AMD newsroom — FSR 4.1 expands](https://newsroom.amd.com/news/amd-fsr-upscaling-4-1-boosts-frames-image-quality/) · [GamingOnLinux — FSR 4.1 / Vulkan SDK 2.4](https://www.gamingonlinux.com/2026/10/amd-fsr-upscaling-4-1-expands-support-to-older-desktops-laptops-handhelds-consoles-and-vulkan/) · [GPUOpen — FSR plugin for UE 5.8](https://gpuopen.com/learn/amd-fsr-plugin-updated-for-unreal-engine-58/) · [PCGH — OptiScaler Vulkan FSR 4](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/OptiScaler-mit-Mega-Update-1524309/)
- [NVIDIA — Cyberpunk 2077 DLSS 4.5 / MW4](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-cyberpunk-2077-planet-zoo-2-modern-warfare-4/) · [NVIDIA — DLSS 5 faster model](https://www.nvidia.com/en-us/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/) · [The Verge — DLSS 5 Hogwarts Legacy](https://www.theverge.com/news/1009904/dlss-5-hogwarts-legacy-changes-improvements)
- [Unity — path to a single render pipeline in Unity 7](https://unity.com/ja/topics/the-path-to-a-single-render-pipeline-in-unity-7) · [Mixed — Intel XeSS XMX under Proton](https://mixed-news.com/en/proton-xess-xmx-intel-requests-10289-wine-369-dxvk-5992/) · [Apple — What's new in Metal (WWDC26)](https://developer.apple.com/metal/whats-new/)
- [MONZikWasTaken/Framegen](https://github.com/MONZikWasTaken/Framegen) · arXiv abs 2501.06838 (version history), 2606.29400; arXiv RSS 2026-10-22; arXiv API (partial).
