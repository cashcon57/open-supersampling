# OSS Research Brief — 2026-08-28

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke`.
- **Open issues:** **12**. Immediate gates remain packed low-rank canvas layout / VRAM / L2 profiling (#12–#14), then tile binning (#3), student distillation (#4/#7/#9), and cross-vendor kernels (#8).[1]
- **README snapshot:** pre-alpha v7 remains the N-D `(x,y,t)` Gaussian-canvas **teacher**; the ≤1M shipping student and native CUDA/HIP/Metal/Level Zero/Vulkan kernels are still unbuilt.[2]
- **Training dashboard:** `srcnn-v7.0-pico-005` remains at **36,650 / 100,000** with total loss **0.19268**, SR Charbonnier **0.02360**, FG Charbonnier **0.07643**, and **2,304** canvas elements.[3] At 00:01Z on Aug 29 UTC the trainer was **offline**, “stuck at step 36650 for 121h,” while Watcher, CF Worker, and R2 were healthy.[4] The dashboard’s “training now” copy is therefore stale/conflicting.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **M³ISR: A Multi-Modal Multi-View Benchmark for 3D/4D Gaussian Splatting and Feedforward Compression** (arXiv:2608.22465; Xinhui Liu, Lei Liu, Zhenghao Chen, Lebin Zhou, Wei Wang, Wei Jiang; 2026-08-23) — A controlled synthetic 3D/4DGS benchmark with 25 scenes, synchronized 1080p views, RGB/depth/semantic/static-dynamic annotations, and streaming/compression tracks. **Why it matters to OSS:** newly surfaced and material for #12: use its rate-distortion and dynamic-update tracks as an external control for canvas residency, storage, and parent/child density, while retaining TartanAir/engine-motion-vector evaluation for actual game relevance.[5]
- The fresh broad arXiv sweep also surfaced dynamic-4DGS/SLAM/world-model papers, but none cleared the architecture/training/inference materiality gate for OSS today. No new bibliography entry was created for those indirect leads.

### Super-Resolution / Upscaling
- No new directly relevant real-time game/VSR student result was verified. Existing NanoVSR / reparameterizable-CNN work remains the best student-backbone lead; do not change the ≤1M deployment plan without fused 1080p-LR measurements.

### Frame Extrapolation / Generation
- No new material frame-extrapolation paper cleared the gate. Keep Amulet as the explicit visibility-age, disocclusion, and tile-refresh systems baseline for OSS-FX.

### Inference Optimization
- No new TensorRT/INT8/kernel-fusion paper materially changes execution order: measure packed layout and L2 behavior (#12/#13) before selecting attributes, precision, sort keys, or fusion boundaries.

## Archive.org Findings
- Advanced Search remains noisy: only three GitHub snapshot artifacts for `"gaussian splatting" super-resolution`; `"frame generation" deep learning` and `"real-time upscaling" game` were dominated by unrelated media/items.[6]
- Targeted Wayback checks found an archived copy for the already-tracked joint SR/interpolation paper **3DGS³**; no archived records were returned for GS-STVSR or Amulet. This is limited coverage, **not** evidence that a paper was withdrawn/taken down.

## GitHub / Open Source
- **IE-SRGS** (1 star, no asserted SPDX license; pushed Aug 27) is an official 3DGS-SR implementation, but its README requires large SwinIR and Depth Anything V2 models plus conventional offline datasets. Watch only; it is neither a latency control nor a deployable OSS student.[7]
- **OSSS** (MIT, 0 stars; pushed Aug 25) is a window-level classical optical-flow frame-interpolation/upscaling pipeline. Its explicit non-extrapolation design and known disocclusion/fast-motion limits make it a useful integration/pacing watch, not a quality or architectural threat to engine-MV OSS-FX.[8]
- No credible new open-source DLSS/FSR analysis tool surfaced in GitHub REST searches.

## News
- **NVIDIA DLSS 4.5 Ray Reconstruction is out via NVIDIA App.** NVIDIA’s Gamescom announcement confirms the release; this remains the only verified vendor-side competitive change in the 14-day window.[9]
- **OSS implication:** product integration needs model-version/capability manifests and per-title temporal controls. NVIDIA’s model-selection UX raises the baseline for debuggability and rollout, independently of image-quality comparison.
- A Google News lead reported AMD’s FSR plugin update for Unreal Engine 5.8, but the guessed direct GPUOpen page returned 404; treat it as an unverified lead, not a product fact. No primary Intel XeSS, Apple MetalFX, Unity, or Unreal upscaling announcement was verified in this run.

## Action Items / Follow-ups
- [ ] **P0 ops:** resolve or explicitly close the trainer; it is unchanged at 36,650 with a 121h offline/stuck state while publication services are healthy.
- [ ] **#12/#13:** add M³ISR’s dynamic-streaming/compression rate-distortion protocol as a secondary canvas-residency control; retain game-motion and temporal-identity gates as the primary acceptance criteria.
- [ ] **Student gate:** keep reparameterizable CNN exploration behind a fused ≤1M, 1080p-LR latency benchmark.
- [ ] **OSS-FX:** maintain Amulet-style visibility age, refresh cadence, and frame-pacing telemetry alongside PSNR/LPIPS.
- [ ] **Product positioning:** prototype per-title model/capability manifests before claiming an integration UX competitive with DLSS 4.5.

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[2] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[3] https://opensupersampling.com/data.json
[4] https://opensupersampling.com/status.json
[5] https://arxiv.org/abs/2608.22465
[6] https://archive.org/advancedsearch.php
[7] https://github.com/ZheXues/IE-SRGS
[8] https://github.com/Lenovive/OSSS
[9] https://www.nvidia.com/en-us/geforce/news/gamescom-2026-nvidia-geforce-rtx-dlss-4-5-announcements
