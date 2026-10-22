# OSS Research Brief — 2026-05-31

## Project Status Summary

**Latest commits (top 10):**
1. `0660f55` wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc
2. `56a9a65` trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)
3. `d1255c8` trainer: WindowsPath shim on Linux for cross-OS checkpoint resume
4. `e1ab7fe` hat.py: use flash-attn 2 directly when available, fallback to SDPA
5. `e9b4b65` trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)
6. `26cd9da` dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB
7. `83b4f07` v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast
8. `baf4b02` worker: apex .com → .org 301 redirect
9. `eb0ec2e` trainer: bf16 autocast around forward + loss
10. `5c2f2e7` trainer: add throughput baselines (cudnn.benchmark, TF32, torch.compile)

**Training dashboard:** `srcnn-v7.0-pico-005` at step 9,100 / 100,000 (9.1%). Trainer degraded — stuck at step 9100 for 7min at time of check. Loss: 0.01242 sr_charbonnier. ETA ~1.6 days. Watcher/CF Worker/R2 all healthy.

**Open issues:** Same 10 open issues. No new issues or closures since last scan.

**Key observation:** Trainer moved from step 8,900 (yesterday) to 9,100 — 200 steps in ~24h. This is extremely slow (~1 step every 7 min vs normal ~4.5 s/step). The WSL2 deadlock revert at `56a9a65` may not have fully resolved the root cause, or there's a new bottleneck. The "degraded" status with 7min stuck suggests the trainer may be hanging intermittently.

---

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **[R5DGS: Semantic-Aware 4D Gaussian Splatting with Rigid Body Constraints]** (arXiv:2605.25909, May 25 2026) — Augments physics-driven 4DGS with Identity Encoding vectors for Gaussian-to-object association. Introduces rigid-body inference constraint: predict dynamics only for object centroids, propagate to member Gaussians via relative transformations — yielding 11 FPS speedup during extrapolation. **Why it matters to OSS:** The rigid-body centroid propagation trick is a pure optimization — reduce degrees of freedom during inference by grouping Gaussians with shared motion. This is structurally analogous to what OSS-FX could do with scene-semantic cueing: group canvas Gaussians by motion coherence and extrapolate groups rather than individual primitives. The 11 FPS speedup magnitude suggests this class of optimization is worth investigating for the OSS-FX path once it's wired. Code: github.com/be2rlab/r5dgs.

- **[MotionScale: Scalable 4D Gaussian Splatting for Dynamic Scenes]** (arXiv:2603.29296, CVPR 2026) — Cluster-centric basis transformations for scalable motion field that adaptively captures diverse motion patterns. Two-stage progressive optimization with background extension + foreground propagation. **Why it matters to OSS:** The cluster-centric motion basis is conceptually similar to OSS v7's parent-child loss-adaptive density — both group Gaussians by shared motion characteristics. MotionScale's progressive optimization strategy (adapt to new regions → refine motion consistency) could inform how OSS v7 handles disocclusion spawns during training. The explicit transient shadow modeling is less relevant to game SR but the motion field architecture is.

- **[4C4D: 4 Camera 4D Gaussian Splatting]** (arXiv:2604.04063, CVPR 2026) — Reconstructs 4D dynamic scenes from as few as 4 cameras using Neural Decaying Function on Gaussian opacities to bias gradients toward geometry learning over appearance. **Why it matters to OSS:** Low relevance for game SR (we have exact camera + motion vectors, not sparse capture). But the insight that geometry learning is harder than appearance in sparse settings might inform OSS's cross-scene generalization training — if geometric priors need more gradient signal during distillation.

- **[3DGS for Efficient Retrospective Dynamic Scene NVS + Standardized Benchmark]** (arXiv:2605.12437, CVPR 2026) — Argues explicit temporal coupling is unnecessary in synchronized multi-view settings; propagates optimized Gaussians over time without temporal deformation constraints. Introduces Blender-based dynamic MV dataset framework. **Why it matters to OSS:** The counter-claim that temporal coupling may be *unnecessary* in synchronized settings is directly relevant to v7 — if true, it suggests OSS v7's V_xt cross-correlation in the Cholesky (native time axis) may be over-modeling the temporal dimension compared to a simpler frame-to-frame propagation approach. The Blender benchmark framework could be repurposed for OSS evaluation scenes. Worth a skeptical deep-read.

- **[Arbitrary-Scale 3D Gaussian Super-Resolution]** (arXiv:2508.16467, AAAI 2026) — First integrated 3DGS SR framework for arbitrary-scale factors (integer + non-integer). Scale-aware rendering + generative prior-guided optimization + progressive super-resolving. +6.59 dB PSNR over vanilla 3DGS, 85 FPS at 1080p. **Why it matters to OSS:** The arbitrary-scale capability is architecturally interesting — OSS currently targets fixed 2×/3×/4× scales. If the v7 canvas could support arbitrary non-integer scales (e.g., 1440p → 4K = 1.5×), it would be a differentiator. The 85 FPS at 1080p on 3DGS is promising but uses 3D Gaussians (not OSS's 2D canvas) and is likely on datacenter GPUs — need to translate to 3080 Ti numbers.

- **[Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos]** (arXiv:2511.11175, AAAI 2026) — Coarse-to-fine temporal alignment module for 4DGS from unsynchronized multi-view video. **Why it matters to OSS:** Low relevance — OSS has exact engine timestamps, not unsynchronized capture. The coarse-to-fine alignment technique could be repurposed for frame pacing correction if OSS ever needs to handle variable-refresh-rate inputs.

### Super-Resolution / Upscaling

- **[NTIRE 2026 Efficient Super-Resolution Challenge]** (arXiv:2604.03198, CVPRW 2026) — Annual benchmark. Samsung SRC-B won x4 SR track with cascaded framework. Multiple teams demonstrated INT8 quantization pipelines achieving >4× inference speedup with <0.5 dB PSNR drop. **Why it matters to OSS:** The INT8 quantization results from competing teams provide a concrete baseline for OSS's student model INT8/FP8 path. The cascaded architecture may offer insights into multi-stage refinement for the student model. The challenge's standardized efficiency metrics (runtime/params/FLOPs) are the right benchmarks for OSS's ≤1M-param student.

- **[Efficient INT8 Single-Image Super-Resolution via Deployment-Oriented Quantization]** (arXiv:2604.20291, Apr 2026) — Extract-refine-upsample design with INT8 quantization for deployment. Student performs most computation in INT8 with minimal FP32 overhead. **Why it matters to OSS:** Directly relevant to the OSS student model quantization path. The extract-refine-upsample pipeline design mirrors the backbone → rasterizer → composite structure OSS uses. The INT8 deployment methodology could be adapted for OSS's TensorRT FP8 export.

### Frame Extrapolation / Generation

- **[X4Rate: A Scene-Adaptive Extrapolation Framework for 4× Frame Rate]** (ICPRAM 2026) — Scene-adaptive framework for 4× frame rate extrapolation. Outperforms ExtraNet by 22.13%. No single extrapolation algorithm performs best across all scenes, so X4Rate selects adaptively. **Why it matters to OSS:** The adaptive selection approach acknowledges a fundamental challenge OSS also faces: no single model works well on all content. The 4× frame rate target (e.g., 30→120 FPS) matches what OSS-FX aims to achieve. Their adaptive selector could inform an OSS-FX quality/performance tradeoff knob that switches between α values or extrapolation modes per scene region. **Note:** This appears to be a rendering optimization paper (sparse shading), not neural frame generation — relevant but different domain.

- **[RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting]** (arXiv:2603.13783, CVPR 2026) — Already covered yesterday. Updated: CVPR 2026 paper now available with supplemental materials. Temporal opacity regularization + spline-based spatial positioning enables rendering at arbitrary timestamps without ghosting. **Why it matters to OSS:** The temporal aliasing → ghosting diagnosis directly maps to OSS-FX at α∈(0,1). The triple-rendering supervision (past, current, future) is worth investigating for OSS-FX training loss design.

### Inference Optimization

- **[DNN Compilation with Virtual Tensors for Data Movement Elimination]** (arXiv:2604.09558, Apr 2026) — Eliminates redundant data movement in DNN inference through virtual tensor fusion. Demonstrated on LLM decoding, but technique generalizes to any DNN. **Why it matters to OSS:** The data-movement optimization is directly applicable to OSS's per-vendor kernel strategy. GPU inference bottlenecks are often memory-bandwidth-bound, not compute-bound — virtual tensor fusion could reduce HBM traffic in the OSS rasterizer + backbone composite. Low priority but worth bookmarking for the kernel optimization phase.

---

## Archive.org Findings

No substantive new archival discoveries this scan. The Gaussian splatting SR paper landscape is well-covered by arXiv and major conferences (CVPR/ICCV/NeurIPS/AAAI 2026). Key papers (GSASR, GaussianSR, ContinuousSR, SuperGS) all have live arXiv pages and active GitHub repos — no takedowns or version discrepancies detected. The 3DGS³ paper (arXiv:2605.11489) code is still "pending acceptance" — archive the preprint page in case of takedown.

---

## GitHub / Open Source

- **[R5DGS](https://github.com/be2rlab/r5dgs)** — **NEW.** Code for the semantic-aware 4DGS with rigid-body constraints paper (arXiv:2605.25909). Just published (May 25). The centroid-propagation optimization for 11 FPS speedup during extrapolation is worth profiling if the code runs on consumer GPUs.

- **[OptiScaler](https://github.com/optiscaler/OptiScaler)** — Continues to demonstrate real market demand for cross-GPU upscaler swapping. The expanding FSR 4 compatibility list is the leading indicator of how quickly the community backports vendor-locked features. **Why it matters to OSS:** Validates the DLL-shim integration approach. If OptiScaler can swap DLSS→FSR→XeSS at the DLL level, OSS can do the same. The community is doing the reverse-engineering work OSS would need for its own shim.

- **[NVIDIA RTX Neural Shaders + Slang (GDC 2026)]** — NVIDIA open-sourced more RTX Neural Shading SDK components. The Slang shader compiler now supports inline cooperative vector operations for neural texture compression and radiance caching. **Why it matters to OSS:** NVIDIA is making neural rendering a first-class shader concept. If in-shader neural models become standard, OSS's student model may need to compete not just with DLSS the driver component but with neural shaders embedded directly in game rendering pipelines. The DirectX Compute Graph Compiler (private preview Summer 2026) will make full-model ML deployment in games trivial — potentially commoditizing what OSS is building.

- **[Microsoft DirectX Linear Algebra (Public Preview April 2026)]** — DX now has first-class matrix-matrix operations in HLSL (Shader Model 6.9+). DirectX Compute Graph Compiler coming Summer 2026 for full-model ML graphs. All four GPU vendors committed (AMD, Intel, NVIDIA, Qualcomm). **Why it matters to OSS:** This is THE platform play for neural upscaling. DirectX Linear Algebra means OSS's student model could theoretically run as an inline HLSL compute shader without going through TensorRT/DirectML. The Compute Graph Compiler would handle model → GPU optimization automatically across vendors — potentially making OSS's cross-vendor kernel effort partially redundant. **This is a major development — DirectX is becoming the universal ML runtime for games.**

---

## News

### NVIDIA DLSS 5 — Neural Rendering Model (GTC 2026, Fall 2026 ship)
DLSS 5 is NOT an upscaler. Jensen Huang called it "the GPT moment for graphics" — it blends handcrafted rendering with a learned neural model that infers photorealistic lighting and materials. Single unified model across all content (no per-title training). Deterministic and temporally stable. Full artist control with per-region dials. Ships as a driver update to existing RTX 50 Series cards. **Why it matters to OSS:** DLSS 5 represents a category shift — from "AI upscaler" to "AI rendering co-processor." If NVIDIA's neural model handles lighting/materials inference, the SR-only problem OSS targets becomes a subset of a larger capability NVIDIA already ships. OSS's SR+FG unification thesis becomes less differentiated if DLSS 5's unified model also handles SR, FG, AND neural rendering in one architecture. That said, DLSS 5 is RTX 50-series only and closed-source — OSS's cross-vendor, Apache-2.0 positioning remains a unique value proposition.

### AMD FSR 4 Open Source Status
FSR 4 source code reveals potential support for RDNA 3 (RX 7000) and possibly RDNA 2 (RX 6000) GPUs — contradicting the official RDNA 4-only stance. FSR 4.1 expansion to RX 7000 (July 2026) and RX 6000 (early 2027) announced. FSR 4 Unreal Engine 5.7 plugin updated. **Why it matters to OSS:** AMD is back-filling the cross-vendor gap OSS targets. If FSR 4 runs on NVIDIA GPUs (as OptiScaler compatibility lists suggest), AMD becomes the de facto "open" ML upscaler — competing directly with OSS's value proposition. Time-to-market for OSS to claim the "first truly open, cross-vendor ML upscaler" position is narrowing rapidly.

### Apple Metal 4 + MetalFX (WWDC 2025)
Metal 4 introduces MetalFX Frame Interpolation and Denoising (experimental). Shader ML lets developers embed neural networks directly in shaders. Metal 4 supports sparse resources and expanded instruction set for gaming. **Why it matters to OSS:** Apple entering the neural upscaling + frame interpolation space with first-party APIs validates the market. MetalFX Frame Interpolation is a direct competitor to OSS-FX on Apple Silicon. The Shader ML feature is Apple's answer to DirectX Linear Algebra — same concept, different platform. OSS's Metal kernel path (currently designed, zero lines of code) now has a moving target to integrate with.

### Intel XeSS 3 / Arc B770
Intel XeSS 3 Multi-Frame Generation shipping on Arc B580 and B770 GPUs. SDK publicly released — 3× and 4× frame generation modes. Intel confirmed XeSS 3 comes to Arc B580, making it the budget GPU recommendation for frame-gen gaming. **Why it matters to OSS:** XeSS 3's open SDK and multi-vendor compatibility (it runs on AMD/NVIDIA GPUs via DP4a fallback) makes it the most philosophically aligned competitor to OSS. If XeSS 3 quality catches up to DLSS 4 while remaining open, it becomes the benchmark OSS needs to beat for the "open upscaler" crown.

### Unreal Engine 5.7
UE 5.7 now shipping with updated FSR 4 plugin support. DLSS 4.5 Super Resolution plugin issues reported by developers (not working in 5.7.4). TSR (Temporal Super Resolution) remains the built-in platform-agnostic option. **Why it matters to OSS:** UE 5.7 is the current shipping engine version — OSS's cross-engine strategy should target 5.7 as the primary integration target. The DLSS 4.5 plugin issues suggest NVIDIA's UE integration isn't seamless — opportunity for a drop-in replacement that "just works."

---

## Action Items / Follow-ups

- [ ] **CRITICAL: Investigate trainer stall.** v7-pico-005 stuck at step 9,100 for 7min+. Only 200 steps in 24h. The WSL2 deadlock revert may not have fixed the root cause. Check if the process is alive, GPU utilization, and whether manual restart is needed.
- [ ] **Deep-read R5DGS (arXiv:2605.25909).** Rigid-body centroid propagation for 11 FPS extrapolation speedup. Code available at github.com/be2rlab/r5dgs — benchmark the optimization technique.
- [ ] **Deep-read 3DGS³ (arXiv:2605.11489).** Joint SR + frame interpolation in a single post-rendering framework. Still waiting for code release — archive the preprint.
- [ ] **Track DirectX Compute Graph Compiler (Summer 2026 private preview).** This could commoditize cross-vendor ML deployment in games — potentially making OSS's per-vendor kernels redundant. Join the preview if possible.
- [ ] **Reassess OSS positioning vs DLSS 5.** DLSS 5's "unified neural rendering model" thesis is structurally similar to OSS's "one architecture for SR+FG" thesis, with the addition of lighting/material inference. OSS needs to articulate what it offers that DLSS 5 doesn't: cross-vendor, open-source, no RTX requirement.
- [ ] **Monitor FSR 4.1 RX 7000 launch (July 2026).** If AMD's ML upscaling becomes available on older GPUs and runs on NVIDIA hardware via OptiScaler, AMD becomes the default "open" option — OSS's window is closing.
- [ ] **Benchmark RetimeGS temporal anti-aliasing approach for OSS-FX.** The triple-rendering supervision strategy could improve OSS-FX ghosting prevention at α∈(0,1).
- [ ] **Survey X4Rate scene-adaptive approach for OSS-FX.** If no single α value or extrapolation mode works best across all game content, build an adaptive selector into OSS-FX.
