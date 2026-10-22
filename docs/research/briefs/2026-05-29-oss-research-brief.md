# OSS Research Brief — 2026-05-29

## Project Status Summary

**Latest commits (top 10):**
```
0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc
56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)
d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume
e1ab7fe hat.py: use flash-attn 2 directly when available, fallback to SDPA
e9b4b65 trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)
26cd9da dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB
83b4f07 v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast
baf4b02 worker: apex .com → .org 301 redirect
eb0ec2e trainer: bf16 autocast around forward + loss
5c2f2e7 trainer: add throughput baselines (cudnn.benchmark, TF32, torch.compile)
```

**Same commits as yesterday — no new activity.** v7-pico-005 trainer still appears offline at step 5,000.

**Open issues (12):** #14 (VRAM footprint), #13 (Nsight L2 profile), #12 (packed low-rank canvas), #10 (decoupled G-buffer encoder), #9 (RepVGG/ECBSR/RepSR), #8 (cross-vendor kernels), #7 (FSR 4 distillation), #6 (v6.2-pico-002 launch), #5 (No-ML shader), #4 (HAT-Tiny student distillation), #3 (counting-sort tile binning), #1 (R=4-8 low-rank latent rasterizer). No new issues opened. No issues closed.

**Training dashboard:** Trainer appears idle. No new training metrics since step-5000 snapshot. Recovery doc written (commit `0660f55`) suggests intentional pause.

---

## 🚨 MAJOR INDUSTRY SHIFTS — DLSS 5 & FSR Diamond

### NVIDIA DLSS 5 (Announced GTC 2026, arriving Fall 2026)

**This is the biggest competitive move since DLSS 1.0.** NVIDIA announced DLSS 5 at GTC 2026 (March 16) — a **real-time neural rendering model that infuses pixels with photorealistic lighting and materials**. Key details:

- **Not just upscaling anymore** — DLSS 5 is a full neural rendering pipeline. It takes game engine output and *re-renders* it with AI-generated photorealistic lighting, materials, and detail. Jensen Huang called it the "GPT moment" for real-time neural rendering.
- **DLSS 4.5** (the bridging release) introduced a second-generation transformer model for Super Resolution, available in 400+ titles via NVIDIA app overrides, plus Dynamic Multi-Frame Generation.
- **DLSS 5** goes beyond: single model that handles SR + FG + ray reconstruction + neural material synthesis. Available to developers via Unreal Engine plugin and direct SDK.
- Backed by Bethesda, CAPCOM, NetEase, NCSOFT, S-Game, Tencent, Ubisoft at launch.
- **Competitive threat level: CRITICAL.** DLSS 5's unified neural rendering model validates OSS's "one architecture not two" thesis, but NVIDIA is shipping it commercially first. The "photorealistic materials" claim suggests DLSS 5 is doing something akin to neural texture synthesis, not just pixel reconstruction — this raises the bar for what "super-resolution" means.

### AMD FSR Diamond (Announced GDC 2026, RDNA 5 / Project Helix)

AMD announced FSR Diamond at GDC 2026 — their answer to DLSS 5:

- **Full neural rendering stack**: ML-based upscaling, multi-frame generation, neural rendering, and **ray regeneration** (AMD's answer to DLSS Ray Reconstruction).
- **Co-engineered with Microsoft** for Project Helix (next-gen Xbox). RDNA 5 exclusive — will NOT run on RDNA 4 or earlier hardware.
- **FSR 4.1** (recently announced) brings ML upscaling to RX 9000 series, but FSR Diamond is the architectural leap.
- **FSR Redstone** (current gen) delivers 4 neural rendering features for RDNA 4: neural texture compression, neural denoising, etc.
- **FSR SDK 2.2** (March 2026) did NOT include multi-frame generation — confirming MFG is reserved for Diamond.
- **Competitive threat level: HIGH.** The RDNA 5 exclusivity fragments AMD's ecosystem (FSR was historically open), but Diamond's MS co-engineering means it'll be the default on next-gen Xbox. This validates OSS's cross-vendor positioning — if FSR becomes hardware-locked, OSS's vendor-neutral pitch gets stronger.

### Intel XeSS 3 (Now shipping)

- Intel officially rolled out XeSS 3 Multi-Frame Generation for Arc Alchemist and Battlemage GPUs.
- XeSS 3 support also enabled on new Panther Lake (Core Ultra Series 3) integrated graphics with Xe3 architecture.
- **However:** Intel cancelled Xe3P "Celestial" discrete gaming GPUs — roadmap now focuses on AI and workstation. This narrows the discrete GPU upscaling landscape to NVIDIA + AMD.
- **Competitive threat level: LOW-MEDIUM.** XeSS 3 is shipping but Intel's discrete GPU pullback reduces its ecosystem importance.

### Arm Neural Super Sampling (NSS)

Arm announced Neural Super Sampling — a mobile-optimized ML upscaling technique targeting late 2026 hardware:

- **4ms at 540p → 1080p** on simulated next-gen Arm GPU silicon (Mali-derived).
- Open-source plugin for Unreal Engine already available: `github.com/arm/neural-graphics-for-unreal`
- Pretrained models on Hugging Face: `huggingface.co/Arm/neural-super-sampling`
- **This is a direct competitor to OSS's Pico tier** (mobile/Steam Deck). Arm ships hardware + software integrated; OSS would need to compete on quality and openness.
- **Competitive threat level: MEDIUM.** Shipping late 2026. Arm has the integration advantage (first-party silicon) but OSS has the architecture advantage (unified SR+FG, not two networks).

---

## New Papers (arXiv)

### Gaussian Splatting / Temporal

1. **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047, Apr 27 2026)
   - Mingyu Shi, Xin Di, Long Peng, Bo Cao, Anran Wu, Zheng Feng, Jun Guo, Renjun Pei
   - **First 2D Gaussian Splatting framework for continuous space-time video super-resolution.** Represents frames as deformable 2D Gaussians whose parameters evolve temporally via learned kernels. Claims ultra-efficient C-STVSR.
   - **Why it matters to OSS: DIRECT ARCHITECTURAL OVERLAP.** GS-STVSR uses 2D Gaussians + temporal deformation — the same primitive class as OSS v7's N-D Gaussian canvas. Their approach to spatiotemporal evolution of Gaussian parameters could inform OSS's V_xt cross-correlation design. This paper is already in OSS's bibliography (see `docs/research/2026-05-13-gs-stvsr-classification.md`) and classified as "mirroring" OSS's deformation approach.
   - **Action:** Re-read the latest version; check if their temporal kernel learning has improved since the earlier classification.

2. **Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling (ContinuousSR)** (ICLR 2026, arXiv:2503.06617)
   - Long Peng, Anran Wu et al. (USTC + Huawei Noah's Ark Lab)
   - A Pixel-to-Gaussian paradigm that reconstructs 2D continuous HR signals from LR images using Gaussian Splatting. Claims ultra-fast arbitrary-scale SR. ICLR 2026 accepted.
   - **Why it matters to OSS:** Directly adjacent — 2D Gaussian SR, arbitrary-scale, fast. Their approach to mapping pixels→Gaussians could inform OSS's canvas initialization and the student model's Gaussian prediction head. ICLR acceptance signals this is recognized as a competitive direction.
   - **Action:** Deep-read their rasterization efficiency; code is open-source at `github.com/peylnog/ContinuousSR`.

3. **SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting** (arXiv:2512.02172)
   - CVPR 2026. Selectively applies SR only in regions lacking high-frequency detail, preserving multi-view consistency.
   - **Why it matters to OSS:** The selective SR approach — applying super-resolution only where needed — is an optimization strategy OSS could adopt for sparse canvas regions. Less relevant to real-time inference but useful for training efficiency.

4. **Progressive Zoom-in Generative 3D Gaussian Splatting (GaussianZoom)** (arXiv:2605.18252, May 2026)
   - Iterative progressive framework combining geometry-guided refinement with Gaussian splatting for zoom-in reconstruction.
   - **Why it matters to OSS:** Progressive refinement strategy could inspire multi-pass rendering approaches for OSS.

5. **R5DGS: Semantic-Aware 4D Gaussian Splatting with Rigid Body Constraints** (arXiv:2605.25909, May 25 2026)
   - Semantic-aware 4D Gaussians for dynamic scenes. Already in OSS bibliography from May 28 brief.
   - No new version since last check.

### Super-Resolution / Upscaling

6. **Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Gaming Content** (arXiv:2602.11339)
   - Benchmark of real-time SR methods specifically on gaming content. Evaluates existing methods against game-specific challenges.
   - **Why it matters to OSS:** Directly relevant — this is a survey of exactly the space OSS operates in. Their benchmark methodology and identified failure modes for game content should inform OSS's eval pipeline.
   - **Action:** Deep-read their game-specific findings; this could provide an external validation baseline.

7. **OSDEnhancer: Taming Real-World Space-Time Video Super-Resolution with One-Step Diffusion** (arXiv:2601.20308)
   - Shuoyan Wei, Huihui Bai et al. First framework achieving real-world STVSR in one-step diffusion. Uses divide-and-conquer with TC-LoRA and TE-LoRA fine-tuned on a shared DiT backbone.
   - **Why it matters to OSS:** Validates the unified SR+FG approach (single model does both). Their one-step diffusion is still too heavy for real-time games, but the architectural pattern — shared backbone + task-specific adapters — is exactly what OSS's v7 canvas achieves through different α values.
   - **Action:** Note the LoRA-based task specialization pattern for potential student distillation approach.

8. **FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution** (CVPR 2026, arXiv:2510.12747)
   - Block-sparse attention + tiny conditional decoder + three-stage distillation. ~17 FPS at 768×1408 on A100.
   - **Why it matters to OSS:** The block-sparse attention pattern could apply to OSS's rasterizer attention. Their three-stage distillation pipeline is a case study for OSS's teacher→student path. But 17 FPS on A100 for video SR means still 2-3× too slow for real-time gaming at equivalent resolutions.
   - **Action:** Study their block-sparse attention implementation — could inform H004 attention optimizations.

9. **CDA-VSR: Compressed-Domain-Aware Online Video Super-Resolution** (CVPR 2026 Highlight, arXiv:2603.07694)
   - Motion-vector-guided deformable alignment using compressed-domain information (motion vectors, residual maps, frame types).
   - **Why it matters to OSS:** Uses motion vectors explicitly for SR alignment — exactly what OSS does with engine motion vectors. CVPR Highlight status signals this approach is recognized as important. Their MV-guided deformable alignment module could directly inform OSS's canvas warping.
   - **Action:** High-priority deep-read. This is the closest academic work to OSS's motion-vector-driven architecture. Check their code at `github.com/sspBIT/CDA-VSR`.

### Frame Extrapolation / Generation

10. **Consecutive Frame Extrapolation with Predictive Sparse Shading** (ACM SIGGRAPH 2025 / TOG)
    - Yazhen Yuan et al. Neural network for consecutive frame extrapolation using sparse shading in real-time rendering.
    - **Why it matters to OSS:** Directly relevant to OSS-FX. Uses sparse shading — analogous to OSS's sparse Gaussian canvas — for frame extrapolation. Their approach to predicting future frames from sparse samples is architecturally aligned.
    - **Action:** Deep-read; this may be the closest published work to OSS-FX's approach.

11. **DiffST: Spatiotemporal-Aware Diffusion for Real-World Space-Time Video Super-Resolution** (arXiv:2605.13182, May 13 2026)
    - Already in bibliography from May 28 brief. No new version.

### Inference Optimization

12. **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization** (arXiv:2604.20291, Apr 2026)
    - Deployment-aware quantization pipeline: QAT on fused deploy graph with weight clipping + BatchNorm recalibration. Won MAI 2026 Quantized 4K Image Super-Resolution challenge.
    - **Why it matters to OSS:** Directly applicable to OSS's student model deployment. Their deployment-aware QAT approach (quantize the fused graph, not the training graph) is exactly the technique needed for TensorRT FP8 export of the ≤1M-param student. MAI 2026 challenge win validates the approach.
    - **Action:** Apply their BN recalibration + fused-graph QAT approach to OSS student distillation pipeline (issue #7 territory).

13. **Accelerating 3D Gaussian Splatting using Tensor Cores** (arXiv:2605.17855, May 18 2026)
    - Already in bibliography. FP16 3DGS on Tensor Cores. Directly applicable to inference optimization.

### Mobile / Edge SR

14. **NTIRE 2026 Challenge on Mobile Real-World Image Super-Resolution** (arXiv:2604.17306)
    - Review of all submitted solutions to NTIRE 2026 mobile SR challenge. Documents SoTA approaches for mobile-constrained SR.
    - **Why it matters to OSS:** The winning architectures and efficiency tricks are directly relevant to OSS's Pico tier (Steam Deck, integrated GPUs). Documents the current efficiency frontier.

15. **NTIRE 2026 Efficient Super-Resolution Challenge** (arXiv:2604.03198)
    - Broader efficient SR challenge results. Covers both image and video SR efficiency techniques.
    - **Why it matters to OSS:** Documents the efficiency techniques the community considers state-of-the-art. Useful benchmark for OSS's student model targets.

---

## Archive.org / GitHub / Open Source

### No major new archive.org findings
Most relevant papers (GS-STVSR, GSASR, ContinuousSR) are live on arXiv and not taken down. No critical papers appear to have been removed or updated in ways requiring archive retrieval.

### GitHub Activity

- **ContinuousSR** (`github.com/peylnog/ContinuousSR`) — ICLR 2026, open source. 2D Gaussian arbitrary-scale SR. Active repo.
- **CDA-VSR** (`github.com/sspBIT/CDA-VSR`) — CVPR 2026 Highlight. Motion-vector-driven video SR. Code now public.
- **FlashVSR** (`github.com/OpenImagingLab/FlashVSR`) — CVPR 2026. Real-time diffusion VSR. Code public.
- **Arm Neural Super Sampling** (`github.com/arm/neural-graphics-for-unreal`) — Unreal Engine plugin, open source. Mobile DLSS competitor.
- **GaussianSR** (`github.com/tljxyys/GaussianSR`) — Existing repo, no major updates detected.
- **GSASR** (`github.com/chrisdud0257/gsasr`) — Existing ICCV 2025 repo, no major updates.

### Notable: Arm NSS is the biggest new entrant
Arm's Neural Super Sampling plugin for Unreal Engine represents the first serious mobile-focused neural upscaler with first-party silicon integration. This creates a three-way competitive landscape: NVIDIA (DLSS, high-end), AMD (FSR Diamond, mid-to-high), Arm (NSS, mobile). OSS's cross-vendor positioning becomes more valuable in this fragmented landscape.

---

## News

### GPU Market Dynamics
- **NVIDIA may not release new gaming GPUs in 2026** — focusing on AI/data center during RAM shortage. This means DLSS 5 will target existing RTX 50-series hardware.
- **AMD RDNA 5** expected to launch alongside FSR Diamond — Computex 2026 anticipated as the announcement venue.
- **Intel cancels Xe3P discrete gaming GPUs** — shifts to AI/workstation only. The discrete GPU upscaling market is now a two-horse race (NVIDIA/AMD).
- **Apple M5 Pro/Max** shipping with up to 40 GPU cores + Neural Accelerators per core. Metal 4 adds shader ML — embedding neural networks directly in shaders. ~70 TFLOPS FP16 through Neural Accelerators. This makes Apple Silicon a viable target for OSS's Metal backend.
- **Metal Shader ML** (WWDC 2025) allows embedding neural networks within shaders — could this be OSS's Metal inference path?

### Upscaling Ecosystem
- **DLSS 4.5** (shipping now): Second-gen transformer SR + Dynamic Multi-Frame Generation. 400+ titles.
- **DLSS 5** (Fall 2026): Full neural rendering. "Photorealistic lighting and materials." This redefines the category from "upscaling" to "neural rendering."
- **FSR Diamond** (TBA, likely with RDNA 5 / Project Helix): ML upscaling + MFG + ray regeneration + neural rendering. RDNA 5 exclusive.
- **FSR 4.1** (shipping now): ML upscaling for RX 9000 series.
- **XeSS 3 MFG** (shipping now): Multi-frame generation on Arc Battlemage.
- **Arm NSS** (late 2026): Mobile neural upscaling, 4ms at 540p→1080p.

### Game Engine Integration
- **Unreal Engine**: DLSS 5 plugin confirmed. DLSS 4.5 already integrated. FSR Diamond likely to follow via Microsoft partnership.
- **Unity**: Current upscaling features limited (DLSS 3.5, FSR 3.1). Discussions ongoing about newer versions. Unity's slower adoption of new upscaling tech is notable.

---

## Analysis & Strategic Implications for OSS

### 1. The "one architecture" thesis is being validated — by competitors
DLSS 5, FSR Diamond, and OSDEnhancer all move toward **unified neural rendering** rather than separate SR + FG networks. NVIDIA's DLSS 5 is the most aggressive: a single neural rendering model handling SR, FG, ray reconstruction, and material synthesis. OSS's thesis that "one canvas does both SR and FG" is now the industry direction — but the bar has been raised from "two networks → one network" to "one network → neural re-rendering."

### 2. Fragmentation creates opportunity for OSS's vendor-neutral pitch
- FSR Diamond is RDNA 5 exclusive (abandoning AMD's "open FSR" history).
- DLSS 5 requires RTX 50-series Tensor Cores.
- Intel is exiting discrete GPUs.
- Arm NSS requires Arm Mali-next-gen hardware.
- **OSS is the only architecture targeting ALL vendors from Steam Deck to RTX 5090 with one codebase.**

### 3. GPU market pause in 2026 is a window
If NVIDIA isn't releasing new gaming GPUs and Intel is exiting discrete, the installed base of RTX 30/40/50 and RX 6000/7000/9000 is the target. OSS's Pico→Standard→Heavy tier system maps well to this installed base. The 2026 GPU drought means games shipping in 2026-2027 will need upscaling that works on existing hardware — not just the latest generation.

### 4. Academic convergence on Gaussian temporal SR
GS-STVSR, ContinuousSR, CDA-VSR, and OSDEnhancer all use motion vectors or temporal deformation with Gaussian primitives. This is converging toward exactly what OSS v7 is building. The academic frontier is validating the approach; OSS's advantage is the engineering focus on real-time (<2ms) inference, which none of these academic papers address.

### 5. Student distillation is more urgent than ever
DLSS 5's "neural rendering" model is almost certainly a large transformer (given the "GPT moment" framing). OSS's student distillation pipeline (HAT-Tiny teacher → ≤1M-param CNN → TensorRT FP8) is the correct architectural response: match NVIDIA's teacher quality but ship a student that runs on all GPUs, not just RTX 50-series.

---

## Action Items / Follow-ups

- [ ] **URGENT: Deep-read DLSS 5 technical details** — what does "neural rendering" actually mean architecturally? Is it a diffusion model? A transformer? Understanding DLSS 5's architecture is critical competitive intelligence.
- [ ] **URGENT: Deep-read CDA-VSR** (CVPR 2026 Highlight) — motion-vector-guided deformable alignment is directly applicable to OSS's canvas warping. Code is public.
- [ ] **Deep-read GS-STVSR latest version** — check if temporal kernel learning has improved. This is the closest academic paper to OSS v7's approach.
- [ ] **Deep-read ContinuousSR** (ICLR 2026) — their Pixel→Gaussian mapping and rasterization efficiency is the closest open-source 2D Gaussian SR implementation.
- [ ] **Apply INT8 deployment-aware quantization** (arXiv:2604.20291) to OSS student pipeline — their fused-graph QAT + BN recalibration approach is directly applicable.
- [ ] **Investigate Arm NSS architecture** — what model does it use? Can OSS beat 4ms at 540p→1080p on equivalent hardware?
- [ ] **FSR Diamond threat assessment** — if RDNA 5 + FSR Diamond is Xbox-next, that's the console baseline OSS needs to beat. Start tracking RDNA 5 specs.
- [ ] **Metal Shader ML investigation** — Apple's "neural networks in shaders" could be OSS's Metal inference path. Worth a spike.
- [ ] **Block-sparse attention from FlashVSR** — could inform H004 attention optimization for OSS rasterizer.
