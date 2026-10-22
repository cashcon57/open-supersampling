# OSS Research Brief — 2026-09-07

## Project Status Summary

- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke` (the local checkout currently exposes only this commit in its log).
- **Open issues:** 12. Near-term architecture/runtime work remains #1/#3/#12/#13/#14 (canvas layout, tile sorting, VRAM, Nsight); shipping work remains #4/#7/#8/#9 (≤1M student, FSR 4 teacher signal, vendor kernels, reparameterized student).[1]
- **README snapshot:** pre-alpha v7 research; the unified native `(x,y,t)` Gaussian canvas and fractional-time OSS-FX teacher are the active architecture, while the ≤1M TensorRT student and non-CUDA kernel ports do not yet exist.[2]
- **Training dashboard:** `srcnn-v7.0-pico-005` remains at step **36,650 / 100,000**, total loss **0.1926767** (SR Charbonnier 0.0235991; FG Charbonnier 0.0764263). `data.json` still lists the run; `status.json` at 2026-09-08T00:04:27Z says Trainer **offline, stuck for 197h**, while watcher, Worker, and R2 are healthy.[3][4]
- **Operational conclusion:** no step progress from the September 6 brief; the trainer-offline counter advanced by 12h, confirming no recovery. Treat trainer recovery as P0. Chrome was unavailable in this cron environment, so dashboard state was read from its public data/status endpoints rather than a rendered-browser snapshot.

## New Papers (arXiv)

Targeted arXiv Atom searches completed. No paper newer than the already-tracked September 3 set cleared the OSS materiality gate. The most relevant results re-confirmed in this scan are already present in `BIBLIOGRAPHY.md`; no duplicate bibliography entry was appended.

### Gaussian Splatting / Temporal

- **TileGS: Tile-Local Depth Binning for Gaussian Splatting Rasterization** — Wei Tan, Matias Turkulainen, Lauri Ilola, Hamed Rezazadegan Tavakoli, Juho Kannala; 2026-09-03; arXiv:2609.03613. Tile-local depth ranges with selective ordering repair target less raster traversal while preserving gsplat-equivalent output. **Why it matters to OSS:** direct #3/#12/#13 control for the 2D temporal tile bin; profile end-to-end time, L2/DRAM, instruction count, and compositing error rather than importing a 3DGS-only speedup.[5]
- **CC-4DGS: Computational Deformation and Point-Cloud Compression for Storage-Efficient Dynamic Gaussian Splatting** — Kyungdae Park, Chae Eun Rhee; 2026-09-02; arXiv:2609.02184. Replaces heavy deformation storage with compact decoders and conditionally compresses canonical attributes. **Why it matters:** a packed-canvas reference for #12, gated on preserving temporal primitive identity and disocclusion behavior under exact engine motion vectors.[6]
- **EvoGS: Modeling Deformation Evolution for Dynamic Gaussian Splatting** — Wei Dong, Shahram Shirani, Jun Chen, Han Zhou; 2026-09-01; arXiv:2609.00994. Persistent deformation state plus residual-memory correction and uncertainty-aware densification improves dynamic-GS stability. **Why it matters:** a teacher-side residual-confidence/spawn-gating ablation after the exact-MV warp, not a replacement for engine motion vectors.[7]

### Super-Resolution / Upscaling

- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** — Filip Pawlicki, et al.; 2026-07-11; arXiv:2607.10495. A structurally reparameterized all-convolution VSR model collapses to deployment-friendly standard convolutions. **Why it matters:** still the cleanest student-backbone control for #4/#9; benchmark the fused ≤1M TensorRT graph at OSS resolutions rather than extrapolating Jetson results.[8]
- No newly material real-time game-SR, engine-MV-conditioned, ≤1M student-distillation paper surfaced.

### Frame Extrapolation / Generation

- No newly material single-network frame extrapolation result surfaced. Existing **MGMVFI** remains a useful unreliable-motion/contextual-synthesis control after engine-MV warp, but is not an OSS-FX replacement.[9]
- OSS’s differentiated claim remains intact: query the same temporal Gaussian canvas at another `t_query`, rather than train and deploy a second frame-generation model.

### Inference Optimization

- TileGS remains the actionable rasterization lead. The prior art consistently reinforces that tile/sort changes must be evaluated with bandwidth counters and error, not occupancy alone.[5]
- No fresh TensorRT/INT8/kernel-fusion paper changes the existing deployment plan: first establish the student and packed-canvas baseline, then quantify fusion/precision tradeoffs on each vendor backend.

## Archive.org Findings

- Archive.org advanced search for `"gaussian splatting" super-resolution` returned three unrelated archived GitHub-listing items; `"frame generation" deep learning` and `real-time upscaling game` were similarly noisy.
- Targeted Archive.org searches returned no usable GS-STVSR or STSS record; `GSASR` produced an unrelated CIA item and `GRAPE` produced unrelated media. These are inventory-quality results, **not** evidence that a paper/archive copy was removed.
- A targeted Wayback CDX request for known arXiv URLs returned HTTP 400 in this environment, so no preserved-copy conclusion was drawn.

## GitHub / Open Source

- **Integration watch:** [ShyVortex/dlss-unlocked](https://github.com/ShyVortex/dlss-unlocked) is MIT, created September 4, and had 102 stars at scan time. It claims feature unlocking for DLSS SR/FG/neural-rendering APIs in DLSS2/3 DX12 titles. **Threat/opportunity:** validates active demand for a game-side compatibility/interception layer; do not treat its compatibility claims as OSS benchmark evidence.[10]
- **Offline FG tooling:** [HECer/DLSSG-Stream-Worker](https://github.com/HECer/DLSSG-Stream-Worker) is a new MIT D3D12 worker for *offline* DLSS Frame Generation. It is not a real-time OSS competitor, but is worth watching for D3D12 integration ergonomics.[11]
- **Window-level competitors are weak signals:** [Lenovive/OSSS](https://github.com/Lenovive/OSSS) advertises MIT D3D11 optical-flow interpolation/upscaling but had zero stars; it is architecturally distinct from engine-buffer-conditioned reconstruction.[12]
- **Research implementation:** [ZheXues/IE-SRGS](https://github.com/ZheXues/IE-SRGS) is a low-activity AAAI 2026 3DGS-SR repo with no declared license. Read for ideas only; it is not incorporable and does not establish a shipping competitor.[13]

## News

- **NVIDIA:** the August 25 Gamescom primary post remains the confirmed bar: DLSS 4.5 Ray Reconstruction, Dynamic/6× Multi Frame Generation, enhanced FG, and a second-generation RR transformer. OSS should not imply quality or feature parity.[14]
- **AMD:** GPUOpen currently advertises an “AMD FSR Redstone SDK + the neural rendering future” and “FSR Upscaling 4.1 RDNA 3 support now available.” This is a primary-site lead; inspect release notes/API before deriving architecture changes.[15]
- **Apple:** Apple continues to position MetalFX Upscaling, Frame Interpolation, and Denoising as Metal game-performance features. No fresh Apple-Silicon performance announcement was verified in the scan window.[16]
- **Intel / engines:** no primary Intel XeSS/Arc, Unreal, or Unity upscaling release that changes OSS positioning was verified in the last 14 days. Google News RSS contained secondary/low-quality claims only and was not used as confirmation.

## Action Items / Follow-ups

- [ ] **P0:** diagnose/recover `srcnn-v7.0-pico-005`; it is unchanged at step 36,650 and dashboard health reports 185h offline despite healthy publication infrastructure.
- [ ] **#3 / #12 / #13:** run TileGS-style tile-local binning only after collecting the packed-canvas baseline; compare frame time, L2 hit rate, DRAM bytes, instruction count, and output error.
- [ ] **#4 / #9:** build/benchmark a NanoVSR-like reparameterized ≤1M CNN versus RepVGG/ECBSR/RepSR in the intended fused TensorRT deployment graph.
- [ ] **#12:** use CC-4DGS as a compression control; reject any layout that loses temporal identity/disocclusion quality even if it improves bytes/Gaussian.
- [ ] **Integration watch:** inspect DLSS-unlocked/Stream-Worker only for clean-room interception and D3D12 integration lessons; do not import implementation assumptions or market their claims as verified.

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[2] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[3] https://opensupersampling.com/data.json
[4] https://opensupersampling.com/status.json
[5] https://arxiv.org/abs/2609.03613
[6] https://arxiv.org/abs/2609.02184
[7] https://arxiv.org/abs/2609.00994
[8] https://arxiv.org/abs/2607.10495
[9] https://arxiv.org/abs/2608.22861
[10] https://api.github.com/repos/ShyVortex/dlss-unlocked
[11] https://api.github.com/repos/HECer/DLSSG-Stream-Worker
[12] https://api.github.com/repos/Lenovive/OSSS
[13] https://api.github.com/repos/ZheXues/IE-SRGS
[14] https://www.nvidia.com/en-us/geforce/news/gamescom-2026-nvidia-geforce-rtx-dlss-4-5-announcements/
[15] https://gpuopen.com/
[16] https://developer.apple.com/metal/
