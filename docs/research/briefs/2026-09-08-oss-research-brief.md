# OSS Research Brief — 2026-09-08

## Project Status Summary

- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke` (the local checkout currently exposes only this commit in its log).
- **Open issues:** 12. Immediate architecture/runtime work remains #1/#3/#12/#13/#14 (latent rasterizer, tile sorting, packed canvas, VRAM and Nsight); shipping work remains #4/#7/#8/#9 (≤1M student, FSR 4 teacher signal, vendor kernels, reparameterized backbone).[1]
- **README snapshot:** pre-alpha v7 research. The native `(x,y,t)` Gaussian canvas / fractional-time OSS-FX teacher is the active architecture; the ≤1M TensorRT student and non-CUDA backends do not exist yet.[2]
- **Training dashboard:** `srcnn-v7.0-pico-005` remains at **36,650 / 100,000**, total loss **0.1926767** (SR Charbonnier 0.0235991; FG Charbonnier 0.0764263). `data.json` still presents it as active/training, while `status.json` checked at 2026-09-09T00:05:11Z says trainer **offline, stuck for 221h**; watcher, Worker, and R2 are healthy.[3][4]
- **Operational conclusion:** step/loss remain unchanged while the offline counter advanced from 209h to 221h. This is a persistent trainer outage, not recovery; the dashboard's “training now” copy conflicts with its health endpoint. Browser rendering was unavailable because Chrome is absent in this cron host, so the public dashboard JSON endpoints were used.

## New Papers (arXiv)

Targeted arXiv Atom queries across Gaussian-temporal, VSR/distillation, frame generation, inference, and engine-MV terms were rate-limited in this run (four HTTP 429 responses and one HTTP 503). **No newly verified primary-metadata paper was added.** The material results below are carried forward from the prior verified scan and already tracked in `BIBLIOGRAPHY.md`, so no duplicate bibliography entry was appended.

### Gaussian Splatting / Temporal

- **TileGS: Tile-Local Depth Binning for Gaussian Splatting Rasterization** (Wei Tan, Matias Turkulainen, Lauri Ilola, Hamed Rezazadegan Tavakoli, Juho Kannala; 2026-09-03; arXiv:2609.03613) — selectively repairs ordering inside tile-local depth ranges to reduce sort/raster work. **Why it matters to OSS:** direct control for #3/#12/#13; test its idea only against the 2D temporal tile bin with frame time, L2/DRAM bytes, instructions, and compositing error—not a transplanted 3DGS speedup claim.[5]
- **CC-4DGS**, **EvoGS**, and **Atlas** remain the useful recent temporal-canvas references, but were already catalogued September 1–4. No new direct 2D-Gaussian-SR or exact-engine-MV method surfaced.

### Super-Resolution / Upscaling

- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11; arXiv:2607.10495) — uses structural reparameterization to deploy as standard convolutions. **Why it matters to OSS:** best current student-backbone control for #4/#9; benchmark a fused ≤1M TensorRT graph at OSS resolutions rather than extrapolating its edge-device results.[6]
- No new real-time game-SR, engine-MV-conditioned, ≤1M student-distillation result was material.

### Frame Extrapolation / Generation

- No newly material single-network extrapolation result surfaced. Recent generic VFI hits remain unsuitable substitutes for OSS-FX because OSS has exact engine motion vectors and a shared time-indexed canvas.
- Competitive differentiation remains intact at the research level: rendering the same canvas at another `t_query` avoids a second FG network, but still needs measured quality/latency before comparison with vendor FG.

### Inference Optimization

- TileGS remains the actionable new-ish raster lead; no fresh TensorRT/INT8/kernel-fusion paper alters the sequencing: establish packed-canvas + student baselines, then measure fusion and precision per backend.[5][6]

## Archive.org Findings

- Archive.org advanced searches for `"gaussian splatting" super-resolution`, `"frame generation" deep learning`, and `"real-time upscaling" game` returned either unrelated GitHub/media inventory or no hits.
- Targeted labels were similarly noisy: GS-STVSR had no result, GSASR resolved to an unrelated CIA item, and GRAPE/STSS were acronym collisions. This is an inventory-quality result—not evidence that copies were removed or papers were taken down.[11]
- No usable archived GS-STVSR, GSASR, GRAPE, or STSS copy was discovered this run.

## GitHub / Open Source

- **Integration watch / opportunity:** [ShyVortex/dlss-unlocked](https://github.com/ShyVortex/dlss-unlocked) is a new MIT project (created Sep 4; **176 stars** at the latest scan) claiming DLSS SR/MFG/neural-rendering feature unlocking for native DX12 DLSS2/3 titles. It validates demand for a compatibility/interception surface, but its claims are not OSS benchmark evidence. Read only for clean-room integration ergonomics.[7]
- [HECer/DLSSG-Stream-Worker](https://github.com/HECer/DLSSG-Stream-Worker) is an MIT D3D12 *offline* DLSS Frame Generation worker. Not a real-time competitor; potentially useful only as a D3D12 workflow reference.[8]
- Searches produced no credible new open-source DLSS/FSR analysis tool. Most other hits were low-activity, video/ComfyUI, or no-license projects and were rejected.

## News

- **NVIDIA — verified:** NVIDIA says DLSS 4.5 Ray Reconstruction is available via the NVIDIA App and describes a second-generation RR transformer; its Gamescom post also references DLSS 4.5 Super Resolution, Dynamic MFG, 6× MFG, and enhanced FG. This raises the vendor feature/quality bar; OSS must not imply parity until independently measured.[9]
- **AMD — primary-site lead:** GPUOpen currently links “AMD FSR ‘Redstone’ SDK + the neural rendering future” and “FSR Upscaling 4.1 RDNA 3 support now available.” Inspect the actual SDK/release API before translating this into architecture work.[10]
- **Intel / Apple / engines:** no primary Intel XeSS/Arc, Apple MetalFX, Unreal, or Unity update in the last 14 days was verified as changing OSS positioning. Google News RSS yielded mostly secondary, rumor-quality, or unrelated claims and was not treated as confirmation.

## Action Items / Follow-ups

- [ ] **P0:** recover or formally stop `srcnn-v7.0-pico-005`; it is stuck at 36,650 with trainer health reporting 221h offline despite healthy publication infrastructure.
- [ ] **#3 / #12 / #13:** establish packed-canvas telemetry first, then run a TileGS-style local-depth-binning ablation; record frame time, L2 rate, DRAM bytes, instructions, and output error.
- [ ] **#4 / #9:** benchmark a NanoVSR-style reparameterized ≤1M CNN against RepVGG/ECBSR/RepSR in the intended fused TensorRT graph.
- [ ] **#12:** retain CC-4DGS/EvoGS as compression/spawn controls, rejecting layouts that harm temporal identity or disocclusion even if bytes/Gaussian improve.
- [ ] **Integration watch:** inspect `dlss-unlocked` and `DLSSG-Stream-Worker` for clean-room D3D12/DLL-shim ergonomics only; do not import behavior or market their compatibility claims.

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[2] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[3] https://opensupersampling.com/data.json
[4] https://opensupersampling.com/status.json
[5] https://arxiv.org/abs/2609.03613
[6] https://arxiv.org/abs/2607.10495
[7] https://api.github.com/repos/ShyVortex/dlss-unlocked
[8] https://api.github.com/repos/HECer/DLSSG-Stream-Worker
[9] https://www.nvidia.com/en-us/geforce/news/gamescom-2026-nvidia-geforce-rtx-dlss-4-5-announcements
[10] https://gpuopen.com
[11] https://archive.org/advancedsearch.php
