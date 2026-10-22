# OSS Research Brief — 2026-07-09

## Project Status Summary
- Latest commits: `a97a180 feat: add tailnet-gated training controls to dashboard`; `fcf8084 feat: tailnet-gated training control API + dashboard panel`; `0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc`; plus DataLoader/WindowsPath/flash-attn/TF32/dashboard/rasterizer redirect fixes in the remaining recent history.
- Open issues: #14 D3: measure v6.2-pico-002 VRAM footprint after v6.2 wiring and checkpoint exist; #13 H004: run Nsight L2 hit-rate profile on 3080 Ti CUDA host; #12 H004: implement packed low-rank canvas layout before claiming 32B/Gaussian; #10 v6.2.1 ablation: decoupled G-buffer encoder before concat-fusion; #9 v6.2.1 ablation: RepVGG/ECBSR/RepSR reparameterization for student backbone; #8 v6.2: cross-vendor kernel ports (HIP / Metal / Vulkan / SYCL); #7 v6.2.1: distillation pipeline with FSR 4 binary as teacher signal; #6 v6.2-pico-002: launch on 3080 Ti once codex T1-T5 land; #5 v6.2: No-ML shader variant (ships alongside v6.2); #4 v6.2: HAT-Tiny student distillation (≤1M params).
- README status snapshot: pre-alpha; v7 is the current research/teacher architecture (`srcnn-v7.0-pico-005`) with native-time N-D Gaussian canvas, parent-child density, Mip-Splatting filters, and α=0.5 OSS-FX supervision. Shipping student remains not yet trained; target remains ≤1M params, TensorRT FP8/custom cross-vendor kernels, <2ms-class inference.
- Training dashboard: headline still says `v7.0 Pico OSS-FX is training right now`; active run `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz step **26,000**, updated/live. Health tooltip says **Trainer — offline, stuck at step 30300 for 92h**; **Watcher — healthy, published 0s ago**; **CF Worker — healthy, HTTP 200 in 98ms**; **R2 origin — healthy, HTTP 200**. GPU stats unavailable: “host offline or rsync failed.” Compared with the existing same-day draft, stuck-duration advanced **68h→92h** with no step/loss progress.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **DeGenseGS: Geometrically and Semantically Decoupled Surgical Scene Understanding in 4D Gaussian Splatting** (Yimo Wang, Bin Kang, Shuojue Yang, Yueming Jin; arXiv:2607.04761v1; 2026-07-06) — Decouples geometry and semantics for real-time, text-promptable 4D surgical scene reconstruction. Why it matters to OSS: Off-domain surgical 4DGS, but the geometry/semantic decoupling principle is relevant if OSS later adds object-aware/transient canvas channels rather than baking semantics into motion warp state.
- **GaussianEmoTalker: Real-Time Emotional Talking Head Synthesis with Audio-Driven and Blendshape-Based 3D Gaussian Splatting** (Haijie Yang, Zhenyu Zhang, Yixuan Dong, Jianjun Qian, Jian Yang; arXiv:2607.00959v1; 2026-07-01) — Uses audio/blendshape controls over 3D Gaussian avatars for real-time expressive talking-head synthesis. Why it matters to OSS: Off-domain talking-head work, but another real-time dynamic-3DGS datapoint; useful only as a low-priority reference for feed-forward deformation/control, not a direct SR/FX architecture lead.
- **GUSH3R: Everyone Everywhere All at Once as Gaussians** (Keito Abe, Kaede Shiohara, Takashi Otonari, Toshihiko Yamasaki; arXiv:2607.05243v1; 2026-07-06; already in 2026-07-08 bibliography) — Feed-forward online dynamic human-scene Gaussian reconstruction. Why it matters to OSS: off-domain but useful for transient/dynamic-object canvas separation; no new bibliography append needed.

### Super-Resolution / Upscaling
- **PhyMRI-SR: Toward Physics-Aware MRI Image Super-Resolution** (Lihua Wei, Huatong Gao, Jia Gong, Zhiyu Tan, Hao Li, Jun Liu, Zhihua Ren; arXiv:2607.06238v1; 2026-07-07) — Uses 2D Gaussian Splatting / coordinate reconstruction for physics-aware MRI super-resolution under acquisition constraints. Why it matters to OSS: Mostly off-domain; the transferable lesson is degradation-physics-aware SR, not game runtime architecture.
- No fresh, clearly game-focused SR/upscaling paper surfaced today. Prior active references remain GS-STVSR, GRAPE, GSASR, Resonant Brane Splatting, adaptive Gaussian allocation, AVSR-Diff, and LCS.

### Frame Extrapolation / Generation
- No genuinely new frame-extrapolation / single-network frame-generation arXiv lead surfaced today beyond already-tracked ANVIL, GFFE, CausalGS, VeloGauss, Predictive Differentiable Rendering, and RSTT.

### Inference Optimization
- No fresh TensorRT/INT8/kernel-fusion SR paper surfaced in today's primary/fallback checks. Current follow-up remains QAT/FP8-INT8 validation plus fused kernels for the ≤1M-param student; export-only claims are still insufficient.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 hits; sample: github.com-Awesome-Image-Registration-Organization-awesome-image-regis, github.com-pbsds-starred_-_2025-08-25_04-40-27. No actionable new paper mirror/update verified from the broad inventory.
- `"frame generation" "deep learning"`: 4 hits; sample: DLSS Swapper builds for Windows, Minecraft-Radiance - Radiance 2026-06-13. No actionable new paper mirror/update verified from the broad inventory.
- `"real-time upscaling" game`: 0 hits; sample: no useful sample. No actionable new paper mirror/update verified from the broad inventory.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16613 hits; sample: EPISODE44 - 'Will that be One Grape or Two?', Nadia Grape Favorites. No actionable new paper mirror/update verified from the broad inventory.
- Targeted Wayback/known-paper monitoring remains best for GS-STVSR / GSASR / GRAPE / STSS; broad acronym search is still noisy, especially `GRAPE`.

## GitHub / Open Source
- Query `gaussian splatting super-resolution pushed:>=2026-06-25`: 1 hit — **GiulioFede/ResonantBraneSplatting**, pushed 2026-07-01, Python, 0 stars, no license. Same lead as prior brief; use as paper/idea only unless licensing changes. <https://github.com/GiulioFede/ResonantBraneSplatting>
- Query `frame generation open source pushed:>=2026-06-25`: 2 low-signal hits — **OULDRABAH95/AllPen-framework-** and **Shepherd217/TuneFrames**; neither currently looks like a game-frame-generation implementation lead from repository metadata.
- Query `neural upscaler pushed:>=2026-06-25`: 7 hits; most are docs/landing pages. **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** remains the actionable lead: pushed 2026-07-02, Python, MIT, 4 stars; static FSR 4.1.0 upscaler analysis / weight-layout reconnaissance. <https://github.com/Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling>
- Query `FSR 4.1 upscaling pushed:>=2026-06-25`: 1 hit — same **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** repo. Verify claims before using it for issue #7.

## News
- `DLSS 4 developments` RSS leads: none
- `FSR 4 developments` RSS leads: Foreign Subsidies Regulation Quarterly (FSRQ) Q2 2026 - White & Case LLP; Why AMD FSR 4 Backward Support Matters For PC Gamers In 2026 Breeze Airways International Routes 2026 (KUXbAbUuih) - Mshale
- `Intel Arc XeSS updates` RSS leads: Intel Arc G3 Extreme: 12 Xe Cores Hit Handhelds [2026] - tech-insider.org; Arc G3-powered MSI Claw 8 EX AI+ handheld goes up for pre-order - KitGuru; RTX 5070 Vs ARC B580 Vs RX9060XT | DLSS 4.5 Vs XeSS 3.0 Vs FSR 4.1 | Test Game 1440p Cristel Carrisi (WPyfXZ4EAL) - Mshale
- `Apple Metal MetalFX performance` RSS leads: Metal Gaming Will Decide the Mac’s Next Performance Leap - AppleMagazine - AppleMagazine; Game Porting Toolkit Gives Apple a New Pitch to PC Developers - AppleMagazine - AppleMagazine
- `Unreal Engine Unity upscaling update` RSS leads: none
- Primary-page spot checks still show NVIDIA DLSS, GPUOpen/FidelityFX, Intel XeSS, and Apple Metal pages reachable, but no verified new technical delta was found from primary sources during this run. Treat RSS items as headline-level leads, not confirmed SDK/API changes.

## Action Items / Follow-ups
- [ ] CRITICAL ops: trainer remains offline/stuck at step 30,300, now **92h**; check trainer process, metrics emission, rsync/upload path, and dashboard health-state semantics. Watcher/CF/R2 are healthy, so focus on the 3080 Ti host/trainer.
- [ ] Deep-read **TemporalGS** + **FreeTimeGS++** + **Flow-based 4DGS** together for v7 temporal cache, active-mask, density-duration, and motion-representation choices.
- [ ] Keep **DeGenseGS**, **GaussianEmoTalker**, and **PhyMRI-SR** low priority; all are real primary arXiv results but off-domain for game SR/FX.
- [ ] Verify **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** claims before using it for FSR4 teacher/operator-layout work.
- [ ] Continue watching AMD FSR 4.x RDNA3/open-source movement and DLSS 4/4.5 messaging as competitor-positioning threats while OSS lacks student/runtime evidence.
