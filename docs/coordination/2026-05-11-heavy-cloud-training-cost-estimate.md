# Heavy Model Cloud-Training Cost Estimate

**Filed:** 2026-05-11
**Status:** estimate, anchored on measured pico-002 step time + public cloud GPU pricing as of 2026-05
**Audience:** funders, hardware-loan donors, OSS planning

## What is being estimated

Cost to train **OSS v6.x Heavy** (the ~17M-param HAT-L-derived teacher described in [`RESEARCH.md`](../../RESEARCH.md)) on cloud GPUs to a quality level competitive with **DLSS 4 SR transformer** on cross-engine game footage (UE5, Unity HDRP, Source 2, etc.) — not just on a single synthetic dataset.

Three cost layers, increasing in scope:

1. **Single Heavy training run** — one config, no ablations, no cross-engine fine-tunes. The minimum check that the architecture converges at Heavy scale.
2. **Heavy training cycle** — Heavy teacher + ablations + hyperparameter sweep + Standard/Pico distillation runs + cross-engine fine-tune. The minimum to publish a credible v6.x Heavy result.
3. **Multi-cycle path to DLSS-class quality** — what it takes to actually reach DLSS 4 SR quality cross-engine, not just clear an internal quality bar.

Bottom line up front (best estimate, May 2026 cloud spot pricing):

| Scope | H100-hours | Spot cost (~$1.50/hr) | On-demand cost (~$3/hr) |
| --- | --- | --- | --- |
| Single Heavy training run (300K steps, 1 config) | 1.5K – 3K | $2.3K – $4.5K | $4.5K – $9K |
| Heavy training **cycle** (Heavy + ablations + sweep + distillation) | 10K – 20K | $15K – $30K | $30K – $60K |
| **Path to DLSS-class quality (multi-cycle, cross-engine, 18–24 mo)** | **50K – 150K** | **$75K – $225K** | **$150K – $450K** |

Storage, dataset acquisition, and engineering time are separate; itemized in §6 below.

## 1. Anchor measurements (what we actually know)

Measured on the running v6.2-pico-002 trainer, 2026-05-11:

| Setting | Value |
| --- | --- |
| GPU | RTX 3080 Ti (12 GB, Ampere, FP16 tensor cores) |
| Backbone | HAT-Tiny (~3M G params, ~4.3M D params) |
| Batch | 4 |
| Grad accum | 4 (effective batch 16) |
| Patch | 128² |
| Trajectory length | 4 |
| Precision | bf16 |
| Observed step rate | **~4.5 s/step** (440 steps in 1,990 s; trainer log 12:35:47 → 13:08:57) |

To reach 100K steps at this rate: 100,000 × 4.5 s = 125 GPU-hours on a 3080 Ti, FP16, no torch.compile (Triton missing). Pico-002 will finish in ~5.2 days from cold start; we are at step ~53K (53%) after ~3 days of wall clock.

Dataset on disk: **550 GB** of extracted TartanAir frames (RGB + depth + motion + normals). This is the *only* dataset pico-002 has seen.

## 2. Heavy-vs-Pico scaling factors

What changes going Pico-tier → Heavy-tier (per [`RESEARCH.md` §5](../../RESEARCH.md) and [`docs/architecture/2026-05-08-v62-arch-v4-spec.md`](../architecture/2026-05-08-v62-arch-v4-spec.md)):

| Dimension | Pico-002 (now) | Heavy (target) | Cost factor |
| --- | --- | --- | --- |
| Backbone params | ~3M | ~17M | ~5.7× forward FLOPs (HAT-L attention block stacking) |
| Patch size | 128² | 256² | 4× pixels through every conv/attn |
| Batch size | 4 | 4 | 1× (memory-limited; can't grow batch on Heavy) |
| Grad accum | 4 | 4–8 | 1–2× steps to achieve same effective batch |
| Trajectory length | 4 | 6–8 | 1.5–2× frames per step |
| Canvas | ~2K Gaussians | ~15K Gaussians | ~7× tokens cross-attended |
| Channels per attn head | smaller | full HAT-L width | additional FLOPs |

Best-case Heavy/Pico per-step cost ratio: **~25×** (5.7 × 4 × 0.7 partial-amortization × 1.5). Worst-case: **~80×** if cross-attention dominates and the canvas hits 15K tokens. Use **40–50× per-step slowdown vs Pico-002 on the same GPU** as the planning anchor.

H100 vs 3080 Ti for this workload: FP16 / FP8 throughput on H100 80GB is ~3–4× the 3080 Ti for transformer SR-class models, but Heavy will not fit in a single H100 at patch 256² with trajectory 6 — distributed training (DDP or ZeRO) introduces 1.2–1.5× overhead. Plan **H100 step time = (3080 Ti step time × 40–50) / 3 = 60–75 s/step on H100 single-GPU equivalent**.

### Why H100 is the anchor (and what changes on B200)

Every estimate in this memo is quoted in **H100-hours** because H100 is the GPU with mature, stable, multi-cloud pricing as of 2026-05 — every grant program, research credit, and cloud provider lists it. **B200 (Blackwell) is the better physical choice if available**, and is the realistic deployment target for any sponsor in a position to allocate Blackwell capacity. Trade-offs:

| Metric | H100 80GB SXM | B200 192GB SXM |
| --- | --- | --- |
| Memory | 80 GB HBM3 | **192 GB HBM3e** (2.4×) |
| Memory bandwidth | 3.35 TB/s | **8 TB/s** (2.4×) |
| BF16 dense throughput | 1,979 TFLOPS | **~4,500 TFLOPS** (2.3×) |
| FP8 dense throughput | 3,958 TFLOPS | **~9,000 TFLOPS** (2.3×) |
| Cloud on-demand (May 2026) | ~$3–4/hr | ~$5–7/hr |
| Cloud spot / capacity blocks | ~$1.50–2.50/hr | ~$3–5/hr (pilot programs) |
| Multi-cloud availability | every major cloud | AWS Capacity Blocks, Lambda 1-click, CoreWeave, GCP A4 |

For the OSS Heavy training workload specifically, B200 vs H100:

- **Per-step ~2.0–2.2× faster.** Heavy is not pure dense matmul (conv + canvas warp + cross-attention cap the speedup below the 2.4× theoretical), so realized speedup is below peak.
- **192 GB lets Heavy run at patch 256² + trajectory 8 + batch 8 on a single B200**, eliminating the DDP overhead that 8× H100 incurs because Heavy doesn't fit per-H100 without ZeRO/sharding.
- **Effective $/result on 8× B200 is ~0.85–0.95× the 8× H100 cost** because the per-step speedup outpaces the price premium, **as long as the cluster stays saturated**. B200 idle time costs more in absolute dollars.
- **Wall clock**: one Heavy cycle on 8× B200 lands in ~3 weeks vs ~5–6 weeks on 8× H100.

**Bottom line:** if a sponsor is in a position to allocate 8× B200 capacity (AWS Capacity Blocks for ML, CoreWeave reserved Blackwell, Lambda Reserved Cloud, NVIDIA Inception Blackwell tier), take it — the wall-clock compression is the most valuable thing the cycle can buy. Convert this memo's H100-hour estimates to B200-hours by dividing by ~2.0 for budget purposes.

## 3. Single Heavy training run (300K steps, 1 config)

[`RESEARCH.md` §5](../../RESEARCH.md): "Teacher target: 300K steps."

- Per-step: 60–75 s on H100 single-GPU equivalent.
- 300K steps × 67.5 s/step = 20.25M s = **5,625 H100-hours single-GPU equivalent**.
- Wall clock on 8× H100 DDP at 80% scaling: 5,625 / (8 × 0.8) = **879 hours = 36.6 days**.

This is a single config, no ablations, no cross-engine fine-tunes. **Real planning lower bound: 1,500 H100-hours** (assume warm restart from Standard checkpoint shortens to 200K Heavy-specific steps, plus better tensor-core utilization than 80%). **Realistic upper bound: 3,000 H100-hours** if the first run is from scratch and includes a Net2Net expansion phase.

**Cost: $2.3K – $4.5K spot, $4.5K – $9K on-demand.**

## 4. Heavy training cycle (one credible publication)

To stand behind "v6.x Heavy beats DLSS-class on game-engine footage" requires more than one well-tuned run:

| Activity | H100-hours | Notes |
| --- | --- | --- |
| Heavy teacher training (1 main run) | 1.5K – 3K | per §3 |
| Ablations (5–10) at 100K steps each | 3K – 6K | concat vs cross-attention, R=4 vs R=16, disocclusion-only spawner vs random, etc. |
| Hyperparameter sweep (LR, weight decay, λ_gan, λ_lpips) | 2K – 4K | even Bayesian search needs ~30–50 trial points |
| Standard tier distillation (~5M params, 80K steps × 8-GPU) | 0.5K – 1K | trained against the Heavy teacher |
| Pico tier distillation (~0.4M nano-CNN, 80K steps × 8-GPU) | 0.3K – 0.6K | the actual *shipping* student |
| Cross-engine fine-tune cycles (3–5 datasets, 50K steps each) | 1K – 2K | UE5 City Sample, Unity HDRP, Source 2 captures, etc. |
| Held-out eval, viz, debug burn | 1K – 2K | the "I forgot to commit, restart" tax |
| **Subtotal** | **9.3K – 18.6K H100-hours** | |
| Slack 20% | 1.9K – 3.7K | |
| **Cycle total** | **11.2K – 22.3K H100-hours** | |

**Cost: $17K – $33K spot, $34K – $67K on-demand.**

This funds **one credible v6.x Heavy result on one architecture iteration**. If v6.2 architecture changes — e.g., if the R=16 latent rasterizer turns out to need rework, or fusion needs to be re-ablated — the cycle repeats.

## 5. Multi-cycle path to DLSS-class quality

DLSS reached today's quality through ~6 years of iteration (DLSS 1 → 4), backed by NVIDIA's internal compute (estimated tens to hundreds of thousands of GPU-hours across that span), proprietary game-capture pipelines from dozens of studios, and a dedicated team of researchers and engineers.

OSS reaching DLSS-class quality is a **multi-cycle** problem, not a single training run.

What "DLSS-class quality cross-engine" actually means:

- ≥30 dB PSNR / ≤0.15 LPIPS on held-out batches from 3+ game engines (UE5, Unity HDRP, Source 2, or equivalent) at Quality preset (1.5× upscale).
- ≤2 ms inference latency on RTX 4070-class GPU at 1080p → 4K (achieved by the **student**, not the Heavy teacher).
- No structural artifacts on disocclusion, transparency, fast camera motion, fine geometry (foliage, hair), or HDR specular highlights.

Plan for the multi-cycle path:

| Phase | H100-hours | Months | Goal |
| --- | --- | --- | --- |
| **Cycle 1** — v6.x Heavy on TartanAir + synthetic UE5 | 12K – 22K | 3–4 | Architecture validated at Heavy scale; first cross-engine generalization measurement |
| **Cycle 2** — Real game captures via OSS Capture Tool (Cyberpunk 2077, Alan Wake 2) | 15K – 30K | 4–6 | First captures from real shipping titles; data-scaling laws measured |
| **Cycle 3** — Multi-engine + HDR retrain | 15K – 30K | 4–6 | UE5 / Unity HDRP / Source 2 all in training mix; HDR-encoded loss path validated |
| **Cycle 4** — Quality polish + student-distillation refinements + DLSS-shim integration | 10K – 70K | 6–10 | The "polish" cycle: many small experiments to close the last few dB and the last few LPIPS points; per-vendor kernel work runs in parallel; the *shipping* student gets its real training; capture-tool feedback loop tunes the data mix |
| **Total** | **50K – 150K** | **18–24 months** | DLSS-class quality, cross-engine, with shipped students per tier |

**Cost: $75K – $225K spot, $150K – $450K on-demand.**

This estimate is honest but uncertain at the upper end. If the architecture has fundamental capacity limits at 17M params, Cycle 4 expands. If a data-scaling law turns out to be steeper than assumed (DLSS-class requires 5× the planned capture volume), the storage + capture costs dominate. Conversely, if FSR-4 or another OSS competitor open-sources useful checkpoints during the project, Cycle 3 shrinks.

## 6. Adjacent costs (not in the GPU table)

- **Storage**: Heavy training corpus is targeted at 50–200 TB (TartanAir 550 GB is a single dataset; real game captures multiply this by 50–200×). Cloudflare R2 at $15/TB/month: **$750 – $3,000/month** for 18–24 months = **$13K – $72K total**.
- **Data acquisition**: The OSS Capture Tool (under active implementation) ingests data from contributors at zero per-frame cost beyond storage. Capturing 100 hours of paired LR/HR/motion/depth from 5–10 supported titles via the contributor network is the lever; if contributor capture is insufficient, paid capture (someone playing the games on a metered rig) is **$50–150/hour of capture × ~500 hours = $25K – $75K**.
- **Engineering**: solo maintainer + AI-assisted development is the current state. Cost: $0 marginal beyond living expenses. If the project funded one full-time research engineer + one full-time kernel engineer to run the cycles above: **$300K – $500K/year fully loaded** in US salary + benefits markets.
- **Inference benchmarking**: a multi-vendor inference rig (one RTX 4070-class card, one RX 7900-class, one Arc B-series, one M-series Mac, one Snapdragon dev kit) for the cross-vendor kernel work: **$5K – $10K one-time**.

## 7. Honest caveats

- **Pico-002 step time anchor is on a 3080 Ti running shared with other GPU consumers earlier in this run.** The clean-idle bench (H006 → H007) showed contamination can roughly double effective step time. Real Heavy step time on H100 could be **30–50% better than estimated** if FP16 tensor cores saturate and we get torch.compile + flash attention working (currently blocked by missing Triton on Windows).
- **Heavy memory footprint at patch 256² is unmeasured.** If Heavy doesn't fit in 80 GB H100 even with sharding, the per-step number rises. The [v6.2 VRAM footprint probe](../research/2026-05-08-v62-vram-footprint.md) is the right place to extend with a Heavy-tier dry-run *before* committing to a cloud bill.
- **Cross-engine generalization scaling is the riskiest number.** v5-pixel-temporal got 25.7 dB on TartanAir alone. Closing to DLSS-class (30+ dB) on cross-engine footage is the gating uncertainty; if the architecture caps out below 30 dB, no amount of compute fixes it.
- **DLSS-class quality is a moving target.** DLSS 4 transformer (2025) is what we're aiming at, but DLSS 5 / FSR 5 / XeSS 3 (2026–2027) will land during the multi-cycle timeline. The estimate assumes "DLSS-class" remains within ~1 generation of today's quality bar.

## 8. Funding posture — why each tier of sponsorship matters

OSS is solo-maintained, AI-augmented, and pre-alpha. The compute and storage above are the gating constraint, not the engineering. The numbers below are what each tier of sponsorship actually unlocks — pitch this as the leverage point, not the line item.

### Cloud GPU sponsors (NVIDIA, AMD, Intel, Modal, Lambda, RunPod, Cloudflare, Vast.ai, Together, …)

- **$2K – $9K of credits** funds a single Heavy training run. That is one credibility checkpoint at Heavy scale — the difference between "this works at Pico" and "this works at Heavy too." Realistic budget for an early-stage corporate dev-rel program or a research grant pilot.
- **$17K – $67K of credits** funds a complete Heavy training cycle: teacher + ablations + hyperparameter sweep + Standard/Pico student distillation + cross-engine fine-tune. One full publishable v6.x Heavy result with student tiers ready to ship. Fits an NSF ACCESS / NAIRR allocation, an Oracle for Research credit, an HF community grant, or a strategic OSS allocation from Modal / Lambda / NVIDIA Inception.
- **$75K – $450K of credits** funds the multi-cycle path to DLSS-class cross-engine quality (18–24 months). This is the credit allocation a cloud provider or GPU vendor offers when they have a strategic interest in vendor-neutral SR (e.g. AMD or Intel wanting a DLSS alternative their GPUs run on natively; a cloud provider wanting an Apache-licensed reference workload to benchmark).

### Hardware-loaner sponsors (GPU vendors, OEMs)

A loaned **MI300X, B200, or H100 8-GPU node for 90 days** is roughly equivalent to **$30K – $70K of cloud spot credit** at sustained 80% utilization, and is what one Heavy training cycle physically runs on. A loaned **Intel Battlemage Arc, Tenstorrent Wormhole/Blackhole, M4 Max Mac Studio, or Snapdragon dev kit** unlocks the cross-vendor kernel sprint that turns "the model trains" into "the model ships on AMD / Intel / Apple / handheld hardware natively."

### Studios + game engines

A studio that has shipped DLSS / FSR / XeSS-integrated titles has the leverage to **co-fund the OSS Capture Tool pipeline against their own title's footage**, in exchange for a vendor-neutral DLL-shim integration tuned for their engine. The marginal cost vs. their existing per-title vendor-stack tuning is small; the payoff is one upscaler that ships on every GPU their players own.

### Job, contract, or full-time role

Any of the three above can also be packaged as a **headcount-funded research-engineer or kernel-engineer role**. Cost of one senior US AI-lab engineer-year fully loaded ($300K – $500K including benefits, equity, on-costs) is in the same band as the multi-cycle DLSS-class compute estimate. The difference is whether the sponsor wants compute → result, or engineer → result + IP-aligned advisor.

### What you get in return

OSS is **Apache-2.0** licensed. Trained model weights are planned for **CC-BY-4.0** when they ship. Cross-vendor SR is the throughline: any sponsor underwriting this work funds a public-good upscaler that **runs everywhere**, with no licensing footprint, no proprietary runtime dependency, and no per-title integration fee. That's the bet.

---

Estimates here are honest planning numbers, not promises. The next concrete step is a **Heavy VRAM/step-time dry-run on a single H100 instance (~$3 of cloud spend)** to anchor the H100 step time more tightly than the 5.7× / 3× extrapolation in §2 — exactly the kind of small allocation a pilot credit program covers.

Reach out: <cashcon57@gmail.com>.
