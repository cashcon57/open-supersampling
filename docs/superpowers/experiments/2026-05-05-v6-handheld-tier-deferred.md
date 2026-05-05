# 2026-05-05 — v6 handheld tier: deferred to post-launch

**Status:** Deferred. Will revisit when one of the trigger conditions in this doc is met.

## What we're deferring

A third student tier targeting Steam Deck and similar handheld / integrated-GPU hardware:

| Constraint | Detail |
|---|---|
| Target hardware | Steam Deck (RDNA2 8 CUs), AMD APUs, low-end integrated GPUs |
| Inference budget | <3 ms at 720p→1080p (had to be tighter because no tensor cores) |
| Backbone | RRDB-mini (~500K params) or similar tiny CNN |
| Runtime | NCNN + Vulkan compute |

## Why we're deferring

### 1. Bad launch comparison

Steam Deck already has FSR 2/3 baked into SteamOS. Our best-case at 500K params is "matches FSR 2 with extra latency." That's not a release narrative. Beating FSR 2 hand-tuned shaders with a 500K-param ML model is genuinely hard.

### 2. Worst engineering ROI

| Cost | Value |
|---|---|
| ~12 h GPU on 3080 Ti for the distillation run | smallest user slice |
| ~1-2 weeks engineering (NCNN export + Vulkan compute path + RDNA2 driver quirks + Deck-specific perf tuning) | "matches FSR" outcome |

Every other tier amortizes the same training data + loss recipe. Handheld tier amortizes the same training but adds an entirely separate runtime stack.

### 3. No tensor core path

Every tier above handheld benefits from FP16/FP8 acceleration. Handheld can't (RDNA2 has no ML accelerator), which forces design compromises that don't help any other tier. Handheld is its own self-contained engineering problem.

### 4. FSR 4 won't reach Deck (RDNA4 only)

The "ML-powered SR for Deck" gap is real and won't be filled by AMD any time soon. So the handheld market opportunity exists. But it's a *post-launch* opportunity, not a *launch* opportunity.

## When to revisit

Revisit when **any one** of these conditions is met:

1. **Community demand** — OSS gets traction and "does it run on my Deck" becomes a frequently-asked question on issues / Discord / Reddit.
2. **Free GPU cycles** — Desktop + NVIDIA tiers have shipped, there's no other priority queued for the 3080 Ti, and we want to extend hardware reach.
3. **Volunteer engineering** — Someone contributes the NCNN export path or RDNA2 perf tuning. That's not Cash + Claude work; it's the kind of thing OSS being open invites.
4. **New handheld market lands** — PS handheld, Switch 2, ROG Ally X variants ship and the addressable market visibly expands beyond Steam Deck.

## What we keep working when we come back

- Teacher checkpoint (HAT-Base or whatever is current at the time)
- Distillation infrastructure
- Loss recipe (Charbonnier + LPIPS + GAN + wavelet)
- Training data pipeline
- Held-out eval set

What changes:
- Add a third student backbone (RRDB-mini, ~500K params, no attention, NCNN-friendly ops only)
- Add NCNN export path
- Add Vulkan compute runtime integration in the OSS plugin
- Add Steam Deck perf tuning (RDNA2-specific, possibly per-driver-version)

## Honest framing for marketing post-launch

When we ship desktop + NVIDIA tiers, anyone asking "does it run on my Deck?" should get:

> Steam Deck support is on the roadmap as a dedicated handheld tier. The desktop model can run on Deck at lower quality and higher latency, but the proper handheld build is a follow-up. We'd rather ship a real handheld optimization than a downsized desktop model.

This is honest (matches reality), invites contribution (OSS being open), and doesn't promise a date.
