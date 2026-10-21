# AMD FSR 4: licence status and repo policy

**Last reviewed:** 2026-10-21. This is not legal advice.

## What happened

On 2025-08-18 AMD pushed FidelityFX SDK 2.0.0 to GitHub (commit `01446e6a`). It included the FSR 4 ML upscaler: HLSL source, INT8 and FP8 model weights (`fsr4_model_v07_*`), and signed DLLs. The repository root carried an MIT licence file. A few days later AMD force-pushed the tag to a commit without the FSR 4 source and said the code had been "published in error".

From 2026-05 until 2026-10-21 this repo vendored that commit at `oss/third_party/fidelityfx-sdk-2.0.0-mit/` (345 files, including 12 weight files and AMD's signed DLLs). A separate public mirror, `cashcon57/fidelityfx-sdk-mit-archive`, held the same content.

## Why we removed it

The vendored copy rested on an "irrevocable MIT grant", and that claim is weaker than our docs said:

- 134 of the 183 FSR 4 files carry `Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.` headers. Only 5 carry MIT text. The file headers contradict the root licence.
- AMD has said publicly that the release was a mistake.
- Every SDK release from v2.1.0 (Dec 2025) through v2.3.0 (Jun 2026) is binary-only, with no reverse engineering allowed. Our old `MIT-PROVENANCE.md` wrongly said v2.1.0 was still MIT.
- No court has ruled on whether a licence grant made by mistake and then withdrawn can be relied on, or whether model weights are copyrightable.

The licence is uncertain, and hosting the files ourselves would make us a redistributor of AMD's code, weights and binaries. That combination is the main exposure, so the files are gone.

The removal applies to the current tree only. Git history still contains the files, so rewriting history is a separate decision (see "Open decisions").

## Policy from now on

| Use | Allowed? | Notes |
|---|---|---|
| Hosting, vendoring or redistributing AMD FSR 4 source, weights or DLLs | **No** | Includes test fixtures and mirrors. |
| Architecture ideas (FasterNet block, SqrSwish formula, FP16 NHWC + INT8 layout) | **Yes** | FasterNet is published (Chen et al., CVPR 2023). SqrSwish is a one-line formula, implemented independently in `oss/sr/v6/activations.py`. |
| FSR 4.1 as a **measured baseline**, run privately through OptiScaler (`Fsr4ForceEnableInt8`) on our own NVIDIA hardware | **Yes, privately** | Users install OptiScaler themselves; we only document how. Never commit the DLLs. |
| FSR 4 as a **distillation teacher** for weights we publish | **Gray** | Via OptiScaler, this relies on leaked or check-bypassed AMD binaries. If it proves useful, switch to AMD's official FSR 4.1 DLL on RDNA 3/4 hardware first. |

## Open decisions

- **Git history:** the removed files are still in history on GitHub. Purging them needs `git filter-repo` plus a force push, which rewrites every commit hash and breaks old links. That is the owner's call.
- **Mirror repo:** `cashcon57/fidelityfx-sdk-mit-archive` is being made private.

## Sources

- AMD's statement that the release was an error: <https://hothardware.com/news/amd-responds-accidental-fsr4-release>
- Release coverage: <https://www.tweaktown.com/news/107229/amd-accidentally-posted-fsr-4-source-code-as-part-of-its-fidelityfx-sdk-2-0-launch/index.html>
- Upstream SDK repo and licence history: <https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK>
- Arguments for and against the MIT grant holding: <https://www.profesionalreview.com/2025/08/23/amd-fsr-4-irreversible-licencia-mit/>, <https://www.igorslab.de/en/amd-fsr-4-leak-and-fidelityfx-sdk-2-0-a-sober-evaluation/>
- OptiScaler INT8 path: <https://github.com/optiscaler/OptiScaler/releases>
