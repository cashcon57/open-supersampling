# 2026-05-08 — H002 Test A1: v6.1-pico-001 SVD rank analysis

## Purpose

Run H002 Test A1 on the existing `srcnn-v6.1-pico-001` checkpoint:

> SVD of feat tensor across a batch; report top-R singular value energy. If
> top-8 energy / total > 0.95, R=8 is feasible.

The tensor measured here is the v6.1 per-Gaussian `colors`/feature payload
emitted by `V6Model.gaussian_spawner`, shape `(B, K, 64)`, because that is the
64-channel per-Gaussian payload H002 proposes to replace with a low-rank
latent.

## Asset discovery

Checked paths:

| Path | Result |
|---|---|
| `runs/` | Missing in repo |
| `/e/checkpoints/` | Missing |
| `/E/checkpoints/` | Missing |
| `/mnt/e/checkpoints/` | Missing |
| `/Volumes/` | Exists, no relevant checkpoint found during targeted check |
| `/Users/cashconway/checkpoints/` | Missing |
| `/Users/cashconway/runs/` | Missing |
| `/Users/cashconway/mnt/gpu-e/checkpoints/srcnn-v6.1-pico-001/` | Found real v6.1 run/checkpoints |
| `/Users/cashconway/mnt/gpu-e/datasets/tartanair_extracted/` | Found real TartanAir data |
| `dashboard-public/runs/srcnn-v6.1-pico-001/` | Found dashboard metrics/viz only |

Checkpoint used:

`/Users/cashconway/mnt/gpu-e/checkpoints/srcnn-v6.1-pico-001/step-00014000.pt`

Data used:

`/Users/cashconway/mnt/gpu-e/datasets/tartanair_extracted/oldtown/Easy/P000`

Frames:

- `image_left/000000_left.png`
- `image_left/000001_left.png`
- `image_left/000002_left.png`

## Method

- Loaded checkpoint with `venv-py312/bin/python` and `torch 2.11.0`.
- Reconstructed `V6Model(V6Config(...))` from checkpoint args:
  - `backbone=hat-tiny`
  - `in_channels=9`
  - `scale=2`
  - `color_activation=hdr`
  - `spawn_offset_random=True`
  - `spawn_subpixel_jitter=False`
  - `rasterizer_overlap=8`
- Used training-matched LR synthesis:
  - `scale=2.0`
  - Halton jitter enabled
  - TAA blur enabled
  - JPEG disabled
  - `blur_sigma=0.5`
- Used a deterministic centered `128x128` LR crop, aligned to 8 px tiles. This
  matches the checkpoint's `patch_size=128` and yields 256 Gaussian feature
  vectors per frame.
- Captured `GaussianSpawner.forward(...).colors` by forward hook.
- Concatenated 3 frames: `768 x 64` feature matrix.
- Reported two energies:
  - `raw`: SVD of the raw feature matrix, matching the literal H002 wording.
  - `centered`: SVD after subtracting the feature mean, the stricter PCA-style
    intrinsic-rank test. A downstream decoder bias can absorb the mean, so this
    is the more informative compression-risk number.

## Commands run

```bash
rg -n "Test A1|\bA1\b|H002|low-rank|SVD|singular|energy" docs -g '*.md'
find /Users/cashconway/mnt/gpu-e/checkpoints/srcnn-v6.1-pico-001 -maxdepth 2 -type f -print
find /Users/cashconway/mnt/gpu-e/checkpoints/srcnn-v6.1-pico-001 -maxdepth 2 -type f -exec ls -lh {} +
venv-py312/bin/python - <<'PY'
# Loaded step-00014000.pt, replayed oldtown/Easy/P000 frames 0..2,
# hooked model.gaussian_spawner, and ran torch.linalg.svdvals on the
# captured 64-channel per-Gaussian features.
PY
```

## Feature capture sanity

| Frame | Feature shape | Output shape | Feature mean | Feature std |
|---:|---:|---:|---:|---:|
| 0 | `256 x 64` | `1 x 3 x 256 x 256` | 0.04093037 | 0.13129702 |
| 1 | `256 x 64` | `1 x 3 x 256 x 256` | 0.03514386 | 0.08643561 |
| 2 | `256 x 64` | `1 x 3 x 256 x 256` | 0.03453538 | 0.08667650 |

## Energy table

Aggregate over all three frames, `768 x 64`:

| Rank R | Raw energy | Centered energy |
|---:|---:|---:|
| 1 | 0.850325 | 0.646732 |
| 2 | 0.939307 | 0.824281 |
| 4 | 0.975188 | 0.887217 |
| 8 | 0.988945 | 0.947917 |
| 16 | 0.998400 | 0.991836 |
| 32 | 0.999949 | 0.999756 |
| 64 | 1.000000 | 1.000000 |

Per-frame top-8 energy:

| Frame | Raw top-8 | Centered top-8 |
|---:|---:|---:|
| 0 | 0.994153 | 0.926473 |
| 1 | 0.986871 | 0.897115 |
| 2 | 0.986679 | 0.894715 |

Top 16 aggregate singular values:

| Index | Raw | Centered |
|---:|---:|---:|
| 1 | 22.495668 | 8.595128 |
| 2 | 7.277108 | 4.503497 |
| 3 | 4.186920 | 2.082747 |
| 4 | 1.955409 | 1.688573 |
| 5 | 1.688052 | 1.483165 |
| 6 | 1.436548 | 1.396786 |
| 7 | 1.356556 | 1.198657 |
| 8 | 1.197292 | 1.160247 |
| 9 | 1.159673 | 1.017485 |
| 10 | 0.976025 | 0.972300 |
| 11 | 0.967425 | 0.934562 |
| 12 | 0.880100 | 0.818799 |
| 13 | 0.758252 | 0.713952 |
| 14 | 0.694945 | 0.679907 |
| 15 | 0.560292 | 0.526386 |
| 16 | 0.497338 | 0.493328 |

## Verdict

H002 A1 does not clear the requested centered-SVD gate. The acceptance test is
`torch.linalg.svd(feat - feat.mean(dim=0, keepdim=True))`, and the aggregate
centered top-8 energy is `0.947917`, below the `0.95` threshold. Per-frame
centered top-8 energy is lower still, `0.894715` to `0.926473`.

Verdict: `R=8 captures <95% energy -> retarget pico-002 at R=16`.

Blocking issue: https://github.com/cashcon57/open-supersampling/issues/11

The raw, uncentered matrix reaches `0.988945` at R=8, but that pass is carried
by a dominant mean/common feature direction. That is useful evidence that a
bias or explicit mean restoration matters, but it is not the criterion specified
for this gate.

Recommendation:

- Block R=8 as the default for `v6.2-pico-002` until an R=8 retraining ablation
  proves it in RGB PSNR/LPIPS.
- Use R=16 as the conservative pico-002 target; centered top-16 energy is
  `0.991836`, while still cutting raster payload by 4x versus 64 channels.
- If R=8 remains in the ablation, use a decoder/projection with bias or explicit
  mean restoration; a pure no-bias `B z` factorization is not what the centered
  data supports.
