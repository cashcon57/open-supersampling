# OSS Results Gallery

Side-by-side qualitative comparisons of OpenSuperSampling models on held-out batches. Each scene shows low-resolution input, bicubic upscale, the OSS model output, ground-truth high-resolution reference, and an absolute-error heatmap, alongside numeric PSNR / LPIPS metrics.

> **Status: SCAFFOLD.** This directory is currently a placeholder layout. Actual images are generated from real checkpoints on the remote 3080 Ti training host (see `~/.claude/projects/.../tailnet_3080ti.md`) and will be filled in here as v5 and v6 training completes. Do not commit ad-hoc image files to this gallery; only regenerate from the held-out evaluation script against a tracked checkpoint.

---

## Purpose

The gallery is the qualitative companion to the quantitative held-out evaluation memos under `docs/superpowers/experiments/*-held-out*.md`. PSNR and LPIPS numbers settle whether a model wins on average; the gallery is what you look at when you need to see *how* it wins or loses on specific scenes (e.g. high-frequency text, disocclusions, fast camera roll, transparent surfaces).

We treat the gallery as evidence in lab-notebook discipline: every result-driving conclusion in a memo should be reproducible from a checkpoint hash plus the held-out manifest plus this gallery.

---

## Layout

```
docs/results/
  README.md                              <-- this file
  v4-baseline/                           <-- frozen v4 SRCNN baseline (single-frame)
    <scene-id>/
      lr.png
      bicubic.png
      model.png
      gt.png
      error.png
      metrics.json
  v5-pixel-temporal/                     <-- v5 pixel-track temporal SR
    <scene-id>/
      ... (same files)
  v5-gaussian-temporal/                  <-- v5 Gaussian-track temporal SR
    <scene-id>/
      ... (same files)
  v6/                                    <-- v6 covariance-resampled online Gaussian-temporal SR
    <scene-id>/
      ... (same files)
```

Each top-level subdirectory corresponds to one architecture version. Inside each version, every held-out scene gets its own subdirectory.

### Scene-id convention

`<dataset-tag>-<trajectory-id>-frame-<NNNN>`

Examples:

- `oldtown-traj-P000-frame-0042` — TartanAir `oldtown` environment, trajectory `P000`, frame index `0042`.
- `sintel-ambush-3-frame-0017` — Sintel held-out manifest, scene `ambush_3`, frame `0017`.

The trajectory and frame index must come from a tracked held-out manifest (see *Held-out manifests* below). Never include a scene whose source frame is not pinned in a versioned manifest.

### Per-scene files

| File | Contents |
|---|---|
| `lr.png` | Low-resolution input fed to the model (bilinearly downsampled from GT, or natively rendered LR if from the OSS capture pipeline). |
| `bicubic.png` | Bicubic upscale of `lr.png` to GT resolution — the floor every model must beat. |
| `model.png` | The model's output at GT resolution. |
| `gt.png` | Ground-truth high-resolution frame. |
| `error.png` | Absolute-error heatmap, `|model - gt|`, normalized to a fixed scale (documented in `metrics.json`). |
| `metrics.json` | PSNR, LPIPS, SSIM for both `bicubic` and `model` against `gt`, plus the error-heatmap normalization scale and the source checkpoint hash. |

### `metrics.json` schema

```json
{
  "scene_id": "oldtown-traj-P000-frame-0042",
  "checkpoint": "<git-hash-or-content-hash>",
  "checkpoint_path": "checkpoints/srcnn-prod-v4-lpips/step-00385000.pt",
  "scale": 2,
  "metrics": {
    "bicubic": { "psnr": 0.0, "lpips": 0.0, "ssim": 0.0 },
    "model":   { "psnr": 0.0, "lpips": 0.0, "ssim": 0.0 }
  },
  "error_heatmap": {
    "norm_max": 0.25,
    "colormap": "magma"
  }
}
```

---

## Held-out manifests

Scenes are drawn from versioned, tracked manifests, never ad-hoc:

- TartanAir held-out trajectories: see the v5 design specs and the `docs/superpowers/experiments/2026-05-05-v5-pixel-temporal-held-out.md` memo for the canonical list.
- Sintel held-out manifest: `docs/superpowers/experiments/v5_held_out_manifest_sintel.json`.

Adding a new scene to the gallery means adding it to one of these manifests first; the gallery follows the manifest, not the other way around.

---

## How to (re)generate the gallery

Generation runs on the training host (`<train-host>`, see the tailnet memory note), against a specific checkpoint and a specific manifest. The script under `scripts/sr_temporal_held_out.py` (or the equivalent v6 inference script once it lands) is the canonical entry point — it produces all five PNGs and `metrics.json` for every scene in the manifest.

Indicative usage (refer to the script's own `--help` for the authoritative flag list):

```
python scripts/sr_temporal_held_out.py \
    --checkpoint <path-to-pt> \
    --manifest   docs/superpowers/experiments/v5_held_out_manifest_sintel.json \
    --out-dir    docs/results/<version-subdir>/ \
    --scale      2 \
    --emit-error-heatmap
```

Rules of thumb:

1. One checkpoint per `<version-subdir>`. If you re-run with a newer checkpoint, replace the contents of that subdirectory wholesale; do not mix outputs from two checkpoints under the same version.
2. Record the checkpoint hash (or content hash) in every `metrics.json` so the gallery cannot drift from its source.
3. Do **not** generate images locally on developer machines — only from the training host against tracked checkpoints. The gallery is evidence, not art direction.
4. Do not commit binary blobs unrelated to a tracked manifest.

---

## Current state (2026-05-05)

- `v4-baseline/` — placeholder, populated when v4 held-out gallery is regenerated against the production checkpoint.
- `v5-pixel-temporal/` — pending; v5-pixel-temporal held-out memo is in flight (see `docs/superpowers/experiments/2026-05-05-v5-pixel-temporal-held-out.md`).
- `v5-gaussian-temporal/` — pending; v5-gaussian-temporal training is in progress on the 3080 Ti host.
- `v6/` — pending; v6 implementation gated on v5 convergence and the canonical architecture lock-in (`docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md`).
