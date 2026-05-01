# Vendored Image-GS

Source: https://github.com/NYU-ICL/image-gs
Paper: https://arxiv.org/abs/2407.01866 (SIGGRAPH 2025)
Pinned commit: `03088368d42684fb54225c981cfd94b58cc0393a` (heads/main as of 2026-05-01)
License: see `LICENSE.image_gs` (preserved from upstream)

## Contents
- `image_gs/` — full upstream repo as git submodule
- `image_gs/gsplat/` — Image-GS's bundled gsplat CUDA backend
- `image_gs/main.py`, `model.py` — their reference training/rendering scripts

## Usage
OSS-Gaussian's renderer wrapper at `oss/gaussian/renderer/rasterizer.py` calls into the vendored Image-GS rasterizer through `oss/gaussian/renderer/ext/` (CUDA extension).

## Updating
```bash
cd oss/gaussian/renderer/vendor/image_gs
git pull origin main
cd -
git add oss/gaussian/renderer/vendor/image_gs
git commit -m "vendor(image_gs): bump to <new-commit>"
```

After update, run Sprint 1 forward + backward tests to confirm no regression.
