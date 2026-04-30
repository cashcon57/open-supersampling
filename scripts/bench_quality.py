from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from ors.bench import QualityRunner


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _pairs(root: Path) -> list[tuple[str, Path, Path]]:
    if (root / "lr").is_dir() and (root / "hr").is_dir():
        return [
            (p.stem, p, root / "hr" / p.name)
            for p in sorted((root / "lr").iterdir())
            if p.is_file() and (root / "hr" / p.name).exists()
        ]
    pairs = []
    for p in sorted(root.iterdir()):
        if p.is_file() and "_lr" in p.stem:
            q = p.with_name(p.name.replace("_lr", "_hr", 1))
            if q.exists():
                pairs.append((p.stem.replace("_lr", "", 1), p, q))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runner = QualityRunner(ckpt_path=args.ckpt)
    rows = []
    for scene, lr_path, hr_path in _pairs(args.input_dir):
        result = runner.run_methods(_read_rgb(lr_path), _read_rgb(hr_path))
        for method, stats in result.items():
            rows.append(
                {
                    "scene": scene,
                    "method": method,
                    "psnr": stats["psnr"],
                    "ssim": stats["ssim"],
                    "lpips": stats["lpips"],
                    "ms_per_frame": stats["ms_per_frame"],
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "method", "psnr", "ssim", "lpips", "ms_per_frame"])
        writer.writeheader()
        writer.writerows(rows)

    print("method      psnr     ssim    lpips   ms/frame")
    for method in sorted({r["method"] for r in rows}):
        vals = [r for r in rows if r["method"] == method]
        mean = lambda k: sum(v[k] for v in vals) / max(1, len(vals))
        print(f"{method:10s} {mean('psnr'):7.3f} {mean('ssim'):7.4f} {mean('lpips'):7.4f} {mean('ms_per_frame'):9.3f}")


if __name__ == "__main__":
    main()
