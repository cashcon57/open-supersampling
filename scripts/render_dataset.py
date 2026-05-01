"""Render a dataset of paired image triplets via render_pair."""
from __future__ import annotations
import argparse
from pathlib import Path
from tqdm import tqdm
from oss.render import render_pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="bistro")
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--spp-noisy", type=int, default=1)
    p.add_argument("--spp-gt", type=int, default=4096)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for v in tqdm(range(args.views), desc=f"rendering {args.scene}"):
        render_pair(scene_name=args.scene, view_index=v,
                    spp_noisy=args.spp_noisy, spp_gt=args.spp_gt,
                    resolution=(args.width, args.height), out_dir=args.out)


if __name__ == "__main__":
    main()
