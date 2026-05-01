"""CLI for procedural Mitsuba dataset generation.

Usage:
    python scripts/generate_dataset.py \\
        --n-sequences 100000 \\
        --out-dir /data/oss-synthetic \\
        --workers 4 \\
        --resolution 512 512 \\
        --seq-len 8 \\
        --spp-noisy 1 \\
        --spp-gt 1024 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))


def _worker(
    idx: int,
    out_dir: str,
    resolution: tuple[int, int],
    seq_len: int,
    spp_noisy: int,
    spp_gt: int,
    seed: int,
    scene_type: str | None,
) -> int:
    """Render one sequence and write it. Returns bytes written."""
    import numpy as np
    from oss.data.mitsuba_gen.scene_builder import build_scene
    from oss.data.mitsuba_gen.render_worker import render_sequence
    from oss.data.mitsuba_gen.zarr_writer import write_sequence

    out_path = Path(out_dir) / f"scene{idx:04d}.zip"
    rng = np.random.default_rng(seed + idx)
    spec = build_scene(rng, scene_type=scene_type, seq_len=seq_len, resolution=resolution)
    buffers = render_sequence(spec, spp_noisy=spp_noisy, spp_gt=spp_gt, seed_base=seed + idx * 65536)
    write_sequence(buffers, out_path)
    return out_path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate procedural Mitsuba dataset")
    parser.add_argument("--n-sequences", type=int, default=100000)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--spp-noisy", type=int, default=1)
    parser.add_argument("--spp-gt", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scene-type",
        type=str,
        default="random",
        choices=["room", "corridor", "outdoor", "random"],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution = (args.resolution[0], args.resolution[1])
    scene_type = None if args.scene_type == "random" else args.scene_type

    indices = [
        i for i in range(args.n_sequences)
        if not (out_dir / f"scene{i:04d}.zip").exists()
    ]
    if not indices:
        print(f"All {args.n_sequences} sequences already exist in {out_dir}. Nothing to do.")
        return

    print(f"Generating {len(indices)} sequences "
          f"({args.n_sequences - len(indices)} already done) → {out_dir}")

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(indices), unit="seq", dynamic_ncols=True)
    except ImportError:
        progress = None

    t0 = time.monotonic()
    total_bytes = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _worker,
                idx,
                str(out_dir),
                resolution,
                args.seq_len,
                args.spp_noisy,
                args.spp_gt,
                args.seed,
                scene_type,
            ): idx
            for idx in indices
        }
        for fut in as_completed(futures):
            try:
                nbytes = fut.result()
                total_bytes += nbytes
            except Exception as exc:
                idx = futures[fut]
                print(f"\nsequence {idx} failed: {exc}", file=sys.stderr)
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    elapsed = time.monotonic() - t0
    gb = total_bytes / (1024 ** 3)
    print(f"\nDone. {len(indices)} sequences in {elapsed:.1f}s "
          f"({len(indices) / max(elapsed, 1e-6):.2f} seq/s), {gb:.3f} GB written.")


if __name__ == "__main__":
    main()
