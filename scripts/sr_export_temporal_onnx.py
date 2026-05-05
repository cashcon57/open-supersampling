#!/usr/bin/env python
"""Export a v5 pixel-temporal checkpoint through the stateless ONNX wrapper.

The export graph has explicit temporal state inputs:

    lr_inputs, prev_hr_input, depth_hr_curr, depth_hr_prev, motion_lr

and returns:

    out_hr, disocclusion_mask

Heavy ML imports are deferred until after argparse so ``--help`` works on a
plain Python interpreter.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True, help="v5 temporal checkpoint")
    p.add_argument("--output", type=Path, required=True, help="Output ONNX path")
    p.add_argument("--lr-h", type=int, required=True, help="LR export height")
    p.add_argument("--lr-w", type=int, required=True, help="LR export width")
    p.add_argument("--opset", type=int, default=18, help="ONNX opset (default: 18)")
    p.add_argument("--device", default="cpu", help="Export device (default: cpu)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.lr_h <= 0 or args.lr_w <= 0:
        print(f"FAIL: --lr-h/--lr-w must be positive; got {args.lr_h}x{args.lr_w}")
        return 1

    import torch
    from oss.sr.temporal.stateless_export import TemporalSRModelStateless

    model = TemporalSRModelStateless.from_temporal_checkpoint(
        args.ckpt, device=args.device
    )
    model.train(False)

    scale = model.scale
    h_lr, w_lr = int(args.lr_h), int(args.lr_w)
    h_hr, w_hr = h_lr * scale, w_lr * scale
    x = torch.zeros(1, model.in_channels, h_lr, w_lr, device=args.device)
    prev_hr = torch.zeros(1, 3, h_hr, w_hr, device=args.device)
    depth_curr = torch.zeros(1, 1, h_hr, w_hr, device=args.device)
    depth_prev = torch.zeros(1, 1, h_hr, w_hr, device=args.device)
    motion = torch.zeros(1, 2, h_lr, w_lr, device=args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (x, prev_hr, depth_curr, depth_prev, motion),
            args.output,
            opset_version=int(args.opset),
            input_names=[
                "lr_inputs",
                "prev_hr_input",
                "depth_hr_curr",
                "depth_hr_prev",
                "motion_lr",
            ],
            output_names=["out_hr", "disocclusion_mask"],
            dynamic_axes={
                "lr_inputs": {2: "H_lr", 3: "W_lr"},
                "motion_lr": {2: "H_lr", 3: "W_lr"},
                "prev_hr_input": {2: "H_hr", 3: "W_hr"},
                "depth_hr_curr": {2: "H_hr", 3: "W_hr"},
                "depth_hr_prev": {2: "H_hr", 3: "W_hr"},
                "out_hr": {2: "H_hr", 3: "W_hr"},
                "disocclusion_mask": {2: "H_hr", 3: "W_hr"},
            },
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
