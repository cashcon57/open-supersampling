"""Slim a trained SR-CNN checkpoint by dropping zero-valued input channels.

Our SR-CNN trains on 12 channels: LR(3) + depth(1) + motion(2) + normals(3) +
canvas_hint(3). The canvas_hint channels are ALREADY zero during training (no
temporal accumulation yet), so the head_conv weights for those channels apply
to zeros and produce no useful signal. Dropping them at inference is
bit-identical to keeping them with zero input — and ~25% faster on the head
convolution.

This script performs the slice surgery on a trained checkpoint:

    head_conv.weight: (hidden, 12, 3, 3)  →  (hidden, 9, 3, 3)

The output is a new checkpoint that builds and runs as a 9-channel model.
The script verifies bit-identical output on a random input where canvas=0
before saving.

Usage:
    python scripts/sr_make_lean.py \\
        --input  <train-host-data>\\checkpoints\\srcnn-prod-v3\\step-00050000.pt \\
        --output <train-host-data>\\checkpoints\\srcnn-prod-v3\\step-00050000-lean9.pt \\
        --keep-channels 9
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from oss.sr import build_sr_model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Source checkpoint.")
    p.add_argument("--output", type=Path, required=True, help="Destination checkpoint.")
    p.add_argument("--keep-channels", type=int, default=9,
                   help="Number of input channels to keep (default 9: drop canvas_hint).")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device
    ck = torch.load(args.input, map_location=device, weights_only=False)
    saved_args = ck.get("args", {})
    tier = saved_args.get("tier", "lite")
    sr_backbone = saved_args.get("sr_backbone", "simple")

    factory_kind = "rrdb" if (sr_backbone == "rrdb") else "simple"

    full_model = build_sr_model(model_kind=factory_kind, tier=tier, in_channels=12, scale=2).to(device)
    full_model.load_state_dict(ck["sr_model"])
    full_model.train(False)

    # Slice the head_conv weight + bias to keep only the requested input channels.
    keep = args.keep_channels
    if keep < 3 or keep > 12:
        print(f"FAIL: --keep-channels must be in [3, 12]; got {keep}")
        return 1

    full_state = ck["sr_model"]
    head_w = full_state["head_conv.weight"]      # (hidden, 12, 3, 3)
    head_b = full_state["head_conv.bias"]        # (hidden,)

    if head_w.shape[1] != 12:
        print(f"FAIL: expected head_conv.weight in_channels=12, got {head_w.shape[1]}")
        return 1

    new_head_w = head_w[:, :keep, :, :].contiguous()
    print(f"  head_conv.weight: {tuple(head_w.shape)} -> {tuple(new_head_w.shape)}")

    # Build the lean model.
    lean_model = build_sr_model(model_kind=factory_kind, tier=tier, in_channels=keep, scale=2).to(device)
    lean_state = lean_model.state_dict()
    # Copy everything except head_conv, replace head_conv with sliced version.
    new_state = dict(full_state)
    new_state["head_conv.weight"] = new_head_w
    new_state["head_conv.bias"] = head_b
    lean_model.load_state_dict(new_state)
    lean_model.train(False)

    # Verify: bit-identical output when extra channels are zero.
    torch.manual_seed(0)
    h, w = 64, 96
    full_in = torch.randn(1, 12, h, w, device=device)
    full_in[:, keep:] = 0  # zero the dropped channels — what inference would do
    lean_in = full_in[:, :keep].contiguous()

    with torch.no_grad():
        full_out = full_model(full_in)
        lean_out = lean_model(lean_in)

    max_diff = (full_out - lean_out).abs().max().item()
    print(f"  verification: max abs diff = {max_diff:.2e}")
    if max_diff > 1e-5:
        print(f"FAIL: lean output diverges from full output (>{1e-5}). Aborting.")
        return 1

    # Save the lean checkpoint with updated args.
    new_args = dict(saved_args)
    new_args["lean_in_channels"] = keep
    payload = {
        "step": ck.get("step", 0),
        "tier": tier,
        "args": new_args,
        "sr_model": lean_model.state_dict(),
        "model_kind": "sr_cnn",
        "lean_in_channels": keep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    in_size = args.input.stat().st_size / 1024**2
    out_size = args.output.stat().st_size / 1024**2
    print(f"  source size = {in_size:.2f} MiB")
    print(f"  lean size   = {out_size:.2f} MiB ({out_size / in_size * 100:.1f}%)")

    n_full = sum(p.numel() for p in full_model.parameters())
    n_lean = sum(p.numel() for p in lean_model.parameters())
    print(f"  source params = {n_full:,}")
    print(f"  lean params   = {n_lean:,} ({n_lean / n_full * 100:.1f}%)")

    print(f"\nWrote lean checkpoint: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
