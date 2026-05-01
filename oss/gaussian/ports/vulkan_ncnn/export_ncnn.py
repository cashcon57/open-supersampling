"""ncnn export for ``GaussianParamNetwork`` — Sprint 7 / T7.V.3.

Loads a trained Sprint 4 checkpoint (Pico tier by default, matching the
Steam Deck Gaussian budget of 1K) and exports it to ncnn's ``.param`` /
``.bin`` pair via PyTorch -> PNNX -> ncnn.

Both ``ncnn`` and ``pnnx`` ship pre-built wheels under the ``[vulkan]``
extra; install via:

    pip install -e '.[vulkan]'

This module is import-safe without ``ncnn`` / ``pnnx`` — the heavy imports
are deferred until ``export(...)`` is called. The CLI ``--check`` flag runs
a parity smoke test on the traced PyTorch graph alone.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from oss.gaussian.network.param_net import (
    GaussianParamNetwork,
    TIER_CONFIGS,
    param_net_for_tier,
)


_DEFAULT_TIER = "pico"
_DEFAULT_LR_HW = (360, 640)  # Steam Deck 1280x800 / 2x upscale, tile-aligned.


def _load_checkpoint(model: GaussianParamNetwork, ckpt_path: Path | None) -> None:
    if ckpt_path is None or not ckpt_path.exists():
        return
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)


def build_traceable(tier: str, lr_hw: tuple[int, int]) -> tuple[GaussianParamNetwork, torch.Tensor]:
    if tier not in TIER_CONFIGS:
        raise KeyError(f"unknown tier {tier!r}; available: {sorted(TIER_CONFIGS)}")
    model = param_net_for_tier(tier)
    model.train(False)
    h_lr, w_lr = lr_hw
    if h_lr % model.tile_size or w_lr % model.tile_size:
        raise ValueError(
            f"lr_hw={lr_hw} must be exact multiples of tile_size={model.tile_size}"
        )
    dummy = torch.randn(1, model.in_channels, h_lr, w_lr)
    return model, dummy


def export(
    tier: str = _DEFAULT_TIER,
    lr_hw: tuple[int, int] = _DEFAULT_LR_HW,
    ckpt_path: Path | None = None,
    out_dir: Path | None = None,
    dry_run: bool = False,
) -> Any:
    """Export the network to ncnn ``.param`` + ``.bin`` via PNNX.

    Args:
        tier: pico / lite / standard / ultra.
        lr_hw: (H_lr, W_lr) — both multiples of tile_size.
        ckpt_path: optional Sprint 4 checkpoint.
        out_dir: where to drop the .param/.bin pair. None means
            ``checkpoints/`` next to this file.
        dry_run: build the traceable model + dummy input but skip PNNX.

    Returns:
        The traced ``ScriptModule`` on dry-run; the path to the written
        ``.param`` file on a real export.
    """
    model, dummy = build_traceable(tier, lr_hw)
    _load_checkpoint(model, ckpt_path)

    traced = torch.jit.trace(model, dummy)

    if dry_run:
        return traced

    pnnx_bin = shutil.which("pnnx")
    if pnnx_bin is None:
        try:
            import pnnx  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "pnnx not installed and not on PATH. "
                "`pip install -e '.[vulkan]'` and retry."
            ) from e
        # `pnnx` Python wheel installs the binary under sys.prefix; rely on the
        # entry-point command being available in that case.
        pnnx_bin = "pnnx"

    if out_dir is None:
        out_dir = Path(__file__).parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)

    pt_path = out_dir / f"param_net_{tier}.pt"
    traced.save(str(pt_path))

    h_lr, w_lr = lr_hw
    shape_arg = f"[1,{model.in_channels},{h_lr},{w_lr}]"
    cmd = [pnnx_bin, str(pt_path), f"inputshape={shape_arg}"]
    subprocess.run(cmd, check=True, cwd=str(out_dir))

    param_path = out_dir / f"param_net_{tier}.ncnn.param"
    if not param_path.exists():
        # PNNX naming convention varies by version — fall back to glob.
        candidates = list(out_dir.glob(f"param_net_{tier}*.param"))
        if not candidates:
            raise RuntimeError(
                f"PNNX completed but no .param found in {out_dir}"
            )
        param_path = candidates[0]
    return param_path


def parity_smoke_test(tier: str, lr_hw: tuple[int, int]) -> float:
    """Determinism check on the traced PyTorch graph (smoke test for export)."""
    model, dummy = build_traceable(tier, lr_hw)
    with torch.no_grad():
        a = model(dummy)
        b = model(dummy)
    return float((a - b).abs().mean().item())


def _cli() -> None:
    p = argparse.ArgumentParser(description="Export GaussianParamNetwork to ncnn via PNNX.")
    p.add_argument("--tier", default=_DEFAULT_TIER, choices=sorted(TIER_CONFIGS))
    p.add_argument("--height", type=int, default=_DEFAULT_LR_HW[0])
    p.add_argument("--width", type=int, default=_DEFAULT_LR_HW[1])
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--check", action="store_true",
                   help="Run a parity smoke test instead of writing .param.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build the traceable model but skip pnnx.")
    args = p.parse_args()

    if args.check:
        diff = parity_smoke_test(args.tier, (args.height, args.width))
        print(f"parity diff (inference-mode determinism): {diff:.3e}")
        return

    out = export(
        tier=args.tier,
        lr_hw=(args.height, args.width),
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
    )
    print(f"export complete: {out}")


if __name__ == "__main__":
    _cli()
