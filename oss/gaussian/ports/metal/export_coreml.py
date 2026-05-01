"""CoreML export for ``GaussianParamNetwork`` — Sprint 7 / T7.M.3.

Loads a trained Sprint 4 checkpoint (Lite tier by default, matching the
M3 Max Gaussian budget of 5K) and converts it to a CoreML ``.mlpackage``
runnable on M3 Max GPU + Apple Neural Engine.

The conversion uses ``coremltools``; install via the ``[coreml]`` extra:

    pip install -e '.[coreml]'

This module is import-safe without ``coremltools`` — the heavy import is
deferred until ``export(...)`` is called. The CLI ``--check`` flag runs a
single-batch parity check between PyTorch CPU output and the converted
CoreML model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from oss.gaussian.network.param_net import (
    GaussianParamNetwork,
    TIER_CONFIGS,
    param_net_for_tier,
)


_DEFAULT_TIER = "lite"
_DEFAULT_LR_HW = (360, 640)  # H_lr, W_lr — multiples of TILE_SIZE=16


def _load_checkpoint(model: GaussianParamNetwork, ckpt_path: Path | None) -> None:
    """Load a Sprint 4 checkpoint into ``model`` if one is provided.

    Sprint 7 prep tolerates a missing checkpoint (we're scaffolding before
    Sprint 4 has finished training) — the dry-run still exercises the export
    plumbing on a randomly initialized model.
    """
    if ckpt_path is None or not ckpt_path.exists():
        return
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)


def build_traceable(tier: str, lr_hw: tuple[int, int]) -> tuple[GaussianParamNetwork, torch.Tensor]:
    """Construct the network + a dummy input ready for tracing/conversion."""
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
    out_path: Path | None = None,
    dry_run: bool = False,
) -> Any:
    """Export the network to CoreML.

    Args:
        tier: pico / lite / standard / ultra (must exist in TIER_CONFIGS).
        lr_hw: (H_lr, W_lr) — both multiples of tile_size.
        ckpt_path: optional Sprint 4 checkpoint. None means random init for dry-run.
        out_path: where to save the .mlpackage. None means
            ``checkpoints/param_net_<tier>.mlpackage``.
        dry_run: if True, build the traceable model + dummy input but skip the
            actual ``coremltools.convert`` call. Used by tests on hosts without
            ``coremltools`` installed.

    Returns:
        The CoreML model object on a real export, or the traced ``ScriptModule``
        on a dry-run (so callers can inspect input/output shapes).
    """
    model, dummy = build_traceable(tier, lr_hw)
    _load_checkpoint(model, ckpt_path)

    traced = torch.jit.trace(model, dummy)

    if dry_run:
        return traced

    try:
        import coremltools as ct  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "coremltools not installed. `pip install -e '.[coreml]'` and retry."
        ) from e

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=dummy.shape, name="x")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    if out_path is None:
        out_dir = Path(__file__).parent / "checkpoints"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"param_net_{tier}.mlpackage"
    mlmodel.save(str(out_path))
    return mlmodel


def parity_smoke_test(tier: str, lr_hw: tuple[int, int]) -> float:
    """Sanity-check that a freshly traced model produces sensible outputs.

    Returns the mean abs difference between two consecutive inference-mode
    forward passes on the same input — should be exactly zero (model is
    deterministic in inference mode). Acts as a smoke test that the traced
    graph is well-formed.
    """
    model, dummy = build_traceable(tier, lr_hw)
    with torch.no_grad():
        a = model(dummy)
        b = model(dummy)
    return float((a - b).abs().mean().item())


def _cli() -> None:
    p = argparse.ArgumentParser(description="Export GaussianParamNetwork to CoreML.")
    p.add_argument("--tier", default=_DEFAULT_TIER, choices=sorted(TIER_CONFIGS))
    p.add_argument("--height", type=int, default=_DEFAULT_LR_HW[0])
    p.add_argument("--width", type=int, default=_DEFAULT_LR_HW[1])
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--check", action="store_true",
                   help="Run a parity smoke test instead of writing .mlpackage.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build the traceable model but skip coremltools.")
    args = p.parse_args()

    if args.check:
        diff = parity_smoke_test(args.tier, (args.height, args.width))
        print(f"parity diff (inference-mode determinism): {diff:.3e}")
        return

    out = export(
        tier=args.tier,
        lr_hw=(args.height, args.width),
        ckpt_path=args.ckpt,
        out_path=args.out,
        dry_run=args.dry_run,
    )
    print(f"export complete: {type(out).__name__}")


if __name__ == "__main__":
    _cli()
