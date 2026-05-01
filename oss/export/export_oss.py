"""Export any OSS model to ONNX (or CoreML .mlpackage).

Usage
-----
python -m oss.export.export_oss --model oss      --tier standard --output model.onnx
python -m oss.export.export_oss --model oss_rg   --tier heavy    --output denoiser.onnx
python -m oss.export.export_oss --model oss_pico              --output pico.onnx
python -m oss.export.export_oss --model oss_fx                --output fx.onnx
python -m oss.export.export_oss --model oss_pico --format coreml --output pico.mlpackage

All spatial axes are dynamic (batch, height, width). Opset 17.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-model export specs
# ---------------------------------------------------------------------------

def _export_oss(model, out_path: Path, tier: str) -> Path:
    """OSS: color(opt), depth, motion, aux(opt), features(opt)."""
    from oss.model.oss_rg import HANDOFF_FEATURE_CHANNELS

    B, H, W = 1, 64, 64
    depth   = torch.randn(B, 1, H, W)
    motion  = torch.randn(B, 2, H, W)

    mode = model.input_mode
    color_in    = torch.randn(B, 3, H, W) if mode in ("rgb", "rgb_aux") else torch.zeros(B, 3, H, W)
    aux_in      = torch.randn(B, 6, H, W) if mode == "rgb_aux" else torch.zeros(B, 6, H, W)
    features_in = torch.randn(B, HANDOFF_FEATURE_CHANNELS, H, W) if mode == "features" else torch.zeros(B, HANDOFF_FEATURE_CHANNELS, H, W)

    spatial = {0: "batch", 2: "height", 3: "width"}
    dynamic_axes = {
        "color":    spatial,
        "depth":    spatial,
        "motion":   spatial,
        "aux":      spatial,
        "features": spatial,
        "output":   {0: "batch", 2: "out_height", 3: "out_width"},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (color_in, depth, motion, aux_in, features_in),
        str(out_path),
        opset_version=17,
        input_names=["color", "depth", "motion", "aux", "features"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    return out_path


def _export_oss_rg(model, out_path: Path, tier: str) -> Path:
    """OSSRG: noisy, aux, history -> rgb, features.

    Forward: (noisy, aux, history) -> (rgb, features)
    noisy:   (B, 3, H, W)
    aux:     (B, 11, H, W)  albedo(3) + normal(3) + depth(1) + roughness(1) + spec_hit_dist(1) + motion(2)
    history: (B, 3, H, W)
    """
    B, H, W = 1, 64, 64
    noisy   = torch.randn(B, 3, H, W)
    aux     = torch.randn(B, 11, H, W)
    history = torch.randn(B, 3, H, W)

    spatial = {0: "batch", 2: "height", 3: "width"}
    dynamic_axes = {
        "noisy":    spatial,
        "aux":      spatial,
        "history":  spatial,
        "rgb":      spatial,
        "features": spatial,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (noisy, aux, history),
        str(out_path),
        opset_version=17,
        input_names=["noisy", "aux", "history"],
        output_names=["rgb", "features"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    return out_path


def _export_oss_pico(model, out_path: Path, tier: str) -> Path:
    """OSSPico: 7 inputs -> rgb_hr, new_hidden_state.

    Forward: (color_lr, depth_lr, motion_lr, normals_lr, albedo_lr,
               history_hr, hidden_state) -> (rgb_hr, new_hidden_state)

    hidden_state is None at sequence start; we trace the explicit-zeros path
    because ONNX has no concept of Python None — callers pass zeros for the
    first frame.
    """
    B, H_lr, W_lr = 1, 64, 64
    scale  = model.scale_factor
    H_hr   = int(H_lr * scale)
    W_hr   = int(W_lr * scale)

    color_lr    = torch.randn(B, 3, H_lr, W_lr)
    depth_lr    = torch.randn(B, 1, H_lr, W_lr)
    motion_lr   = torch.randn(B, 2, H_lr, W_lr)
    normals_lr  = torch.randn(B, 3, H_lr, W_lr)
    albedo_lr   = torch.randn(B, 3, H_lr, W_lr)
    history_hr  = torch.randn(B, 3, H_hr, W_hr)
    hidden_zero = torch.zeros(B, model.HIDDEN_CHANNELS, H_lr // 4, W_lr // 4)

    lr_axes  = {0: "batch", 2: "h_lr", 3: "w_lr"}
    hr_axes  = {0: "batch", 2: "h_hr", 3: "w_hr"}
    bot_axes = {0: "batch", 2: "h_lr_q", 3: "w_lr_q"}
    dynamic_axes = {
        "color_lr":         lr_axes,
        "depth_lr":         lr_axes,
        "motion_lr":        lr_axes,
        "normals_lr":       lr_axes,
        "albedo_lr":        lr_axes,
        "history_hr":       hr_axes,
        "hidden_state":     bot_axes,
        "rgb_hr":           hr_axes,
        "new_hidden_state": bot_axes,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden_zero),
        str(out_path),
        opset_version=17,
        input_names=[
            "color_lr", "depth_lr", "motion_lr", "normals_lr",
            "albedo_lr", "history_hr", "hidden_state",
        ],
        output_names=["rgb_hr", "new_hidden_state"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    return out_path


def _export_oss_fx(model, out_path: Path, tier: str) -> Path:
    """OSSFx: warped, depth, history, alpha -> frame, new_history.

    Forward: (warped, depth, history, alpha) -> (frame, new_history)
    warped:  (B, 3, H, W)
    depth:   (B, 1, H, W)
    history: (B, HISTORY_CH=32, H, W)
    alpha:   (B,)           temporal offset in (0, 1]; 1-D, batch-only dynamic.
    """
    from oss.model.oss_fx import HISTORY_CH
    B, H, W = 1, 64, 64

    warped  = torch.randn(B, 3, H, W)
    depth   = torch.randn(B, 1, H, W)
    history = torch.randn(B, HISTORY_CH, H, W)
    alpha   = torch.rand(B)

    spatial = {0: "batch", 2: "height", 3: "width"}
    dynamic_axes = {
        "warped":      spatial,
        "depth":       spatial,
        "history":     spatial,
        "alpha":       {0: "batch"},
        "frame":       spatial,
        "new_history": spatial,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (warped, depth, history, alpha),
        str(out_path),
        opset_version=17,
        input_names=["warped", "depth", "history", "alpha"],
        output_names=["frame", "new_history"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    return out_path


# ---------------------------------------------------------------------------
# CoreML conversion (called when --format coreml)
# ---------------------------------------------------------------------------

def _to_coreml(onnx_path: Path, out_path: Path) -> Path:
    """Convert an ONNX model to CoreML .mlpackage via coremltools."""
    import coremltools as ct

    mlmodel = ct.convert(
        str(onnx_path),
        source="onnx",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    log.info("CoreML model saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_MODEL_KEYS = ("oss", "oss_rg", "oss_pico", "oss_fx")
_EXPORTERS  = {
    "oss":      _export_oss,
    "oss_rg":   _export_oss_rg,
    "oss_pico": _export_oss_pico,
    "oss_fx":   _export_oss_fx,
}


def _build_model(model_key: str, tier: str):
    """Construct a fresh (randomly-initialised) model in inference mode."""
    if model_key == "oss":
        from oss.model.oss import OSS
        m = OSS(tier=tier)
    elif model_key == "oss_rg":
        from oss.model.oss_rg import OSSRG
        m = OSSRG(tier=tier)
    elif model_key == "oss_pico":
        from oss.model.oss_pico import OSSPico
        m = OSSPico()
    elif model_key == "oss_fx":
        from oss.model.oss_fx import OSSFx
        m = OSSFx()
    else:
        raise ValueError(f"Unknown model key {model_key!r}; choices: {_MODEL_KEYS}")
    m.train(False)
    return m


def export(
    model_key: Literal["oss", "oss_rg", "oss_pico", "oss_fx"],
    out_path: Path,
    tier: str = "standard",
    fmt: Literal["onnx", "coreml"] = "onnx",
) -> Path:
    """Export a randomly-initialised model to ONNX (or CoreML).

    Args:
        model_key: One of "oss", "oss_rg", "oss_pico", "oss_fx".
        out_path:  Destination path (.onnx or .mlpackage).
        tier:      Weight tier — ignored by oss_pico and oss_fx.
        fmt:       "onnx" (default) or "coreml".

    Returns:
        Absolute path to the written file.
    """
    out_path = Path(out_path).resolve()
    model = _build_model(model_key, tier)
    exporter = _EXPORTERS[model_key]

    if fmt == "onnx":
        with torch.no_grad():
            result = exporter(model, out_path, tier)
        kb = out_path.stat().st_size / 1024
        log.info("ONNX export complete: %s (%.1f KB)", out_path, kb)
        return result

    # CoreML: ONNX -> tmp file -> coremltools convert.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with torch.no_grad():
            exporter(model, tmp_path, tier)
        return _to_coreml(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Export OSS models to ONNX or CoreML.")
    p.add_argument("--model",  required=True, choices=_MODEL_KEYS)
    p.add_argument("--tier",   default="standard", choices=("lite", "standard", "heavy"),
                   help="Weight tier (ignored for oss_pico, oss_fx)")
    p.add_argument("--output", type=Path, default=Path("model.onnx"))
    p.add_argument("--format", dest="fmt", default="onnx", choices=("onnx", "coreml"))
    args = p.parse_args()
    out = export(args.model, args.output, tier=args.tier, fmt=args.fmt)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
