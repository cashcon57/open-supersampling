#!/usr/bin/env python
"""Phase 4 Tier-2 statistics probe for v6/v6.1 pipeline-elegance audit.

The script tries to run one real v6/v6.1 forward from ``--ckpt`` and
``--input``. If either artifact is missing or incompatible, it falls back to a
seeded synthetic v6.1-pico forward and marks every result as fallback.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from oss.sr.v6.model import V6Config, V6Model


def _tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    x = x.detach().float().cpu()
    if x.numel() == 0:
        return {"shape": list(x.shape), "mean": None, "std": None, "min": None, "max": None}
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.amin().item()),
        "max": float(x.amax().item()),
    }


def _finite_fraction(x: torch.Tensor) -> float:
    return float(torch.isfinite(x).float().mean().item())


def _load_ckpt(path: Path, device: torch.device) -> tuple[V6Model, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint is {type(ckpt).__name__}, expected dict")

    args = ckpt.get("args", {})
    args = args if isinstance(args, dict) else {}
    cfg_data = ckpt.get("v6_config", {})
    cfg_data = cfg_data if isinstance(cfg_data, dict) else {}
    valid = {f.name for f in fields(V6Config)}
    cfg_kwargs = {k: v for k, v in cfg_data.items() if k in valid}
    cfg_kwargs.setdefault("backbone", args.get("backbone", "hat-tiny"))
    cfg_kwargs.setdefault("in_channels", int(args.get("in_channels", 9)))
    cfg_kwargs.setdefault("scale", int(args.get("scale", 2)))
    cfg_kwargs.setdefault("color_activation", args.get("color_activation", "hdr"))
    cfg_kwargs.setdefault("spawn_offset_random", bool(args.get("spawn_offset_random", "v6.1" in str(path))))
    cfg_kwargs.setdefault("rasterizer_overlap", int(args.get("rasterizer_overlap", 8 if "v6.1" in str(path) else 0)))
    model = V6Model(V6Config(**cfg_kwargs)).to(device)

    state = None
    for key in ("generator", "v6_model", "model_state_dict", "model", "state_dict"):
        if isinstance(ckpt.get(key), dict):
            state = ckpt[key]
            break
    if state is None:
        raise KeyError(f"no v6 state key found; keys={list(ckpt.keys())[:20]}")
    result = model.load_state_dict(state, strict=False)
    if len(result.missing_keys) > 50 or len(result.unexpected_keys) > 50:
        raise RuntimeError(
            "checkpoint state is not compatible with v6 model "
            f"(missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)})"
        )
    model.eval()
    info = {
        "loaded": True,
        "kind": ckpt.get("kind"),
        "step": ckpt.get("step"),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "config": {k: getattr(model.cfg, k) for k in valid},
    }
    return model, info


def _coerce_lr_input(value: Any, *, in_channels: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, dict):
        for key in ("lr_inputs", "lr_input", "input", "x"):
            if torch.is_tensor(value.get(key)):
                value = value[key]
                break
        else:
            parts = []
            for key in ("lr_frame", "depth", "motion", "normals"):
                if torch.is_tensor(value.get(key)):
                    t = value[key]
                    if t.ndim == 3:
                        t = t.unsqueeze(0)
                    parts.append(t)
            if parts:
                value = torch.cat(parts, dim=1)
            else:
                raise KeyError("input dict has no lr_inputs/input/x or v6 frame parts")
    if not torch.is_tensor(value):
        raise TypeError(f"input payload is {type(value).__name__}, expected tensor/dict")
    x = value.detach().float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"input tensor must be BCHW or CHW, got shape={list(x.shape)}")
    if x.shape[1] < in_channels:
        x = F.pad(x, (0, 0, 0, 0, 0, in_channels - x.shape[1]))
    elif x.shape[1] > in_channels:
        x = x[:, :in_channels]
    return x.to(device=device).contiguous()


def _load_input(path: Path, *, in_channels: int, device: torch.device) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return _coerce_lr_input(torch.load(path, map_location=device, weights_only=False), in_channels=in_channels, device=device)
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PIL/numpy are required for image inputs") from exc
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0
    return _coerce_lr_input(arr, in_channels=in_channels, device=device)


def _synthetic_input(*, h: int, w: int, in_channels: int, device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    rgb = torch.rand((1, 3, h, w), generator=gen, device=device)
    rest = torch.zeros((1, in_channels - 3, h, w), device=device)
    if in_channels >= 4:
        rest[:, 0:1].uniform_(0.1, 1.0, generator=gen)
    if in_channels >= 6:
        rest[:, 1:3] = 0.0
    if in_channels >= 9:
        rest[:, 3:6].uniform_(-1.0, 1.0, generator=gen)
    return torch.cat([rgb, rest], dim=1).contiguous()


def _make_fallback_model(device: torch.device, seed: int) -> V6Model:
    torch.manual_seed(seed)
    model = V6Model(
        V6Config(
            backbone="hat-tiny",
            canvas_capacity=128,
            spawn_offset_random=True,
            rasterizer_overlap=8,
            color_activation="sdr",
        )
    ).to(device)
    model.eval()
    return model


def _collect(model: V6Model, lr_inputs: torch.Tensor) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def spawner_hook(_module, _inputs, output) -> None:
        captured["spawned_positions"] = output.positions.detach()
        captured["spawned_scales"] = output.scales.detach()
        captured["spawned_opacities"] = output.opacities.detach()
        captured["spawned_colors"] = output.colors.detach()
        captured["spawned_confidence"] = output.confidence.detach()

    def composite_hook(_module, _inputs, output) -> None:
        captured["composite_delta"] = output.detach()

    h1 = model.gaussian_spawner.register_forward_hook(spawner_hook)
    h2 = model.composite_head.register_forward_hook(composite_hook)
    model.reset_state(device=lr_inputs.device)
    try:
        with torch.inference_mode():
            out = model(lr_inputs)
    finally:
        h1.remove()
        h2.remove()

    bicubic = F.interpolate(
        lr_inputs[:, :3],
        size=out.shape[-2:],
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    delta = captured.get("composite_delta", out - bicubic)
    canvas = model._canvas_state
    canvas_count = int(canvas.count) if canvas is not None else 0
    alive_ratio = float(canvas_count / max(1, int(model.cfg.canvas_capacity)))
    spawned_scales = captured.get("spawned_scales")
    spawned_opacities = captured.get("spawned_opacities")
    spawned_conf = captured.get("spawned_confidence")

    return {
        "common": {
            "input": _tensor_stats(lr_inputs),
            "output": _tensor_stats(out),
            "output_finite_fraction": _finite_fraction(out),
            "canvas_count_after_forward": canvas_count,
            "canvas_capacity": int(model.cfg.canvas_capacity),
        },
        "technique_b": {
            "bicubic_skip_output_l1": float((out - bicubic).abs().mean().item()),
            "bicubic_skip_output_l2": float(torch.sqrt(torch.mean((out - bicubic).float() ** 2)).item()),
            "bicubic_finite_fraction": _finite_fraction(bicubic),
        },
        "technique_f": {
            "spawned_count": int(captured.get("spawned_positions", torch.empty(0)).numel() // 2),
            "spawn_confidence": _tensor_stats(spawned_conf) if spawned_conf is not None else None,
            "spawn_opacity": _tensor_stats(spawned_opacities) if spawned_opacities is not None else None,
        },
        "technique_g": {
            "spawn_scale": _tensor_stats(spawned_scales) if spawned_scales is not None else None,
            "spawn_scale_anisotropy_mean": (
                float((spawned_scales[..., 0] / spawned_scales[..., 1].clamp_min(1.0e-6)).mean().item())
                if spawned_scales is not None else None
            ),
        },
        "technique_i": {
            "composite_delta": _tensor_stats(delta),
            "delta_to_output_l1_ratio": float(delta.abs().mean().item() / max(out.abs().mean().item(), 1.0e-12)),
        },
        "technique_k": {
            "canvas_alive_ratio": alive_ratio,
            "model_step_count": int(model._step_count.item()),
            "has_st_score_state": bool(model._st_state is not None),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--synthetic-h", type=int, default=32)
    p.add_argument("--synthetic-w", type=int, default=48)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    fallback_reasons: list[str] = []

    model: V6Model | None = None
    ckpt_info: dict[str, Any] = {"loaded": False}
    if args.ckpt is None:
        fallback_reasons.append("--ckpt not provided")
    elif not args.ckpt.exists():
        fallback_reasons.append(f"--ckpt not found: {args.ckpt}")
    else:
        try:
            model, ckpt_info = _load_ckpt(args.ckpt, device)
        except Exception as exc:  # noqa: BLE001
            fallback_reasons.append(f"ckpt load failed: {type(exc).__name__}: {exc}")

    if model is None:
        model = _make_fallback_model(device, args.seed)

    lr_inputs: torch.Tensor | None = None
    if args.input is None:
        fallback_reasons.append("--input not provided")
    elif not args.input.exists():
        fallback_reasons.append(f"--input not found: {args.input}")
    else:
        try:
            lr_inputs = _load_input(args.input, in_channels=model.cfg.in_channels, device=device)
        except Exception as exc:  # noqa: BLE001
            fallback_reasons.append(f"input load failed: {type(exc).__name__}: {exc}")

    if lr_inputs is None:
        lr_inputs = _synthetic_input(
            h=args.synthetic_h,
            w=args.synthetic_w,
            in_channels=model.cfg.in_channels,
            device=device,
            seed=args.seed + 1,
        )

    stats = _collect(model, lr_inputs)
    payload = {
        "schema_version": 1,
        "script": str(Path(__file__).relative_to(ROOT)),
        "fallback": bool(fallback_reasons),
        "fallback_reasons": fallback_reasons,
        "ckpt": str(args.ckpt) if args.ckpt is not None else None,
        "input": str(args.input) if args.input is not None else None,
        "device": str(device),
        "seed": int(args.seed),
        "ckpt_info": ckpt_info,
        "model_config": {
            "backbone": model.cfg.backbone,
            "in_channels": model.cfg.in_channels,
            "scale": model.cfg.scale,
            "canvas_capacity": model.cfg.canvas_capacity,
            "spawn_offset_random": model.cfg.spawn_offset_random,
            "rasterizer_overlap": model.cfg.rasterizer_overlap,
            "color_activation": model.cfg.color_activation,
        },
        "stats": stats,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if math.isfinite(stats["common"]["output"]["mean"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
