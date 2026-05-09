#!/usr/bin/env python
"""In-flight visualization for temporal SR training.

Watches a checkpoint dir, loads the latest ``step-XXXXX.pt`` periodically,
renders a fixed set of held-out frames as a comparison strip:

    v5 primary:
        [ LR | bicubic | v4-baseline | v5-pixel-temporal | GT | |err| ]

    v6 primary:
        [ LR | bicubic | v5-pixel-temporal | v6 | GT | |err v5| | |err v6| ]

Writes ``output_dir/viz/step-XXXXX.png`` after each iteration. Designed to
run as a background loop alongside training; uses CPU inference to avoid
GPU contention with the live training process.

Pair selection is read from the deterministic held-out manifest produced by
``scripts/sr_freeze_held_out_manifest.py``. Default 4 pairs (a small subset
of the full 64 — keeps each iteration under ~30 s on CPU).

Usage::

    python scripts/sr_temporal_inflight_viz.py \\
        --output-dir <train-host-data>/checkpoints/srcnn-v5-pixel-temporal \\
        --manifest <train-host-data>/checkpoints/v5_held_out_manifest.json \\
        --tartanair-root <train-host-data>/datasets/tartanair_extracted \\
        --interval 300 \\
        --n-pairs 4

A companion static file server (``python -m http.server``) can serve the viz
dir to a browser; see launch-status notes for the actual orphan-spawn
command.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow ``python scripts/...`` from a system Python without installing the
# package. Mirrors the other v5-pixel-temporal scripts.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Training output dir containing step-*.pt checkpoints.")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Held-out manifest JSON (from sr_freeze_held_out_manifest.py). "
                        "If omitted for v6, infer from the v6 ckpt args when possible.")
    p.add_argument("--tartanair-root", type=Path, default=None,
                   help="TartanAir root for resolving manifest pair paths.")
    p.add_argument("--n-pairs", type=int, default=4,
                   help="Number of pairs from the manifest to render per iteration.")
    p.add_argument("--interval", type=int, default=300,
                   help="Seconds between viz iterations (default 300 = 5 min).")
    p.add_argument("--device", default="cpu",
                   help="Inference device. Default cpu (avoids contention with training GPU).")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Optional explicit primary checkpoint to render. "
                        "Default uses latest step-*.pt in --output-dir.")
    p.add_argument("--primary-version",
                   choices=("v4", "v5-pixel", "v5-gaussian", "v6"),
                   default=None,
                   help="Checkpoint family in --output-dir. Default infers from run name.")
    p.add_argument("--backbone", choices=("hat-tiny", "hat-small", "hat-l"),
                   default="hat-l",
                   help="Fallback v6 backbone if a ckpt is missing v6_config/args.")
    p.add_argument("--traj-length", type=int, default=None,
                   help="Consecutive frames to replay per manifest entry. Defaults to "
                        "v6 ckpt args.trajectory_length when present, else 2.")
    p.add_argument("--once", action="store_true",
                   help="Render one iteration and exit (smoke / one-shot).")
    p.add_argument("--ckpt-baseline", type=Path, default=None,
                   help="Optional v4-baseline ckpt path; if provided, viz adds "
                        "a v4-baseline column for direct A/B with v5-temporal.")
    p.add_argument("--ckpt-v5-gaussian", type=Path, default=None,
                   help="Optional v5-gaussian-temporal ckpt path; with --ckpt-v6, "
                        "adds a v5-Gaussian comparison column.")
    p.add_argument("--ckpt-v5", type=Path, default=None,
                   help="Required v5-pixel-temporal comparison ckpt for v6-primary. "
                        "If omitted, auto-resolve the validated v5 run's latest ckpt.")
    p.add_argument("--ckpt-v6", type=Path, default=None,
                   help="Optional v6 ckpt path; when provided, viz adds a v6 "
                        "comparison column using oss.sr.v6.model.V6Model.")
    p.add_argument("--err-scale", type=float, default=0.2,
                   help="Error heatmap normalization (per-channel L1 absolute "
                        "error mapped to [0, err_scale] -> red colormap).")
    return p.parse_args(argv)


def _latest_ckpt(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    ckpts = sorted(output_dir.glob("step-*.pt"))
    return ckpts[-1] if ckpts else None


class NonFiniteCheckpointError(RuntimeError):
    """Checkpoint contains non-finite model weights and should be skipped."""


def _infer_primary_version(output_dir: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    name = output_dir.name.lower()
    if name.startswith("srcnn-v6-") or name == "srcnn-v6":
        return "v6"
    # v6.1 runs are excluded from the comparison strip until a `.viz_validated`
    # marker is present in the run dir. See
    # docs/superpowers/experiments/2026-05-07-v6.1-pico-grid-artifact-architectural-fix.md.
    if name.startswith("srcnn-v6.1") and (output_dir / ".viz_validated").exists():
        return "v6"
    if name.startswith("srcnn-v5-pixel-temporal"):
        return "v5-pixel"
    if name.startswith("srcnn-v5-gaussian"):
        return "v5-gaussian"
    if name.startswith("srcnn-v4") or "v4" in name:
        return "v4"
    return "v5-pixel"


def _v6_column_allowed(ckpt_path: Path | None) -> bool:
    """v6.1 run-dirs gate the v6 column behind a `.viz_validated` marker."""
    if ckpt_path is None:
        return False
    run_dir = ckpt_path.parent
    name = run_dir.name.lower()
    if name.startswith("srcnn-v6.1"):
        return (run_dir / ".viz_validated").exists()
    return True


def _step_from_ckpt_name(ckpt_path: Path) -> int:
    step_str = ckpt_path.stem.split("-")[-1]
    try:
        return int(step_str)
    except ValueError:
        return -1


def _state_has_nonfinite(state: dict) -> bool:
    import torch

    for value in state.values():
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            if not bool(torch.isfinite(value).all().item()):
                return True
    return False


def _latest_from_candidates(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            latest = _latest_ckpt(candidate)
            if latest is not None:
                return latest
    return None


def _auto_resolve_v5_ckpt(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    return _latest_from_candidates([
        Path(r"<train-host-data>\checkpoints\srcnn-v5-pixel-temporal-validated"),
        Path("/tmp/oss-runs/srcnn-v5-pixel-temporal-validated"),
        Path(r"<train-host-data>\checkpoints\srcnn-v5-pixel-temporal"),
        Path("/tmp/oss-runs/srcnn-v5-pixel-temporal"),
    ])


def _ckpt_args(ckpt_path: Path, device: str = "cpu") -> dict:
    import torch

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and isinstance(ck.get("args"), dict):
        return ck["args"]
    return {}


def _auto_resolve_manifest(
    *,
    explicit: Path | None,
    primary_ckpt: Path | None,
    output_dir: Path,
    device: str,
) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    candidates: list[Path] = [
        output_dir / "held_out_manifest.json",
        output_dir / "v5_held_out_manifest.json",
    ]
    held_out_envs: list[str] = []
    if primary_ckpt is not None and primary_ckpt.exists():
        saved = _ckpt_args(primary_ckpt, device=device)
        raw = saved.get("held_out_envs", [])
        if isinstance(raw, str):
            held_out_envs = [raw]
        elif isinstance(raw, (list, tuple)):
            held_out_envs = [str(v) for v in raw]
    if not held_out_envs or held_out_envs == ["oldtown"]:
        candidates.extend([
            Path(r"<train-host-data>\checkpoints\v5_held_out_manifest.json"),
            Path("/tmp/oss-runs/v5_held_out_manifest.json"),
        ])
    for env in held_out_envs:
        slug = env.strip().lower()
        if not slug:
            continue
        candidates.extend([
            Path(rf"<train-host-data>\checkpoints\v5_held_out_manifest_{slug}.json"),
            Path(f"/tmp/oss-runs/v5_held_out_manifest_{slug}.json"),
            _REPO_ROOT / "docs" / "superpowers" / "experiments" / f"v5_held_out_manifest_{slug}.json",
        ])
    candidates.append(_REPO_ROOT / "docs" / "superpowers" / "experiments" / "v5_held_out_manifest.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_v6_model(ckpt_path: Path, device: str, fallback_backbone: str = "hat-l"):
    import torch
    from oss.sr.v6.model import V6Config, V6Model

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck.get("args", {}) if isinstance(ck, dict) else {}
    cfg_data = ck.get("v6_config", {}) if isinstance(ck, dict) else {}
    if not isinstance(cfg_data, dict):
        cfg_data = {}
    cfg_kwargs = dict(cfg_data)
    cfg_kwargs.setdefault("backbone", args.get("backbone", fallback_backbone))
    cfg_kwargs.setdefault("in_channels", int(args.get("in_channels", 9)))
    cfg_kwargs.setdefault("scale", int(args.get("scale", 2)))
    cfg_kwargs.setdefault("color_activation", args.get("color_activation", "softplus"))
    # v6.2 architectural switches must be honored or we instantiate the
    # wrong fusion/spawner path and silently render misleading viz strips.
    if "fusion_mode" in args:
        cfg_kwargs.setdefault("fusion_mode", str(args["fusion_mode"]))
    if "spawner_mode" in args:
        cfg_kwargs.setdefault("spawner_mode", str(args["spawner_mode"]))
    if "latent_rank" in args:
        cfg_kwargs.setdefault("latent_rank", int(args["latent_rank"]))
    model = V6Model(V6Config(**cfg_kwargs)).to(device)

    state = None
    if isinstance(ck, dict):
        for key in ("v6_model", "model", "model_state_dict", "generator", "state_dict"):
            if key in ck:
                state = ck[key]
                break
    if state is None and isinstance(ck, dict) and all(hasattr(v, "shape") for v in ck.values()):
        state = ck
    if state is not None:
        if _state_has_nonfinite(state):
            raise NonFiniteCheckpointError(f"{ckpt_path} contains non-finite v6 weights")
        model.load_state_dict(state, strict=False)
    model.train(False)
    return model


def _load_v5_pixel_model(ckpt_path: Path, device: str):
    import torch

    from oss.sr.temporal import TemporalSRModel

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ck.get("args", {})
    tier = saved.get("tier", "standard")
    backbone_kind = saved.get("backbone_kind")
    if backbone_kind is None:
        backbone_kind = "rrdb" if saved.get("sr_backbone") == "rrdb" else "simple"
    if "zero_gbuffer_into_backbone" in saved:
        zero_flag = bool(saved["zero_gbuffer_into_backbone"])
    else:
        zero_flag = bool(saved.get("warm_start"))
    model = TemporalSRModel(
        in_channels=int(saved.get("in_channels", 12)),
        scale=int(saved.get("scale", 2)),
        tier=tier,
        backbone_kind=backbone_kind,
        zero_gbuffer_into_backbone=zero_flag,
    ).to(device)
    state = ck.get("temporal_model")
    if state is None and "sr_model" in ck:
        model.backbone.load_state_dict(ck["sr_model"])
    elif state is not None:
        if _state_has_nonfinite(state):
            raise NonFiniteCheckpointError(f"{ckpt_path} contains non-finite v5 weights")
        model.load_state_dict(state)
    else:
        raise KeyError(f"{ckpt_path} has no temporal_model or sr_model")
    model.train(False)
    return model


def _load_v5_gaussian_engine(ckpt_path: Path, device: str):
    from oss.sr.inference import GaussianTemporalSRInferenceEngine

    return GaussianTemporalSRInferenceEngine.from_checkpoint(
        ckpt_path, device=device, fp16=False, scene_cut_motion_threshold=32.0,
    )


def _v6_label_from_ckpt(ckpt_path) -> str:
    """Derive 'v6' / 'v6.1' / 'v6.2' column label from the run-dir name."""
    if ckpt_path is None:
        return "v6"
    name = ckpt_path.parent.name.lower()
    # Match srcnn-v6.<digit>-* → v6.<digit>; srcnn-v6-* → v6.
    import re
    m = re.match(r"srcnn-v(6(?:\.\d+)?)-", name)
    if m:
        return "v" + m.group(1)
    return "v6"


def _comparison_panels(
    *,
    lr_up,
    bicubic,
    baseline=None,
    pixel,
    gt,
    err_rgb,
    err_rgb_v6=None,
    gaussian=None,
    v6=None,
    v6_label="v6",
):
    if v6 is None:
        baseline = bicubic if baseline is None else baseline
        return (
            [lr_up, bicubic, baseline, pixel, gt, err_rgb],
            ["LR-bilinear", "bicubic", "v4-baseline", "v5-temporal", "GT", "|err| heatmap"],
        )
    panels = [lr_up, bicubic]
    labels = ["LR-bilinear", "bicubic"]
    if baseline is not None:
        panels.append(baseline)
        labels.append("v4-baseline")
    panels.append(pixel)
    labels.append("v5-pixel-temporal")
    if gaussian is not None:
        panels.append(gaussian)
        labels.append("v5-Gaussian")
    panels.extend([v6, gt, err_rgb, err_rgb_v6 if err_rgb_v6 is not None else err_rgb])
    labels.extend([v6_label, "GT", f"|err v5|", f"|err {v6_label}|"])
    return panels, labels


def _error_heatmap(pred, gt, err_scale: float):
    import torch

    err = (pred - gt).abs().mean(dim=0, keepdim=True)
    err_norm = (err / max(err_scale, 1e-6)).clamp(0.0, 1.0)
    red = (err_norm * 2.0).clamp(0.0, 1.0)
    green = (err_norm * 2.0 - 1.0).clamp(0.0, 1.0)
    blue = torch.zeros_like(err_norm)
    return torch.cat([red, green, blue], dim=0)


def _render_iteration(
    *,
    ckpt_path: Path,
    manifest_path: Path,
    tartanair_root: Path,
    output_dir: Path,
    n_pairs: int,
    device: str,
    primary_version: str = "v5-pixel",
    ckpt_baseline: Path | None = None,
    ckpt_v5_gaussian: Path | None = None,
    ckpt_v6: Path | None = None,
    ckpt_v5: Path | None = None,
    fallback_backbone: str = "hat-l",
    traj_length: int | None = None,
    err_scale: float = 0.2,
) -> Path | None:
    """Render one checkpoint into ``viz/step-XXXXXXXX.png``."""
    import torch
    import torch.nn.functional as F

    from oss.gaussian.data import EngineAliasedLRSynth, TartanAirGaussianDataset
    from oss.sr import build_sr_model
    from oss.sr.temporal import adapt_tartanair, make_first_frame_prev_hr
    from oss.sr.temporal.held_out_manifest import load_manifest, manifest_to_pairs

    step = _step_from_ckpt_name(ckpt_path)

    viz_dir = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True, parents=True)
    out_path = viz_dir / f"step-{step:08d}.png"
    if out_path.exists():
        return None  # already rendered this step

    if primary_version == "v6":
        v6_ckpt = ckpt_path
        v5_ckpt = ckpt_v5
        if v5_ckpt is None:
            raise FileNotFoundError(
                "v6-primary viz requires a v5-pixel-temporal comparison ckpt. "
                "Pass --ckpt-v5, or make the validated v5 run reachable at "
                r"<train-host-data>\checkpoints\srcnn-v5-pixel-temporal-validated or "
                "/tmp/oss-runs/srcnn-v5-pixel-temporal-validated."
            )
    else:
        v5_ckpt = ckpt_path
        v6_ckpt = ckpt_v6 if ckpt_v6 is not None and ckpt_v6.exists() else None
        if not _v6_column_allowed(v6_ckpt):
            v6_ckpt = None

    model = _load_v5_pixel_model(v5_ckpt, device)

    v6_model = None
    if v6_ckpt is not None and v6_ckpt.exists():
        v6_model = _load_v6_model(v6_ckpt, device, fallback_backbone=fallback_backbone)

    # Load v4-baseline single-frame model (optional column).
    baseline = None
    if ckpt_baseline is not None and ckpt_baseline.exists():
        bck = torch.load(ckpt_baseline, map_location=device, weights_only=False)
        bsaved = bck.get("args", {})
        b_tier = bsaved.get("tier", "standard")
        b_backbone = bsaved.get("sr_backbone", "simple")
        b_kind = "rrdb" if b_backbone == "rrdb" else "simple"
        baseline = build_sr_model(
            model_kind=b_kind, tier=b_tier, in_channels=12, scale=2,
        ).to(device)
        baseline.load_state_dict(bck["sr_model"])
        baseline.train(False)

    gaussian_engine = None
    if ckpt_v5_gaussian is not None and ckpt_v5_gaussian.exists():
        gaussian_engine = _load_v5_gaussian_engine(ckpt_v5_gaussian, device)

    if traj_length is None:
        saved = _ckpt_args(ckpt_path, device=device)
        traj_length = int(saved.get("trajectory_length", 2))
    traj_length = max(2, int(traj_length))

    # Load manifest + dataset. Codex R5 review fixed three issues:
    #
    #   (a) Distribution match — the dataset is built with
    #       EngineAliasedLRSynth(...) using the manifest's saved LR-synth
    #       config, so the LR fed to the model matches the held-out script's
    #       LR generation regime (rather than a too-clean box-downsample).
    #
    #   (b) Real temporal eval — the manifest's (idx_t, idx_t_plus_1) pair
    #       is honored. We seed prev_hr by running the model on frame t,
    #       then visualize the OUTPUT on frame t+1 using t_motion. That is
    #       the same regime sr_temporal_held_out.py uses; without it the
    #       viz only shows the first-frame fallback path and never exercises
    #       the temporal head's prev-HR warp.
    #
    #   (c) Use manifest_to_pairs(manifest, base) for hard-fail-on-mismatch
    #       pair resolution rather than a silent skip on missing frames.
    manifest = load_manifest(manifest_path)
    pairs_meta = manifest["pairs"][:n_pairs]
    lr_synth = EngineAliasedLRSynth(
        scale=manifest["lr_scale"], **manifest.get("lr_synth_args", {})
    )
    base = adapt_tartanair(
        TartanAirGaussianDataset(
            root=tartanair_root, scale=manifest["lr_scale"], lr_synth=lr_synth,
        )
    )

    # Resolve each manifest pair to base-dataset indices. ``manifest_to_pairs``
    # raises clearly on a path/frame mismatch; subset to the first N_pairs we
    # actually want to visualize.
    sliced = dict(manifest)
    sliced["pairs"] = pairs_meta
    sliced["n_pairs"] = len(pairs_meta)
    resolved = manifest_to_pairs(sliced, base)

    rendered_strips: list[torch.Tensor] = []
    panel_labels: list[str] = []

    def _to_x12(ex):
        lr = ex.lr_frame.to(device)
        depth_lr = ex.depth.to(device)
        motion_lr = ex.motion.to(device)
        normals = (ex.normals if ex.normals is not None else
                   torch.zeros((3, *lr.shape[-2:]), dtype=lr.dtype)).to(device)
        canvas = (ex.canvas_hint if ex.canvas_hint is not None else
                  torch.zeros((3, *lr.shape[-2:]), dtype=lr.dtype)).to(device)
        x12 = torch.cat([lr, depth_lr, motion_lr, normals, canvas], dim=0).unsqueeze(0)
        x9 = torch.cat([lr, depth_lr, motion_lr, normals], dim=0).unsqueeze(0)
        h_hr, w_hr = lr.shape[-2] * model.scale, lr.shape[-1] * model.scale
        depth_hr = F.interpolate(
            depth_lr.unsqueeze(0), size=(h_hr, w_hr),
            mode="bilinear", align_corners=False,
        )
        return lr, motion_lr, x12, x9, depth_hr

    for (base_idx_t, _base_idx_tp1) in resolved:
        traj_key = base.trajectory_key(base_idx_t)
        seq_indices = [base_idx_t + offset for offset in range(traj_length)]
        if (
            seq_indices[-1] >= len(base)
            or any(base.trajectory_key(idx) != traj_key for idx in seq_indices)
        ):
            print(
                f"  skip manifest trajectory {traj_key}: shorter than "
                f"--traj-length={traj_length}",
                flush=True,
            )
            continue

        examples = [base[idx] for idx in seq_indices]
        v6_out = None
        prev_v5 = None
        prev_depth_hr = None
        prev_motion = None
        if gaussian_engine is not None:
            gaussian_engine.reset()
        if v6_model is not None:
            v6_model.reset_state(device=torch.device(device))

        for frame_index, ex in enumerate(examples):
            lr, motion_lr, x12, x9, depth_hr = _to_x12(ex)
            h_hr, w_hr = depth_hr.shape[-2:]
            motion_for_temporal = (
                torch.zeros_like(motion_lr).unsqueeze(0)
                if frame_index == 0 or prev_motion is None
                else prev_motion.unsqueeze(0)
            )

            with torch.no_grad():
                if prev_v5 is None:
                    prev_hr = make_first_frame_prev_hr(x12[:, :3], scale=model.scale)
                    prev_depth = depth_hr
                else:
                    prev_hr = prev_v5
                    prev_depth = prev_depth_hr
                out_v5 = model(
                    lr_inputs=x12, prev_hr=prev_hr,
                    depth_hr_curr=depth_hr, depth_hr_prev=prev_depth,
                    motion_lr=motion_for_temporal,
                ).clamp(0.0, 1.0)

                gauss_out = None
                if gaussian_engine is not None:
                    gauss_out = gaussian_engine(
                        lr_inputs=x12,
                        motion_lr=motion_for_temporal,
                    ).clamp(0.0, 1.0)

                if v6_model is not None:
                    in_channels = int(v6_model.cfg.in_channels)
                    if in_channels <= x9.shape[1]:
                        v6_inputs = x9[:, :in_channels]
                    elif in_channels <= x12.shape[1]:
                        v6_inputs = x12[:, :in_channels]
                    else:
                        raise ValueError(
                            f"v6 ckpt expects {in_channels} channels, but viz can "
                            f"supply only {x12.shape[1]}"
                        )
                    v6_out = v6_model(
                        lr_inputs=v6_inputs,
                        motion_lr=None if frame_index == 0 else motion_for_temporal,
                        depth_hr_curr=depth_hr,
                        depth_hr_prev=depth_hr if prev_depth_hr is None else prev_depth_hr,
                        frame_index=frame_index,
                    ).clamp(0.0, 1.0)

            if frame_index > 0:
                bicubic = F.interpolate(
                    x12[:, :3], size=(h_hr, w_hr), mode="bicubic", antialias=True
                ).clamp(0.0, 1.0)
                lr_up = F.interpolate(
                    x12[:, :3], size=(h_hr, w_hr), mode="bilinear", align_corners=False
                ).clamp(0.0, 1.0)
                gt = ex.gt_hr_frame.unsqueeze(0).to(device).clamp(0.0, 1.0)

                base_out = None
                if baseline is not None:
                    base_in = torch.zeros_like(x12)
                    base_in[:, :3] = x12[:, :3]
                    if base_in.shape[1] >= 7:
                        base_in[:, 6] = 1.0
                    with torch.no_grad():
                        base_out = baseline(base_in).clamp(0.0, 1.0)

                err_v5 = _error_heatmap(out_v5[0], gt[0], err_scale)
                err_v6 = (
                    _error_heatmap(v6_out[0], gt[0], err_scale)
                    if v6_out is not None else None
                )
                panels, panel_labels = _comparison_panels(
                    lr_up=lr_up[0],
                    bicubic=bicubic[0],
                    baseline=base_out[0] if base_out is not None else (
                        None if v6_out is not None else bicubic[0]
                    ),
                    pixel=out_v5[0],
                    gaussian=gauss_out[0] if gauss_out is not None else None,
                    v6=v6_out[0] if v6_out is not None else None,
                    gt=gt[0],
                    err_rgb=err_v5,
                    err_rgb_v6=err_v6,
                    v6_label=_v6_label_from_ckpt(v6_ckpt),
                )
                rendered_strips.append(torch.cat(panels, dim=-1).cpu())

            prev_v5 = out_v5.detach()
            prev_depth_hr = depth_hr.detach()
            prev_motion = motion_lr.detach()
    if not rendered_strips:
        return None

    # Stack vertically across the n_pairs strips.
    composite = torch.cat(rendered_strips, dim=-2)

    # Convert to PIL + draw bottom-right panel labels so each preview is
    # self-identifying when viewed in isolation. Each panel is ``W_hr`` wide;
    # labels go inside that panel's bottom-right corner with a dark scrim.
    from PIL import Image, ImageDraw
    arr = (composite.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = Image.fromarray(arr)
    drawer = ImageDraw.Draw(img, mode="RGBA")
    panel_w = img.width // len(panel_labels)
    for i, label in enumerate(panel_labels):
        # Estimate text width for default font (~6px per char).
        text_w = 6 * len(label) + 12
        text_h = 18
        x_right = (i + 1) * panel_w - 6
        x_left = x_right - text_w
        # Per-strip Y stride: place at bottom of EACH stacked sub-strip so the
        # label is visible even when scrubbing.
        n_strips = len(rendered_strips)
        strip_h = img.height // n_strips
        for s in range(n_strips):
            y_bottom = (s + 1) * strip_h - 6
            y_top = y_bottom - text_h
            # Dark scrim under the text for legibility on bright frames.
            drawer.rectangle([(x_left, y_top), (x_right, y_bottom)], fill=(0, 0, 0, 160))
            # Centre-align text inside the scrim box.
            drawer.text((x_left + 6, y_top + 2), label, fill=(255, 255, 255, 255))
    img.save(out_path, format="PNG", optimize=False)
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    torch.set_num_threads(2)
    if args.device != "cpu":
        print("forcing --device cpu for in-flight viz to avoid GPU contention", flush=True)
        args.device = "cpu"

    primary_version = _infer_primary_version(args.output_dir, args.primary_version)
    if args.tartanair_root is None or not args.tartanair_root.is_dir():
        print(f"--tartanair-root must point to an existing dir", file=sys.stderr)
        return 1

    print(f"in-flight viz: output_dir={args.output_dir} interval={args.interval}s "
          f"n_pairs={args.n_pairs} primary={primary_version} device={args.device}",
          flush=True)

    last_step = -1
    iters = 0
    while True:
        ckpt = args.ckpt if args.ckpt is not None else _latest_ckpt(args.output_dir)
        if ckpt is None:
            print(f"  no checkpoints yet at {args.output_dir}", flush=True)
        elif not ckpt.is_file():
            print(f"  checkpoint not found: {ckpt}", file=sys.stderr, flush=True)
            if args.once:
                return 1
        else:
            step = _step_from_ckpt_name(ckpt)
            if step != last_step:
                t0 = time.monotonic()
                try:
                    ckpt_v5 = (
                        _auto_resolve_v5_ckpt(args.ckpt_v5)
                        if primary_version == "v6" else args.ckpt_v5
                    )
                    manifest = _auto_resolve_manifest(
                        explicit=args.manifest,
                        primary_ckpt=ckpt,
                        output_dir=args.output_dir,
                        device=args.device,
                    )
                    if manifest is None:
                        raise FileNotFoundError(
                            "held-out manifest not found. Pass --manifest, or place it at "
                            "<output-dir>/held_out_manifest.json / "
                            r"<train-host-data>\checkpoints\v5_held_out_manifest*.json / "
                            "/tmp/oss-runs/v5_held_out_manifest*.json."
                        )
                    out = _render_iteration(
                        ckpt_path=ckpt, manifest_path=manifest,
                        tartanair_root=args.tartanair_root, output_dir=args.output_dir,
                        n_pairs=args.n_pairs, device=args.device,
                        primary_version=primary_version,
                        ckpt_baseline=args.ckpt_baseline,
                        ckpt_v5_gaussian=args.ckpt_v5_gaussian,
                        ckpt_v6=args.ckpt_v6,
                        ckpt_v5=ckpt_v5,
                        fallback_backbone=args.backbone,
                        traj_length=args.traj_length,
                        err_scale=args.err_scale,
                    )
                    elapsed = time.monotonic() - t0
                    if out is None:
                        print(f"  step {step}: no new viz", flush=True)
                    else:
                        print(f"  step {step}: rendered {out} in {elapsed:.1f}s", flush=True)
                    last_step = step
                except NonFiniteCheckpointError as e:
                    print(f"  step {step}: checkpoint has NaN/Inf weights; skipping: {e}", flush=True)
                    last_step = step
                except Exception as e:
                    print(f"  step {step}: render failed: {e}", flush=True)
            else:
                print(f"  step {step}: unchanged, skipping", flush=True)
        iters += 1
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
