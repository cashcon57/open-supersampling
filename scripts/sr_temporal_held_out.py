"""Held-out fixed-batch eval: v5-pixel-temporal vs v4 baseline.

Mirrors ``scripts/sr_v3_vs_v4_ab.py`` but adapted for the v5 pixel-temporal
track. Wraps each held-out dataset (Sintel + TartanAir) in
``SequentialPairDataset`` so each batch entry yields ``(t, t+1)`` from the
SAME trajectory, then scores three reconstructions on frame ``t+1``:

  - bicubic upsample (lower-bound),
  - the v4 single-frame baseline checkpoint, and
  - the v5 temporal model (cold-started: ``prev_hr_t = bicubic(LR_t)``,
    then ``prev_hr_{t+1} = baseline_output_at_t.detach()`` — that's the
    regime the deployed inference engine uses on the first frame).

Reports:

  - PSNR + LPIPS for v5-temporal (B), v4-baseline (A), bicubic, with the
    same per-sample win-count format as ``sr_v3_vs_v4_ab.py``.
  - "Temporal stability" block: ``mean(|warp(out_t, motion_t->t+1) -
    out_{t+1}|_1)`` for both v5 and v4 — lower is more temporally stable.

Writes ``held_out_results.json`` next to the temporal checkpoint.

The verification gate for Task 8 of the v5-pixel-temporal plan is just::

    python scripts/sr_temporal_held_out.py --help

returning exit code 0. The actual eval requires real datasets and trained
checkpoints and is run after Sprint-5 training completes (Task 10).

Usage:
    python scripts/sr_temporal_held_out.py \\
        --ckpt-temporal <train-host-data>/checkpoints/srcnn-v5-pixel-temporal/step-XXXXX.pt \\
        --ckpt-baseline <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt \\
        --tartanair-root <train-host-data>/datasets/tartanair_extracted \\
        --sintel-root <train-host-data>/datasets/sintel \\
        --n-samples 64
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

# Allow ``python scripts/sr_temporal_held_out.py`` to import ``oss.*`` when
# the package isn't installed into the active interpreter (e.g. tests invoke
# us via ``sys.executable``). Mirrors ``scripts/sr_train_temporal.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._score_log_io import append_score_log_row

# NOTE: torch and oss imports are deferred into ``main()`` and helpers below
# so that ``--help`` (the Task 8 verification gate) works on a vanilla
# Python interpreter without the heavy ML stack installed.

if TYPE_CHECKING:
    import torch  # noqa: F401
    from torch.utils.data import DataLoader  # noqa: F401

    from oss.sr.temporal import TemporalSRModel  # noqa: F401


DEFAULT_SCALE = 2.0
DEFAULT_LR_SYNTH_ARGS: dict[str, bool | int | float] = {
    "enable_jitter": True,
    "enable_taa_blur": True,
    "enable_jpeg": False,
    "jpeg_quality": 85,
    "blur_sigma": 0.5,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _psnr(pred, target) -> float:
    import torch.nn.functional as F  # local: keep --help torch-free
    mse = float(F.mse_loss(pred.float(), target.float()).item())
    mse = max(mse, 1e-12)
    return float(-10.0 * math.log10(mse))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_baseline(ckpt_path: Path, device: str):
    """Load a v4 single-frame SR-CNN checkpoint (same loader as v3-vs-v4 ab)."""
    import torch
    from oss.sr import build_sr_model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    tier = saved_args.get("tier", "standard")
    sr_backbone = saved_args.get("sr_backbone", "simple")
    factory_kind = "rrdb" if sr_backbone == "rrdb" else "simple"
    model = build_sr_model(
        model_kind=factory_kind, tier=tier, in_channels=12, scale=2,
    ).to(device)
    model.load_state_dict(ckpt["sr_model"])
    model.train(False)
    return model


def _load_temporal(ckpt_path: Path, device: str):
    """Load a v5-pixel-temporal or v6 generator checkpoint."""
    import torch
    from oss.sr.temporal import TemporalSRModel
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ck.get("args", {})

    if any(key in ck for key in ("v6_model", "generator", "model_state_dict")):
        from oss.sr.v6.model import V6Config, V6Model

        cfg_data = ck.get("v6_config", {})
        if not isinstance(cfg_data, dict):
            cfg_data = {}
        cfg_kwargs = dict(cfg_data)
        cfg_kwargs.setdefault("backbone", saved.get("backbone", "hat-tiny"))
        cfg_kwargs.setdefault("in_channels", int(saved.get("in_channels", 9)))
        cfg_kwargs.setdefault("scale", int(saved.get("scale", 2)))
        cfg_kwargs.setdefault("color_activation", saved.get("color_activation", "hdr"))
        cfg_kwargs.setdefault(
            "spawn_offset_random", bool(saved.get("spawn_offset_random", False))
        )
        cfg_kwargs.setdefault(
            "rasterizer_overlap", int(saved.get("rasterizer_overlap", 0))
        )
        # v6.2 architectural switches: pico-002 trains with fusion_mode=concat
        # + spawner_mode=disocclusion + latent_rank=16. Without these the
        # eval instantiates V6Model with the v6.1-default cross_attention path
        # which (a) cannot load the concat-trained weights and (b) OOMs on
        # 12 GB GPUs trying to allocate the global Q@K^T tensor.
        if "fusion_mode" in saved:
            cfg_kwargs.setdefault("fusion_mode", str(saved["fusion_mode"]))
        if "spawner_mode" in saved:
            cfg_kwargs.setdefault("spawner_mode", str(saved["spawner_mode"]))
        if "latent_rank" in saved:
            cfg_kwargs.setdefault("latent_rank", int(saved["latent_rank"]))
        model = V6Model(V6Config(**cfg_kwargs)).to(device)
        state = None
        for key in ("v6_model", "model", "model_state_dict", "generator", "state_dict"):
            if key in ck:
                state = ck[key]
                break
        if state is None:
            raise KeyError(f"checkpoint {ckpt_path} has no v6 state dict")
        model.load_state_dict(state, strict=False)
        model.train(False)
        return model

    tier = saved.get("tier", "standard")
    backbone_kind = saved.get("backbone_kind", "simple")
    # Read the conditional channel-zero flag (added in commit d25b3b9). New
    # ckpts persist the explicit flag; legacy ckpts predate it but warm-
    # started runs (args.warm_start truthy) needed zeroing to match the
    # v4-on-SRGD distribution. Without this restoration at eval time, the
    # eval pathway feeds real TartanAir G-buffers into a backbone that was
    # trained on zeroed G-buffers — producing the rainbow chromatic-
    # dispersion garbage we already debugged in commit b2fa647.
    if "zero_gbuffer_into_backbone" in saved:
        zero_flag = bool(saved["zero_gbuffer_into_backbone"])
    else:
        zero_flag = bool(saved.get("warm_start"))
    model = TemporalSRModel(
        in_channels=12, scale=2, tier=tier, backbone_kind=backbone_kind,
        zero_gbuffer_into_backbone=zero_flag,
    ).to(device)
    if "temporal_model" in ck:
        model.load_state_dict(ck["temporal_model"])
    elif "sr_model" in ck:
        model.backbone.load_state_dict(ck["sr_model"])
    else:
        raise KeyError(
            f"checkpoint {ckpt_path} has neither 'temporal_model' nor "
            f"'sr_model' key (got {list(ck.keys())})"
        )
    model.train(False)
    return model


def _is_v6_model(model: Any) -> bool:
    return model.__class__.__name__ == "V6Model"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _build_pair_loader(kind: str, root: Path, batch_size: int):
    """Build a deterministic SequentialPairDataset loader.

    ``shuffle=False`` and zero-worker keep frame order stable across runs;
    the caller seeds ``torch.manual_seed(0)`` before iterating.
    """
    from torch.utils.data import DataLoader
    from oss.gaussian.data import (
        SintelGaussianDataset,
        TartanAirGaussianDataset,
    )
    from oss.sr.temporal import (
        SequentialPairDataset, adapt_sintel, adapt_tartanair,
        default_collate_pair,
    )
    if kind == "tartanair":
        ds = adapt_tartanair(TartanAirGaussianDataset(root=root, scale=2.0))
    elif kind == "sintel":
        ds = adapt_sintel(
            SintelGaussianDataset(root=root, scale=2.0, pass_name="clean")
        )
    else:
        raise ValueError(f"unknown dataset kind: {kind!r}")
    pair = SequentialPairDataset(ds)
    if len(pair) == 0:
        return None
    return DataLoader(
        pair, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=default_collate_pair, drop_last=True,
    )


class _ExplicitPairDataset:
    """Pair dataset whose pair order is pinned by a held-out manifest."""

    def __init__(self, base: Any, pairs: list[tuple[int, int]]) -> None:
        self.base = base
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        i, j = self.pairs[idx]
        prev_key = self.base.trajectory_key(i - 1) if i > 0 else None
        cur_key = self.base.trajectory_key(i)
        return {
            "t": self.base[i],
            "t_plus_1": self.base[j],
            "is_first_in_seq": bool(prev_key != cur_key),
        }


def _lr_synth_args_from_cli(args: argparse.Namespace) -> dict[str, bool | int | float]:
    cfg = dict(DEFAULT_LR_SYNTH_ARGS)
    cfg["enable_jpeg"] = bool(args.enable_jpeg)
    cfg["jpeg_quality"] = int(args.jpeg_quality)
    cfg["blur_sigma"] = float(args.blur_sigma)
    return cfg


def _validate_manifest_config(
    manifest: Mapping[str, Any],
    *,
    scale: float,
    lr_synth_args: Mapping[str, Any],
) -> None:
    manifest_scale = float(manifest["lr_scale"])
    if abs(manifest_scale - float(scale)) > 1e-9:
        raise ValueError(
            f"manifest lr_scale mismatch: manifest={manifest_scale}, "
            f"script --scale={float(scale)}"
        )
    manifest_lr = dict(manifest.get("lr_synth_args", {}))
    expected_lr = dict(lr_synth_args)
    if manifest_lr != expected_lr:
        raise ValueError(
            "manifest lr_synth_args mismatch: "
            f"manifest={manifest_lr}, script={expected_lr}"
        )


def _split_manifest_paths(manifest_arg: str | Path | None) -> list[Path]:
    if manifest_arg is None:
        return []
    return [
        Path(part.strip())
        for part in str(manifest_arg).split(",")
        if part.strip()
    ]


def _infer_manifest_kind(manifest: Mapping[str, Any]) -> str:
    kind = manifest.get("dataset_kind")
    if kind in {"tartanair", "sintel"}:
        return str(kind)
    trajectories = [str(p["trajectory"]).lower() for p in manifest["pairs"]]
    if any("sintel" in t or "training\\clean" in t or "training/clean" in t for t in trajectories):
        return "sintel"
    return "tartanair"


def _build_manifest_base_dataset(
    kind: str,
    root: Path,
    *,
    scale: float,
    lr_synth_args: Mapping[str, Any],
) -> Any:
    from oss.gaussian.data import (
        SintelGaussianDataset,
        TartanAirGaussianDataset,
    )
    from oss.gaussian.data.lr_synthesis import EngineAliasedLRSynth
    from oss.sr.temporal import adapt_sintel, adapt_tartanair

    lr_synth = EngineAliasedLRSynth(scale=scale, **dict(lr_synth_args))
    if kind == "tartanair":
        return adapt_tartanair(
            TartanAirGaussianDataset(root=root, scale=scale, lr_synth=lr_synth)
        )
    if kind == "sintel":
        return adapt_sintel(
            SintelGaussianDataset(
                root=root, scale=scale, pass_name="clean", lr_synth=lr_synth
            )
        )
    raise ValueError(f"unknown dataset kind: {kind!r}")


def _build_manifest_loader(
    kind: str,
    root: Path,
    manifest_path: Path,
    batch_size: int,
    *,
    scale: float,
    lr_synth_args: Mapping[str, Any],
):
    """Build a loader whose pair order is resolved from a frozen manifest."""
    from torch.utils.data import DataLoader
    from oss.sr.temporal import default_collate_pair
    from oss.sr.temporal.held_out_manifest import load_manifest, manifest_to_pairs

    manifest = load_manifest(manifest_path)
    _validate_manifest_config(
        manifest, scale=scale, lr_synth_args=lr_synth_args,
    )
    base = _build_manifest_base_dataset(
        kind, root, scale=scale, lr_synth_args=lr_synth_args,
    )
    pairs = manifest_to_pairs(manifest, base)
    pair_ds = _ExplicitPairDataset(base, pairs)
    if len(pair_ds) == 0:
        return None
    return DataLoader(
        pair_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=default_collate_pair, drop_last=False,
    )


def _build_manifest_loaders(
    manifest_paths: list[Path],
    *,
    tartanair_root: Path | None,
    sintel_root: Path | None,
    batch_size: int,
    scale: float,
    lr_synth_args: Mapping[str, Any],
) -> list[tuple[str, Any]]:
    from oss.sr.temporal.held_out_manifest import load_manifest

    loaders: list[tuple[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = load_manifest(manifest_path)
        kind = _infer_manifest_kind(manifest)
        if kind == "tartanair":
            if tartanair_root is None:
                raise ValueError(
                    f"manifest {manifest_path} is TartanAir but --tartanair-root was not provided"
                )
            root = tartanair_root
        elif kind == "sintel":
            if sintel_root is None:
                raise ValueError(
                    f"manifest {manifest_path} is Sintel but --sintel-root was not provided"
                )
            root = sintel_root
        else:
            raise ValueError(f"unknown manifest dataset_kind {kind!r} in {manifest_path}")
        loader = _build_manifest_loader(
            kind, root, manifest_path, batch_size,
            scale=scale, lr_synth_args=lr_synth_args,
        )
        if loader is not None:
            loaders.append((kind, loader))
    return loaders


# ---------------------------------------------------------------------------
# Eval core
# ---------------------------------------------------------------------------


def _make_12ch(lr, depth, motion, normals, canvas):
    import torch
    return torch.cat([lr, depth, motion, normals, canvas], dim=1)


def _save_chw_as_png(tensor, dest: Path) -> None:
    """Save a (3, H, W) tensor as an 8-bit RGB PNG.

    Atomic write: encode to a sibling .png.tmp file then os.replace into
    place. Without this, a watcher mid-copy could publish a partial PNG
    -- the dashboard's video player would then render a corrupt frame
    or 404 a half-existent file.
    """
    import os
    import torch
    from PIL import Image
    arr = tensor.detach().clamp(0.0, 1.0).cpu().float()
    if arr.ndim != 3 or arr.shape[0] not in (1, 3):
        raise ValueError(f"expected (C, H, W) with C in (1,3); got {tuple(arr.shape)}")
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    arr_u8 = (arr.mul(255.0).round().to(torch.uint8)
              .permute(1, 2, 0).contiguous().numpy())
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    Image.fromarray(arr_u8, mode="RGB").save(tmp, format="PNG")
    os.replace(tmp, dest)


def _eval_loader(
    loader,
    *,
    model_temporal,
    model_baseline,
    lpips_fn,
    n_samples_remaining: int,
    device: str,
    frames_dir: Path | None = None,
    sample_offset: int = 0,
) -> dict[str, list[float]]:
    """Run held-out eval on a single dataset loader.

    For each batch we evaluate frame ``t+1`` for the three reconstructions
    plus the temporal-stability metric.

    Returns a dict of lists, one entry per held-out frame consumed.
    """
    import torch
    import torch.nn.functional as F
    from oss.sr.temporal import make_first_frame_prev_hr, warp_prev_hr

    psnr_temp: list[float] = []
    psnr_base: list[float] = []
    psnr_bic: list[float] = []
    lpips_temp: list[float] = []
    lpips_base: list[float] = []
    lpips_bic: list[float] = []
    tstab_temp: list[float] = []
    tstab_base: list[float] = []

    def _lpips(pred, target) -> float | None:
        if lpips_fn is None:
            return None
        p = pred.unsqueeze(0).clamp(0.0, 1.0) * 2.0 - 1.0
        t = target.unsqueeze(0).clamp(0.0, 1.0) * 2.0 - 1.0
        return float(lpips_fn(p, t).item())

    with torch.no_grad():
        for batch in loader:
            if len(psnr_temp) >= n_samples_remaining:
                break

            t_lr = batch["t_lr"].to(device)
            t_depth = batch["t_depth"].to(device)
            t_motion = batch["t_motion"].to(device)
            t_normals = batch["t_normals"].to(device)
            t_canvas = batch["t_canvas"].to(device)

            p_lr = batch["tp1_lr"].to(device)
            p_depth = batch["tp1_depth"].to(device)
            p_motion = batch["tp1_motion"].to(device)
            p_normals = batch["tp1_normals"].to(device)
            p_canvas = batch["tp1_canvas"].to(device)
            p_gt = batch["tp1_gt_hr"].to(device)

            H_hr, W_hr = p_gt.shape[-2:]
            scale = int(getattr(model_temporal, "scale", 2))

            # Baseline at t (used as cold-start prev_hr for v5 at t+1).
            # The baseline (v4) was trained on SRGD where depth/motion/normals
            # were zero everywhere except normals[2]=1.0; feeding real
            # TartanAir G-buffers into the backbone here triggers the
            # distribution-shift bug (chromatic-dispersion garbage; see
            # commit b2fa647). Zero non-RGB channels with the same SRGD
            # default-up convention before calling the baseline. The v5
            # temporal model handles this internally via its
            # ``zero_gbuffer_into_backbone`` flag; the baseline is a raw
            # SRCNNSimple/RRDB so we apply the same masking here.
            def _baseline_input(x: "torch.Tensor") -> "torch.Tensor":
                masked = torch.zeros_like(x)
                masked[:, :3] = x[:, :3]
                if masked.shape[1] >= 7:
                    masked[:, 6] = 1.0
                return masked

            x_t = _make_12ch(t_lr, t_depth, t_motion, t_normals, t_canvas)
            base_out_t = model_baseline(_baseline_input(x_t)).clamp(0.0, 1.0)

            # Baseline at t+1 (single-frame) — competitor.
            x_tp1 = _make_12ch(p_lr, p_depth, p_motion, p_normals, p_canvas)
            base_out_tp1 = model_baseline(_baseline_input(x_tp1)).clamp(0.0, 1.0)

            depth_hr_t = F.interpolate(
                t_depth, size=(H_hr, W_hr), mode="bilinear", align_corners=False
            )
            depth_hr_tp1 = F.interpolate(
                p_depth, size=(H_hr, W_hr), mode="bilinear", align_corners=False
            )

            if _is_v6_model(model_temporal):
                if hasattr(model_temporal, "reset_state"):
                    model_temporal.reset_state(device=torch.device(device))

                def _v6_input(x12: "torch.Tensor") -> "torch.Tensor":
                    in_channels = int(model_temporal.cfg.in_channels)
                    x9 = x12[:, :9]
                    if in_channels <= x9.shape[1]:
                        return x9[:, :in_channels]
                    if in_channels <= x12.shape[1]:
                        return x12[:, :in_channels]
                    raise ValueError(
                        f"v6 ckpt expects {in_channels} channels, "
                        f"but eval can supply only {x12.shape[1]}"
                    )

                temp_out_t = model_temporal(
                    lr_inputs=_v6_input(x_t),
                    motion_lr=None,
                    depth_hr_curr=depth_hr_t,
                    depth_hr_prev=depth_hr_t,
                    frame_index=0,
                ).clamp(0.0, 1.0)
                temp_out_tp1 = model_temporal(
                    lr_inputs=_v6_input(x_tp1),
                    motion_lr=t_motion,
                    depth_hr_curr=depth_hr_tp1,
                    depth_hr_prev=depth_hr_t,
                    frame_index=1,
                ).clamp(0.0, 1.0)
            else:
                # v5 temporal at t (cold-started with bilinear up of t_lr) —
                # used both for the t+1 prev_hr feed AND the temporal-stability metric.
                prev_hr_t = make_first_frame_prev_hr(t_lr, scale=scale)
                temp_out_t = model_temporal(
                    lr_inputs=x_t, prev_hr=prev_hr_t,
                    depth_hr_curr=depth_hr_t, depth_hr_prev=depth_hr_t,
                    motion_lr=t_motion,
                ).clamp(0.0, 1.0)

                # v5 temporal at t+1 with cold-start regime: prev_hr =
                # baseline_output_at_t.detach() (matches the deployed inference
                # engine's first-frame behaviour). Motion fed in is ``t_motion``
                # (forward flow t -> t+1, lives at frame t, used as small-motion
                # approximation when sampling at frame t+1's grid).
                temp_out_tp1 = model_temporal(
                    lr_inputs=x_tp1, prev_hr=base_out_t.detach(),
                    depth_hr_curr=depth_hr_tp1, depth_hr_prev=depth_hr_t,
                    motion_lr=t_motion,
                ).clamp(0.0, 1.0)

            # Bicubic upsample of LR_{t+1}.
            bic_tp1 = F.interpolate(
                p_lr, size=(H_hr, W_hr), mode="bicubic", antialias=True
            ).clamp(0.0, 1.0)

            # Temporal stability:
            #   |warp(out_t, motion_{t->t+1}) - out_{t+1}|_1, mean per sample.
            # Same convention: t->t+1 forward flow lives at frame t, i.e.
            # ``t_motion`` (NOT ``p_motion`` which is t+1->t+2).
            warped_temp = warp_prev_hr(temp_out_t, t_motion, scale=scale)
            warped_base = warp_prev_hr(base_out_t, t_motion, scale=scale)

            for b_idx in range(p_lr.shape[0]):
                if len(psnr_temp) >= n_samples_remaining:
                    break
                # Per-sample frame dump for the held-out video player. We
                # write the current ckpt's prediction every eval (it changes
                # per ckpt). GT/bicubic/baseline are deterministic per
                # held-out batch, so we only write them when missing -- that
                # collapses storage from ~180 MB/eval to ~45 MB/eval after
                # the first run.
                if frames_dir is not None:
                    sample_idx = sample_offset + len(psnr_temp)  # global sample id
                    sample_str = f"sample-{sample_idx:03d}"
                    model_path = frames_dir / "model" / f"{sample_str}.png"
                    _save_chw_as_png(temp_out_tp1[b_idx], model_path)
                    for stream, tensor in (
                        ("gt", p_gt[b_idx]),
                        ("bicubic", bic_tp1[b_idx]),
                        ("baseline", base_out_tp1[b_idx]),
                    ):
                        dest = frames_dir / stream / f"{sample_str}.png"
                        if not dest.exists():
                            _save_chw_as_png(tensor, dest)

                psnr_temp.append(_psnr(temp_out_tp1[b_idx], p_gt[b_idx]))
                psnr_base.append(_psnr(base_out_tp1[b_idx], p_gt[b_idx]))
                psnr_bic.append(_psnr(bic_tp1[b_idx], p_gt[b_idx]))

                lt = _lpips(temp_out_tp1[b_idx], p_gt[b_idx])
                lb = _lpips(base_out_tp1[b_idx], p_gt[b_idx])
                lc = _lpips(bic_tp1[b_idx], p_gt[b_idx])
                if lt is not None:
                    lpips_temp.append(lt)
                    lpips_base.append(lb)
                    lpips_bic.append(lc)

                tstab_temp.append(float(
                    (warped_temp[b_idx] - temp_out_tp1[b_idx]).abs().mean().item()
                ))
                tstab_base.append(float(
                    (warped_base[b_idx] - base_out_tp1[b_idx]).abs().mean().item()
                ))

    return {
        "psnr_temporal": psnr_temp,
        "psnr_baseline": psnr_base,
        "psnr_bicubic": psnr_bic,
        "lpips_temporal": lpips_temp,
        "lpips_baseline": lpips_base,
        "lpips_bicubic": lpips_bic,
        "tstab_temporal": tstab_temp,
        "tstab_baseline": tstab_base,
    }


def _merge_results(
    *results: dict[str, list[float]]
) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for r in results:
        for k, v in r.items():
            merged.setdefault(k, []).extend(v)
    return merged


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _step_from_ckpt(ckpt_path: Path) -> int:
    try:
        return int(ckpt_path.stem.split("-")[-1])
    except ValueError:
        return -1


def _dashboard_score_row(
    *,
    ckpt: Path,
    manifest_paths: list[Path],
    result: dict[str, list[float]],
) -> dict[str, Any]:
    model_psnr = result["psnr_temporal"]
    bic_psnr = result["psnr_bicubic"]
    model_lpips = result["lpips_temporal"]
    bic_lpips = result["lpips_bicubic"]
    return {
        "step": _step_from_ckpt(ckpt),
        "model_psnr_mean": _mean(model_psnr),
        "bicubic_psnr_mean": _mean(bic_psnr),
        "per_frame_psnr": list(model_psnr),
        "per_frame_bicubic_psnr": list(bic_psnr),
        "model_lpips_mean": _mean(model_lpips) if model_lpips else None,
        "bicubic_lpips_mean": _mean(bic_lpips) if bic_lpips else None,
        "per_frame_lpips": list(model_lpips),
        "per_frame_bicubic_lpips": list(bic_lpips),
        "model_beats_bicubic_count": sum(1 for m, b in zip(model_psnr, bic_psnr) if m > b),
        "model_beats_bicubic_lpips_count": (
            sum(1 for m, b in zip(model_lpips, bic_lpips) if m < b)
            if model_lpips else None
        ),
        "n_samples": len(model_psnr),
        "manifest": ",".join(str(p) for p in manifest_paths),
        "ckpt": str(ckpt),
    }


def _print_compact_result_block(label: str, result: dict[str, list[float]]) -> None:
    n = len(result["psnr_temporal"])
    if n == 0:
        return
    psnr_a = result["psnr_baseline"]
    psnr_b = result["psnr_temporal"]
    psnr_c = result["psnr_bicubic"]
    lpips_a = result["lpips_baseline"]
    lpips_b = result["lpips_temporal"]
    lpips_c = result["lpips_bicubic"]
    tstab_a = result["tstab_baseline"]
    tstab_b = result["tstab_temporal"]

    print()
    print(f"=== {label} held-out (n={n}) ===")
    print(
        f"PSNR A={_mean(psnr_a):6.3f}  B={_mean(psnr_b):6.3f}  "
        f"bicubic={_mean(psnr_c):6.3f}  B-vs-A={_mean(psnr_b)-_mean(psnr_a):+6.3f}"
    )
    if lpips_a:
        print(
            f"LPIPS A={_mean(lpips_a):6.4f}  B={_mean(lpips_b):6.4f}  "
            f"bicubic={_mean(lpips_c):6.4f}  B-vs-A={_mean(lpips_b)-_mean(lpips_a):+7.4f}"
        )
    print(
        f"Temporal stability A={_mean(tstab_a):7.5f}  "
        f"B={_mean(tstab_b):7.5f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_device() -> str:
    """Best-effort cuda/cpu default; falls back to ``"cpu"`` if torch is absent.

    Keeps ``--help`` working in environments where torch isn't installed
    (the Task 8 verification gate runs on a vanilla interpreter).
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-temporal", type=Path, required=True,
                   help="v5 pixel-temporal checkpoint (.pt). "
                        "held_out_results.json is written next to it.")
    p.add_argument("--ckpt-baseline", type=Path, required=True,
                   help="v4 single-frame baseline checkpoint (.pt).")
    p.add_argument("--tartanair-root", type=Path, default=None,
                   help="Held-out TartanAir trajectory root.")
    p.add_argument("--sintel-root", type=Path, default=None,
                   help="Held-out Sintel root (uses 'clean' pass).")
    p.add_argument("--device", type=str, default=_default_device(),
                   help="cuda if available else cpu")
    p.add_argument("--n-samples", type=int, default=64,
                   help="Total held-out frames to evaluate across both datasets.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--manifest", default=None,
                   help="Frozen held-out manifest JSON path, or a comma-separated "
                        "list of paths. When set, frame pairs are replayed from "
                        "the manifest(s) instead of discovered from "
                        "SequentialPairDataset order.")
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                   help="HR/LR scale factor for manifest config checks.")
    p.add_argument("--enable-jpeg", action="store_true",
                   help="LR synth config flag checked against --manifest.")
    p.add_argument("--blur-sigma", type=float, default=0.5,
                   help="LR synth blur sigma checked against --manifest.")
    p.add_argument("--jpeg-quality", type=int, default=85,
                   help="LR synth JPEG quality checked against --manifest.")
    p.add_argument("--write-frames-to", type=Path, default=None,
                   help="When set, save each held-out sample's prediction + GT/bicubic/baseline "
                        "as PNG sequences into this directory. Used by the dashboard's per-step "
                        "video player. GT/bicubic/baseline are written only if missing.")
    p.add_argument("--score-log", type=Path, default=None,
                   help="Optional dashboard-compatible JSON score log to append/update.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Defer torch import to inside main() so ``--help`` works without torch.
    import torch
    device = args.device
    torch.manual_seed(args.seed)

    manifest_paths = _split_manifest_paths(args.manifest)

    if args.tartanair_root is None and args.sintel_root is None:
        print("FAIL: provide at least one of --tartanair-root / --sintel-root")
        return 1

    print(f"Loading temporal: {args.ckpt_temporal}")
    model_temporal = _load_temporal(args.ckpt_temporal, device)
    print(f"Loading baseline: {args.ckpt_baseline}")
    model_baseline = _load_baseline(args.ckpt_baseline, device)

    # LPIPS — single instance shared between all three reconstructions.
    lpips_fn = None
    try:
        import lpips  # type: ignore[import-not-found]
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device)
        lpips_fn.train(False)
    except Exception as e:
        print(f"WARN: LPIPS unavailable ({e}) - PSNR + temporal stability only")

    # Build deterministic loaders.
    loaders: list[tuple[str, "DataLoader"]] = []
    lr_synth_args = _lr_synth_args_from_cli(args)
    if manifest_paths:
        try:
            loaders = _build_manifest_loaders(
                manifest_paths,
                tartanair_root=args.tartanair_root,
                sintel_root=args.sintel_root,
                batch_size=args.batch_size,
                scale=args.scale,
                lr_synth_args=lr_synth_args,
            )
        except Exception as e:
            print(f"FAIL: {e}")
            return 1
        for name, loader in loaders:
            print(f"{name} manifest held-out pairs: {len(loader.dataset)}")
    else:
        if args.tartanair_root is not None:
            loader = _build_pair_loader(
                "tartanair", args.tartanair_root, args.batch_size,
            )
            if loader is not None:
                loaders.append(("tartanair", loader))
                print(f"tartanair held-out pairs: {len(loader.dataset)}")
        if args.sintel_root is not None:
            loader = _build_pair_loader(
                "sintel", args.sintel_root, args.batch_size,
            )
            if loader is not None:
                loaders.append(("sintel", loader))
                print(f"sintel held-out pairs: {len(loader.dataset)}")

    if not loaders:
        print("FAIL: neither dataset produced any sequential pairs")
        return 1

    # Default path preserves the old total-budget behavior. Manifest mode
    # evaluates up to n_samples from each frozen manifest, so a 64-pair
    # TartanAir manifest + a 64-pair Sintel manifest produces n=128 aggregate.
    per_loader = (
        max(1, args.n_samples)
        if manifest_paths
        else max(1, math.ceil(args.n_samples / len(loaders)))
    )
    per_dataset_results: dict[str, dict[str, list[float]]] = {}
    sample_offset = 0
    for name, loader in loaders:
        print(f"-- evaluating {name} (target ~{per_loader} samples) --")
        # Frames dir gets one subdir per dataset so loaders can't clobber
        # each other's per-sample writes when both contribute to the same
        # held-out batch (TartanAir + Sintel).
        frames_subdir = (args.write_frames_to / name) if args.write_frames_to else None
        per_dataset_results[name] = _eval_loader(
            loader,
            model_temporal=model_temporal,
            model_baseline=model_baseline,
            lpips_fn=lpips_fn,
            n_samples_remaining=per_loader,
            device=device,
            frames_dir=frames_subdir,
            sample_offset=sample_offset,
        )
        sample_offset += len(per_dataset_results[name].get("psnr_temporal", []))

    merged = _merge_results(*per_dataset_results.values())
    n = len(merged["psnr_temporal"])
    if n == 0:
        print("FAIL: no samples evaluated")
        return 1

    if manifest_paths and len(per_dataset_results) > 1:
        for name, result in per_dataset_results.items():
            display = {"tartanair": "TartanAir", "sintel": "Sintel"}.get(name, name)
            _print_compact_result_block(display, result)

    # ---- Print results (mirror sr_v3_vs_v4_ab.py format) ----
    psnr_a = merged["psnr_baseline"]   # A = baseline (v4)
    psnr_b = merged["psnr_temporal"]   # B = v5 temporal
    psnr_c = merged["psnr_bicubic"]
    lpips_a = merged["lpips_baseline"]
    lpips_b = merged["lpips_temporal"]
    lpips_c = merged["lpips_bicubic"]
    tstab_b = merged["tstab_temporal"]
    tstab_a = merged["tstab_baseline"]

    print()
    print(f"=== held-out fixed-batch eval (n={n}) ===")
    print(f"  ckpt_temporal (B) = {args.ckpt_temporal.name}")
    print(f"  ckpt_baseline (A) = {args.ckpt_baseline.name}")
    print()
    print(f"PSNR (dB, higher is better)")
    print(f"  A (baseline) : {_mean(psnr_a):6.3f}")
    print(f"  B (temporal) : {_mean(psnr_b):6.3f}")
    print(f"  bicubic      : {_mean(psnr_c):6.3f}")
    print(f"  B-vs-A       : {_mean(psnr_b)-_mean(psnr_a):+6.3f} dB")
    print(f"  A>bicubic    : {sum(1 for a,c in zip(psnr_a,psnr_c) if a>c)}/{n}")
    print(f"  B>bicubic    : {sum(1 for b,c in zip(psnr_b,psnr_c) if b>c)}/{n}")
    print(f"  B>A          : {sum(1 for a,b in zip(psnr_a,psnr_b) if b>a)}/{n}")
    if lpips_a:
        a_mean = _mean(lpips_a)
        b_mean = _mean(lpips_b)
        rel_pct = 100.0 * (b_mean - a_mean) / a_mean if a_mean else 0.0
        print()
        print(f"LPIPS-VGG (lower is better)")
        print(f"  A (baseline) : {a_mean:6.4f}")
        print(f"  B (temporal) : {b_mean:6.4f}")
        print(f"  bicubic      : {_mean(lpips_c):6.4f}")
        print(f"  B-vs-A       : {b_mean - a_mean:+7.4f}  ({rel_pct:+5.1f}%)")
        print(f"  A<bicubic    : {sum(1 for a,c in zip(lpips_a,lpips_c) if a<c)}/{n}")
        print(f"  B<bicubic    : {sum(1 for b,c in zip(lpips_b,lpips_c) if b<c)}/{n}")
        print(f"  B<A          : {sum(1 for a,b in zip(lpips_a,lpips_b) if b<a)}/{n}")
    print()
    print(f"Temporal stability (mean(|warp(out_t, motion_t->t+1) - out_t+1|_1), lower is better)")
    print(f"  A (baseline) : {_mean(tstab_a):7.5f}")
    print(f"  B (temporal) : {_mean(tstab_b):7.5f}")
    if _mean(tstab_a) > 0:
        ratio = _mean(tstab_b) / _mean(tstab_a)
        print(f"  B/A ratio    : {ratio:5.3f}  (spec target: <= 0.5)")
    print()

    # Per-sample win counts on the joint criterion (PSNR > bicubic AND LPIPS < bicubic).
    n_joint_b_beats_bic = 0
    if lpips_a:
        for pb, pc, lb, lc in zip(psnr_b, psnr_c, lpips_b, lpips_c):
            if pb > pc and lb < lc:
                n_joint_b_beats_bic += 1
        pct = 100.0 * n_joint_b_beats_bic / n
        print(
            f"  B beats bicubic on PSNR AND LPIPS : "
            f"{n_joint_b_beats_bic}/{n}  ({pct:.1f}%)  (spec target: >= 95%)"
        )
        print()

    # ---- Write JSON next to the temporal checkpoint ----
    out: dict[str, Any] = {
        "n_samples": n,
        "ckpt_temporal": str(args.ckpt_temporal),
        "ckpt_baseline": str(args.ckpt_baseline),
        "manifests": [str(p) for p in manifest_paths],
        "datasets": {name: len(r["psnr_temporal"]) for name, r in per_dataset_results.items()},
        "psnr": {
            "baseline_mean": _mean(psnr_a),
            "temporal_mean": _mean(psnr_b),
            "bicubic_mean": _mean(psnr_c),
            "delta_b_minus_a": _mean(psnr_b) - _mean(psnr_a),
            "B_gt_A": sum(1 for a, b in zip(psnr_a, psnr_b) if b > a),
            "B_gt_bicubic": sum(1 for b, c in zip(psnr_b, psnr_c) if b > c),
            "A_gt_bicubic": sum(1 for a, c in zip(psnr_a, psnr_c) if a > c),
        },
        "lpips": {
            "baseline_mean": _mean(lpips_a) if lpips_a else None,
            "temporal_mean": _mean(lpips_b) if lpips_b else None,
            "bicubic_mean": _mean(lpips_c) if lpips_c else None,
            "B_lt_A": sum(1 for a, b in zip(lpips_a, lpips_b) if b < a) if lpips_a else None,
            "B_lt_bicubic": sum(1 for b, c in zip(lpips_b, lpips_c) if b < c) if lpips_a else None,
        },
        "temporal_stability": {
            "baseline_mean": _mean(tstab_a),
            "temporal_mean": _mean(tstab_b),
            "ratio_b_over_a": (_mean(tstab_b) / _mean(tstab_a)) if _mean(tstab_a) > 0 else None,
        },
        "joint_b_beats_bicubic": n_joint_b_beats_bic if lpips_a else None,
        "joint_b_beats_bicubic_pct": (100.0 * n_joint_b_beats_bic / n) if lpips_a and n else None,
    }
    json_path = args.ckpt_temporal.parent / "held_out_results.json"
    with json_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {json_path}")
    if args.score_log is not None:
        row = _dashboard_score_row(
            ckpt=args.ckpt_temporal,
            manifest_paths=manifest_paths,
            result=merged,
        )
        append_score_log_row(args.score_log, row)
        print(f"updated {args.score_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
