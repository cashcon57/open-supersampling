"""v6 held-out eval on the frozen v5 TartanAir/Sintel manifest.

Writes dashboard-compatible rows to ``<output-dir>/score_log.json``. The row
schema matches the training dashboard's existing score reader:

    step, model_psnr_mean, bicubic_psnr_mean,
    model_lpips_mean, bicubic_lpips_mean,
    model_ssim_mean, bicubic_ssim_mean,
    model_beats_bicubic_count, model_beats_bicubic_lpips_count

The eval mirrors the v5 fixed-batch path: replay manifest pairs in order,
run the model on full frames, score frame ``t+1`` against the HR target, and
compare against bicubic upsample of the same LR input.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_SCALE = 2.0
DEFAULT_TARTANAIR_ROOT = Path("E:/datasets/tartanair_extracted")
DEFAULT_MANIFEST = Path("E:/checkpoints/v5_held_out_manifest.json")
DEFAULT_LR_SYNTH_ARGS: dict[str, bool | int | float] = {
    "enable_jitter": True,
    "enable_taa_blur": True,
    "enable_jpeg": False,
    "jpeg_quality": 85,
    "blur_sigma": 0.5,
}


def _psnr(pred, target) -> float:
    import torch.nn.functional as F

    mse = float(F.mse_loss(pred.float(), target.float()).item())
    return float(-10.0 * math.log10(max(mse, 1e-12)))


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _step_from_ckpt(path: Path) -> int:
    m = re.search(r"step-(\d+)\.pt$", path.name)
    if m:
        return int(m.group(1))
    return 0


def _latest_ckpt(output_dir: Path) -> Path | None:
    ckpts = sorted(output_dir.glob("step-*.pt"))
    return ckpts[-1] if ckpts else None


def _state_has_nonfinite(state: Mapping[str, Any]) -> bool:
    import torch

    for value in state.values():
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
            return True
    return False


def _load_v6_model(ckpt_path: Path, device: str):
    import torch
    from oss.sr.v6.model import V6Config, V6Model

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck.get("args", {}) if isinstance(ck, dict) else {}
    cfg_data = ck.get("v6_config", {}) if isinstance(ck, dict) else {}
    if not isinstance(cfg_data, dict):
        cfg_data = {}

    valid_cfg_keys = {f.name for f in fields(V6Config)}
    cfg_kwargs = {k: v for k, v in dict(cfg_data).items() if k in valid_cfg_keys}
    cfg_kwargs.setdefault("backbone", args.get("backbone", "hat-l"))
    cfg_kwargs.setdefault("in_channels", int(args.get("in_channels", 9)))
    cfg_kwargs.setdefault("scale", int(args.get("scale", 2)))
    cfg_kwargs.setdefault("color_activation", args.get("color_activation", "hdr"))
    model = V6Model(V6Config(**cfg_kwargs)).to(device)

    state = None
    if isinstance(ck, dict):
        for key in ("v6_model", "generator", "model_state_dict", "model", "state_dict"):
            if key in ck:
                state = ck[key]
                break
    if state is None and isinstance(ck, dict) and all(hasattr(v, "shape") for v in ck.values()):
        state = ck
    if state is None:
        keys = list(ck.keys()) if isinstance(ck, dict) else type(ck)
        raise KeyError(
            f"{ckpt_path} has no v6 model state; got keys={keys}"
        )
    if _state_has_nonfinite(state):
        raise ValueError(f"{ckpt_path} contains non-finite v6 weights")
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            "WARN: checkpoint schema drift: "
            f"missing={result.missing_keys} unexpected={result.unexpected_keys}",
            flush=True,
        )
    model.train(False)
    return model


def _make_9ch(lr, depth, motion, normals):
    import torch

    return torch.cat([lr, depth, motion, normals], dim=1).contiguous()


def _ssim(pred, target) -> float:
    try:
        from pytorch_msssim import ssim as _ssim_fn  # type: ignore[import-not-found]

        return float(
            _ssim_fn(
                pred.unsqueeze(0).float().clamp(0.0, 1.0),
                target.unsqueeze(0).float().clamp(0.0, 1.0),
                data_range=1.0,
                size_average=True,
            ).item()
        )
    except Exception:
        from oss.valuation.metrics import ssim as _fallback_ssim

        return float(
            _fallback_ssim(
                pred.unsqueeze(0).float().clamp(0.0, 1.0),
                target.unsqueeze(0).float().clamp(0.0, 1.0),
            ).item()
        )


class _LPIPSMetric:
    def __init__(self, device: str) -> None:
        self.fn = None
        if os.environ.get("OSS_V6_HELD_OUT_LPIPS_FALLBACK", "").strip() == "1":
            return
        try:
            import lpips  # type: ignore[import-not-found]

            self.fn = lpips.LPIPS(net="vgg", verbose=False).to(device)
            self.fn.train(False)
            for p in self.fn.parameters():
                p.requires_grad_(False)
        except Exception as e:
            print(f"WARN: LPIPS-VGG unavailable ({e}); using L1 proxy for LPIPS columns")

    def __call__(self, pred, target) -> float:
        p = pred.unsqueeze(0).float().clamp(0.0, 1.0)
        t = target.unsqueeze(0).float().clamp(0.0, 1.0)
        if self.fn is None:
            return float((p - t).abs().mean().item())
        return float(self.fn(p * 2.0 - 1.0, t * 2.0 - 1.0).float().item())


class _ExplicitPairDataset:
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


def _validate_manifest_config(manifest: Mapping[str, Any], *, scale: float) -> None:
    manifest_scale = float(manifest["lr_scale"])
    if abs(manifest_scale - float(scale)) > 1e-9:
        raise ValueError(f"manifest lr_scale={manifest_scale} does not match scale={scale}")
    manifest_lr = dict(manifest.get("lr_synth_args", {}))
    if manifest_lr != DEFAULT_LR_SYNTH_ARGS:
        raise ValueError(
            "manifest lr_synth_args mismatch: "
            f"manifest={manifest_lr}, expected={DEFAULT_LR_SYNTH_ARGS}"
        )


def _remap_manifest_trajectories(manifest: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Map Windows-authored trajectory dirs onto the caller's dataset root.

    Frozen v5 manifests store absolute trajectory paths from the training host.
    ``manifest_to_pairs`` compares those strings against the local dataset's
    ``trajectory_key()``, so a Mac-side eval needs the same env/level/traj under
    its own ``--tartanair-root``.
    """

    out = dict(manifest)
    pairs = []
    for entry in manifest["pairs"]:
        e = dict(entry)
        parts = [p for p in str(e["trajectory"]).replace("\\", "/").split("/") if p]
        if len(parts) >= 3:
            e["trajectory"] = str(root / parts[-3] / parts[-2] / parts[-1])
        pairs.append(e)
    out["pairs"] = pairs
    return out


def _build_manifest_base_dataset(kind: str, root: Path, *, scale: float) -> Any:
    from oss.gaussian.data import TartanAirGaussianDataset
    from oss.gaussian.data.lr_synthesis import EngineAliasedLRSynth
    from oss.sr.temporal import adapt_tartanair

    lr_synth = EngineAliasedLRSynth(scale=scale, **DEFAULT_LR_SYNTH_ARGS)
    if kind == "tartanair":
        return adapt_tartanair(
            TartanAirGaussianDataset(root=root, scale=scale, lr_synth=lr_synth)
        )
    raise ValueError(f"unsupported manifest dataset_kind={kind!r}; v6 held-out expects TartanAir")


def _build_manifest_loader(
    manifest_path: Path,
    tartanair_root: Path,
    batch_size: int,
    *,
    scale: float,
):
    from torch.utils.data import DataLoader
    from oss.sr.temporal import default_collate_pair
    from oss.sr.temporal.held_out_manifest import load_manifest, manifest_to_pairs

    manifest = load_manifest(manifest_path)
    kind = str(manifest.get("dataset_kind", "tartanair"))
    if kind != "tartanair":
        raise ValueError(f"{manifest_path} is {kind!r}; v6 oldtown eval requires TartanAir")
    _validate_manifest_config(manifest, scale=scale)
    base = _build_manifest_base_dataset(kind, tartanair_root, scale=scale)
    manifest = _remap_manifest_trajectories(manifest, tartanair_root)
    pairs = manifest_to_pairs(manifest, base)
    ds = _ExplicitPairDataset(base, pairs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=default_collate_pair,
        drop_last=False,
    )


def _eval_loader(loader, *, model, lpips_metric, device: str) -> dict[str, list[float]]:
    import torch
    import torch.nn.functional as F

    model_psnr: list[float] = []
    bicubic_psnr: list[float] = []
    model_lpips: list[float] = []
    bicubic_lpips: list[float] = []
    model_ssim: list[float] = []
    bicubic_ssim: list[float] = []

    with torch.no_grad():
        for batch in loader:
            t_lr = batch["t_lr"].to(device)
            t_depth = batch["t_depth"].to(device)
            t_motion = batch["t_motion"].to(device)
            t_normals = batch["t_normals"].to(device)

            p_lr = batch["tp1_lr"].to(device)
            p_depth = batch["tp1_depth"].to(device)
            p_normals = batch["tp1_normals"].to(device)
            p_gt = batch["tp1_gt_hr"].to(device)

            model.reset_state(device=torch.device(device))
            first_motion = t_motion.new_zeros(t_motion.shape)
            x_t = _make_9ch(t_lr, t_depth, first_motion, t_normals)
            _ = model(x_t, motion_lr=None, frame_index=0)

            # The manifest's ``t_motion`` is the forward flow t -> t+1; v6
            # training shifts motion this same way for frame 1.
            x_p = _make_9ch(p_lr, p_depth, t_motion, p_normals)
            pred = model(x_p, motion_lr=t_motion, frame_index=1).clamp(0.0, 1.0)
            bic = F.interpolate(
                p_lr,
                size=p_gt.shape[-2:],
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0.0, 1.0)

            for b_idx in range(p_gt.shape[0]):
                m = pred[b_idx]
                b = bic[b_idx]
                gt = p_gt[b_idx]
                model_psnr.append(_psnr(m, gt))
                bicubic_psnr.append(_psnr(b, gt))
                model_lpips.append(lpips_metric(m, gt))
                bicubic_lpips.append(lpips_metric(b, gt))
                model_ssim.append(_ssim(m, gt))
                bicubic_ssim.append(_ssim(b, gt))

    return {
        "model_psnr_per_sample": model_psnr,
        "bicubic_psnr_per_sample": bicubic_psnr,
        "model_lpips_per_sample": model_lpips,
        "bicubic_lpips_per_sample": bicubic_lpips,
        "model_ssim_per_sample": model_ssim,
        "bicubic_ssim_per_sample": bicubic_ssim,
    }


def _score_row(*, ckpt: Path, manifest: Path, result: dict[str, list[float]]) -> dict[str, Any]:
    model_psnr = result["model_psnr_per_sample"]
    bic_psnr = result["bicubic_psnr_per_sample"]
    model_lpips = result["model_lpips_per_sample"]
    bic_lpips = result["bicubic_lpips_per_sample"]
    model_ssim = result["model_ssim_per_sample"]
    bic_ssim = result["bicubic_ssim_per_sample"]
    return {
        "step": _step_from_ckpt(ckpt),
        "ckpt": str(ckpt),
        "manifest": str(manifest),
        "n_samples": len(model_psnr),
        "model_psnr_mean": _mean(model_psnr),
        "bicubic_psnr_mean": _mean(bic_psnr),
        "per_frame_psnr": model_psnr,
        "per_frame_bicubic_psnr": bic_psnr,
        "model_psnr_per_sample": model_psnr,
        "bicubic_psnr_per_sample": bic_psnr,
        "model_beats_bicubic_count": sum(1 for m, b in zip(model_psnr, bic_psnr) if m > b),
        "model_lpips_mean": _mean(model_lpips),
        "bicubic_lpips_mean": _mean(bic_lpips),
        "per_frame_lpips": model_lpips,
        "per_frame_bicubic_lpips": bic_lpips,
        "model_lpips_per_sample": model_lpips,
        "bicubic_lpips_per_sample": bic_lpips,
        "model_beats_bicubic_lpips_count": sum(
            1 for m, b in zip(model_lpips, bic_lpips) if m < b
        ),
        "model_ssim_mean": _mean(model_ssim),
        "bicubic_ssim_mean": _mean(bic_ssim),
        "model_ssim_per_sample": model_ssim,
        "bicubic_ssim_per_sample": bic_ssim,
    }


def _read_score_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [dict(row) for row in data]


def _write_score_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(rows, f, indent=2)
    tmp.replace(path)


def _try_scp_missing_file(local: Path, remote_path: str) -> None:
    if local.exists():
        return
    host = os.environ.get("OSS_TRAIN_HOST") or os.environ.get("REMOTE_HOST")
    if not host:
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{host}:{remote_path}"
    try:
        subprocess.run(["scp", "-B", "-p", "-q", remote, str(local)], check=False, timeout=60)
    except Exception:
        return


def _resolve_manifest(explicit: Path | None, output_dir: Path, ckpt: Path) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        if str(explicit).startswith("E:"):
            local = Path("/tmp/oss-runs") / explicit.name
            _try_scp_missing_file(local, str(explicit))
            if local.exists():
                return local
        raise FileNotFoundError(f"manifest not found: {explicit}")

    candidates = [
        output_dir / "held_out_manifest.json",
        output_dir / "v5_held_out_manifest.json",
        DEFAULT_MANIFEST,
        Path("/tmp/oss-runs/v5_held_out_manifest.json"),
        _REPO_ROOT / "docs/superpowers/experiments/v5_held_out_manifest.json",
    ]
    _try_scp_missing_file(Path("/tmp/oss-runs/v5_held_out_manifest.json"), str(DEFAULT_MANIFEST))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "no TartanAir held-out manifest found; pass --manifest or sync "
        "E:/checkpoints/v5_held_out_manifest.json to /tmp/oss-runs/"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True, help="v6 run dir for score_log.json")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Specific v6 checkpoint. Defaults to latest in output-dir.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"Frozen held-out manifest. Default: {DEFAULT_MANIFEST}",
    )
    p.add_argument("--tartanair-root", type=Path, default=DEFAULT_TARTANAIR_ROOT)
    p.add_argument("--device", type=str, default="cpu", help="Default cpu to avoid GPU contention")
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to existing score_log.json instead of overwriting",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ckpt = args.ckpt or _latest_ckpt(args.output_dir)
    if ckpt is None:
        print(f"FAIL: no step-*.pt checkpoint found in {args.output_dir}")
        return 1
    if not ckpt.is_file():
        print(f"FAIL: checkpoint not found: {ckpt}")
        return 1

    try:
        manifest = _resolve_manifest(args.manifest, args.output_dir, ckpt)
        loader = _build_manifest_loader(
            manifest,
            args.tartanair_root,
            args.batch_size,
            scale=args.scale,
        )
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    print(f"Loading v6 checkpoint: {ckpt}")
    model = _load_v6_model(ckpt, args.device)
    lpips_metric = _LPIPSMetric(args.device)

    print(f"Evaluating {len(loader.dataset)} held-out pairs from {manifest}")
    result = _eval_loader(loader, model=model, lpips_metric=lpips_metric, device=args.device)
    row = _score_row(ckpt=ckpt, manifest=manifest, result=result)

    score_path = args.output_dir / "score_log.json"
    rows = _read_score_log(score_path) if args.append else []
    rows.append(row)
    _write_score_log(score_path, rows)

    print(
        "held-out: "
        f"step={row['step']} "
        f"model_psnr={row['model_psnr_mean']:.3f} "
        f"bicubic_psnr={row['bicubic_psnr_mean']:.3f} "
        f"model_lpips={row['model_lpips_mean']:.4f} "
        f"bicubic_lpips={row['bicubic_lpips_mean']:.4f} "
        f"beats={row['model_beats_bicubic_count']}/{row['n_samples']}"
    )
    print(f"wrote {score_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
