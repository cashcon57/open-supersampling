#!/usr/bin/env python3
"""Emit a reproducibility manifest for a training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    for cwd in (Path.cwd(), ROOT):
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        value = proc.stdout.strip()
        if value:
            return value
    return None


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("torch is required to read checkpoint files") from exc

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"{path} did not contain a checkpoint object")
    return ckpt


def torch_versions() -> tuple[str | None, str | None]:
    try:
        import torch
    except ImportError:
        return None, None
    return str(torch.__version__), (str(torch.version.cuda) if torch.version.cuda else None)


def jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is normally present with torch
        np = None  # type: ignore[assignment]
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if torch is not None and isinstance(value, torch.Tensor):
        cpu = value.detach().cpu()
        return {
            "type": "torch.Tensor",
            "dtype": str(cpu.dtype),
            "shape": list(cpu.shape),
            "values": cpu.tolist(),
        }
    if np is not None and isinstance(value, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return str(value)


def checkpoint_step(path: Path, ckpt: dict[str, Any]) -> int | None:
    raw = ckpt.get("step")
    try:
        return int(raw)
    except (TypeError, ValueError):
        pass
    match = __import__("re").search(r"step-(\d+)", path.name)
    return int(match.group(1)) if match else None


def checkpoint_git_sha(ckpt: dict[str, Any]) -> str | None:
    for key in ("git_sha", "commit_sha", "commit", "sha"):
        value = ckpt.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = ckpt.get("meta") or ckpt.get("metadata")
    if isinstance(meta, dict):
        for key in ("git_sha", "commit_sha", "commit", "sha"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return git_head()


def resolve_existing_path(raw: Any, *, ckpt_path: Path) -> Path | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        for item in raw:
            resolved = resolve_existing_path(item, ckpt_path=ckpt_path)
            if resolved is not None:
                return resolved
        return None
    if isinstance(raw, str) and "," in raw:
        for item in raw.split(","):
            resolved = resolve_existing_path(item.strip(), ckpt_path=ckpt_path)
            if resolved is not None:
                return resolved
        return None
    path = Path(str(raw))
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, ROOT / path, ckpt_path.parent / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def manifest_path_from_args(args: dict[str, Any], ckpt_path: Path) -> Path | None:
    preferred = (
        "dataset_manifest",
        "data_manifest",
        "train_manifest",
        "training_manifest",
        "manifest",
        "manifest_path",
        "held_out_manifest",
    )
    for key in preferred:
        path = resolve_existing_path(args.get(key), ckpt_path=ckpt_path)
        if path is not None:
            return path
    for key, value in args.items():
        if "manifest" not in str(key).lower():
            continue
        path = resolve_existing_path(value, ckpt_path=ckpt_path)
        if path is not None:
            return path
    for name in ("train_manifest.json", "dataset_manifest.json", "held_out_manifest.json", "v5_held_out_manifest.json"):
        path = ckpt_path.parent / name
        if path.is_file():
            return path
    return None


def arg_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def args_to_cli(args: dict[str, Any]) -> str:
    parts = ["python", "scripts/sr_train_v6.py"]
    for key in sorted(args):
        value = args[key]
        if value is None or value is False:
            continue
        flag = arg_flag(str(key))
        if value is True:
            parts.append(flag)
        elif isinstance(value, (list, tuple)):
            if value:
                parts.append(flag)
                parts.extend(shlex.quote(str(item)) for item in value)
        else:
            parts.extend([flag, shlex.quote(str(value))])
    return " ".join(parts)


def state_dict_param_count(state: Any) -> int | None:
    if not isinstance(state, dict):
        return None
    total = 0
    found = False
    for value in state.values():
        if hasattr(value, "numel"):
            try:
                total += int(value.numel())
                found = True
            except (TypeError, ValueError):
                continue
    return total if found else None


def model_arch_and_params(ckpt: dict[str, Any], args: dict[str, Any]) -> tuple[str | None, int | None]:
    kind = ckpt.get("kind")
    backbone = args.get("backbone")
    if kind and backbone:
        arch = f"{kind}:{backbone}"
    elif kind:
        arch = str(kind)
    elif backbone:
        arch = f"v6:{backbone}"
    else:
        arch = None

    for key in ("generator", "model", "state_dict", "temporal_model", "sr_model"):
        count = state_dict_param_count(ckpt.get(key))
        if count is not None:
            if arch is None:
                arch = key
            return arch, count
    return arch, None


def emit_manifest(ckpt_path: Path, *, manifest_path: Path | None = None) -> dict[str, Any]:
    ckpt_path = Path(ckpt_path)
    ckpt = load_checkpoint(ckpt_path)
    args = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}
    found_manifest = manifest_path or manifest_path_from_args(args, ckpt_path)
    torch_version, cuda_version = torch_versions()
    arch, param_count = model_arch_and_params(ckpt, args)
    seed = args.get("seed") if isinstance(args, dict) else None

    return {
        "git_sha": checkpoint_git_sha(ckpt),
        "dataset_sha": sha256_file(found_manifest) if found_manifest else None,
        "rng_state": jsonable(ckpt.get("rng") or ckpt.get("rng_state")),
        "cli_invocation": args_to_cli(args) if args else None,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "model_arch": arch,
        "param_count": param_count,
        "timestamp_utc": utc_now_iso(),
        "checkpoint_path": str(ckpt_path),
        "step": checkpoint_step(ckpt_path, ckpt),
        "args": jsonable(args) if args else None,
        "rng_seed": int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else seed,
        "dataset_manifest": str(found_manifest) if found_manifest else None,
}


def build_manifest(ckpt_path: str | Path) -> dict[str, Any]:
    """Backward-compatible import name for dashboard callers/tests."""

    return emit_manifest(Path(ckpt_path))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, default=None, help="Explicit training-data manifest to hash.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = emit_manifest(args.checkpoint, manifest_path=args.manifest)
    json.dump(payload, sys.stdout, indent=2 if args.pretty else None, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
