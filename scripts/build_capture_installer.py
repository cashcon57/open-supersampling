"""Per-game capture-installer config builder.

Generates the **input files** that an MSI compiler (WiX / candle+light)
will bundle into a per-game ``oss_capture_<game>_v<ver>.msi``. The MSI
compilation step itself lives outside this script (it requires Windows
tooling); we just produce the JSON config + installer manifest that the
Codex-side DLL build pipeline consumes.

Usage:

    python scripts/build_capture_installer.py \
        --game cyberpunk-2077 \
        --game-exe-name Cyberpunk2077.exe \
        --proxy-dll-name dxgi.dll \
        --output dist/oss_capture_cyberpunk_v1.0.0/

This writes:

    <output>/config.json           (bundled into %LOCALAPPDATA%\\oss-capture\\)
    <output>/installer_manifest.json   (consumed by WiX/candle build)

The ``install_token`` is a freshly minted UUID4 — the unique key that
identifies this installer build to the ingest server.

Note: this is the **build-time** config, baked into the MSI. At
install-time the user agrees to the consent dialog described in the
design memo §"One-click installer".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from typing import Any, Dict, Optional, Sequence


CAPTURE_API_DEFAULT = "https://capture.oss-supersampling.dev"
CAPTURE_MODES = ("trickle", "lite", "regular", "INSANE")
INSANE_SUPERSAMPLE_GT_CONSENT = (
    "INSANE mode runs an automatic 256-frame supersample ground-truth pass "
    "when the camera settles for \u22651.5 s. This briefly stutters the game "
    "(~250 ms) and is the source of OSS's beyond-DLSS quality data. By "
    "accepting INSANE mode you accept this trade-off."
)

_GAME_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_DLL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}\.dll$")
_EXE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}\.exe$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?$")


def build_config(
    *,
    game_id: str,
    game_exe_name: str,
    proxy_dll_name: str,
    installer_version: str,
    capture_api_base: str,
    install_token: Optional[str] = None,
    suggested_capture_rate_per_min: float = 3.0,
    capture_mode: str = "lite",
) -> Dict[str, Any]:
    """Return the install-time ``config.json`` dict.

    Pure function — no I/O, no UUID side-effect unless ``install_token``
    is None. Useful for unit tests and for scripted batch builds.
    """
    if not _GAME_ID_RE.match(game_id):
        raise ValueError(
            "game_id must match [a-z0-9][a-z0-9._-]{0,63} (lowercase, fs-safe)"
        )
    if not _EXE_NAME_RE.match(game_exe_name):
        raise ValueError(f"game_exe_name {game_exe_name!r} not a valid .exe name")
    if not _DLL_NAME_RE.match(proxy_dll_name):
        raise ValueError(f"proxy_dll_name {proxy_dll_name!r} not a valid .dll name")
    if not _VERSION_RE.match(installer_version):
        raise ValueError(
            f"installer_version {installer_version!r} not semver"
        )
    if capture_mode not in CAPTURE_MODES:
        raise ValueError(f"unknown capture_mode {capture_mode!r}")

    if install_token is None:
        install_token = uuid.uuid4().hex

    return {
        "schema_version": 1,
        "game_id": game_id,
        "game_exe_name": game_exe_name,
        "proxy_dll_name": proxy_dll_name,
        "install_token": install_token,
        "installer_version": installer_version,
        "capture_mode": capture_mode,
        "capture_api_base": capture_api_base.rstrip("/"),
        "endpoints": {
            "ingest": capture_api_base.rstrip("/") + "/ingest",
            "session_start": capture_api_base.rstrip("/") + "/session/start",
            "stats": capture_api_base.rstrip("/") + "/stats",
        },
        "suggested_capture_rate_per_min": float(suggested_capture_rate_per_min),
        # The DLL+uploader pair reads these knobs at runtime (from
        # %LOCALAPPDATA%\oss-capture\config.json).
        "pending_dir_cap_bytes": 2 * 1024 * 1024 * 1024,  # 2 GB hard cap
        "max_frame_bytes": 16 * 1024 * 1024,  # 16 MB hard cap (server-side too)
        "uploader_retry_attempts": 5,
        "uploader_retry_max_seconds": 30 * 60,  # 30 minutes total
        "consent": {
            "mode": capture_mode,
            "standard_disclosure": (
                "By installing, you agree to upload anonymized gameplay frames "
                f"from {game_id} to OSS for AI training. Captures are deleted "
                "from your machine after upload or terminal rejection."
            ),
            "insane_supersample_gt_disclosure": (
                INSANE_SUPERSAMPLE_GT_CONSENT if capture_mode == "INSANE" else ""
            ),
        },
    }


def build_installer_manifest(
    config: Dict[str, Any],
    *,
    output_dir: str,
) -> Dict[str, Any]:
    """Return the manifest the WiX build step consumes.

    Lists the files the MSI must bundle, their target paths, and the
    metadata an uninstaller will need (registry keys, scheduled-task
    name, backup behavior for an existing ``dxgi.dll``).
    """
    game_id = config["game_id"]
    return {
        "schema_version": 1,
        "build_dir": output_dir,
        "config_path": os.path.join(output_dir, "config.json"),
        "files": [
            {
                "src": "oss_capture.dll",
                "dst_token": "GAME_BIN_X64_DIR",
                "dst_name": config["proxy_dll_name"],
                "behavior": "backup-existing-then-replace",
            },
            {
                "src": "oss_capture_uploader.exe",
                "dst_token": "LOCALAPPDATA_OSS_CAPTURE",
                "dst_name": "oss_capture_uploader.exe",
            },
            {
                "src": "config.json",
                "dst_token": "LOCALAPPDATA_OSS_CAPTURE",
                "dst_name": "config.json",
            },
        ],
        "scheduled_task": {
            "name": f"OSS-Capture-Uploader-{game_id}",
            "exe": "oss_capture_uploader.exe",
            "interval_minutes": 10,
        },
        "consent": config["consent"],
        "verify": {
            "expected_game_exe": config["game_exe_name"],
            "expected_relative_path": "bin/x64/",
        },
        "uninstall": {
            "restore_backup": True,
            "remove_pending_dir": True,
            "remove_scheduled_task": True,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_capture_installer",
        description=(
            "Generate per-game capture-installer config + manifest. "
            "MSI compilation is a separate step (WiX/candle)."
        ),
    )
    parser.add_argument("--game", required=True, dest="game_id")
    parser.add_argument(
        "--game-exe-name",
        required=True,
        help="e.g., Cyberpunk2077.exe",
    )
    parser.add_argument(
        "--proxy-dll-name",
        default="dxgi.dll",
        help="Filename to drop our DLL as (default: dxgi.dll).",
    )
    parser.add_argument(
        "--installer-version",
        default="1.0.0",
        help="Semver string baked into the MSI metadata.",
    )
    parser.add_argument(
        "--capture-api-base",
        default=CAPTURE_API_DEFAULT,
    )
    parser.add_argument(
        "--mode",
        choices=list(CAPTURE_MODES),
        default="lite",
        help=(
            "Capture mode preset. Default 'lite' is the 99%% case "
            "(~500 MB/h, v5-temporal-optimized). 'trickle' (~100 MB/h) "
            "for users who don't want to notice it. 'regular' (~2 GB/h) "
            "for material-aware contributors with uncapped fiber. "
            "'INSANE' (~20-50 GB/h) for data-warriors with high-end GPUs "
            "+ uncapped uplink (note: periodic supersample-GT pass briefly "
            "stutters the game when camera is settled)."
        ),
    )
    parser.add_argument(
        "--install-token",
        default=None,
        help="Pin this exact token instead of generating a UUID4 "
        "(useful for repeatable test builds).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. Will be created if missing.",
    )

    args = parser.parse_args(argv)

    try:
        config = build_config(
            game_id=args.game_id,
            game_exe_name=args.game_exe_name,
            proxy_dll_name=args.proxy_dll_name,
            installer_version=args.installer_version,
            capture_api_base=args.capture_api_base,
            install_token=args.install_token,
            capture_mode=args.mode,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.output, exist_ok=True)
    config_path = os.path.join(args.output, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")

    manifest = build_installer_manifest(config, output_dir=args.output)
    manifest_path = os.path.join(args.output, "installer_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {config_path}")
    print(f"wrote {manifest_path}")
    print(f"install_token: {config['install_token']}")
    print(
        "next step: run the WiX/candle build with --manifest "
        f"{manifest_path} (separate Windows-only tooling)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
