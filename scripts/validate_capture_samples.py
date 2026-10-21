#!/usr/bin/env python3
"""Validate locally captured OSS training samples.

The validator is intentionally read-only: it reports malformed samples but does
not delete, upload, or rewrite anything. It is suitable for Cyberpunk smoke
runs where ``capture_storage_mode=local`` should leave samples on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT_CANDIDATES = (
    Path(__file__).resolve().parents[1],
    Path.cwd(),
    Path("E:/Temp/oss-capture-src-current"),
)
for candidate in REPO_ROOT_CANDIDATES:
    if (candidate / "oss" / "capture" / "uploader.py").is_file():
        REPO_ROOT = candidate
        break
else:
    REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oss.capture.uploader import ValidationIssue, validate_capture_tree


def _latest_session_dirs(pending_dir: Path, count: int) -> list[Path]:
    sessions: list[tuple[int, Path]] = []
    for session_dir in pending_dir.glob("*/*"):
        if not session_dir.is_dir():
            continue
        newest = 0
        for child in session_dir.iterdir():
            if child.is_file():
                try:
                    newest = max(newest, child.stat().st_mtime_ns)
                except OSError:
                    continue
        if newest:
            sessions.append((newest, session_dir))
    sessions.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in sessions[:count]]


def _storage_mode_counts(pending_dir: Path) -> dict[str, int]:
    counts = {"local": 0, "upload": 0, "missing": 0, "invalid": 0}
    for meta_path in sorted(pending_dir.glob("*/*/*.json")):
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            counts["invalid"] += 1
            continue
        if not isinstance(raw, dict):
            counts["invalid"] += 1
            continue
        mode = raw.get("capture_storage_mode")
        if mode in {"local", "upload"}:
            counts[str(mode)] += 1
        elif mode is None:
            counts["missing"] += 1
        else:
            counts["invalid"] += 1
    return counts


def _storage_mode_counts_for_sessions(session_dirs: Sequence[Path]) -> dict[str, int]:
    counts = {"local": 0, "upload": 0, "missing": 0, "invalid": 0}
    for session_dir in session_dirs:
        for meta_path in sorted(session_dir.glob("*.json")):
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                counts["invalid"] += 1
                continue
            if not isinstance(raw, dict):
                counts["invalid"] += 1
                continue
            mode = raw.get("capture_storage_mode")
            if mode in {"local", "upload"}:
                counts[str(mode)] += 1
            elif mode is None:
                counts["missing"] += 1
            else:
                counts["invalid"] += 1
    return counts


def _capture_file_count_for_sessions(session_dirs: Sequence[Path]) -> int:
    total = 0
    for session_dir in session_dirs:
        total += sum(1 for _ in session_dir.glob("*.exr"))
        total += sum(1 for _ in session_dir.glob("*.exr.raw"))
        total += sum(1 for _ in session_dir.glob("*.json"))
    return total


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pending_dir", type=Path, help="Pending capture root to validate")
    parser.add_argument(
        "--disallow-local",
        action="store_true",
        help="Fail samples whose metadata marks capture_storage_mode='local'.",
    )
    parser.add_argument(
        "--skip-exr",
        action="store_true",
        help="Validate JSON/path/burst metadata only; skip EXR channel and numeric checks.",
    )
    parser.add_argument(
        "--no-strict-bursts",
        action="store_true",
        help="Do not require observed multi-frame burst indices to be contiguous.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable validation report.",
    )
    parser.add_argument(
        "--latest-session-count",
        type=int,
        default=0,
        help="Validate only the N newest capture session directories.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    session_dirs = None
    if args.latest_session_count < 0:
        raise SystemExit("--latest-session-count must be >= 0")
    if args.latest_session_count:
        session_dirs = _latest_session_dirs(args.pending_dir, args.latest_session_count)
    scope_issues: list[ValidationIssue] = []
    scoped_file_count = 0
    if session_dirs is not None:
        scoped_file_count = _capture_file_count_for_sessions(session_dirs)
        if not session_dirs:
            scope_issues.append(
                ValidationIssue(args.pending_dir, "no capture sessions found")
            )
        elif scoped_file_count == 0:
            scope_issues.append(
                ValidationIssue(
                    session_dirs[0],
                    "latest capture session has no sample files",
                )
            )
    issues = validate_capture_tree(
        args.pending_dir,
        allow_local=not args.disallow_local,
        validate_exr=not args.skip_exr,
        strict_bursts=not args.no_strict_bursts,
        session_dirs=session_dirs,
    )
    report = {
        "pending_dir": str(args.pending_dir),
        "validated_session_dirs": [str(path) for path in session_dirs or []],
        "valid": not issues and not scope_issues,
        "issue_count": len(issues) + len(scope_issues),
        "scoped_file_count": scoped_file_count,
        "storage_modes": (
            _storage_mode_counts_for_sessions(session_dirs)
            if session_dirs is not None
            else _storage_mode_counts(args.pending_dir)
        ),
        "issues": [
            {
                "path": str(issue.path),
                "message": issue.message,
            }
            for issue in [*scope_issues, *issues]
        ],
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        all_issues = [*scope_issues, *issues]
        if all_issues:
            print(f"FAIL: {len(all_issues)} capture validation issue(s)")
            for issue in all_issues:
                print(f"{issue.path}: {issue.message}")
        else:
            print(f"OK: capture samples valid under {args.pending_dir}")
    return 1 if issues or scope_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
