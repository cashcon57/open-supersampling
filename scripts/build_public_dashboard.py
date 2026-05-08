#!/usr/bin/env python3
"""Build the static public training dashboard.

Input lives under dashboard-public/runs/<run-name>/ and is restored from the
gh-pages branch before the GitHub Action fetches fresh training-host files.
The output is a static index.html plus data.json. No third-party Python deps.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = ROOT / "dashboard-public"
RUNS_DIR = PUBLIC_DIR / "runs"
README = ROOT / "README.md"
DATA_JSON = PUBLIC_DIR / "data.json"
INDEX_HTML = PUBLIC_DIR / "index.html"
DASHBOARD_PITCH = (
    "OpenSuperSampling is training a unified game reconstruction pipeline for "
    "super-resolution and frame extrapolation from one temporal model."
)

RUN_CONFIG = {
    "srcnn-v6.1-pico-001": {
        "active": True,
        "default_open": True,
        "history_title": "v6.1 Pico",
        "status": "active, training now",
        "summary": "current v6.1 Pico run from the 3080 Ti training host",
        "note": "",
        "headline": [
            {"label": "Status", "value": "training now", "caption": "live run"},
            {"label": "Latest step", "value_from": "latest_step", "caption": "from metrics.json"},
            {"label": "Loss", "value_from": "loss_total", "caption": "latest total"},
        ],
    },
    "srcnn-v5-pixel-temporal-validated": {
        "active": False,
        "default_open": False,
        "history_title": "v5-pixel-temporal",
        "status": "measured, in-distribution",
        "summary": "See measured v5 result: 25.703 dB, 0.1666 LPIPS, 0.337x temporal ratio, and viz strips.",
        "note": "",
        "headline": [
            {"label": "PSNR", "value": "25.703", "caption": "dB, higher is better"},
            {"label": "LPIPS", "value": "0.1666", "caption": "lower is better"},
            {"label": "Temporal ratio", "value": "0.337x", "caption": "versus v4 baseline"},
        ],
    },
    "srcnn-v5-pixel-temporal-clean-restart-override": {
        "active": False,
        "default_open": False,
        "history_title": "v5 first try",
        "status": "superseded by validated v5",
        "summary": "Earlier v5 pixel-temporal attempt before the validated restart; kept for comparison with its first three viz strips.",
        "note": "Superseded by srcnn-v5-pixel-temporal-validated after the clean restart produced the canonical v5 measurements.",
        "headline": [
            {"label": "Status", "value": "superseded", "caption": "replaced by validated v5"},
            {"label": "Final step", "value_from": "latest_step", "caption": "from metrics.json"},
            {"label": "Viz", "value": "3 strips", "caption": "steps 2K, 4K, 6K"},
        ],
    },
    "srcnn-prod-v4-lpips": {
        "active": False,
        "default_open": False,
        "history_title": "v4",
        "status": "single-frame baseline",
        "summary": "See both v4 held-out contexts: 33.67 dB / 0.270 LPIPS on SRGD, but 11.718 dB / 0.6367 LPIPS after shifting to TartanAir oldtown.",
        "note": "v4 has two held-out contexts: SRGD in-distribution at 33.67 dB / 0.270 LPIPS, and the same model distribution-shifted to TartanAir oldtown at 11.718 dB / 0.6367 LPIPS. TensorRT FP16 latency: 15.6 ms engine-only, 37.6 ms end-to-end.",
        "headline": [
            {"label": "SRGD PSNR", "value": "33.67 dB", "caption": "in-distribution peak"},
            {"label": "TartanAir PSNR", "value": "11.718 dB", "caption": "distribution-shifted"},
            {"label": "TartanAir LPIPS", "value": "0.6367", "caption": "same batch as v5/v6.1"},
        ],
    },
    "srcnn-v6-pico-001": {
        "active": False,
        "default_open": False,
        "history_title": "v6 Pico",
        "status": "superseded by v6.1, 2026-05-07",
        "summary": "Review the stopped v6 run: loss through step 20K and the documented 16-pixel grid artifact.",
        "note": "Stopped after the structural 16-pixel grid artifact was diagnosed; v6.1 supersedes.",
        "headline": [
            {"label": "Status", "value": "superseded", "caption": "v6.1 replaced this run"},
            {"label": "Final step", "value_from": "latest_step", "caption": "expected around 20K"},
            {"label": "Artifact", "value": "16 px grid", "caption": "documented in viz strip"},
        ],
    },
    "srcnn-v6-heavy-001": {
        "active": False,
        "default_open": False,
        "history_title": "v6 Heavy",
        "status": "parked",
        "summary": "Heavy HAT v6 attempt that started on 2026-05-06 and was parked after the v6 grid-artifact diagnosis.",
        "note": "Only the first 68 training metric rows are present in the mirror; no public score or viz strips were produced before it was parked.",
        "headline": [
            {"label": "Status", "value": "parked", "caption": "pre-v6.1 heavy run"},
            {"label": "Final step", "value_from": "latest_step", "caption": "short startup trace"},
            {"label": "Backbone", "value": "HAT-L", "caption": "heavy v6 configuration"},
        ],
    },
}

RUN_ORDER = [
    "srcnn-v6.1-pico-001",
    "srcnn-v6-pico-001",
    "srcnn-v6-heavy-001",
    "srcnn-v5-pixel-temporal-validated",
    "srcnn-v5-pixel-temporal-clean-restart-override",
    "srcnn-prod-v4-lpips",
]
DENY_RE = re.compile(
    r"(aborted|smoke|sanity|test|leak|preflight|paramprobe|init-fix|diag)",
    re.IGNORECASE,
)
VIZ_RE = re.compile(r"step-(\d+)\.png$")
V6_RUN_RE = re.compile(r"srcnn-v(6(?:\.\d+)?)-", re.IGNORECASE)
RUN_MAX_STEPS = {
    "srcnn-v6.1-pico-001": 300_000,
    "srcnn-v6-pico-001": 300_000,
    "srcnn-v6-heavy-001": 300_000,
    "srcnn-v5-pixel-temporal-validated": 80_000,
    "srcnn-v5-pixel-temporal-clean-restart-override": 80_000,
    "srcnn-prod-v4-lpips": 420_000,
}
V4_CROSS_VERSION_POINTS = [
    {
        "name": "v4 (SRGD)",
        "label_full": "v4 single-frame on SRGD held-out (in-distribution)",
        "psnr": 33.67,
        "lpips": 0.270,
        "step": 300000,
        "manifest": "SRGD held-out, 8 frames",
        "color_class": "v4",
        "is_in_distribution": True,
    },
    {
        "name": "v4 (TartanAir, distribution-shifted)",
        "label_full": "v4 single-frame on TartanAir oldtown (NOT in v4's training distribution)",
        "psnr": 11.718,
        "lpips": 0.6367,
        "step": 300000,
        "manifest": "TartanAir oldtown, 64 frames (same as v5/v6.1)",
        "color_class": "v4",
        "is_in_distribution": False,
    },
]
REFERENCE_MODEL_IDS = ("bicubic", "fsr3", "xess1", "dlss4")
REFERENCE_ID_LABELS = {
    "bicubic": "Bicubic",
    "fsr3": "FSR 3",
    "xess1": "XeSS 1.x",
    "dlss4": "DLSS 4",
}
REFERENCE_OUTPUT_IDS = {
    "bicubic": "bicubic",
    "fsr3": "fsr-3",
    "xess1": "xess-1.x",
    "dlss4": "dlss-4",
}


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def configure_paths(runs_dir: Path, out_dir: Path) -> None:
    global PUBLIC_DIR, RUNS_DIR, DATA_JSON, INDEX_HTML
    PUBLIC_DIR = out_dir
    RUNS_DIR = runs_dir
    DATA_JSON = PUBLIC_DIR / "data.json"
    INDEX_HTML = PUBLIC_DIR / "index.html"


def v6_revision_from_run_name(name: str) -> str | None:
    match = V6_RUN_RE.match(name)
    return f"v{match.group(1)}" if match else None


def label_for_run(name: str, config: dict[str, Any]) -> str:
    v6_revision = v6_revision_from_run_name(name)
    if v6_revision:
        if "heavy" in name.lower():
            return f"{v6_revision} Heavy (parked)"
        if config.get("active"):
            return f"{v6_revision} Pico (active)"
        return f"{v6_revision} Pico (superseded)"
    if name == "srcnn-v5-pixel-temporal-validated":
        return "v5-pixel-temporal (measured)"
    if name == "srcnn-v5-pixel-temporal-clean-restart-override":
        return "v5 first try (superseded)"
    if name.startswith("srcnn-prod-v4") or name.startswith("srcnn-v4"):
        return "v4 (single-frame baseline)"
    return str(config.get("history_title") or name)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array, a dict with train rows, or JSONL rows."""

    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("train") or payload.get("rows") or payload.get("metrics")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        return [payload]

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Ignore torn final JSONL writes from an interrupted rsync.
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def primitive_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    return None


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def numeric_list(row: dict[str, Any], *keys: str) -> list[float] | None:
    for key in keys:
        value = row.get(key)
        if not isinstance(value, list):
            continue
        numbers: list[float] = []
        for item in value:
            number = finite_float(item)
            if number is None:
                return None
            numbers.append(number)
        return numbers
    return None


def percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def stddev(values: list[float]) -> float | None:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def iqr(values: list[float]) -> float | None:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    q1 = percentile(sorted_values, 0.25)
    q3 = percentile(sorted_values, 0.75)
    if q1 is None or q3 is None:
        return None
    return q3 - q1


def wilson95(success: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    phat = success / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    spread = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def per_frame_payload(row: dict[str, Any]) -> tuple[dict[str, list[float]] | None, dict[str, Any] | None]:
    psnr = numeric_list(row, "per_frame_psnr", "model_psnr_per_sample", "psnr_per_sample")
    lpips = numeric_list(row, "per_frame_lpips", "model_lpips_per_sample", "lpips_per_sample")
    bicubic_psnr = numeric_list(row, "per_frame_bicubic_psnr", "bicubic_psnr_per_sample")
    bicubic_lpips = numeric_list(row, "per_frame_bicubic_lpips", "bicubic_lpips_per_sample")
    if psnr is None or lpips is None or bicubic_psnr is None or bicubic_lpips is None:
        return None, None
    frame_count = len(psnr)
    if not (len(lpips) == len(bicubic_psnr) == len(bicubic_lpips) == frame_count):
        return None, None
    delta_psnr = [model - bicubic for model, bicubic in zip(psnr, bicubic_psnr)]
    delta_lpips = [model - bicubic for model, bicubic in zip(lpips, bicubic_lpips)]
    beats_count = sum(1 for model, bicubic in zip(psnr, bicubic_psnr) if model > bicubic)
    wilson_lo, wilson_hi = wilson95(beats_count, frame_count)
    return (
        {
            "psnr": psnr,
            "lpips": lpips,
            "delta_psnr_vs_bicubic": delta_psnr,
            "delta_lpips_vs_bicubic": delta_lpips,
        },
        {
            "psnr_std": stddev(psnr),
            "psnr_iqr": iqr(psnr),
            "lpips_std": stddev(lpips),
            "lpips_iqr": iqr(lpips),
            "beats_bicubic_count": beats_count,
            "beats_bicubic_wilson95_lo": wilson_lo,
            "beats_bicubic_wilson95_hi": wilson_hi,
        },
    )


def slim_row(row: dict[str, Any], *, enrich_score: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        primitive = primitive_value(value)
        if primitive is not None or value is None:
            out[str(key)] = primitive
    if not enrich_score:
        return out
    per_frame, stats = per_frame_payload(row)
    out["per_frame"] = per_frame
    out["stats"] = stats
    return out


def row_with_derived_metrics(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    # PSNR proxy from any L1ish reconstruction loss key the trainer wrote.
    # v6/v6.1 trainer uses `loss_charbonnier`; v5 trainer wrote `t_l1`;
    # earlier runs sometimes used bare `l1`. All are L1-magnitude on
    # roughly normalized images, so -10*log10(x²) gives a usable trend
    # (biased ~3-5 dB high vs MSE-true PSNR).
    if out.get("psnr_db") is None:
        for key in ("loss_charbonnier", "t_l1", "l1", "tp1_l1"):
            if key in out:
                try:
                    lc = float(out[key]) or 1e-6
                except (TypeError, ValueError):
                    lc = 1e-6
                out["psnr_proxy"] = -10.0 * math.log10(max(lc * lc, 1e-12))
                break
    return out


def step_value(row: dict[str, Any]) -> int:
    raw = row.get("step", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def viz_step(name: str) -> int:
    match = VIZ_RE.match(name)
    return int(match.group(1)) if match else -1


def list_viz_pngs(run_dir: Path) -> list[str]:
    viz_dir = run_dir / "viz"
    if not viz_dir.is_dir():
        return []
    return sorted((path.name for path in viz_dir.glob("step-*.png")), key=viz_step)


def viz_columns_for_run(name: str) -> list[str]:
    """Return the comparison-strip column order used by sr_temporal_inflight_viz."""

    revision = v6_revision_from_run_name(name)
    if revision:
        return [
            "LR-bilinear",
            "bicubic",
            "v5-pixel-temporal",
            revision,
            "GT",
            "|err v5|",
            f"|err {revision}|",
        ]
    if name in {
        "srcnn-v5-pixel-temporal-validated",
        "srcnn-v5-pixel-temporal-clean-restart-override",
    }:
        return ["LR-bilinear", "bicubic", "v4-baseline", "v5-pixel-temporal", "GT", "|err|"]
    if name.startswith("srcnn-prod-v4") or name.startswith("srcnn-v4"):
        return ["LR-bilinear", "bicubic", "v4", "GT", "|err|"]
    return []


def max_steps_for_run(name: str, metrics: list[dict[str, Any]], previous: dict[str, Any] | None) -> int:
    for row in reversed(metrics):
        for key in ("max_steps", "max_step", "target_steps"):
            try:
                value = int(row.get(key, 0))
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                return value
    previous_target = previous.get("max_target_steps") if previous else None
    try:
        if int(previous_target) > 0:
            return int(previous_target)
    except (TypeError, ValueError):
        pass
    return RUN_MAX_STEPS.get(name, 100_000)


def read_gpu_status(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "gpu_status.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    allowed = {
        "captured_at",
        "gpu_name",
        "memory_used_mib",
        "memory_total_mib",
        "memory_used_pct",
        "utilization_pct",
    }
    return {key: primitive_value(payload.get(key)) for key in allowed if key in payload}


def load_previous_runs() -> dict[str, dict[str, Any]]:
    if not DATA_JSON.is_file():
        return {}
    try:
        data = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    runs = data.get("runs")
    if not isinstance(runs, list):
        return {}
    return {str(run.get("name")): run for run in runs if isinstance(run, dict) and run.get("name")}


def build_run(name: str, previous: dict[str, Any] | None) -> dict[str, Any] | None:
    if DENY_RE.search(name) or name not in RUN_CONFIG:
        return None

    config = RUN_CONFIG[name]
    run_dir = RUNS_DIR / name
    metrics = read_rows(run_dir / "metrics.json")
    scores = read_rows(run_dir / "score_log.json")
    viz_pngs = list_viz_pngs(run_dir)

    previous_has_data = bool(
        previous
        and (
            previous.get("latest_step")
            or previous.get("loss_curve")
            or previous.get("score_log")
            or previous.get("viz_pngs")
        )
    )
    if not metrics and not scores and not viz_pngs and previous_has_data:
        cached = dict(previous)
        if isinstance(cached.get("score_log"), list):
            cached["score_log"] = [
                {
                    **row,
                    "per_frame": row.get("per_frame") if isinstance(row, dict) else None,
                    "stats": row.get("stats") if isinstance(row, dict) else None,
                }
                for row in cached["score_log"]
                if isinstance(row, dict)
            ]
        cached["cached"] = True
        cached["label"] = label_for_run(name, config)
        cached["active"] = config["active"]
        cached["history"] = history_for_run(name, config)
        cached["gpu_status"] = read_gpu_status(run_dir) or cached.get("gpu_status")
        cached["viz_columns"] = viz_columns_for_run(name) or cached.get("viz_columns", [])
        cached["max_target_steps"] = max_steps_for_run(name, [], cached)
        cached["cross_version_points"] = cross_version_points_for_run(name)
        return cached

    if not metrics and not scores and not viz_pngs:
        return {
            "name": name,
            "label": label_for_run(name, config),
            "active": config["active"],
            "history": history_for_run(name, config),
            "latest_step": 0,
            "latest_metrics": {},
            "loss_curve": [],
            "score_log": [],
            "viz_pngs": [],
            "viz_columns": viz_columns_for_run(name),
            "max_target_steps": max_steps_for_run(name, [], previous),
            "gpu_status": read_gpu_status(run_dir),
            "cross_version_points": cross_version_points_for_run(name),
        }

    metrics = [row_with_derived_metrics(row) for row in sorted(metrics, key=step_value)]
    latest = slim_row(metrics[-1]) if metrics else {}
    latest_step = step_value(metrics[-1]) if metrics else 0
    viz_columns = viz_columns_for_run(name)

    return {
        "name": name,
        "label": label_for_run(name, config),
        "active": config["active"],
        "history": history_for_run(name, config),
        "latest_step": latest_step,
        "latest_metrics": latest,
        "loss_curve": [slim_row(row) for row in metrics[-1000:]],
        "score_log": [slim_row(row, enrich_score=True) for row in scores],
        "viz_pngs": viz_pngs,
        "viz_columns": viz_columns,
        "max_target_steps": max_steps_for_run(name, metrics, previous),
        "gpu_status": read_gpu_status(run_dir),
        "cross_version_points": cross_version_points_for_run(name),
    }


def cross_version_points_for_run(name: str) -> list[dict[str, Any]]:
    if name == "srcnn-prod-v4-lpips":
        return [dict(point) for point in V4_CROSS_VERSION_POINTS]
    return []


def history_for_run(name: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": config.get("history_title") or label_for_run(name, config),
        "status": config.get("status", ""),
        "summary": config.get("summary", ""),
        "note": config.get("note", ""),
        "default_open": bool(config.get("default_open")),
        "headline": config.get("headline", []),
    }


def score_metric(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = finite_float(row.get(key))
        if value is not None:
            return value
    return None


def format_step_label(step: int | None) -> str:
    if step is None:
        return ""
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}K"
    if step >= 1000:
        return f"{step / 1000:.1f}K".rstrip("0").rstrip(".")
    return str(step)


def model_label_for_run(name: str, step: int | None) -> str:
    step_label = format_step_label(step)
    if name == "srcnn-v6.1-pico-001":
        return f"v6.1 step {step_label}" if step_label else "v6.1"
    if name == "srcnn-v6-pico-001":
        return f"v6 step {step_label}" if step_label else "v6"
    if name == "srcnn-v5-pixel-temporal-validated":
        return f"v5 step {step_label}" if step_label else "v5"
    if name.startswith("srcnn-prod-v4") or name.startswith("srcnn-v4"):
        return f"v4 step {step_label}" if step_label else "v4"
    return f"{name} step {step_label}" if step_label else name


def model_id_for_run(name: str, step: int | None) -> str:
    suffix = f"-step-{step}" if step is not None else ""
    return re.sub(r"[^a-z0-9._-]+", "-", f"{name}{suffix}".lower()).strip("-")


def model_from_score_run(run: dict[str, Any]) -> dict[str, Any] | None:
    score_log = run.get("score_log")
    if not isinstance(score_log, list) or not score_log:
        return None
    latest = max((row for row in score_log if isinstance(row, dict)), key=step_value, default=None)
    if latest is None:
        return None
    psnr = score_metric(latest, "model_psnr_mean", "psnr_mean", "psnr")
    lpips = score_metric(latest, "model_lpips_mean", "lpips_mean", "lpips")
    if psnr is None or lpips is None:
        return None
    step = step_value(latest)
    return {
        "id": model_id_for_run(str(run["name"]), step),
        "label": model_label_for_run(str(run["name"]), step),
        "run_name": str(run["name"]),
        "step": step,
        "psnr_mean": psnr,
        "lpips_mean": lpips,
        "active": bool(run.get("active")),
    }


def model_from_run_config(run: dict[str, Any]) -> dict[str, Any] | None:
    name = str(run.get("name") or "")
    config = RUN_CONFIG.get(name)
    if config is None:
        return None
    values: dict[str, float] = {}
    for item in config.get("headline", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        if label not in {"psnr", "lpips"}:
            continue
        value = finite_float(item.get("value"))
        if value is not None:
            values[label] = value
    if "psnr" not in values or "lpips" not in values:
        return None
    step = RUN_MAX_STEPS.get(name)
    return {
        "id": model_id_for_run(name, step),
        "label": model_label_for_run(name, step),
        "run_name": name,
        "step": step,
        "psnr_mean": values["psnr"],
        "lpips_mean": values["lpips"],
        "active": bool(run.get("active")),
    }


def v4_srgd_model() -> dict[str, Any]:
    point = next(point for point in V4_CROSS_VERSION_POINTS if point.get("is_in_distribution"))
    return {
        "id": "v4-srgd",
        "label": "v4 SRGD",
        "run_name": "srcnn-prod-v4-lpips",
        "step": int(point["step"]),
        "psnr_mean": float(point["psnr"]),
        "lpips_mean": float(point["lpips"]),
        "active": False,
    }


def extract_reference_models() -> list[dict[str, Any]]:
    source = ROOT / "dashboard-public" / "index.html"
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    match = re.search(r"const\s+comparisonReferenceLines\s*=\s*\[(.*?)\];", text, re.S)
    if not match:
        return []
    models: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, str]] = {}
    for raw_obj in re.findall(r"\{[^{}]*\}", match.group(1)):
        id_match = re.search(r'id:\s*"([^"]+)"', raw_obj)
        psnr_match = re.search(r"psnr:\s*([0-9.]+)", raw_obj)
        lpips_match = re.search(r"lpips:\s*([0-9.]+)", raw_obj)
        latest_match = re.search(r"latest:\s*true", raw_obj)
        if not id_match or not psnr_match or not lpips_match or not latest_match:
            continue
        by_id[id_match.group(1)] = {
            "psnr": psnr_match.group(1),
            "lpips": lpips_match.group(1),
        }
    for reference_id in REFERENCE_MODEL_IDS:
        values = by_id.get(reference_id)
        if values is None:
            continue
        models.append(
            {
                "id": REFERENCE_OUTPUT_IDS[reference_id],
                "label": REFERENCE_ID_LABELS[reference_id],
                "run_name": None,
                "step": None,
                "psnr_mean": float(values["psnr"]),
                "lpips_mean": float(values["lpips"]),
                "active": False,
            }
        )
    return models


def build_models(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(model: dict[str, Any] | None) -> None:
        if model is None or model["id"] in seen:
            return
        seen.add(model["id"])
        models.append(model)

    for run in runs:
        add(model_from_score_run(run) or model_from_run_config(run))
    add(v4_srgd_model())
    for model in extract_reference_models():
        add(model)
    return models


def extract_pitch() -> str:
    return DASHBOARD_PITCH


def build_data() -> dict[str, Any]:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    previous = load_previous_runs()
    runs = []
    for name in RUN_ORDER:
        run = build_run(name, previous.get(name))
        if run is not None:
            runs.append(run)

    return {
        "schema_version": "2026-05-07",
        "generated_at": utc_now_iso(),
        "runs": runs,
        "models": build_models(runs),
    }


def write_data(data: dict[str, Any]) -> None:
    DATA_JSON.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_index(pitch: str) -> None:
    if INDEX_HTML.exists() and "__PITCH_HTML__" not in INDEX_HTML.read_text(encoding="utf-8", errors="replace"):
        return
    pitch_html = html.escape(pitch)
    index = HTML_TEMPLATE.replace("__PITCH_HTML__", pitch_html)
    INDEX_HTML.write_text(index, encoding="utf-8")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OSS Training Dashboard</title>
  <link rel="icon" href="data:,">
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = { darkMode: "class" };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
  <style>
    html { color-scheme: dark; }
    body { background: #09090b; }
    details > summary { list-style: none; }
    details > summary::-webkit-details-marker { display: none; }
    details[open] .summary-chevron { transform: rotate(90deg); }
    details[open] .closed-hint { display: none; }
    .viz-strip { scrollbar-width: thin; }
    .chart-wrap { height: 18rem; min-height: 16rem; max-width: 100%; overflow: hidden; }
    .chart-wrap canvas { max-width: 100% !important; }
    .score-scroll { scrollbar-width: thin; }
    .light-mode { background: #f8fafc; color: #18181b; color-scheme: light; }
    .light-mode [data-surface] { background-color: rgba(255, 255, 255, 0.92) !important; border-color: #d4d4d8 !important; }
    .light-mode [data-muted-surface] { background-color: #f4f4f5 !important; border-color: #d4d4d8 !important; }
    .light-mode [data-text] { color: #18181b !important; }
    .light-mode [data-subtext] { color: #3f3f46 !important; }
    .light-mode [data-dim] { color: #71717a !important; }
    @media (max-width: 640px) {
      .chart-wrap { height: 15rem; }
      .viz-frame, .viz-frame img { width: 18rem; }
    }
  </style>
</head>
<body class="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
  <main class="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-5 sm:px-6 lg:px-8">
    <header class="border-b border-zinc-800 pb-5">
      <div class="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div class="max-w-4xl">
          <p class="text-xs font-semibold uppercase tracking-wide text-cyan-300">OpenSuperSampling</p>
          <h1 class="mt-2 text-3xl font-semibold tracking-normal text-zinc-50 sm:text-4xl" data-text>Public training dashboard</h1>
          <p class="mt-3 max-w-3xl text-sm leading-6 text-zinc-300" data-subtext>__PITCH_HTML__</p>
          <div class="mt-4 flex flex-wrap items-center gap-3 text-sm text-zinc-400" data-subtext>
            <a class="font-medium text-cyan-300 hover:text-cyan-200" href="https://github.com/cashcon57/open-supersampling">GitHub repo</a>
            <span class="hidden text-zinc-700 sm:inline">/</span>
            <span id="updated-line">updated just now</span>
            <span id="data-state" class="rounded border border-zinc-700 px-2 py-0.5 text-xs text-zinc-300">loading</span>
          </div>
        </div>
        <button id="theme-toggle" class="inline-flex h-9 w-fit items-center justify-center rounded-md border border-zinc-700 px-3 text-sm text-zinc-200 hover:border-cyan-500 hover:text-white" type="button">Light mode</button>
      </div>
    </header>

    <section class="grid items-start gap-4 lg:grid-cols-[1.4fr_1fr_1fr]" aria-label="At a glance">
      <article class="rounded-md border border-cyan-900/70 bg-cyan-950/20 p-4" data-surface>
        <div class="flex items-center gap-2 text-sm font-medium text-cyan-200">
          <span class="h-2.5 w-2.5 rounded-full bg-emerald-400"></span>
          <span id="hero-active-status">training status loading</span>
        </div>
        <h2 id="hero-active-title" class="mt-2 text-xl font-semibold text-zinc-50" data-text>v6.1 Pico</h2>
        <p id="hero-active-meta" class="mt-1 text-sm text-zinc-300" data-subtext>waiting for metrics.json</p>
      </article>
      <article class="rounded-md border border-zinc-800 bg-zinc-900/60 p-4" data-surface>
        <div class="text-sm uppercase text-zinc-500" data-dim>Latest measured result</div>
        <div class="mt-2 font-mono text-2xl font-semibold text-emerald-300">25.703 dB</div>
        <p class="mt-1 text-sm text-zinc-400" data-subtext>v5-pixel-temporal, LPIPS 0.1666, temporal ratio 0.337x</p>
      </article>
      <article id="hero-gpu" class="rounded-md border border-zinc-800 bg-zinc-900/60 p-4" data-surface>
        <div class="text-sm uppercase text-zinc-500" data-dim>GPU usage</div>
        <div class="mt-2 text-sm text-zinc-400" data-subtext>loading GPU status</div>
      </article>
    </section>

    <section class="flex flex-col gap-3" aria-labelledby="run-history-title">
      <div class="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 id="run-history-title" class="text-xl font-semibold text-zinc-50" data-text>Run history</h2>
          <p class="text-sm text-zinc-400" data-subtext>Active training first, then measured and superseded references.</p>
        </div>
      </div>
      <div id="run-history" class="flex flex-col gap-3"></div>
      <div id="load-error" class="hidden rounded-md border border-red-900 bg-red-950/30 p-4 text-sm text-red-200"></div>
    </section>

    <section class="rounded-md border border-zinc-800 bg-zinc-900/60 p-5" data-surface>
      <h2 class="text-xl font-semibold text-zinc-50" data-text>Architecture memo</h2>
      <p class="mt-2 max-w-3xl text-sm leading-6 text-zinc-300" data-subtext>The canonical v6 design is the source of truth for the persistent Gaussian canvas, covariance-resampled rasterizer, cross-attention fusion path, and OSS-FX frame extrapolation plan.</p>
      <a class="mt-4 inline-flex rounded-md border border-cyan-500 px-3 py-2 text-sm font-medium text-cyan-300 hover:bg-cyan-500 hover:text-zinc-950" href="https://github.com/cashcon57/open-supersampling/blob/main/docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md">Open canonical memo</a>
    </section>
  </main>

<script>
"use strict";

const colors = {
  total: "#38bdf8",
  charbonnier: "#f59e0b",
  lpips: "#fb7185",
  psnr: "#34d399",
  bicubic: "#94a3b8",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
}

function fmtNumber(value, digits = 4) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
  const num = Number(value);
  if (Math.abs(num) >= 1000) return num.toLocaleString();
  return num.toFixed(digits).replace(/\.?0+$/, "");
}

function stepFromName(name) {
  const match = /step-(\d+)\.png$/.exec(name || "");
  return match ? Number(match[1]) : 0;
}

function agoText(iso) {
  const ts = Date.parse(iso || "");
  if (!Number.isFinite(ts)) return "updated unknown";
  const seconds = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (seconds < 60) return "updated just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `updated ${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `updated ${hours} hr ago`;
  return `updated ${Math.floor(hours / 24)} days ago`;
}

function runUrl(run, file) {
  return `runs/${encodeURIComponent(run.name)}/viz/${encodeURIComponent(file)}`;
}

function xy(rows, key, ...aliases) {
  return (rows || []).map((row) => {
    let value = row[key];
    for (const alias of aliases) {
      if (value !== undefined && value !== null) break;
      value = row[alias];
    }
    if (value === undefined || value === null) return null;
    return { x: Number(row.step || 0), y: Number(value) };
  }).filter(Boolean);
}

function headlineValue(run, item) {
  if (item.value !== undefined) return item.value;
  if (item.value_from === "latest_step") return Number(run.latest_step || 0).toLocaleString();
  if (item.value_from === "loss_total") {
    const latest = run.latest_metrics || {};
    return fmtNumber(latest.loss_total ?? latest.loss, 5);
  }
  return "--";
}

function setTheme(light) {
  document.documentElement.classList.toggle("dark", !light);
  document.body.classList.toggle("light-mode", light);
  localStorage.setItem("oss-dashboard-theme", light ? "light" : "dark");
  document.getElementById("theme-toggle").textContent = light ? "Dark mode" : "Light mode";
}

function chartOptions({ yTitle = "loss", componentAxis = false } = {}) {
  const scales = {
    x: {
      type: "linear",
      title: { display: true, text: "step", color: "#a1a1aa" },
      grid: { color: "rgba(113,113,122,0.22)" },
      ticks: { color: "#a1a1aa", maxTicksLimit: 6 },
    },
    y: {
      title: { display: true, text: yTitle, color: "#a1a1aa" },
      grid: { color: "rgba(113,113,122,0.22)" },
      ticks: { color: "#a1a1aa", maxTicksLimit: 6 },
    },
  };
  if (componentAxis) {
    scales.y1 = {
      type: "linear",
      position: "right",
      grid: { drawOnChartArea: false },
      ticks: { color: "#a1a1aa", maxTicksLimit: 6 },
      title: { display: true, text: "component loss", color: "#a1a1aa" },
    };
  }
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    parsing: false,
    interaction: { mode: "nearest", intersect: false },
    scales,
    plugins: { legend: { labels: { color: "#d4d4d8", boxWidth: 12 } } },
  };
}

function renderLossChart(canvas, run) {
  const rows = run.loss_curve || [];
  const datasets = [
    { label: "loss_total", yAxisID: "y", data: xy(rows, "loss_total", "loss"), borderColor: colors.total, backgroundColor: colors.total, pointRadius: 0, borderWidth: 2, tension: 0 },
    { label: "loss_charbonnier", yAxisID: "y1", data: xy(rows, "loss_charbonnier", "l1", "t_l1"), borderColor: colors.charbonnier, backgroundColor: colors.charbonnier, pointRadius: 0, borderWidth: 1.5, borderDash: [5, 3], tension: 0 },
    { label: "loss_lpips", yAxisID: "y1", data: xy(rows, "loss_lpips", "t_lpips"), borderColor: colors.lpips, backgroundColor: colors.lpips, pointRadius: 0, borderWidth: 1.5, tension: 0 },
  ].filter((dataset) => dataset.data.length);
  if (!datasets.length) return null;
  return new Chart(canvas, { type: "line", data: { datasets }, options: chartOptions({ yTitle: "total loss", componentAxis: true }) });
}

function renderScoreTable(host, run) {
  const rows = (run.score_log || []).slice(-6).reverse();
  if (!rows.length) {
    host.innerHTML = `<p class="rounded border border-zinc-800 bg-zinc-950/40 px-4 py-5 text-sm text-zinc-400" data-muted-surface data-subtext>No held-out score_log.json rows published for this run.</p>`;
    return;
  }
  const keys = Array.from(new Set(rows.flatMap((row) => Object.keys(row))))
    .filter((key) => /step|psnr|lpips|temporal|ratio|latency/i.test(key))
    .slice(0, 6);
  const head = keys.map((key) => `<th class="px-3 py-2 text-left font-medium">${escapeHtml(key)}</th>`).join("");
  const body = rows.map((row) => `<tr class="border-t border-zinc-800">${keys.map((key) => `<td class="px-3 py-2 font-mono">${escapeHtml(fmtNumber(row[key], 4))}</td>`).join("")}</tr>`).join("");
  host.innerHTML = `<div class="score-scroll overflow-x-auto rounded border border-zinc-800"><table class="min-w-full text-sm text-zinc-300"><thead class="bg-zinc-950/50 text-xs uppercase text-zinc-500"><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderVizStrip(host, run, limit) {
  const files = (run && run.viz_pngs ? run.viz_pngs : []).slice(-limit).reverse();
  host.replaceChildren();
  if (!files.length) {
    const empty = document.createElement("div");
    empty.className = "rounded border border-zinc-800 px-4 py-6 text-sm text-zinc-400";
    empty.textContent = "No visualization PNGs published yet.";
    host.appendChild(empty);
    return;
  }
  for (const file of files) {
    const frame = document.createElement("figure");
    frame.className = "viz-frame w-72 shrink-0";
    const img = document.createElement("img");
    img.className = "aspect-video w-72 rounded border border-zinc-800 bg-black object-contain";
    img.loading = "lazy";
    img.alt = `${run.label} ${file}`;
    img.src = runUrl(run, file);
    const caption = document.createElement("figcaption");
    caption.className = "mt-2 text-center font-mono text-xs text-zinc-400";
    caption.textContent = `step ${stepFromName(file).toLocaleString()}`;
    frame.append(img, caption);
    host.appendChild(frame);
  }
}

function gpuPanelHtml(run) {
  const gpu = run.gpu_status;
  if (!gpu) {
    return `<div class="rounded-md border border-zinc-800 bg-zinc-950/40 p-4 opacity-70" data-muted-surface>
      <div class="text-sm uppercase text-zinc-500" data-dim>GPU usage</div>
      <div class="mt-2 text-sm text-zinc-400" data-subtext>GPU stats unavailable</div>
      <p class="mt-1 text-xs text-zinc-500" data-dim>host offline or rsync failed</p>
    </div>`;
  }
  const pct = Math.max(0, Math.min(100, Number(gpu.memory_used_pct || 0)));
  const roundedPct = Math.round(pct);
  return `<div class="rounded-md border border-emerald-900/60 bg-emerald-950/20 p-4" data-surface>
    <div class="flex items-center justify-between gap-3">
      <div class="text-sm uppercase text-zinc-500" data-dim>GPU usage</div>
      <div class="text-sm font-medium text-emerald-300">${escapeHtml(gpu.utilization_pct)}% util</div>
    </div>
    <div class="mt-3 h-2 rounded-full bg-zinc-800">
      <div class="h-2 rounded-full bg-emerald-400" style="width: ${pct}%"></div>
    </div>
    <div class="mt-2 text-sm text-zinc-300" data-subtext>${escapeHtml(gpu.memory_used_mib)} / ${escapeHtml(gpu.memory_total_mib)} MiB (${roundedPct}%)</div>
    <p class="mt-1 text-xs text-zinc-500" data-dim>${escapeHtml(gpu.gpu_name || "GPU")} sampled ${escapeHtml(agoText(gpu.captured_at))}</p>
  </div>`;
}

function renderHero(data) {
  const active = (data.runs || []).find((run) => run.active) || (data.runs || [])[0];
  if (!active) return;
  document.getElementById("hero-active-status").textContent = active.history?.status || "active run";
  document.getElementById("hero-active-title").textContent = active.label || active.name;
  const latest = active.latest_metrics || {};
  document.getElementById("hero-active-meta").textContent = `step ${Number(active.latest_step || 0).toLocaleString()} / loss ${fmtNumber(latest.loss_total ?? latest.loss, 5)}`;
  document.getElementById("hero-gpu").innerHTML = gpuPanelHtml(active).replace("rounded-md border ", "");
}

function renderRun(run, index) {
  const history = run.history || {};
  const activeClass = run.active ? "border-cyan-800 bg-cyan-950/20" : "border-zinc-800 bg-zinc-900/60";
  const open = history.default_open ? "open" : "";
  const headline = (history.headline || []).map((item) => `
    <div class="rounded border border-zinc-800 bg-zinc-950/35 px-3 py-3" data-muted-surface>
      <div class="text-xs uppercase text-zinc-500" data-dim>${escapeHtml(item.label)}</div>
      <div class="mt-1 font-mono text-lg font-semibold text-zinc-100" data-text>${escapeHtml(headlineValue(run, item))}</div>
      <div class="mt-1 text-xs text-zinc-500" data-dim>${escapeHtml(item.caption || "")}</div>
    </div>`).join("");
  const note = history.note ? `<p class="rounded border border-amber-900/60 bg-amber-950/20 px-4 py-3 text-sm text-amber-100">${escapeHtml(history.note)}</p>` : "";
  const gpu = run.active ? gpuPanelHtml(run) : "";
  const chartId = `loss-chart-${index}`;
  const scoreId = `score-table-${index}`;
  const vizId = `viz-strip-${index}`;
  const rows = run.loss_curve || [];
  const chartBody = rows.length
    ? `<div class="chart-wrap mt-3"><canvas id="${chartId}"></canvas></div>`
    : `<p class="mt-3 rounded border border-zinc-800 bg-zinc-950/40 px-4 py-5 text-sm text-zinc-400" data-muted-surface data-subtext>No loss metrics published yet.</p>`;
  return `<details ${open} class="rounded-md border ${activeClass}" data-surface>
    <summary class="flex cursor-pointer items-start gap-3 p-4">
      <span class="summary-chevron mt-1 text-zinc-500 transition-transform">›</span>
      <span class="min-w-0 flex-1">
        <span class="flex flex-wrap items-center gap-2">
          <span class="text-base font-semibold text-zinc-50" data-text>${escapeHtml(run.label || run.name)}</span>
          <span class="rounded border ${run.active ? "border-emerald-700 text-emerald-300" : "border-zinc-700 text-zinc-300"} px-2 py-0.5 text-xs">${escapeHtml(history.status || "")}</span>
        </span>
        <span class="mt-1 block text-sm text-zinc-400" data-subtext>${escapeHtml(history.summary || run.name)}</span>
      </span>
      <span class="hidden text-right text-sm text-zinc-500 sm:block" data-dim>
        <span class="closed-hint">Open details</span>
        <span class="block font-mono">step ${Number(run.latest_step || 0).toLocaleString()}</span>
      </span>
    </summary>
    <div class="grid min-w-0 gap-4 border-t border-zinc-800 p-4 lg:grid-cols-[1.15fr_0.85fr]">
      <div class="flex min-w-0 flex-col gap-4">
        <div class="grid min-w-0 gap-3 sm:grid-cols-3">${headline}</div>
        ${gpu}
        ${note}
        <article class="min-w-0 rounded-md border border-zinc-800 bg-zinc-950/25 p-4" data-muted-surface>
          <div class="flex items-center justify-between gap-3">
            <h3 class="text-base font-semibold text-zinc-50" data-text>Loss curve</h3>
            <span class="text-sm text-zinc-500" data-dim>${rows.length} points</span>
          </div>
          ${chartBody}
        </article>
      </div>
      <div class="flex min-w-0 flex-col gap-4">
        <article class="min-w-0 rounded-md border border-zinc-800 bg-zinc-950/25 p-4" data-muted-surface>
          <h3 class="text-base font-semibold text-zinc-50" data-text>Held-out scores</h3>
          <div id="${scoreId}" class="mt-3"></div>
        </article>
        <article class="min-w-0 rounded-md border border-zinc-800 bg-zinc-950/25 p-4" data-muted-surface>
          <div class="flex items-center justify-between gap-3">
            <h3 class="text-base font-semibold text-zinc-50" data-text>Viz strips</h3>
            <span class="text-sm text-zinc-500" data-dim>${(run.viz_pngs || []).length} images</span>
          </div>
          <div id="${vizId}" class="viz-strip mt-3 flex gap-3 overflow-x-auto pb-2"></div>
        </article>
      </div>
    </div>
  </details>`;
}

function renderRuns(data) {
  const host = document.getElementById("run-history");
  const runs = data.runs || [];
  host.innerHTML = runs.map(renderRun).join("");
  const charts = [];
  runs.forEach((run, index) => {
    const canvas = document.getElementById(`loss-chart-${index}`);
    if (canvas) {
      const chart = renderLossChart(canvas, run);
      if (chart) charts.push(chart);
    }
    renderScoreTable(document.getElementById(`score-table-${index}`), run);
    renderVizStrip(document.getElementById(`viz-strip-${index}`), run, 4);
  });
  host.querySelectorAll("details").forEach((details) => {
    details.addEventListener("toggle", () => {
      if (details.open) window.requestAnimationFrame(() => charts.forEach((chart) => chart.resize()));
    });
  });
}

async function loadDashboard() {
  const state = document.getElementById("data-state");
  try {
    const response = await fetch("data.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    state.textContent = "loaded";
    state.classList.remove("border-zinc-700");
    state.classList.add("border-emerald-700", "text-emerald-300");
    document.getElementById("updated-line").textContent = agoText(data.generated_at);
    renderHero(data);
    renderRuns(data);
    window.setInterval(() => {
      document.getElementById("updated-line").textContent = agoText(data.generated_at);
    }, 30000);
  } catch (error) {
    state.textContent = "data load failed";
    state.classList.add("border-red-700", "text-red-300");
    const errorBox = document.getElementById("load-error");
    errorBox.classList.remove("hidden");
    errorBox.textContent = `Could not load data.json: ${error.message}`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setTheme(localStorage.getItem("oss-dashboard-theme") === "light");
  document.getElementById("theme-toggle").addEventListener("click", () => {
    setTheme(document.documentElement.classList.contains("dark"));
  });
  loadDashboard();
});
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=PUBLIC_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(args.runs_dir, args.out)
    data = build_data()
    write_data(data)
    write_index(extract_pitch())
    try:
        data_path = DATA_JSON.relative_to(ROOT)
        index_path = INDEX_HTML.relative_to(ROOT)
    except ValueError:
        data_path = DATA_JSON
        index_path = INDEX_HTML
    print(f"wrote {data_path} and {index_path}")


if __name__ == "__main__":
    main()
