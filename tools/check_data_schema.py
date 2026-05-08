#!/usr/bin/env python3
"""Validate the public dashboard data.json contract."""

import argparse
import json
import math
from pathlib import Path
import sys


SCHEMA_VERSION = "2026-05-07"


def type_name(value: object) -> str:
    return type(value).__name__


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_utc_iso8601(value: str) -> bool:
    if not value.endswith("Z") or "T" not in value:
        return False
    date, time = value[:-1].split("T", 1)
    date_parts = date.split("-")
    time_parts = time.split(":")
    if len(date_parts) != 3 or len(time_parts) < 3:
        return False
    if not all(part.isdigit() for part in date_parts):
        return False
    sec = time_parts[2]
    if "." in sec:
        sec = sec.split(".", 1)[0]
    return time_parts[0].isdigit() and time_parts[1].isdigit() and sec.isdigit()


def add_error(errors: list[str], path: str, expected: str, value: object) -> None:
    errors.append(f"{path}: expected {expected}, got {type_name(value)}")


def require_key(obj: dict[str, object], key: str, path: str, errors: list[str]) -> bool:
    if key in obj:
        return True
    errors.append(f"{path}.{key}: missing required key")
    return False


def require_str(obj: dict[str, object], key: str, path: str, errors: list[str]) -> None:
    if require_key(obj, key, path, errors) and not isinstance(obj[key], str):
        add_error(errors, f"{path}.{key}", "str", obj[key])


def require_bool(obj: dict[str, object], key: str, path: str, errors: list[str]) -> None:
    if require_key(obj, key, path, errors) and not isinstance(obj[key], bool):
        add_error(errors, f"{path}.{key}", "bool", obj[key])


def require_optional_str(obj: dict[str, object], key: str, path: str, errors: list[str]) -> None:
    if not require_key(obj, key, path, errors):
        return
    value = obj[key]
    if value is not None and not isinstance(value, str):
        add_error(errors, f"{path}.{key}", "str | null", value)


def require_int_ge_zero(obj: dict[str, object], key: str, path: str, errors: list[str]) -> None:
    if not require_key(obj, key, path, errors):
        return
    value = obj[key]
    if not is_int(value):
        add_error(errors, f"{path}.{key}", "int", value)
    elif value < 0:
        errors.append(f"{path}.{key}: expected int >= 0, got {value}")


def require_optional_int(obj: dict[str, object], key: str, path: str, errors: list[str]) -> None:
    if not require_key(obj, key, path, errors):
        return
    value = obj[key]
    if value is not None and not is_int(value):
        add_error(errors, f"{path}.{key}", "int | null", value)


def require_optional_object(obj: dict[str, object], key: str, path: str, errors: list[str]) -> None:
    if not require_key(obj, key, path, errors):
        return
    value = obj[key]
    if value is not None and not isinstance(value, dict):
        add_error(errors, f"{path}.{key}", "object | null", value)


def require_object(obj: dict[str, object], key: str, path: str, errors: list[str]) -> None:
    if require_key(obj, key, path, errors) and not isinstance(obj[key], dict):
        add_error(errors, f"{path}.{key}", "object", obj[key])


def require_object_list(obj: dict[str, object], key: str, path: str, errors: list[str]) -> None:
    if not require_key(obj, key, path, errors):
        return
    value = obj[key]
    if not isinstance(value, list):
        add_error(errors, f"{path}.{key}", "list[object]", value)
        return
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            add_error(errors, f"{path}.{key}[{index}]", "object", item)


def require_str_list(obj: dict[str, object], key: str, path: str, errors: list[str]) -> None:
    if not require_key(obj, key, path, errors):
        return
    value = obj[key]
    if not isinstance(value, list):
        add_error(errors, f"{path}.{key}", "list[str]", value)
        return
    for index, item in enumerate(value):
        item_path = f"{path}.{key}[{index}]"
        if not isinstance(item, str):
            add_error(errors, item_path, "str", item)
        elif key == "viz_pngs" and Path(item).is_absolute():
            errors.append(f"{item_path}: expected relative path, got absolute path")


def require_finite_float(
    obj: dict[str, object],
    key: str,
    path: str,
    errors: list[str],
    *,
    gt: float | None = None,
    ge: float | None = None,
    le: float | None = None,
) -> None:
    if not require_key(obj, key, path, errors):
        return
    value = obj[key]
    value_path = f"{path}.{key}"
    if not is_finite_number(value):
        add_error(errors, value_path, "finite float", value)
        return
    number = float(value)
    if gt is not None and not number > gt:
        errors.append(f"{value_path}: expected > {gt}, got {number}")
    if ge is not None and not number >= ge:
        errors.append(f"{value_path}: expected >= {ge}, got {number}")
    if le is not None and not number <= le:
        errors.append(f"{value_path}: expected <= {le}, got {number}")


def validate_float_list(value: object, path: str, errors: list[str]) -> int | None:
    if not isinstance(value, list):
        add_error(errors, path, "list[float]", value)
        return None
    for index, item in enumerate(value):
        if not is_finite_number(item):
            add_error(errors, f"{path}[{index}]", "finite float", item)
    return len(value)


def validate_per_frame(value: object, path: str, errors: list[str]) -> int | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        add_error(errors, path, "object | null", value)
        return None
    lengths: list[int] = []
    for key in ("psnr", "lpips", "delta_psnr_vs_bicubic", "delta_lpips_vs_bicubic"):
        if require_key(value, key, path, errors):
            length = validate_float_list(value[key], f"{path}.{key}", errors)
            if length is not None:
                lengths.append(length)
    if lengths and any(length != lengths[0] for length in lengths):
        errors.append(f"{path}: expected all arrays to have equal length, got {lengths}")
    return lengths[0] if lengths else None


def validate_stats(value: object, path: str, errors: list[str], frame_count: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        add_error(errors, path, "object | null", value)
        return
    for key in ("psnr_std", "psnr_iqr", "lpips_std", "lpips_iqr"):
        require_finite_float(value, key, path, errors, ge=0.0)
    if not require_key(value, "beats_bicubic_count", path, errors):
        pass
    elif not is_int(value["beats_bicubic_count"]):
        add_error(errors, f"{path}.beats_bicubic_count", "int", value["beats_bicubic_count"])
    elif value["beats_bicubic_count"] < 0:
        errors.append(f"{path}.beats_bicubic_count: expected int >= 0, got {value['beats_bicubic_count']}")
    elif frame_count is not None and value["beats_bicubic_count"] > frame_count:
        errors.append(
            f"{path}.beats_bicubic_count: expected <= frame_count {frame_count}, "
            f"got {value['beats_bicubic_count']}"
        )
    require_finite_float(value, "beats_bicubic_wilson95_lo", path, errors, ge=0.0, le=1.0)
    require_finite_float(value, "beats_bicubic_wilson95_hi", path, errors, ge=0.0, le=1.0)
    lo = value.get("beats_bicubic_wilson95_lo")
    hi = value.get("beats_bicubic_wilson95_hi")
    if is_finite_number(lo) and is_finite_number(hi) and float(lo) > float(hi):
        errors.append(f"{path}: expected wilson95_lo <= wilson95_hi")


def validate_score_row(row: object, path: str, errors: list[str], warnings: list[str]) -> None:
    if not isinstance(row, dict):
        add_error(errors, path, "object", row)
        return
    if "per_frame" not in row:
        errors.append(f"{path}.per_frame: missing required key")
        frame_count = None
    else:
        frame_count = validate_per_frame(row["per_frame"], f"{path}.per_frame", errors)
        if row["per_frame"] is None and row.get("ckpt") is not None:
            warnings.append(f"WARNING {path}.per_frame: null on checkpoint-backed score row")
    if "stats" not in row:
        errors.append(f"{path}.stats: missing required key")
    else:
        validate_stats(row["stats"], f"{path}.stats", errors, frame_count)
        if row["stats"] is None and row.get("ckpt") is not None:
            warnings.append(f"WARNING {path}.stats: null on checkpoint-backed score row")


def validate_model(model: object, index: int, errors: list[str]) -> None:
    path = f"models[{index}]"
    if not isinstance(model, dict):
        add_error(errors, path, "object", model)
        return
    require_str(model, "id", path, errors)
    if isinstance(model.get("id"), str) and not model["id"]:
        errors.append(f"{path}.id: expected non-empty str")
    require_str(model, "label", path, errors)
    require_optional_str(model, "run_name", path, errors)
    require_optional_int(model, "step", path, errors)
    require_finite_float(model, "psnr_mean", path, errors, gt=0.0)
    require_finite_float(model, "lpips_mean", path, errors, ge=0.0, le=1.0)
    require_bool(model, "active", path, errors)


def validate_run(run: object, index: int, errors: list[str], warnings: list[str]) -> None:
    path = f"runs[{index}]"
    if not isinstance(run, dict):
        add_error(errors, path, "object", run)
        return

    require_str(run, "name", path, errors)
    require_str(run, "label", path, errors)
    require_bool(run, "active", path, errors)
    require_int_ge_zero(run, "latest_step", path, errors)
    require_optional_int(run, "max_target_steps", path, errors)
    require_object(run, "latest_metrics", path, errors)
    require_object(run, "history", path, errors)
    require_object_list(run, "loss_curve", path, errors)
    require_object_list(run, "score_log", path, errors)
    require_str_list(run, "viz_pngs", path, errors)
    require_str_list(run, "viz_columns", path, errors)
    require_object_list(run, "cross_version_points", path, errors)
    require_optional_object(run, "gpu_status", path, errors)

    if run.get("active") is True and isinstance(run.get("loss_curve"), list) and not run["loss_curve"]:
        warnings.append(f"WARNING {path}.loss_curve: active run has no rows")
    if isinstance(run.get("score_log"), list):
        for row_index, row in enumerate(run["score_log"]):
            validate_score_row(row, f"{path}.score_log[{row_index}]", errors, warnings)


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate(data: object) -> tuple[int, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        return 1, ["root: expected object, got " + type_name(data)], warnings

    if "schema_version" not in data:
        errors.append("schema_version: missing required key")
    elif not isinstance(data["schema_version"], str):
        add_error(errors, "schema_version", "str", data["schema_version"])
    elif data["schema_version"] != SCHEMA_VERSION:
        return 2, [f"schema_version: expected {SCHEMA_VERSION}, got {data['schema_version']}"], warnings

    if "generated_at" not in data:
        errors.append("generated_at: missing required key")
    elif not isinstance(data["generated_at"], str):
        add_error(errors, "generated_at", "str", data["generated_at"])
    elif not is_utc_iso8601(data["generated_at"]):
        errors.append("generated_at: expected ISO-8601 UTC string ending in Z")

    if "runs" not in data:
        errors.append("runs: missing required key")
    elif not isinstance(data["runs"], list):
        add_error(errors, "runs", "list[Run]", data["runs"])
    else:
        for index, run in enumerate(data["runs"]):
            validate_run(run, index, errors, warnings)

    if "models" not in data:
        errors.append("models: required field missing")
    elif not isinstance(data["models"], list):
        add_error(errors, "models", "list[Model]", data["models"])
    else:
        for index, model in enumerate(data["models"]):
            validate_model(model, index, errors)

    return (1 if errors else 0), errors, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = load_json(args.data_json)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{args.data_json}: not readable: {exc}", file=sys.stderr)
        return 3

    code, errors, warnings = validate(data)
    for warning in warnings:
        print(warning, file=sys.stderr)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return code

    print(
        f"OK schema_version={data['schema_version']} "
        f"runs={len(data['runs'])} models={len(data['models'])} "
        f"generated_at={data['generated_at']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
