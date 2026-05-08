#!/usr/bin/env python3
"""Validate the public dashboard data.json contract."""

import argparse
import json
from pathlib import Path
import sys


SCHEMA_VERSION = "2026-05-07"


def type_name(value: object) -> str:
    return type(value).__name__


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
        f"runs={len(data['runs'])} generated_at={data['generated_at']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
