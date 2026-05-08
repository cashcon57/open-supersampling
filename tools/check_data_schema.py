#!/usr/bin/env python3
"""Validate the public dashboard data.json contract."""

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys


SCHEMA_VERSION = "2026-05-07"
STATUS_SERVICE_IDS = {"trainer", "watcher", "worker", "r2", "dns"}
STATUS_VALUES = {"healthy", "degraded", "offline"}


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


def is_utc_iso8601_datetime(value: str) -> bool:
    if not value.endswith("Z") or "T" not in value:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


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


def require_non_empty_str(obj: dict[str, object], key: str, path: str, errors: list[str]) -> None:
    if not require_key(obj, key, path, errors):
        return
    value = obj[key]
    if not isinstance(value, str):
        add_error(errors, f"{path}.{key}", "non-empty str", value)
    elif not value.strip():
        errors.append(f"{path}.{key}: expected non-empty str")


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


def validate_event(event: object, path: str, errors: list[str]) -> None:
    if not isinstance(event, dict):
        add_error(errors, path, "Event", event)
        return
    require_int_ge_zero(event, "step", path, errors)
    require_non_empty_str(event, "kind", path, errors)
    require_non_empty_str(event, "label", path, errors)
    for key in ("ts", "detail", "commit", "doc"):
        if key not in event:
            continue
        value = event[key]
        if value is not None and not isinstance(value, str):
            add_error(errors, f"{path}.{key}", "str | null", value)


def validate_cost_projection(projection: object, path: str, errors: list[str]) -> None:
    if not isinstance(projection, dict):
        add_error(errors, path, "object", projection)
        return
    require_finite_float(projection, "gpu_hours_to_dlss4_quality", path, errors, ge=0.0)
    require_finite_float(projection, "usd_at_runpod_rate", path, errors, ge=0.0)


def validate_cost(cost: object, path: str, errors: list[str]) -> None:
    if not isinstance(cost, dict):
        add_error(errors, path, "object", cost)
        return
    require_finite_float(cost, "kwh", path, errors, ge=0.0)
    require_finite_float(cost, "usd", path, errors, ge=0.0)
    require_finite_float(cost, "gpu_hours", path, errors, ge=0.0)
    if require_key(cost, "projections", path, errors):
        projections = cost["projections"]
        if not isinstance(projections, dict):
            add_error(errors, f"{path}.projections", "object", projections)
        else:
            expected = {"B200", "H100", "A100", "4090"}
            seen = set(projections)
            if seen != expected:
                errors.append(f"{path}.projections: expected keys {sorted(expected)}, got {sorted(seen)}")
            for gpu_class, projection in projections.items():
                validate_cost_projection(projection, f"{path}.projections.{gpu_class}", errors)


def validate_gpu_mem_log(value: object, path: str, errors: list[str]) -> None:
    if not isinstance(value, list):
        add_error(errors, path, "list[[unix_ts, mb]]", value)
        return
    timestamps: list[float] = []
    for index, sample in enumerate(value):
        sample_path = f"{path}[{index}]"
        if not isinstance(sample, list):
            add_error(errors, sample_path, "list[unix_ts, mb]", sample)
            continue
        if len(sample) != 2:
            errors.append(f"{sample_path}: expected 2 entries, got {len(sample)}")
            continue
        ts, mb = sample
        if not is_finite_number(ts):
            add_error(errors, f"{sample_path}[0]", "finite unix timestamp", ts)
            continue
        if float(ts) < 0:
            errors.append(f"{sample_path}[0]: expected >= 0, got {ts}")
            continue
        timestamps.append(float(ts))
        if not is_finite_number(mb):
            add_error(errors, f"{sample_path}[1]", "finite MB value", mb)
        elif float(mb) < 0:
            errors.append(f"{sample_path}[1]: expected >= 0, got {mb}")
    if timestamps != sorted(timestamps):
        errors.append(f"{path}: expected samples sorted by unix timestamp")
    if timestamps and max(timestamps) - min(timestamps) > 1800:
        errors.append(f"{path}: expected 30-min sliding window, got {max(timestamps) - min(timestamps):.1f}s")


def validate_repro_manifest_item(item: object, path: str, errors: list[str]) -> None:
    if not isinstance(item, dict):
        add_error(errors, path, "object", item)
        return
    for key in (
        "git_sha",
        "dataset_sha",
        "python_version",
        "torch_version",
        "cuda_version",
        "model_arch",
    ):
        require_optional_str(item, key, path, errors)
    require_optional_str(item, "cli_invocation", path, errors)
    require_optional_int(item, "param_count", path, errors)
    if require_key(item, "rng_state", path, errors):
        rng_state = item["rng_state"]
        if rng_state is not None and not isinstance(rng_state, dict):
            add_error(errors, f"{path}.rng_state", "object | null", rng_state)
    if require_key(item, "timestamp_utc", path, errors):
        value = item["timestamp_utc"]
        if not isinstance(value, str):
            add_error(errors, f"{path}.timestamp_utc", "UTC ISO-8601 str", value)
        elif not is_utc_iso8601_datetime(value):
            errors.append(f"{path}.timestamp_utc: expected ISO-8601 UTC string ending in Z")


def validate_repro_manifest(manifest: object, path: str, errors: list[str]) -> None:
    if not isinstance(manifest, dict):
        add_error(errors, path, "object", manifest)
        return
    for key, item in manifest.items():
        if not isinstance(key, str):
            add_error(errors, f"{path} key", "str", key)
        validate_repro_manifest_item(item, f"{path}.{key}", errors)


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
    if require_key(run, "events", path, errors):
        if not isinstance(run["events"], list):
            add_error(errors, f"{path}.events", "list[Event]", run["events"])
        else:
            for event_index, event in enumerate(run["events"]):
                validate_event(event, f"{path}.events[{event_index}]", errors)
    require_object_list(run, "cross_version_points", path, errors)
    require_optional_object(run, "gpu_status", path, errors)
    if require_key(run, "gpu_mem_log", path, errors):
        validate_gpu_mem_log(run["gpu_mem_log"], f"{path}.gpu_mem_log", errors)
    if require_key(run, "repro_manifest", path, errors):
        validate_repro_manifest(run["repro_manifest"], f"{path}.repro_manifest", errors)
    if require_key(run, "cost", path, errors):
        validate_cost(run["cost"], f"{path}.cost", errors)

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


def validate_status_service(service: object, index: int, errors: list[str]) -> None:
    path = f"services[{index}]"
    if not isinstance(service, dict):
        add_error(errors, path, "object", service)
        return

    for key in ("id", "name", "status", "detail", "tooltip"):
        require_str(service, key, path, errors)

    service_id = service.get("id")
    if isinstance(service_id, str) and service_id not in STATUS_SERVICE_IDS:
        errors.append(f"{path}.id: expected one of {sorted(STATUS_SERVICE_IDS)}, got {service_id}")

    status = service.get("status")
    if isinstance(status, str) and status not in STATUS_VALUES:
        errors.append(f"{path}.status: expected one of {sorted(STATUS_VALUES)}, got {status}")


def validate_status(data: object) -> tuple[int, list[str], list[str]]:
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

    if "checked_at" not in data:
        errors.append("checked_at: missing required key")
    elif not isinstance(data["checked_at"], str):
        add_error(errors, "checked_at", "str", data["checked_at"])
    elif not is_utc_iso8601_datetime(data["checked_at"]):
        errors.append("checked_at: expected ISO-8601 UTC string ending in Z")

    if "services" not in data:
        errors.append("services: missing required key")
    elif not isinstance(data["services"], list):
        add_error(errors, "services", "list[Service]", data["services"])
    else:
        if len(data["services"]) != len(STATUS_SERVICE_IDS):
            errors.append(f"services: expected length {len(STATUS_SERVICE_IDS)}, got {len(data['services'])}")
        seen_ids: list[str] = []
        for index, service in enumerate(data["services"]):
            validate_status_service(service, index, errors)
            if isinstance(service, dict) and isinstance(service.get("id"), str):
                seen_ids.append(service["id"])
        seen_id_set = set(seen_ids)
        if seen_id_set != STATUS_SERVICE_IDS:
            errors.append(f"services: expected ids {sorted(STATUS_SERVICE_IDS)}, got {sorted(seen_id_set)}")
        duplicate_ids = sorted({service_id for service_id in seen_ids if seen_ids.count(service_id) > 1})
        if duplicate_ids:
            errors.append(f"services: duplicate ids {duplicate_ids}")

    return (1 if errors else 0), errors, warnings


def self_test() -> int:
    data = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "2026-05-07T00:00:00Z",
        "runs": [
            {
                "name": "self-test-run",
                "label": "self-test",
                "active": False,
                "latest_step": 1,
                "max_target_steps": None,
                "latest_metrics": {},
                "history": {},
                "loss_curve": [],
                "score_log": [],
                "viz_pngs": [],
                "viz_columns": [],
                "events": [
                    {
                        "step": 1,
                        "kind": "note",
                        "label": "self-test event",
                        "ts": "2026-05-07T00:00:00Z",
                        "detail": None,
                        "commit": "abc1234",
                        "doc": None,
                        "extra": {"preserved": True},
                    }
                ],
                "cross_version_points": [],
                "gpu_status": None,
                "gpu_mem_log": [[1778198400, 7561], [1778198460, 7563]],
                "repro_manifest": {
                    "1": {
                        "git_sha": "abc123",
                        "dataset_sha": None,
                        "rng_state": None,
                        "cli_invocation": "python scripts/sr_train_v6.py --max-steps 1",
                        "python_version": "3.11.0",
                        "torch_version": "2.4.0",
                        "cuda_version": None,
                        "model_arch": "v6 hat-tiny",
                        "param_count": 123,
                        "timestamp_utc": "2026-05-07T00:00:00Z",
                    },
                },
                "cost": {
                    "kwh": 0.1,
                    "usd": 0.015,
                    "gpu_hours": 0.25,
                    "projections": {
                        "B200": {
                            "gpu_hours_to_dlss4_quality": 0.01,
                            "usd_at_runpod_rate": 0.0598,
                        },
                        "H100": {
                            "gpu_hours_to_dlss4_quality": 0.02,
                            "usd_at_runpod_rate": 0.0598,
                        },
                        "A100": {
                            "gpu_hours_to_dlss4_quality": 0.03,
                            "usd_at_runpod_rate": 0.0567,
                        },
                        "4090": {
                            "gpu_hours_to_dlss4_quality": 0.04,
                            "usd_at_runpod_rate": 0.0276,
                        },
                    },
                },
            }
        ],
        "models": [],
    }

    code, errors, _warnings = validate(data)
    if code != 0:
        print("self-test valid event payload failed unexpectedly:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    broken = copy.deepcopy(data)
    del broken["runs"][0]["events"][0]["kind"]
    code, errors, _warnings = validate(broken)
    expected = "runs[0].events[0].kind: missing required key"
    if code == 0 or expected not in errors:
        print("self-test failed to reject event without kind clearly", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("OK self-test event validation rejects missing kind")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_json", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true", help="run validator self-tests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.data_json is None:
        print("data_json is required unless --self-test is set", file=sys.stderr)
        return 3
    try:
        data = load_json(args.data_json)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{args.data_json}: not readable: {exc}", file=sys.stderr)
        return 3

    is_status_json = args.data_json.name == "status.json"
    code, errors, warnings = validate_status(data) if is_status_json else validate(data)
    for warning in warnings:
        print(warning, file=sys.stderr)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return code

    if is_status_json:
        print(f"OK schema_version={data['schema_version']} services={len(data['services'])} checked_at={data['checked_at']}")
    else:
        print(
            f"OK schema_version={data['schema_version']} "
            f"runs={len(data['runs'])} models={len(data['models'])} "
            f"generated_at={data['generated_at']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
