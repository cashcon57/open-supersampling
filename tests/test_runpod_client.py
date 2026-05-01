"""Tests for RunPodClient — mock-based, no network.

Each test stubs `runpod.get_pods`, `runpod.get_gpus`, `runpod.create_pod` and
`runpod.terminate_pod` so the client logic is exercised in isolation.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip the entire module if the runpod SDK isn't installed (allows
# Lambda-only environments to run the rest of the test suite).
runpod = pytest.importorskip("runpod")

from oss.cloud.runpod_client import (
    RunPodClient,
    RunPodInstance,
    RUNPOD_DEFAULT_PRICING,
    _canonical_status,
)


# ----- API key resolution --------------------------------------------------

def test_api_key_from_kwarg(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    c = RunPodClient(api_key="kwarg-key", live_pricing=False)
    assert c._api_key == "kwarg-key"


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "env-key")
    c = RunPodClient(live_pricing=False)
    assert c._api_key == "env-key"


def test_api_key_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    keyfile = tmp_path / "rp.txt"
    keyfile.write_text("file-key\n")
    c = RunPodClient(key_path=keyfile, live_pricing=False)
    assert c._api_key == "file-key"


def test_api_key_missing_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    nonexistent = tmp_path / "absent.txt"
    with pytest.raises(RuntimeError, match="no RunPod API key"):
        RunPodClient(key_path=nonexistent, live_pricing=False)


def test_invalid_cloud_type_rejected(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "x")
    with pytest.raises(ValueError, match="cloud_type"):
        RunPodClient(cloud_type="WAT", live_pricing=False)


# ----- pod parsing ---------------------------------------------------------

def test_parse_pod_with_public_ssh_port():
    pod = {
        "id": "pod-abc",
        "desiredStatus": "RUNNING",
        "lastStatusChange": 1714502400000,
        "machine": {"gpuDisplayName": "H100 SXM"},
        "runtime": {
            "ports": [
                {"privatePort": 22, "publicPort": 12345, "ip": "1.2.3.4", "isIpPublic": True, "type": "tcp"},
                {"privatePort": 8888, "publicPort": 22222, "ip": "10.0.0.1", "isIpPublic": False, "type": "http"},
            ]
        },
    }
    inst = RunPodClient._parse_pod(pod)
    assert isinstance(inst, RunPodInstance)
    assert inst.instance_id == "pod-abc"
    assert inst.status == "active"
    assert inst.ip == "1.2.3.4"
    assert inst.instance_type == "H100 SXM"
    assert inst.hostname.endswith("@ssh.runpod.io")


def test_parse_pod_canonical_status():
    assert _canonical_status("RUNNING") == "active"
    assert _canonical_status("EXITED") == "terminated"
    assert _canonical_status("PROVISIONING") == "booting"
    assert _canonical_status("PAUSED") == "stopped"
    assert _canonical_status(None) == "unknown"
    assert _canonical_status("WHATEVER") == "unknown"


# ----- list_instances ------------------------------------------------------

def _make_client(monkeypatch, *, gpus=None, pods=None):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    c = RunPodClient(live_pricing=False)
    # Patch the bound `_runpod` reference rather than the module-global, so
    # multiple clients in one test don't stomp on each other.
    fake_sdk = MagicMock()
    fake_sdk.get_gpus.return_value = gpus or []
    fake_sdk.get_pods.return_value = pods or []
    fake_sdk.get_pod.side_effect = lambda pid: next(
        (p for p in (pods or []) if p.get("id") == pid), None
    )
    c._runpod = fake_sdk
    return c, fake_sdk


def test_list_instances_maps_to_cloudinstance(monkeypatch):
    pods = [
        {
            "id": "p1",
            "desiredStatus": "RUNNING",
            "lastStatusChange": 1714502400000,
            "machine": {"gpuDisplayName": "H100 SXM"},
            "runtime": {"ports": [{"privatePort": 22, "publicPort": 1, "ip": "1.1.1.1", "isIpPublic": True}]},
        },
        {
            "id": "p2",
            "desiredStatus": "EXITED",
            "lastStatusChange": 1714502401000,
            "machine": {"gpuDisplayName": "A100 80GB"},
            "runtime": None,
        },
    ]
    c, _ = _make_client(monkeypatch, pods=pods)
    insts = c.list_instances()
    assert [i.instance_id for i in insts] == ["p1", "p2"]
    assert insts[0].status == "active"
    assert insts[1].status == "terminated"


# ----- pricing -------------------------------------------------------------

def test_live_pricing_overlay(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    # `_refresh_pricing` calls `get_gpu(id)` per default-priced GPU. Stub each.
    by_id = {
        "NVIDIA H100 80GB HBM3": {"id": "NVIDIA H100 80GB HBM3", "securePrice": 3.50, "communityPrice": 2.20},
        "NVIDIA A40":            {"id": "NVIDIA A40",            "securePrice": 0.45, "communityPrice": 0.30},
    }
    def fake_get_gpu(gid, *a, **kw):
        return by_id.get(gid)
    with patch("runpod.get_gpu", side_effect=fake_get_gpu):
        c = RunPodClient(live_pricing=True)
    # We always pick max(default, live) so the budget cap stays conservative.
    assert c.hourly_rate("NVIDIA H100 80GB HBM3") == 3.50    # live > default 2.99
    assert c.hourly_rate("NVIDIA A40") == max(0.45, RUNPOD_DEFAULT_PRICING["NVIDIA A40"])


def test_live_pricing_failure_falls_back_to_defaults(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    # Simulate per-GPU lookup raising on every call. The client should swallow
    # individual failures inside `_refresh_pricing`'s try/except and never
    # raise out to the caller.
    with patch("runpod.get_gpu", side_effect=RuntimeError("api down")):
        c = RunPodClient(live_pricing=True)
    # Constructor must not raise; defaults remain in place.
    assert c.hourly_rate("NVIDIA H100 80GB HBM3") == RUNPOD_DEFAULT_PRICING["NVIDIA H100 80GB HBM3"]


def test_canonicalize_gpu_id():
    assert RunPodClient.canonicalize_gpu_id("NVIDIA H100 80GB HBM3") == "gpu_1x_h100_sxm5"
    assert RunPodClient.canonicalize_gpu_id("NVIDIA H100 PCIe") == "gpu_1x_h100_pcie"
    assert RunPodClient.canonicalize_gpu_id("Some Future GPU") is None


# ----- launch + orphan recovery -------------------------------------------

def test_launch_returns_pod_id(monkeypatch):
    c, sdk = _make_client(monkeypatch)
    sdk.create_pod.return_value = {"id": "pod-new"}
    ids = c.launch(
        instance_type_name="NVIDIA H100 PCIe",
        region_name="",
        ssh_key_names=[],
        name="t",
    )
    assert ids == ["pod-new"]
    sdk.create_pod.assert_called_once()
    kwargs = sdk.create_pod.call_args.kwargs
    assert kwargs["gpu_type_id"] == "NVIDIA H100 PCIe"
    assert kwargs["cloud_type"] == "SECURE"
    assert kwargs["start_ssh"] is True


def test_launch_orphan_recovery_after_create_raises(monkeypatch):
    """If create_pod raises but a pod was actually created server-side,
    the client must adopt it via the list_instances poll."""
    c, sdk = _make_client(monkeypatch)
    pre_existing = []
    new_pod = {
        "id": "pod-orphan",
        "desiredStatus": "RUNNING",
        "machine": {"gpuDisplayName": "H100 PCIe"},
        "runtime": {"ports": []},
    }
    sdk.get_pods.side_effect = [pre_existing, [new_pod], [new_pod]]
    sdk.create_pod.side_effect = RuntimeError("timeout, but it actually launched")
    ids = c.launch(
        instance_type_name="NVIDIA H100 PCIe",
        region_name="",
        ssh_key_names=[],
    )
    assert ids == ["pod-orphan"]


def test_launch_no_orphan_raises(monkeypatch):
    c, sdk = _make_client(monkeypatch)
    sdk.get_pods.return_value = []
    sdk.create_pod.side_effect = RuntimeError("real failure")
    # Patch sleep so the 6 retry attempts don't actually wait 60s.
    with patch("ors.cloud.runpod_client.time.sleep", return_value=None):
        with pytest.raises(RuntimeError, match="real failure"):
            c.launch(
                instance_type_name="NVIDIA H100 PCIe",
                region_name="",
                ssh_key_names=[],
            )


def test_launch_empty_response_attempts_orphan_recovery(monkeypatch):
    """If create_pod returns {} (no id), still try to find an orphan."""
    c, sdk = _make_client(monkeypatch)
    sdk.get_pods.side_effect = [
        [],  # pre-existing snapshot
        [{"id": "pod-found", "desiredStatus": "RUNNING",
          "machine": {"gpuDisplayName": "H100 PCIe"}, "runtime": None}],
    ]
    sdk.create_pod.return_value = {}  # no id
    with patch("ors.cloud.runpod_client.time.sleep", return_value=None):
        ids = c.launch(
            instance_type_name="NVIDIA H100 PCIe",
            region_name="",
            ssh_key_names=[],
        )
    assert ids == ["pod-found"]


# ----- terminate -----------------------------------------------------------

def test_terminate_idempotent_on_already_gone(monkeypatch):
    c, sdk = _make_client(monkeypatch)
    sdk.terminate_pod.side_effect = [None, RuntimeError("not found")]
    res = c.terminate(["alive", "already-gone"])
    # Both pods reported, neither raises.
    assert "data" in res
    assert res["data"]["alive"] == "ok"
    assert res["data"]["already-gone"].startswith("err:")


def test_terminate_empty_list_is_noop(monkeypatch):
    c, sdk = _make_client(monkeypatch)
    res = c.terminate([])
    assert res == {}
    sdk.terminate_pod.assert_not_called()


# ----- protocol surface ----------------------------------------------------

def test_terminate_endpoint_and_auth(monkeypatch):
    c, _ = _make_client(monkeypatch)
    assert c.vendor_name == "runpod"
    assert c.terminate_endpoint() == "https://api.runpod.io/graphql"
    assert c.terminate_auth_header("KKK") == "Authorization: Bearer KKK"
    assert c.terminate_curl_auth_flag("KKK") is None
    body = c.terminate_request_body("pod-x")
    assert "podTerminate" in body
    assert "pod-x" in body
